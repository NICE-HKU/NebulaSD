"""Owner-thread token client; buffers are local presentation state, never IPC."""
from collections import deque
from math import isfinite
from time import monotonic
from uuid import uuid4
from nebulasd.api import RequestHandle,StreamEvent
from nebulasd.core.enums import Lifecycle
from .admission import AdmissionRejected


class TokenEngine:
    def __init__(self,config,engine):
        self._config,self._engine = config,engine
        self._epoch = uuid4().int  # Reject handles from another Engine session.
        self._handles,self._buffers,self._records = {},{},{}
        self._next_id = 0
        engine.outputs.on_tokens = self._on_tokens

    @property
    def config(self):
        return self._config

    def submit(self,prompt_token_ids,generation_config):
        self._engine._check_owner()
        if getattr(generation_config,'proposal_depth',0)>self.config.max_proposal_depth:
            raise AdmissionRejected('proposal_depth exceeds Worker configuration')
        prompt = tuple(prompt_token_ids)
        request = RequestHandle(self._next_id,self._epoch)
        self._engine.admit(str(request.request_id),prompt,generation_config)
        self._records[request.request_id] = self._engine.registry.records[self._engine.registry.identities[str(request.request_id)]]
        profile = getattr(self._engine, 'profiler', None)
        if profile is not None:
            record = self._records[request.request_id]
            from time import perf_counter_ns
            profile.record('client.admitted', perf_counter_ns(), keys=[[record.input.slot, record.input.epoch, None]],
                           request_id=request.request_id)
        self._handles[request.request_id] = request
        self._buffers[request.request_id] = deque()
        self._next_id += 1
        return request

    def poll(self):
        """Drive one Engine iteration for all requests; no blocking sleep."""
        return self._engine.step()

    def read(self,request):
        """Consume currently buffered chunks exactly once; does not drive work."""
        self._record(request)
        buffer = self._buffers[request.request_id]
        events = tuple(buffer)
        buffer.clear()
        return events

    def stream(self,request,*,timeout=60):
        """Drive all requests while yielding this request's buffered chunks.

        Other requests retain their output. Closing this iterator or timing out
        does not cancel the request; cancellation is explicit. Timeout includes
        time spent by the caller consuming yielded events.
        """
        if not isfinite(timeout) or timeout<=0:
            raise ValueError('timeout must be finite and positive')
        record = self._record(request)
        deadline = monotonic()+timeout
        while True:
            self._record(request)
            buffer = self._buffers[request.request_id]
            while buffer:
                yield buffer.popleft()
            if record.lifecycle != Lifecycle.ACTIVE:
                return
            if monotonic()>=deadline:
                raise TimeoutError('stream timed out; request remains active')
            if not self.poll():
                self._engine.supervisor.bell.wait(min(.001,max(0,deadline-monotonic())))

    def drain(self,*,timeout=60):
        """Wait for terminal requests, all-owner retirement and cohort recycling."""
        from .client_idle import quiescent
        if not isfinite(timeout) or timeout<=0:
            raise ValueError('timeout must be finite and positive')
        deadline = monotonic()+timeout
        while True:
            progressed = self.poll()
            if quiescent(self._engine) and not self._engine.recycler.busy and not self._engine.registry.records:
                return
            if monotonic()>=deadline:
                raise TimeoutError('Engine drain timed out')
            if not progressed:
                self._engine.supervisor.bell.wait(min(.001,max(0,deadline-monotonic())))

    def result(self,request):
        """Return terminal output; cancellation returns its already accepted prefix."""
        record = self._record(request)
        if record.lifecycle == Lifecycle.ACTIVE:
            raise RuntimeError('request has not completed')
        return record.output

    def cancel(self,request):
        record = self._record(request)
        if record.lifecycle == Lifecycle.ACTIVE:
            self._engine.cancel(str(request.request_id))
            self._buffers[request.request_id].append(StreamEvent(request,(),Lifecycle.CANCELLED))

    def release(self, request):
        """Drop terminal client output; independent of physical resource retirement."""
        record = self._record(request)
        if record.lifecycle == Lifecycle.ACTIVE:
            raise RuntimeError('cancel or finish the request before release')
        del self._handles[request.request_id]
        del self._buffers[request.request_id]
        del self._records[request.request_id]

    def metrics(self):
        self._engine._check_owner()
        return dict(recycled_cohorts=self._engine.recycler.completed,
                    profile_directory=str(self._engine.profiler.directory) if getattr(self._engine, 'profiler', None) else None,
                    scheduling=self._engine.schedule_latency.summary(),
                    dispatch=self._engine.dispatch_latency.summary(),engine_step=self._engine.step_latency.summary())

    def reset_metrics(self):
        """Start a new measurement epoch; request and Worker state are untouched."""
        self._engine._check_owner()
        for metric in (self._engine.schedule_latency,self._engine.dispatch_latency,self._engine.step_latency):
            metric.samples.clear()
            metric.count = 0

    def close(self):
        """Stop admission, join Worker owners, then unmap resources."""
        engine = self._engine
        if engine._closed:
            return
        # Owner identity remains enforced on shutdown, including poisoned state.
        from threading import get_ident
        if get_ident()!=engine._owner:
            raise RuntimeError('close must run on the Engine owner thread')
        engine.close()

    def _record(self,request):
        self._engine._check_owner()
        if not isinstance(request,RequestHandle) or self._handles.get(request.request_id)!=request:
            raise ValueError('unknown request or stale Engine session handle')
        return self._records[request.request_id]

    def _on_tokens(self,identity,tokens,lifecycle):
        request = self._handles[int(identity)]
        self._buffers[request.request_id].append(StreamEvent(request,tokens,lifecycle))

    def __enter__(self):
        return self

    def __exit__(self,kind,error,tb):
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            if hasattr(error,'add_note'):
                error.add_note(f'Engine cleanup also failed: {cleanup}')
        return False
