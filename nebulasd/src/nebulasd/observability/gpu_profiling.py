"""Canonical CUDA event tracing; synchronize only at setup/report, never per step."""
from collections import deque
from threading import Lock
from time import perf_counter_ns


class GPUProfile:
    def __init__(self, recorder, torch):
        self.recorder, self.torch = recorder, torch
        self.device = torch.cuda.current_device()
        self.lock = Lock()
        self.context = []
        self.events = deque(maxlen=recorder.events.maxlen)
        self.total = 0
        candidates = []
        for _ in range(3):
            origin = torch.cuda.Event(enable_timing=True)
            begin = perf_counter_ns()
            origin.record()
            origin.synchronize()
            end = perf_counter_ns()
            candidates.append((end-begin, (begin+end)//2, origin))
        self.uncertainty, self.host_origin, self.origin = min(candidates, key=lambda x:x[0])

    def add(self, name, keys, start, done):
        with self.lock:
            self.total += 1
            self.events.append((name, list(keys), start, done))

    def instrument(self, model, name):
        original = model._forward
        def forward(*args, **kwargs):
            keys = list(self.context)
            start, done = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
            stream = self.torch.cuda.current_stream(self.device)
            start.record(stream)
            result = original(*args, **kwargs)
            done.record(stream)
            self.add(name, keys, start, done)
            return result
        model._forward = forward

    def finish(self):
        for name, keys, start, done in self.events:
            done.synchronize()  # Shutdown report only, after owner retirement.
            begin = self.host_origin + int(self.origin.elapsed_time(start)*1e6)
            end = self.host_origin + int(self.origin.elapsed_time(done)*1e6)
            self.recorder.record(name, begin, end, keys=keys, category='gpu', device=self.device,
                                 clock_uncertainty_ns=self.uncertainty)
        self.recorder.record('gpu.coverage', perf_counter_ns(), keys=[],
                             gpu_dropped_events=self.total-len(self.events))


def attach_gpu(raw, recorder):
    """Only canonical CUDA backends get kernel intervals; CPU injection stays CPU."""
    backend = raw.worker.backend if getattr(raw, 'draft_banked', False) else raw.worker._backend
    if raw.target:
        facade = getattr(backend, 'bank_facade', None)
        model = getattr(getattr(facade, 'worker', None), 'model', None)
    else:
        sessions = getattr(backend, '_adapter', None)
        model = getattr(getattr(sessions, '_adapter', None), 'model', None)
    if model is None or not getattr(getattr(model, 'k_cache', None), 'is_cuda', True):
        return None
    import torch
    gpu = GPUProfile(recorder, torch)
    gpu.instrument(model, 'gpu.target' if raw.target else 'gpu.draft')
    if raw.target:
        for method in ('prefill_batch_async', 'verify_batch_async'):
            original = getattr(backend, method)
            def wrap(original):
                async def compute(items):
                    gpu.context = [[i.request_slot,i.request_epoch,i.round_id] for i in items]
                    return await original(items)
                return compute
            setattr(backend, method, wrap(original))
    if raw.target or getattr(raw, 'draft_banked', False):
        copy = raw.worker.copy_lane.executor.backend
        launch = copy.launch
        def traced_launch(plan):
            ticket = launch(plan)
            if hasattr(ticket, 'start') and hasattr(ticket, 'done'):
                keys = [[r.extent.request_slot,r.extent.request_epoch,n]
                        for r,n in zip(plan.regions,plan.round_ids)]
                gpu.add('gpu.'+plan.direction, keys, ticket.start, ticket.done)
            return ticket
        copy.launch = traced_launch
    if not raw.target:
        from nebulasd.workers.draft.backend import session_key_for
        current = {}
        original_run = backend.run_batch
        def run(items):
            current.update({session_key_for(i.request_slot,i.request_epoch):
                            [i.request_slot,i.request_epoch,i.round_id] for i in items})
            try:
                return original_run(items)
            finally:
                current.clear()
        backend.run_batch = run
        for method in ('prefill_batch', 'decode_batch'):
            original = getattr(sessions, method)
            def wrap(original):
                def forward(items):
                    gpu.context = [current[i.key] for i in items]
                    return original(items)
                return forward
            setattr(sessions, method, wrap(original))
    return gpu
