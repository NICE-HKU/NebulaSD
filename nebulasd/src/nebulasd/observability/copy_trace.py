"""Opt-in CopyPlan/retirement provenance; buffered by existing recorders."""
from dataclasses import asdict
from time import perf_counter_ns, thread_time_ns


def plan_fields(plan):
    return dict(direction=plan.direction, regions=[asdict(r) for r in plan.regions],
                round_ids=list(plan.round_ids), dependency_count=len(plan.dependencies))


def attach_execution(service):
    metrics = service.metrics
    if metrics.profile is None:
        return
    from contextvars import ContextVar
    from nebulasd.workers.execution.protocol import Kind
    from nebulasd.workers.execution.copy_wire import decode_copy
    import os
    from pathlib import Path
    current = ContextVar('copy_operation', default=None)
    executors = getattr(service.copy, "executors", (service.copy,))
    backend = executors[0].backend
    contexts = {}
    device_clock = None
    if os.environ.get('STARSD_COPY_DEVICE_PROFILE') == '1':
        import torch
        with torch.cuda.device(backend.k_cache.device):
            anchor = torch.cuda.Event(enable_timing=True)
            before = perf_counter_ns()
            anchor.record()
            anchor.synchronize()  # Initialization only, before admitting work.
            after = perf_counter_ns()
        device_clock = anchor
        metrics.record('copy.device.clock', before, after,
            metadata_stream=getattr(getattr(getattr(service, 'sessions', getattr(service, 'support', None)), '_metadata_stream', None), 'cuda_stream', None),
            default_stream=torch.cuda.default_stream(backend.k_cache.device).cuda_stream)
    for executor in executors:
        _attach_dma_cost(executor, metrics, contexts, device_clock)
    now = perf_counter_ns()
    metrics.record('copy.layout', now, now, descriptor=asdict(service.host.descriptor),
        gpu=str(backend.k_cache.device), cache_shape=list(backend.k_cache.shape),
        affinity=sorted(os.sched_getaffinity(0)), numa_maps=Path('/proc/self/numa_maps').read_text())
    original_execute = service._execute
    async def execute(message, activated=None):
        if message.kind not in (Kind.IMPORT, Kind.EXPORT):
            return await original_execute(message, activated)
        start = perf_counter_ns()
        key, plan, versions = decode_copy(message.payload,
            direction='H2D' if message.kind is Kind.IMPORT else 'D2H', max_batch_size=service.spec.max_batch_size)
        fields = dict(operation=message.operation, bank_key=key, gpu_versions=versions, **plan_fields(plan))
        metrics.record('copy.operation.start', start, start, **fields)
        token = current.set(fields)
        failed = True
        try:
            result = await original_execute(message, activated)
            failed = False
            return result
        finally:
            metrics.record('copy.operation.end', start, perf_counter_ns(), operation=message.operation, failed=failed)
            current.reset(token)
    service._execute = execute
    method = "submit_batch" if hasattr(service.copy, "submit_batch") else "submit"
    original_submit = getattr(service.copy, method)
    def submit(plan, *args):
        fields = current.get()
        contexts[id(plan)] = fields
        future = original_submit(plan, *args)
        def done(f):
            if f.cancelled() or f.exception() is not None:
                metrics.record('copy.failed', perf_counter_ns(), perf_counter_ns(), operation=fields['operation'])
            else:
                receipt = f.result()
                metrics.record('copy.dma', receipt.enqueued_ns, receipt.completed_ns,
                    operation=fields['operation'], dependency_count=len(plan.dependencies), timing=asdict(receipt))
        future.add_done_callback(done)
        return future
    setattr(service.copy, method, submit)
    original_acquire = backend._registration.acquire
    def acquire(arena):
        start=perf_counter_ns()
        result=original_acquire(arena)
        metrics.record('copy.registration',start,perf_counter_ns(),bytes=arena.descriptor.total_bytes,
                       numa_maps=Path('/proc/self/numa_maps').read_text())
        return result
    backend._registration.acquire=acquire
    if os.environ.get("STARSD_BANK_TURNAROUND_PROFILE") == "1":
        from .bank_turnaround import attach_execution as attach_turnaround
        attach_turnaround(service)


def attach_control(adapter, recorder):
    lane=adapter.worker.copy_lane
    facts=lane.host_facts if adapter.target else lane.facts
    # Capture actual source-demand handoff, including Target's local dirty FIFO.
    obj,method=(lane.dirty_sink,'put') if adapter.target else (lane,'enqueue_dirty')
    original_dirty=getattr(obj,method)
    def dirty(batch,*args,**kwargs):
        start=perf_counter_ns()
        result=original_dirty(batch,*args,**kwargs)
        recorder.record('copy.dirty.enqueue',start,perf_counter_ns(),keys=[],
                        bank_key=[batch.bank_id,batch.bank_epoch,batch.batch_seq])
        return result
    setattr(obj,method,dirty)
    original_launch=lane._launch
    def launch(plan,batch,pins,*args,**kwargs):
        start=perf_counter_ns()
        fields=plan_fields(plan)
        fields['bank_key']=[getattr(batch,'bank_id',getattr(batch,'standby_bank_id',None)),
                            getattr(batch,'bank_epoch',getattr(batch,'next_bank_epoch',None)),batch.batch_seq]
        recorder.record('copy.control.launch',start,keys=[],**fields)
        return original_launch(plan,batch,pins,*args,**kwargs)
    lane._launch=launch
    previous=lane.on_receipt
    def receipt(plan,result):
        recorder.record('copy.control.receipt',result.enqueued_ns,result.retired_ns,keys=[],
                        **plan_fields(plan),timing=asdict(result))
        if previous is not None:previous(plan,result)
    lane.on_receipt=receipt
    for method in ('accept_prepare','enqueue_dirty','discard_prepare','discard_pending_for_shutdown'):
        recorder.wrap(lane,method,'copy.control.'+method,command=method in ('accept_prepare','enqueue_dirty'))
    import os
    if os.environ.get("STARSD_TURNAROUND_LEAN") == "1":
        return
    # Track logical pins separately from CUDA page registration; no extra polling.
    allocator=facts.allocator
    for method in ('pin_prepared' if hasattr(allocator, 'pin_prepared') else 'pin', 'release_pinned' if hasattr(allocator, 'release_pinned') else 'unpin'):
        original=getattr(allocator,method)
        def wrapped(extent,*,write,original=original,method=method):
            start=perf_counter_ns();result=original(extent,write=write)
            recorder.record('copy.'+('unpin' if method in ('release_pinned', 'unpin') else 'pin'),start,perf_counter_ns(),keys=[],slot=extent.request_slot,
                epoch=extent.request_epoch,write=write,success=result if method in ('pin', 'pin_prepared') else True)
            return result
        setattr(allocator,method,wrapped)
    original_source=facts.source_ready
    states={}
    def source(*args,**kwargs):
        start=perf_counter_ns();result=original_source(*args,**kwargs)
        arg=args[0];slot=arg.request_slot;epoch=arg.request_epoch
        version=getattr(arg,'snapshot_version',getattr(arg,'source_host_version',None))
        state=(epoch,version,result is not None)
        if states.get(slot)!=state:
            states[slot]=state
            recorder.record('copy.source',start,perf_counter_ns(),keys=[],slot=slot,epoch=epoch,
                            version=version,available=result is not None)
        return result
    facts.source_ready=source


def _attach_dma_cost(executor, metrics, contexts, device_clock=None):
    # Opt-in only. One record per DMA, never one record per query.
    original_run, original_launch = executor._run, executor.backend.launch
    queries = [0]
    device_state = {}
    def launch(plan):
        chunk_index = len(device_state.setdefault('chunks', []))
        if device_clock is not None:
            dependencies_ready = [d.query() for d in plan.dependencies]
        else:
            dependencies_ready = []
        ticket = original_launch(plan)
        if device_clock is not None:
            device_state['chunks'].append(dict(plan=plan, ticket=ticket,
                dependencies_ready=dependencies_ready, chunk_index=chunk_index))
        class CountedTicket:
            def query(self):
                queries[0] += 1
                return ticket.query()
            def duration_ms(self):
                return ticket.duration_ms()
        return CountedTicket()
    def run(plan, enqueued_ns):
        fields = contexts.pop(id(plan))
        start, cpu = perf_counter_ns(), thread_time_ns()
        queries[0] = 0
        device_state['chunks'] = []
        succeeded = False
        try:
            result = original_run(plan, enqueued_ns)
            succeeded = True
            return result
        finally:
            if succeeded and device_clock is not None:
                chunks = device_state.get('chunks', [])
                chunk_count = len(chunks) or 1
                for chunk in chunks:
                    ticket = chunk['ticket']
                    chunk_plan = chunk['plan']
                    descriptor = executor.backend.arena.descriptor
                    chunk_bytes = sum(r.block_count for r in chunk_plan.regions) * descriptor.block_bytes * 2
                    metrics.record('copy.device.span', start, perf_counter_ns(),
                        operation=fields['operation'], bank_key=fields['bank_key'], direction=plan.direction,
                        stream=executor.backend._stream.cuda_stream,
                        dependencies_ready=chunk['dependencies_ready'],
                        chunk_index=chunk['chunk_index'], chunk_count=chunk_count,
                        chunk_bytes=chunk_bytes,
                        oversized_chunk=bool(chunk_bytes > getattr(executor, '_h2d_chunk_bytes', 0) > 0),
                        device_start_ms=device_clock.elapsed_time(ticket.start),
                        device_end_ms=device_clock.elapsed_time(ticket.done))
            metrics.record('copy.thread.cost', start, perf_counter_ns(),
                operation=fields['operation'], bank_key=fields['bank_key'],
                direction=plan.direction, cpu_ns=thread_time_ns()-cpu, queries=queries[0])
    executor.backend.launch = launch
    executor._run = run
