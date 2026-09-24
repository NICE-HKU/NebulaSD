"""Opt-in banked Draft instrumentation, independent of legacy worker internals."""
from dataclasses import asdict
from time import perf_counter_ns


def attach(adapter, recorder):
    worker = adapter.worker
    if recorder.mode == "light":
        from .light_profiling import attach_forward
        attach_forward(adapter, recorder)
    from .backend_work import attach_backend_work
    from .gpu_profiling import attach_gpu
    if recorder.mode == 'full':
        attach_backend_work(adapter, recorder)
        recorder.gpu = attach_gpu(adapter, recorder)
    recorder.attach_table(adapter.resources.table)
    recorder.attach_table(adapter.resources.registry)
    recorder.wrap(worker, 'execute_async', 'worker.execute', command=True, category='inclusive')
    recorder.wrap(worker.compiler, 'compile', 'worker.compile')
    recorder.wrap(worker.backend, 'run_batch', 'backend.run_batch', items=True, category='backend_wall')
    recorder.wrap(worker.store, 'publish', 'worker.publish_result')
    for name in ('reserve_batch', 'begin_compute', 'end_compute', 'complete_import', 'retire_batch'):
        recorder.wrap(worker.sessions, name, f'draft.bank.{name}')
    for name in ('accept_prepare', 'consume_ready'):
        recorder.wrap(worker.copy_lane, name, f'copy.{name}', command=True)
    original = adapter.consumer.consume
    def consume(**kwargs):
        start = perf_counter_ns()
        envelope = original(**kwargs)
        if envelope is not None:
            from .profiling import request_keys
            command = envelope.decode(worker_id=worker.worker_id)
            recorder.record('command.consume', start, perf_counter_ns(), keys=request_keys(command),
                command_seq=command.command_seq, worker=worker.worker_id, kind=command.kind.name)
        return envelope
    adapter.consumer.consume = consume
    previous = worker.copy_lane.on_receipt
    def receipt(plan, result):
        recorder.record('copy.receipt', result.enqueued_ns, result.retired_ns,
            keys=[[r.extent.request_slot, r.extent.request_epoch, n] for r, n in zip(plan.regions, plan.round_ids)],
            direction=plan.direction, timing=asdict(result))
        if previous is not None:
            previous(plan, result)
    worker.copy_lane.on_receipt = receipt
