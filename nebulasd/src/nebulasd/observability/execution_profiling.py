"""Control-only tracing for isolated workers; never imports model code."""
from time import perf_counter_ns, thread_time_ns
from .bank_turnaround import bank_key as trace_bank_key


def attach_result_sender(link, metrics):
    """One boundary per result, including activation; no payload logging."""
    original = link.results.send
    def send(message):
        start = perf_counter_ns()
        result = original(message)
        if result:
            metrics.record('execution.result.sent', start, perf_counter_ns(),
                           operation=message.operation, kind=message.kind.name)
        return result
    link.results.send = send


def attach_control(adapter, recorder):
    from .copy_trace import attach_control as attach_copy_control
    attach_copy_control(adapter, recorder)
    worker, remote = adapter.worker, adapter.execution_backend
    original_receive = remote.client.link.results.receive
    def receive():
        result = original_receive()
        if result is not None:
            recorder.record('execution.result.received', perf_counter_ns(), keys=[],
                            operation=result.operation, kind=result.kind.name)
        return result
    remote.client.link.results.receive = receive
    lane = worker.copy_lane
    original_complete = lane._complete_flight
    def complete(flight):
        start, cpu = perf_counter_ns(), thread_time_ns()
        result = original_complete(flight)
        recorder.record('copy.owner.complete', start, perf_counter_ns(), keys=[],
                        direction=flight.plan.direction, bank_key=trace_bank_key(flight.batch), cpu_ns=thread_time_ns()-cpu)
        return result
    lane._complete_flight = complete
    for name in ("_start_h2d", "_start_import", "_start_d2h", "_start_export", "_launch"):
        recorder.wrap(lane, name, "copy.owner." + name)
    for name in ("retire", "drain_complete", "complete_h2d", "complete_import"):
        recorder.wrap(lane.banks, name, "copy.owner." + name)
    if adapter.target:
        for name in ("gpu_ready", "host_ready"):
            recorder.wrap(lane.publisher, name, "copy.owner." + name)
    recorder.attach_table(adapter.resources.table)
    recorder.attach_table(adapter.resources.registry)
    recorder.wrap(worker, 'execute_async', 'worker.execute', command=True, category='inclusive')
    if adapter.target:
        compiler, state = worker._input_compiler, worker._local_state
        for name in ('compile_prefill', 'compile_prepare', 'compile_verify'):
            recorder.wrap(compiler, name, 'control.' + name)
        for name in ('commit_prefill', 'commit_verify', 'commit_prepare', 'mark_gpu_ready'):
            recorder.wrap(state, name, 'control.' + name)
    else:
        recorder.wrap(worker.compiler, 'compile', 'control.compile')
        from nebulasd.workers.draft import execution_control
        for name in ('compile_draft_execution', 'finalize_draft_execution'):
            recorder.wrap(execution_control, name, 'control.' + name)
        for name in ('restore', 'run_inputs', 'publish'):
            recorder.wrap(worker.store, name, 'control.snapshot.' + name)
    original = remote.client.submit
    def submit(kind, payload=b'', **kwargs):
        start = perf_counter_ns()
        operation = original(kind, payload, **kwargs)
        def complete(future):
            if not future.cancelled() and future.exception() is None:
                recorder.record('execution.round_trip', start, perf_counter_ns(), keys=[],
                    kind=kind.name, operation=operation.operation, request_bytes=64 + len(payload),
                    result_bytes=64 + len(future.result()) + (64 + len(operation._started.result()) if operation._started is not None else 0))
        operation._future.add_done_callback(complete)
        return operation
    remote.client.submit = submit

    import os
    if os.environ.get("STARSD_BANK_TURNAROUND_PROFILE") == "1":
        from .bank_turnaround import attach_control as attach_turnaround
        attach_turnaround(adapter, recorder)
