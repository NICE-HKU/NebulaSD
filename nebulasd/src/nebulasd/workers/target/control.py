"""CPU protocol publication and physical Runtime owner for isolated workers.

Only execution owns the model. Engine facts follow shared publication. Explicit
diagnostic execution entries retain the original WORK/input forwarding adapter."""
from collections import deque
from contextlib import ExitStack
from queue import Empty, Full
from time import monotonic, perf_counter_ns


def run_control(options, engine_commands, execution_commands,
                execution_results, wake, execution_wake, ready, execution_stopped, *, publisher_factory=None, result_kind="RESULT", import_kind="IMPORTED", projection_factory=None):
    from nebulasd.workers.channel import LocalChannel
    from nebulasd.workers.work import Work, MAX_OUTSTANDING
    from nebulasd.workers.dependencies import Dependencies
    from nebulasd.workers.input_facts import InputFact, FACT_BUDGET
    from nebulasd.workers.completion import CompletionArena
    from nebulasd.table.native_storage import request_table, close_table_partitions
    from nebulasd.data.shared_arenas import SharedTokenArena
    from .publication import TargetPublisher
    with ExitStack() as stack:
        channels = []
        for descriptor in (engine_commands,execution_commands,execution_results):
            channel = LocalChannel(descriptor)
            stack.callback(channel.close)
            channels.append(channel)
        incoming,commands,results = channels
        event = None
        if 'event' in options:
            from nebulasd.ipc.native_ring import NativeStateChangeRing
            event = NativeStateChangeRing(options['event_capacity'],descriptor=options['event'],doorbell=options['engine_bell'])
            stack.callback(event.close)
        table = request_table(options['slots'],descriptors=options['table'],ring=event)
        stack.callback(close_table_partitions,table._partitions)
        completions = CompletionArena(descriptor=options['completions'])
        stack.callback(completions.close)
        from nebulasd.workers.resources import attach_router
        from nebulasd.data.shared_arenas import SharedConfigArena
        configs, config_arenas = attach_router(options['configs'], SharedConfigArena)
        outputs, output_arenas = attach_router(options['tokens'], SharedTokenArena)
        for arena in (*config_arenas, *output_arenas):
            stack.callback(arena.close)
        if publisher_factory is None:
            publisher = TargetPublisher(table,completions,block_bytes=options['host'].block_bytes,block_size=options.get('block_size',16), outputs=outputs, configs=configs)
        else:
            publisher = publisher_factory(options, table, completions, stack)
        if 'global_registry' in options:
            from nebulasd.workers.observation import Projection
            projection_factory = Projection
        projection = None if projection_factory is None else projection_factory(options,stack)
        dependencies = Dependencies(table, configs=configs, period_s=options.get('period_s', .0001),
                                    profile=options.get('profile', False))
        records, forwards = {},deque()
        import os
        if options.get("profile", False) and os.environ.get("STARSD_ENGINE_DETAIL_DIR"):
            from nebulasd.observability.engine_detail import attach_control
            attach_control(options, publisher, event, stack)

        def forward():
            sent = False
            for _ in range(8):
                if not forwards:
                    break
                try:
                    if runtime_owner is None:
                        commands.put_nowait(forwards[0])
                    else:
                        kind, value = forwards[0]
                        if kind == 'WORK':
                            runtime_owner.accept(value)
                        elif kind == 'RETIRE_RECORD':
                            runtime_owner.retire(value)
                        else:
                            runtime_owner.stop(kind == 'SHUTDOWN')
                except Full:
                    break
                forwards.popleft()
                sent = True
            if sent and runtime_owner is None:
                execution_wake.set()
            return sent

        runtime_owner = None
        if options.get('isolated_compute'):
            from nebulasd.workers.isolated import ControlRuntime
            runtime_owner = ControlRuntime(options, stack, commands, results, wake, execution_wake)
            results = runtime_owner
            publisher.handoff_trace = runtime_owner.trace

        registration = None
        stopping = False
        publication_deadline = 0.0
        publication_delay = options.get('diagnostic_publication_delay_s',0.0)
        if publication_delay < 0 or options.get('diagnostic_publication_busy_s',0.0) < 0:
            raise ValueError('negative publication delay')
        def consume_message(message):
            nonlocal publication_deadline
            kind, seq, data = message
            record = records[seq]
            if kind == 'PHYSICAL':
                dependencies.retire(seq)
                record['inputs'].clear()
            publisher.consume(message)
            record['messages'][kind] = data
            if kind == result_kind:
                publication_deadline = max(publication_deadline,monotonic()+publication_delay)
                if options.get('profile',False):
                    record['times'].append(('CONTROL_RESULT_RECEIVED',perf_counter_ns()))
                if runtime_owner is not None and not publication_delay and not options.get('diagnostic_publication_busy_s',0):
                    publisher.publish_results(seq)
        if runtime_owner is not None:
            runtime_owner.on_result = consume_message

        ready.set()
        while True:
            wake.clear()
            changed = runtime_owner.step() if runtime_owner is not None else False
            received = False
            for _ in range(4):
                try:
                    message = results.get_nowait()
                except Empty:
                    break
                changed = True
                kind,seq,data = message
                received = True
                consume_message(message)
            if received:
                execution_wake.set()  # Also retry any bounded result backpressure.
            if projection is not None:
                changed |= projection.step()
            # The sole observer lives here, even when the Engine/result rings
            # are Empty. Pending facts occupy at most three entries per member.
            for event in dependencies.poll(FACT_BUDGET):
                record = records[event.key[0]]
                record['inputs'].append(InputFact.capture(record['work'], event))
                changed = True
            # FIFO first sends every earlier WORK/control. A full ring retains
            # compact facts on their existing WORK record, without blocking.
            inputs_sent = False
            if not forwards:
                for seq, record in records.items():
                    events = record['inputs']
                    if not events:
                        continue
                    work = record['work']
                    count = len(events) if runtime_owner is not None else FACT_BUDGET
                    try:
                        inputs = dict(worker_id=work.worker_id,
                            worker_generation=work.worker_generation,events=events[:count])
                        if runtime_owner is None:
                            commands.put_nowait(('INPUTS',seq,inputs))
                        else:
                            runtime_owner.receive_inputs(seq, inputs)
                    except Full:
                        break
                    del events[:count]
                    inputs_sent = changed = True
            if inputs_sent:
                if runtime_owner is not None:
                    changed |= runtime_owner.step()
                else:
                    execution_wake.set()
            # Bounded cooperative cold registration. Existing WORK handoffs
            # run above between rows, instead of waiting for a large reservation.
            if registration is not None:
                work, payload, iterator = registration
                deadline = perf_counter_ns() + 200_000
                try:
                    # Amortize the owner loop, but cap cold work between hot
                    # completion checks. Dependency registration is batched so
                    # the native watch set is rebuilt once per WORK.
                    while True:
                        next(iterator)
                        if perf_counter_ns() >= deadline:
                            break
                except StopIteration:
                    records[work.work_seq] = dict(work=work,messages={},times=[],retired=False,busy_done=False,inputs=[])
                    forwards.append(('WORK', work if runtime_owner is not None else payload))
                    changed |= forward()
                    dependencies.register(work)
                    registration = None
                changed = True
            if registration is None:
                for _ in range(4):
                    if (len(records) >= MAX_OUTSTANDING or len(forwards) >= MAX_OUTSTANDING) and incoming.peek_kind() == 'WORK':
                        break
                    try:
                        kind, payload = incoming.get_nowait()
                    except Empty:
                        break
                    changed = True
                    if kind == 'WORK':
                        if stopping or len(records) >= MAX_OUTSTANDING:
                            raise RuntimeError('Engine exceeded WORK credit or dispatched after drain')
                        work = Work.from_bytes(payload)
                        if 'worker_id' in options and (work.worker_id, work.worker_generation) != (options['worker_id'], options['worker_generation']):
                            raise ValueError('WORK belongs to another worker generation')
                        if work.work_seq in records:
                            raise RuntimeError('duplicate Engine WORK')
                        registration = (work, payload, publisher.reserve_steps(work))
                        break
                    elif kind in ('DRAIN', 'SHUTDOWN'):
                        if stopping and kind != 'SHUTDOWN':
                            raise RuntimeError('duplicate drain')
                        stopping = True
                        if kind == 'SHUTDOWN':
                            dependencies.clear()
                            for record in records.values():
                                record['inputs'].clear()
                    else:
                        raise ValueError('Engine may send only WORK, DRAIN or SHUTDOWN')
                    forwards.append((kind, payload))
                    changed |= forward()
            changed |= forward()
            # Explicit diagnostic: occupy THIS interpreter's GIL with Python
            # work. Only WORKs whose required facts have reached execution are
            # independent of this interval; uncaptured inputs wait here.
            busy_s = options.get('diagnostic_publication_busy_s',0.0)
            for record in records.values():
                if busy_s and result_kind in record['messages'] and not record['busy_done']:
                    record['busy_done'] = True
                    begin = perf_counter_ns()
                    until = begin+int(busy_s*1e9)
                    while perf_counter_ns() < until:
                        pass
                    if options.get('profile',False):
                        record['times'].extend((('CONTROL_PUBLICATION_BUSY_BEGIN',begin),
                                                ('CONTROL_PUBLICATION_BUSY_END',perf_counter_ns())))
            if monotonic() >= publication_deadline:
                retired = publisher.step(options.get('publication_event_budget',1))
                changed |= bool(publisher.last_published_rows or retired)
                for seq,kind,at in publisher.take_times():
                    records[seq]['times'].append((kind,at))
                for seq in retired:
                    record = records[seq]
                    record['retired'] = True
                    if options.get('profile',False):
                        record['times'].append(('CONTROL_WORK_PUBLISHED',perf_counter_ns()))
                    # This credit releases only the execution result record.
                    forwards.append(('RETIRE_RECORD',seq))
            for seq, record in tuple(records.items()):
                if record['retired']:
                    del records[seq]
            changed |= forward()
            if len(forwards) > 2*MAX_OUTSTANDING+2:
                raise RuntimeError('bounded execution control backlog exceeded')
            if execution_stopped.is_set():
                if not stopping or records or forwards or publisher.records or registration is not None:
                    raise RuntimeError('execution exited before protocol retirement')
                break
            if not changed:
                # Results/Engine commands signal this worker-local event. The
                # finite wait also observes process exit and diagnostic deadlines.
                wake.wait(min(options.get('control_period_s',.001), dependencies.period_s)
                          if dependencies.pending else options.get('control_period_s',.001))
