"""Opt-in lifecycle observations. Never advances/polls resource state itself."""
from time import perf_counter_ns, thread_time_ns
import inspect


def bank_key(value):
    if value is None:
        return None
    value = getattr(value, 'command', value)
    if hasattr(value, 'standby_bank_id'):
        return (value.standby_bank_id, value.next_bank_epoch, value.batch_seq)
    if hasattr(value, 'active_bank_id'):
        return (value.active_bank_id, value.active_bank_epoch, value.expected_batch_seq)
    if hasattr(value, 'bank_id'):
        return (value.bank_id, value.bank_epoch, value.batch_seq)
    return None


def attach_control(adapter, recorder):
    lane = adapter.worker.copy_lane
    target = adapter.target
    banks = lane.banks
    previous = {}
    def emit_changed(name, state, **fields):
        if previous.get(name) != state:
            previous[name] = state
            recorder.record('turn.' + name, perf_counter_ns(), keys=[], state=state, **fields)
    def snapshot():
        pending = lane._pending
        command = pending.compiled.command if target and pending is not None else pending
        future = pending.prepare_future if target and pending is not None else getattr(lane, '_prepare_future', None)
        ready = bool(pending.locations) if target and pending is not None else getattr(lane, '_prepared_bank', None) is not None
        if target:
            bs = tuple((b.bank_id, b.bank_epoch, b.batch_seq, b.state.name, b.role.name, b.drain_leased, b.alloc_rows)
                       for b in banks.manager.snapshots())
            dirty = tuple(bank_key(b) for b in lane._dirty.values())
        else:
            bs = tuple((b.bank_id, b.epoch, bank_key(banks.sessions._batches.get(b.bank_id)),
                        banks.sessions.batch_state(b.bank_id)[0], len(b.request_ranges)) for b in banks.sessions.describe_banks())
            dirty = tuple(bank_key(b) for b, _, _ in lane._dirty)
        return dict(pending=bank_key(command), flights=tuple((f.plan.direction, bank_key(f.batch), f.future.done())
                          for f in lane._flights.values()),
            prepare_future=None if future is None else future.done(), prepared=ready, banks=bs, dirty=dirty)
    def state(where):
        emit_changed('state', snapshot(), where=where)
    # Observe actual progress calls, including the bound callback captured at construction.
    original_poll = lane.poll
    def poll():
        state('poll.enter')
        result = original_poll()
        state('poll.exit')
        return result
    lane.poll = poll
    lane._progress._progress = poll
    original_can = banks.can_prepare
    def can(command):
        result = original_can(command)
        conflicts = []
        if not result:
            if target:
                for row in command.requests:
                    old = banks.local_state.get(row.request_slot)
                    if old is not None:
                        loc = old.bank_location
                        conflicts.append((row.request_slot, loc.bank_id, loc.bank_epoch))
            else:
                keys = {f'slot:{r.request_slot}:epoch:{r.request_epoch}' for r in command.requests}
                for old in banks.sessions._batches.values():
                    if any(i.key.request_id in keys for i in old.items):
                        conflicts.append(bank_key(old))
        emit_changed('can_prepare', (bank_key(command), result, tuple(conflicts)), snapshot=snapshot())
        return result
    banks.can_prepare = can
    def wrap(obj, method):
        original = getattr(obj, method, None)
        if original is None:
            return
        def fields(args):
            return dict(bank_key=bank_key(args[0]) if args else None)
        if inspect.iscoroutinefunction(original):
            async def call(*args, **kwargs):
                start = perf_counter_ns(); f = fields(args)
                recorder.record('turn.' + method + '.start', start, keys=[], **f)
                result = await original(*args, **kwargs)
                recorder.record('turn.' + method, start, perf_counter_ns(), keys=[], **f)
                state(method)
                return result
        else:
            def call(*args, **kwargs):
                start, cpu = perf_counter_ns(), thread_time_ns(); f = fields(args)
                state(method + '.enter')
                result = original(*args, **kwargs)
                recorder.record('turn.' + method, start, perf_counter_ns(), keys=[], cpu_ns=thread_time_ns()-cpu, **f)
                state(method + '.exit')
                return result
        setattr(obj, method, call)
    for method in ('accept_prepare', '_launch'):
        if method == '_launch':
            original = lane._launch
            def launch(plan, batch, pins, *args, original=original, **kwargs):
                state('launch.' + plan.direction)
                return original(plan, batch, pins, *args, **kwargs)
            lane._launch = launch
        else:
            wrap(lane, method)
    for method in ('prepare_async', 'retire', 'drain_complete', 'complete_h2d', 'complete_import'):
        wrap(banks, method)
    if not target:
        for method in ('begin_compute', 'end_compute'):
            wrap(banks.sessions, method)
    # Record failed import/export checks only on reason/request transitions.
    facts = lane.host_facts if target else lane.facts
    for method in ('_start_h2d', '_start_import', '_start_d2h', '_start_export'):
        original = getattr(lane, method, None)
        if original is None:
            continue
        def attempt(*args, original=original, method=method, **kwargs):
            begin, cpu = perf_counter_ns(), thread_time_ns()
            result = original(*args, **kwargs)
            emit_changed('attempt.' + method, (snapshot(), result), begin_ns=begin, cpu_ns=thread_time_ns()-cpu)
            return result
        setattr(lane, method, attempt)
    # Existing calls only: no extra source reads or pin acquisition.
    original_source = facts.source_ready
    def source(*args, **kwargs):
        result = original_source(*args, **kwargs)
        if result is None:
            row = args[0]
            emit_changed('missing_source', (bank_key(lane._pending.compiled.command if target and lane._pending else lane._pending),
                row.request_slot, row.request_epoch, getattr(row, 'snapshot_version', getattr(row, 'source_host_version', None))))
        return result
    facts.source_ready = source
    state('attach')


def attach_execution(service):
    if service.metrics.profile is None:
        return
    from nebulasd.workers.execution.protocol import Kind
    from nebulasd.workers.execution.wire import Reader
    original = service._execute
    async def execute(message, activated=None):
        start = perf_counter_ns()
        key = None
        if message.kind in (Kind.PREPARE, Kind.RUN, Kind.RUN_START, Kind.PREFILL, Kind.RETIRE, Kind.ACTIVATE):
            key = Reader(message.payload).fields('IQQ')
        service.metrics.record('turn.execution.start', start, start, operation=message.operation,
                               kind=message.kind.name, bank_key=key)
        result = await original(message, activated)
        end = perf_counter_ns()
        service.metrics.record('turn.execution', start, end, operation=message.operation,
                               kind=message.kind.name, bank_key=key)
        return result
    service._execute = execute


def attach_engine(engine, recorder):
    original = engine.scheduling_progress.dispatch_work
    def dispatch(command, plan, import_plan):
        start, cpu = perf_counter_ns(), thread_time_ns()
        try:
            return original(command, plan, import_plan)
        finally:
            recorder.record('turn.dispatch', start, perf_counter_ns(), keys=[],
                worker=command.worker_id, command_seq=command.command_seq,
                kind=command.kind.name, cpu_ns=thread_time_ns()-cpu)
    engine.scheduling_progress.dispatch_work = dispatch
