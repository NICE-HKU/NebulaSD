"""Input responsibility boundary: real codec/control loop, execution has no table."""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from queue import Empty
from threading import Thread, Event
from time import monotonic, sleep
from types import SimpleNamespace as NS
import pytest

from nebulasd.core.enums import Lifecycle, StateChangeBlockKind as K
from nebulasd.core.handles import ArenaHandle
from nebulasd.table.writers import EngineTableWriter
from nebulasd.workers.input_facts import InputFact, FIELDS, FACT_BUDGET
from nebulasd.workers.channel import LocalChannel, encode, decode
from nebulasd.workers.dependencies import Dependencies
from nebulasd.workers.runtime import Runtime
from nebulasd.workers.banks import Banks
from nebulasd.workers.work import Selector, Outcome
from test_autonomous_runtime import Jobs, successor, publish_source
from test_autonomous_target_control import resources
from test_autonomous_work import work


def fact(name, selector, **values):
    return InputFact(0, name, selector, values)


def deliver(runtime, w, *events, **identity):
    runtime.receive_inputs(w.work_seq, dict(worker_id=w.worker_id,
        worker_generation=w.worker_generation, events=events) | identity)


@pytest.mark.parametrize('selector', list(Selector))
def test_compact_input_codec_all_selectors_and_truncations(selector):
    values = {n: ArenaHandle(117, 8, 71) if n.endswith('_handle') else 37 for n in FIELDS[selector]}
    name = 'classified' if selector == Selector.CLASSIFIED else 'source' if selector in (Selector.TARGET_HOST, Selector.DRAFT_HOST) else 'predecessor'
    data = dict(worker_id=9, worker_generation=37, events=(InputFact(255, name, selector, values, 123),))
    raw = encode('INPUTS', data)
    assert decode('INPUTS', raw) == data
    for n in range(len(raw)):
        with pytest.raises(ValueError):
            decode('INPUTS', raw[:n])
    with pytest.raises(ValueError):
        decode('INPUTS', raw+b'\0')
    with pytest.raises(ValueError):
        encode('INPUTS', data | {'events': data['events']*(FACT_BUDGET+1)})
    channel = LocalChannel()
    try:
        channel.put_nowait(('WORK', work(91).to_bytes()))
        channel.put_nowait(('INPUTS', 91, data))
        assert channel.get_nowait()[0] == 'WORK'
        assert channel.get_nowait() == ('INPUTS', 91, data)
    finally:
        channel.close(unlink=True)


def test_source_fact_starts_real_runtime_h2d_before_delta_and_merge_compute():
    jobs = Jobs()
    runtime = Runtime(Banks(16, 1), jobs, jobs)
    w = successor(0)
    assert runtime.accept(w)
    assert not hasattr(runtime, 'dependencies')
    runtime.step()
    assert not jobs.jobs
    deliver(runtime, w, fact('source', Selector.TARGET_HOST, logical_kv_len=4, ready_version=1))
    runtime.step()
    assert ('input', 2) in jobs.jobs and ('plan', 2) not in jobs.jobs
    jobs.finish('input', 2, NS(copy_plan=lambda _: 'h2d'))
    runtime.step()
    jobs.finish('metadata', 0, object())
    runtime.step()
    assert ('h2d', 0) in jobs.jobs and ('plan', 2) not in jobs.jobs
    deliver(runtime, w, fact('predecessor', Selector.PROPOSAL, proposal_handle=ArenaHandle(0, 8, 37)),
            fact('classified', Selector.CLASSIFIED, lifecycle=Lifecycle.ACTIVE))
    runtime.step()
    jobs.finish('plan', 2, 2)
    jobs.finish('h2d', 0)
    runtime.step()
    assert ('compute', 2) in jobs.jobs
    jobs.finish('compute', 2, NS(executed_rows=(0,), export_plan='d2h'))
    runtime.step()
    jobs.finish('d2h', 0)
    runtime.step()
    assert runtime.records[2].physical_done and runtime.banks.free_rows == {0}
    # No control/publication acknowledgement authorized the above free.
    runtime.publication_retired(2)
    with pytest.raises(ValueError, match='retired'):
        deliver(runtime, w, fact('source', Selector.TARGET_HOST))


@pytest.mark.parametrize('invalid', ['worker', 'generation', 'member', 'selector', 'unknown', 'duplicate'])
def test_fact_boundary_identity_member_selector_and_duplicate(invalid):
    jobs = Jobs(); runtime = Runtime(Banks(16, 1), jobs, jobs); w = successor(0)
    runtime.accept(w)
    event = fact('source', Selector.TARGET_HOST, ready_version=1, logical_kv_len=4)
    identity = {}
    if invalid == 'worker': identity['worker_id'] = w.worker_id+1
    if invalid == 'generation': identity['worker_generation'] = w.worker_generation+1
    if invalid == 'member': event = replace(event, index=1)
    if invalid == 'selector': event = replace(event, selector=Selector.DRAFT_HOST)
    if invalid == 'unknown': w = replace(w, work_seq=999)
    if invalid == 'duplicate': deliver(runtime, w, event)
    with pytest.raises(ValueError):
        deliver(runtime, w, event, **identity)


def test_partial_finished_late_facts_and_shutdown_retirement():
    jobs = Jobs(); runtime = Runtime(Banks(32, 2), jobs, jobs); w = successor(0)
    row = w.rows[0]
    second = replace(row, slot=1, destination_offset=8,
        source=replace(row.source, slot=1), predecessor=replace(row.predecessor, slot=1),
        classified=replace(row.classified, slot=1))
    w = replace(w, rows=(row, second))
    runtime.accept(w)
    deliver(runtime, w, fact('classified', Selector.CLASSIFIED, lifecycle=Lifecycle.FINISHED),
        fact('source', Selector.TARGET_HOST, ready_version=1, logical_kv_len=4))
    runtime.step()
    assert runtime.records[2].outcomes == [Outcome.SKIPPED_FINISHED, None]
    assert not jobs.jobs
    runtime.shutdown()
    deliver(runtime, w, replace(fact('source', Selector.TARGET_HOST), index=1))
    runtime.step()
    assert runtime.records[2].outcomes == [Outcome.SKIPPED_FINISHED, Outcome.SKIPPED_SHUTDOWN]
    runtime.publication_retired(2)
    assert runtime.quiescent


def classify(table, *, ticket=1, lifecycle=Lifecycle.ACTIVE):
    EngineTableWriter(table).publish_active(slot=0, publish_seq=0, request_epoch=1,
        current_round_id=0, classified_result_ticket=ticket, arrival_seq=0,
        prompt_token_count=4, max_new_tokens=16, spec_token_limit=4, lifecycle=lifecycle)


def test_finished_removes_absent_inputs_and_observer_tombstones_are_bounded(resources):
    _, table, *_ = resources
    classify(table, ticket=0, lifecycle=Lifecycle.FINISHED)
    deps = Dependencies(table)
    for seq in range(100):
        deps.register(replace(successor(0), work_seq=seq))
        events = deps.poll(1)
        assert len(events) == 1 and events[0].key[2] == 'classified'
        assert not deps.pending
        deps.retire(seq)
        assert not deps.order


def test_profile_off_observation_has_no_diagnostic_clock(resources, monkeypatch):
    _, table, *_ = resources
    classify(table)
    from nebulasd.workers import dependencies
    monkeypatch.setattr(dependencies, 'perf_counter_ns', lambda: pytest.fail('profile-off clock'))
    deps = Dependencies(table); deps.register(successor(0))
    events = deps.poll()
    assert len(events) == 1 and events[0].observed_ns == 0


@contextmanager
def control(options):
    """Actual controller, with the test driving the execution ends of its rings."""
    from nebulasd.workers.target.control import run_control
    with ExitStack() as stack:
        channels = [LocalChannel() for _ in range(3)]
        for c in channels: stack.callback(c.close, unlink=True)
        incoming, commands, results = channels
        wake, execution_wake, ready, stopped = (Event() for _ in range(4))
        errors = []
        def run():
            try:
                run_control(options, *(c.descriptor for c in channels), wake, execution_wake, ready, stopped)
            except BaseException as error:
                errors.append(error)
        thread = Thread(target=run)
        thread.start()
        assert ready.wait(3)
        try:
            yield NS(incoming=incoming, commands=commands, results=results,
                     wake=wake, errors=errors)
        finally:
            stopped.set(); wake.set(); thread.join(3)
            assert not thread.is_alive()
            # Deliberate teardown may interrupt an outstanding test WORK.
            assert all(str(e) == 'execution exited before protocol retirement' for e in errors), errors


def get(channel, timeout=3):
    deadline = monotonic()+timeout
    while monotonic() < deadline:
        try: return channel.get_nowait()
        except Empty: sleep(.001)
    raise TimeoutError('missing control message')


def test_control_work_immediate_source_independent_and_empty_ring_polling(resources):
    options, table, *_ = resources
    with control(options) as c:
        w = successor(0)
        c.incoming.put_nowait(('WORK', w.to_bytes())); c.wake.set()
        assert get(c.commands) == ('WORK', w.to_bytes())
        with pytest.raises(Empty): c.commands.get_nowait()
        publish_source(table)
        # No command or producer wakeup: control must discover the shared level.
        msg = get(c.commands)
        assert msg[0:2] == ('INPUTS', 2)
        assert [e.name for e in msg[2]['events']] == ['source']
        classify(table)
        assert [e.name for e in get(c.commands)[2]['events']] == ['classified']
        assert not c.errors


def test_control_full_fact_channel_still_publishes_and_retains_fact(resources):
    options, table, _, completions = resources
    with control(options | {'publication_event_budget': 1}) as c:
        w = successor(0)
        c.incoming.put_nowait(('WORK', w.to_bytes())); c.wake.set()
        assert get(c.commands)[0] == 'WORK'
        # Accept an independent prefill before saturating the private ring.
        from test_autonomous_target_control import allocated_work
        b = replace(allocated_work(resources),work_seq=3,bank_id=1,completion_offset=256,
                    rows=(replace(allocated_work(resources).rows[0],slot=1,prompt_count=4),))
        c.incoming.put_nowait(('WORK', b.to_bytes())); c.wake.set()
        assert get(c.commands)[0] == 'WORK'
        # Fill only while no inputs/credits can be sent by the controller.
        for _ in range(c.commands.CAPACITY): c.commands.put_nowait(('DRAIN', None))
        publish_source(table)
        sleep(.02)
        c.results.put_nowait(('RESULT', 3, dict(compute_start_ns=1, compute_end_ns=2,
            rows=[dict(index=0,tokens=(42,),accepted=0,logical=4,version=1,dirty_begin=0,dirty_blocks=1)])))
        c.results.put_nowait(('PHYSICAL', 3, dict(d2h_submitted_ns=3,observed_ns=4,outcomes=(1,),facts=())))
        c.wake.set()
        deadline=monotonic()+3
        while completions.read(256) is None and monotonic()<deadline:sleep(.001)
        assert completions.read(256).physical_done_ns == 4
        for _ in range(c.commands.CAPACITY): assert get(c.commands)[0] == 'DRAIN'
        assert get(c.commands) == ('RETIRE_RECORD', 3)
        msg = get(c.commands)
        assert msg[0:2] == ('INPUTS', 2) and msg[2]['events'][0].name == 'source'
        assert not c.errors


def test_observer_streams_large_work_in_bounded_frames():
    from nebulasd.table.storage import RequestSchedulingTable, FieldValue
    table = RequestSchedulingTable(96)
    w = successor(0); row = w.rows[0]
    w = replace(w, rows=tuple(replace(row, slot=i, destination_offset=8*i,
        source=replace(row.source, slot=i), predecessor=replace(row.predecessor, slot=i),
        classified=replace(row.classified, slot=i)) for i in range(96)))
    for i in range(96):
        table.partition(K.REQUEST_D2H)._publish(i, 0, tuple(FieldValue(k, v) for k, v in
            dict(request_epoch=1, ready_version=1, logical_kv_len=4, status=2).items()))
    deps = Dependencies(table); deps.register(w)
    events = []
    for _ in range(5):
        batch = deps.poll(FACT_BUDGET)
        assert len(batch) <= FACT_BUDGET
        events.extend(batch)
    assert len(events) == 96 and len({e.key for e in events}) == 96
    assert all(e.key[2] == 'source' for e in events)
    assert len(deps.pending) == 192  # No waiting for those inputs to send source.
    deps.retire(w.work_seq)
    assert not deps.pending and not deps.order


@pytest.mark.parametrize('epoch,ticket,error', [(2,1,'cohort'), (1,2,'overwritten')])
def test_control_gate_rejects_stale_lifetimes(resources, epoch, ticket, error):
    from nebulasd.table.storage import FieldValue
    _, table, *_ = resources
    table.partition(K.REQUEST_D2H)._publish(0,0,tuple(FieldValue(k,v) for k,v in
        dict(request_epoch=epoch,ready_version=ticket,status=2).items()))
    deps = Dependencies(table); deps.register(successor(0))
    with pytest.raises(RuntimeError, match=error): deps.poll()


def test_uncaptured_ready_table_does_not_advance_execution(resources):
    _, table, *_ = resources
    jobs = Jobs(); runtime = Runtime(Banks(16,1), jobs, jobs); w = successor(0)
    runtime.accept(w)
    publish_source(table)
    for _ in range(5): runtime.step()
    assert not jobs.jobs  # Control busy/unavailable: no execution-side fallback.
    deps = Dependencies(table); deps.register(w)
    for event in deps.poll(): deliver(runtime,w,InputFact.capture(w,event))
    runtime.step()
    assert ('input',w.work_seq) in jobs.jobs
