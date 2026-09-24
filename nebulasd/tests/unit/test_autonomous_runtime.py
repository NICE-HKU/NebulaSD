from concurrent.futures import Future
from types import SimpleNamespace
from dataclasses import replace
from nebulasd.workers.runtime import Runtime
from nebulasd.workers.banks import Banks
from nebulasd.workers.dependencies import Dependencies
from nebulasd.workers.work import WorkKind, TableDependency, Selector, Outcome
from nebulasd.core.enums import StateChangeBlockKind as K, Lifecycle
from nebulasd.table.storage import RequestSchedulingTable
from nebulasd.table.writers import EngineTableWriter
from test_autonomous_work import work


# Explicit in-process control harness. Execution Runtime has no observer.
observers = {}

def make_runtime(banks, dependencies, dma, role, **kwargs):
    runtime = Runtime(banks, dma, role, **kwargs)
    dependencies.profile = kwargs.get('profile', False)
    observers[runtime] = dependencies
    return runtime

def accept(runtime, work):
    accepted = runtime.accept(work)
    if accepted:
        observers[runtime].register(work)
    return accepted

def step(runtime):
    from nebulasd.workers.input_facts import InputFact
    for event in observers[runtime].poll():
        work = runtime.records[event.key[0]].spec
        runtime.receive_inputs(work.work_seq, dict(worker_id=work.worker_id,
            worker_generation=work.worker_generation, events=(InputFact.capture(work, event),)))
    return runtime.step()

def shutdown(runtime):
    observers[runtime].clear()
    runtime.shutdown()


class Jobs:
    def __init__(self):
        self.jobs = {}
    def job(self, kind, key):
        future = Future()
        self.jobs[kind, key] = future
        return future
    def compile_import(self, spec, captured, layout, live):
        return self.job('input', spec.work_seq)
    def compile_compute(self, spec, captured, layout, live):
        return self.job('plan', spec.work_seq)
    def write_metadata(self, layout, inputs):
        return self.job('metadata', layout.bank_id)
    def execute(self, plan):
        return self.job('compute', plan)
    def submit(self, bank, plan):
        return self.job(plan, bank)
    def finish(self, kind, key, value=None):
        self.jobs[kind, key].set_result(value)


def prepare(runtime, jobs, w):
    assert accept(runtime, w)
    step(runtime)
    jobs.finish('input', w.work_seq, SimpleNamespace(copy_plan=lambda _: 'h2d'))
    jobs.finish('plan', w.work_seq, w.work_seq)
    step(runtime)
    jobs.finish('metadata', w.bank_id, object())
    step(runtime)


def test_slow_bank_h2d_does_not_block_other_d2h_free_or_pending_work():
    jobs = Jobs()
    runtime = make_runtime(Banks(16, 2), Dependencies(RequestSchedulingTable(2)), jobs, jobs)
    a, b = work(), replace(work(2, 1), rows=(replace(work().rows[0], slot=1),))
    prepare(runtime, jobs, a)
    jobs.finish('h2d', 0)
    step(runtime)
    prepare(runtime, jobs, b)
    assert not jobs.jobs['h2d', 1].done()
    jobs.finish('compute', 1, SimpleNamespace(executed_rows=(0,), export_plan='d2h'))
    step(runtime)
    assert ('d2h', 0) in jobs.jobs
    assert not jobs.jobs['h2d', 1].done()
    jobs.finish('d2h', 0)
    step(runtime)
    assert runtime.records[1].physical_done
    assert runtime.banks.banks[0].layout is None
    # Keep old result unconsumed; physical Bank reuse proceeds anyway.
    c = work(3, 0, 2)
    assert accept(runtime, c)
    step(runtime)
    assert runtime.banks.banks[0].layout.bank_epoch == 2
    assert runtime.records[1].outcomes == [Outcome.EXECUTED]
    assert not jobs.jobs['h2d', 1].done()


def test_finished_without_future_payload_retires_no_fake_compute_and_recycles():
    table = RequestSchedulingTable(1)
    EngineTableWriter(table).publish_active(slot=0, publish_seq=0, request_epoch=1,
        current_round_id=0, classified_result_ticket=1, arrival_seq=0, prompt_token_count=4,
        max_new_tokens=1, spec_token_limit=4, lifecycle=Lifecycle.FINISHED)
    w = work()
    row = replace(w.rows[0], owner_epoch=7, layout_id=19, predecessor=TableDependency(K.REQUEST_TARGET_COMPUTE, 0, 1, 0, Selector.DELTA),
        classified=TableDependency(K.REQUEST_ENGINE, 0, 1, 1, Selector.CLASSIFIED))
    w = replace(w, operation=WorkKind.DRAFT_INITIAL, rows=(row,))
    jobs = Jobs()
    runtime = make_runtime(Banks(16, 2), Dependencies(table), jobs, jobs, profile=True)
    assert accept(runtime, w)
    runtime.drain()
    step(runtime)
    state = runtime.records[1]
    assert state.physical_done
    assert state.outcomes == [Outcome.SKIPPED_FINISHED]
    assert not jobs.jobs
    assert not any(kind in ('COMPUTE_DONE', 'RESULT_READY', 'H2D_DONE') for kind, _, _ in state.facts)
    assert not runtime.quiescent
    runtime.publication_retired(1)
    assert runtime.quiescent
    runtime.recycle()
    assert accept(runtime, work(2, 0, 2))


def test_finished_retires_started_h2d_before_reusing_rows():
    table = RequestSchedulingTable(1)
    w = work()
    w = replace(w, operation=WorkKind.DRAFT_DECODE, rows=(replace(w.rows[0], owner_epoch=8, layout_id=19,
        source=TableDependency(K.REQUEST_DRAFT_D2H, 0, 1, 1, Selector.DRAFT_HOST),
        predecessor=TableDependency(K.REQUEST_TARGET_COMPUTE, 0, 1, 0, Selector.DELTA),
        classified=TableDependency(K.REQUEST_ENGINE, 0, 1, 1, Selector.CLASSIFIED)),))
    publish_source(table, draft=True)
    jobs = Jobs()
    runtime = make_runtime(Banks(16, 1), Dependencies(table), jobs, jobs)
    assert accept(runtime, w)
    step(runtime)
    jobs.finish('input', 1, SimpleNamespace(copy_plan=lambda _: 'h2d'))
    step(runtime)
    jobs.finish('metadata', 0, object())
    step(runtime)
    assert not jobs.jobs['h2d', 0].done()
    EngineTableWriter(table).publish_active(slot=0, publish_seq=0, request_epoch=1,
        current_round_id=0, classified_result_ticket=1, arrival_seq=0, prompt_token_count=4,
        max_new_tokens=1, spec_token_limit=4, lifecycle=Lifecycle.FINISHED)
    step(runtime)
    assert not runtime.records[1].physical_done
    assert not runtime.banks.free_rows
    jobs.finish('h2d', 0)
    step(runtime)
    assert runtime.records[1].physical_done
    assert runtime.banks.free_rows == {0}
    assert ('compute', 1) not in jobs.jobs


def test_failed_job_is_fail_stop_and_never_releases_referenced_layout():
    import pytest
    jobs = Jobs()
    runtime = make_runtime(Banks(16, 1), Dependencies(RequestSchedulingTable(1)), jobs, jobs)
    assert accept(runtime, work())
    step(runtime)
    jobs.jobs['input', 1].set_exception(RuntimeError('injected metadata input failure'))
    with pytest.raises(RuntimeError, match='injected'):
        step(runtime)
    assert not runtime.records[1].physical_done
    assert not runtime.banks.free_rows


def test_impossible_row_capacity_rejected_before_acceptance():
    import pytest
    jobs = Jobs()
    runtime = make_runtime(Banks(32, 1), Dependencies(RequestSchedulingTable(2)), jobs, jobs)
    w = work()
    w = replace(w, rows=(w.rows[0], replace(w.rows[0], slot=1, destination_offset=8)))
    with pytest.raises(ValueError, match='total physical row'):
        accept(runtime, w)
    assert not runtime.records


def publish_source(table, draft=False):
    from nebulasd.table.storage import FieldValue
    table._publish_owned(owner='draft_source_copy_lane' if draft else 'target_copy_lane',
        block_kind=K.REQUEST_DRAFT_D2H if draft else K.REQUEST_D2H, row=0, publish_seq=0,
        fields=tuple(FieldValue(k, v) for k, v in dict(request_epoch=1, ready_version=1, status=2).items()))


def successor(bank):
    w = work(2, bank)
    return replace(w, operation=WorkKind.TARGET_VERIFY, rows=(replace(w.rows[0], round_id=1,
        source=TableDependency(K.REQUEST_D2H, 0, 1, 1, Selector.TARGET_HOST),
        predecessor=TableDependency(K.REQUEST_DRAFT, 0, 1, 1, Selector.PROPOSAL),
        classified=TableDependency(K.REQUEST_ENGINE, 0, 1, 1, Selector.CLASSIFIED)),))


import pytest


@pytest.mark.parametrize('producer_bank', [0, 1])
def test_successor_cannot_steal_only_row_from_source_producer(producer_bank):
    jobs = Jobs()
    table = RequestSchedulingTable(1)
    runtime = make_runtime(Banks(16, 1), Dependencies(table), jobs, jobs)
    assert accept(runtime, work(1, producer_bank))
    assert accept(runtime, successor(1-producer_bank))
    step(runtime)
    assert runtime.records[1].layout is not None
    assert runtime.records[2].layout is None
    assert ('input', 1) in jobs.jobs
    jobs.finish('input', 1, SimpleNamespace(copy_plan=lambda _: None))
    jobs.finish('plan', 1, 1)
    step(runtime)
    jobs.finish('metadata', producer_bank, object())
    step(runtime)
    jobs.finish('compute', 1, SimpleNamespace(executed_rows=(0,), export_plan='d2h'))
    step(runtime)
    jobs.finish('d2h', producer_bank)
    step(runtime)
    publish_source(table)
    step(runtime)
    # Early import survives: neither proposal nor classification has arrived.
    assert runtime.records[2].layout is not None
    assert ('input', 2) in jobs.jobs
    assert ('plan', 2) not in jobs.jobs


def test_shutdown_missing_inputs_retires_but_normal_drain_waits():
    jobs = Jobs()
    runtime = make_runtime(Banks(16, 1), Dependencies(RequestSchedulingTable(1)), jobs, jobs)
    assert accept(runtime, successor(0))
    runtime.drain()
    step(runtime)
    assert not runtime.records[2].physical_done
    shutdown(runtime)
    step(runtime)
    assert runtime.records[2].outcomes == [Outcome.SKIPPED_SHUTDOWN]
    assert runtime.records[2].physical_done
    assert not jobs.jobs and not observers[runtime].pending
    runtime.publication_retired(2)
    assert runtime.quiescent
    with pytest.raises(RuntimeError, match='terminal'):
        runtime.recycle()


@pytest.mark.parametrize('stage', ['metadata', 'h2d', 'compute', 'd2h'])
def test_shutdown_waits_real_jobs_and_finishes_started_compute(stage):
    jobs = Jobs()
    runtime = make_runtime(Banks(16, 1), Dependencies(RequestSchedulingTable(1)), jobs, jobs)
    assert accept(runtime, work())
    step(runtime)
    jobs.finish('input', 1, SimpleNamespace(copy_plan=lambda _: 'h2d'))
    jobs.finish('plan', 1, 1)
    step(runtime)
    if stage != 'metadata':
        jobs.finish('metadata', 0, object())
        step(runtime)
    if stage in ('compute', 'd2h'):
        jobs.finish('h2d', 0)
        step(runtime)
    result = SimpleNamespace(executed_rows=(0,), export_plan='d2h')
    if stage == 'd2h':
        jobs.finish('compute', 1, result)
        step(runtime)
    shutdown(runtime)
    step(runtime)
    assert not runtime.records[1].physical_done
    assert not runtime.banks.free_rows
    if stage == 'metadata':
        jobs.finish('metadata', 0, object())
    elif stage == 'h2d':
        jobs.finish('h2d', 0)
    elif stage == 'compute':
        jobs.finish('compute', 1, result)
        step(runtime)
        assert ('d2h', 0) in jobs.jobs
        jobs.finish('d2h', 0)
    else:
        jobs.finish('d2h', 0)
    step(runtime)
    assert runtime.records[1].physical_done
    expected = Outcome.EXECUTED if stage in ('compute', 'd2h') else Outcome.SKIPPED_SHUTDOWN
    assert runtime.records[1].outcomes == [expected]


def test_profile_off_has_no_trace_clock_or_retained_diagnostic_objects(monkeypatch):
    from nebulasd.workers import runtime as module
    calls = []
    monkeypatch.setattr(module, 'perf_counter_ns', lambda: calls.append(123) or 123)
    jobs = Jobs()
    runtime = make_runtime(Banks(16, 2), Dependencies(RequestSchedulingTable(1)), jobs, jobs)
    prepare(runtime,jobs,work())
    receipt = SimpleNamespace(submitted_ns=11)
    jobs.finish('h2d',0,receipt)
    step(runtime)
    assert calls == []
    assert runtime.records[1].facts is None
    assert runtime.records[1].h2d_receipt is receipt
    jobs.finish('compute',1,SimpleNamespace(executed_rows=(0,),export_plan='d2h'))
    step(runtime)
    exported = SimpleNamespace(submitted_ns=22)
    jobs.finish('d2h',0,exported)
    step(runtime)
    assert calls == [123]  # Completion ABI still needs a physical retirement timestamp.
    assert runtime.records[1].physical_done_ns == 123
    assert runtime.records[1].d2h_receipt is exported
    assert runtime.records[1].facts is None
