"""Engine protocol boundary tests with explicit in-process transport doubles."""
from types import SimpleNamespace
from dataclasses import replace
from queue import Empty, Full
import pytest
from nebulasd.core.enums import WorkerRole, StateChangeBlockKind as K
from nebulasd.engine.resources import ControlResources
from nebulasd.engine.core import Engine
from nebulasd.scheduler.views import WorkerSpec, value as v
from nebulasd.scheduler.completion import CompletionScheduler
from nebulasd.scheduler.measured_placement import MeasuredPlacementEstimator
from nebulasd.table.prepared import PreparedRow
from nebulasd.workers.work import Outcome
from nebulasd.workers.completion import WorkCompletion, MemberCompletion
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from test_stage_planner import estimator


class Endpoint:
    def __init__(self):
        self.work = []
        self.blocked = False
    def submit(self, work):
        if self.blocked:
            raise Full
        self.work.append(work)


@pytest.fixture
def engine(tmp_path):
    specs = (WorkerSpec(0,WorkerRole.DRAFT,draft_banked=True),WorkerSpec(1,WorkerRole.TARGET))
    resources = ControlResources(specs,slots=8,host_blocks=128)
    sup = SimpleNamespace(autonomous=True,pairs={i:Endpoint() for i in (0,1)},check=lambda:None,
        pump=lambda *a:False,close=lambda:None,output_dir=tmp_path,
        bell=SimpleNamespace(drain=lambda:None),management={},command_bells={})
    e = Engine(resources,sup,scheduler=CompletionScheduler(estimator=estimator()))
    for w in specs:
        PreparedRow(resources.registry.partition(K.WORKER_COMMON),w.worker_id,
            dict(worker_id=w.worker_id,worker_generation=w.generation,status=1,role=w.role),()).publish(())
        kind = K.WORKER_DRAFT_BANK if w.role == WorkerRole.DRAFT else K.WORKER_BANK
        for bank in (0,1):
            PreparedRow(resources.registry.partition(kind),w.worker_id*2+bank,
                dict(bank_id=bank,bank_epoch=0,state=0,role=2,capacity_rows=8,capacity_blocks=128,alloc_rows=0),()).publish(())
    yield e
    e.close()


def test_first_decision_full_work_reservation_and_retry_identity(engine):
    e=engine
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4))
    e.supervisor.pairs[1].blocked=True
    e.step()
    pending=e.scheduling_progress.pending['T'][0][0]
    head=e.resources.completions.next_offset
    assert pending.rows[0].output == e.registry.records[0].reservation
    assert pending.bank_epoch == 1
    for _ in range(3):
        e.step()
    assert e.resources.completions.next_offset == head
    assert not e.ledger.records
    e.supervisor.pairs[1].blocked=False
    e.step()
    assert e.supervisor.pairs[1].work == [pending]
    assert e.rows[K.REQUEST_DISPATCH,0].get('target_run_seq') == pending.rows[0].run_seq


def test_two_free_standby_banks_reservations_and_monotone_epoch(engine):
    e=engine
    for i in range(8):
        e.admit(str(i),(1,2,3),DraftGenerationConfig(8,4))
    e.step();e.step()
    works=e.supervisor.pairs[1].work
    assert len(works)==2 and {w.bank_id for w in works}=={0,1}
    assert all(w.bank_epoch==1 for w in works)
    assert len(e.ledger.records)==2
    assert e.ledger.capacity(e._scheduling_view(),e.resources.specs[1]) is None


def test_physical_before_output_keeps_result_identity(engine):
    e=engine
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    work=e.supervisor.pairs[1].work[0]
    completion=WorkCompletion(work.worker_generation,work.work_seq,work.bank_epoch,100,work.bank_id,
        tuple(MemberCompletion(r.slot,r.epoch,r.round_id,Outcome.EXECUTED) for r in work.rows))
    e.resources.completions.publish(work.completion_offset,completion)
    e.ledger.observe_completions();e.ledger.refresh(e.rows)
    assert e.ledger.records
    row=SimpleNamespace(block_kind=K.REQUEST_TARGET_COMPUTE,row=0,get=lambda n:dict(status=2,result_code=0,request_epoch=1,round_id=0,observed_run_seq=1,
        target_id=1,target_generation=1,bank_id=work.bank_id,bank_epoch=work.bank_epoch)[n])
    e.ledger.result_applied(e.ledger.validate_fact(row), None);e.ledger.observe_completions();e.ledger.refresh(e.rows)
    assert not e.ledger.records


def test_skipped_finished_retires_without_compute_result(engine):
    e=engine
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    work=e.supervisor.pairs[1].work[0]
    e.resources.completions.publish(work.completion_offset,WorkCompletion(1,work.work_seq,work.bank_epoch,100,work.bank_id,
        (MemberCompletion(0,1,0,Outcome.SKIPPED_FINISHED),)))
    e._observe_facts()
    e.ledger.observe_completions();e.ledger.refresh(e.rows)
    assert not e.ledger.records and not e.ledger.request_index
    assert e.scheduler.compute_times.end_ns(0, 1, 'T', 0) is None


def publish_target(engine, work):
    from nebulasd.workers.target.publication import TargetPublisher
    e=engine
    p=TargetPublisher(e.resources.table,e.resources.completions,block_bytes=16, outputs=e.resources.token_router, configs=e.resources.configs)
    # This is an explicit CPU test double for the control process writer.
    p.reserve(work)
    result=('RESULT',work.work_seq,dict(rows=[dict(index=i,tokens=(4,),accepted=0,logical=3,
        version=1,dirty_begin=0,dirty_blocks=1) for i,r in enumerate(work.rows)],compute_start_ns=10,compute_end_ns=20))
    physical=('PHYSICAL',work.work_seq,dict(outcomes=[1]*len(work.rows),observed_ns=30,d2h_submitted_ns=21))
    p.consume(result);p.consume(physical)
    while p.records: p.step(8)


def test_initial_and_continuation_selectors_issued_before_target_result(engine):
    from nebulasd.workers.draft.publication import DraftPublisher
    e=engine
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    target=e.supervisor.pairs[1].work[0]
    publish_target(e,target);e.step()
    assert e.ledger.key(target) not in e.ledger.records
    assert e.scheduler.compute_times.end_ns(0, 1, 'T', 0) == 20
    initial=e.supervisor.pairs[0].work[0]
    row=initial.rows[0]
    # TARGET_DECISION names the Target round, unlike legacy CLASSIFIED's round+1.
    assert row.predecessor.expected_ticket==0 and row.classified.expected_ticket==0 and row.source is None
    assert row.output.length==8*4 and row.output==e.registry.records[0].reservation
    # Publish a real binary snapshot using a CPU-supplied compact model result.
    e.resources.proposals[0]._writer=True;e.resources.snapshots[0]._writer=True
    pub=DraftPublisher(e.resources.table,e.resources.proposals[0],e.resources.snapshots[0],e.resources.completions,
        host=e.resources.draft_host.descriptor)
    pub.reserve(initial)
    result=('DRAFT_RESULT',initial.work_seq,dict(rows=[dict(index=0,proposal=(5,6,7,8),proposal_kind=1,
        logical=7,version=1,dirty_begin=0,dirty_blocks=1,committed_count=1)],compute_start_ns=40,compute_end_ns=50))
    physical=('PHYSICAL',initial.work_seq,dict(outcomes=[1],observed_ns=60,d2h_submitted_ns=51))
    pub.consume(result);pub.consume(physical)
    while pub.records: pub.step(8)
    e.step()
    assert e.ledger.key(initial) not in e.ledger.records
    assert e.scheduler.compute_times.end_ns(0, 1, 'D', row.round_id) == 50
    assert e.scheduler.compute_times.end_ns(0, 1, 'T', 0) == 20
    verify=e.supervisor.pairs[1].work[-1]
    assert verify.operation.name=='TARGET_VERIFY'
    continuation=e.supervisor.pairs[0].work[-1]
    assert continuation.operation.name=='DRAFT_DECODE'
    r=continuation.rows[0]
    assert r.source.expected_ticket==1 and r.predecessor.expected_ticket==1 and r.classified.expected_ticket==1
    assert r.owner_epoch==1 and r.output==row.output
    assert v(e.rows[K.REQUEST_TARGET_COMPUTE,0],'round_id')==0  # delta round 1 is still future.


def test_same_candidates_batches_placement_with_protocol_capacity_adapter(engine):
    from nebulasd.core.enums import ComputeStatus
    from nebulasd.scheduler.completion import NumericRow
    e=engine
    for i in range(6): e.admit(str(i),(1,2,3),DraftGenerationConfig(8,4))
    e._observe_facts()
    view=e._scheduling_view()
    rows=dict(view.rows)
    for w in view.workers:
        kind=K.WORKER_DRAFT_BANK if w.role==WorkerRole.DRAFT else K.WORKER_BANK
        for i in (0,1):
            fields={f.name:f.value for f in rows[kind,w.worker_id*2+i].fields}
            rows[kind,w.worker_id*2+i]=NumericRow(fields|dict(role=1 if i==0 else 2))
        rows[K.WORKER_DRAFT_RUNTIME if w.role==WorkerRole.DRAFT else K.WORKER_TARGET_COMPUTE_RUNTIME,w.worker_id]=NumericRow(compute_status=0)
    legacy=replace(view,rows=rows,work_state=None)
    a=e.scheduler.schedule(view,phase='T');b=e.scheduler.schedule(legacy,phase='T')
    identity=lambda commands:[(c.worker_id,tuple(r.request_slot for r in c.requests)) for c in commands]
    assert identity(a)==identity(b)


def test_completed_epoch_survives_missed_free_observation(engine):
    from nebulasd.scheduler.completion import NumericRow
    e=engine
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    w=e.supervisor.pairs[1].work[0]
    rows=dict(e.rows)
    rows[K.WORKER_BANK,2+w.bank_id]=NumericRow(bank_epoch=w.bank_epoch+1,state=2)
    e.ledger.observe_completions();e.ledger.refresh(rows)
    record=next(iter(e.ledger.records.values()))
    assert not record.physical_done and record.compute_done
    assert record.completion is None  # Physical facts do not erase logical receipts.


def test_completion_admission_budget_fails_before_any_arena_mutation(engine):
    from nebulasd.engine.admission import AdmissionCapacityError
    e=engine;r=e.resources
    original=r.completions
    heads=[a._head for a in (*r.tokens,r.configs)]
    try:
        r.completions=SimpleNamespace(segment=SimpleNamespace(descriptor=SimpleNamespace(size=8)))
        with pytest.raises(AdmissionCapacityError,match='completions'):
            e.admit('a',(1,2,3),DraftGenerationConfig(8,4))
        assert heads==[a._head for a in (*r.tokens,r.configs)]
        assert not e.registry.records
    finally:
        r.completions=original


@pytest.mark.parametrize('worker', [0,1])
def test_control_projection_global_bank_rows_and_actual_compute_clock(engine,worker):
    from contextlib import ExitStack
    from nebulasd.workers.observation import Observation, Projection
    from nebulasd.workers.banks import Banks
    from nebulasd.table.native_storage import table_descriptors
    e=engine;r=e.resources;w=r.specs[worker]
    obs=Observation()
    options=dict(event_capacity=r.descriptor.state_ring_capacity,event=r.events[worker].segment.descriptor,engine_bell=None,
        global_registry=table_descriptors(r.registry),worker_count=2,observation=obs.segment.descriptor,
        worker_id=worker,worker_generation=2,role=w.role,max_batch_size=4,max_batch_tokens=256,
        blocks_per_bank=128,capacity_rows=8,host=r.host.descriptor)
    try:
        with ExitStack() as stack:
            p=Projection(options,stack)
            obs.write(SimpleNamespace(banks=Banks(128,8)),(17,100,0))
            assert p.step()
            kind=K.WORKER_DRAFT_BANK if worker==0 else K.WORKER_BANK
            assert r.registry.partition(kind).read_stable(worker*2+1).get('bank_id')==1
            runtime=r.registry.partition(K.WORKER_DRAFT_RUNTIME if worker==0 else K.WORKER_TARGET_COMPUTE_RUNTIME).read_stable(worker)
            assert runtime.get('compute_status')==1 and runtime.get('compute_start_time_ns')==100
            obs.write(SimpleNamespace(banks=Banks(128,8)),(17,100,200))
            assert p.step()
            assert r.registry.partition(runtime.block_kind).read_stable(worker).get('compute_status')==0
    finally:
        obs.close(unlink=True)


def test_completion_work_progress_ignores_copy_row_churn(engine):
    e=engine
    e._observe_facts()
    p=e.scheduling_progress
    p.advance()
    before=dict(p.schedule_counts)
    for _ in range(20):
        for kind in (K.REQUEST_H2D, K.REQUEST_DRAFT_H2D, K.REQUEST_D2H, K.REQUEST_DRAFT_D2H):
            p.observed(SimpleNamespace(block_kind=kind,row=0),None)
        p.advance()
    assert p.schedule_counts==before
    assert not any(p.dirty.values())


def test_compute_and_physical_opportunities_are_identity_deduplicated(engine):
    from nebulasd.scheduler.completion import NumericRow
    e=engine
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    w=e.supervisor.pairs[1].work[0]
    e.ledger.scheduling_events.clear()
    rows=dict(e.rows)
    rows[K.WORKER_BANK,2+w.bank_id]=NumericRow(bank_epoch=w.bank_epoch,state=1)
    for _ in range(3):e.ledger.observe_completions();e.ledger.refresh(rows)
    assert [k for k,_ in e.ledger.scheduling_events]==['compute']
    rows[K.WORKER_BANK,2+w.bank_id]=NumericRow(bank_epoch=w.bank_epoch,state=0)
    for _ in range(3):e.ledger.observe_completions();e.ledger.refresh(rows)
    assert [k for k,_ in e.ledger.scheduling_events]==['compute']
    e.resources.completions.publish(w.completion_offset, WorkCompletion(w.worker_generation,w.work_seq,w.bank_epoch,100,w.bank_id,
        tuple(MemberCompletion(r.slot,r.epoch,r.round_id,Outcome.EXECUTED) for r in w.rows)))
    for _ in range(3): e.ledger.observe_completions();e.ledger.refresh(rows)
    assert [k for k,_ in e.ledger.scheduling_events]==['compute','physical']


def test_completion_coalesces_batch_events_and_does_not_plan_without_capacity(engine):
    e=engine;p=e.scheduling_progress
    e._observe_facts();p.advance()
    before=dict(p.schedule_counts)
    for _ in range(100):p._wake('D','test')
    p.advance()
    assert p.schedule_counts['D']==before['D']+1
    assert p.schedule_counts['T']==before['T']
    for _ in range(5):p.advance()
    assert p.schedule_counts['D']==before['D']+1


def test_late_bank_free_reopens_completion_admission(engine):
    from nebulasd.scheduler.completion import NumericRow
    p=engine.scheduling_progress
    class Row(NumericRow):
        block_kind=K.WORKER_BANK
        row=2
    p.dirty={'D':set(),'T':set()}
    p.observed(Row(state=0,bank_epoch=7,alloc_rows=0),Row(state=3,bank_epoch=7,alloc_rows=1))
    assert p.dirty=={'D':set(),'T':{1}}
    p.dirty['T'].clear()
    p.observed(Row(state=0,bank_epoch=7,alloc_rows=0),Row(state=0,bank_epoch=7,alloc_rows=0))
    assert not p.dirty['T']
