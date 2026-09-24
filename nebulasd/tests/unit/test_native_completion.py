"""Same input/clock decisions, including live records and changed source fences."""
from dataclasses import replace
from types import SimpleNamespace as NS
import random
import pytest
from nebulasd.scheduler.completion import CompletionScheduler
from nebulasd.scheduler.measured_placement import MeasuredPlacementEstimator
from nebulasd.scheduler.cost_table import CostTable
from nebulasd.engine.work_ledger import WorkLedger
from nebulasd.core.enums import StateChangeBlockKind as K, WorkerRole
from test_cost_table import payload
from test_draft_placement import migration_view
from test_stage_planner import issued_target_view
from test_wp08_scheduler import patch


def make(stage, count):
    view = migration_view(count) if stage == 'D' else issued_target_view(count)
    ledger=WorkLedger(NS(specs=view.workers))
    view=replace(view,work_state=ledger)
    for w in view.workers:
        kind=K.WORKER_DRAFT_BANK if w.role==WorkerRole.DRAFT else K.WORKER_BANK
        for bank in (0,1):patch(view,kind,w.worker_id*2+bank,state=0)
    data=payload();table=CostTable(data,expected=data['compatibility'])
    scheduler=CompletionScheduler(estimator=MeasuredPlacementEstimator(table,16,draft_block_bytes=4),clock=lambda:1_000_000_000)
    return view,scheduler


@pytest.mark.parametrize('stage,prefix', [('D', 'draft'), ('T', 'target')])
def test_actual_compute_ends_update_native_cache_without_table_changes(stage, prefix, monkeypatch):
    from nebulasd.workers.work import WorkKind
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL', 'check')
    view, scheduler = make('T', 4)
    scheduler.enable_native_updates()
    before = scheduler.schedule(view, phase='T')
    r = view.requests[0]
    work = NS(operation=WorkKind.DRAFT_DECODE if stage == 'D' else WorkKind.TARGET_VERIFY,
              rows=(NS(slot=r.slot, epoch=r.epoch, round_id=3),))
    scheduler.observe_compute_result(work, dict(compute_start_ns=10, compute_end_ns=20,
                                               rows=[dict(index=0)]))
    after = scheduler.schedule(view, phase='T')
    numeric = scheduler._native.requests[r.slot][0]
    assert getattr(numeric, prefix + '_compute_round') == 3
    assert getattr(numeric, prefix + '_compute_end_ns') == 20
    # The old policy does not consume these clocks; decisions remain identical.
    assert after == before
    scheduler.reset_compute_times()
    scheduler.schedule(view, phase='T')
    assert getattr(numeric, prefix + '_compute_end_ns') == -1
    scheduler.observe_compute_result(work, dict(compute_start_ns=10, compute_end_ns=20,
                                               rows=[dict(index=0)]))
    view.requests[0] = replace(r, epoch=r.epoch+1)
    scheduler.schedule(view, phase='T')
    assert getattr(numeric, prefix + '_compute_end_ns') == -1


@pytest.mark.parametrize('stage',['D','T'])
@pytest.mark.parametrize('seed',range(30))
def test_same_decisions_with_varied_shapes_and_fences(stage,seed,monkeypatch):
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    rng=random.Random(seed);view,scheduler=make(stage,rng.randrange(1,40))
    for slot,r in list(view.requests.items()):
        view.requests[slot]=replace(r,prompt_count=rng.randrange(1,30),capacity_blocks=32+rng.randrange(1,40),
            proposal_depth=rng.randrange(1,5),ready_ns=rng.randrange(0,1000000000))
        if rng.random()<.15:patch(view,K.REQUEST_ENGINE,slot,lifecycle=2)
        if rng.random()<.2:patch(view,K.REQUEST_DRAFT if stage=='T' else K.REQUEST_TARGET_COMPUTE,slot,status=1)
    for _ in range(2):
        commands=scheduler.schedule(view,phase=stage)
        assert scheduler.last_metrics['implementation']=='cpp'
        for command in commands:
            items=getattr(command,'requests',getattr(command,'new_requests',()))
            requests=[view.requests[item.request_slot] for item in items]
            initial=command.kind.name in ('DRAFT_BATCH','TARGET_PREFILL_BATCH')
            worker=next(w for w in view.workers if w.worker_id==command.worker_id)
            reference=scheduler._estimator.stage_prediction(view,stage,worker,requests,1000000000,initial=initial)
            actual=scheduler.native_predictions[worker.worker_id,command.command_seq][1]
            assert actual==pytest.approx(reference,rel=1e-12,abs=1e-12)
    # Reusing the scheduler must see new immutable rows and changed RequestInput.
    patch(view,K.REQUEST_DISPATCH,0,draft_worker_generation=9)
    scheduler.schedule(view,phase=stage)


@pytest.mark.parametrize('stage',['D','T'])
@pytest.mark.parametrize('seed',range(20))
def test_pending_work_and_host_versions(stage,seed,monkeypatch):
    from nebulasd.workers.work import WorkKind
    from nebulasd.engine.work_ledger import Record
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    rng=random.Random(seed);view,scheduler=make(stage,16)
    view.work_state.direct_imports=True
    for slot in view.requests:
        patch(view,K.REQUEST_DRAFT_D2H,slot,request_epoch=1,status=2,ready_version=1,
            snapshot_version=1,source_op_seq=1,owner_epoch=0,source_worker_generation=1)
        target=view.row(K.REQUEST_TARGET_COMPUTE,slot)
        patch(view,K.REQUEST_D2H,slot,request_epoch=1,status=2,ready_version=target.get('target_kv_version'))
    # Live records on both roles exercise predecessor and copy-free predictions.
    for w in view.workers:
        d=w.role==WorkerRole.DRAFT
        rows=tuple(NS(slot=i,capacity_blocks=4,source=None if seed%3==0 else object()) for i in range(4))
        op=(WorkKind.DRAFT_INITIAL if seed%3==0 else WorkKind.DRAFT_DECODE) if d else WorkKind.TARGET_VERIFY
        work=NS(worker_id=w.worker_id,work_seq=7,bank_id=0,rows=rows,operation=op)
        rec=Record(work,None,compute_done=bool(seed%2),physical_done=bool(seed%5==0))
        view.work_state.records[w.worker_id,1,7]=rec
        view.work_state.by_worker[w.worker_id][7]=rec
        kind=K.WORKER_DRAFT_RUNTIME if d else K.WORKER_TARGET_COMPUTE_RUNTIME
        patch(view,kind,w.worker_id,compute_status=seed%2,compute_start_time_ns=950000000,
            **{'current_batch_seq' if d else 'compute_batch_seq':7})
    for _ in range(3):
        scheduler.schedule(view,phase=stage)
        slot=rng.randrange(16)
        patch(view,K.REQUEST_DRAFT_D2H if stage=='D' else K.REQUEST_D2H,slot,ready_version=rng.randrange(1,3))


def test_initial_and_slot_reuse(monkeypatch):
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    view,scheduler=make('T',12)
    for slot in view.requests:patch(view,K.REQUEST_DISPATCH,slot,target_run_seq=0)
    scheduler.initial_batch_limit=2
    assert scheduler.schedule(view,phase='T')
    for slot,r in list(view.requests.items()):
        view.requests[slot]=replace(r,epoch=2)
        patch(view,K.REQUEST_ENGINE,slot,request_epoch=2)
    assert scheduler.schedule(view,phase='T')
    view.requests.clear()
    assert not scheduler.schedule(view,phase='T')
    assert not scheduler._native.requests


def test_incremental_updates_match_fresh_snapshot(monkeypatch):
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    view,scheduler=make('T',16)
    scheduler.enable_native_updates()
    scheduler.schedule(view,phase='T')
    pointer=scheduler._native.pointers
    for slot in range(16):
        patch(view,K.REQUEST_ENGINE,slot,lifecycle=2)
        scheduler.observe_table(view.row(K.REQUEST_ENGINE,slot))
        scheduler.schedule(view,phase='T')
        assert scheduler._native.pointers is pointer
    # Cohort reset/reuse is keyed by live membership and new snapshots.
    del view.requests[0]
    scheduler.schedule(view,phase='T')
    assert 0 not in scheduler._native.requests


@pytest.mark.parametrize('stage',['D','T'])
def test_compact_plan_builds_identical_work(stage,monkeypatch):
    from nebulasd.engine.work_progress import WorkProgress
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    view,scheduler=make(stage,4)
    if stage=='T':
        for r in view.requests.values():
            patch(view,K.REQUEST_HOSTKV,r.slot,capacity_blocks=r.capacity_blocks,offset_blocks=0,
                host_slot_generation=1,host_slot=r.slot,writer_lease_generation=1)
    commands=scheduler.schedule(view,phase=stage)
    assert commands
    engine=NS(resources=NS(specs=view.workers,completions=NS(reserve=lambda count:0)),
        ledger=view.work_state,rows=view.rows,
        registry=NS(records={r.slot:NS(input=r,reservation=r.output) for r in view.requests.values()}))
    progress=WorkProgress.__new__(WorkProgress);progress.engine=engine
    for command in commands:
        initial=command.kind.name in ('DRAFT_BATCH','TARGET_PREFILL_BATCH')
        old=command.materialize(scheduler,view,scheduler._native.banks[command.worker_id,initial])
        assert progress.build(command).to_bytes()==progress.build(old).to_bytes()


def test_more_than_eight_destinations_preserves_beam_ties(monkeypatch):
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    view,scheduler=make('T',20)
    target=next(w for w in view.workers if w.role==WorkerRole.TARGET)
    extras=tuple(replace(target,worker_id=i,max_batch_size=2) for i in range(10,19))
    view=replace(view,workers=view.workers+extras)
    view.work_state.resources.specs=view.workers
    for w in extras:
        view.work_state.sequences[w.worker_id]=1
        view.sequences[w.worker_id]=1
        for bank in (0,1):
            view.work_state.bank_epochs[w.worker_id,bank]=0
            patch(view,K.WORKER_BANK,w.worker_id*2+bank,bank_id=bank,bank_epoch=0,state=0,capacity_blocks=128)
        patch(view,K.WORKER_COMMON,w.worker_id,worker_id=w.worker_id,worker_generation=w.generation,status=1)
    scheduler.schedule(view,phase='T',destinations={w.worker_id for w in extras})


@pytest.mark.parametrize('stage',['D','T'])
def test_native_projection_uses_immutable_observed_bytes(stage,monkeypatch):
    from nebulasd.table.native_storage import request_table,close_table_partitions
    from nebulasd.table.storage import FieldValue
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    view,scheduler=make(stage,8)
    table=request_table(8)
    try:
        for (kind,slot),row in list(view.rows.items()):
            if kind not in table._partitions:continue
            partition=table.partition(kind)
            fields=tuple(FieldValue(f.name,f.value) for f in row.fields if f.name in partition._field_by_name and f.name!='publish_seq')
            partition._publish(slot,0,fields)
            view.rows[kind,slot]=partition.read_stable(slot)
            assert view.rows[kind,slot].payload is not None
        scheduler.enable_native_updates()
        before=scheduler.schedule(view,phase=stage)
        partition=table.partition(K.REQUEST_ENGINE)
        partition._publish(0,1,(FieldValue('lifecycle',2),))
        # A later shared-memory write is invisible until Engine merges its snapshot.
        assert scheduler.schedule(view,phase=stage)==before
        row=partition.read_stable(0)
        view.rows[K.REQUEST_ENGINE,0]=row
        scheduler.observe_table(row)
        after=scheduler.schedule(view,phase=stage)
        assert all(r.request_slot!=0 for c in after for r in c.requests)
        assert partition.read_stable(0,field_names=('lifecycle',)).payload is None
    finally:
        close_table_partitions(table._partitions,unlink=True)
