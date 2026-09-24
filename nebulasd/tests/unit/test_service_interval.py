"""Budget protection and unchanged Engine reservation/trigger contract."""
from dataclasses import replace
from types import SimpleNamespace as NS
import random
import pytest

from nebulasd.config import NebulaSDConfig
from nebulasd.core.enums import StateChangeBlockKind as K, WorkerRole
from nebulasd.scheduler.completion import CompletionScheduler
from nebulasd.scheduler.factory import completion_scheduler
from nebulasd.scheduler.service_interval import ServiceIntervalScheduler
from nebulasd.workers.work import WorkKind
from test_native_completion import make
from test_wp08_scheduler import patch
from test_autonomous_engine import engine, publish_target
from nebulasd.data.generation_config_arena import DraftGenerationConfig


def sources(view):
    for r in view.requests.values():
        patch(view, K.REQUEST_DRAFT_D2H, r.slot, request_epoch=r.epoch, status=2,
              ready_version=1, snapshot_version=1, source_op_seq=1, owner_epoch=0,
              source_worker_generation=1)
        patch(view, K.REQUEST_D2H, r.slot, request_epoch=r.epoch, status=2,
              ready_version=view.row(K.REQUEST_TARGET_COMPUTE, r.slot).get('target_kv_version'))


def actual(scheduler, view, stage, slot, end):
    r = view.requests[slot]
    d = stage == 'D'
    dispatch = view.row(K.REQUEST_DISPATCH, slot)
    work = NS(operation=WorkKind.TARGET_VERIFY if d else WorkKind.DRAFT_DECODE,
              rows=(NS(slot=slot, epoch=r.epoch,
                       round_id=dispatch.get('target_round_id' if d else 'draft_round_id')),))
    scheduler.observe_compute_result(work, dict(compute_start_ns=end-1, compute_end_ns=end,
                                                rows=[dict(index=0)]))


class PredictionModel:
    draft_block_bytes = 4
    def __init__(self, late=None, per_member=0, free=1.):
        self.late, self.per_member, self.free = late or {}, per_member, free
    def input_ready(self, view, stage, request, now):
        return self.late.get(request.slot, 1.)
    def worker_ready(self, *args):
        return self.free
    def stage_prediction(self, view, stage, worker, batch, now, *, input_times, initial=False):
        kv = max(self.late.get(r.slot, 1.) for r in batch) + self.per_member * (len(batch)-1)
        start = max(1., self.free, kv, *input_times.values())
        return dict(start_s=start, finish_s=start+.01, compute_s=.01,
                    input_ready_s=max(input_times.values()), kv_ready_s=kv,
                    worker_ready_s=self.free, cost_s=0.)


def setup(stage='T', count=4, batch=2, model=None, gap=20, delay=0):
    view, old = make(stage, count)
    view = replace(view, workers=tuple(replace(w, max_batch_size=batch) for w in view.workers))
    sources(view)
    scheduler = ServiceIntervalScheduler(estimator=model or PredictionModel(), clock=lambda:1_000_000_000,
        draft_service_gap_ms=gap, target_service_gap_ms=gap, service_batch_delay_ms=delay)
    worker = next(w.worker_id for w in view.workers if (w.role == WorkerRole.DRAFT) == (stage == 'D'))
    return view, scheduler, worker


def selected(scheduler, view, stage, worker):
    return [r.request_slot for c in scheduler.schedule(view, phase=stage, destinations={worker}) for r in c.requests]


@pytest.mark.parametrize('stage', ['D','T'])
def test_drop_delayer_keep_waiting_anchor_no_refill(stage):
    view, s, worker = setup(stage, model=PredictionModel(late={1:1.03}))
    for slot, end in enumerate((990_000_000, 995_000_000, 996_000_000, 997_000_000)):
        actual(s, view, stage, slot, end)
    assert selected(s, view, stage, worker) == [0]
    assert s.last_metrics['frontier'] == 2  # slots 2/3 are not used to refill


@pytest.mark.parametrize('stage', ['D','T'])
def test_overdue_singleton_served_and_not_delayed_by_batch(stage):
    view, s, worker = setup(stage, model=PredictionModel(per_member=.001))
    actual(s, view, stage, 0, 900_000_000)
    assert selected(s, view, stage, worker) == [0]
    s._estimator = PredictionModel()  # peers can ride along without delaying it
    assert selected(s, view, stage, worker) == [0,1]


def test_completed_age_precedes_matching_and_future_is_not_urgent():
    view, s, worker = setup(model=PredictionModel(late={0:1.02}), batch=1)
    actual(s, view, 'T', 0, 900_000_000)
    assert selected(s, view, 'T', worker) == [0]
    # Unknown completed time is not invented from ready_ns/admitted_ns.
    s.reset_compute_times()
    assert selected(s, view, 'T', worker) == [1]


def test_equal_hidden_ready_spread_uses_age_not_absolute_ready():
    view, s, worker = setup(model=PredictionModel(late={0:1.01, 1:1.02}, free=1.03), batch=2)
    assert selected(s, view, 'T', worker) == [0,1]


def test_equal_delayers_are_removed_until_anchor_can_start():
    view, s, worker = setup(count=4, batch=3, model=PredictionModel(late={1:1.05,2:1.05}))
    for slot in range(4):actual(s,view,'T',slot,990_000_000+slot)
    assert selected(s, view, 'T', worker) == [0]


def test_configuration_rejects_paths_without_existing_source_ready_trigger(monkeypatch):
    from nebulasd.engine.bootstrap import create
    config=NebulaSDConfig(scheduler_policy='service_interval',scheduler_execution='completion',
        draft_placement='stagewise',target_placement='stagewise',cost_model='table',backend_cost_table='table.json',
        draft_service_gap_ms=20,target_service_gap_ms=20)
    with pytest.raises(ValueError,match='autonomous'):
        create(config,worker_factory=lambda:None,worker_options=None,kv_layout=None,observer=None)
    monkeypatch.setenv('STARSD_DMA_MODE','thread')
    with pytest.raises(ValueError,match='HostKV-ready triggers'):
        create(config,worker_factory=None,worker_options=None,kv_layout=None,observer=None)


@pytest.mark.parametrize('resource', ['blocks','tokens','rows'])
def test_capacity_first_fit_skips_oversized_request(resource):
    view, s, worker = setup(count=5, batch=4)
    if resource == 'blocks':
        view.requests[0] = replace(view.requests[0], capacity_blocks=1000)
    elif resource == 'tokens':
        view.requests[0] = replace(view.requests[0], proposal_depth=1000)
    else:
        for bank in (0,1):
            patch(view,K.WORKER_BANK,worker*2+bank,capacity_rows=1)
        view = replace(view, workers=tuple(replace(w,bank_rows=1) if w.worker_id==worker else w for w in view.workers))
    ids = selected(s, view, 'T', worker)
    assert ids == ([0] if resource == 'rows' else [1,2,3,4])


def test_source_unavailable_and_stale_identity_do_not_enter_top_k():
    view, s, worker = setup(count=4)
    patch(view,K.REQUEST_D2H,0,status=1)
    patch(view,K.REQUEST_DISPATCH,1,draft_worker_generation=99)
    assert selected(s, view, 'T', worker) == [2,3]


@pytest.mark.parametrize('stage', ['D','T'])
@pytest.mark.parametrize('seed', range(20))
@pytest.mark.parametrize('ignore_kv', [False, True])
def test_native_matches_reference_with_budgets_shapes_and_fences(stage, seed, ignore_kv, monkeypatch):
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    rng=random.Random(seed)
    view, old = make(stage, 12)
    sources(view)
    view=replace(view,workers=tuple(replace(w,max_batch_size=4) for w in view.workers))
    s=ServiceIntervalScheduler(estimator=replace(old._estimator,ignore_kv_time=ignore_kv),clock=lambda:1_000_000_000,
        draft_service_gap_ms=rng.choice((0,1,20,1000)),target_service_gap_ms=rng.choice((0,1,20,1000)),service_batch_delay_ms=rng.choice((0,5,30,100)))
    s.enable_native_updates()
    for slot,r in list(view.requests.items()):
        view.requests[slot]=replace(r,prompt_count=rng.randrange(1,80),proposal_depth=rng.randrange(1,5),
                                    capacity_blocks=rng.randrange(32,65))
        if rng.random()<.7: actual(s,view,stage,slot,rng.randrange(800_000_000,1_000_000_000))
        if rng.random()<.1: patch(view,K.REQUEST_ENGINE,slot,lifecycle=2)
        if rng.random()<.1: patch(view,K.REQUEST_D2H if stage=='T' else K.REQUEST_DRAFT_D2H,slot,status=1)
    for _ in range(2):
        commands=s.schedule(view,phase=stage)
        assert s.last_metrics['implementation']=='cpp'
        slots=[r.request_slot for c in commands for r in c.requests]
        assert len(slots)==len(set(slots))
        for c in commands:
            initial=c.kind.name in ('DRAFT_BATCH','TARGET_PREFILL_BATCH')
            batch=[view.requests[r.request_slot] for r in c.requests]
            worker=next(w for w in view.workers if w.worker_id==c.worker_id)
            expected=s._estimator.stage_prediction(view,stage,worker,batch,1_000_000_000,initial=initial)
            assert s.native_predictions[c.worker_id,c.command_seq][1]==pytest.approx(expected)


def test_prefill_matches_existing_policy(monkeypatch):
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    view, old=make('T',12)
    for slot in view.requests:patch(view,K.REQUEST_DISPATCH,slot,target_run_seq=0)
    s=ServiceIntervalScheduler(estimator=old._estimator,clock=old._clock,
                               draft_service_gap_ms=0,target_service_gap_ms=0)
    assert s.schedule(view,phase='T') == old.schedule(view,phase='T')


def test_prefill_admission_keeps_old_mixed_frontier(monkeypatch):
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    view,old=make('T',80);sources(view)
    for slot in range(60,80):patch(view,K.REQUEST_DISPATCH,slot,target_run_seq=0)
    s=ServiceIntervalScheduler(estimator=old._estimator,clock=old._clock,
                               draft_service_gap_ms=20,target_service_gap_ms=20)
    prefill=lambda cmds:tuple(c for c in cmds if c.kind.name=='TARGET_PREFILL_BATCH')
    assert prefill(s.schedule(view,phase='T'))==prefill(old.schedule(view,phase='T'))


def test_configuration_keeps_old_default_and_requires_explicit_budgets():
    config=NebulaSDConfig()
    assert config.scheduler_policy=='existing'
    assert type(completion_scheduler(config, PredictionModel())) is CompletionScheduler
    with pytest.raises(ValueError,match='requires completion'):
        replace(config,scheduler_policy='service_interval')
    config=replace(config,scheduler_policy='service_interval',scheduler_execution='completion',
        draft_placement='stagewise',target_placement='stagewise',cost_model='table',backend_cost_table='table.json',
        draft_service_gap_ms=0,target_service_gap_ms=20)
    assert isinstance(completion_scheduler(config, PredictionModel()),ServiceIntervalScheduler)
    with pytest.raises(ValueError,match='finite'):
        replace(config,target_service_gap_ms=float('nan'))


def install(engine):
    old=engine.scheduler
    s=ServiceIntervalScheduler(estimator=old._estimator,draft_service_gap_ms=1000,target_service_gap_ms=1000)
    s.enable_native_updates()
    engine.scheduler=s
    return s


def test_pending_and_issued_reservations_keep_exact_work(engine):
    e=engine;install(e)
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4))
    e.supervisor.pairs[1].blocked=True
    e.step()
    frozen=e.scheduling_progress.pending['T'][0][0]
    for _ in range(3):
        e.scheduling_progress.changed();e.step()
        assert e.scheduling_progress.pending['T'][0][0] is frozen
        assert not e.ledger.records
    e.supervisor.pairs[1].blocked=False;e.step()
    assert e.supervisor.pairs[1].work == [frozen]
    for _ in range(3):
        e.scheduling_progress.changed();e.step()
    assert e.supervisor.pairs[1].work == [frozen]
    publish_target(e,frozen);e.step()
    assert e.scheduler.compute_times.end_ns(0,1,'T',0)==20
    assert e.supervisor.pairs[0].work[0].rows[0].round_id==1

@pytest.mark.parametrize('mode', ['python','check'])
def test_same_round_never_moves_to_peer_after_dispatch(mode, monkeypatch):
    """Exercise the real WORK builder, ledger and dispatch gates on 2D+2T."""
    from nebulasd.engine.core import Engine
    from nebulasd.engine.resources import ControlResources
    from nebulasd.scheduler.views import WorkerSpec
    from nebulasd.table.prepared import PreparedRow
    from nebulasd.workers.target.publication import TargetPublisher
    from nebulasd.workers.draft.publication import DraftPublisher
    from test_autonomous_engine import Endpoint
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL',mode)
    specs=tuple(WorkerSpec(i,WorkerRole.DRAFT if i<2 else WorkerRole.TARGET,
                          max_batch_size=2,draft_banked=i<2) for i in range(4))
    resources=ControlResources(specs,slots=8,host_blocks=128)
    sup=NS(autonomous=True,pairs={i:Endpoint() for i in range(4)},check=lambda:None,
           pump=lambda *a:False,close=lambda:None,bell=NS(drain=lambda:None),management={},command_bells={})
    _, old=make('T',4)
    scheduler=ServiceIntervalScheduler(estimator=old._estimator,draft_service_gap_ms=1000,target_service_gap_ms=1000)
    e=Engine(resources,sup,scheduler=scheduler)
    try:
        for w in specs:
            PreparedRow(resources.registry.partition(K.WORKER_COMMON),w.worker_id,
                dict(worker_id=w.worker_id,worker_generation=1,status=1,role=w.role),()).publish(())
            kind=K.WORKER_DRAFT_BANK if w.worker_id<2 else K.WORKER_BANK
            for b in (0,1):
                PreparedRow(resources.registry.partition(kind),w.worker_id*2+b,
                    dict(bank_id=b,bank_epoch=0,state=0,role=2,capacity_rows=8,capacity_blocks=128,alloc_rows=0),()).publish(())
        for slot in range(4):e.admit(str(slot),(1,2,3),DraftGenerationConfig(8,4))
        # One endpoint accepts while its peer is backpressured. Even global
        # management triggers must retain the pending WORK's exact membership.
        sup.pairs[3].blocked=True;e.step()
        pending=e.scheduling_progress.pending['T']
        assert pending and sup.pairs[2].work
        frozen=pending[0][0]
        for _ in range(3):e.scheduling_progress.changed();e.step()
        assert e.scheduling_progress.pending['T'][0][0] is frozen
        sup.pairs[3].blocked=False;e.step()
        prefills=[w for i in (2,3) for w in sup.pairs[i].work]
        assert sorted(r.slot for w in prefills for r in w.rows)==list(range(4))
        assert len({w.worker_id for w in prefills})==2
        for w in prefills:
            pub=TargetPublisher(resources.table,resources.completions,block_bytes=16,outputs=resources.token_router,configs=resources.configs)
            pub.reserve(w)
            result=('RESULT',w.work_seq,dict(rows=[dict(index=i,tokens=(4,),accepted=0,logical=3,
                version=1,dirty_begin=0,dirty_blocks=1) for i in range(len(w.rows))],compute_start_ns=10,compute_end_ns=20))
            physical=('PHYSICAL',w.work_seq,dict(outcomes=[1]*len(w.rows),observed_ns=30,d2h_submitted_ns=21))
            pub.consume(result);pub.consume(physical)
            while pub.records:pub.step(8)
        e.step()
        drafts=[w for i in (0,1) for w in sup.pairs[i].work]
        assert sorted(r.slot for w in drafts for r in w.rows)==list(range(4))
        before=[len(sup.pairs[i].work) for i in range(4)]
        for _ in range(3):e.scheduling_progress.changed();e.step()
        assert [len(sup.pairs[i].work) for i in range(4)]==before
        # Duplicate stage/round authorization cannot be redirected to a peer.
        with pytest.raises(ValueError,match='duplicate request authorization'):
            w=drafts[0]
            e.ledger.sent(replace(w,worker_id=1-w.worker_id,work_seq=999),None)
        for w in drafts:
            resources.proposals[w.worker_id]._writer=True;resources.snapshots[w.worker_id]._writer=True
            pub=DraftPublisher(resources.table,resources.proposals[w.worker_id],resources.snapshots[w.worker_id],
                               resources.completions,host=resources.draft_host.descriptor)
            pub.reserve(w)
            result=('DRAFT_RESULT',w.work_seq,dict(rows=[dict(index=i,proposal=(5,6,7,8),proposal_kind=1,
                logical=7,version=1,dirty_begin=0,dirty_blocks=1,committed_count=1) for i in range(len(w.rows))],
                compute_start_ns=40,compute_end_ns=50))
            physical=('PHYSICAL',w.work_seq,dict(outcomes=[1]*len(w.rows),observed_ns=60,d2h_submitted_ns=51))
            pub.consume(result);pub.consume(physical)
            while pub.records:pub.step(8)
        e.step()
        verifies=[w for i in (2,3) for w in sup.pairs[i].work if w.operation==WorkKind.TARGET_VERIFY]
        assert sorted(r.slot for w in verifies for r in w.rows)==list(range(4))
        for _ in range(3):e.scheduling_progress.changed();e.step()
        all_rows=[(w.operation.name.startswith('DRAFT'),r.slot,r.epoch,r.round_id)
                  for endpoint in sup.pairs.values() for w in endpoint.work for r in w.rows]
        assert len(all_rows)==len(set(all_rows))
        if mode=='check':assert scheduler.last_metrics['implementation']=='cpp'
    finally:
        e.close()


@pytest.mark.parametrize('stage', ['D','T'])
@pytest.mark.parametrize('seed', range(6))
def test_native_inflight_predecessor_and_dma_match(stage, seed, monkeypatch):
    from nebulasd.engine.work_ledger import Record
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    view,old=make(stage,8);sources(view)
    s=ServiceIntervalScheduler(estimator=old._estimator,clock=old._clock,
                              draft_service_gap_ms=seed,target_service_gap_ms=seed)
    view.work_state.direct_imports=True
    for w in view.workers:
        d=w.role==WorkerRole.DRAFT
        rows=tuple(NS(slot=i,capacity_blocks=4,source=object()) for i in range(2))
        work=NS(worker_id=w.worker_id,work_seq=7,bank_id=0,rows=rows,
                operation=WorkKind.DRAFT_DECODE if d else WorkKind.TARGET_VERIFY)
        record=Record(work,None,compute_done=bool(seed%2),physical_done=False)
        view.work_state.records[w.worker_id,1,7]=record
        view.work_state.by_worker[w.worker_id][7]=record
        patch(view,K.WORKER_DRAFT_RUNTIME if d else K.WORKER_TARGET_COMPUTE_RUNTIME,w.worker_id,
              compute_status=seed%2,compute_start_time_ns=999_000_000,
              **{'current_batch_seq' if d else 'compute_batch_seq':7})
    if stage=='T':
        for r in view.requests.values():patch(view,K.REQUEST_DRAFT,r.slot,status=1)
    else:
        for r in view.requests.values():patch(view,K.REQUEST_TARGET_COMPUTE,r.slot,status=1)
    s.schedule(view,phase=stage)
    assert s.last_metrics['implementation']=='cpp'


@pytest.mark.parametrize('stage', ['D', 'T'])
def test_thirty_ms_budget_is_total_batch_delay_not_per_addition(stage):
    view, scheduler, worker = setup(stage, count=8, batch=8, gap=0,
        delay=30, model=PredictionModel(per_member=.01))
    for slot in range(8):actual(scheduler,view,stage,slot,900_000_000+slot)
    assert selected(scheduler,view,stage,worker)==[0,1,2,3]


def test_single_pass_skips_delayer_then_accepts_later_top_k_member():
    view,s,worker=setup(count=4,batch=4,gap=0,delay=30,
                       model=PredictionModel(late={1:1.031,2:1.03}))
    for slot in range(4):actual(s,view,'T',slot,900_000_000+slot)
    assert selected(s,view,'T',worker)==[0,2,3]


def test_filter_checks_at_most_one_addition_per_candidate():
    from nebulasd.scheduler.service_interval import filter_batch
    batch=[NS(slot=i) for i in range(128)];calls=[]
    def predict(rows):
        calls.append(tuple(r.slot for r in rows))
        return dict(start_s=1.04)
    assert filter_batch(batch,predict,{i:1. for i in range(128)},.03)==batch[:1]
    assert len(calls)==127
    assert all(len(c)==2 and c[0]==0 for c in calls)


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, None])
def test_invalid_batch_delay_rejected(value):
    with pytest.raises(ValueError,match='finite'):
        NebulaSDConfig(service_batch_delay_ms=value)
    with pytest.raises(ValueError,match='finite'):
        ServiceIntervalScheduler(estimator=PredictionModel(),draft_service_gap_ms=20,
                                 target_service_gap_ms=20,service_batch_delay_ms=value)


def test_batch_delay_default_and_factory_forwarding():
    c=NebulaSDConfig(scheduler_policy='service_interval',scheduler_execution='completion',
        draft_placement='stagewise',target_placement='stagewise',cost_model='table',
        backend_cost_table='table.json',draft_service_gap_ms=160,target_service_gap_ms=130)
    assert completion_scheduler(c,PredictionModel()).service_batch_delay_s==.03
    assert completion_scheduler(replace(c,service_batch_delay_ms=7),PredictionModel()).service_batch_delay_s==.007


@pytest.mark.parametrize('stage', ['D','T'])
def test_ignore_kv_time_skips_copy_queries_but_keeps_compute_and_dependency(stage,monkeypatch):
    from nebulasd.scheduler.measured_placement import MeasuredPlacementEstimator
    view,old=make(stage,4);sources(view)
    estimator=replace(old._estimator,ignore_kv_time=True)
    def forbidden(*args,**kwargs):raise AssertionError('KV prediction must not be consulted')
    monkeypatch.setattr(MeasuredPlacementEstimator,'_copy',forbidden)
    monkeypatch.setattr(MeasuredPlacementEstimator,'_copy_free',forbidden)
    worker=next(w for w in view.workers if (w.role==WorkerRole.DRAFT)==(stage=='D'))
    requests=list(view.requests.values())[:2]
    p=estimator.for_invocation().stage_prediction(view,stage,worker,requests,1_000_000_000,
                      input_times={r.slot:1.02 for r in requests})
    assert p['kv_ready_s']==1.
    assert p['start_s']==1.02
    assert p['compute_s']>0
    assert p['finish_s']==p['start_s']+p['compute_s']


@pytest.mark.parametrize('stage', ['D','T'])
def test_ignore_kv_native_needs_no_copy_cost_family_and_preserves_source_gate(stage,monkeypatch):
    from nebulasd.scheduler.cost_table import CostTable
    from test_cost_table import payload
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL','check')
    view,old=make(stage,4);sources(view)
    data=payload();data['rows']=[r for r in data['rows'] if r['stage'] not in ('H2D','D2H')]
    estimator=replace(old._estimator,table=CostTable(data,expected=data['compatibility']),ignore_kv_time=True)
    scheduler=ServiceIntervalScheduler(estimator=estimator,clock=old._clock,
        draft_service_gap_ms=160,target_service_gap_ms=130,service_batch_delay_ms=30)
    patch(view,K.REQUEST_D2H if stage=='T' else K.REQUEST_DRAFT_D2H,0,status=1)
    commands=scheduler.schedule(view,phase=stage)
    assert commands and scheduler.last_metrics['implementation']=='cpp'
    assert 0 not in [r.request_slot for c in commands for r in c.requests]
    for prediction in scheduler.native_predictions.values():assert prediction[1]['kv_ready_s']==old._clock()/1e9


def test_ignore_kv_configuration_is_explicit_and_preserves_thirty_ms_allowance():
    assert NebulaSDConfig().scheduler_ignore_kv_time is False
    with pytest.raises(ValueError,match='requires service_interval'):
        NebulaSDConfig(scheduler_ignore_kv_time=True)
    with pytest.raises(ValueError,match='boolean'):
        NebulaSDConfig(scheduler_ignore_kv_time=1)
    _,old=make('D',4)
    c=NebulaSDConfig(scheduler_policy='service_interval',scheduler_execution='completion',
        draft_placement='stagewise',target_placement='stagewise',cost_model='table',backend_cost_table='table.json',
        draft_service_gap_ms=160,target_service_gap_ms=130,scheduler_ignore_kv_time=True)
    s=completion_scheduler(c,old._estimator)
    assert s._estimator.ignore_kv_time is True and s.service_batch_delay_s==.03
    assert old._estimator.ignore_kv_time is False
