"""Batch locality must remain a preference inside existing safety fences."""
from dataclasses import replace
import pytest
from nebulasd.core.enums import *
from nebulasd.ipc.protocol import PrepareTargetBankCommand, RunTargetBatchCommand
from nebulasd.scheduler.policy import Scheduler
from nebulasd.scheduler.grouped_placement import related_batches
from nebulasd.scheduler.views import WorkerSpec
from test_wp08_scheduler import world, patch, runnable

K = StateChangeBlockKind


def multi(count=8):
    view = world(count)
    workers = (*view.workers[:2], *(WorkerSpec(i, WorkerRole.TARGET, max_batch_size=8) for i in (2,3,4)))
    view = replace(view, workers=workers, sequences={i:1 for i in range(5)})
    for w in workers[2:]:
        patch(view,K.WORKER_COMMON,w.worker_id,worker_id=w.worker_id,worker_generation=1,status=WorkerStatus.ONLINE)
        patch(view,K.WORKER_TARGET_COMPUTE_RUNTIME,w.worker_id,compute_status=ComputeStatus.IDLE)
        for b in (0,1):
            patch(view,K.WORKER_BANK,w.worker_id*2+b,bank_id=b,bank_epoch=1,
                  role=BankRole.ACTIVE if b==0 else BankRole.STANDBY,state=BankState.EMPTY,
                  alloc_rows=0,capacity_rows=16,capacity_blocks=128)
    for r in view.requests.values():
        patch(view,K.REQUEST_TARGET_COMPUTE,r.slot,bank_id=0,bank_epoch=1)
    return view


def prepares(view, policy='batch_adaptive', estimator=None):
    return [c for c in Scheduler(target_placement=policy, estimator=estimator).schedule(view)
            if isinstance(c,PrepareTargetBankCommand)]


class Scatter:
    def draft_worker_score(self,*args): return 0
    def score(self,view,worker,request,now): return 0 if worker.worker_id == 2+request.slot%3 else 1


def test_whole_batch_even_when_individual_scores_scatter():
    view=multi()
    old=prepares(view,'adaptive',Scatter())
    new=prepares(view,estimator=Scatter())
    assert sorted(len(c.requests) for c in old)==[2,3,3]
    assert len(new)==1 and len(new[0].requests)==8
    assert Scheduler()._target_placement=='adaptive'


@pytest.mark.parametrize('limit', ['requests','tokens','blocks','rows'])
def test_split_respects_all_cumulative_limits(limit):
    view=multi(6)
    if limit in ('requests','tokens'):
        view=replace(view,workers=tuple(replace(w,**({'max_batch_size':2} if limit=='requests'
                          else {'verify_max_batch_tokens':6})) if w.role==WorkerRole.TARGET else w for w in view.workers))
    for w in view.workers[2:]:
        if limit=='blocks': patch(view,K.WORKER_BANK,w.worker_id*2+1,capacity_blocks=8)
        if limit=='rows':
            patch(view,K.WORKER_BANK,w.worker_id*2+1,capacity_rows=8)
            patch(view,K.WORKER_BANK,w.worker_id*2,alloc_rows=6)
    cmds=prepares(view)
    assert sorted(len(c.requests) for c in cmds)==[2,2,2]
    assert sorted(r.request_slot for c in cmds for r in c.requests)==list(range(6))
    for c in cmds:
        assert sum(r.destination_capacity_blocks for r in c.requests)==8
        assert [r.destination_bank_offset_blocks for r in c.requests]==[0,4]


def test_live_members_only_and_no_cached_cohort_on_recycle():
    view=multi(4)
    patch(view,K.REQUEST_ENGINE,1,lifecycle=Lifecycle.FINISHED)
    patch(view,K.REQUEST_TARGET_COMPUTE,2,status=TargetStatus.IN_TARGET)
    patch(view,K.REQUEST_TARGET_COMPUTE,3,request_epoch=0)
    assert [r.request_slot for c in prepares(view) for r in c.requests]==[0]
    patch(view,K.REQUEST_TARGET_COMPUTE,2,status=TargetStatus.READY_DRAFT)
    assert [r.request_slot for c in prepares(view) for r in c.requests]==[0,2]
    view.requests[0]=replace(view.requests[0],epoch=2)
    assert [r.request_slot for c in prepares(view) for r in c.requests]==[2]
    patch(view,K.REQUEST_ENGINE,0,request_epoch=2)
    patch(view,K.REQUEST_TARGET_COMPUTE,0,request_epoch=2,bank_epoch=3)
    groups=related_batches(view,view.requests.values())
    assert [[r.slot for r in g] for g in groups]==[[0],[2]]


def test_bank_and_worker_generation_separate_equal_rounds():
    view=multi(4)
    patch(view,K.REQUEST_TARGET_COMPUTE,1,bank_epoch=2)
    patch(view,K.REQUEST_TARGET_COMPUTE,2,target_id=3)
    patch(view,K.REQUEST_TARGET_COMPUTE,3,target_generation=2)
    assert [[r.slot for r in g] for g in related_batches(view,view.requests.values())]==[[0],[1],[2]]


def test_busy_whole_fit_does_not_hide_idle_partial_resources():
    view=multi(6)
    view.inflight[2]=object()
    patch(view,K.WORKER_TARGET_COMPUTE_RUNTIME,2,compute_status=ComputeStatus.RUNNING)
    patch(view,K.WORKER_BANK,7,capacity_rows=2)
    patch(view,K.WORKER_COMMON,4,status=WorkerStatus.FAILED)
    cmds=prepares(view)
    assert [(c.worker_id,len(c.requests)) for c in cmds]==[(2,4),(3,2)]
    view.prepared.update({c.worker_id:c for c in cmds})
    assert prepares(view)==[]


def test_unavailable_recovers_and_oversize_does_not_starve_small():
    view=multi(2)
    for w in view.workers[2:]: patch(view,K.WORKER_BANK,w.worker_id*2+1,state=BankState.PREPARING)
    assert prepares(view)==[]
    patch(view,K.WORKER_BANK,7,state=BankState.EMPTY)
    view.requests[0]=replace(view.requests[0],capacity_blocks=129)
    assert [r.request_slot for c in prepares(view) for r in c.requests]==[1]
    view.requests[0]=replace(view.requests[0],capacity_blocks=4)
    assert [r.request_slot for c in prepares(view) for r in c.requests]==[0,1]


@pytest.mark.parametrize('policy',['fixed','adaptive','batch_adaptive'])
def test_frozen_order_and_late_draft_member(policy):
    view=runnable()
    scheduler=Scheduler(target_placement=policy)
    cmds=scheduler.schedule(view)
    run=next(c for c in cmds if isinstance(c,RunTargetBatchCommand))
    assert [r.request_slot for r in run.requests]==[r.request_slot for r in view.prepared[2].requests]
    patch(view,K.REQUEST_DRAFT,1,status=DraftStatus.IN_DRAFT)
    assert not any(isinstance(c,(RunTargetBatchCommand,PrepareTargetBankCommand)) for c in scheduler.schedule(view))
    patch(view,K.REQUEST_DRAFT,1,status=DraftStatus.READY_TARGET)
    assert any(isinstance(c,RunTargetBatchCommand) for c in scheduler.schedule(view))


def test_whole_group_migrates_to_idle_target_without_affinity_change():
    view=multi()
    view.inflight[2]=object()
    patch(view,K.WORKER_TARGET_COMPUTE_RUNTIME,2,compute_status=ComputeStatus.RUNNING)
    cmds=prepares(view)
    assert len(cmds)==1 and cmds[0].worker_id in (3,4) and len(cmds[0].requests)==8


def test_profile_diagnostics_do_not_change_commands(tmp_path):
    from nebulasd.observability.profiling import ProfileRecorder
    from nebulasd.observability.scheduler_batches import attach_batch_diagnostics
    view=multi()
    scheduler=Scheduler(clock=lambda:1000,target_placement='batch_adaptive')
    expected=scheduler.schedule(view)
    recorder=ProfileRecorder(tmp_path,'engine',100)
    attach_batch_diagnostics(scheduler,recorder)
    assert scheduler.schedule(view)==expected
    event=recorder.events[-1]
    assert len(event['groups'])==1 and len(event['groups'][0])==8
    assert len(event['prepares'])==1
