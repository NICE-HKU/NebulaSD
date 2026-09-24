"""Pure policy tests: fences, stable affinity and immutable whole-batch issuance."""
from dataclasses import replace
import pytest

from nebulasd.core.enums import *
from nebulasd.core.handles import ArenaHandle, HostKVArenaHandle
from nebulasd.ipc.protocol import DraftBatchCommand, PrepareTargetBankCommand, RunTargetBatchCommand
from nebulasd.scheduler.batching import BatchWindow
from nebulasd.scheduler.policy import Scheduler
from nebulasd.scheduler.views import SchedulingView, RequestInput, WorkerSpec
from nebulasd.table.storage import BlockSnapshot, FieldValue

K = StateChangeBlockKind


def row(kind, index, **fields):
    return BlockSnapshot(kind, index, 1, tuple(FieldValue(k, v) for k, v in fields.items()))


def world(count=2):
    workers = (WorkerSpec(0, WorkerRole.DRAFT), WorkerSpec(1, WorkerRole.DRAFT), WorkerSpec(2, WorkerRole.TARGET))
    rows = {}
    def put(kind, index, **fields):
        rows[kind, index] = row(kind, index, **fields)
    for w in workers:
        put(K.WORKER_COMMON, w.worker_id, worker_id=w.worker_id, worker_generation=1, status=WorkerStatus.ONLINE)
    put(K.WORKER_TARGET_COMPUTE_RUNTIME, 2, compute_status=ComputeStatus.IDLE)
    for i in (0, 1):
        put(K.WORKER_BANK, 4+i, bank_id=i, bank_epoch=0, role=BankRole.ACTIVE if i == 0 else BankRole.STANDBY,
            state=BankState.DRAINING if i == 0 else BankState.EMPTY, alloc_rows=0, capacity_rows=8, capacity_blocks=128)
    requests = {}
    for slot in range(count):
        requests[slot] = RequestInput(slot, 1, slot, 3, 16, 2, ArenaHandle(0,12,1), ArenaHandle(12,16,1),
            ArenaHandle(32,4,1), 1, HostKVArenaHandle(slot*4,4,1), 4, 100)
        put(K.REQUEST_ENGINE, slot, request_epoch=1, lifecycle=Lifecycle.ACTIVE, current_round_id=0)
        put(K.REQUEST_DISPATCH, slot, target_round_id=0, target_run_seq=1, target_prepare_seq=0, draft_issue_seq=0)
        put(K.REQUEST_TARGET_COMPUTE, slot, request_epoch=1, target_id=2, target_generation=1, round_id=0,
            observed_run_seq=1, status=TargetStatus.READY_DRAFT, result_code=0, logical_kv_len=3,
            target_kv_version=1, committed_delta_handle=ArenaHandle(32,4,1))
        put(K.REQUEST_HOSTKV, slot, request_epoch=1, host_slot=slot, host_slot_generation=1, writer_lease_generation=1)
    return SchedulingView(requests, workers, rows, {}, {}, {0:1,1:1,2:1})


def patch(view, kind, slot, **fields):
    old = view.rows.get((kind, slot))
    merged = {} if old is None else {f.name:f.value for f in old.fields}
    merged.update(fields)
    view.rows[kind,slot] = row(kind,slot,**merged)


def runnable(count=2):
    view = world(count)
    prepare = next(c for c in Scheduler().schedule(view) if isinstance(c, PrepareTargetBankCommand))
    view.prepared[2] = prepare
    patch(view, K.WORKER_BANK, 5, state=BankState.READY, bank_epoch=prepare.next_bank_epoch, batch_seq=prepare.batch_seq)
    for item in prepare.requests:
        slot = item.request_slot
        patch(view, K.REQUEST_DISPATCH, slot, draft_worker_id=0, draft_worker_generation=1, draft_round_id=1,
              draft_issue_seq=1, target_prepare_seq=item.op_seq, planned_target_id=2, planned_target_generation=1,
              planned_bank_id=1, planned_bank_epoch=1)
        patch(view, K.REQUEST_DRAFT, slot, request_epoch=1, round_id=1, worker_id=0, worker_generation=1,
              status=DraftStatus.READY_TARGET, result_code=0, observed_issue_seq=1, proposal_handle=ArenaHandle(0,8,1))
        patch(view, K.REQUEST_H2D, slot, request_epoch=1, round_id=1, observed_prepare_seq=item.op_seq,
              target_id=2, target_generation=1, destination_bank_id=1, destination_bank_epoch=1,
              status=H2DStatus.GPU_READY, result_code=0, source_host_version=1, gpu_ready_version=1, copied_blocks=1)
        patch(view, K.REQUEST_D2H, slot, request_epoch=1, host_slot_generation=1, writer_version=1,
              ready_version=1, status=D2HStatus.HOST_READY, result_code=0)
    return view


def test_deterministic_draft_independent_of_h2d_and_one_prepare():
    view = world()
    policy = Scheduler(clock=lambda:1000)
    decisions = policy.schedule(view)
    assert decisions == policy.schedule(view)
    assert len([c for c in decisions if isinstance(c, DraftBatchCommand)]) == 1
    prepare = next(c for c in decisions if isinstance(c, PrepareTargetBankCommand))
    view.prepared[2] = prepare
    assert not any(isinstance(c, PrepareTargetBankCommand) for c in policy.schedule(view))


def test_whole_ordered_batch_only():
    view = runnable()
    run = next(c for c in Scheduler().schedule(view) if isinstance(c, RunTargetBatchCommand))
    assert tuple(r.request_slot for r in run.requests) == (0,1)
    patch(view, K.REQUEST_DRAFT, 1, status=DraftStatus.IN_DRAFT)
    assert not any(isinstance(c, RunTargetBatchCommand) for c in Scheduler().schedule(view))


@pytest.mark.parametrize('kind,field,bad', [
    (K.REQUEST_ENGINE,'lifecycle',Lifecycle.CANCELLED), (K.REQUEST_ENGINE,'request_epoch',2),
    (K.REQUEST_DRAFT,'worker_generation',2), (K.REQUEST_DRAFT,'round_id',0),
    (K.REQUEST_DRAFT,'observed_issue_seq',99), (K.REQUEST_H2D,'destination_bank_epoch',2),
    (K.REQUEST_H2D,'source_host_version',2), (K.REQUEST_H2D,'copied_blocks',0),
    (K.REQUEST_D2H,'result_code',1), (K.REQUEST_D2H,'writer_version',2),
    (K.REQUEST_HOSTKV,'host_slot_generation',2), (K.REQUEST_DISPATCH,'target_prepare_seq',99),
])
@pytest.mark.parametrize('policy', ['fixed', 'adaptive', 'batch_adaptive'])
def test_stale_fence_never_runs(kind,field,bad,policy):
    view = runnable()
    patch(view,kind,1,**{field:bad})
    assert not any(isinstance(c, RunTargetBatchCommand) for c in Scheduler(target_placement=policy).schedule(view))


@pytest.mark.parametrize('policy', ['fixed', 'adaptive', 'batch_adaptive'])
def test_affinity_does_not_move_to_idle_draft(policy):
    view = world(1)
    patch(view,K.REQUEST_DISPATCH,0,draft_worker_id=0,draft_worker_generation=1,draft_issue_seq=1,draft_round_id=0)
    view.inflight[0] = object()
    assert not any(isinstance(c,DraftBatchCommand) for c in Scheduler(target_placement=policy).schedule(view))


def test_optional_window_waits_but_default_never_waits():
    view = world(1)
    assert Scheduler(clock=lambda:101).schedule(view)
    assert not Scheduler(clock=lambda:101,window=BatchWindow(100)).schedule(view)
    assert Scheduler(clock=lambda:201,window=BatchWindow(100)).schedule(view)


def test_capacity_skips_oversized_candidate_without_blocking_smaller():
    view = world()
    view.requests[0] = replace(view.requests[0], capacity_blocks=129)
    command = next(c for c in Scheduler().schedule(view) if isinstance(c,PrepareTargetBankCommand))
    assert tuple(r.request_slot for r in command.requests) == (1,)


def test_shared_block_table_row_capacity_includes_other_bank():
    view = world()
    patch(view,K.WORKER_BANK,4,alloc_rows=7)
    command = next(c for c in Scheduler().schedule(view) if isinstance(c,PrepareTargetBankCommand))
    assert len(command.requests)==1


def test_role_specific_max_batch_sizes_bound_policy():
    view = world(3)
    view = replace(view, workers=(
        WorkerSpec(0, WorkerRole.DRAFT, max_batch_size=3),
        WorkerSpec(1, WorkerRole.DRAFT, max_batch_size=3),
        WorkerSpec(2, WorkerRole.TARGET, max_batch_size=1),
    ))
    decisions = Scheduler().schedule(view)
    draft = next(c for c in decisions if isinstance(c,DraftBatchCommand))
    prepare = next(c for c in decisions if isinstance(c,PrepareTargetBankCommand))
    assert len(draft.new_requests)+len(draft.cached_request_deltas)==3
    assert len(prepare.requests)==1


def test_target_prefill_and_verify_use_independent_token_budgets():
    view = world(3)
    for slot in range(3):
        patch(view,K.REQUEST_DISPATCH,slot,target_run_seq=0)
    rows = dict(view.rows)
    for slot in range(3):
        del rows[K.REQUEST_TARGET_COMPUTE, slot]
    requests = {slot: replace(request, prompt_count=80) for slot,request in view.requests.items()}
    view = replace(view, requests=requests, rows=rows, workers=(
        WorkerSpec(0, WorkerRole.DRAFT),
        WorkerSpec(1, WorkerRole.DRAFT),
        WorkerSpec(2, WorkerRole.TARGET, max_batch_size=3, max_batch_tokens=200,
                   prefill_max_batch_tokens=200, verify_max_batch_tokens=4),
    ))
    prefill = next(c for c in Scheduler().schedule(view) if c.kind.name == "TARGET_PREFILL_BATCH")
    assert len(prefill.requests)==2

    view = world(3)
    view = replace(view, workers=(
        WorkerSpec(0, WorkerRole.DRAFT),
        WorkerSpec(1, WorkerRole.DRAFT),
        WorkerSpec(2, WorkerRole.TARGET, max_batch_size=3, max_batch_tokens=200,
                   prefill_max_batch_tokens=200, verify_max_batch_tokens=4),
    ))
    prepare = next(c for c in Scheduler().schedule(view) if isinstance(c,PrepareTargetBankCommand))
    assert len(prepare.requests)==1

@pytest.mark.parametrize('kind,field,bad', [
    (K.REQUEST_ENGINE, 'lifecycle', Lifecycle.CANCELLED),
    (K.REQUEST_ENGINE, 'request_epoch', 2),
    (K.REQUEST_TARGET_COMPUTE, 'target_generation', 2),
    (K.REQUEST_TARGET_COMPUTE, 'observed_run_seq', 99),
    (K.REQUEST_TARGET_COMPUTE, 'result_code', 1),
    (K.WORKER_COMMON, 'worker_generation', 2),
])
def test_reused_scheduler_rechecks_target_facts_each_call(kind, field, bad):
    view = world(1)
    policy = Scheduler(clock=lambda: 1000)
    expected = policy.schedule(view)
    assert expected
    index = 2 if kind == K.WORKER_COMMON else 0
    old = view.rows[kind, index]
    patch(view, kind, index, **{field: bad})
    assert not policy.schedule(view)
    view.rows[kind, index] = old
    assert policy.schedule(view) == expected


def test_reused_scheduler_rechecks_previously_missing_target_result():
    view = world(1)
    policy = Scheduler(clock=lambda: 1000)
    row = view.rows.pop((K.REQUEST_TARGET_COMPUTE, 0))
    assert not policy.schedule(view)
    view.rows[K.REQUEST_TARGET_COMPUTE, 0] = row
    assert policy.schedule(view)
