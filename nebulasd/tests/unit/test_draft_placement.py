"""Pure Step4 joins, capacity exclusion and immutable ordered preparation."""
from dataclasses import replace
import pytest
from nebulasd.core.enums import StateChangeBlockKind as K, BankState, BankRole, ComputeStatus, DraftStatus, H2DStatus, D2HStatus, Lifecycle
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.draft_contracts import DraftHostAllocation
from nebulasd.ipc.draft_protocol import RunDraftBatchCommand
from nebulasd.scheduler.draft_placement import schedule
from nebulasd.table.draft_fences import allocation_values, prepare_values, h2d_values
from unit.test_wp08_scheduler import world, patch


def migration_view(count=1):
    view = world(count)
    view = replace(view, workers=tuple(replace(w, draft_banked=w.worker_id != 2) for w in view.workers))
    for worker in (0, 1):
        patch(view, K.WORKER_DRAFT_RUNTIME, worker, compute_status=ComputeStatus.IDLE)
        for bank in (0, 1):
            patch(view, K.WORKER_DRAFT_BANK, worker*2+bank, bank_id=bank, bank_epoch=0,
                role=BankRole.ACTIVE if bank == 0 else BankRole.STANDBY, state=BankState.EMPTY,
                alloc_rows=0, capacity_rows=8, capacity_blocks=128)
    for slot in view.requests:
        allocation = DraftHostAllocation(1, 1, 1, 1, 1, slot*4, slot, 4, 16)
        patch(view, K.REQUEST_DRAFT_HOSTKV, slot, request_epoch=1, **allocation_values(allocation))
        patch(view, K.REQUEST_ENGINE, slot, current_round_id=1)
        patch(view, K.REQUEST_TARGET_COMPUTE, slot, round_id=1)
        patch(view, K.REQUEST_DISPATCH, slot, target_round_id=1, draft_round_id=1, draft_issue_seq=1,
            draft_worker_id=0, draft_worker_generation=1, draft_owner_epoch=0)
        patch(view, K.REQUEST_DRAFT, slot, request_epoch=1, round_id=1, observed_issue_seq=1,
            worker_id=0, worker_generation=1, owner_epoch=0, snapshot_version=1, logical_kv_len=5,
            valid_blocks=1, draft_state_handle=ArenaHandle(slot*200, 200, 1),
            status=DraftStatus.READY_TARGET, result_code=0)
    return view


def decide(view):
    return schedule(view, list(view.requests.values()), view.workers, dict(view.sequences), placement='worker_id')


def prepared_view():
    view = migration_view(2)
    p = decide(view)[0]
    view.prepared[p.worker_id] = p
    patch(view, K.WORKER_DRAFT_BANK, p.worker_id*2+p.standby_bank_id,
          state=BankState.READY, bank_epoch=p.next_bank_epoch, batch_seq=p.batch_seq)
    for item in p.requests:
        i = item.source
        patch(view, K.REQUEST_DISPATCH, item.request_slot, **prepare_values(p, item))
        patch(view, K.REQUEST_DRAFT_H2D, item.request_slot, **h2d_values(p, item), status=H2DStatus.GPU_READY,
            result_code=0, gpu_ready_version=1, copied_blocks=1)
        patch(view, K.REQUEST_DRAFT_D2H, item.request_slot, request_epoch=1, status=D2HStatus.HOST_READY,
            ready_version=1, snapshot_version=1, snapshot_round_id=1, snapshot_handle=item.snapshot_handle,
            source_worker_id=0, source_worker_generation=1, source_op_seq=1, owner_epoch=0,
            logical_kv_len=5, valid_blocks=1, result_code=0, **allocation_values(i.allocation))
    return view


def test_capacity_excludes_destination_without_changing_source_owner():
    view = migration_view()
    patch(view, K.WORKER_DRAFT_BANK, 1, capacity_blocks=3)
    before = dict(view.rows)
    command = decide(view)[0]
    assert command.worker_id == 1 and command.requests[0].source.worker_id == 0
    assert view.rows == before and not view.prepared
    patch(view, K.WORKER_DRAFT_BANK, 3, capacity_blocks=3)
    assert decide(view) == []


def test_preparing_while_other_bank_computes_obeys_shared_row_capacity():
    view = migration_view()
    patch(view, K.WORKER_DRAFT_RUNTIME, 0, compute_status=ComputeStatus.RUNNING)
    view.inflight[0] = object()
    assert decide(view)[0].worker_id == 0  # compute does not block independent prepare
    patch(view, K.WORKER_DRAFT_BANK, 0, alloc_rows=8)
    assert decide(view)[0].worker_id == 1


def test_early_prepare_waits_for_verify_issue_but_not_target_result():
    view = migration_view()
    # Target's frozen batch may still be waiting for a different request's
    # Draft proposal. Do not occupy the only Draft prepare slot for round 2.
    patch(view, K.REQUEST_DISPATCH, 0, target_round_id=0)
    patch(view, K.REQUEST_TARGET_COMPUTE, 0, round_id=0)
    assert decide(view) == []
    patch(view, K.REQUEST_DISPATCH, 0, target_round_id=1)
    assert decide(view)  # verify issued, result deliberately still round 0


@pytest.mark.parametrize('kind,field,value', [
    (K.REQUEST_ENGINE, 'lifecycle', Lifecycle.CANCELLED),
    (K.REQUEST_DRAFT, 'owner_epoch', 99), (K.REQUEST_DRAFT, 'snapshot_version', 2),
    (K.REQUEST_DRAFT_HOSTKV, 'host_slot_generation', 2),
    (K.REQUEST_DISPATCH, 'draft_prepare_seq', 99),
    (K.REQUEST_DRAFT_H2D, 'destination_bank_epoch', 99),
    (K.REQUEST_DRAFT_H2D, 'destination_worker_generation', 99),
    (K.REQUEST_DRAFT_D2H, 'source_worker_generation', 99),
    (K.REQUEST_DRAFT_D2H, 'ready_version', 2),
    (K.REQUEST_TARGET_COMPUTE, 'round_id', 0)])
def test_frozen_run_joins_every_member_identity(kind, field, value):
    view = prepared_view()
    command = next(c for c in decide(view) if isinstance(c, RunDraftBatchCommand))
    assert tuple(r.request_slot for r in command.requests) == (0, 1)
    patch(view, kind, 1, **{field: value})
    assert not any(isinstance(c, RunDraftBatchCommand) for c in decide(view))


@pytest.mark.parametrize('count', [1, 2])
def test_initial_short_budget_clamps_command_and_batch_token_fit(count):
    from nebulasd.ipc.protocol import DraftBatchCommand
    view = migration_view(count)
    # 14 prompt + 2 maximum output fits exactly one 16-token block.
    # Target has already produced the anchor: only one proposal token remains.
    for slot, r in view.requests.items():
        view.requests[slot] = replace(r, prompt_count=14, max_new_tokens=2, proposal_depth=8,
            prompt=ArenaHandle(0, 56, 1), capacity_blocks=1)
        patch(view, K.REQUEST_DISPATCH, slot, draft_issue_seq=0)
    view = replace(view, workers=tuple(replace(w, max_batch_tokens=16*count) for w in view.workers))
    commands = [c for c in decide(view) if isinstance(c, DraftBatchCommand)]
    assert len(commands) == 1
    command = commands[0]
    assert tuple(r.request_slot for r in command.new_requests) == tuple(range(count))
    assert tuple(r.scheduled_token_count for r in command.new_requests) == (1,)*count
    assert command.bank.capacity_blocks == (1,)*count
    command.bank.validate_requests(command.new_requests, (), ())
    # The wire guard must still reject a sender that forgets the truncation.
    with pytest.raises(ValueError, match='proposal growth'):
        command.bank.validate_requests(tuple(replace(r, scheduled_token_count=8) for r in command.new_requests), (), ())
