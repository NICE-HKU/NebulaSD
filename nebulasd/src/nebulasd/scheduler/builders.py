"""Translate already-selected batches into existing immutable wire commands."""

from nebulasd.core.enums import BankState, StateChangeBlockKind as K
from nebulasd.core.handles import ArenaHandle, HostKVArenaHandle
from nebulasd.ipc.protocol import (DraftBatchCommand, NewRequestData, CachedRequestDelta,
    TargetPrefillBatchCommand, TargetPrefillRequest, PrepareTargetBankCommand,
    TargetPrepareRequest, RunTargetBatchCommand, RunTargetRequest)
from .views import value as v


def draft(view, worker, sequence, requests):
    new, cached = [], []
    for r in requests:
        target, dispatch = view.row(K.REQUEST_TARGET_COMPUTE, r.slot), view.row(K.REQUEST_DISPATCH, r.slot)
        round_id = v(target, "round_id") + 1
        op = round_id
        if v(dispatch, "draft_issue_seq", 0) == 0:
            new.append(NewRequestData(r.slot, r.epoch, round_id, op, r.proposal_depth,
                r.prompt, r.output, r.config, ArenaHandle.null()))
        else:
            cached.append(CachedRequestDelta(r.slot, r.epoch, round_id, op,
                v(target, "committed_delta_handle"), ArenaHandle.null(),
                HostKVArenaHandle(0, 0, 0), ArenaHandle.null(), r.proposal_depth))
    return DraftBatchCommand(worker.worker_id, worker.generation, sequence, tuple(new), tuple(cached))


def prefill(worker, sequence, requests, bank):
    offset, items = 0, []
    epoch = v(bank, "bank_epoch") + (1 if v(bank, "state") == BankState.DRAINING else 0)
    for r in requests:
        items.append(TargetPrefillRequest(r.slot, r.epoch, 0, 1, r.prompt, r.config,
            1, r.max_new_tokens, v(bank, "bank_id"), epoch, offset, r.capacity_blocks))
        offset += r.capacity_blocks
    return TargetPrefillBatchCommand(worker.worker_id, worker.generation, sequence, sequence + 1,
                                     v(bank, "bank_id"), epoch, tuple(items))


def prepare(view, worker, sequence, requests, bank):
    items, offset = [], 0
    bank_id, epoch = v(bank, "bank_id"), v(bank, "bank_epoch") + 1
    for r in requests:
        target, host = view.row(K.REQUEST_TARGET_COMPUTE, r.slot), view.row(K.REQUEST_HOSTKV, r.slot)
        round_id = v(target, "round_id") + 1
        logical = v(target, "logical_kv_len")
        valid = (logical + worker.block_size - 1) // worker.block_size
        dispatch = view.row(K.REQUEST_DISPATCH, r.slot)
        op = max(round_id * 2, v(dispatch, "target_prepare_seq", 0) + 1,
                 v(dispatch, "target_run_seq", 0) + 1)
        items.append(TargetPrepareRequest(r.slot, r.epoch, round_id, op,
            r.output, r.output_count, r.prompt_count, r.config, r.host,
            v(host, "host_slot"), v(host, "host_slot_generation"), v(host, "writer_lease_generation"),
            v(target, "target_kv_version"), logical, valid, valid, bank_id, epoch, offset, r.capacity_blocks))
        offset += r.capacity_blocks
    return PrepareTargetBankCommand(worker.worker_id, worker.generation, sequence, sequence + 1,
                                    bank_id, epoch, tuple(items))


def run(view, worker, sequence, prepared, active):
    items = tuple(RunTargetRequest(r.request_slot, r.request_epoch, r.round_id, r.op_seq + 1,
        v(view.row(K.REQUEST_DRAFT, r.request_slot), "proposal_handle"), r.committed_output_handle)
        for r in prepared.requests)
    return RunTargetBatchCommand(worker.worker_id, worker.generation, sequence, prepared.batch_seq,
        prepared.standby_bank_id, prepared.next_bank_epoch, v(active, "bank_epoch"), items)
