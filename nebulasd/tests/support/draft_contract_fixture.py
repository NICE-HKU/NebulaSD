"""CPU fact fixture; does not perform or claim any GPU migration."""
from nebulasd.core.draft_contracts import DraftHostAllocation, DraftSnapshotIdentity, DraftSnapshot
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import RequestFence
from nebulasd.table.draft_fences import publish
from nebulasd.table.writers import EngineTableWriter, DispatcherTableWriter
from nebulasd.table.draft_writers import DraftHostKVAllocatorWriter, DraftMigrationComputeWriter
from nebulasd.ipc.draft_protocol import DraftPrepareRequest, PrepareDraftBankCommand, DraftRunRequest, RunDraftBatchCommand


def snapshot():
    allocation = DraftHostAllocation(4, 7, 987, 2, 3, 8, 0, 64, 16)
    identity = DraftSnapshotIdentity(0, 5, 1, 10, 1, 9, 0, 1, 33, 3, allocation)
    return DraftSnapshot(identity, ArenaHandle(0, 120, 1), ArenaHandle(0, 16, 2),
                         ArenaHandle(120, 4, 1), ArenaHandle(0, 32, 3), 30, 1, 3)


def seed(table):
    s = snapshot()
    EngineTableWriter(table).publish_active(slot=0, publish_seq=0, request_epoch=5,
        current_round_id=1, arrival_seq=1, prompt_token_count=30, max_new_tokens=64, spec_token_limit=4)
    DispatcherTableWriter(table).publish_draft_command_sent(slot=0, publish_seq=0, request_epoch=5,
        draft_issue_seq=10, draft_worker_generation=9, draft_round_id=1, draft_worker_id=1)
    # Initial Bank preparation is a phase-two runtime concern. Seed its facts.
    publish(table, K.REQUEST_DISPATCH, 0, dict(draft_run_bank_id=0, draft_run_bank_epoch=1, draft_run_batch_seq=10))
    DraftHostKVAllocatorWriter(table).publish_allocation(request=RequestFence(0, 5), allocation=s.identity.allocation)
    handle = ArenaHandle(0, DraftSnapshot.byte_size, 10)
    compute(table, s, handle, 0, 1, 10)
    return s, handle


def compute(table, s, handle, bank, epoch, batch):
    DraftMigrationComputeWriter(table).publish_ready(snapshot=s, snapshot_handle=handle,
        bank_id=bank, bank_epoch=epoch, batch_seq=batch, dirty_begin_block=0, dirty_block_count=s.identity.valid_blocks)


def prepare(s, handle, *, worker=2, seq=0, prepare_seq=1, epoch=1):
    i = s.identity
    row = DraftPrepareRequest(i, handle, prepare_seq, i.round_id + 1, i.owner_epoch + 1, 0, 3, 5)
    return PrepareDraftBankCommand(worker, 9, seq, 20 + prepare_seq, 1, epoch, (row,))


def run(p):
    return RunDraftBatchCommand(p.worker_id, 9, p.command_seq + 1, p.batch_seq, p.standby_bank_id, p.next_bank_epoch,
        tuple(DraftRunRequest(r.request_slot, r.request_epoch, r.next_round_id, r.source.op_seq + 1,
            r.next_owner_epoch, r.source.snapshot_version, 4, ArenaHandle(200, 8, 1)) for r in p.requests))


def target_ready(table, s):
    # Explicit fake Target fact. No backend/validation work is performed here.
    publish(table, K.REQUEST_DISPATCH, 0, dict(target_run_seq=100 + s.identity.round_id))
    publish(table, K.REQUEST_TARGET_COMPUTE, 0, dict(request_epoch=s.identity.request_epoch,
        round_id=s.identity.round_id, observed_run_seq=100 + s.identity.round_id,
        committed_delta_handle=ArenaHandle(200, 8, 1), status=2, result_code=0))
