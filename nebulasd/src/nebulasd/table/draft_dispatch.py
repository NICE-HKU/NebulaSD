"""Draft dispatch preflight and post-ring facts; owner changes only on run."""
from nebulasd.core.enums import StateChangeBlockKind as K, H2DStatus, TargetStatus
from nebulasd.core.ids import U64, OP_SEQ
from nebulasd.ipc.draft_protocol import PrepareDraftBankCommand, validate_prepared_run
from .draft_fences import (expect, read, publish, validate_source, validate_prepare,
                           validate_host_ready, prepare_values, h2d_values)
from .storage import TableProtocolError


class DraftDispatchContract:
    def __init__(self, table):
        self.table = table
        self.pending = {}  # At most one frozen next batch per destination worker.

    def preflight(self, command):
        if isinstance(command, PrepareDraftBankCommand):
            if command.worker_id in self.pending:
                raise TableProtocolError('unconsumed Draft prepare cannot be overwritten')
            occupied = {r.request_slot for p in self.pending.values() for r in p.requests}
            for row in command.requests:
                if row.request_slot in occupied:
                    raise TableProtocolError('request already has an outstanding Draft prepare')
                validate_source(self.table, row.source, row.snapshot_handle)
                dispatch = read(self.table, K.REQUEST_DISPATCH, row.request_slot)
                expect(dispatch, request_epoch=row.request_epoch, draft_owner_epoch=row.source.owner_epoch,
                       draft_worker_id=row.source.worker_id, draft_worker_generation=row.source.worker_generation,
                       draft_round_id=row.source.round_id, draft_issue_seq=row.source.op_seq)
                if not OP_SEQ.is_newer(row.prepare_seq, dispatch.get('draft_prepare_seq')):
                    raise TableProtocolError('stale Draft prepare sequence')
        else:
            prepare = self.pending.get(command.worker_id)
            if prepare is None:
                raise TableProtocolError('Draft run has no frozen prepare')
            validate_prepared_run(prepare, command)
            for p, r in zip(prepare.requests, command.requests, strict=True):
                validate_prepare(self.table, prepare, p)
                validate_host_ready(self.table, p.source, p.snapshot_handle)
                expect(read(self.table, K.REQUEST_DRAFT_H2D, r.request_slot), **h2d_values(prepare, p),
                       status=int(H2DStatus.GPU_READY), result_code=0,
                       gpu_ready_version=p.source.snapshot_version, copied_blocks=p.source.valid_blocks)
                dispatch = read(self.table, K.REQUEST_DISPATCH, r.request_slot)
                expect(dispatch, draft_owner_epoch=p.source.owner_epoch, draft_worker_id=p.source.worker_id,
                       draft_worker_generation=p.source.worker_generation, draft_issue_seq=p.source.op_seq)
                expect(read(self.table, K.REQUEST_TARGET_COMPUTE, r.request_slot),
                       request_epoch=r.request_epoch, round_id=p.source.round_id,
                       observed_run_seq=dispatch.get('target_run_seq'), committed_delta_handle=r.token_delta_handle,
                       status=int(TargetStatus.READY_DRAFT), result_code=0)

    def sent(self, command):
        # Called only after successful ring publication; any error is fail-stop.
        if isinstance(command, PrepareDraftBankCommand):
            for row in command.requests:
                publish(self.table, K.REQUEST_DISPATCH, row.request_slot, prepare_values(command, row))
            self.pending[command.worker_id] = command
        else:
            for row in command.requests:
                publish(self.table, K.REQUEST_DISPATCH, row.request_slot, dict(
                    request_epoch=row.request_epoch, draft_issue_seq=row.run_seq, draft_round_id=row.round_id,
                    draft_worker_id=command.worker_id, draft_worker_generation=command.worker_generation,
                    draft_owner_epoch=row.owner_epoch, draft_run_bank_id=command.active_bank_id,
                    draft_run_bank_epoch=command.active_bank_epoch, draft_run_batch_seq=command.expected_batch_seq))
            del self.pending[command.worker_id]
