"""Single-owner Draft allocation, compute and independent copy facts.

Sequence numbers are read from the shared row, never restarted at destination.
The dispatcher guarantees one authorized writer per partition and request.
"""
from nebulasd.core.enums import StateChangeBlockKind as K, DraftStatus, D2HStatus, H2DStatus, validate_enum
from nebulasd.core.ids import U32, U64
from nebulasd.core.draft_contracts import require_snapshot_handle
from .draft_copy_transitions import validate_copy_transition
from .draft_fences import (expect, expect_read, read, publish, allocation_values, validate_allocation,
                           validate_source, validate_prepare, h2d_values)


def d2h_values(identity, snapshot_handle, bank_id, bank_epoch, batch_seq, status,
               copy_start_time_ns=0, copy_bytes=0):
    return dict(
            request_epoch=identity.request_epoch, snapshot_round_id=identity.round_id,
            snapshot_version=identity.snapshot_version, snapshot_handle=snapshot_handle,
            logical_kv_len=identity.logical_kv_len, valid_blocks=identity.valid_blocks,
            source_worker_id=identity.worker_id, source_worker_generation=identity.worker_generation,
            source_op_seq=identity.op_seq, owner_epoch=identity.owner_epoch,
            source_bank_id=bank_id, source_bank_epoch=bank_epoch, source_batch_seq=batch_seq,
            ready_version=identity.snapshot_version if status == D2HStatus.HOST_READY else 0,
            status=int(status), result_code=0, copy_start_time_ns=copy_start_time_ns, copy_bytes=copy_bytes,
            **allocation_values(identity.allocation))



class DraftHostKVAllocatorWriter:
    def __init__(self, table): self.table = table

    def publish_allocation(self, *, request, allocation):
        expect_read(self.table, K.REQUEST_ENGINE, request.request_slot, request_epoch=request.request_epoch)
        publish(self.table, K.REQUEST_DRAFT_HOSTKV, request.request_slot,
                dict(request_epoch=request.request_epoch, **allocation_values(allocation)))


class DraftMigrationComputeWriter:
    def __init__(self, table): self.table = table

    def publish_initial_running(self, *, command, request):
        from nebulasd.core.handles import ArenaHandle
        from .draft_fences import validate_initial_dispatch
        validate_initial_dispatch(self.table, command, request)
        bank = command.bank
        publish(self.table, K.REQUEST_DRAFT, request.request_slot, dict(
            request_epoch=request.request_epoch, round_id=request.round_id,
            observed_issue_seq=request.op_seq, worker_id=command.worker_id,
            worker_generation=command.worker_generation, owner_epoch=0,
            snapshot_version=1, logical_kv_len=0, valid_blocks=0,
            draft_state_handle=ArenaHandle.null(), proposal_handle=ArenaHandle.null(),
            proposal_token_count=0, bank_id=bank.bank_id, bank_epoch=bank.bank_epoch,
            batch_seq=bank.batch_seq, dirty_begin_block=0, dirty_block_count=0,
            status=int(DraftStatus.IN_DRAFT), result_code=0))

    def publish_running(self, *, command, request):
        from nebulasd.core.handles import ArenaHandle
        from .draft_fences import validate_run_dispatch
        if request not in command.requests:
            raise ValueError('compute request is not a run member')
        validate_run_dispatch(self.table, command, request)
        publish(self.table, K.REQUEST_DRAFT, request.request_slot, dict(
            request_epoch=request.request_epoch, round_id=request.round_id,
            observed_issue_seq=request.run_seq, worker_id=command.worker_id,
            worker_generation=command.worker_generation, owner_epoch=request.owner_epoch,
            snapshot_version=U64.next(request.snapshot_version), logical_kv_len=0, valid_blocks=0,
            draft_state_handle=ArenaHandle.null(), proposal_handle=ArenaHandle.null(), proposal_token_count=0,
            bank_id=command.active_bank_id, bank_epoch=command.active_bank_epoch,
            batch_seq=command.expected_batch_seq, dirty_begin_block=0, dirty_block_count=0,
            status=int(DraftStatus.IN_DRAFT), result_code=0))

    def publish_ready(self, *, snapshot, snapshot_handle, bank_id, bank_epoch, batch_seq,
                      dirty_begin_block, dirty_block_count):
        from nebulasd.ipc.draft_protocol import bank_id as check_bank
        check_bank(bank_id)
        U64.validate(bank_epoch)
        U64.validate(batch_seq)
        U32.validate(dirty_begin_block)
        U32.validate(dirty_block_count)
        require_snapshot_handle(snapshot_handle)
        identity = snapshot.identity
        validate_allocation(self.table, identity)
        dispatch = read(self.table, K.REQUEST_DISPATCH, identity.request_slot)
        expect(dispatch, request_epoch=identity.request_epoch,
            draft_issue_seq=identity.op_seq, draft_round_id=identity.round_id,
            draft_worker_id=identity.worker_id, draft_worker_generation=identity.worker_generation,
            draft_owner_epoch=identity.owner_epoch, draft_run_bank_id=bank_id,
            draft_run_bank_epoch=bank_epoch, draft_run_batch_seq=batch_seq)
        if identity.snapshot_version != U64.next(dispatch.get('draft_source_snapshot_version')):
            raise ValueError('compute snapshot version must advance dispatched source version')
        if dirty_begin_block + dirty_block_count != identity.valid_blocks:
            raise ValueError('dirty range must cover the actual KV tail; an empty range starts at valid_blocks')
        publish(self.table, K.REQUEST_DRAFT, identity.request_slot, dict(
            request_epoch=identity.request_epoch, round_id=identity.round_id,
            observed_issue_seq=identity.op_seq, worker_id=identity.worker_id,
            worker_generation=identity.worker_generation, owner_epoch=identity.owner_epoch,
            snapshot_version=identity.snapshot_version, logical_kv_len=identity.logical_kv_len,
            valid_blocks=identity.valid_blocks, draft_state_handle=snapshot_handle,
            proposal_handle=snapshot.proposal_handle, proposal_token_count=snapshot.proposal_count,
            bank_id=bank_id, bank_epoch=bank_epoch, batch_seq=batch_seq,
            dirty_begin_block=dirty_begin_block, dirty_block_count=dirty_block_count,
            status=int(DraftStatus.READY_TARGET), result_code=0))


class DraftSourceCopyWriter:
    def __init__(self, table): self.table = table

    def publish_d2h(self, *, identity, snapshot_handle, bank_id, bank_epoch, batch_seq,
                    status, copy_start_time_ns=0, copy_bytes=0):
        status = validate_enum(D2HStatus, status)
        if status not in (D2HStatus.IN_D2H, D2HStatus.HOST_READY):
            raise ValueError('expected IN_D2H or HOST_READY')
        validate_source(self.table, identity, snapshot_handle)
        expect_read(self.table, K.REQUEST_DRAFT, identity.request_slot,
               bank_id=bank_id, bank_epoch=bank_epoch, batch_seq=batch_seq)
        values = d2h_values(identity, snapshot_handle, bank_id, bank_epoch, batch_seq, status,
                            copy_start_time_ns, copy_bytes)
        values = validate_copy_transition(self.table, K.REQUEST_DRAFT_D2H, identity.request_slot, values, direction='D2H')
        publish(self.table, K.REQUEST_DRAFT_D2H, identity.request_slot, values)


class DraftDestinationCopyWriter:
    def __init__(self, table):
        self.table = table
        from .draft_wait import DraftWaitEncoder
        self._wait_encoder = DraftWaitEncoder(table.partition(K.REQUEST_DRAFT_H2D))

    def validate_wait_batch(self, command, *, dispatch_validated=False):
        """Freeze authorization and old row sequences before accepting intent."""
        from .draft_fences import validate_prepare_dispatch
        from .storage import TableProtocolError
        partition = self.table.partition(K.REQUEST_DRAFT_H2D)
        sequences = {}
        for request in command.requests:
            if not dispatch_validated:
                validate_prepare_dispatch(self.table, command, request)
            seq = partition.read_publish_seq(request.request_slot)
            if seq != U64.invalid:
                old = partition.read_stable(request.request_slot,
                    field_names=('request_epoch', 'observed_prepare_seq'))
                seq = old.publish_seq
                for name, after in (('request_epoch', request.request_epoch),
                                    ('observed_prepare_seq', request.prepare_seq)):
                    before = old.get(name)
                    if before != after:
                        if not U64.is_newer(after, before):
                            raise TableProtocolError('stale Draft copy operation')
                        break
                else:
                    # WAIT cannot follow any state of the same operation.
                    raise TableProtocolError('duplicate or regressing Draft copy fact')
            sequences[request.request_slot] = seq
        return sequences

    def publish_wait_batch(self, command, *, dispatch_validated=False):
        sequences = self.validate_wait_batch(command, dispatch_validated=dispatch_validated)
        return self._wait_encoder.capture(command, sequences).commit()

    def publish_h2d(self, *, command, request, status, local_row=0, copy_start_time_ns=0, copy_bytes=0):
        if request not in command.requests:
            raise ValueError('H2D request is not a prepared member')
        status = validate_enum(H2DStatus, status)
        if status not in (H2DStatus.WAIT_HOST, H2DStatus.IN_H2D, H2DStatus.GPU_READY):
            raise ValueError('invalid Draft H2D status')
        U32.validate(local_row)
        validate_prepare(self.table, command, request, host_ready=status != H2DStatus.WAIT_HOST)
        values = dict(
            **h2d_values(command, request), status=int(status), result_code=0, local_row=local_row,
            gpu_ready_version=request.source.snapshot_version if status == H2DStatus.GPU_READY else 0,
            copied_blocks=request.source.valid_blocks if status == H2DStatus.GPU_READY else 0,
            copy_start_time_ns=copy_start_time_ns, copy_bytes=copy_bytes)
        values = validate_copy_transition(self.table, K.REQUEST_DRAFT_H2D, request.request_slot, values, direction='H2D')
        publish(self.table, K.REQUEST_DRAFT_H2D, request.request_slot, values)
