"""Read-only Draft fact joins shared by dispatch, copy writers and consumers."""
from nebulasd.core.enums import StateChangeBlockKind as K, DraftStatus, D2HStatus, H2DStatus, TargetStatus
from nebulasd.core.ids import U64, OP_SEQ
from .storage import TableProtocolError, FieldValue


def expect(row, **values):
    for name, value in values.items():
        if row.get(name) != value:
            raise TableProtocolError(f'Draft fence mismatch: {name}')


def expect_read(table, kind, slot, **values):
    # Decode only fields participating in this boundary join. The payload is
    # still copied and checked under the unchanged stable-read protocol.
    expect(table.partition(kind).read_stable(slot, field_names=tuple(values)), **values)


def read(table, kind, slot):
    return table.partition(kind).read_stable(slot)


def has_draft_dispatch(row):
    # Target prefill may have published this partition first. Its untouched
    # Draft fields are all zero; a canonical initial reservation has epoch >= 1.
    return any(row.get(n) for n in ('draft_issue_seq', 'draft_round_id', 'draft_worker_id',
        'draft_worker_generation', 'draft_run_bank_epoch', 'draft_run_batch_seq'))


def reset_dispatch_fields_for_epoch(table, slot, fields):
    from nebulasd.core.handles import ArenaHandle
    partition = table.partition(K.REQUEST_DISPATCH)
    values = {f.name: f.value for f in fields}
    if partition.read_publish_seq(slot) != U64.invalid:
        old = partition.read_stable(slot)
        if old.get('request_epoch') != values['request_epoch']:
            # Target prefill may be the first publisher after slot reuse. Do
            # not label the previous lifetime's Draft fields with the new epoch.
            return tuple(FieldValue(f.name, values.get(f.name,
                ArenaHandle.null() if f.type.name == 'arena_handle' else 0))
                for f in partition.layout.fields if f.name != 'publish_seq')
    return fields


def validate_initial_dispatch(table, command, request):
    if command.bank is None or request not in command.new_requests:
        raise TableProtocolError('request is not an initial Draft Bank member')
    expect_read(table, K.REQUEST_ENGINE, request.request_slot, request_epoch=request.request_epoch)
    expect_read(table, K.REQUEST_DISPATCH, request.request_slot,
        request_epoch=request.request_epoch, draft_issue_seq=request.op_seq,
        draft_round_id=request.round_id, draft_worker_id=command.worker_id,
        draft_worker_generation=command.worker_generation, draft_owner_epoch=0,
        draft_source_snapshot_version=0, draft_run_bank_id=command.bank.bank_id,
        draft_run_bank_epoch=command.bank.bank_epoch, draft_run_batch_seq=command.bank.batch_seq)


def next_seq(table, kind, slot):
    current = table.partition(kind).read_publish_seq(slot)
    return 0 if current == U64.invalid else OP_SEQ.next(current)


def publish(table, kind, slot, values):
    partition = table.partition(kind)
    table._publish_owned(owner=partition.layout.owner, block_kind=kind, row=slot,
        publish_seq=next_seq(table, kind, slot), fields=tuple(FieldValue(k, v) for k, v in values.items()))


def allocation_values(allocation):
    return {n: getattr(allocation, n) for n in allocation.__dataclass_fields__}


def validate_allocation(table, identity):
    expect_read(table, K.REQUEST_ENGINE, identity.request_slot, request_epoch=identity.request_epoch)
    expect_read(table, K.REQUEST_DRAFT_HOSTKV, identity.request_slot,
           request_epoch=identity.request_epoch, **allocation_values(identity.allocation))


def validate_source(table, identity, handle, **fields):
    validate_allocation(table, identity)
    expect_read(table, K.REQUEST_DRAFT, identity.request_slot,
        request_epoch=identity.request_epoch, round_id=identity.round_id,
        observed_issue_seq=identity.op_seq, worker_id=identity.worker_id,
        worker_generation=identity.worker_generation, owner_epoch=identity.owner_epoch,
        snapshot_version=identity.snapshot_version, logical_kv_len=identity.logical_kv_len,
        valid_blocks=identity.valid_blocks, draft_state_handle=handle,
        status=int(DraftStatus.READY_TARGET), result_code=0, **fields)


def prepare_values(command, row):
    return dict(request_epoch=row.request_epoch, draft_prepare_seq=row.prepare_seq,
        planned_draft_id=command.worker_id, planned_draft_generation=command.worker_generation,
        planned_draft_bank_id=command.standby_bank_id, planned_draft_bank_epoch=command.next_bank_epoch,
        draft_prepared_batch_seq=command.batch_seq, draft_source_snapshot_version=row.source.snapshot_version,
        draft_snapshot_round_id=row.source.round_id, draft_snapshot_handle=row.snapshot_handle,
        draft_next_owner_epoch=row.next_owner_epoch)


def validate_prepare(table, command, row, *, host_ready=False):
    validate_allocation(table, row.source)
    validate_prepare_dispatch(table, command, row)
    if host_ready:
        # One synchronous join checks the same allocation once. Nothing is
        # cached across calls, pin acquisition, or a copy's physical lifetime.
        _validate_host_ready_source(table, row.source, row.snapshot_handle)


def validate_prepare_dispatch(table, command, row):
    expect_read(table, K.REQUEST_DISPATCH, row.request_slot, **prepare_values(command, row),
        draft_owner_epoch=row.source.owner_epoch, draft_worker_id=row.source.worker_id,
        draft_worker_generation=row.source.worker_generation)


def validate_host_ready(table, identity, handle):
    validate_allocation(table, identity)
    _validate_host_ready_source(table, identity, handle)


def _validate_host_ready_source(table, identity, handle):
    expect_read(table, K.REQUEST_DRAFT_D2H, identity.request_slot,
        request_epoch=identity.request_epoch, snapshot_round_id=identity.round_id,
        snapshot_version=identity.snapshot_version, ready_version=identity.snapshot_version,
        snapshot_handle=handle, source_worker_id=identity.worker_id,
        source_worker_generation=identity.worker_generation, owner_epoch=identity.owner_epoch,
        source_op_seq=identity.op_seq, valid_blocks=identity.valid_blocks,
        logical_kv_len=identity.logical_kv_len, status=int(D2HStatus.HOST_READY), result_code=0,
        **allocation_values(identity.allocation))


def h2d_values(command, row):
    return dict(request_epoch=row.request_epoch, snapshot_round_id=row.source.round_id,
        snapshot_version=row.source.snapshot_version, snapshot_handle=row.snapshot_handle,
        logical_kv_len=row.source.logical_kv_len, next_round_id=row.next_round_id,
        observed_prepare_seq=row.prepare_seq, destination_worker_id=command.worker_id,
        destination_worker_generation=command.worker_generation, next_owner_epoch=row.next_owner_epoch,
        destination_bank_id=command.standby_bank_id, destination_bank_epoch=command.next_bank_epoch,
        prepared_batch_seq=command.batch_seq, **allocation_values(row.source.allocation))


def validate_run_dispatch(table, command, row):
    expect_read(table, K.REQUEST_ENGINE, row.request_slot, request_epoch=row.request_epoch)
    expect_read(table, K.REQUEST_DISPATCH, row.request_slot, request_epoch=row.request_epoch,
        draft_issue_seq=row.run_seq, draft_round_id=row.round_id, draft_worker_id=command.worker_id,
        draft_worker_generation=command.worker_generation, draft_owner_epoch=row.owner_epoch,
        draft_run_bank_id=command.active_bank_id, draft_run_bank_epoch=command.active_bank_epoch,
        draft_run_batch_seq=command.expected_batch_seq, draft_source_snapshot_version=row.snapshot_version)
