"""Read-only guards: copy completion is terminal within one operation.

Progress publication is optional, completion is not. Repeated fact writes are
strictly rejected (including repeated completion); notification delivery may
repeat freely because it never invokes a writer. New identities are separately
authorized by allocation/source/dispatch fences before reaching these guards.
"""
from nebulasd.core.enums import D2HStatus, H2DStatus
from nebulasd.core.ids import U64
from .storage import TableProtocolError


def validate_copy_transition(table, kind, slot, values, *, direction):
    """Validate every fence and return the fields needed for this publication."""
    partition = table.partition(kind)
    if partition.read_publish_seq(slot) == U64.invalid:
        return values
    old = partition.read_stable(slot)
    version = 'snapshot_version' if direction == 'D2H' else 'observed_prepare_seq'
    for field in ('request_epoch', version):
        before, after = old.get(field), values[field]
        if before != after:
            if not U64.is_newer(after, before):
                raise TableProtocolError('stale Draft copy operation')
            return values  # A new operation must publish its full identity.

    mutable = {'status', 'result_code', 'copy_start_time_ns', 'copy_bytes',
               'ready_version', 'gpu_ready_version', 'copied_blocks', 'local_row'}
    for field, value in values.items():
        if field not in mutable:
            previous = old.get(field)
            if previous != value:
                raise TableProtocolError(f'Draft copy identity changed within operation: {field}')
            if type(previous) is not type(value):
                # Equality alone does not certify the scalar representation
                # (False == 0). Let the normal typed encoder validate it.
                mutable.add(field)
    if direction == 'D2H':
        allowed = {D2HStatus.IN_D2H: {D2HStatus.HOST_READY}, D2HStatus.HOST_READY: set()}
    else:
        allowed = {H2DStatus.WAIT_HOST: {H2DStatus.IN_H2D, H2DStatus.GPU_READY},
                   H2DStatus.IN_H2D: {H2DStatus.GPU_READY}, H2DStatus.GPU_READY: set()}
        if old.get('status') == H2DStatus.IN_H2D and old.get('local_row') != values['local_row']:
            raise TableProtocolError('Draft H2D row changed during copy')
    if values['status'] not in allowed.get(old.get('status'), set()):
        raise TableProtocolError('duplicate or regressing Draft copy fact')
    # The authorized partition has one owner. All identity fields above
    # matched the current row, so retain those bytes and publish only mutable
    # fields under the same seqlock/sequence/ring protocol. No old-row cache.
    return {field: value for field, value in values.items() if field in mutable}
