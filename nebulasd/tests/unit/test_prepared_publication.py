"""Prepared patches retain row seqlocks, notification hints and exact lifetimes."""
import os
import pytest
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.ipc.state_change_ring import StateChangeRing
from nebulasd.table.storage import RequestSchedulingTable, FieldValue, TableProtocolError
from nebulasd.table.prepared import PreparedPublication


@pytest.mark.parametrize('native', [False, True])
def test_batch_prechecks_all_rows_and_rejects_replaced_or_replayed_commit(native):
    ring = StateChangeRing(8)
    if native:
        if not os.environ.get('STARSD_NEXT_NATIVE_LIBRARY'):
            pytest.skip('native control library required')
        from nebulasd.table.native_storage import request_table, close_table_partitions
        table = request_table(2, ring=ring)
    else:
        table = RequestSchedulingTable(2, ring=ring)
    try:
        partition = table.partition(K.REQUEST_H2D)
        for row in range(2):
            partition._publish(row, 0, (FieldValue('request_epoch', 7), FieldValue('status', 1)))
        patch = PreparedPublication.capture(table, K.REQUEST_H2D,
            ((row, dict(status=2, gpu_ready_version=3)) for row in range(2)))
        partition._publish(1, 1, (FieldValue('request_epoch', 8),))
        with pytest.raises(TableProtocolError, match='stale'):
            patch.commit()
        assert partition.read_publish_seq(0) == 0  # No partial batch on stale last row.
        patch = PreparedPublication.capture(table, K.REQUEST_H2D, [(0, dict(status=2))])
        patch.commit()
        row = partition.read_stable(0, field_names=('request_epoch', 'status'))
        assert row.get('request_epoch') == 7 and row.get('status') == 2
        with pytest.raises(TableProtocolError, match='duplicate'):
            patch.commit()
        assert len(ring.drain().entries) == 4
    finally:
        if native:
            close_table_partitions(table._partitions, unlink=True)


@pytest.mark.parametrize('native', [False, True])
def test_draft_wait_fixed_encoder_matches_generic_bytes_and_sequences(native):
    from nebulasd.core.enums import H2DStatus
    from nebulasd.core.ids import U64
    from nebulasd.table.draft_fences import h2d_values
    from nebulasd.table.draft_wait import DraftWaitEncoder
    from support.draft_contract_fixture import snapshot, prepare
    from nebulasd.core.handles import ArenaHandle
    if native:
        if not os.environ.get('STARSD_NEXT_NATIVE_LIBRARY'):
            pytest.skip('native control library required')
        from nebulasd.table.native_storage import request_table, close_table_partitions
        tables = [request_table(1), request_table(1)]
    else:
        tables = [RequestSchedulingTable(1), RequestSchedulingTable(1)]
    try:
        command = prepare(snapshot(), ArenaHandle(1232, 200, 10))
        r = command.requests[0]
        values = dict(**h2d_values(command, r), status=int(H2DStatus.WAIT_HOST),
            result_code=0, local_row=0, gpu_ready_version=0, copied_blocks=0,
            copy_start_time_ns=0, copy_bytes=0)
        generic = PreparedPublication.capture(tables[0], K.REQUEST_DRAFT_H2D,
            [(r.request_slot, values)], initial=True)
        fast = DraftWaitEncoder(tables[1].partition(K.REQUEST_DRAFT_H2D)).capture(
            command, {r.request_slot: U64.invalid})
        assert generic.commit() == fast.commit() == {r.request_slot: 0}
        a, b = [t.partition(K.REQUEST_DRAFT_H2D) for t in tables]
        assert a._copy_payload(0) == b._copy_payload(0)
        with pytest.raises(TableProtocolError, match='duplicate'):
            fast.commit()
    finally:
        if native:
            for table in tables:
                close_table_partitions(table._partitions, unlink=True)
