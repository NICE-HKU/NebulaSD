"""Owner batching preserves immutable snapshots and bounded loss-tolerant hints."""
import multiprocessing as mp
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.engine.local_state import SchedulingRow
from nebulasd.ipc.native_ring import NativeStateChangeRing
from nebulasd.ipc.state_change_ring import StateChangeEntry
from nebulasd.table.storage import RequestSchedulingTable


def test_partial_dispatch_preserves_other_stage_and_old_snapshot():
    old = SchedulingRow.local(K.REQUEST_DISPATCH, 0, 0,
                    dict(request_epoch=7, target_run_seq=13, target_round_id=12))
    new = SchedulingRow.local(K.REQUEST_DISPATCH, 0, 1,
                    dict(old._values, draft_issue_seq=14, draft_round_id=13))
    assert new.get('target_run_seq') == 13 and new.get('request_epoch') == 7
    assert new.get('draft_issue_seq') == 14 and 'draft_issue_seq' not in old._values
    assert {f.name:f.value for f in old.fields}['target_round_id'] == 12


def test_bulk_hint_drain_limit_wrap_overflow_and_zero():
    ring = NativeStateChangeRing(4)
    try:
        for i in range(4):ring.push(StateChangeEntry(K.REQUEST_D2H, i, i))
        assert not ring.push(StateChangeEntry(K.REQUEST_D2H, 4, 4)).accepted
        batch = ring.drain(0)
        assert batch.overflowed and not batch.entries and ring.has_pending()
        assert [e.row for e in ring.drain(3).entries] == [0, 1, 2]
        for i in range(4, 7):ring.push(StateChangeEntry(K.REQUEST_D2H, i, i))
        assert [e.row for e in ring.drain().entries] == [3, 4, 5, 6]
        assert not ring.has_pending() and not ring.drain().overflowed
    finally:
        ring.close();ring.segment.unlink()


def _produce(descriptor, count):
    ring = NativeStateChangeRing(16, descriptor)
    try:
        for i in range(count):
            # Test producer backpressure, not production busy polling.
            while not ring.push(StateChangeEntry(K.REQUEST_D2H, i, i)).accepted:pass
    finally:ring.close()


def test_bulk_hint_drain_concurrent_producer_keeps_order():
    ring = NativeStateChangeRing(16)
    process = mp.get_context('spawn').Process(target=_produce,args=(ring.segment.descriptor, 10000))
    try:
        process.start();rows=[]
        from time import monotonic
        deadline=monotonic()+20
        while len(rows)<10000 and monotonic()<deadline:
            rows.extend(e.row for e in ring.drain(7).entries)
        process.join(5)
        assert process.exitcode == 0
        assert rows == list(range(10000))
    finally:
        if process.is_alive():process.terminate();process.join()
        ring.close();ring.segment.unlink()
