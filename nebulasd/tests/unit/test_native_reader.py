"""Native and Python observation have identical bounded/recovery semantics."""
import random
import pytest
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.core.ids import U64
from nebulasd.table.native_storage import request_table,worker_table,close_table_partitions
from nebulasd.table.native_reader import NativeTableReader
from nebulasd.table.reader import IncrementalTableReader, ENGINE_IGNORED_KINDS
from nebulasd.engine.local_state import SchedulingSnapshotReader
from nebulasd.table.storage import FieldValue
from nebulasd.ipc.native_ring import NativeStateChangeRing
from nebulasd.ipc.state_change_ring import StateChangeEntry

@pytest.fixture
def readers():
    table,workers=request_table(12),worker_table(2)
    rings=[NativeStateChangeRing(8) for _ in range(4)]
    priority=((K.WORKER_BANK,0),(K.WORKER_BANK,1))
    native=NativeTableReader(request_table=table,worker_registry=workers,rings=tuple(rings[:2]),priority_rows=priority,max_entries=3)
    python=IncrementalTableReader(request_table=table,worker_registry=workers,rings=tuple(rings[2:]))
    python.max_entries=3;python.priority_rows=priority
    python.ignored_kinds=ENGINE_IGNORED_KINDS;python.read_snapshot=SchedulingSnapshotReader()
    yield native,python,table,workers,rings
    native._finalizer()
    for r in rings:r.close();r.segment.unlink()
    close_table_partitions(table._partitions,unlink=True)
    close_table_partitions(workers._partitions,unlink=True)


def compare(native,python):
    a,b=native.poll(),python.poll()
    for batch in (a,b):
        assert len({(r.block_kind,r.row) for r in batch.views}) == len(batch.views)
    key=lambda batch:[(r.block_kind,r.row,r.publish_seq,r.payload) for r in batch.views]
    assert key(a)==key(b)
    assert a.scanned_rows==b.scanned_rows
    assert a.overflow_recovered==b.overflow_recovered
    assert native.has_pending()==python.has_pending()
    return a


def test_native_reader_matches_bounded_rotation_overflow_and_coalescing(readers):
    n,p,t,w,rings=readers;rng=random.Random(73);seqs={}
    for turn in range(180):
        for _ in range(rng.randrange(20)):
            kind=rng.choice((K.REQUEST_D2H,K.REQUEST_DRAFT_D2H,K.REQUEST_ENGINE,
                             K.REQUEST_H2D,K.REQUEST_DRAFT_H2D,K.WORKER_BANK))
            slot=rng.randrange(4 if kind==K.WORKER_BANK else 12)
            table=w if kind==K.WORKER_BANK else t
            key=(kind,slot);seq=seqs[key]=seqs.get(key,0)+1
            table.partition(kind)._publish(slot,seq,(FieldValue('bank_epoch' if kind==K.WORKER_BANK else 'request_epoch',seq),))
            index=rng.randrange(2)
            for r in (rings[index],rings[index+2]):r.push(StateChangeEntry(kind,slot,seq))
        compare(n,p)
    for _ in range(500):
        compare(n,p)
        if not n.has_pending():break
    else:pytest.fail('recovery did not converge')


def test_native_reader_retries_invalid_row_without_new_hint_and_keeps_snapshot(readers):
    n,p,t,w,rings=readers;part=t.partition(K.REQUEST_D2H)
    for r in (rings[0],rings[2]):r.push(StateChangeEntry(K.REQUEST_D2H,0,1))
    assert not compare(n,p).views and n.has_pending()
    part._publish(0,1,(FieldValue('request_epoch',1),))
    first=compare(n,p).views[0]
    part._publish(0,2,(FieldValue('request_epoch',2),))
    for r in (rings[0],rings[2]):r.push(StateChangeEntry(K.REQUEST_D2H,0,2))
    assert compare(n,p).views[0].get('request_epoch')==2
    assert first.get('request_epoch')==1


def test_native_reader_reset_observes_reused_request_version(readers):
    n,p,t,w,rings=readers;part=t.partition(K.REQUEST_D2H)
    for epoch in (1,2):
        part.native.sd_store(part.segment.address,U64.invalid)
        part._publish(0,0,(FieldValue('request_epoch',epoch),))
        for r in (rings[0],rings[2]):r.push(StateChangeEntry(K.REQUEST_D2H,0,0))
        assert compare(n,p).views[0].get('request_epoch')==epoch
        n.reset_requests({K.REQUEST_D2H});p.reset_requests({K.REQUEST_D2H});p.reset_pending()


def test_native_reader_version_wrap_and_priority_deduplicate(readers):
    n,p,t,w,rings=readers;part=w.partition(K.WORKER_BANK)
    for seq in (U64.max_valid,0,1):
        part._publish(0,seq,(FieldValue('bank_epoch',seq),))
        for r in (rings[0],rings[2]):r.push(StateChangeEntry(K.WORKER_BANK,0,seq))
        assert len(compare(n,p).views)==1
        assert not compare(n,p).views


def _publish_concurrently(table_descriptor,ring_descriptor,count):
    ring=NativeStateChangeRing(8,ring_descriptor)
    table=request_table(12,descriptors=table_descriptor,ring=ring)
    try:
        for seq in range(1,count+1):
            table.partition(K.REQUEST_D2H)._publish(0,seq,
                (FieldValue('request_epoch',seq),FieldValue('committed_blocks',seq)))
    finally:
        close_table_partitions(table._partitions);ring.close()


def test_native_reader_concurrent_publication_never_accepts_torn_fields(readers):
    import multiprocessing as mp
    from time import monotonic
    from nebulasd.table.native_storage import table_descriptors
    n,p,t,w,rings=readers
    process=mp.get_context('spawn').Process(target=_publish_concurrently,
        args=(table_descriptors(t),rings[0].segment.descriptor,5000))
    process.start();last=0;deadline=monotonic()+20
    try:
        while monotonic()<deadline and (process.is_alive() or n.has_pending() or last<5000):
            for row in n.poll().views:
                if row.block_kind==K.REQUEST_D2H and row.row==0:
                    assert row.get('request_epoch')==row.get('committed_blocks')==row.publish_seq
                    assert row.publish_seq>last
                    last=row.publish_seq
        process.join(5)
        assert process.exitcode==0 and last==5000
    finally:
        if process.is_alive():process.terminate();process.join()
