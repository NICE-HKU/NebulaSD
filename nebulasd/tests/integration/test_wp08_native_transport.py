"""Cross-process atomic visibility and bounded queue recovery, not a Python mock."""
import multiprocessing as mp
import os
from time import monotonic
import pytest

from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.ipc.native_ring import NativeRing, NativeStateChangeRing
from nebulasd.table.native_storage import request_table, table_descriptors, close_table_partitions
from nebulasd.table.reader import IncrementalTableReader
from nebulasd.table.storage import FieldValue, StableReadConflict

pytestmark = pytest.mark.skipif(not os.environ.get('STARSD_NEXT_NATIVE_LIBRARY'),reason='native library required')


def publish_rows(descriptors, count):
    table = request_table(1,descriptors=descriptors)
    try:
        p = table.partition(K.REQUEST_ENGINE)
        for seq in range(count):
            p._publish(0,seq,(FieldValue('request_epoch',seq+1),FieldValue('arrival_seq',seq+1)))
    finally:
        close_table_partitions(table._partitions)


def test_spawned_writer_never_exposes_torn_payload():
    table = request_table(1)
    process = mp.get_context('spawn').Process(target=publish_rows,args=(table_descriptors(table),2000))
    try:
        process.start()
        p = table.partition(K.REQUEST_ENGINE)
        seen, deadline = 0, monotonic()+10
        while process.is_alive():
            assert monotonic()<deadline
            try:
                row = p.read_stable(0)
            except StableReadConflict:
                continue
            assert row.get('request_epoch') == row.get('arrival_seq') == row.publish_seq+1
            seen += 1
        process.join()
        assert process.exitcode == 0 and seen > 0
        assert p.read_stable(0).publish_seq == 1999
    finally:
        if process.is_alive():
            process.kill(); process.join()
        close_table_partitions(table._partitions,unlink=True)


def test_u64_ring_wrap_and_backpressure():
    ring = NativeRing(4,8)
    try:
        near_wrap = (1<<64)-2
        ring.native.sd_store(ring.address,near_wrap)
        ring.native.sd_store(ring.address+64,near_wrap)
        for i in range(4):
            assert ring.can_push() and ring.push(i.to_bytes(8,'little'))
        assert not ring.can_push() and not ring.push(b'xxxxxxxx')
        for i in range(4):
            assert int.from_bytes(ring.peek(),'little')==i
            ring.ack()
        assert ring.peek() is None and ring.can_push()
    finally:
        ring.close(); ring.segment.unlink()


def test_overflow_recovers_latest_rows_and_conflicted_hint_is_retried(monkeypatch):
    ring = NativeStateChangeRing(2)
    table = request_table(4,ring=ring)
    reader = IncrementalTableReader(request_table=table,rings=(ring,))
    p = table.partition(K.REQUEST_ENGINE)
    try:
        for slot in range(4):
            p._publish(slot,0,(FieldValue('request_epoch',1),))
        batch = reader.poll()
        assert batch.overflow_recovered and {r.row for r in batch.views}==set(range(4))
        p._publish(0,1,(FieldValue('request_epoch',2),))
        original = p.read_stable
        def conflict(*args,**kwargs):
            raise StableReadConflict('injected concurrent publication')
        monkeypatch.setattr(p,'read_stable',conflict)
        assert not reader.poll().views
        monkeypatch.setattr(p,'read_stable',original)
        assert reader.poll().views[0].get('request_epoch')==2
    finally:
        close_table_partitions(table._partitions,unlink=True)
        ring.close(); ring.segment.unlink()


def test_command_payload_is_reused_only_after_copy_and_generation_is_checked():
    from nebulasd.ipc.native_command import NativeCommandRing, NativeCommandArena
    from nebulasd.ipc.command_arena import CommandBackpressure
    from nebulasd.ipc.command_ring import StaleWorkerGeneration
    from nebulasd.ipc.protocol import CommandHeader
    from nebulasd.ipc.protocol import CommandKind
    ring = NativeCommandRing(1)
    arena = NativeCommandArena(ring,slot_bytes=32)
    try:
        handle = arena.allocate(command_seq=1,payload=b'original')
        ring.publish(CommandHeader(1,1,CommandKind.DRAFT_BATCH,handle.offset,handle.length))
        with pytest.raises(CommandBackpressure):
            arena.allocate(command_seq=2,payload=b'overwrite')
        with pytest.raises(StaleWorkerGeneration):
            ring.consume(expected_worker_generation=2,arena=arena)
        assert ring.head()==0
        envelope = ring.consume(expected_worker_generation=1,arena=arena)
        arena.allocate(command_seq=2,payload=b'replacement')
        assert envelope.payload==b'original'
    finally:
        arena.close(); arena.segment.unlink()
        ring.close(); ring.segment.unlink()


def test_mapped_schema_mismatch_is_rejected():
    from dataclasses import replace
    from nebulasd.ipc.mapped_segment import MappedSegment
    segment = MappedSegment.create(64,'expected')
    try:
        with pytest.raises(ValueError,match='schema mismatch'):
            MappedSegment(replace(segment.descriptor,schema='wrong'))
    finally:
        segment.close(); segment.unlink()
