"""Native incremental observation retains exact versions and stable row reads."""
from contextlib import contextmanager
from dataclasses import replace
import multiprocessing as mp
import pytest
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.core.handles import ArenaHandle
from nebulasd.table.native_storage import request_table,close_table_partitions,table_descriptors
from nebulasd.table.prepared import PreparedRow
from nebulasd.workers.dependencies import Dependencies
from nebulasd.workers.work import TableDependency,Selector

@contextmanager
def table(n):
    t=request_table(n)
    try:yield t
    finally:close_table_partitions(t._partitions,unlink=True)

def test_native_captures_full_registered_batch_once_and_reregisters_published_rows():
    with table(128) as t:
        deps=Dependencies(t)
        for i in range(128):
            dep=TableDependency(K.REQUEST_DRAFT,i,1,7,Selector.PROPOSAL)
            deps.add((1,i,'predecessor'),dep)
        assert deps.poll()==[]
        for i in range(128):
            PreparedRow(t.partition(K.REQUEST_DRAFT),i,dict(request_epoch=1,round_id=7,status=2,
                proposal_handle=ArenaHandle(i*12,12,1)),()).publish(())
        events=deps.poll()
        assert len(events)==128 and len({e.key for e in events})==128
        assert deps.poll()==[] and not deps.pending
        deps.add((2,0,'predecessor'),TableDependency(K.REQUEST_DRAFT,0,1,7,Selector.PROPOSAL))
        assert len(deps.poll())==1

@pytest.mark.parametrize('epoch,round_id,error',[(2,7,'cohort'),(1,8,'overwritten')])
def test_native_rejects_stale_epoch_and_overwritten_ticket(epoch,round_id,error):
    with table(1) as t:
        deps=Dependencies(t)
        deps.add((1,0,'predecessor'),TableDependency(K.REQUEST_DRAFT,0,1,7,Selector.PROPOSAL))
        PreparedRow(t.partition(K.REQUEST_DRAFT),0,dict(request_epoch=epoch,round_id=round_id,status=2,
            proposal_handle=ArenaHandle(0,12,1)),()).publish(())
        with pytest.raises(RuntimeError,match=error):deps.poll()

def publish_racing(descriptors,start,done):
    t=request_table(1,descriptors=descriptors)
    try:
        p=PreparedRow(t.partition(K.REQUEST_TARGET_COMPUTE),0,dict(request_epoch=1,status=2,result_code=0),
            ('round_id','logical_kv_len','last_committed_token'))
        start.wait()
        for i in range(1,3001):p.publish((i,i,i))
    finally:
        close_table_partitions(t._partitions);done.set()

def test_native_scan_never_exposes_torn_result_under_concurrent_publication():
    from nebulasd.workers.dependencies import _FIELDS,_RULES,Watch
    from nebulasd.workers.native_dependencies import NativeDependencies
    with table(1) as t:
        ctx=mp.get_context('spawn');start=ctx.Event();done=ctx.Event()
        writer=ctx.Process(target=publish_racing,args=(table_descriptors(t),start,done));writer.start()
        dep=TableDependency(K.REQUEST_TARGET_COMPUTE,0,1,3000,Selector.TARGET_DECISION)
        watch=Watch((1,0,'classified'),dep);pending={watch.key:watch};scanner=NativeDependencies(t,_FIELDS,_RULES)
        seen=[];start.set()
        try:
            while not done.is_set():
                for row in scanner.scan(pending).values():
                    assert row['round_id']==row['logical_kv_len']==row['last_committed_token']
                    seen.append(row['round_id']);watch.last_seq=row.publish_seq
            for row in scanner.scan(pending).values():seen.append(row['round_id'])
            writer.join(10);assert writer.exitcode==0
            assert seen and max(seen)==3000
        finally:
            if writer.is_alive():writer.terminate();writer.join()

def test_fused_publication_keeps_shared_fact_when_hint_ring_overflows():
    from nebulasd.ipc.native_ring import NativeStateChangeRing
    ring=NativeStateChangeRing(2)
    t=request_table(3,ring=ring)
    try:
        for i in range(3):
            PreparedRow(t.partition(K.REQUEST_DRAFT),i,dict(request_epoch=1,round_id=7,status=2,
                proposal_handle=ArenaHandle(i*12,12,1)),()).publish(())
        hints=ring.drain()
        assert hints.overflowed and len(hints.entries)==2
        assert [e.row for e in hints.entries]==[0,1]
        deps=Dependencies(t)
        for i in range(3):deps.add((1,i,'predecessor'),TableDependency(K.REQUEST_DRAFT,i,1,7,Selector.PROPOSAL))
        assert len(deps.poll())==3
    finally:
        close_table_partitions(t._partitions,unlink=True);ring.close();ring.segment.unlink()
