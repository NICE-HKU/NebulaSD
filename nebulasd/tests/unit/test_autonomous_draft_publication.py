from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.core.draft_contracts import DraftSnapshot
from nebulasd.table.storage import RequestSchedulingTable
from nebulasd.data.shared_arenas import SharedProposalArena
from nebulasd.data.draft_snapshot_arena import SharedDraftSnapshotArena
from nebulasd.workers.completion import CompletionArena
from nebulasd.workers.draft.publication import DraftPublisher
from nebulasd.workers.work import WorkKind,TableDependency,Selector
from test_autonomous_work import work


def draft_work():
    w=work()
    row=replace(w.rows[0],owner_epoch=21,layout_id=77,prompt=ArenaHandle(0,16,17),
        output=ArenaHandle(64,64,23),config=ArenaHandle(0,16,41),
        predecessor=TableDependency(K.REQUEST_TARGET_COMPUTE,0,1,0,Selector.DELTA),
        classified=TableDependency(K.REQUEST_ENGINE,0,1,1,Selector.CLASSIFIED))
    return replace(w,operation=WorkKind.DRAFT_INITIAL,rows=(row,))


@pytest.mark.parametrize('physical_first',[True,False])
def test_payload_before_host_and_completion_budget(physical_first):
    proposals=SharedProposalArena(128,generation=51,writer=True)
    snapshots=SharedDraftSnapshotArena(1024,generation=61,writer=True)
    completions=CompletionArena(4096)
    table=RequestSchedulingTable(1)
    pub=DraftPublisher(table,proposals,snapshots,completions,host=NS(block_bytes=1024,descriptor_generation=13))
    w=draft_work()
    result=('DRAFT_RESULT',1,dict(compute_start_ns=10,compute_end_ns=20,rows=[dict(index=0,proposal=(20,21),proposal_kind=1,logical=6,
        version=1,dirty_begin=0,dirty_blocks=1,committed_count=1)]))
    physical=('PHYSICAL',1,dict(outcomes=[1],observed_ns=80,d2h_submitted_ns=60))
    try:
        pub.reserve(w)
        pub.consume(physical if physical_first else result)
        assert table.partition(K.REQUEST_DRAFT).read_publish_seq(0)==(1<<64)-1
        pub.step(1)
        assert table.partition(K.REQUEST_DRAFT_D2H).read_publish_seq(0)==(1<<64)-1
        pub.consume(result if physical_first else physical)
        while pub.records:
            pub.step(1)
            assert pub.last_published_rows<=1
        host=table.partition(K.REQUEST_DRAFT_D2H).read_stable(0)
        h=host.get('snapshot_handle')
        snap=DraftSnapshot.from_bytes(bytes(snapshots._bytes[h.offset:h.end_offset]))
        assert snap.identity.owner_epoch==21 and snap.identity.allocation.arena_generation==13
        assert snap.identity.logical_kv_len==6 and snap.proposal_count==2
        assert proposals.read_proposal(snap.proposal_handle).draft_token_ids==(20,21)
        assert completions.read(0).physical_done_ns==80
    finally:
        for a in (proposals,snapshots):
            a.close(); a.segment.unlink()
        completions.close(unlink=True)


def test_cross_arena_failure_is_atomic():
    proposals=SharedProposalArena(128,generation=51,writer=True)
    snapshots=SharedDraftSnapshotArena(DraftSnapshot.byte_size,generation=61,writer=True)
    completions=CompletionArena(4096)
    pub=DraftPublisher(RequestSchedulingTable(2),proposals,snapshots,completions,
        host=NS(block_bytes=1024,descriptor_generation=13))
    w=draft_work()
    r=w.rows[0]
    second=replace(r,slot=1,destination_offset=8,predecessor=replace(r.predecessor,slot=1),
        classified=replace(r.classified,slot=1))
    w=replace(w,rows=(r,second))
    try:
        for _ in range(3):
            with pytest.raises(ValueError,match='capacity'):
                pub.reserve(w)
            assert proposals._head==snapshots._head==0 and not pub.records
    finally:
        for a in (proposals,snapshots):
            a.close(); a.segment.unlink()
        completions.close(unlink=True)


def test_projection_coalesces_bank_epochs_without_replaying_old_free():
    from nebulasd.workers.draft.observation import Observation,Projection
    from nebulasd.workers.banks import Banks
    from nebulasd.table.storage import WorkerSchedulingRegistry
    banks=Banks(16,2)
    writer=Observation()
    reader=Observation(writer.segment.descriptor)
    table=WorkerSchedulingRegistry(1)
    projection=Projection(table,reader,16,2)
    try:
        writer.write(banks)
        projection.step()
        w=draft_work()
        banks.banks[0].current=NS(spec=w,jobs={})
        layout=banks.allocate(w)
        writer.write(banks)
        banks.release(layout)
        writer.write(banks)
        w2=replace(w,work_seq=2,bank_epoch=2)
        banks.banks[0].current=NS(spec=w2,jobs={})
        banks.allocate(w2)
        writer.write(banks)
        projection.step()
        fact=table.partition(K.WORKER_DRAFT_BANK).read_stable(0)
        assert fact.get('bank_epoch')==2 and fact.get('state')==2 and fact.get('alloc_rows')==1
        assert not projection.step()
    finally:
        reader.close();writer.close(unlink=True)


def test_all_finished_draft_has_no_result_snapshot_or_host_fact():
    proposals=SharedProposalArena(128,generation=51,writer=True)
    snapshots=SharedDraftSnapshotArena(1024,generation=61,writer=True)
    completions=CompletionArena(4096)
    table=RequestSchedulingTable(1)
    pub=DraftPublisher(table,proposals,snapshots,completions,host=NS(block_bytes=1024,descriptor_generation=13))
    try:
        pub.reserve(draft_work())
        pub.consume(('PHYSICAL',1,dict(outcomes=[2],observed_ns=80,d2h_submitted_ns=0)))
        assert pub.step()==(1,)
        assert pub.times is None and pub.take_times()==()
        assert completions.read(0).members[0].outcome==2
        for kind in (K.REQUEST_DRAFT,K.REQUEST_DRAFT_D2H,K.REQUEST_DRAFT_H2D):
            assert table.partition(kind).read_publish_seq(0)==(1<<64)-1
    finally:
        for arena in (proposals,snapshots):
            arena.close();arena.segment.unlink()
        completions.close(unlink=True)
