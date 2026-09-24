from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from nebulasd.workers.draft.inputs import reconcile
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from test_autonomous_work import work


@pytest.mark.parametrize('accepted,delta,retained,suffix',[(0,(9,),7,(9,)),(2,(10,11,9),9,(9,)),(4,(10,11,12,13,9),10,(13,9))])
def test_reconcile_actual_cached_prefix(accepted,delta,retained,suffix):
    row=replace(work().rows[0],max_new_tokens=32)
    source=NS(proposal_count=4,prompt_count=4,committed_output_count=3,identity=NS(snapshot_version=8))
    result=reconcile(row,DraftGenerationConfig(32,4),delta,accepted,source,())
    assert result == (retained,suffix,3+len(delta),9,4)


def test_initial_and_depth_one_length_budget():
    row=replace(work().rows[0],max_new_tokens=3,token_budget=1)
    assert reconcile(row,DraftGenerationConfig(3,4),(8,),0,None,(1,2,3,4)) == (0,(1,2,3,4,8),1,1,1)
    source=NS(proposal_count=1,prompt_count=4,committed_output_count=1,identity=NS(snapshot_version=1))
    with pytest.raises(ValueError,match='budget'):
        reconcile(row,DraftGenerationConfig(3,4),(8,9),1,source,())


def test_source_snapshot_generation_allocation_and_epoch_boundary():
    from nebulasd.core.draft_contracts import DraftSnapshot,DraftSnapshotIdentity
    from nebulasd.core.handles import ArenaHandle
    from nebulasd.data.draft_snapshot_arena import SharedDraftSnapshotArena
    from nebulasd.data.shared_arenas import ArenaRouter
    from nebulasd.workers.draft.inputs import read_source,allocation
    from nebulasd.workers.work import TableDependency,Selector
    from nebulasd.core.enums import StateChangeBlockKind as K
    from test_autonomous_draft_publication import draft_work
    row=replace(draft_work().rows[0],round_id=4,run_seq=8,owner_epoch=22,
        source=TableDependency(K.REQUEST_DRAFT_D2H,0,1,3,Selector.DRAFT_HOST))
    alloc=allocation(row,13,16)
    identity=DraftSnapshotIdentity(0,1,3,7,2,37,21,3,6,1,alloc)
    source=DraftSnapshot(identity,row.prompt,row.config,ArenaHandle(64,4,23),ArenaHandle(0,16,51),4,1,2)
    arenas=[SharedDraftSnapshotArena(1024,generation=g,writer=True) for g in (61,67)]
    try:
        handle=arenas[1].write_snapshot(source)
        fields={n:getattr(alloc,n) for n in alloc.__dataclass_fields__}
        fields.update(snapshot_round_id=3,source_op_seq=7,source_worker_id=2,source_worker_generation=37,
            owner_epoch=21,snapshot_version=3,logical_kv_len=6,valid_blocks=1,ready_version=3,snapshot_handle=handle)
        router=ArenaRouter(arenas)
        assert read_source(row,fields,router,13,16)==source
        for bad in (replace(row,host_slot=99),replace(row,owner_epoch=21),replace(row,epoch=2)):
            with pytest.raises(ValueError):
                read_source(bad,fields,router,13,16)
    finally:
        for arena in arenas:
            arena.close();arena.segment.unlink()


def test_import_job_needs_no_target_delta_and_captures_source_once():
    from concurrent.futures import ThreadPoolExecutor
    from nebulasd.workers.draft.inputs import DraftInputs
    from nebulasd.core.draft_contracts import DraftSnapshot,DraftSnapshotIdentity
    from nebulasd.core.handles import ArenaHandle
    from nebulasd.data.draft_snapshot_arena import SharedDraftSnapshotArena
    from nebulasd.data.shared_arenas import ArenaRouter
    from nebulasd.workers.draft.inputs import allocation
    from nebulasd.workers.work import TableDependency,Selector
    from nebulasd.core.enums import StateChangeBlockKind as K
    from test_autonomous_draft_publication import draft_work
    row=replace(draft_work().rows[0],round_id=4,run_seq=8,owner_epoch=22,
        source=TableDependency(K.REQUEST_DRAFT_D2H,0,1,3,Selector.DRAFT_HOST))
    alloc=allocation(row,13,16)
    ident=DraftSnapshotIdentity(0,1,3,7,2,37,21,3,6,1,alloc)
    snapshot=DraftSnapshot(ident,row.prompt,row.config,ArenaHandle(64,4,23),ArenaHandle(0,16,51),4,1,2)
    arena=SharedDraftSnapshotArena(1024,generation=67,writer=True)
    role=DraftInputs(input_pool=ThreadPoolExecutor(2), tokens=None, configs=None, proposals=None,
        layout_id=row.layout_id, host_arena_id=row.host_arena, snapshots=ArenaRouter((arena,)),
        host=NS(descriptor=NS(descriptor_generation=13),make_extent=lambda **kw:NS(capacity_blocks=kw['capacity_blocks'])))
    try:
        handle=arena.write_snapshot(snapshot)
        fields={n:getattr(alloc,n) for n in alloc.__dataclass_fields__}
        fields.update(snapshot_round_id=3,source_op_seq=7,source_worker_id=2,source_worker_generation=37,
            owner_epoch=21,snapshot_version=3,logical_kv_len=6,valid_blocks=1,ready_version=3,snapshot_handle=handle)
        spec=NS(work_seq=19,rows=(row,))
        layout=NS(offsets=(8,),rows=(3,))
        # No predecessor/classification payload exists; the import job still completes.
        plan=role.compile_import(spec,({'source':fields},),layout,(0,)).result(1)
        assert [(r.index,r.local_row,r.blocks) for r in plan.rows]==[(0,3,1)]
        assert plan.regions[0].gpu_begin_block==8
        assert plan.copy_plan('metadata').dependencies==('metadata',)
        # Only the actual live import set is traversed, even with unavailable source.
        skipped=role.compile_import(spec,({},),layout,()).result(1)
        assert not skipped.rows and skipped.copy_plan('metadata') is None
        assert len(role.sources)==1
        role.retire(19)
        assert not role.sources
    finally:
        role.input_pool.shutdown()
        arena.close();arena.segment.unlink()
