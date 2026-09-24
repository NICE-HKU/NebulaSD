"""Latest shared facts replace receipt transactions; there is no batch gate."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from test_autonomous_engine import engine
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from nebulasd.workers.target.publication import TargetPublisher
from nebulasd.workers.work import WorkKind
from nebulasd.table.prepared import PreparedRow


def publisher(e):
    return TargetPublisher(e.resources.table, e.resources.completions,
        block_bytes=16, outputs=e.resources.token_router, configs=e.resources.configs)


def result(pub, work, tokens=(4,), logical=3):
    pub.reserve(work)
    pub.consume(('RESULT', work.work_seq, dict(rows=[dict(index=i, tokens=tokens,
        accepted=0, logical=logical, version=work.work_seq, dirty_begin=0, dirty_blocks=1)
        for i in range(len(work.rows))], compute_start_ns=10, compute_end_ns=20)))


def test_complete_request_schedules_while_peer_unpublished(engine):
    e = engine
    for name in ('a','b'): e.admit(name, (1,2,3), DraftGenerationConfig(12,4))
    e.step()
    work = e.supervisor.pairs[1].work[0]
    p = publisher(e); result(p, work)
    # Stop the real publisher between A and B, with no Engine receipt.
    assert p._publish_result(p.records[work.work_seq])
    e.step()
    assert e.registry.records[0].input.output_count == 1
    assert e.registry.records[1].input.output_count == 0
    drafts = e.supervisor.pairs[0].work
    assert drafts and {r.slot for w in drafts for r in w.rows} == {0}
    assert e.ledger.records[e.ledger.key(work)].completion is None


def test_coalesced_versions_deliver_entire_prefix_once(engine):
    e = engine
    e.admit('a', (1,2,3), DraftGenerationConfig(12,4)); e.step()
    first = e.supervisor.pairs[1].work[0]
    p = publisher(e); result(p, first)
    p.publish_results(first.work_seq)
    old = e.resources.table.partition(K.REQUEST_TARGET_COMPUTE).read_stable(0)
    # Authorize successors explicitly, as a protocol fixture. Engine observation
    # remains paused across all three publications.
    latest = first
    for round_id, tokens, logical in ((1,(5,6),5),(2,(7,8,9),8)):
        row = replace(first.rows[0], round_id=round_id, run_seq=round_id+1)
        latest = replace(first, work_seq=round_id+1, rows=(row,),
            bank_epoch=round_id+1, completion_offset=e.resources.completions.reserve(1))
        e.ledger.sent(latest, None)
        result(p, latest, tokens, logical); p.publish_results(latest.work_seq)
    e._observe_facts(); e.outputs.flush()
    assert e.registry.records[0].output == (4,5,6,7,8,9)
    assert e.registry.records[0].current_round == 2
    row = e.rows[K.REQUEST_TARGET_COMPUTE,0]
    # Manual entry follows reader uniqueness; duplicates across polls are inert.
    assert not e.apply_facts((old,))
    assert not e.apply_facts((row,))
    assert not e.apply_facts((row,))
    e._observe_facts();e.outputs.flush()
    assert e.registry.records[0].output == (4,5,6,7,8,9)
    assert all(0 in r.applied for r in e.ledger.records.values())


@pytest.mark.parametrize('field,value', [('bank_epoch',99),('target_id',0),('target_generation',9),('request_epoch',9),('observed_run_seq',9)])
def test_wrong_result_authorization_rejected_before_output(engine, field, value):
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    w=e.supervisor.pairs[1].work[0];p=publisher(e);result(p,w);p.publish_results(w.work_seq)
    part=e.resources.table.partition(K.REQUEST_TARGET_COMPUTE)
    PreparedRow(part,0,{field:value},()).publish(())
    with pytest.raises(ValueError): e._observe_facts()
    assert e.registry.records[0].output == ()


def test_partial_skips_retire_after_completion_and_executed_output(engine):
    e=engine
    for n in ('a','b'):e.admit(n,(1,2,3),DraftGenerationConfig(8,4))
    e.step();w=e.supervisor.pairs[1].work[0];p=publisher(e);p.reserve(w)
    p.consume(('RESULT',w.work_seq,dict(rows=[dict(index=0,tokens=(4,),accepted=0,logical=3,
        version=1,dirty_begin=0,dirty_blocks=1)],compute_start_ns=10,compute_end_ns=20)))
    p.publish_results(w.work_seq);e._observe_facts();e.ledger.observe_completions();e.ledger.refresh(e.rows)
    assert not e.ledger.records[e.ledger.key(w)].physical_done
    p.consume(('PHYSICAL',w.work_seq,dict(outcomes=[1,2],observed_ns=30,d2h_submitted_ns=21)))
    while p.records:p.step(8)
    e._observe_facts();e.ledger.observe_completions();e.ledger.refresh(e.rows);e.outputs.flush()
    assert e.ledger.key(w) not in e.ledger.records
    assert e.registry.records[0].output == (4,) and e.registry.records[1].output == ()


def test_migration_appends_same_prefix_and_keeps_old_delta_alive(engine):
    from nebulasd.workers.resources import PayloadDescriptor, attach_router
    from nebulasd.data.shared_arenas import SharedTokenArena
    from nebulasd.workers.work import TableDependency, Selector
    from nebulasd.scheduler.views import WorkerSpec
    from nebulasd.core.enums import WorkerRole
    e=engine
    e.resources.specs += (WorkerSpec(2,WorkerRole.TARGET),)
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    w=e.supervisor.pairs[1].work[0];first=publisher(e);result(first,w);first.publish_results(w.work_seq)
    old=e.resources.table.partition(K.REQUEST_TARGET_COMPUTE).read_stable(0)
    delta=old.get('committed_delta_handle')
    e._observe_facts();e.outputs.flush()
    # A distinct Target mapping has no allocation-head authority, only the
    # frozen request range. Its predecessor is still readable during append.
    router, attachments=attach_router(tuple(PayloadDescriptor.of(a) for a in e.resources.tokens),SharedTokenArena)
    try:
        row=replace(w.rows[0],round_id=1,run_seq=2,
            source=TableDependency(K.REQUEST_D2H,0,1,1,Selector.TARGET_HOST),
            predecessor=TableDependency(K.REQUEST_DRAFT,0,1,1,Selector.PROPOSAL),
            classified=TableDependency(K.REQUEST_TARGET_COMPUTE,0,1,0,Selector.TARGET_DECISION))
        migrated=replace(w,worker_id=2,work_seq=1,operation=WorkKind.TARGET_VERIFY,rows=(row,),
            completion_offset=e.resources.completions.reserve(1))
        e.ledger.sent(migrated,None)
        target=TargetPublisher(e.resources.table,e.resources.completions,outputs=router,
                              configs=e.resources.configs,block_bytes=16)
        result(target,migrated,(5,6),5);target.publish_results(migrated.work_seq)
        e._observe_facts();e.outputs.flush()
        assert e.registry.records[0].output==(4,5,6)
        assert router.read_tokens(delta)==(4,)
        assert all(not a._writer for a in attachments)
        assert e.rows[K.REQUEST_TARGET_COMPUTE,0].get('target_id')==2
    finally:
        for a in attachments:a.close()


def test_completion_mismatched_executed_members_is_rejected(engine):
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    w=e.supervisor.pairs[1].work[0];p=publisher(e);result(p,w)
    p.consume(('PHYSICAL',w.work_seq,dict(outcomes=[2],observed_ns=30,d2h_submitted_ns=21)))
    with pytest.raises(ValueError,match='completion members'):
        while p.records:p.step(8)
    assert e.resources.completions.read(w.completion_offset) is None


def test_completion_slot_cannot_be_reused_before_cohort(engine):
    from nebulasd.workers.completion import WorkCompletion, MemberCompletion
    from nebulasd.workers.work import Outcome
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    w=e.supervisor.pairs[1].work[0]
    c=WorkCompletion(w.worker_generation,w.work_seq,w.bank_epoch,20,w.bank_id,
                    (MemberCompletion(0,1,0,Outcome.SKIPPED_FINISHED),))
    e.resources.completions.publish(w.completion_offset,c);e.ledger.observe_completions();e.ledger.refresh(e.rows)
    assert not e.ledger.records
    with pytest.raises(RuntimeError,match='overwrite'):e.resources.completions.publish(w.completion_offset,c)


def test_profiling_all_current_engine_hooks(engine, tmp_path, monkeypatch):
    from nebulasd.observability.profiling import ProfileRecorder,attach_engine
    e=engine
    e.supervisor.bell.wait=lambda timeout:None
    e.supervisor.start=lambda:None
    monkeypatch.setenv('STARSD_ENGINE_DETAIL_DIR',str(tmp_path/'detail'))
    monkeypatch.setenv('STARSD_COMPLETION_TRACE_DIR',str(tmp_path/'completion'))
    monkeypatch.setenv('STARSD_BANK_TURNAROUND_PROFILE','1')
    recorder=ProfileRecorder(tmp_path,'engine',1000,mode='full')
    attach_engine(e,recorder)
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    recorder.flush()
    assert (tmp_path/'engine.json').is_file()
    assert 'engine.apply_facts' in (tmp_path/'engine.json').read_text()
    assert e.supervisor.pairs[1].work


def test_direct_import_backpressure_keeps_frozen_ranges(engine):
    from queue import Full
    e=engine;endpoint=e.supervisor.pairs[1]
    endpoint.direct_imports=True
    attempts=[]
    def submit(work, *, import_plan):
        attempts.append((work,import_plan))
        if len(attempts)<3:raise Full
        endpoint.work.append(work)
    endpoint.submit=submit
    e.admit('a',(1,2,3),DraftGenerationConfig(8,4))
    for _ in range(4):e.step()
    assert len(attempts)==3
    assert all(w is attempts[0][0] and ranges is attempts[0][1] for w,ranges in attempts)
    assert len(e.ledger.records)==1 and len(endpoint.work)==1
