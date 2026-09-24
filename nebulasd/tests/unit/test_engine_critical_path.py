"""Single scan, independent credit, bounded indexing and per-batch dirty work."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from test_autonomous_engine import engine, publish_target
from test_work_publications import publisher, result
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from nebulasd.engine.local_state import SchedulingRow
from nebulasd.engine.autonomous_supervisor import AutonomousSupervisor
from nebulasd.workers.completion import WorkCompletion, MemberCompletion
from nebulasd.workers.work import Outcome


def issued(e):
    e.admit('a',(1,2,3),DraftGenerationConfig(12,4));e.step()
    return e.supervisor.pairs[1].work[0]


def complete(e,w):
    e.resources.completions.publish(w.completion_offset,WorkCompletion(w.worker_generation,
        w.work_seq,w.bank_epoch,30,w.bank_id,
        tuple(MemberCompletion(r.slot,r.epoch,r.round_id,Outcome.EXECUTED) for r in w.rows)))


def test_completion_credit_released_before_dispatch_but_authorization_kept(engine,monkeypatch):
    e=engine;w=issued(e)
    for i in range(1,4):
        other=replace(w,work_seq=i+1,bank_epoch=i+1,rows=(replace(w.rows[0],round_id=i,run_seq=i+1),),completion_offset=e.resources.completions.reserve(1))
        e.ledger.sent(other,None)
    for wid,p in e.supervisor.pairs.items():
        p.pending_work=dict(e.ledger.by_worker[wid]);p.retire=lambda seq,p=p:p.pending_work.pop(seq)
    e.supervisor.retire_completed=lambda ledger:AutonomousSupervisor.retire_completed(e.supervisor,ledger)
    p=publisher(e);result(p,w)
    complete(e,w);reads=[];read=e.resources.completions.read
    monkeypatch.setattr(e.resources.completions,'read',lambda offset:(reads.append(offset),read(offset))[1])
    def dispatch_boundary():
        assert len(e.supervisor.pairs[1].pending_work)==3
        assert len(e.ledger.by_worker[1])==4 and len(e.ledger.request_index)==4
        assert not e.ledger.records[e.ledger.key(w)].applied
        return 0
    monkeypatch.setattr(e.scheduling_progress,'advance',dispatch_boundary)
    e.step();assert len(reads)==4
    reads.clear();e.step();assert len(reads)==3 and w.completion_offset not in reads
    p.publish_results(w.work_seq)
    e._observe_facts();e.ledger.refresh(e.rows)
    assert e.ledger.key(w) not in e.ledger.records
    assert len(e.ledger.request_index)==3
    assert (0,1) not in e.ledger.by_request[False,0,1]


@pytest.mark.parametrize('phase',[1,2])
def test_completion_after_scan_waits_for_next_step(engine,monkeypatch,phase):
    e=engine;w=issued(e);e._observe_facts()
    e.scheduling_progress.dirty={'D':set(),'T':set()}
    e._retry_dispatch=True
    reads=[];read=e.resources.completions.read
    monkeypatch.setattr(e.resources.completions,"read",lambda offset:(reads.append(offset),read(offset))[1])
    original=e._observe_facts;calls=[];scans=[];scan=e.ledger.observe_completions
    monkeypatch.setattr(e.ledger,'observe_completions',lambda:(scans.append(1),scan())[1])
    def observe():
        calls.append(1)
        if len(calls)==phase+1:complete(e,w)
        return original()
    monkeypatch.setattr(e,'_observe_facts',observe)
    e.step()
    assert len(calls)==3 and len(scans)==1 and reads==[w.completion_offset]
    assert e.ledger.records[e.ledger.key(w)].completion is None
    e.step();assert len(scans)==2
    assert e.ledger.records[e.ledger.key(w)].completion is not None


def test_failed_output_validation_does_not_mark_result_applied(engine):
    e=engine;w=issued(e);p=publisher(e);result(p,w);p.publish_results(w.work_seq)
    from nebulasd.table.prepared import PreparedRow
    PreparedRow(e.resources.table.partition(K.REQUEST_TARGET_COMPUTE),0,dict(output_count=100),()).publish(())
    with pytest.raises(ValueError,match='cumulative output'):e._observe_facts()
    assert not e.ledger.records[e.ledger.key(w)].applied


@pytest.mark.parametrize('batch',[1,16,64])
def test_authorization_lookup_never_iterates_work_members(engine,batch):
    e=engine;w=issued(e)
    # Isolate validation from completion member checking, which remains linear.
    e.ledger.records.clear();e.ledger.request_index.clear();e.ledger.by_request.clear();e.ledger.by_worker[1].clear()
    w=replace(w,rows=tuple(replace(w.rows[0],slot=i,destination_offset=i*w.rows[0].capacity_blocks) for i in range(batch)))
    e.ledger.sent(w,None)
    record=e.ledger.records[e.ledger.key(w)]
    class NoIteration(tuple):
        def __iter__(self):raise AssertionError('linear WORK member lookup')
    object.__setattr__(record.work, "rows", NoIteration(w.rows))
    for i in range(batch):
        row=SchedulingRow.local(K.REQUEST_TARGET_COMPUTE,i,1,dict(status=2,request_epoch=1,
            round_id=0,observed_run_seq=1,target_id=1,target_generation=1,bank_id=w.bank_id,bank_epoch=w.bank_epoch,result_code=0))
        context=e.ledger.validate_fact(row)
        assert context.member.slot==i and not record.applied
        copy=SchedulingRow.local(K.REQUEST_H2D,i,1,dict(status=3,request_epoch=1,round_id=0,
            observed_prepare_seq=1,target_id=1,target_generation=1,destination_bank_id=w.bank_id,
            destination_bank_epoch=w.bank_epoch,result_code=0))
        assert e.ledger.validate_fact(copy)


def test_skipped_versions_visit_only_same_request_authorizations(engine):
    e=engine;w=issued(e)
    later=replace(w,work_seq=2,rows=(replace(w.rows[0],round_id=3,run_seq=4),),bank_epoch=2)
    unrelated=replace(w,work_seq=3,rows=(replace(w.rows[0],slot=1),))
    e.ledger.sent(later,None);e.ledger.sent(unrelated,None)
    class NoScan(dict):
        def values(self):raise AssertionError('all WORKs scanned')
    e.ledger.records=NoScan(e.ledger.records)
    row=SchedulingRow.local(K.REQUEST_TARGET_COMPUTE,0,1,dict(status=2,request_epoch=1,
        round_id=3,observed_run_seq=4,target_id=1,target_generation=1,bank_id=later.bank_id,bank_epoch=2,result_code=0))
    context=e.ledger.validate_fact(row)
    assert not context.record.applied
    e.ledger.result_applied(context,None)
    assert e.ledger.records[e.ledger.key(w)].applied=={0}
    assert context.record.applied=={0}
    assert not e.ledger.records[e.ledger.key(unrelated)].applied


def test_batch_dirty_merges_bounded_and_all_scheduler_rows_updated(engine,monkeypatch):
    e=engine
    for i in range(4):e.admit(str(i),(1,2,3),DraftGenerationConfig(12,4))
    e.step();w=e.supervisor.pairs[1].work[0];publish_target(e,w)
    merges=[];observations=[];invalidations=[]
    class Dirty(set):
        def update(self,values):merges.append(tuple(values));return super().update(values)
    e.scheduling_progress.dirty={k:Dirty(v) for k,v in e.scheduling_progress.dirty.items()}
    original=e.scheduler.observe_result
    monkeypatch.setattr(e.scheduler,'observe_result',lambda row:(observations.append((row.block_kind,row.row)),original(row))[1])
    invalidate=e.scheduler.requests_changed
    monkeypatch.setattr(e.scheduler,'requests_changed',lambda slots:(invalidations.append(set(slots)),invalidate(slots))[1])
    monkeypatch.setattr(e.scheduler,'observe_table',lambda row:pytest.fail('generic per-row scheduler notification'))
    e._observe_facts()
    assert len(merges)==2
    assert {slot for kind,slot in observations if kind==K.REQUEST_TARGET_COMPUTE}=={r.slot for r in w.rows}
    assert invalidations==[{r.slot for r in w.rows}]
    assert not any(kind==K.REQUEST_ENGINE for kind,slot in e.rows)
    assert all(e.candidates[r.slot].output_count==1 for r in w.rows)


@pytest.mark.parametrize('reverse',[False,True])
def test_copy_and_result_validate_against_prebatch_cache(engine,reverse):
    e=engine;w=issued(e);publish_target(e,w)
    updates=e.reader.poll().views
    facts=tuple(r for r in updates if r.block_kind in (K.REQUEST_D2H,K.REQUEST_TARGET_COMPUTE))
    if reverse:facts=facts[::-1]
    e.apply_facts(facts);e.ledger.observe_completions();e.ledger.refresh(e.rows)
    assert e.registry.records[0].input.output_count==1 and not e.ledger.records
    assert e.rows[K.REQUEST_D2H,0].get('ready_version')==1
    assert not e.apply_facts(facts)


@pytest.mark.parametrize('budget',[1,8])
def test_budgeted_result_observation_uses_cached_completion(engine,monkeypatch,budget):
    e=engine
    for i in range(4):e.admit(str(i),(1,2,3),DraftGenerationConfig(12,4))
    e.step();works=tuple(e.supervisor.pairs[1].work)
    for w in works:publish_target(e,w)
    e.reader.max_entries=budget
    monkeypatch.setattr(e.scheduler,'schedule',lambda *args,**kw:())
    reads=[];read=e.resources.completions.read
    monkeypatch.setattr(e.resources.completions,'read',lambda offset:(reads.append(offset),read(offset))[1])
    for _ in range(200):
        e.step()
        if not e.ledger.records:break
    else:pytest.fail('budgeted result observation stalled')
    assert reads==[w.completion_offset for w in works]
    assert not e.ledger.request_index and not e.ledger.by_request
    assert all(r.output==(4,) for r in e.registry.records.values())
