"""Peer readiness must not depend on Engine classification/output progress."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from nebulasd.core.enums import StateChangeBlockKind as K, Lifecycle
from nebulasd.table.storage import RequestSchedulingTable
from nebulasd.table.prepared import PreparedRow
from nebulasd.workers.dependencies import Dependencies
from nebulasd.workers.input_facts import InputFact, encode_inputs, decode_inputs
from nebulasd.workers.work import Selector, TableDependency
from test_autonomous_work import work


@pytest.mark.parametrize('logical,last,finished', [(5,7,False),(19,7,True),(5,99,True)])
def test_peer_decision_without_engine_publication(logical,last,finished):
    table = RequestSchedulingTable(1)
    w = work()
    dep = TableDependency(K.REQUEST_TARGET_COMPUTE,0,1,0,Selector.TARGET_DECISION)
    # Missing proposal must not keep a naturally finished request waiting.
    proposal = TableDependency(K.REQUEST_DRAFT,0,1,1,Selector.PROPOSAL)
    w = replace(w,rows=(replace(w.rows[0],classified=dep,predecessor=proposal),))
    configs = SimpleNamespace(read_config=lambda _: SimpleNamespace(all_stop_token_ids=(99,)))
    watcher = Dependencies(table,configs=configs)
    watcher.register(w)
    assert watcher.poll()==[]
    PreparedRow(table.partition(K.REQUEST_TARGET_COMPUTE),0,
        dict(request_epoch=1,round_id=0,status=2,result_code=0,
             logical_kv_len=logical,last_committed_token=last,output_count=logical+1-w.rows[0].prompt_count,
             output_finished=int(finished)),()).publish(())
    events=watcher.poll()
    assert len(events)==1
    snapshot=events[0].snapshot
    assert snapshot['lifecycle']==(Lifecycle.FINISHED if finished else Lifecycle.ACTIVE)
    assert ((w.work_seq,0,'predecessor') in watcher.pending)==(not finished)
    fact=InputFact.capture(w,events[0])
    payload=dict(worker_id=w.worker_id,worker_generation=w.worker_generation,events=(fact,))
    assert decode_inputs(encode_inputs(payload))['events'][0].snapshot==snapshot
    # No Engine row was published at any point.
    assert table.partition(K.REQUEST_ENGINE).read_publish_seq(0)==(1<<64)-1


def test_target_anchor_does_not_read_engine_output():
    from concurrent.futures import Future
    from nebulasd.workers.target.inputs import TargetInputs
    from nebulasd.workers.work import WorkKind
    from nebulasd.core.enums import ProposalKind

    class Inline:
        def submit(self, fn):
            f = Future()
            f.set_result(fn())
            return f

    def engine_output_unavailable(_):
        raise AssertionError('model input read Engine output arena')

    backend = TargetInputs(
        host=SimpleNamespace(make_extent=lambda **kw: None),
        configs=SimpleNamespace(read_config=lambda _: SimpleNamespace(proposal_depth=4,all_stop_token_ids=())),
        tokens=SimpleNamespace(read_tokens=engine_output_unavailable),
        proposals=SimpleNamespace(read_proposal=lambda _: SimpleNamespace(draft_token_ids=(8,9),kind=ProposalKind.LINEAR)),
        input_pool=Inline(), max_batch_tokens=512)
    base = work()
    decision = TableDependency(K.REQUEST_TARGET_COMPUTE,0,1,0,Selector.TARGET_DECISION)
    w = replace(base,operation=WorkKind.TARGET_VERIFY,rows=(replace(base.rows[0],classified=decision,
        source=TableDependency(K.REQUEST_D2H,0,1,1,Selector.TARGET_HOST),
        predecessor=TableDependency(K.REQUEST_DRAFT,0,1,1,Selector.PROPOSAL)),))
    captured = ({'source':{'logical_kv_len':5,'ready_version':1},
                 'predecessor':{'proposal_handle':None},
                 'classified':{'logical_kv_len':5,'last_committed_token':7}},)
    plan = backend.compile_compute(w,captured,SimpleNamespace(rows=(0,),offsets=(0,)),(0,)).result()
    assert plan.rows[0].anchor == 7
    assert plan.rows[0].remaining == 14
