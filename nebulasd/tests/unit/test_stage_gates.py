import pytest
from nebulasd.observability.stage_gates import predicates
from nebulasd.core.enums import StateChangeBlockKind as K
from test_wp08_scheduler import runnable,patch
from test_draft_placement import prepared_view

@pytest.mark.parametrize('make',[runnable,prepared_view])
def test_ready_batch_and_exact_fences(make):
    view=make();p=next(iter(view.prepared.values()));w=next(w for w in view.workers if w.worker_id==p.worker_id)
    gates=predicates(view,w,p)
    assert all(check(view) for check in gates.values())
    draft=p.kind.name=='PREPARE_DRAFT_BANK'
    patch(view,K.REQUEST_DRAFT_H2D if draft else K.REQUEST_H2D,p.requests[0].request_slot,destination_bank_epoch=999)
    assert not gates['KV'](view)
    assert gates['free'](view)

@pytest.mark.parametrize('make',[runnable,prepared_view])
def test_ready_input_requires_epoch_and_classification(make):
    view=make();p=next(iter(view.prepared.values()));w=next(w for w in view.workers if w.worker_id==p.worker_id)
    gates=predicates(view,w,p)
    patch(view,K.REQUEST_ENGINE,p.requests[0].request_slot,request_epoch=999)
    assert not gates['in'](view)

def test_trace_retains_provenance_and_does_not_change_decisions(tmp_path):
    from types import SimpleNamespace
    from nebulasd.observability.stage_gates import attach
    from nebulasd.observability.profiling import ProfileRecorder
    view=prepared_view();commands=[]
    scheduler=SimpleNamespace(schedule=lambda v:commands)
    rec=ProfileRecorder(tmp_path,'engine',100,mode='light')
    attach(scheduler,rec)
    assert scheduler.schedule(view) is commands
    assert len(rec.events)==5 and all(e['ready'] for e in rec.events)
    kv=next(e for e in rec.events if e['gate']=='KV')
    assert any('destination_bank_epoch' in d['fields'] for d in kv['evidence'])
    scheduler.schedule(view)
    assert len(rec.events)==5
    p=next(iter(view.prepared.values()))
    patch(view,K.REQUEST_DRAFT_H2D,p.requests[0].request_slot,destination_bank_epoch=999)
    # Fixture patch() keeps seq=1; real changed facts increment publish_seq.
    from dataclasses import replace
    key=(K.REQUEST_DRAFT_H2D,p.requests[0].request_slot)
    view.rows[key]=replace(view.rows[key],publish_seq=2)
    scheduler.schedule(view)
    assert rec.events[-1]['gate']=='KV' and not rec.events[-1]['ready']
