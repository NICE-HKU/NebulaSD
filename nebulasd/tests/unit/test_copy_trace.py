from types import SimpleNamespace as NS
from nebulasd.observability.copy_trace import attach_execution, attach_control
from nebulasd.observability.profiling import ProfileRecorder

def test_disabled_execution_installs_nothing():
    service=NS(metrics=NS(profile=None))
    before=vars(service).copy();attach_execution(service)
    assert vars(service)==before

def test_control_keyword_only_cancel_and_source_transitions(tmp_path):
    allocator=NS(pin=lambda e,write: True,unpin=lambda e,write: None)
    facts=NS(allocator=allocator,source_ready=lambda e: e)
    lane=NS(dirty_sink=NS(put=lambda batch:None),host_facts=facts,_launch=lambda *a:None,on_receipt=None,
            accept_prepare=lambda c:None,discard_prepare=lambda *,bank_id:bank_id,
            discard_pending_for_shutdown=lambda:True)
    rec=ProfileRecorder(tmp_path,'test',100,mode='light')
    attach_control(NS(worker=NS(copy_lane=lane),target=True),rec)
    assert lane.discard_prepare(bank_id=7)==7
    assert lane.discard_pending_for_shutdown()
    e=NS(request_slot=0,request_epoch=3,source_host_version=9)
    facts.source_ready(e);facts.source_ready(e)
    assert sum(x['name']=='copy.source' for x in rec.events)==1
    assert allocator.pin(e,write=False)
    allocator.unpin(e,write=False)
    assert [x['name'] for x in rec.events][-2:]==['copy.pin','copy.unpin']
