from types import SimpleNamespace as NS
from support.steady_metrics import steady_summary


def test_excludes_prefill_ramp_and_completion_tail():
    metrics=NS(completed={0:5000000000,1:9000000000}, chunks=[
        dict(observed_ns=t,request_id=0,tokens=[1,2]) for t in (1000000000,3000000000,5000000000,6000000000)])
    result=steady_summary({(0,1):1000000000,(1,1):2000000000},[(0,1),(1,1)],metrics,[])
    assert result['duration_s']==3 and result['output_tokens']==4
    assert result['concurrent_requests']==2
    assert result['output_tokens_per_second']==4/3


def test_no_common_window_is_not_reported_as_steady():
    metrics=NS(completed={0:10,1:30})
    assert not steady_summary({(0,1):5,(1,1):20},[(0,1),(1,1)],metrics,[])['valid']
    assert not steady_summary({},[(0,1)],metrics,[])['valid']
