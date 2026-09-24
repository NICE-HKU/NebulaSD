"""A missing overlap window must not masquerade as either blocking or proof."""
from types import SimpleNamespace
from support.wp08_draft_progress import summarize_draft_progress
from support.copy_measurements import summarize_copy_receipts
from support.gpu_timeline import GPUTimeline


def evidence(start=500,round_id=3,width=10):
    copy = dict(direction='D2H',measured=True,requests=[7],rounds=[2],copy_id=10,device=1,
        timestamps=dict(ready_fact_published_ns=1200),estimated_gpu_completed_ns=1000)
    draft = dict(kind='draft',slots=[7],round_ids=[round_id],device=0,start_ns=start,end_ns=start+100)
    reports = [dict(worker_id=1,intervals=[draft],clock_alignment_uncertainty_ns={0:width,1:width},
                    copy_receipts=dict(rows=[copy]))]
    dispatch = [dict(kind='DRAFT_BATCH',dispatched_ns=400,
        command=dict(new_requests=[dict(request_slot=7,round_id=3)],cached_request_deltas=[]))]
    observed = [dict(kind='REQUEST_TARGET_COMPUTE',slot=7,observed_ns=200,fields=dict(status=2,round_id=2))]
    return reports,dispatch,observed


def test_matching_next_round_draft_proves_progress():
    result = summarize_draft_progress(*evidence())
    assert result['status']=='demonstrated' and result['gpu_started_before_d2h_end']==1
    assert result['dispatch_before_host_ready']==1
    assert result['target_observed_to_draft_dispatch']['p50_ms']==.0002


def test_short_copy_without_window_is_not_observed_not_blocked():
    result = summarize_draft_progress(*evidence(start=1500))
    assert result['status']=='not_observed' and result['gpu_started_before_d2h_end']==0
    assert result['dispatch_before_host_ready']==1  # Dispatch alone is insufficient proof of compute.


def test_wrong_round_and_clock_uncertainty_cannot_produce_witness():
    assert summarize_draft_progress(*evidence(round_id=2))['status']=='incomplete_trace'
    assert summarize_draft_progress(*evidence(start=990,width=10))['status']=='not_observed'
    reports,dispatch,observed = evidence()
    reports[0]['clock_alignment_uncertainty_ns'] = {}
    assert summarize_draft_progress(reports,dispatch,observed)['status']=='not_observed'


def test_terminal_round_without_next_draft_is_not_an_opportunity():
    reports,dispatch,observed = evidence()
    result = summarize_draft_progress(reports,[],observed)
    assert result['status']=='no_samples' and result['without_next_draft']==1


def test_next_verify_metrics_do_not_skip_same_request_rounds():
    from nebulasd.kv.transfer import CopyPlan,CopyRegion,CopyReceipt
    plan = CopyPlan('D2H',(CopyRegion(SimpleNamespace(request_slot=0,capacity_blocks=1),0,0,1),),round_ids=(2,))
    receipt = CopyReceipt(2000,5000,.2,enqueued_ns=1000)
    target = SimpleNamespace(world=SimpleNamespace(arena=SimpleNamespace(descriptor=SimpleNamespace(block_bytes=16))),
                             receipts=[(plan,receipt)])
    intervals = [dict(copy_id=id(plan),kind='D2H',device=0,slots=[0],start_ns=2000,end_ns=3000),
                 dict(kind='target_verify',device=0,slots=[0],start_ns=4000,end_ns=6000),
                 dict(kind='target_verify',device=0,slots=[1],start_ns=1_000_000_000,end_ns=1_100_000_000)]
    row = summarize_copy_receipts([target],intervals,{0:0},1)['rows'][0]
    assert row['enqueue_to_next_any_verify_gpu_estimate_ms']==.003
    assert row['enqueue_to_next_same_request_verify_gpu_estimate_ms']==.003
    assert row['enqueue_to_next_independent_verify_gpu_estimate_ms']==999.999
    assert 'enqueue_to_next_verify_gpu_estimate_ms' not in row




def test_draft_gate_is_separate_from_legacy_four_way_gate(tmp_path):
    import json
    from support.wp08_runner import merge_gpu_reports
    reports,dispatch,observed = evidence()
    reports[0]['copy_receipts']['summary'] = {}
    path = tmp_path/'worker-1'
    path.mkdir()
    (path/'report.json').write_text(json.dumps(reports[0]))
    (path/'trace.json').write_text(json.dumps(dict(traceEvents=[])))
    result = merge_gpu_reports(tmp_path,1,False,require_draft_progress=True,
                               dispatch_events=dispatch,observations=observed)
    assert result['status']=='passed' and not result['four_way_overlap_passed']
    assert merge_gpu_reports(tmp_path,1,True,require_draft_progress=True,
        dispatch_events=dispatch,observations=observed)['status']=='failed_overlap'
    reports[0]['intervals'][0]['start_ns'] = 1500
    reports[0]['intervals'][0]['end_ns'] = 1600
    (path/'report.json').write_text(json.dumps(reports[0]))
    assert merge_gpu_reports(tmp_path,1,False,require_draft_progress=True,
        dispatch_events=dispatch,observations=observed)['status']=='inconclusive_draft_progress'
