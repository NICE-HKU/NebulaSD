"""Offline evidence of next-round Draft progress before source D2H completes.

No overlap is not proof of blocking: short copies, busy Draft workers and clock
uncertainty can all hide the window. The strict optional gate requires a positive
witness; the controlled CPU test separately checks the dependency invariant.
"""
from nebulasd.core.enums import TargetStatus
from nebulasd.observability.copy_timing import percentiles


def summarize_draft_progress(reports, dispatch_events, observations):
    intervals = [r for report in reports for r in report['intervals'] if r['kind']=='draft']
    clocks = {int(device):width for report in reports
              for device,width in report['clock_alignment_uncertainty_ns'].items()}
    dispatch = {}
    for event in dispatch_events:
        if event['kind']=='DRAFT_BATCH':
            command = event['command']
            for item in command['new_requests']+command['cached_request_deltas']:
                dispatch[item['request_slot'],item['round_id']] = event['dispatched_ns']
    observed = {}
    for row in observations:
        fields = row['fields']
        if row['kind']=='REQUEST_TARGET_COMPUTE' and fields['status']==TargetStatus.READY_DRAFT:
            observed.setdefault((row['slot'],fields['round_id']),row['observed_ns'])
    forwards = {}
    for interval in intervals:
        for key in zip(interval['slots'],interval['round_ids'],strict=True):
            if key not in forwards or interval['start_ns'] < forwards[key]['start_ns']:
                forwards[key] = interval
    rows, without_next_draft = [], 0
    for report in reports:
        for copy in report['copy_receipts']['rows']:
            if copy['direction']!='D2H' or not copy['measured']:
                continue
            for slot, source_round in zip(copy['requests'],copy['rounds'],strict=True):
                key = slot,source_round+1
                issued, forward = dispatch.get(key),forwards.get(key)
                if issued is None:
                    # Terminal rounds need no subsequent Draft. Do not count
                    # them as progress opportunities or fabricate a latency.
                    without_next_draft += 1
                    continue
                stamps = copy['timestamps']
                ready = stamps.get('ready_fact_published_ns',0)
                observed_ns = observed.get((slot,source_round))
                error = None
                if forward and copy.get('device') in clocks and forward['device'] in clocks:
                    error = clocks[copy['device']]+clocks[forward['device']]
                gpu_end = copy.get('estimated_gpu_completed_ns')
                margin = (gpu_end-forward['start_ns']-error
                          if forward and gpu_end is not None and error is not None else None)
                rows.append(dict(slot=slot,source_round=source_round,draft_round=source_round+1,
                    target_worker_id=report['worker_id'],copy_id=copy['copy_id'],
                    draft_dispatched_ns=issued,d2h_ready_ns=ready,
                    draft_gpu_start_ns=forward['start_ns'] if forward else None,
                    d2h_gpu_end_estimate_ns=gpu_end,clock_uncertainty_ns=error,
                    dispatch_before_host_ready=bool(ready and issued<ready),
                    gpu_started_before_d2h_end=margin is not None and margin>0,
                    conservative_start_margin_ns=margin,
                    target_observed_to_draft_dispatch_ms=(issued-observed_ns)/1e6 if observed_ns is not None else None,
                    dispatch_to_draft_gpu_estimate_ms=(forward['start_ns']-issued)/1e6 if forward else None))
    witnesses = sum(r['gpu_started_before_d2h_end'] for r in rows)
    latencies = ('target_observed_to_draft_dispatch_ms','dispatch_to_draft_gpu_estimate_ms')
    missing = sum(r['draft_gpu_start_ns'] is None for r in rows)
    return dict(status='incomplete_trace' if missing else 'demonstrated' if witnesses else 'not_observed' if rows else 'no_samples',
        sample_unit='request/source-round with a measured D2H and an issued next Draft',
        samples=len(rows),without_next_draft=without_next_draft,
        dispatch_before_host_ready=sum(r['dispatch_before_host_ready'] for r in rows),
        gpu_started_before_d2h_end=witnesses,
        missing_draft_gpu_samples=missing,
        **{name.removesuffix('_ms'):percentiles([r[name] for r in rows if r[name] is not None]) for name in latencies},
        rows=rows)
