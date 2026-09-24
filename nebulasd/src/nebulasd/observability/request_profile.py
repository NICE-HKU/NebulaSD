"""Offline full-lifecycle interval unions, including admission and prefill.

Shared batches count fully for each member's experienced latency, never as
exclusive device cost. Uninstrumented time is not labelled scheduler overhead.
"""
from collections import defaultdict


def merged(start, end, intervals):
    out = []
    for a, b in sorted((max(start, a), min(end, b)) for a, b in intervals if a < end and b > start):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([a, b])
    return out


def summarize_requests(events, *, dropped=0, gpu_dropped=0, gpu_incomplete=False):
    keyed = defaultdict(list)
    shared_schedule = []
    visible = {}
    for e in events:
        for key in {(k[0], k[1]) for k in e['keys']}:
            keyed[key].append(e)
        if e['name'] == 'engine.schedule':
            shared_schedule.append((e['start_ns'], e['end_ns']))
        if e['name'] == 'command.visible':
            visible[e['worker'], e['command_seq']] = e
    rows = []
    for (slot, epoch), spans in sorted(keyed.items()):
        def first(name, terminal=False):
            return next((e for e in sorted(spans, key=lambda e:e['start_ns'])
                         if e['name'] == name and (not terminal or e.get('terminal'))), None)
        submitted = first('client.submitted')
        admitted = first('client.admitted')
        available = first('client.output_available', True)
        observed = first('client.output_observed', True)
        if not (submitted or admitted):
            continue
        missing = [name for name, e in [('submitted',submitted), ('terminal_available',available),
                                        ('terminal_observed',observed)] if e is None]
        row = dict(slot=slot, epoch=epoch, request_id=(submitted or admitted).get('request_id'),
                   complete=not(missing or dropped or gpu_dropped or gpu_incomplete), missing=missing)
        rows.append(row)
        if submitted is None or available is None:
            continue
        start = submitted['start_ns']
        end = (observed or available)['start_ns']
        if end <= start:
            row['complete'] = False
            row['missing'].append('positive_lifecycle_duration')
            continue
        groups = defaultdict(list)
        for e in spans:
            interval = (e['start_ns'], e['end_ns'])
            name = e['name']
            if e.get('category') == 'backend_wall':
                stage = 'draft' if name == 'backend.run_batch' else 'target'
                groups['backend_'+stage].append(interval)
            if name in ('gpu.draft', 'gpu.target'):
                groups[name.replace('.', '_')].append(interval)
            if e.get('category') == 'gpu' and name not in ('gpu.draft','gpu.target'):
                groups['gpu_copy'].append(interval)
            if name == 'command.consume':
                issued = visible.get((e['worker'], e['command_seq']))
                if issued is not None:
                    groups['command_transport_wait'].append((issued['start_ns'], e['end_ns']))
        groups['backend_compute'] = groups['backend_draft'] + groups['backend_target']
        groups['gpu_compute'] = groups['gpu_draft'] + groups['gpu_target']
        groups['gpu_compute_or_copy'] = groups['gpu_compute'] + groups['gpu_copy']
        groups['shared_engine_schedule'] = shared_schedule
        gpu_present = bool(groups['gpu_compute'])
        intervals = {k: merged(start, end, v) for k, v in groups.items()}
        durations = {k:sum(b-a for a,b in v)/1e6 for k,v in intervals.items()}
        total = (end-start)/1e6
        # Exclusive GPU buckets partition the lifecycle; category totals above overlap.
        compute = durations['gpu_compute']
        combined = durations['gpu_compute_or_copy']
        row.update(start_ns=start, end_ns=end, terminal_available_ns=available['start_ns'],
                   lifecycle=available.get('lifecycle'), total_ms=total,
                   submit_to_output_available_ms=(available['start_ns']-start)/1e6,
                   output_available_to_observed_ms=(end-available['start_ns'])/1e6 if observed else None,
                   interval_union_ms=durations, intervals_ns=intervals,
                   backend_service_ratio=durations['backend_compute']/total,
                   non_backend_ms=total-durations['backend_compute'],
                   initial_backend_wait_ms=(intervals['backend_compute'][0][0]-start)/1e6 if intervals['backend_compute'] else None,
                   gpu_compute_ratio=compute/total if gpu_present else None,
                   gpu_clock_uncertainty_ns=max((e.get('clock_uncertainty_ns',0) for e in spans if e.get('category')=='gpu'),default=None),
                   exclusive_gpu_ms=dict(compute=compute, copy_without_compute=combined-compute,
                                         other=total-combined) if gpu_present else None)
    return dict(schema=1, requests=rows, complete_requests=sum(r['complete'] for r in rows),
                dropped_events=dropped, gpu_dropped_events=gpu_dropped, gpu_incomplete=gpu_incomplete,
                notes=[
                    'Lifecycle: submit entry to terminal output read/stream observation; includes prefill. Missing observation falls back to output available and marks incomplete.',
                    'Backend service includes host work and awaits; GPU forward event spans include stream gaps, not a kernel-active or SM-utilization metric.',
                    'Intervals are clipped and unioned per slot/epoch across all rounds. Shared batch duration is not divided by batch size.',
                    'GPU copy overlaps compute; exclusive buckets subtract overlaps. Other includes resource/dependency waits, control, uninstrumented work and clock error, not pure scheduling.',
                    'shared_engine_schedule is wall overlap with all Engine schedule calls, not exclusive per-request CPU cost or scheduling delay.',
                    'command_transport_wait spans visibility to consumption for all related commands and can overlap useful compute.',
                    'Dropped events invalidate completeness; surviving intervals are partial coverage. No GPU observations yields null ratios. Profiling perturbs timing.',
                ])
