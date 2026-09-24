"""Offline per-request/epoch/round attribution; overlapping stages are not summed."""
from collections import defaultdict
import json
from pathlib import Path
from .copy_timing import covered_ns, percentiles


def summarize(directory):
    reports = [json.loads(p.read_text()) for p in sorted(Path(directory).glob('*.json'))
               if p.name != 'summary.json']
    if any(r.get('mode') == 'draft' for r in reports):
        raise ValueError('Draft-only trace cannot provide full lifecycle attribution; use the Draft comparison analyzer')
    events = sorted((e for r in reports for e in r['events']), key=lambda e:e['start_ns'])
    dropped = sum(r['dropped_events'] for r in reports)
    gpu_dropped = sum(e.get('gpu_dropped_events',0) for e in events)
    gpu_incomplete = any(e.get('gpu_incomplete',False) for e in events)
    publications, visible, consumed = {}, {}, {}
    completed = defaultdict(dict)
    for e in events:
        if e['name'] == 'fact.publish' and e['keys']:
            epoch = e['keys'][0][1]
            publications[e['kind'],e['row'],epoch,e['publish_seq']] = e
            if e['kind'] == 'REQUEST_TARGET_COMPUTE' and e['status'] == 2:
                slot,epoch,round_id = e['keys'][0]
                completed[slot,epoch][round_id] = e
        if e['name'] == 'command.visible':
            visible[e['worker'],e['command_seq']] = e
        if e['name'] == 'command.consume':
            consumed[e['worker'],e['command_seq']] = e
    # Derive edges with host timestamps, not polling configuration or GPU guesses.
    edges = []
    for e in events:
        if e['name'] == 'fact.observe' and e['keys']:
            pub = publications.get((e['kind'],e['row'],e['keys'][0][1],e['publish_seq']))
            if pub is not None:
                # Publication wrapper ends after the store. A very fast reader
                # can observe before wrapper exit: keep the measurement bound.
                edges.append(dict(name='wait.publish_to_observe.'+e['kind'],
                    start_ns=pub['end_ns'], end_ns=e['start_ns'], keys=e['keys'],
                    uncertainty_ns=pub['end_ns']-pub['start_ns']))
    for key, command in visible.items():
        peer = consumed.get(key)
        if peer is not None:
            edges.append(dict(name='wait.command_visible_to_consume',start_ns=command['start_ns'],
                              end_ns=peer['end_ns'],keys=command['keys']))
    rounds = []
    for (slot,epoch), results in sorted(completed.items()):
        for round_id, end_event in sorted(results.items()):
            if round_id == 0:
                continue
            previous = results.get(round_id-1)
            if previous is None:
                rounds.append(dict(slot=slot,epoch=epoch,round=round_id,complete=False,
                                   missing=['previous_target_result']))
                continue
            start,end = previous['end_ns'],end_event['end_ns']
            def relevant(e):
                return (any(k[0]==slot and k[1]==epoch and k[2] in (round_id-1,round_id) for k in e['keys'])
                        or (not e['keys'] and e.get('owner')=='engine'))
            spans = [e for e in events if e['end_ns']>start and e['start_ns']<end and relevant(e)]
            stages = defaultdict(list)
            for e in spans:
                stages[e['name']].append(e)
            durations = {name: dict(samples=len(rows), wall_union_ms=covered_ns(start,end,
                [(e['start_ns'],e['end_ns']) for e in rows])/1e6)
                for name,rows in stages.items()}
            cpu_spans = [e for e in spans if e.get('category')=='control' and e.get('cpu_ns') is not None
                         and e['start_ns']>=start and e['end_ns']<=end]
            cpu = 0
            for e in cpu_spans:
                # Sum only outermost synchronous control spans. Their thread
                # CPU counters already include nested control spans.
                if not any(p is not e and p['owner']==e['owner'] and p['thread']==e['thread']
                           and p['start_ns']<=e['start_ns'] and p['end_ns']>=e['end_ns']
                           for p in cpu_spans):
                    cpu += e['cpu_ns']
            gpu = [e for e in spans if e.get('category')=='gpu']
            gpu_union = covered_ns(start,end,[(e['start_ns'],e['end_ns']) for e in gpu])
            copies = [dict(direction=e['direction'], keys=e['keys'], timing=e['timing'])
                      for e in spans if e['name']=='copy.receipt']
            waits = [dict(name=e['name'],duration_ms=(e['end_ns']-e['start_ns'])/1e6,
                          uncertainty_ns=e.get('uncertainty_ns',0)) for e in edges
                     if start<=e['end_ns']<=end and relevant(e)]
            # Named milestones retain their raw timestamps. Shared batch events
            # may occur for many requests; never multiply them into GPU totals.
            def first(name, *, kind=None, current=True, status=None):
                matches = [e for e in events if e['name']==name
                    and (kind is None or e.get('kind')==kind)
                    and (status is None or e.get('status')==status)
                    and [slot,epoch,round_id if current else round_id-1] in e['keys']
                    and start-1000000<=e['end_ns']<=end]
                return min((e['end_ns'] for e in matches),default=None)
            milestones = dict(previous_target_ready_ns=start,
                target_observed_ns=first('fact.observe',kind='REQUEST_TARGET_COMPUTE',current=False),
                draft_dispatch_ns=(first('command.visible',kind='RUN_DRAFT_BATCH')
                                   or first('command.visible',kind='DRAFT_BATCH')),
                draft_ready_ns=first('fact.publish',kind='REQUEST_DRAFT',status=2),
                h2d_ready_ns=first('fact.publish',kind='REQUEST_H2D',status=3),
                target_run_dispatch_ns=first('command.visible',kind='RUN_TARGET_BATCH'),
                current_target_ready_ns=end)
            for label,left,right in (
                ('wait.target_observed_to_draft_dispatch','target_observed_ns','draft_dispatch_ns'),
                ('wait.draft_ready_to_run_dispatch','draft_ready_ns','target_run_dispatch_ns'),
                ('wait.h2d_ready_to_run_dispatch','h2d_ready_ns','target_run_dispatch_ns')):
                if milestones[left] is not None and milestones[right] is not None:
                    waits.append(dict(name=label,duration_ms=(milestones[right]-milestones[left])/1e6))
            missing = [name for name,value in milestones.items() if value is None]
            rounds.append(dict(slot=slot,epoch=epoch,round=round_id,complete=not(dropped or gpu_dropped or gpu_incomplete or missing),missing=missing,
                start_ns=start,end_ns=end,round_wall_ms=(end-start)/1e6,stages=durations,
                measured_control_cpu_ms=cpu/1e6, gpu_covered_union_ms=gpu_union/1e6,
                gpu_clock_uncertainty_ns=max((e.get('clock_uncertainty_ns',0) for e in gpu),default=None),
                remainder_not_attributed_to_gpu_ms=(end-start-gpu_union)/1e6 if gpu else None,
                waits=waits,copies=copies,milestones=milestones))
    stage_samples, wait_samples, copy_samples = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rounds:
        for name, stage in row.get('stages',{}).items():
            stage_samples[name].append(stage['wall_union_ms'])
        for edge in row.get('waits',[]):
            wait_samples[edge['name']].append(edge['duration_ms'])
    for e in events:
        if e['name']!='copy.receipt':
            continue
        t = e['timing']
        copy_samples[e['direction']+'.dma'].append(t['duration_ms'])
        for name,left,right in (
            ('executor_queue','enqueued_ns','submitted_ns'),
            ('submit_to_last_launch_return','submitted_ns','launch_returned_ns'),
            ('launch_call','submitted_ns','launch_returned_ns'),
            ('launch_return_to_observed','launch_returned_ns','completed_ns'),
            ('observed_to_owner','completed_ns','ready_publish_started_ns'),
            ('owner_to_ready','ready_publish_started_ns','ready_fact_published_ns'),
            ('ready_to_retired','ready_fact_published_ns','retired_ns'),
            ('enqueue_to_retired','enqueued_ns','retired_ns')):
            if t.get(left) and t.get(right):
                copy_samples[e['direction']+'.'+name].append((t[right]-t[left])/1e6)
    from .request_profile import summarize_requests
    requests = summarize_requests(events, dropped=dropped, gpu_dropped=gpu_dropped, gpu_incomplete=gpu_incomplete)
    return dict(schema=1,request_lifecycles=requests,dropped_events=dropped,gpu_dropped_events=gpu_dropped,gpu_incomplete=gpu_incomplete,
        complete_rounds=sum(r['complete'] for r in rounds),
        gpu_intervals=sum(e.get('category')=='gpu' for e in events),
        stage_wall_per_round={k:percentiles(v) for k,v in stage_samples.items()},
        wait_edges={k:percentiles(v) for k,v in wait_samples.items()},
        copy_stages={k:percentiles(v) for k,v in copy_samples.items()},
        notes=[
            'Round = previous Target ready publication to current Target ready publication; prefill excluded.',
            'Stage wall unions overlap. Do not sum stages, rounds or per-request shared batch costs.',
            'Control CPU covers fully contained synchronous instrumented spans only; not total process CPU.',
            'Unkeyed Engine work is shared across requests and attributed by time window, not exclusive ownership.',
            'Backend wall spans include compute/await; asynchronous thread CPU is intentionally omitted.',
            'GPU remainder includes scheduling, dependency waits, uninstrumented work and alignment error; NOT pure control overhead.',
            'Small negative publication/dispatch edges can reflect wrapper endpoint uncertainty; raw timestamps retained.',
            'Dropped events or missing endpoints invalidate complete attribution; profiling perturbs performance.'],
        round_latency=percentiles([r['round_wall_ms'] for r in rounds if 'round_wall_ms' in r]), rounds=rounds)
