"""Join full snapshot identities; CPU publication bounds and CUDA intervals stay separate."""
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import statistics


def plain(value):
    return json.loads(json.dumps(value,default=lambda v:asdict(v) if is_dataclass(v) else str(v)))


def identity_key(identity):
    return json.dumps(identity,sort_keys=True)


def stats(values):
    values=sorted(values)
    if not values:return dict(n=0)
    def percentile(p):
        i=(len(values)-1)*p;lo=int(i);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(i-lo)
    return dict(n=len(values),mean=statistics.mean(values),p50=percentile(.5),p95=percentile(.95),p99=percentile(.99),min=values[0],max=values[-1])


def publication_delay(observed,publication):
    """Linearization is inside the measured call; a reader can beat its return."""
    assert publication['begin_ns']<=publication['end_ns']
    assert observed>=publication['begin_ns'],'observation precedes matching publication'
    return dict(lower_ms=max(0,observed-publication['end_ns'])/1e6,
                upper_ms=(observed-publication['begin_ns'])/1e6)


def observed_ready(publication,observations):
    i=publication['identity']
    candidates=[o for o in observations.get((publication['fact_kind'],i['request_slot'],i['request_epoch'],publication['publish_seq']),[])
                if o['fields'].get('status')==publication['status']]
    assert candidates,('missing Engine observation',publication)
    o=min(candidates,key=lambda x:x['observed_ns']);f=o['fields']
    expected=dict(snapshot_round_id=i['round_id'],snapshot_version=i['snapshot_version'],logical_kv_len=i['logical_kv_len'],result_code=0,**i['allocation'])
    if publication['fact_kind']=='REQUEST_DRAFT_H2D':
        expected.update(destination_worker_id=publication['worker'],destination_bank_id=publication['bank_id'],
            destination_bank_epoch=publication['bank_epoch'],prepared_batch_seq=publication['batch_seq'],
            next_owner_epoch=publication['next_owner_epoch'],gpu_ready_version=i['snapshot_version'],copied_blocks=i['valid_blocks'])
    else:
        expected.update(source_worker_id=i['worker_id'],source_worker_generation=i['worker_generation'],
            source_op_seq=i['op_seq'],owner_epoch=i['owner_epoch'],ready_version=i['snapshot_version'])
    assert all(f.get(k)==v for k,v in expected.items()),('mismatched ready identity',expected,f)
    assert f['snapshot_handle']==publication['snapshot_handle']
    return o


def analyze(directory,cohorts,events,observations,dispatch_spans):
    directory=Path(directory);epochs={c['epoch'] for c in cohorts if not c['warmup']}
    events,observations,dispatch_spans=plain(events),plain(observations),plain(dispatch_spans)
    rows=[r for p in directory.glob('latency-worker-*.json') for r in json.loads(p.read_text())['rows']]
    obs=defaultdict(list)
    for o in observations:obs[o['kind'],o['slot'],o['fields'].get('request_epoch'),o['seq']].append(o)
    pubs={};sources={};queue={};copies=[];consumes={};runs={}
    for r in rows:
        if r['kind']=='publish' and r['status']==(2 if r['fact_kind']=='REQUEST_DRAFT_D2H' else 3):
            key=(identity_key(r['identity']),r['fact_kind'],r['worker'],r['bank_id'],r['bank_epoch'],r['batch_seq'])
            assert key not in pubs,'duplicate ready publication';pubs[key]=r
        if r['kind']=='dirty_queued':
            for i in r['identities']:queue[identity_key(i)]=r['at_ns']
        if r['kind']=='copy':
            es={i['request_epoch'] for i in r['identities']};assert len(es)==1
            if not es<=epochs:continue
            r['copy_id']=len(copies);r['epoch']=next(iter(es));r['members']=len(r['identities'])
            if r['direction']=='D2H':
                labels={'initial_full' if i['snapshot_version']==1 else 'dirty_writeback' for i in r['identities']}
                r['copy_class']=next(iter(labels)) if len(labels)==1 else 'mixed_initial_dirty'
                assert all(region['host_begin_block']+region['block_count']==i['valid_blocks'] for region,i in zip(r['regions'],r['identities'],strict=True))
                for i in r['identities']:
                    k=identity_key(i);assert k not in sources,'duplicate D2H snapshot';sources[k]=r
            else:
                r['copy_class']='full_restore'
                assert all(region['host_begin_block']==0 and region['block_count']==i['valid_blocks'] for region,i in zip(r['regions'],r['identities'],strict=True))
            rc=r['receipt'];assert rc['enqueued_ns']<=rc['submitted_ns']<=rc['launch_returned_ns']<=rc['completed_ns']
            assert rc['last_pending_ns']<=rc['completed_ns'] and rc['duration_ms']>=0
            r['effective_GB_s']=r['bytes']/rc['duration_ms']/1e6 if r['bytes'] else None
            copies.append(r)
        if r['kind']=='consume_run':
            c=r['command'];consumes[c['worker_id'],c['command_seq']]=r
        if r['kind']=='backend_run':
            for k in r['keys']:runs[(r['worker'],*k)]=r
    span={(s['command']['worker_id'],s['command']['command_seq']):s for s in dispatch_spans}
    dispatched=[]
    for event in events:
        if event['kind']=='RUN_DRAFT_BATCH':
            for request in event['command']['requests']:dispatched.append((event,request))
    chains=[];terminal=[];seen_runs=set()
    for h in copies:
        if h['direction']!='H2D':continue
        for i in h['identities']:
            k=identity_key(i);assert k in sources,('H2D lacks same-identity D2H',i)
            d=sources[k]
            dp=pubs[k,'REQUEST_DRAFT_D2H',d['worker'],d['bank_id'],d['bank_epoch'],d['batch_seq']]
            hp=pubs.get((k,'REQUEST_DRAFT_H2D',h['worker'],h['bank_id'],h['bank_epoch'],h['batch_seq']))
            matches=[(e,q) for e,q in dispatched if q['request_slot']==i['request_slot'] and q['request_epoch']==i['request_epoch']
                and q['round_id']==i['round_id']+1 and q['snapshot_version']==i['snapshot_version'] and q['owner_epoch']==i['owner_epoch']+1
                and e['command']['worker_id']==h['worker'] and e['command']['expected_batch_seq']==h['batch_seq']
                and e['command']['active_bank_id']==h['bank_id'] and e['command']['active_bank_epoch']==h['bank_epoch']]
            if not matches:
                terminal.append(dict(identity=i,h2d_copy_id=h['copy_id'],gpu_ready_published=hp is not None,classification='prefetched_without_subsequent_run'))
                continue
            assert len(matches)==1 and hp is not None
            e,q=matches[0];c=e['command'];hs=observed_ready(hp,obs);ds=observed_ready(dp,obs)
            assert dp['snapshot_handle']==hp['snapshot_handle']
            run_key=(c['worker_id'],c['command_seq'],q['request_slot'],q['request_epoch'])
            assert run_key not in seen_runs,'run matched more than once'
            seen_runs.add(run_key)
            sp=span[c['worker_id'],c['command_seq']];cons=consumes[c['worker_id'],c['command_seq']]
            run=runs[(c['worker_id'],q['request_slot'],q['request_epoch'],q['round_id'],q['run_seq'])]
            target=[o for o in observations if o['kind']=='REQUEST_TARGET_COMPUTE' and o['slot']==i['request_slot']
                    and o['fields']['request_epoch']==i['request_epoch'] and o['fields']['round_id']==i['round_id'] and o['fields']['status']==2]
            assert target;ts=min(o['observed_ns'] for o in target)
            dr,hr=d['receipt'],h['receipt']
            assert sp['begin_ns']>=max(hs['observed_ns'],ts)
            assert cons['begin_ns']>=sp['begin_ns'] and run['begin_ns']>=cons['begin_ns']
            metrics={
                'dirty_queued_to_D2H_enqueued_ms':(dr['enqueued_ns']-queue[k])/1e6,
                'D2H_cuda_ms':dr['duration_ms'],
                'D2H_enqueue_to_detected_done_ms':(dr['completed_ns']-dr['enqueued_ns'])/1e6,
                'D2H_detected_done_to_host_publish_begin_ms':(dp['begin_ns']-dr['completed_ns'])/1e6,
                'D2H_completion_poll_bracket_ms':(dr['completed_ns']-dr['last_pending_ns'])/1e6,
                'H2D_cuda_ms':hr['duration_ms'],
                'H2D_enqueue_to_detected_done_ms':(hr['completed_ns']-hr['enqueued_ns'])/1e6,
                'H2D_detected_done_to_gpu_publish_begin_ms':(hp['begin_ns']-hr['completed_ns'])/1e6,
                'H2D_completion_poll_bracket_ms':(hr['completed_ns']-hr['last_pending_ns'])/1e6,
                'engine_GPU_READY_seen_to_run_dispatch_begin_ms':(sp['begin_ns']-hs['observed_ns'])/1e6,
                'engine_both_ready_seen_to_run_dispatch_begin_ms':(sp['begin_ns']-max(hs['observed_ns'],ts))/1e6,
                'engine_GPU_READY_seen_to_backend_run_ms':(run['begin_ns']-hs['observed_ns'])/1e6,
                'worker_consume_to_backend_run_ms':(run['begin_ns']-cons['begin_ns'])/1e6,
                'D2H_enqueue_to_backend_run_ms':(run['begin_ns']-dr['enqueued_ns'])/1e6,
            }
            assert all(v>=0 for v in metrics.values()),metrics
            for label,stamp,pub in [('host_ready_to_H2D_enqueue',hr['enqueued_ns'],dp),
                ('host_ready_to_engine_observed',ds['observed_ns'],dp),
                ('GPU_ready_to_engine_observed',hs['observed_ns'],hp),
                ('run_dispatch_to_worker_consume',cons['begin_ns'],sp)]:
                metrics.update({f'{label}_{side}':v for side,v in publication_delay(stamp,pub).items()})
            # Actual GPU completion is inside last-false/first-true host poll brackets.
            for label,rc,pub in [('D2H_end_to_host_ready',dr,dp),('H2D_end_to_GPU_ready',hr,hp)]:
                metrics[f'{label}_lower_ms']=max(0,pub['begin_ns']-rc['completed_ns'])/1e6
                metrics[f'{label}_upper_ms']=(pub['end_ns']-rc['last_pending_ns'])/1e6
            chains.append(dict(identity=i,d2h_copy_id=d['copy_id'],h2d_copy_id=h['copy_id'],d2h_class=d['copy_class'],
                host_publication=dp,gpu_publication=hp,engine_host_seen_ns=ds['observed_ns'],engine_gpu_seen_ns=hs['observed_ns'],
                target_seen_ns=ts,dispatch_span=sp,worker_consume=cons,backend_run=run,metrics=metrics))
    expected_runs={(e['command']['worker_id'],e['command']['command_seq'],q['request_slot'],q['request_epoch']) for e,q in dispatched if q['request_epoch'] in epochs}
    assert seen_runs==expected_runs,('unmatched run commands',expected_runs-seen_runs)
    assert chains and any(c['d2h_class']=='dirty_writeback' for c in chains),'no resumed dirty-writeback chains'
    summaries={k:stats([c['metrics'][k] for c in chains]) for k in chains[0]['metrics']}
    groups=defaultdict(list)
    for c in copies:groups[c['direction'],c['copy_class'],c['members']].append(c)
    copy_summary=[]
    for (direction,copy_class,members),cs in sorted(groups.items()):
        duration=sum(c['receipt']['duration_ms'] for c in cs);size=sum(c['bytes'] for c in cs)
        copy_summary.append(dict(direction=direction,copy_class=copy_class,members=members,tickets=len(cs),bytes=stats([c['bytes'] for c in cs]),
            cuda_ms=stats([c['receipt']['duration_ms'] for c in cs]),effective_GB_s=size/duration/1e6,
            ticket_bandwidth_GB_s=stats([c['effective_GB_s'] for c in cs if c['effective_GB_s'] is not None])))
    (directory/'copy-tickets.json').write_text(json.dumps(copies))
    (directory/'migration-chains.json').write_text(json.dumps(chains))
    (directory/'terminal-prefetch.json').write_text(json.dumps(terminal))
    return dict(status='passed',scope='instrumented real GPU diagnostic; no KV readback and no hardware-clock subtraction',
        measured_epochs=sorted(epochs),joined_request_migrations=len(chains),unconsumed_prefetches=len(terminal),
        cross_worker_migrations=sum(c['identity']['worker_id']!=c['gpu_publication']['worker'] for c in chains),
        copy_summary=copy_summary,request_weighted_latency_ms=summaries,
        caveats=['CUDA time is a ticket interval after dependency waits; CPU enqueue-to-done includes waits and polling.',
                 'Publication/read races are bounds, not negative delays silently discarded.',
                 'Per-request chains can share a batch copy; do not sum their repeated CUDA durations.',
                 'Prefetches without later run are retained separately, never counted as completed migration chains.'])
