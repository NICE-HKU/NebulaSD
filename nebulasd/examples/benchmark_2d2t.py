"""2D2T public Engine benchmark with resident warmups and explicit provenance."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter_ns, monotonic, sleep, thread_time_ns
ROOT=Path(__file__).absolute().parents[2]


class Trace:
    def __init__(self):self.commands=[];self.first={}
    def dispatched(self,work,now):self.commands.append((now,work))
    def observed(self,rows,now):
        from nebulasd.scheduler.views import value as v
        for row in rows:
            if row.block_kind.name=='REQUEST_DRAFT' and v(row,'round_id')==1 and v(row,'status')==2:
                self.first.setdefault((row.row,v(row,'request_epoch')),now)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'nebulasd/src')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--requests',type=int,default=512)
    p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--warmup-repeats',type=int,default=2)
    p.add_argument('--timeout',type=float,default=1200)
    p.add_argument('--draft-batch',type=int,default=64)
    p.add_argument('--target-batch',type=int,default=16)
    p.add_argument('--draft-model', type=Path, required=True)
    p.add_argument('--target-model', type=Path, required=True)
    p.add_argument('--cost-table', type=Path, required=True)
    p.add_argument('--devices', type=int, nargs=4, default=[0,1,2,3])
    a=p.parse_args()
    if min(a.requests,a.tokens,a.draft_batch,a.target_batch)<=0 or a.warmup_repeats<0:
        p.error('counts/batches must be positive and warmup repeats nonnegative')
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(a.source),str(ROOT/'nebulasd/examples')]
    from support.output_metrics import OutputMetrics
    from support.fixed_context_cost import FixedContextCostTable
    from nebulasd.observability.copy_timing import percentiles

    def manifest():
        hashes = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for base in ('nebulasd/src','swiftLLM/swiftllm','nebulasd/examples')
            for f in sorted((ROOT/base).rglob('*.py'))}
        for f in sorted((ROOT/"nebulasd/csrc").rglob("*")):
            if f.suffix in (".cpp", ".h"):
                hashes[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        hashes.update({'selected_source/'+str(f.relative_to(a.source)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(a.source.rglob('*.py'))})
        import os
        native = Path(os.environ["STARSD_NEXT_NATIVE_LIBRARY"])
        hashes["native_library"] = hashlib.sha256(native.read_bytes()).hexdigest()
        return hashes
    source=manifest();(a.output/'source-manifest.json').write_text(json.dumps(source,indent=2))
    from nebulasd.canonical import import_canonical
    paths=import_canonical()
    for path in paths.values():assert Path(path).resolve().is_relative_to((ROOT/'swiftLLM').resolve())
    from nebulasd import create_engine,NebulaSDConfig,GenerationConfig
    config=NebulaSDConfig(draft_worker_count=2,target_worker_count=2,devices=tuple(a.devices),
        scheduler_execution='completion',draft_placement='stagewise',target_placement='stagewise',
        cost_model='table',backend_cost_table=str(a.cost_table.resolve()),
        draft_model_path=str(a.draft_model.resolve()),target_model_path=str(a.target_model.resolve()),
        gpu_memory_fraction=.9,max_proposal_depth=4,draft_max_batch_size=a.draft_batch,target_max_batch_size=a.target_batch,
        draft_max_batch_tokens=4096,max_batch_tokens=4096,target_verify_max_batch_tokens=512,
        # Both banks share model row slots; reserve room for two full batches.
        draft_bank_blocks=2560,target_bank_blocks=768,bank_rows=2*max(a.draft_batch,a.target_batch),host_blocks=max(16384,48*a.requests),payload_capacity=128<<20,
        request_slots=max(256,a.requests+32*a.warmup_repeats),startup_timeout=240,output_dir=str(a.output),
        profiling=a.profile,profile_mode='light',profile_max_events=6000000)
    (a.output/'configuration.json').write_text(json.dumps(dict(config=asdict(config),arguments=vars(a),paths=paths,
        fixed_compute_context=512,initial_batch_size=8,
        warmup_policy='defer optional cohort reclamation until after timed run; same PIDs, extra warmup slots'),default=str,indent=2))
    client=None;trace=Trace();runs=[]
    try:
        client=create_engine(config,observer=trace);e=client._engine
        (a.output/'pids.json').write_text(json.dumps({k:v.pids for k,v in e.supervisor.pairs.items()},indent=2))
        # Reclamation is cold management, not a stage permission. All physical
        # work and publications still retire autonomously during warmup.
        e.recycler.progress=lambda:False
        estimator=e.scheduler._estimator
        fixed=FixedContextCostTable(estimator.table,512)
        e.scheduler._estimator=replace(estimator,table=fixed)
        e.scheduler.initial_batch_limit=8
        scheduler_samples=[]
        schedule=e.scheduler.schedule
        def timed_schedule(view, **kwargs):
            started=perf_counter_ns()
            try:return schedule(view, **kwargs)
            finally:scheduler_samples.append((perf_counter_ns()-started)/1e6)
        e.scheduler.schedule=timed_schedule
        for rep in range(-a.warmup_repeats,1):
            scheduler_samples.clear();poll_samples=[]
            e.scheduling_progress.schedule_counts={'D':0,'T':0}
            e.scheduling_progress.trigger_counts.clear()
            metrics=OutputMetrics();trace.commands.clear();trace.first.clear();fixed.reset();client.reset_metrics()
            count,tokens=(32,64) if rep<0 else (a.requests,a.tokens)
            start=perf_counter_ns();handles=[]
            for i in range(count):
                length=96+(i%4)*16
                admitted=perf_counter_ns()
                h=client.submit(tuple(range(1+i*17,length+1+i*17)),GenerationConfig(tokens,4))
                metrics.admit(h,admitted);handles.append(h)
            deadline=monotonic()+a.timeout;next_report=monotonic()+5
            while len(metrics.completed)<count:
                poll_start=perf_counter_ns();cpu_start=thread_time_ns()
                progressed=client.poll()
                poll_samples.append((poll_start,perf_counter_ns(),thread_time_ns()-cpu_start))
                for h in handles:
                    for event in client.read(h):metrics.observe(event,perf_counter_ns())
                if monotonic()>next_report:
                    state=dict(repeat=rep,elapsed_s=(perf_counter_ns()-start)/1e9,finished=len(metrics.completed),
                        tokens=sum(map(len,metrics.outputs.values())),work_records=len(e.ledger.records),
                        pending={k:len(p.pending_work) for k,p in e.supervisor.pairs.items()})
                    (a.output/'last-progress.json').write_text(json.dumps(state));print(json.dumps(state),flush=True)
                    next_report=monotonic()+5
                if monotonic()>deadline:raise TimeoutError('autonomous benchmark timed out')
                if not progressed:sleep(.0001)
            end=perf_counter_ns();outputs=[list(client.result(h)) for h in handles]
            assert all(len(out)==tokens for out in outputs)
            identities=[(client._records[h.request_id].input.slot,client._records[h.request_id].input.epoch) for h in handles]
            steady_start=max(trace.first[i] for i in identities);steady_end=min(metrics.completed.values())
            chunks=[c for c in metrics.chunks if steady_start<c['observed_ns']<=steady_end]
            steady=dict(valid=steady_end>steady_start,start_ns=steady_start,end_ns=steady_end,
                duration_s=(steady_end-steady_start)/1e9,output_tokens=sum(len(c['tokens']) for c in chunks))
            steady['output_tokens_per_second']=steady['output_tokens']/steady['duration_s'] if steady['valid'] else None
            samples=[r for r in poll_samples if steady_start<=r[0] and r[1]<=steady_end]
            last={};intervals=[]
            for c in chunks:
                if not c['tokens']:continue
                identity=c['request_id']
                if identity in last:intervals.append((c['observed_ns']-last[identity])/1e6)
                last[identity]=c['observed_ns']
            steady.update(engine_cpu_ms=sum(r[2] for r in samples)/1e6,
                engine_poll_calls=len(samples),output_chunk_interval=percentiles(intervals),
                boundary_policy='whole polls/chunk intervals inside latest first-Draft observation to first completion')
            steady['engine_cpu_fraction']=steady['engine_cpu_ms']/1000/steady['duration_s'] if steady['valid'] else None
            batch=Counter((w.operation.name,len(w.rows)) for t,w in trace.commands)
            measured=dict(repeat=rep,**metrics.summary(start,end),steady=steady,metrics=client.metrics(),
                batches=[dict(operation=op,size=size,count=n) for (op,size),n in sorted(batch.items())],
                cost_query_audit=fixed.summary(),outputs=outputs,identities=identities,
                scheduler_triggers=dict(e.scheduling_progress.trigger_counts),
                scheduler_stage_counts=dict(e.scheduling_progress.schedule_counts),
                scheduler_full=dict(count=len(scheduler_samples),total_ms=sum(scheduler_samples),
                    **percentiles(scheduler_samples)))
            measured['full_engine_poll']=dict(calls=len(poll_samples),
                cpu_ms=sum(r[2] for r in poll_samples)/1e6,
                wall_ms=sum(r[1]-r[0] for r in poll_samples)/1e6)
            # Physical retirement is separately timed, outside token throughput.
            retire_start=perf_counter_ns()
            while e.ledger.records or any(e.scheduling_progress.pending.values()):
                client.poll()
                if monotonic()>deadline:raise TimeoutError('benchmark retirement timed out')
                sleep(.0001)
            measured['retirement_ms']=(perf_counter_ns()-retire_start)/1e6
            measured['max_outstanding']=dict(e.ledger.max_outstanding)
            runs.append(measured)
            (a.output/'report.json').write_text(json.dumps(dict(status='running',runs=runs),indent=2))
            if rep==0:
                (a.output/'commands.json').write_text(json.dumps([dict(now=now,work=asdict(w)) for now,w in trace.commands]))
                (a.output/'output-chunks.json').write_text(json.dumps(metrics.chunks))
            print(json.dumps({k:v for k,v in measured.items() if k not in ('outputs','identities','batches','cost_query_audit')}),flush=True)
            for h in handles:client.release(h)
        assert source==manifest(),'source changed during capture'
        (a.output/'report.json').write_text(json.dumps(dict(status='passed',runs=runs),indent=2))
    except BaseException:
        import traceback
        (a.output/'failure.txt').write_text(traceback.format_exc())
        if client is not None:
            client._engine.write_diagnostics('failure-state.json')
            (a.output/'commands-failure.json').write_text(json.dumps([dict(now=now,work=asdict(w)) for now,w in trace.commands]))
        raise
    finally:
        if client is not None:
            client.close()
            (a.output/'exitcodes.json').write_text(json.dumps(client._engine.supervisor.exit_codes))

if __name__=='__main__':main()
