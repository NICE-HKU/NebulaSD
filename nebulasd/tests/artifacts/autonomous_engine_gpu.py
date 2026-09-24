"""Real public Engine + completion/stagewise scheduler acceptance (no simulated Engine)."""
import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from time import monotonic, sleep
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'nebulasd/src'))
from canonical_reference_gpu import reference_many


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--requests',type=int,default=4)
    p.add_argument('--tokens',type=int,default=12)
    p.add_argument('--cohorts',type=int,default=1)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--slow-ms',type=int,default=0)
    p.add_argument('--eos',action='store_true')
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--draft-model', type=Path, required=True)
    p.add_argument('--target-model', type=Path, required=True)
    p.add_argument('--cost-table', type=Path, required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def manifest():
        hashes = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for base in ('nebulasd/src','swiftLLM/swiftllm','nebulasd/tests/artifacts') for f in sorted((ROOT/base).rglob('*.py'))}
        for f in sorted((ROOT/"nebulasd/csrc").rglob("*")):
            if f.suffix in (".cpp", ".h"):
                hashes[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        import os
        native = Path(os.environ["STARSD_NEXT_NATIVE_LIBRARY"])
        hashes["native_library"] = hashlib.sha256(native.read_bytes()).hexdigest()
        return hashes
    source=manifest();(a.output/'source-manifest.json').write_text(json.dumps(source,indent=2))
    from nebulasd.canonical import import_canonical
    import_canonical()
    from nebulasd import create_engine,NebulaSDConfig,GenerationConfig
    config=NebulaSDConfig(draft_worker_count=a.workers,target_worker_count=a.workers,
        scheduler_execution='completion',draft_placement='stagewise',target_placement='stagewise',cost_model='table',
        backend_cost_table=str(a.cost_table.resolve()),
        draft_model_path=str(a.draft_model.resolve()),target_model_path=str(a.target_model.resolve()),
        devices=tuple(range(2*a.workers)),gpu_memory_fraction=.9,
        request_slots=max(8,a.requests),max_batch_size=2,bank_rows=8,bank_blocks=64,host_blocks=512,
        max_batch_tokens=512,max_proposal_depth=a.depth,output_dir=str(a.output),profiling=a.profile,profile_mode='light')
    (a.output/'configuration.json').write_text(json.dumps(dict(config=asdict(config),arguments=vars(a)),default=str,indent=2))
    prompts=[tuple([1]+[100+17*i+j for j in range(13+i%4)]) for i in range(a.requests)]
    ctx=mp.get_context('spawn');expected=[]
    for start in range(0,len(prompts),4):
        q=ctx.Queue();ref=ctx.Process(target=reference_many,args=(config.target_model_path,config.devices[-1],prompts[start:start+4],a.tokens,q))
        ref.start();expected.extend(q.get(timeout=240));ref.join(30);assert ref.exitcode==0
        q.close();q.join_thread()
    # Diagnostic delay is an explicit startup option, not a Scheduler policy override.
    if a.slow_ms:
        from nebulasd.engine.autonomous_supervisor import AutonomousSupervisor
        original=AutonomousSupervisor.start
        def start(self):
            self.diagnostics={w.worker_id:dict(diagnostic_publication_delay_s=a.slow_ms/1000) for w in self.resources.specs}
            return original(self)
        AutonomousSupervisor.start=start
    client=None;reports=[]
    try:
        from types import SimpleNamespace
        works=[]
        observer=SimpleNamespace(observed=lambda *a:None,dispatched=lambda work,at:works.append(work))
        client=create_engine(config,observer=observer);e=client._engine
        for cohort in range(a.cohorts):
            ledger=e.ledger
            configs=[GenerationConfig(a.tokens if i%4!=1 else min(6,a.tokens),a.depth,
                eos_token_id=expected[i][min(4,a.tokens-1)] if a.eos and i%4==2 else None) for i in range(len(prompts))]
            handles=[client.submit(prompt,c) for prompt,c in zip(prompts,configs)]
            for h in handles:
                list(client.stream(h,timeout=180))
            outputs=[list(client.result(h)) for h in handles]
            wanted=[out[:a.tokens if i%4!=1 else min(6,a.tokens)] for i,out in enumerate(expected)]
            for i,c in enumerate(configs):
                if c.eos_token_id is not None and c.eos_token_id in wanted[i]:
                    wanted[i]=wanted[i][:wanted[i].index(c.eos_token_id)+1]
            assert outputs==wanted,(outputs,wanted)
            # Drain is the actual public cohort barrier. Client records survive it.
            client.drain(timeout=240)
            report=dict(cohort=cohort,outputs=outputs,expected=wanted,max_outstanding=ledger.max_outstanding,
                completions=[asdict(c) for c in ledger.completed],work_count=len(ledger.completed),
                worker_generations=[w.generation for w in e.resources.specs],pids={w:p.pids for w,p in e.supervisor.pairs.items()})
            reports.append(report)
            (a.output/'results.json').write_text(json.dumps(reports,indent=2))
            for h in handles: client.release(h)
        (a.output/'works.json').write_text(json.dumps([asdict(w) for w in works],indent=2))
        (a.output/'metrics.json').write_text(json.dumps(client.metrics(),indent=2))
    except BaseException:
        if client is not None:
            client._engine.write_diagnostics('failure-state.json')
            (a.output/'works.json').write_text(json.dumps([asdict(w) for w in works],indent=2))
        raise
    finally:
        if client is not None:
            client.close()
            (a.output/'exitcodes.json').write_text(json.dumps(client._engine.supervisor.exit_codes,indent=2))
    assert source==manifest(),'source changed during capture'
    print(json.dumps(dict(passed=True,cohorts=a.cohorts,requests=a.requests)))

if __name__=='__main__': main()
