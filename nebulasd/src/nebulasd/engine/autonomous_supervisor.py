"""Direct Engine endpoints for the existing autonomous sibling processes."""
from pathlib import Path
from nebulasd.core.enums import WorkerRole
from nebulasd.ipc.process_doorbell import ProcessDoorbell
from nebulasd.workers.resources import PayloadDescriptor as P
from nebulasd.workers.observation import Observation


class AutonomousSupervisor:
    autonomous = True

    def __init__(self, resources, *, options, output_dir):
        self.resources, self.options = resources, options
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.bell = ProcessDoorbell.create()
        self.pairs, self.observations, self.exit_codes = {}, {}, {}
        self.closed = False

    def start(self):
        from nebulasd.workers.target.service import TargetProcesses
        from nebulasd.workers.draft.process import DraftProcesses
        from nebulasd.table.native_storage import table_descriptors
        r,c = self.resources,self.options
        try:
            for w in r.specs:
                draft = w.role == WorkerRole.DRAFT
                obs = Observation()
                self.observations[w.worker_id] = obs
                options = dict(worker_id=w.worker_id,worker_generation=w.generation,role=w.role,
                    worker_count=len(r.specs),slots=r.slots,table=table_descriptors(r.table),
                    global_registry=table_descriptors(r.registry),observation=obs.segment.descriptor,
                    event=r.events[w.worker_id].segment.descriptor,event_capacity=r.descriptor.state_ring_capacity,
                    engine_bell=self.bell,completions=r.completions.segment.descriptor,
                    tokens=tuple(P.of(a) for a in r.tokens),configs=(P.of(r.configs),),
                    proposals=tuple(P.of(a) for a in r.proposals.values()),
                    snapshots=tuple(P.of(a) for a in r.snapshots.values()),
                    host=(r.draft_host if draft else r.host).descriptor,host_arena_id=1,layout_id=r.draft_layout_id,
                    model_path=c.draft_model_path if draft else c.target_model_path,device=c.devices[w.worker_id],
                    blocks_per_bank=w.bank_blocks,capacity_rows=w.bank_rows,block_size=w.block_size,
                    max_batch_size=w.max_batch_size,max_batch_tokens=w.max_batch_tokens,profile=c.profiling)
                if draft:
                    options.update(result_proposals=P.of(r.proposals[w.worker_id]),result_snapshots=P.of(r.snapshots[w.worker_id]))
                options.update(getattr(self,'diagnostics',{}).get(w.worker_id,{}))
                self.pairs[w.worker_id] = (DraftProcesses if draft else TargetProcesses)(options)
            for pair in self.pairs.values():
                pair.wait_ready(c.startup_timeout)
        except BaseException:
            self.stop_workers()
            raise

    def retire_completed(self, ledger):
        for worker, pair in self.pairs.items():
            active = ledger.by_worker.get(worker, {})
            for seq in tuple(pair.pending_work):
                record = active.get(seq)
                if record is None or record.completion is not None:
                    pair.retire(seq)

    def check(self):
        for pair in self.pairs.values():
            pair.check()

    def stop_workers(self):
        errors = []
        for worker,pair in self.pairs.items():
            try:
                pair.close()
                self.exit_codes[worker] = dict(pair.exitcodes)
            except BaseException as error:
                errors.append(error)
        self.pairs.clear()
        for obs in self.observations.values():
            obs.close(unlink=True)
        self.observations.clear()
        if errors:
            raise errors[0]

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.stop_workers()
        finally:
            self.bell.close()
