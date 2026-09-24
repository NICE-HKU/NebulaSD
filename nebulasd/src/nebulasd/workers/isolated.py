"""Control-owned preparation/runtime; execution consumes only complete model jobs.

The existing Runtime remains the single bank/row owner, now in control. GPU
storage stays in execution and is shared once with DMA at startup.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace, fields
from queue import Empty
from types import SimpleNamespace
from typing import NamedTuple


class ComputeExtent(NamedTuple):
    capacity_blocks: int


from .channel import LocalChannel


class InlineJobs:
    """CPU preparation runs to completion in the control owner."""
    inline = True

    def submit(self, fn):
        return fn()


def attach_inputs(options, stack):
    from nebulasd.data.shared_arenas import SharedTokenArena, SharedConfigArena, SharedProposalArena
    from nebulasd.kv.arena import SharedHostKVArena
    from .resources import attach_router
    values = {}
    for name, cls in (('tokens', SharedTokenArena), ('configs', SharedConfigArena), ('proposals', SharedProposalArena)):
        values[name], attachments = attach_router(options[name], cls)
        for arena in attachments:
            stack.callback(arena.close)
    values['host'] = SharedHostKVArena.attach(options['host'])
    stack.callback(values['host'].close)
    return values


class ControlDMA:
    """Only control owns export/metadata RPC futures; H2D still bypasses it."""
    def __init__(self, options):
        from .dma_process import ProcessDMA
        self.direct_imports = bool(options.get('direct_imports'))
        self.import_slots = options.get('import_slots')
        self.connections = options['dma_connections']
        self.metadata = options['metadata_connections']
        self._pool = ThreadPoolExecutor(2, thread_name_prefix='control-dma')
        self._metadata_pool = ThreadPoolExecutor(2, thread_name_prefix='control-metadata')
        self._futures = [None, None]
        self._failure, self._closed = None, False
        self._exchange = ProcessDMA._exchange.__get__(self)
    def check(self):
        if self._closed:
            raise RuntimeError('control DMA closed')
    def import_receipt(self, bank, seq):
        from .direct_import import read
        return read(self.import_slots[bank], seq)
    def submit(self, bank, plan):
        from .dma_process import transport_plan
        if self.direct_imports and plan.direction == 'D2H':
            regions = tuple((r.gpu_begin_block, r.extent.offset_blocks + r.host_begin_block, r.block_count)
                            for r in plan.regions)
            return self._exchange(bank, ('EXPORT', regions))
        return self._exchange(bank, transport_plan(plan))
    def release_bank(self, bank):
        return self._exchange(bank, 'RELEASE')
    def write_metadata(self, layout, imports):
        def exchange():
            connection = self.metadata[layout.bank_id]
            connection.send(layout)
            return connection.recv()
        return self._metadata_pool.submit(exchange)
    def close(self):
        self._metadata_pool.shutdown(wait=True)
        self._pool.shutdown(wait=True)
        self._closed = True
        # Execution owns DMA lifetime and sends shutdown after our final job.
        for connection in (*self.connections, *self.metadata):
            connection.close()


class ControlRuntime:
    """Control-owned input compiler and Runtime; only model jobs cross IPC."""
    def __init__(self, options, stack, commands, results, wake, execution_wake):
        from .banks import Banks
        from .runtime import Runtime
        from .target.inputs import TargetInputs
        from .draft.inputs import DraftInputs
        from .observation import execution_observation
        values = attach_inputs(options, stack)
        draft = options.get('isolated_role') == 'draft'
        if draft:
            from nebulasd.data.draft_snapshot_arena import SharedDraftSnapshotArena
            from .resources import attach_router
            values['snapshots'], arenas = attach_router(options['snapshots'], SharedDraftSnapshotArena)
            for arena in arenas:
                stack.callback(arena.close)
            values.update(layout_id=options['layout_id'], host_arena_id=options['host_arena_id'])
        role = (DraftInputs if draft else TargetInputs)(**values, input_pool=InlineJobs(),
            block_size=options.get('block_size', 16), max_batch_tokens=options.get('max_batch_tokens', 4096))
        self.dma = ControlDMA(options)
        stack.callback(self.dma.close)
        self.role = role
        self.runtime = Runtime(Banks(options['blocks_per_bank'], options['capacity_rows']), self.dma, role,
                               profile=options.get('profile', False), execute=self.execute,
                               write_metadata=self.dma.write_metadata)
        from .handoff_trace import HandoffTrace
        self.trace = HandoffTrace(options, stack, "control")
        self.runtime.handoff_trace = self.trace
        self.runtime.wakeup = wake
        self.options, self.commands, self.results = options, commands, results
        self.execution_wake = execution_wake
        self.observation = execution_observation(options, stack) if 'global_registry' in options else None
        if self.observation is None and draft and 'observation' in options:
            from .draft.observation import Observation
            self.observation = Observation(options['observation'])
            stack.callback(self.observation.close)
        self.sent, self.pending = {}, None
        self.on_result = None
        self.runtime.result_callback = self._notify_result
        self.stopping = self.closed = False
        self.result_kind, self.import_kind = ('DRAFT_RESULT', 'DRAFT_IMPORTED') if draft else ('RESULT', 'IMPORTED')

    def execute(self, plan):
        if self.pending is not None:
            raise RuntimeError('concurrent model jobs')
        # Model code only needs operation/seq; do not deserialize a full WORK in execution.
        spec = SimpleNamespace(work_seq=plan.spec.work_seq, operation=plan.spec.operation)
        rows = tuple(r._replace(extent=ComputeExtent(r.extent.capacity_blocks)) for r in plan.rows)
        self.commands.put_nowait(('COMPUTE', replace(plan, spec=spec, rows=rows)))
        self.pending = Future()
        self.execution_wake.set()
        return self.pending

    def accept(self, work):
        if not self.runtime.accept(work):
            raise RuntimeError('control exceeded reserved WORK credit')

    def receive_inputs(self, seq, inputs):
        self.runtime.receive_inputs(seq, inputs)

    def retire(self, seq):
        self.role.retire(seq)
        self.runtime.publication_retired(seq)
        self.sent.pop(seq, None)

    def stop(self, shutdown):
        self.stopping = True
        self.runtime.shutdown() if shutdown else self.runtime.drain()

    def step(self):
        changed = False
        try:
            kind, result = self.results.get_nowait()
        except Empty:
            pass
        else:
            if kind != 'COMPUTED' or self.pending is None:
                raise RuntimeError('unexpected compute result')
            self.trace.mark("RESULT_AVAILABLE", self.runtime.compute.spec.work_seq)
            if result.export_plan == 'CONTROL_EXPORT':
                from nebulasd.kv.transfer import CopyPlan, CopyRegion, HostCompletedFence
                inputs = self.runtime.compute.plan.rows
                if tuple(r.index for r in inputs) != result.executed_rows or tuple(r.index for r in result.rows) != result.executed_rows:
                    raise RuntimeError('execution returned unexpected members')
                regions = tuple(CopyRegion(row.extent, row.gpu_begin + out.dirty_begin,
                    out.dirty_begin, out.dirty_blocks) for row, out in zip(inputs, result.rows))
                result = replace(result, export_plan=CopyPlan('D2H', regions, (HostCompletedFence(),)))
            future, self.pending = self.pending, None
            future.set_result(result)
            changed = True
        changed |= self.runtime.step()
        if self.observation is not None and 'global_registry' not in self.options:
            self.observation.write(self.runtime.banks)
        elif self.observation is not None:
            with self.options['compute_clock'].get_lock():
                clock = tuple(self.options['compute_clock'][:])
            self.observation.write(self.runtime, clock)
        if self.stopping and self.runtime.quiescent and not self.closed:
            self.commands.put_nowait(('SHUTDOWN', None))
            self.execution_wake.set()
            self.closed = True
            changed = True
        return changed

    def _notify_result(self, state):
        if self.on_result is not None:
            seq = state.spec.work_seq
            self.sent.setdefault(seq, set()).add('RESULT')
            r = state.result
            self.on_result((self.result_kind, seq, dict(rows=[row._asdict() for row in r.rows],
                compute_start_ns=r.compute_start_ns, compute_end_ns=r.compute_end_ns)))

    def get_nowait(self):
        for seq, state in self.runtime.records.items():
            flags = self.sent.setdefault(seq, set())
            if state.imported and state.import_inputs is not None and state.import_inputs.rows and 'IMPORTED' not in flags:
                flags.add('IMPORTED')
                return self.import_kind, seq, dict(rows=[{f.name: getattr(r, f.name) for f in fields(r)} for r in state.import_inputs.rows], submitted_ns=state.h2d_receipt.submitted_ns)
            if state.result is not None and 'RESULT' not in flags:
                flags.add('RESULT')
                r = state.result
                return self.result_kind, seq, dict(rows=[row._asdict() for row in r.rows], compute_start_ns=r.compute_start_ns, compute_end_ns=r.compute_end_ns)
            if state.physical_done and 'PHYSICAL' not in flags:
                flags.add('PHYSICAL')
                facts = [(k, t) for k, t, _ in state.facts] if self.options.get('profile') else []
                if self.options.get('profile'):
                    for direction, receipt in (('H2D', state.h2d_receipt), ('D2H', state.d2h_receipt)):
                        if receipt is not None:
                            facts.extend(((direction+'_LAUNCH', receipt.submitted_ns),
                                          (direction+'_OBSERVED', receipt.completed_ns),
                                          (direction+'_CPU_NS', receipt.cpu_ns)))
                return 'PHYSICAL', seq, dict(d2h_submitted_ns=state.d2h_receipt.submitted_ns if state.d2h_receipt else 0,
                    outcomes=[int(o) for o in state.outcomes], observed_ns=state.physical_done_ns, facts=facts)
        raise Empty


def run_compute(options, commands, results, wake, ready, result_wake, backend_factory):
    """No runtime/inputs/metadata/reply threads run alongside the model."""
    from .target.backend import TargetBackend
    from .dma_process import ProcessDMA
    with ExitStack() as stack:
        from .handoff_trace import HandoffTrace
        trace = HandoffTrace(options, stack, "execution")
        commands, results = LocalChannel(commands), LocalChannel(results)
        stack.callback(commands.close)
        stack.callback(results.close)
        values = attach_inputs(options, stack)
        kwargs = dict(values, model_path=options['model_path'], device=options['device'],
            blocks_per_bank=options['blocks_per_bank'], capacity_rows=options['capacity_rows'],
            block_size=options.get('block_size', 16), max_batch_tokens=options.get('max_batch_tokens', 4096), job_threads=False)
        role = TargetBackend(**kwargs) if backend_factory is None else backend_factory(options, stack, **kwargs)
        stack.callback(role.close)
        role.clock_slot, role.clock_wakeup = options['compute_clock'], result_wake
        dma = ProcessDMA(arena=values['host'], k_cache=role.model.k_cache, v_cache=role.model.v_cache,
                         block_table=role.block_table, options=options, wake=wake)
        stack.callback(dma.close)
        ready.set()
        while True:
            wake.clear()
            dma.check()
            try:
                kind, plan = commands.get_nowait()
            except Empty:
                wake.wait()
                continue
            if kind == 'SHUTDOWN':
                break
            if kind != 'COMPUTE':
                raise ValueError('execution accepts only prepared compute jobs')
            trace.mark("EXECUTION_RECEIVED", plan.spec.work_seq)
            result = role._execute(plan)
            # Actual HostKV identities stay in control; outputs contain all
            # dynamic export ranges, already fenced by _execute.
            result = replace(result, export_plan='CONTROL_EXPORT' if result.export_plan is not None else None)
            results.put_nowait(('COMPUTED', result))
            result_wake.set()
