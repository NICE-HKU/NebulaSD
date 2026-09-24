"""Single physical owner, in control for the isolated process path.

Jobs return facts; publication never releases banks. CPU preparation may return
an already-completed Future. GPU jobs finish before their facts are consumed."""
from dataclasses import dataclass, field, replace
from queue import SimpleQueue, Empty
from threading import Event
from time import perf_counter_ns
from nebulasd.core.enums import Lifecycle
from .banks import Phase
from .work import MAX_OUTSTANDING, Outcome


@dataclass(slots=True)
class State:
    spec: object
    captured: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    jobs: dict = field(default_factory=dict)
    layout: object = None
    import_inputs: object = None
    plan: object = None
    prepared_rows: dict = field(default_factory=dict)
    prepared_template: object = None
    metadata: object = None
    imported: bool = False
    result: object = None
    physical_done: bool = False
    physical_done_ns: int = 0
    h2d_receipt: object = None
    d2h_receipt: object = None
    facts: list | None = None


class Runtime:
    def __init__(self, banks, dma, role, *, profile=False, execute=None, write_metadata=None):
        self.banks, self.dma, self.role = banks, dma, role
        self.execute = execute if execute is not None else role.execute
        self.write_metadata = write_metadata if write_metadata is not None else role.write_metadata
        self.profile = profile
        self.direct_imports = bool(getattr(dma, "direct_imports", False))
        self.records = {}
        self.compute = None
        self.done = SimpleQueue()
        self.wakeup = Event()
        self.shutting_down = False
        self.draining = False
        self.turn = 0
        self.failed = None
        self.handoff_trace = None
        self.result_callback = None

    def accept(self, work):
        if self.failed is not None:
            raise RuntimeError('worker failed; restart requires a new cohort') from self.failed
        if len(work.rows) > self.banks.capacity_rows:
            raise ValueError('WORK exceeds total physical row capacity')
        if self.draining or len(self.records) >= MAX_OUTSTANDING:
            return False
        if work.work_seq in self.records:
            raise ValueError('duplicate WORK')
        bank = self.banks.banks[work.bank_id]
        if bank.current is not None and bank.pending is not None:
            return False
        if sum(r.capacity_blocks for r in work.rows) > self.banks.blocks_per_bank:
            raise ValueError('WORK exceeds Bank capacity')
        state = State(work, [{} for _ in work.rows], [None for _ in work.rows])
        if self.profile:
            state.facts = []
        self.records[work.work_seq] = state
        if bank.current is None:
            bank.current = state
        else:
            bank.pending = state
        self._fact(state, 'WORK_ACCEPTED')
        self.wakeup.set()
        return True

    def _fact(self, state, kind, value=None):
        # At most one occurrence of each physical fact plus 3 dependencies/row.
        if self.profile:
            state.facts.append((kind, perf_counter_ns(), None))

    def _submit(self, state, kind, future):
        if kind in ('IMPORT_INPUT', 'COMPILE') and getattr(getattr(self.role, 'input_pool', None), 'inline', False):
            self._apply(state, kind, future)
            return
        state.jobs[kind] = future
        def complete(done):
            # Completion callback never waits or owns any physical state.
            self.done.put((state, kind, done))
            self.wakeup.set()
        future.add_done_callback(complete)

    def _release(self, state):
        if state.jobs:
            raise RuntimeError('physical retirement while jobs reference WORK')
        bank = self.banks.banks[state.spec.bank_id]
        if state.layout is not None:
            self.banks.release(state.layout)
        state.physical_done = True
        self._fact(state, 'BANK_FREE', bank.observation_seq)
        state.physical_done_ns = perf_counter_ns()
        self._fact(state, 'WORK_PHYSICALLY_DONE')
        bank.current, bank.pending = bank.pending, None

    def _finish(self, state):
        if self.direct_imports:
            self._submit(state, 'D2H', self.dma.release_bank(state.spec.bank_id))
        else:
            self._release(state)

    def _handle(self, state, kind, future):
        del state.jobs[kind]
        value = future.result()  # Called only for a delivered done callback.
        self._apply(state, kind, value)

    def _apply(self, state, kind, value):
        self._fact(state, kind + '_DONE', value)
        if kind == 'IMPORT_INPUT':
            state.import_inputs = value
        elif kind == 'COMPILE':
            state.plan = value
        elif kind == 'METADATA':
            state.metadata = value
        elif kind == 'H2D':
            state.h2d_receipt = value
            state.imported = True
            self.banks.banks[state.spec.bank_id].phase = Phase.READY
        elif kind == 'COMPUTE':
            self.compute = None
            state.result = value
            # execute result must already own host tokens and the compact DMA plan.
            expected = tuple(i for i, outcome in enumerate(state.outcomes) if outcome is None)
            if tuple(value.executed_rows) != expected:
                raise RuntimeError('backend result does not cover the live WORK members')
            for i in value.executed_rows:
                state.outcomes[i] = Outcome.EXECUTED
            if value.export_plan is not None:
                self._fact(state, 'D2H_SUBMITTED')
                self._submit(state, 'D2H', self.dma.submit(state.spec.bank_id, value.export_plan))
                self.banks.banks[state.spec.bank_id].phase = Phase.EXPORTING
            else:
                self._finish(state)
            self._fact(state, 'RESULT_READY', value)
            if self.result_callback is not None:
                self.result_callback(state)
        elif kind == 'D2H':
            state.d2h_receipt = value
            self._release(state)

    def receive_inputs(self, seq, data):
        """Consume facts for an accepted WORK. No shared-table access or fallback.

        Duplicate/unknown/retired identities fail explicitly. In-flight facts for
        terminal members are harmless and ignored until logical retirement.
        """
        state = self.records.get(seq)
        if state is None or (data['worker_id'], data['worker_generation']) != (
                state.spec.worker_id, state.spec.worker_generation):
            raise ValueError('input facts for unknown WORK/generation or retired record')
        for event in data['events']:
            if not 0 <= event.index < len(state.spec.rows) or event.name not in ('source', 'predecessor', 'classified'):
                raise ValueError('input fact member/type mismatch')
            dep = getattr(state.spec.rows[event.index], event.name)
            if dep is None or dep.selector != event.selector:
                raise ValueError('input fact selector mismatch')
            if state.outcomes[event.index] in (Outcome.SKIPPED_FINISHED, Outcome.SKIPPED_SHUTDOWN):
                continue
            if state.physical_done or event.name in state.captured[event.index]:
                raise ValueError('duplicate or expired input fact')
            self._capture(state, event)
        self.wakeup.set()

    def _capture(self, state, event):
        index, name = event.index, event.name
        state.captured[index][name] = event.snapshot
        if self.profile:
            state.facts.append((name.upper() + '_CONTROL_CAPTURED', event.observed_ns, None))
            self._fact(state, name.upper() + '_RECEIVED')
        if name == 'classified' and event.snapshot.get('lifecycle') == Lifecycle.FINISHED:
            state.outcomes[index] = Outcome.SKIPPED_FINISHED

    def _ready(self, state, names):
        return all(state.outcomes[i] == Outcome.SKIPPED_FINISHED or all(
            getattr(row, name) is None or name in state.captured[i] for name in names)
            for i, row in enumerate(state.spec.rows))

    def _advance(self, bank):
        state = bank.current
        if state is None:
            return False
        if self.direct_imports and not state.imported:
            completion = self.dma.import_receipt(bank.bank_id, state.spec.work_seq)
            if completion is not None:
                queued, state.h2d_receipt = completion
                state.imported = True
                if self.profile:
                    state.facts.append(('H2D_SUBMITTED', queued, None))
                self._fact(state, 'H2D_DONE')
        if all(o in (Outcome.SKIPPED_FINISHED, Outcome.SKIPPED_SHUTDOWN) for o in state.outcomes):
            # Already-submitted metadata/import/compile jobs still own their inputs.
            if not state.jobs and (not self.direct_imports or state.imported):
                self._finish(state)
                return True
            return False
        changed = False
        if state.layout is None:
            # A continuation may import before peer compute/classification, but
            # may not own rows until every live same-side source is retired.
            # Initial work has no useful early import: wait for its compute inputs.
            names = ('source',) if any(r.source is not None for r in state.spec.rows) else ('predecessor', 'classified')
            if not self._ready(state, names):
                return False
            state.layout = self.banks.allocate(state.spec)
            if state.layout is None:
                return False
            self._fact(state, 'LAYOUT_ALLOCATED', state.layout)
            changed = True
        if state.import_inputs is None and 'IMPORT_INPUT' not in state.jobs and self._ready(state, ('source',)):
            self._submit(state, 'IMPORT_INPUT', self.role.compile_import(state.spec, tuple(dict(c) for c in state.captured), state.layout,
                tuple(i for i, o in enumerate(state.outcomes) if o is None)))
            changed = True
        if state.plan is None and 'COMPILE' not in state.jobs:
            live = tuple(i for i, o in enumerate(state.outcomes) if o is None)
            inline = (getattr(getattr(self.role, 'input_pool', None), 'inline', False)
                      and getattr(self.role, 'incremental_inputs', False))
            if inline:
                ready = tuple(i for i in live if i not in state.prepared_rows and all(
                    getattr(state.spec.rows[i], name) is None or name in state.captured[i]
                    for name in ('source', 'predecessor', 'classified')))
                complete = all(i in state.prepared_rows or i in ready for i in live)
                if ready or complete:
                    if complete and self.handoff_trace is not None:
                        self.handoff_trace.mark('PLAN_BEGIN', state.spec.work_seq)
                    if ready:
                        part = self.role.compile_compute(state.spec, tuple(state.captured), state.layout, ready)
                        if tuple(row.index for row in part.rows) != ready:
                            raise RuntimeError('partial plan does not cover ready members')
                        state.prepared_template = part
                        for row in part.rows:
                            if row.index in state.prepared_rows:
                                raise RuntimeError('input descriptor prepared twice')
                            state.prepared_rows[row.index] = row
                    if complete:
                        plan = replace(state.prepared_template, rows=tuple(state.prepared_rows[i] for i in live))
                        if state.spec.operation.name.startswith('DRAFT'):
                            total = sum(len(r.suffix) for r in plan.rows)
                        else:
                            total = sum(len(r.prompt) or len(r.proposal)+1 for r in plan.rows)
                        if total > self.role.max_batch_tokens:
                            raise ValueError('WORK exceeds model token capacity')
                        if self.handoff_trace is not None:
                            self.handoff_trace.mark('PLAN_RETURN', state.spec.work_seq)
                        self._apply(state, 'COMPILE', plan)
                    changed = True
            elif self._ready(state, ('source', 'predecessor', 'classified')):
                self._submit(state, 'COMPILE', self.role.compile_compute(state.spec,
                    tuple(dict(c) for c in state.captured), state.layout, live))
                changed = True
        if state.import_inputs is not None and state.metadata is None and 'METADATA' not in state.jobs:
            self._submit(state, 'METADATA', self.write_metadata(state.layout, state.import_inputs))
            changed = True
        if self.direct_imports:
            if bank.phase == Phase.FILLING and state.imported and state.metadata is not None:
                bank.phase = Phase.READY
            return changed
        if state.metadata is not None and not state.imported and 'H2D' not in state.jobs:
            plan = state.import_inputs.copy_plan(state.metadata)
            if plan is None:
                # Metadata job returns only after its physical fence; no fake H2D.
                state.imported = True
                bank.phase = Phase.READY
            else:
                self._fact(state, 'H2D_SUBMITTED')
                self._submit(state, 'H2D', self.dma.submit(bank.bank_id, plan))
            changed = True
        return changed

    def step(self, budget=64):
        if self.failed is not None:
            raise RuntimeError('worker failed; no further progress permitted') from self.failed
        try:
            return self._step(budget)
        except BaseException as error:
            self.failed = error
            raise

    def _step(self, budget):
        changed = False
        for _ in range(budget):
            try:
                state, kind, future = self.done.get_nowait()
            except Empty:
                break
            self._handle(state, kind, future)
            changed = True
        # Oldest authorized WORK gets its existing per-worker priority. Submit
        # as soon as it becomes ready, before preparing the other Bank.
        banks = sorted(self.banks.banks, key=lambda b: b.current.spec.work_seq if b.current else float('inf'))
        for bank in banks:
            changed |= self._advance(bank)
            changed |= self._dispatch_compute()
        return changed

    def _dispatch_compute(self):
        if self.compute is None:
            eligible = [b.current for b in self.banks.banks if b.current is not None
                and b.phase == Phase.READY and b.current.plan is not None
                and not b.current.jobs and any(o is None for o in b.current.outcomes)]
            if eligible:
                state = min(eligible, key=lambda s: s.spec.work_seq)
                self.compute = state
                self.banks.banks[state.spec.bank_id].phase = Phase.COMPUTING
                self._fact(state, 'COMPUTE_SELECTION_OBSERVED')
                self._fact(state, 'COMPUTE_SUBMITTED')
                self._submit(state, 'COMPUTE', self.execute(state.plan))
                return True
        return False

    def publication_retired(self, work_seq):
        state = self.records[work_seq]
        if not state.physical_done:
            raise RuntimeError('result storage retired before physical completion')
        del self.records[work_seq]

    def drain(self):
        self.draining = True

    def shutdown(self):
        """Global producers stopped. Retire unstarted work, never cancel DMA/model.

        Computations already submitted finish their output and automatic export.
        Cohort drain remains distinct and keeps all normal dependencies alive.
        """
        self.draining = self.shutting_down = True
        for state in self.records.values():
            if state.physical_done or state is self.compute or state.result is not None:
                continue
            for i, outcome in enumerate(state.outcomes):
                if outcome is None:
                    state.outcomes[i] = Outcome.SKIPPED_SHUTDOWN
        self.wakeup.set()

    @property
    def quiescent(self):
        return self.draining and not self.records

    def recycle(self):
        if self.shutting_down:
            raise RuntimeError('shutdown is terminal; cannot recycle')
        if not self.quiescent:
            raise RuntimeError('recycle requires physical and publication barrier')
        self.draining = False
