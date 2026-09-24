"""Mechanical single-owner loop: observe, classify, schedule, dispatch."""

from time import perf_counter_ns, monotonic
from types import MappingProxyType
from threading import get_ident

from nebulasd.core.enums import Lifecycle, StateChangeBlockKind as K
from nebulasd.observability.latency import LatencySamples
from nebulasd.scheduler.views import SchedulingView
from nebulasd.table.reader import IncrementalTableReader, ENGINE_IGNORED_KINDS
from .local_state import SchedulingRow, PayloadRow, SchedulingSnapshotReader
from .output_manager import OutputManager
from .request_registry import RequestRegistry, AdmissionRejected
from .failure_policy import StopEnginePolicy
from .recycling import CohortRecycler


class Engine:
    def __init__(self, resources, supervisor, *, scheduler=None, on_tokens=None, observer=None):
        self.resources, self.supervisor = resources, supervisor
        self.observer = observer
        self._retry_dispatch = False
        if scheduler is None:
            raise ValueError("Engine requires an explicit completion scheduler")
        self.scheduler = scheduler
        if not getattr(supervisor, 'autonomous', False):
            raise ValueError('Engine requires autonomous WORK workers')
        if getattr(self.scheduler, 'execution_mode', None) != 'completion':
            raise ValueError('Engine requires the completion scheduler')
        self.autonomous = True
        from .work_progress import WorkProgress
        self.scheduling_progress = WorkProgress(self)
        self.registry = RequestRegistry(resources)
        self.outputs = OutputManager(self.registry, on_tokens=on_tokens)
        self.reader = IncrementalTableReader(request_table=resources.table, worker_registry=resources.registry,
                                             rings=resources.events)
        from .work_ledger import WorkLedger
        self.ledger = WorkLedger(resources)
        self.failure_policy = StopEnginePolicy()
        self.recycler = CohortRecycler(self)
        self.rows, self.candidates = {}, {}
        self.schedule_latency, self.step_latency = LatencySamples(), LatencySamples()
        self.dispatch_latency = LatencySamples()
        self._owner, self._poisoned, self._closed = get_ident(), False, False
        self.reader.read_snapshot = SchedulingSnapshotReader()
        self.reader.max_entries = 128
        from nebulasd.core.enums import WorkerRole
        self.reader.priority_rows = tuple(
            key for w in resources.specs
            for key in (
                *((K.WORKER_DRAFT_BANK if w.role == WorkerRole.DRAFT else K.WORKER_BANK,
                   w.worker_id*2+b) for b in (0, 1)),
                (K.WORKER_DRAFT_RUNTIME if w.role == WorkerRole.DRAFT
                 else K.WORKER_TARGET_COMPUTE_RUNTIME, w.worker_id)))
        self.reader.ignored_kinds = ENGINE_IGNORED_KINDS
        from nebulasd.table.native_storage import NativeTablePartition
        from nebulasd.ipc.native_ring import NativeStateChangeRing
        if (all(isinstance(p, NativeTablePartition) for table in (resources.table, resources.registry)
                for p in table._partitions.values()) and
                all(isinstance(r, NativeStateChangeRing) for r in resources.events)):
            from nebulasd.table.native_reader import NativeTableReader
            self.reader = NativeTableReader(request_table=resources.table,
                worker_registry=resources.registry, rings=resources.events,
                priority_rows=self.reader.priority_rows, max_entries=128)
        self.outputs.defer_delivery = True

    def admit(self, request_id, prompt, config):
        self._check_owner()
        try:
            self.recycler.check_admission()
            slot = self.registry.admit(request_id, prompt, config)
            # Admission is cold. Make static allocations available before the
            # local candidate; bounded worker observation cannot split these.
            kinds = (K.REQUEST_HOSTKV, K.REQUEST_DRAFT_HOSTKV)
            self._merge(tuple(self.resources.table.partition(k).read_stable(slot) for k in kinds))
            self._request_changed(self.registry.records[slot], admitted=True)
            return slot
        except AdmissionRejected:
            raise
        except Exception:
            self._poisoned = True  # Partial arena/admission mutation is fail-fast.
            self.failure_policy.fail(self.supervisor)
            raise

    def cancel(self, request_id):
        self._check_owner()
        raise NotImplementedError("autonomous Engine cancellation is outside the supported completion path")

    def step(self):
        self._check_owner()
        start = perf_counter_ns()
        try:
            self.supervisor.check()
            self.supervisor.bell.drain()
            changed = self._observe_facts()
            changed |= self.ledger.observe_completions()
            changed |= self.ledger.refresh(self.rows)
            if hasattr(self.supervisor, 'retire_completed'):
                self.supervisor.retire_completed(self.ledger)
            recycle_changed = self.recycler.progress()
            if self.recycler.busy or recycle_changed:
                return bool(changed or recycle_changed)
            if not (changed or self._retry_dispatch or self.reader.has_pending()
                    or getattr(self.scheduler, "requires_clock_ticks", True)):
                return False
            sent = self.scheduling_progress.advance()
            return bool(changed or sent)
        except Exception:
            self._poisoned = True
            self.failure_policy.fail(self.supervisor)
            raise
        finally:
            try:
                if not self._poisoned:
                    self.outputs.flush()
            except Exception:
                self._poisoned = True
                self.failure_policy.fail(self.supervisor)
                raise
            finally:
                self.step_latency.add(perf_counter_ns() - start)

    def _publish_local(self, kind, slot, fields):
        self._publish_locals(kind, ((slot, fields),))

    def _publish_locals(self, kind, updates):
        progress = self.scheduling_progress
        progress.begin_updates()
        try:
            for slot, fields in updates:
                self._set_local(kind, slot, fields)
        finally:
            progress.end_updates()

    def _set_local(self, kind, slot, fields):
        previous = self.rows.get((kind, slot))
        values = {} if previous is None else dict(previous._values)
        values.update(fields)
        row = SchedulingRow.local(kind, slot, 0 if previous is None else previous.publish_seq+1, values)
        self.rows[kind, slot] = row
        self.scheduling_progress.observed(row, previous)
        if self.observer is not None:
            self.observer.observed((row,), perf_counter_ns())
        self._retry_dispatch = True

    def _request_changed(self, record, *, admitted=False):
        slot = record.input.slot
        if record.lifecycle == Lifecycle.ACTIVE:
            self.candidates[slot] = record.input
        else:
            self.candidates.pop(slot, None)
        self.scheduling_progress.request_changed(record, admitted=admitted)
        self._retry_dispatch = True

    def _scheduling_view(self):
        return SchedulingView(MappingProxyType(self.candidates), self.resources.specs,
            MappingProxyType(self.rows), MappingProxyType({}),
            MappingProxyType({}), MappingProxyType(self.ledger.sequences),
            self.ledger, MappingProxyType(self.registry.records))

    def _observe_facts(self):
        """One bounded observation; row hints never carry completion evidence."""
        return self.apply_facts(self.reader.poll().views)

    def apply_facts(self, updates):
        """Apply a reader batch unique by (kind, row), retaining cross-poll fences.

        Validate against the pre-batch result cache: late copy acceptance must
        not depend on copy/result ordering inside this batch. Application then
        updates each accepted row once, including output and local candidates.
        """
        self.ledger.rows = self.rows
        accepted = []
        for row in updates:
            previous = self.rows.get((row.block_kind, row.row))
            if previous is not None and row.publish_seq <= previous.publish_seq:
                continue
            context = self.ledger.validate_fact(row)
            if context:
                accepted.append((row, previous, context))
        progress = self.scheduling_progress
        progress.begin_updates()
        try:
            for row, previous, context in accepted:
                if not isinstance(row, (SchedulingRow, PayloadRow)):
                    row = SchedulingRow.observed(row)
                self.rows[row.block_kind, row.row] = row
                progress.observed(row, previous)
                record = self.outputs.consume_row(row)
                if record is not None:
                    self._request_changed(record)
                if context is not True:
                    self.ledger.result_applied(context, previous)
            if self.observer is not None:
                self.observer.observed(tuple(row for row, _, _ in accepted), perf_counter_ns())
        finally:
            progress.end_updates()
        return bool(accepted)

    def _merge(self, updates):
        if self.observer is not None:
            self.observer.observed(updates, perf_counter_ns())
        for row in updates:
            if not isinstance(row, (SchedulingRow, PayloadRow)):
                row = SchedulingRow.observed(row)
            previous = self.rows.get((row.block_kind, row.row))
            self.rows[(row.block_kind, row.row)] = row
            self.scheduling_progress.observed(row, previous)

    def run_until_complete(self, timeout=60):
        deadline = monotonic() + timeout
        while True:
            progressed = self.step()
            if self.registry.records and all(r.lifecycle != Lifecycle.ACTIVE for r in self.registry.records.values()):
                return {r.request_id: r.output for r in self.registry.records.values()}
            if monotonic() > deadline:
                self.write_diagnostics("timeout.json")
                raise TimeoutError("Engine did not complete requests")
            if not progressed:
                self.supervisor.bell.wait(0.001)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.supervisor.close()
        finally:
            self.outputs.on_tokens = None  # Break client callback cycles before the next session.
            self.resources.close()
            if getattr(self, "profiler", None) is not None:
                self.profiler.flush()

    def write_diagnostics(self, name):
        import json
        data = {"rows": [{"kind": kind.name, "row": row, "seq": snapshot.publish_seq,
                         "fields": {f.name: f.value for f in snapshot.fields}}
                        for (kind, row), snapshot in self.rows.items()],
                "authorizations": self.ledger.records,
                "requests": self.registry.records}
        (self.supervisor.output_dir / name).write_text(json.dumps(data, default=str, indent=2))

    def _check_owner(self):
        if get_ident() != self._owner:
            raise RuntimeError("Engine must stay on its ScheduleLoop owner thread")
        if self._closed or self._poisoned:
            raise RuntimeError("Engine closed or poisoned")
