"""Opt-in owner-only causal trace. Bounded tuples, no hot-path I/O or IPC.

Spans are inclusive; consumers must subtract children when attributing time.
Capacity snapshots describe the actual call, without invoking policy again.
"""
from collections import deque
import json
from pathlib import Path
from time import perf_counter_ns, thread_time_ns

from nebulasd.core.enums import StateChangeBlockKind as K, WorkerRole
from nebulasd.scheduler.views import value as v


class CompletionTrace:
    def __init__(self, capacity=2000000):
        self.events = deque(maxlen=capacity)
        self.total = 0

    def record(self, name, start, end, cpu, data=()):
        self.total += 1
        self.events.append((name, start, end, cpu, data))

    def wrap(self, obj, method, name, identity=None):
        original = getattr(obj, method)
        def call(*args, **kwargs):
            data = identity(args, kwargs) if identity else ()
            start, cpu = perf_counter_ns(), thread_time_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.record(name, start, perf_counter_ns(), thread_time_ns()-cpu, data)
        setattr(obj, method, call)

    def flush(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'completion-trace.json').write_text(json.dumps(dict(
            schema=1, columns=['name', 'start_ns', 'end_ns', 'cpu_ns', 'data'],
            clock='perf_counter_ns; same host only', total_events=self.total,
            dropped_events=self.total-len(self.events), events=list(self.events))))


def attach(engine, directory):
    trace = CompletionTrace()
    engine.completion_trace = trace
    progress = engine.scheduling_progress
    for obj, method, name in (
        (engine, 'step', 'step'), (engine.supervisor, 'check', 'check'),
        (engine.supervisor.bell, 'drain', 'bell'),
        (engine, '_observe_facts', 'observe'),
        (engine.reader, 'poll', 'poll'), (engine, '_merge', 'merge'),
        (engine.outputs, 'consume', 'consume'), (engine.outputs, 'flush', 'flush'),
        (engine.recycler, 'progress', 'recycle'),
        (progress, '_events', 'events'), (progress, 'build', 'build'),
        (progress, 'publish_dispatch', 'publish')):
        trace.wrap(obj, method, name)
    def state():
        return {s:dict(dirty=sorted(progress.dirty[s]),
                      pending=[w.work_seq if w is not None else None for w, _, _ in progress.pending[s]]) for s in ('D', 'T')}
    trace.wrap(progress, 'advance', 'advance', lambda a, k: state())
    original_wake = progress._wake
    def wake(stage, reason, worker=None):
        before = set(progress.dirty[stage])
        original_wake(stage, reason, worker)
        added = progress.dirty[stage] - before
        if added:
            now = perf_counter_ns()
            trace.record('wake', now, now, 0, (stage, reason, sorted(added)))
    progress._wake = wake
    active_ledger = [None]
    original_step = engine.step
    def step():
        if engine.ledger is not active_ledger[0]:
            attach_ledger(engine.ledger, trace)
            active_ledger[0] = engine.ledger
        return original_step()
    engine.step = step
    original_close = engine.close
    def close():
        try:
            return original_close()
        finally:
            trace.flush(directory)
    engine.close = close
    return trace


def attach_ledger(ledger, trace):
    trace.wrap(ledger, 'observe_completions', 'completions')
    original_changes = ledger._schedule_changes
    def changes(record, before):
        original_changes(record, before)
        after = record.compute_done, record.physical_done
        if before != after:
            now = perf_counter_ns()
            w = record.work
            trace.record('transition', now, now, 0,
                         (w.worker_id, w.worker_generation, w.work_seq, before, after))
    ledger._schedule_changes = changes
    original_capacity = ledger.capacity
    def capacity(view, worker):
        start, cpu = perf_counter_ns(), thread_time_ns()
        result = original_capacity(view, worker)
        end, elapsed_cpu = perf_counter_ns(), thread_time_ns()-cpu
        records = tuple(ledger.by_worker.get(worker.worker_id, {}).values())
        kind = K.WORKER_DRAFT_BANK if worker.role == WorkerRole.DRAFT else K.WORKER_BANK
        banks = [view.row(kind, worker.worker_id*2+b) for b in (0, 1)]
        trace.record('capacity', start, end, elapsed_cpu, dict(
            worker=worker.worker_id, generation=worker.generation,
            accepted=result is not None, direct_imports=ledger.direct_imports,
            max_batch_size=worker.max_batch_size, bank_rows=worker.bank_rows,
            banks=[{f:v(b,f) for f in ('state','bank_epoch','alloc_rows')} for b in banks],
            records=[(r.work.work_seq,r.work.bank_id,r.compute_done,r.physical_done,
                      r.completion is not None,len(r.applied),len(r.work.rows)) for r in records]))
        return result
    ledger.capacity = capacity
    trace.wrap(ledger, 'sent', 'sent')
    original_refresh = ledger.refresh
    def refresh(*args, **kwargs):
        before = tuple(ledger.records)
        start, cpu = perf_counter_ns(), thread_time_ns()
        try:
            return original_refresh(*args, **kwargs)
        finally:
            end = perf_counter_ns()
            trace.record('refresh', start, end, thread_time_ns()-cpu)
            for key in before:
                if key not in ledger.records:
                    # Retirement happened inside this refresh interval.
                    trace.record('credit', start, end, None, key)
    ledger.refresh = refresh
