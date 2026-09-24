"""Opt-in bounded host tracing. No event IPC, CUDA synchronize or hot file I/O."""
from collections import deque
from contextvars import ContextVar
from functools import wraps
import inspect
import os
import json
from pathlib import Path
from threading import get_ident
from time import perf_counter_ns, thread_time_ns


def request_keys(command):
    command = getattr(command, 'command', command)
    if hasattr(command,'operation') and hasattr(command,'rows'):
        return [[r.slot,r.epoch,r.round_id] for r in command.rows]
    items = getattr(command, 'requests', None)
    if items is None:
        items = (*getattr(command, 'new_requests', ()), *getattr(command, 'cached_request_deltas', ()))
    return [[r.request_slot, r.request_epoch, getattr(r, 'round_id', getattr(r, 'next_round_id', None))] for r in items]


class ProfileRecorder:
    def __init__(self, directory, owner, capacity, *, mode="full"):
        self.mode = mode
        self.directory, self.owner = Path(directory), owner
        self.events = deque(maxlen=capacity)
        self.total = 0
        self.short_calls = {}
        self.gpu = None
        self.keys = ContextVar('profile_keys', default=())

    def record(self, name, start, end=None, *, keys=None, **fields):
        self.total += 1
        self.events.append(dict(name=name, start_ns=start, end_ns=start if end is None else end,
            owner=self.owner, thread=get_ident(), keys=list(self.keys.get() if keys is None else keys), **fields))

    def wrap(self, obj, method, name, *, command=False, items=False, category='control'):
        original = getattr(obj, method, None)
        if original is None:
            return
        def enter(args):
            keys = ([[r.request_slot, r.request_epoch, r.round_id] for r in args[0]]
                    if items else request_keys(args[0]) if command else None)
            token = self.keys.set(keys) if keys is not None else None
            return perf_counter_ns(), thread_time_ns(), token
        def leave(start, cpu, token, asynchronous, failed):
            end = perf_counter_ns()
            cpu_ns = None if asynchronous else thread_time_ns()-cpu
            if self.mode == 'light' and name in ('engine.ledger', 'engine.apply_facts') and end-start < 100_000 and not failed:
                counts = self.short_calls.setdefault(name, [0, 0, 0])
                counts[0] += 1
                counts[1] += end-start
                counts[2] += cpu_ns or 0
            else:
                self.record(name, start, end, category=category, cpu_ns=cpu_ns, failed=failed)
            if token is not None:
                self.keys.reset(token)
        if inspect.iscoroutinefunction(original):
            @wraps(original)
            async def wrapped(*args, **kwargs):
                start, cpu, token = enter(args)
                failed = True
                try:
                    result = await original(*args, **kwargs)
                    failed = False
                    return result
                finally:
                    leave(start, cpu, token, True, failed)
        else:
            @wraps(original)
            def wrapped(*args, **kwargs):
                start, cpu, token = enter(args)
                failed = True
                try:
                    result = original(*args, **kwargs)
                    failed = False
                    return result
                finally:
                    leave(start, cpu, token, False, failed)
        setattr(obj, method, wrapped)

    def attach_table(self, table):
        if os.environ.get("STARSD_TURNAROUND_LEAN") == "1":
            return
        for partition in table._partitions.values():
            self._attach_partition(partition)

    def _attach_partition(self, partition):
        from nebulasd.table.storage import _decode_scalar
        original, encoded_original = partition._publish, partition._publish_encoded
        ordinary = [False]
        def record(row, seq, data, start):
            keys = ([[row, data['request_epoch'], data.get('round_id')]]
                    if 'request_epoch' in data else list(self.keys.get()))
            self.record('fact.publish', start, perf_counter_ns(), keys=keys,
                kind=partition.block_kind.name, row=row, publish_seq=seq,
                fields=data if self.mode == 'light' else None,
                status=int(data['status']) if 'status' in data else None,
                compute_status=int(data['compute_status']) if 'compute_status' in data else None)
        def publish(row, seq, fields):
            start = perf_counter_ns()
            ordinary[0] = True
            try:
                original(row, seq, fields)
            finally:
                ordinary[0] = False
            record(row, seq, {f.name: f.value for f in fields}, start)
        by_offset = {offset: partition._field_by_name[name] for name, offset in partition._offset_by_name.items()}
        def encoded(row, seq, patch):
            start = perf_counter_ns()
            encoded_original(row, seq, patch)
            if ordinary[0]:
                return
            if hasattr(partition, 'segment'):
                offsets, lengths, raw, count = patch
                cursor, data = 0, {}
                for index in range(count):
                    field = by_offset[offsets[index]]
                    length = lengths[index]
                    data[field.name] = _decode_scalar(field.type, raw[cursor:cursor+length])
                    cursor += length
            else:
                data = {name: _decode_scalar(partition._field_by_name[name].type, raw) for name, raw in patch}
            record(row, seq, data, start)
        partition._publish, partition._publish_encoded = publish, encoded

    def observed(self, updates, now):
        if os.environ.get("STARSD_TURNAROUND_LEAN") == "1":
            return
        from nebulasd.scheduler.views import value
        for row in updates:
            epoch = value(row, 'request_epoch')
            self.record('fact.observe', now, keys=[] if epoch is None else [[row.row, epoch, value(row, 'round_id')]],
                        kind=row.block_kind.name, row=row.row, publish_seq=row.publish_seq)

    def dispatched(self, command, now):
        self.record('command.visible', now, keys=request_keys(command),
                    worker=command.worker_id, command_seq=command.command_seq, kind=command.kind.name)

    def flush(self, *, gpu_retired=True):
        if self.gpu is not None:
            if gpu_retired:
                self.gpu.finish()
            else:
                # A failed CUDA operation must not be synchronized for diagnostics.
                self.record('gpu.coverage', perf_counter_ns(), keys=[],
                            gpu_dropped_events=self.gpu.total, gpu_incomplete=True)
            self.gpu = None
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f'{self.owner}.json'
        target.write_text(json.dumps(dict(schema=1, clock='perf_counter_ns; same host only',
            owner=self.owner, mode=self.mode, short_calls=self.short_calls, total_events=self.total, dropped_events=self.total-len(self.events),
            events=list(self.events)), default=str))


class ProfileObserver:
    def __init__(self, recorder, observer):
        self.recorder, self.observer = recorder, observer

    def observed(self, rows, now):
        self.recorder.observed(rows, now)
        if self.observer is not None:
            self.observer.observed(rows, now)

    def dispatched(self, command, now):
        self.recorder.dispatched(command, now)
        if self.observer is not None:
            self.observer.dispatched(command, now)


def attach_engine(engine, recorder):
    if recorder.mode == "draft":
        return  # Client endpoints remain available; no Engine trace hooks.
    from .stage_predictions import attach as attach_predictions
    attach_predictions(engine.scheduler, recorder)
    if recorder.mode == 'light' and os.environ.get('STARSD_TURNAROUND_LEAN') != '1':
        from .stage_gates import attach as attach_gates
        attach_gates(engine.scheduler, recorder)
    engine.observer = ProfileObserver(recorder, engine.observer)
    recorder.attach_table(engine.resources.table)
    if recorder.mode == 'full':
        from .scheduler_batches import attach_batch_diagnostics
        attach_batch_diagnostics(engine.scheduler, recorder)
    else:
        from .light_profiling import attach_scheduler
        attach_scheduler(engine.scheduler, recorder)
    for obj, method, name in ((engine.reader, 'poll', 'engine.observe'),
            (engine, 'apply_facts', 'engine.apply_facts'), (engine.ledger, 'refresh', 'engine.ledger'),
            (engine.scheduler, 'schedule', 'engine.schedule'),
            (engine.registry, 'admit', 'engine.admit'), (engine.recycler, 'progress', 'engine.retire')):
        if recorder.mode == 'light' and method in ('poll', 'progress'):
            continue
        recorder.wrap(obj, method, name)
    from .scheduling_progress import attach as attach_scheduling_progress
    attach_scheduling_progress(engine, recorder)
    original_step = engine.step
    previous_reason = [None]
    def step():
        begin=perf_counter_ns()
        result=original_step()
        # A false step has neither a fact nor a retry to schedule. Record state
        # transitions only, avoiding a measurement event for every empty poll.
        reason='no_new_fact_or_retry' if not result else 'progress'
        if reason != previous_reason[0]:
            recorder.record('engine.decision_state',begin,perf_counter_ns(),keys=[],reason=reason,
                            recycling=engine.recycler.busy)
            previous_reason[0]=reason
        return result
    engine.step=step
    if engine.autonomous and os.environ.get("STARSD_COMPLETION_TRACE_DIR"):
        from .completion_trace import attach
        attach(engine, os.environ["STARSD_COMPLETION_TRACE_DIR"])
    if os.environ.get("STARSD_ENGINE_DETAIL_DIR"):
        from .engine_detail import attach_engine
        attach_engine(engine, recorder)
    if os.environ.get("STARSD_BANK_TURNAROUND_PROFILE") == "1":
        from .bank_turnaround import attach_engine as attach_turnaround
        attach_turnaround(engine, recorder)


def attach_worker(adapter, recorder):
    """Wrap only opted-in instances; normal method calls remain unchanged."""
    from dataclasses import asdict
    raw = adapter
    while hasattr(raw, 'adapter'):
        raw = raw.adapter
    if getattr(raw, 'execution_backend', None) is not None:
        from .execution_profiling import attach_control
        attach_control(raw, recorder)
        return
    if recorder.mode == 'draft':
        if not raw.target:
            from .draft_comparison import attach_draft_backend, attach_copy_receipts
            backend = raw.worker.backend if getattr(raw, 'draft_banked', False) else raw.worker._backend
            attach_draft_backend(backend, recorder)
            if getattr(raw, 'draft_banked', False):
                attach_copy_receipts(raw, recorder)
        return
    if getattr(raw, 'draft_banked', False):
        from .draft_profiling import attach
        attach(raw, recorder)
        return
    worker, runtime = raw.worker, raw.runtime
    if recorder.mode == "light":
        from .light_profiling import attach_forward
        attach_forward(raw, recorder)
    from .backend_work import attach_backend_work
    if recorder.mode == 'full':
        attach_backend_work(raw, recorder)
        from .gpu_profiling import attach_gpu
        recorder.gpu = attach_gpu(raw, recorder)
    recorder.attach_table(raw.resources.table)
    recorder.attach_table(raw.resources.registry)
    consumer = raw.consumer
    original_consume = consumer.consume
    def consume(**kwargs):
        start = perf_counter_ns()
        envelope = original_consume(**kwargs)
        if envelope is not None:
            command = envelope.decode(worker_id=worker.worker_id)
            recorder.record('command.consume', start, perf_counter_ns(), keys=request_keys(command),
                            command_seq=command.command_seq, worker=worker.worker_id, kind=command.kind.name)
        return envelope
    consumer.consume = consume
    recorder.wrap(worker, 'execute_async' if raw.target else 'execute', 'worker.execute', command=True, category='inclusive')
    for method in ('compile', 'compile_prefill', 'compile_verify', 'compile_prepare'):
        recorder.wrap(worker._input_compiler, method, f'worker.{method}')
    for method in ('prepare', 'commit', 'commit_prefill', 'commit_verify', 'commit_prepare', 'mark_gpu_ready'):
        recorder.wrap(worker._local_state, method, f'worker.state.{method}')
    for method in ('run_batch', 'prefill_batch_async', 'verify_batch_async', 'prefill_batch', 'verify_batch'):
        recorder.wrap(worker._backend, method, f'backend.{method}', category='backend_wall')
    if raw.target:
        for method in ('prepare','complete_prepare','activate_for_run_async','begin_compute','finish_compute','release_drain'):
            recorder.wrap(worker._bank_controller, method, f'worker.bank.{method}')
        recorder.wrap(worker._compute_lane._publisher, 'publish_ready_batch', 'worker.publish_result')
        lane = worker.copy_lane
        for method in ('host_ready','gpu_ready','d2h_started','h2d_wait'):
            recorder.wrap(lane.publisher, method, f'copy.publish.{method}', command=True)
        for method in ('accept_prepare', 'consume_ready'):
            recorder.wrap(lane, method, f'copy.{method}', command=True)
        previous = lane.on_receipt
        def receipt(plan, result):
            keys = [[r.extent.request_slot, r.extent.request_epoch, round_id]
                    for r, round_id in zip(plan.regions, plan.round_ids)]
            recorder.record('copy.receipt', result.enqueued_ns, result.retired_ns,
                            keys=keys, direction=plan.direction, timing=asdict(result))
            if previous is not None:
                previous(plan, result)
        lane.on_receipt = receipt
        for method in ('prepare', 'complete_h2d', 'drain_complete'):
            recorder.wrap(lane.banks, method, f'copy.bank.{method}', command=True)

    else:
        recorder.wrap(worker._publisher, 'publish_ready_batch', 'worker.publish_result')
