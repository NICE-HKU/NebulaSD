"""Global, bounded planning for independent compute and Prepare opportunities.

Location is a cost, never affinity. Cheap classification sees all requests;
full identity joins and measured predictions only see the bounded frontier.
No decisions survive here: the Engine owns issued/pending reservations.
"""
from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter_ns

from nebulasd.core.enums import (StateChangeBlockKind as K, WorkerRole, BankRole,
    BankState, ComputeStatus, DraftStatus, TargetStatus, D2HStatus)
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import U64
from nebulasd.ipc.protocol import DraftBatchCommand, NewRequestData
from nebulasd.ipc.draft_protocol import DraftInitialBank, DraftPrepareRequest, PrepareDraftBankCommand
from . import builders
from .batching import BatchWindow
from .draft_placement import source, scheduled_depth
from .eligibility import ScheduleEligibility, active, online
from .policy import Scheduler
from .stage_planner import plan
from .views import value as v


def host_ready(view, stage, request):
    """Filter before the bounded frontier: unreadable sources must not crowd it."""
    draft = stage == 'D'
    dispatch = view.row(K.REQUEST_DISPATCH, request.slot)
    if not v(dispatch, 'draft_issue_seq' if draft else 'target_run_seq', 0):
        return True  # Initial work has no KV import.
    host = view.row(K.REQUEST_DRAFT_D2H if draft else K.REQUEST_D2H, request.slot)
    result = view.row(K.REQUEST_DRAFT if draft else K.REQUEST_TARGET_COMPUTE, request.slot)
    version = v(result, 'snapshot_version' if draft else 'target_kv_version')
    return (v(host, 'request_epoch') == request.epoch and v(host, 'status') == D2HStatus.HOST_READY
            and v(host, 'ready_version') == version)


class NumericRow(dict):
    def get(self, name):
        # Preserve BlockSnapshot's missing-field contract for value(default).
        return self[name]


class InvocationRows:
    def __init__(self, rows):
        self.rows, self.cache = rows, {}

    def get(self, key, default=None):
        if key not in self.cache:
            row = self.rows.get(key)
            self.cache[key] = (None if row is None else
                NumericRow((f.name, f.value) for f in row.fields) if hasattr(row, 'fields') else row)
        return self.cache[key] if self.cache[key] is not None else default


def prepare_capacity(view, worker):
    """Published standby intent capacity, not physical reserve/H2D readiness.

    An issued compute may precede its Bank publication. Wait for that identity
    before trusting role/row counts; in particular an initial EMPTY active Bank
    does not yet describe the rows consumed by the dispatched initial batch.
    """
    if view.work_state is not None:
        return view.work_state.capacity(view, worker)
    kind = K.WORKER_DRAFT_BANK if worker.role == WorkerRole.DRAFT else K.WORKER_BANK
    banks = tuple(view.row(kind, worker.worker_id * 2 + i) for i in (0, 1))
    bank = next((b for b in banks if v(b, 'role') == BankRole.STANDBY), None)
    if v(bank, 'state') not in (BankState.EMPTY, BankState.DRAINING):
        return None
    other = banks[1 - v(bank, 'bank_id')]
    if v(other, 'role') != BankRole.ACTIVE:
        return None
    command = view.inflight.get(worker.worker_id)
    if command is not None:
        if command.kind.name == 'DRAFT_BATCH':
            identity = command.bank.bank_id, command.bank.bank_epoch, command.bank.batch_seq
        elif command.kind.name == 'TARGET_PREFILL_BATCH':
            identity = command.bank_id, command.bank_epoch, command.batch_seq
        else:
            identity = command.active_bank_id, command.active_bank_epoch, command.expected_batch_seq
        if tuple(v(other, f) for f in ('bank_id', 'bank_epoch', 'batch_seq')) != identity:
            return None
    # Both Banks share row IDs. Even a draining ACTIVE Bank can retain them;
    # only the destination's own old rows may be deferred to its retirement.
    rows = min(worker.max_batch_size, v(bank, 'capacity_rows', 0) - v(other, 'alloc_rows', 0))
    return (bank, rows) if rows > 0 else None


def initial_capacity(view, worker):
    if view.work_state is not None:
        return view.work_state.capacity(view, worker)
    draft = worker.role == WorkerRole.DRAFT
    kind = K.WORKER_DRAFT_BANK if draft else K.WORKER_BANK
    bank = next((b for i in (0, 1) if (b := view.row(kind, worker.worker_id * 2 + i))
                 is not None and v(b, 'role') == BankRole.ACTIVE), None)
    state = v(bank, 'state')
    if not (state == BankState.EMPTY or (not draft and state == BankState.DRAINING and v(bank, 'alloc_rows', 0) == 0)):
        return None
    other = view.row(kind, worker.worker_id * 2 + 1 - v(bank, 'bank_id'))
    rows = min(worker.max_batch_size, v(bank, 'capacity_rows', 0) - v(other, 'alloc_rows', 0))
    return (bank, rows) if rows > 0 else None


@dataclass(frozen=True, slots=True)
class RequestSchedulingFacts:
    request: object
    initial: bool
    source: object = None


class BatchAggregates:
    """A prefix costs one addition; reuse it across all destination fits."""
    def __init__(self, stage, initial):
        self.stage, self.initial = stage, initial
        self.cache = {(): (0, 0)}

    def get(self, batch):
        key = tuple(r.slot for r in batch)
        if key not in self.cache:
            missing, prefix = [], len(key)
            while key[:prefix] not in self.cache:
                prefix -= 1
                missing.append(prefix)
            blocks, tokens = self.cache[key[:prefix]]
            for index in reversed(missing):
                r = batch[index]
                cost = (r.prompt_count if self.stage == 'T' else
                        r.prompt_count + r.output_count + scheduled_depth(r)) if self.initial else r.proposal_depth + 1
                blocks, tokens = blocks + r.capacity_blocks, tokens + cost
                self.cache[key[:index + 1]] = blocks, tokens
        return self.cache[key]


class CompletionScheduler:
    execution_mode = 'completion'
    requires_clock_ticks = False
    service_interval = False

    def __init__(self, *, estimator, clock=perf_counter_ns, frontier_factor=2,
                 draft_service_weight=32.0, target_service_weight=32.0):
        if not hasattr(estimator, 'stage_prediction') or getattr(estimator, 'draft_block_bytes', None) is None:
            raise ValueError('completion requires a stage-aware measured estimator and Draft layout')
        if type(frontier_factor) is not int or not 1 <= frontier_factor <= 4:
            raise ValueError('frontier_factor must be an integer in [1, 4]')
        if any(not isfinite(x) or x < 0 for x in (draft_service_weight, target_service_weight)):
            raise ValueError('service weights must be finite and nonnegative')
        self._estimator, self._clock = estimator, clock
        self.frontier_factor = frontier_factor
        self.service_weights = {'D': draft_service_weight, 'T': target_service_weight}
        # This path only joins already-frozen batches; it never reconstructs.
        self._ready = Scheduler(estimator=estimator)
        self.last_metrics = {}
        self.blocked_workers = set()
        self._native_dirty = None  # Explicit snapshot callers still get a full identity scan.
        from .compute_times import ComputeTimes
        self.compute_times = ComputeTimes()

    def observe_compute_result(self, work, result):
        changed = self.compute_times.observe(work, result)
        if self._native_dirty is not None:
            self._native_dirty.update(changed)

    def reset_compute_times(self):
        self.compute_times.clear()
        if self._native_dirty is not None:
            self._native_dirty.update(getattr(getattr(self, '_native', None), 'requests', ()))

    def enable_native_updates(self):
        self._native_dirty = set()

    def requests_changed(self, slots):
        if self._native_dirty is not None:
            self._native_dirty.update(slots)

    def observe_result(self, row):
        self.compute_times.observe_row(row)

    def observe_table(self, row):
        self.compute_times.observe_row(row)
        if self._native_dirty is not None and row.block_kind in (
                K.REQUEST_ENGINE, K.REQUEST_DISPATCH, K.REQUEST_DRAFT, K.REQUEST_TARGET_COMPUTE,
                K.REQUEST_D2H, K.REQUEST_DRAFT_D2H, K.REQUEST_DRAFT_HOSTKV):
            self._native_dirty.add(row.row)

    def schedule(self, view, *, phase, destinations=None, compute_destinations=None, prepare_destinations=None):
        import os
        from .native_completion import NativeCompletion, supported, estimator_base
        mode = os.environ.get('STARSD_SCHEDULER_IMPL', 'cpp')
        if mode not in ('cpp', 'python', 'check'):
            raise ValueError('STARSD_SCHEDULER_IMPL must be cpp, python or check')
        kwargs = dict(phase=phase, destinations=destinations, compute_destinations=compute_destinations,
                      prepare_destinations=prepare_destinations)
        if mode == 'python' or phase == 'ready' or not supported(self._estimator, view):
            return self._schedule_python(view, **kwargs)
        now = self._clock()
        native = getattr(self, '_native', None)
        if native is None or native.estimator is not estimator_base(self._estimator):
            self._native = native = NativeCompletion(self._estimator)
        result = native.schedule(self, view, now=now, **kwargs)
        if mode == 'check':
            metrics, blocked = self.last_metrics, self.blocked_workers
            reference = self._schedule_python(view, _now=now, **kwargs)
            materialized = tuple(c.materialize(self, view, native.banks[c.worker_id,
                c.kind.name in ('DRAFT_BATCH','TARGET_PREFILL_BATCH')]) for c in result)
            if materialized != reference:
                raise AssertionError(f'C++ scheduler decision differs: {materialized!r} != {reference!r}')
            if blocked != self.blocked_workers:
                raise AssertionError('C++ scheduler blocked workers differ')
            self.last_metrics, self.blocked_workers = metrics, blocked
        return result

    def _schedule_python(self, view, *, phase, destinations=None, compute_destinations=None, prepare_destinations=None, _now=None):

        if phase == 'ready':
            return self._ready.schedule(view, phase='ready')
        if phase not in ('D', 'T'):
            raise ValueError('completion scheduling requires an explicit stage')
        start, now = perf_counter_ns(), self._clock() if _now is None else _now
        metrics = dict(scan_ns=0, frontier_ns=0, facts_ns=0, reconstruction_ns=0,
            placement_ns=0, assignment_ns=0, total_ns=0, scanned=0, frontier=0,
            frontier_limit=0, prediction_cells=0, fit_checks=0, partitions=0, batches=0)
        self.last_metrics = metrics
        self.blocked_workers = set()
        # Decode worker rows once; request rows stay untouched until <=K facts.
        cached = replace(view, rows=InvocationRows(view.rows))
        role = WorkerRole.DRAFT if phase == 'D' else WorkerRole.TARGET
        runtime = K.WORKER_DRAFT_RUNTIME if phase == 'D' else K.WORKER_TARGET_COMPUTE_RUNTIME
        workers = [w for w in view.workers if w.role == role
            and (destinations is None or w.worker_id in destinations)
            and w.worker_id not in view.prepared and online(cached, w)]
        compute_workers = [w for w in workers if w.worker_id not in view.inflight
            and (compute_destinations is None or w.worker_id in compute_destinations)
            and (view.work_state is not None or v(cached.row(runtime, w.worker_id), 'compute_status') == ComputeStatus.IDLE)]
        prepare_workers = [w for w in workers
            if prepare_destinations is None or w.worker_id in prepare_destinations]
        if phase == 'D' and any(not w.draft_banked for w in workers):
            raise ValueError('completion requires banked Draft workers')
        capacities = ({w.worker_id: capacity for w in compute_workers
                       if (capacity := initial_capacity(cached, w)) is not None},
                      {w.worker_id: capacity for w in prepare_workers
                       if (capacity := prepare_capacity(cached, w)) is not None})
        compute_workers = [w for w in compute_workers if w.worker_id in capacities[0]]
        prepare_workers = [w for w in prepare_workers if w.worker_id in capacities[1]]
        capable = capacities[0].keys() | capacities[1].keys()
        self.blocked_workers.update(w.worker_id for w in workers if w.worker_id not in capable)
        workers = [w for w in workers if w.worker_id in capable]
        if not workers:
            metrics['total_ns'] = perf_counter_ns() - start
            return ()
        limit = self.frontier_factor * sum(w.max_batch_size for w in workers)
        max_blocks = max(w.bank_blocks for w in workers)
        metrics['frontier_limit'] = limit
        reserved = {r.request_slot for c in view.prepared.values()
            if c.kind.name == ('PREPARE_DRAFT_BANK' if phase == 'D' else 'PREPARE_TARGET_BANK') for r in c.requests}
        # Fixed 16 buckets: increasing age overrides ready/near-ready preference.
        # Each buffer is bounded; tie order is stable global table iteration.
        # Only the selected <=K requests are subsequently sorted and decoded.
        buckets = [[] for _ in range(16)]
        service_candidates = []
        begin = perf_counter_ns()
        for r in view.requests.values():
            metrics['scanned'] += 1
            if r.slot in reserved or r.capacity_blocks > max_blocks or not active(view, r):
                continue
            classification = self._cheap(view, phase, r, allow_initial=bool(compute_workers),
                                         allow_prepare=bool(prepare_workers))
            if classification is None:
                continue
            interval = self.service_interval and (phase == 'D' or
                v(view.row(K.REQUEST_DISPATCH, r.slot), 'target_run_seq', 0))
            if (interval or getattr(getattr(view, 'work_state', None), 'direct_imports', False)) and not host_ready(view, phase, r):
                continue
            if interval:
                service_candidates.append(r)
                if phase == 'D':
                    continue
            age = max(0, now - (r.ready_ns or r.admitted_ns))
            age_bin = min(7, (age // 1_000_000).bit_length())
            bucket = buckets[(7 - age_bin) * 2 + int(not classification)]
            if len(bucket) < limit:
                bucket.append(r)
        metrics['scan_ns'] = perf_counter_ns() - begin
        begin = perf_counter_ns()
        frontier = []
        for bucket in buckets:
            frontier.extend(bucket[:limit - len(frontier)])
            if len(frontier) == limit:
                break
        frontier.sort(key=lambda r: (r.ready_ns or r.admitted_ns, r.arrival_seq, r.slot))
        if self.service_interval and phase == 'T':
            # Preserve prefill admission in the original mixed-stage frontier.
            # Continuations are reconsidered by the per-worker service policy.
            frontier = [r for r in frontier if not v(view.row(K.REQUEST_DISPATCH, r.slot), 'target_run_seq', 0)]
        metrics['frontier'] = len(frontier)
        frontier.extend(service_candidates)
        metrics['frontier_ns'] = perf_counter_ns() - begin
        # Lazy numeric decoding only for frontier, worker facts and bounded
        # issued predecessor batches accessed by the existing estimator.
        begin = perf_counter_ns()
        eligibility = ScheduleEligibility(cached)
        facts = []
        for r in frontier:
            dispatch = cached.row(K.REQUEST_DISPATCH, r.slot)
            initial = not v(dispatch, 'draft_issue_seq' if phase == 'D' else 'target_run_seq', 0)
            if initial:
                if compute_workers and (phase == 'T' or eligibility.target_result(r) is not None):
                    facts.append(RequestSchedulingFacts(r, True))
            elif phase == 'D':
                identity = source(cached, r)
                if identity is not None and v(dispatch, 'target_round_id') == identity.round_id and v(dispatch, 'target_run_seq', 0):
                    facts.append(RequestSchedulingFacts(r, False, identity))
            elif eligibility.draft_issued(r):
                facts.append(RequestSchedulingFacts(r, False))
        metrics['facts_ns'] = perf_counter_ns() - begin
        estimator = self._estimator.for_invocation() if hasattr(self._estimator, 'for_invocation') else self._estimator
        decisions, occupied = [], set()
        sources = {f.request.slot: f.source for f in facts}
        for initial in (True, False):
            requests = [f.request for f in facts if f.initial == initial]
            if not requests:
                continue
            banks, limits = {}, {}
            for worker in (compute_workers if initial else prepare_workers):
                if worker.worker_id in occupied:
                    continue
                bank, rows = capacities[0 if initial else 1][worker.worker_id]
                tokens = worker.max_batch_tokens if phase == 'D' else (worker.prefill_max_batch_tokens if initial else worker.verify_max_batch_tokens)
                banks[worker.worker_id] = bank
                limits[worker.worker_id] = (rows, min(worker.bank_blocks, v(bank, 'capacity_blocks', 0)), tokens)
            destinations = [w for w in workers if w.worker_id in banks]
            aggregates = BatchAggregates(phase, initial)
            def fits(worker, batch):
                rows, blocks, tokens = limits[worker.worker_id]
                if len(batch) > rows:
                    return False
                used_blocks, used_tokens = aggregates.get(batch)
                return used_blocks <= blocks and used_tokens <= tokens
            groups = self._plan(cached, phase, requests, destinations, fits, now, decisions,
                initial=initial, estimator=estimator, metrics=metrics)
            for worker in destinations:
                batch = groups.get(worker.worker_id)
                if not batch:
                    self.blocked_workers.add(worker.worker_id)
                    continue
                bank, seq = banks[worker.worker_id], view.sequences[worker.worker_id]
                if phase == 'T':
                    command = builders.prefill(worker, seq, batch, bank) if initial else builders.prepare(cached, worker, seq, batch, bank)
                else:
                    command = self._draft_command(cached, worker, seq, batch, bank, sources, initial)
                decisions.append(command)
                occupied.add(worker.worker_id)
        self.blocked_workers.difference_update(occupied)
        metrics['batches'] = len(decisions)
        metrics['total_ns'] = perf_counter_ns() - start
        return tuple(decisions)

    @staticmethod
    def _cheap(view, stage, r, *, allow_initial=True, allow_prepare=True):
        dispatch = view.row(K.REQUEST_DISPATCH, r.slot)
        target = view.row(K.REQUEST_TARGET_COMPUTE, r.slot)
        if stage == 'T':
            if not v(dispatch, 'target_run_seq', 0):
                return True if allow_initial else None
            if (not allow_prepare or v(dispatch, 'target_round_id') != v(target, 'round_id')
                    or v(target, 'status') != TargetStatus.READY_DRAFT or not v(dispatch, 'draft_issue_seq', 0)
                    or v(dispatch, 'draft_round_id') != v(target, 'round_id', -2) + 1):
                return None
            draft = view.row(K.REQUEST_DRAFT, r.slot)
            return v(draft, 'status') == DraftStatus.READY_TARGET and v(draft, 'round_id') == v(dispatch, 'draft_round_id')
        if not v(dispatch, 'draft_issue_seq', 0):
            return True if allow_initial and r.output_count and v(target, 'status') == TargetStatus.READY_DRAFT else None
        draft = view.row(K.REQUEST_DRAFT, r.slot)
        if (not allow_prepare or v(dispatch, 'draft_round_id') != v(draft, 'round_id')
                or v(draft, 'status') != DraftStatus.READY_TARGET or not v(dispatch, 'target_run_seq', 0)
                or v(dispatch, 'target_round_id') != v(draft, 'round_id')):
            return None
        return v(target, 'status') == TargetStatus.READY_DRAFT and v(target, 'round_id') == v(draft, 'round_id')

    def _plan(self, view, stage, requests, workers, fits, now, decisions, *, initial=False, estimator=None, metrics=None):
        if not requests or not workers:
            return {}
        cap = getattr(self, 'initial_batch_limit', None)
        if initial and cap is not None:
            original = fits
            fits = lambda w, b: len(b) <= cap and original(w, b)
        return plan(view, stage, requests, workers, fits, estimator or self._estimator, now,
            window=BatchWindow(), initial=initial, service_weight=self.service_weights[stage], metrics=metrics,
            time_aligned=True)

    @staticmethod
    def _draft_command(view, worker, sequence, requests, bank, sources, initial):
        if initial:
            rows = tuple(NewRequestData(r.slot, r.epoch,
                U64.next(v(view.row(K.REQUEST_TARGET_COMPUTE, r.slot), 'round_id')), 1,
                scheduled_depth(r), r.prompt, r.output, r.config, ArenaHandle.null()) for r in requests)
            return DraftBatchCommand(worker.worker_id, worker.generation, sequence, rows, (),
                bank=DraftInitialBank(v(bank, 'bank_id'), U64.next(v(bank, 'bank_epoch')),
                    sequence, worker.block_size, tuple(r.capacity_blocks for r in requests)))
        offset, rows = 0, []
        for r in requests:
            identity = sources[r.slot]
            rows.append(DraftPrepareRequest(identity, v(view.row(K.REQUEST_DRAFT, r.slot), 'draft_state_handle'),
                U64.next(v(view.row(K.REQUEST_DISPATCH, r.slot), 'draft_prepare_seq', 0)),
                U64.next(identity.round_id), U64.next(identity.owner_epoch), offset, r.capacity_blocks,
                max(0, r.prompt_count + r.max_new_tokens - 1 - identity.logical_kv_len)))
            offset += r.capacity_blocks
        return PrepareDraftBankCommand(worker.worker_id, worker.generation, sequence, sequence,
            v(bank, 'bank_id'), U64.next(v(bank, 'bank_epoch')), tuple(rows))
