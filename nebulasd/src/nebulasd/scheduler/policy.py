"""Global Target placement and bounded Draft Bank placement from stable facts."""

from time import perf_counter_ns

from nebulasd.core.enums import BankRole, BankState, ComputeStatus, WorkerRole, StateChangeBlockKind as K
from . import builders
from .batching import BatchWindow, fit
from .eligibility import active, online, ScheduleEligibility, draft_ready, restored
from .placement import PlacementEstimator
from .views import value as v


class Scheduler:
    def __init__(self, *, clock=perf_counter_ns, estimator=None, window=None, target_placement='adaptive', draft_placement='worker_id', execution_mode='legacy'):
        if target_placement not in ('adaptive', 'fixed', 'batch_adaptive', 'stagewise'):
            raise ValueError('unknown Target placement policy')
        if draft_placement not in ('worker_id', 'round_robin', 'stagewise'):
            raise ValueError('unknown Draft placement policy')
        if execution_mode not in ('legacy', 'event', 'event_early'):
            raise ValueError('unknown scheduler execution mode')
        self.execution_mode = execution_mode
        self._target_placement = target_placement
        self._draft_placement = draft_placement
        self._clock = clock
        self._estimator = estimator or PlacementEstimator()
        self._window = window or BatchWindow()
        if 'stagewise' in (target_placement, draft_placement):
            if not hasattr(self._estimator, 'stage_prediction'):
                raise ValueError('stagewise requires a stage-aware measured estimator')
            if draft_placement == 'stagewise' and getattr(self._estimator, 'draft_block_bytes', None) is None:
                raise ValueError('stagewise Draft requires explicit draft_block_bytes')

    @property
    def requires_clock_ticks(self):
        return self._window.delay_ns > 0

    def schedule(self, view, *, phase=None):
        if phase not in (None, "ready", "D", "T"):
            raise ValueError("unknown scheduler phase")
        now = self._clock()
        eligibility = ScheduleEligibility(view)
        if self._draft_placement == 'stagewise' and any(w.role == WorkerRole.DRAFT and not w.draft_banked for w in view.workers):
            raise ValueError('stagewise Draft requires banked workers')
        candidates = [] if phase == "ready" else sorted((r for r in view.requests.values() if active(view, r)),
                            key=lambda r: (r.arrival_seq, r.slot))
        workers = tuple(w for w in view.workers if online(view, w))
        # Use the configured topology, never the current online subset: a missing
        # owner must not silently remap fixed-affinity requests. Slot allocation
        # assigns consecutive slots, providing round-robin ownership per cohort.
        targets = sorted(w.worker_id for w in view.workers if w.role == WorkerRole.TARGET)
        def permits(request, worker):
            return self._target_placement != 'fixed' or worker.worker_id == targets[request.slot % len(targets)]
        sequences = dict(view.sequences)
        decisions, used_slots = [], set()

        def emit(command):
            decisions.append(command)
            sequences[command.worker_id] += 1

        if phase in (None, "ready"):
            # Already-frozen batches run as a whole, in their original order.
            for worker in workers:
                if worker.role != WorkerRole.TARGET or worker.worker_id in view.inflight:
                    continue
                prepared = view.prepared.get(worker.worker_id)
                if prepared is None:
                    continue
                active_bank = self._bank(view, worker, BankRole.ACTIVE)
                bank = view.row(K.WORKER_BANK, worker.worker_id * 2 + prepared.standby_bank_id)
                ready = (v(bank, "state") == BankState.READY and v(bank, "bank_epoch") == prepared.next_bank_epoch
                         and v(bank, "batch_seq") == prepared.batch_seq)
                if ready and self._compute_idle(view, worker) and all(
                    item.request_slot in view.requests
                    and draft_ready(view, view.requests[item.request_slot], item.round_id)
                    and restored(view, view.requests[item.request_slot], prepared, item)
                    for item in prepared.requests
                ):
                    emit(builders.run(view, worker, sequences[worker.worker_id], prepared, active_bank))

        if phase in (None, "ready", "D"):
            from .draft_placement import schedule as schedule_draft
            def draft_plan(requests, destinations, fits, initial, draft_decisions):
                return self._plan(view, 'D', requests, destinations, fits, now,
                                  [*decisions, *draft_decisions], initial=initial)
            decisions.extend(schedule_draft(view, candidates, workers, sequences, placement=self._draft_placement,
                target_result_for=eligibility.target_result, phase=phase,
                planner=draft_plan if self._draft_placement == 'stagewise' else None))

            if phase != "ready":
                # Explicit legacy WorkerSpec injection retains its non-banked protocol.
                draft_workers = sorted((w for w in workers if w.role == WorkerRole.DRAFT and not w.draft_banked),
                    key=lambda w: (self._estimator.draft_worker_score(view, w, now), w.worker_id))
                for worker in draft_workers:
                    if worker.role != WorkerRole.DRAFT or worker.worker_id in view.inflight:
                        continue
                    eligible = []
                    for r in candidates:
                        result = eligibility.target_result(r)
                        dispatch = view.row(K.REQUEST_DISPATCH, r.slot)
                        issued = v(dispatch, "draft_issue_seq", 0)
                        if (r.slot not in used_slots and result is not None
                                and (not issued or v(dispatch, "draft_worker_id") == worker.worker_id)
                                and (not issued or v(dispatch, "draft_round_id", 0) <= v(result, "round_id"))):
                            eligible.append(r)
                    batch = fit(eligible, max_rows=worker.max_batch_size, max_tokens=worker.max_batch_tokens,
                                max_blocks=1 << 60, token_cost=lambda r: r.proposal_depth + (
                                    r.prompt_count + r.output_count if not v(view.row(K.REQUEST_DISPATCH, r.slot), "draft_issue_seq", 0) else 1))
                    if self._allow(batch, worker, now):
                        emit(builders.draft(view, worker, sequences[worker.worker_id], batch))
                        used_slots.update(r.slot for r in batch)

        if phase in (None, "T"):
            # Fresh requests use an idle active Bank; no pretend HostKV before prefill.
            assigned = set()
            initial_targets = None
            if self._target_placement == 'stagewise':
                destinations = [w for w in workers if w.role == WorkerRole.TARGET
                    and w.worker_id not in view.inflight and w.worker_id not in view.prepared
                    and self._compute_idle(view, w) and self._reusable(self._bank(view, w, BankRole.ACTIVE))]
                initial_targets = self._plan(view, 'T', [r for r in candidates
                    if not v(view.row(K.REQUEST_DISPATCH, r.slot), 'target_run_seq', 0)], destinations,
                    lambda w, batch: self._fit_target(batch, w, self._bank(view, w, BankRole.ACTIVE), prefill=True,
                        other_rows=self._other_rows(view, w, self._bank(view, w, BankRole.ACTIVE))) == list(batch),
                    now, decisions, initial=True)
            for worker in workers:
                if worker.role != WorkerRole.TARGET or worker.worker_id in view.inflight or worker.worker_id in view.prepared:
                    continue
                bank = self._bank(view, worker, BankRole.ACTIVE)
                if not self._compute_idle(view, worker) or not self._reusable(bank):
                    continue
                fresh = [r for r in candidates if r.slot not in assigned
                         and permits(r, worker)
                         and not v(view.row(K.REQUEST_DISPATCH, r.slot), "target_run_seq", 0)]
                batch = (self._fit_target(fresh, worker, bank, prefill=True, other_rows=self._other_rows(view, worker, bank))
                         if initial_targets is None else initial_targets.get(worker.worker_id, []))
                if self._allow(batch, worker, now):
                    emit(builders.prefill(worker, sequences[worker.worker_id], batch, bank))
                    assigned.update(r.slot for r in batch)

            # Placement is selected once. A sent prepare cannot be overwritten.
            occupied = set(view.prepared) | {c.worker_id for c in decisions if c.kind.name == "TARGET_PREFILL_BATCH"}
            prepared_slots = {r.request_slot for c in view.prepared.values() if c.kind.name == "PREPARE_TARGET_BANK" for r in c.requests}
            if self._target_placement == 'stagewise':
                destinations = [w for w in workers if w.role == WorkerRole.TARGET and w.worker_id not in occupied
                    and v(self._bank(view, w, BankRole.STANDBY), 'state') in (BankState.EMPTY, BankState.DRAINING)]
                groups = self._plan(view, 'T', [r for r in candidates if r.slot not in prepared_slots
                    and eligibility.target_result(r) is not None and eligibility.draft_issued(r)], destinations,
                    lambda w, batch: self._fit_target(batch, w, self._bank(view, w, BankRole.STANDBY),
                        other_rows=self._other_rows(view, w, self._bank(view, w, BankRole.STANDBY))) == list(batch),
                    now, decisions)
            elif self._target_placement == 'batch_adaptive':
                from .grouped_placement import place_batches
                groups = place_batches(self, view,
                    [r for r in candidates if r.slot not in prepared_slots], workers, occupied, now, decisions)
            else:
                groups = {}
                for r in candidates:
                    if r.slot in prepared_slots or eligibility.target_result(r) is None:
                        continue
                    choices = []
                    for worker in workers:
                        if worker.role != WorkerRole.TARGET or worker.worker_id in occupied:
                            continue
                        if not permits(r, worker):
                            continue
                        bank = self._bank(view, worker, BankRole.STANDBY)
                        if bank is not None and v(bank, "state") in (BankState.EMPTY, BankState.DRAINING):
                            group = groups.get(worker.worker_id, [])
                            if self._fit_target(group + [r], worker, bank, other_rows=self._other_rows(view, worker, bank)) == group + [r]:
                                choices.append((self._placement_score(view, worker, group + [r], now, decisions, legacy_request=r), len(group), worker.worker_id, worker, bank))
                    if choices:
                        _, _, _, worker, bank = min(choices, key=lambda x: x[:3])
                        groups.setdefault(worker.worker_id, []).append(r)
            for worker in workers:
                batch = groups.get(worker.worker_id, [])
                if self._allow(batch, worker, now):
                    emit(builders.prepare(view, worker, sequences[worker.worker_id], batch,
                                          self._bank(view, worker, BankRole.STANDBY)))
        return tuple(decisions)

    def _plan(self, view, stage, requests, workers, fits, now, decisions, *, initial=False):
        if not requests or not workers:
            return {}
        from dataclasses import replace
        from .stage_planner import plan
        inflight = dict(view.inflight)
        inflight.update({c.worker_id: c for c in decisions if c.kind.name in
            ('DRAFT_BATCH', 'RUN_DRAFT_BATCH', 'RUN_TARGET_BATCH', 'TARGET_PREFILL_BATCH')})
        return plan(replace(view, inflight=inflight), stage, requests, workers, fits,
                    self._estimator, now, window=self._window, initial=initial)

    def _placement_score(self, view, worker, requests, now, decisions, legacy_request=None):
        """Score a full provisional batch without changing dispatch state.

        Include compute commands selected in THIS call, which are not in the sent
        ledger yet. Backpressure may delay them, so the estimate is a ranking
        hypothesis only. Capacity and execution fences remain in schedule().
        """
        if hasattr(self._estimator, 'batch_score'):
            from dataclasses import replace
            inflight = dict(view.inflight)
            inflight.update({c.worker_id:c for c in decisions if c.kind.name in ('DRAFT_BATCH','RUN_DRAFT_BATCH','RUN_TARGET_BATCH','TARGET_PREFILL_BATCH')})
            return self._estimator.batch_score(replace(view,inflight=inflight),worker,requests,now)
        if legacy_request is not None:
            return self._estimator.score(view,worker,legacy_request,now)
        return max(self._estimator.score(view,worker,r,now) for r in requests)

    def _allow(self, batch, worker, now):
        return bool(batch) and self._window.allow(now_ns=now,
            oldest_ready_ns=min(r.ready_ns or r.admitted_ns for r in batch), full=len(batch) == worker.max_batch_size)

    @staticmethod
    def _bank(view, worker, role):
        return next((view.row(K.WORKER_BANK, worker.worker_id * 2 + i) for i in (0, 1)
                     if v(view.row(K.WORKER_BANK, worker.worker_id * 2 + i), "role") == role), None)

    @staticmethod
    def _compute_idle(view, worker):
        row = view.row(K.WORKER_TARGET_COMPUTE_RUNTIME, worker.worker_id)
        # Runtime/Bank partitions share their worker's CommonBlock generation.
        return online(view, worker) and v(row, "compute_status") == ComputeStatus.IDLE

    @staticmethod
    def _reusable(bank):
        return v(bank, "state") == BankState.EMPTY or (v(bank, "state") == BankState.DRAINING and v(bank, "alloc_rows") == 0)

    @staticmethod
    def _other_rows(view, worker, bank):
        # Canonical block-table row IDs are shared by both Banks. A standby
        # allocation cannot consume IDs still leased by the active Bank.
        other = view.row(K.WORKER_BANK, worker.worker_id * 2 + (1 - v(bank, "bank_id")))
        return v(other, "alloc_rows", 0)

    @staticmethod
    def _fit_target(candidates, worker, bank, prefill=False, other_rows=0):
        return fit(candidates, max_rows=max(0, min(worker.max_batch_size, v(bank, "capacity_rows", 0) - other_rows)),
            max_tokens=worker.prefill_max_batch_tokens if prefill else worker.verify_max_batch_tokens,
            max_blocks=v(bank, "capacity_blocks", 0),
            token_cost=lambda r: r.prompt_count if prefill else r.proposal_depth + 1)
