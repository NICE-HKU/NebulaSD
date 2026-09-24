"""Per-destination top-B selection and single-pass filtering of batching delays.

Shares only eligibility/capacity/command plumbing with CompletionScheduler.
See docs/current/SCHEDULER.md for the current policy contract.
"""
from math import isfinite
from itertools import groupby
from time import perf_counter_ns

from nebulasd.core.enums import StateChangeBlockKind as K
from .completion import CompletionScheduler
from .views import value as v

TIME_TOLERANCE_S = 1e-9


def filter_batch(batch, predict, bounds, delay_s):
    """One priority-ordered pass: reject additions that delay retained members."""
    batch = list(batch)
    if not batch:
        return []
    kept = batch[:1]
    limit = bounds[kept[0].slot] + delay_s
    for request in batch[1:]:
        next_limit = min(limit, bounds[request.slot] + delay_s)
        trial = kept + [request]
        if predict(trial)['start_s'] <= next_limit + TIME_TOLERANCE_S:
            kept = trial
            limit = next_limit
    return kept


class ServiceIntervalScheduler(CompletionScheduler):
    service_interval = True

    def __init__(self, *, draft_service_gap_ms, target_service_gap_ms, service_batch_delay_ms=30.0, **kwargs):
        gaps = (draft_service_gap_ms, target_service_gap_ms, service_batch_delay_ms)
        if any(isinstance(g, bool) or not isinstance(g, (int, float)) or not isfinite(g) or g < 0 for g in gaps):
            raise ValueError('service gaps must be explicit finite nonnegative milliseconds')
        super().__init__(**kwargs)
        self.service_gaps_s = dict(zip(('D', 'T'), (g / 1000 for g in gaps[:2])))
        self.service_batch_delay_s = service_batch_delay_ms / 1000

    def schedule(self, view, **kwargs):
        if view.work_state is None:
            raise ValueError('service_interval requires autonomous WORK scheduling facts')
        return super().schedule(view, **kwargs)

    def _plan(self, view, stage, requests, workers, fits, now, decisions, *, initial=False,
              estimator=None, metrics=None):
        if stage == 'T' and initial:
            return super()._plan(view, stage, requests, workers, fits, now, decisions,
                                 initial=True, estimator=estimator, metrics=metrics)
        if not requests or not workers:
            return {}
        started = perf_counter_ns()
        estimator = estimator or self._estimator
        actual, times, ends = {}, {}, {}
        for r in requests:
            dispatch = view.row(K.REQUEST_DISPATCH, r.slot)
            predecessor_stage, field = ('T', 'target_round_id') if stage == 'D' else ('D', 'draft_round_id')
            end = self.compute_times.end_ns(r.slot, r.epoch, predecessor_stage, v(dispatch, field))
            actual[r.slot] = end
            if end is not None:
                ends[r.slot] = end / 1e9
                times[r.slot] = max(now / 1e9, end / 1e9)
        # Actual wait orders groups without a profile lookup. Only equal-age
        # groups reached before top-B fills need destination-specific matching.
        def wait_group(r):
            end = actual[r.slot]
            return end is None, end if end is not None else 0
        age_groups = [list(group) for _, group in groupby(sorted(requests, key=wait_group), wait_group)]
        groups, used = {}, set()
        for worker in sorted(workers, key=lambda w: w.worker_id):
            predictions = {}
            def predict(batch):
                key = tuple(sorted(r.slot for r in batch))
                if key not in predictions:
                    for r in batch:
                        if r.slot not in times:
                            ends[r.slot] = estimator.input_ready(view, stage, r, now)
                            times[r.slot] = max(now / 1e9, ends[r.slot])
                    predictions[key] = estimator.stage_prediction(view, stage, worker, batch, now,
                        input_times={r.slot: times[r.slot] for r in batch}, initial=initial)
                    metrics['prediction_cells'] += 1
                return predictions[key]
            def capacity(batch):
                metrics['fit_checks'] += 1
                cap = getattr(self, 'initial_batch_limit', None)
                return (not initial or cap is None or len(batch) <= cap) and fits(worker, batch)
            free = estimator.worker_ready(view, worker, now)
            def order(r):
                delta = max(0.0, predict([r])['start_s'] - free)
                return delta, r.ready_ns or r.admitted_ns, r.arrival_seq, r.slot
            batch = []
            for group in age_groups:
                ranked = sorted((r for r in group if r.slot not in used and capacity([r])), key=order)
                for r in ranked:
                    if capacity(batch + [r]):
                        batch.append(r)
                    if len(batch) == worker.max_batch_size:
                        break
                if len(batch) == worker.max_batch_size:
                    break
            metrics['frontier'] += len(batch)
            if not batch:
                continue
            bounds = {r.slot: max(ends[r.slot] + self.service_gaps_s[stage],
                                  predict([r])['start_s']) for r in batch}
            batch = filter_batch(batch, predict, bounds, self.service_batch_delay_s)
            groups[worker.worker_id] = batch
            used.update(r.slot for r in batch)
        metrics['placement_ns'] += perf_counter_ns() - started
        return groups
