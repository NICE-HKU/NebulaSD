"""One-stage, bounded reconstruction and minimum request-completion cost.

Completion admission aligns issued predecessor times to destination availability;
other callers retain FIFO admission. Optimization compares the SAME admitted
requests, so dropping a costly request cannot masquerade as an improvement.
Exact subset DP for <=8 workers, deterministic 64-state beam above that size.
No plan persists: only issued immutable commands survive a scheduler call.
"""
from time import perf_counter_ns
from .reconstruction import partitions, READY_SLACK_S


def assignment(batches, workers, fits, predict, *, beam=64):
    if len(batches) > len(workers):
        return None
    # Construct each batch x worker cell once, including all four clocks.
    matrix = [[predict(w, b)['cost_s'] if fits(w, b) else None for w in workers] for b in batches]
    states = {0: (0.0, ())}
    for costs in matrix:
        next_states = {}
        for mask, (cost, path) in states.items():
            for index, cell in enumerate(costs):
                if cell is None or mask & (1 << index):
                    continue
                key = mask | (1 << index)
                candidate = (cost + cell, path + (index,))
                if key not in next_states or candidate < next_states[key]:
                    next_states[key] = candidate
        if len(workers) > 8:
            next_states = dict(sorted(next_states.items(), key=lambda item: item[1])[:beam])
        states = next_states
    if not states:
        return None
    cost, path = min(states.values())
    return cost, {workers[i].worker_id: list(batch) for i, batch in zip(path, batches)}


def plan(view, stage, candidates, workers, fits, estimator, now, *, window, initial=False, service_weight=0.0, metrics=None,
         time_aligned=False):
    started = perf_counter_ns() if metrics is not None else 0
    prediction_before = 0 if metrics is None else metrics["placement_ns"]
    assignment_before = 0 if metrics is None else metrics["assignment_ns"]
    # Matching can move a seed batch onto a smaller destination, freeing a
    # larger Bank for an otherwise skipped request. Fill those unused slots in
    # this SAME stage, with at most one pass per worker (no future queue plan).
    # One immutable view, stage and request set per plan(). Keep batch order in
    # the key: capacity callbacks need not be permutation-invariant. Reuse only
    # within this call, including relocation passes; no Bank facts persist.
    capacity = {}
    original_fits = fits
    def fits(worker, batch):
        key = worker.worker_id, tuple(r.slot for r in batch)
        if key not in capacity:
            capacity[key] = original_fits(worker, batch)
            if metrics is not None:
                metrics["fit_checks"] += 1
        return capacity[key]

    times, free = None, None
    if time_aligned and candidates and workers:
        prediction_started = perf_counter_ns() if metrics is not None else 0
        # candidates is Completion's <=K frontier, never the global request set.
        # Cache once across admission, reconstruction and relocation passes.
        times = {r.slot: estimator.input_ready(view, 'target_prefill' if initial and stage == 'T' else stage, r, now)
                 for r in candidates}
        free = {w.worker_id: estimator.worker_ready(view, w, now) for w in workers}
        if metrics is not None:
            metrics['placement_ns'] += perf_counter_ns() - prediction_started
    groups = {}
    remaining = list(candidates)
    destinations = list(workers)
    while remaining and destinations:
        selected = _plan_once(view, stage, remaining, destinations, fits, estimator,
                              now, window=window, initial=initial, service_weight=service_weight, metrics=metrics,
                              input_times=times, worker_times=free)
        if not selected:
            break
        groups.update(selected)
        slots = {r.slot for batch in selected.values() for r in batch}
        remaining = [r for r in remaining if r.slot not in slots]
        destinations = [w for w in destinations if w.worker_id not in selected]
    if metrics is not None:
        metrics["reconstruction_ns"] += (perf_counter_ns() - started
            - (metrics["placement_ns"] - prediction_before) - (metrics["assignment_ns"] - assignment_before))
    return groups


def _aligned_seed(candidates, workers, fits, times, free):
    """One current cohort per destination; no timer or future queue simulation.

    All inputs ready by worker availability are equivalent. Age breaks that
    tie; otherwise pick the earliest legal issued predecessor and its 2ms
    cohort. A lone legal request always seeds a batch, even outside the slack.
    As an issued predecessor finishes it enters the ready/age-prioritized set,
    so slack cannot indefinitely defer old work behind newly ready requests.
    """
    seed, used = {}, set()
    for w in sorted(workers, key=lambda w: (free[w.worker_id], w.worker_id)):
        available = free[w.worker_id]
        batch, high = [], None
        ordered = sorted((r for r in candidates if r.slot not in used),
            key=lambda r: (max(available, times[r.slot]), r.ready_ns or r.admitted_ns, r.arrival_seq, r.slot))
        for r in ordered:
            ready = max(available, times[r.slot])
            if high is not None and ready > high:
                break
            if fits(w, batch + [r]):
                if high is None:
                    high = ready + READY_SLACK_S
                batch.append(r)
                used.add(r.slot)
                if len(batch) == w.max_batch_size:
                    break
        if batch:
            seed[w.worker_id] = batch
    return seed


def _plan_once(view, stage, candidates, workers, fits, estimator, now, *, window, initial=False, service_weight=0.0, metrics=None,
               input_times=None, worker_times=None):
    workers = sorted(workers, key=lambda w: w.worker_id)
    if not workers or not candidates:
        return {}
    # A legal seed bounds admission to sum(max_batch_size) and provides a
    # work-conserving fallback. Completion may admit less to avoid frozen HOL.
    if input_times is not None:
        seed = _aligned_seed(candidates, workers, fits, input_times, worker_times)
        admitted = [r for batch in seed.values() for r in batch]
    else:
        seed = {w.worker_id: [] for w in workers}
        admitted = []
        row_bound = sum(w.max_batch_size for w in workers)
        for r in sorted(candidates, key=lambda r: (r.ready_ns or r.admitted_ns, r.arrival_seq, r.slot)):
            if len(admitted) == row_bound:
                break
            for w in workers:
                if len(seed[w.worker_id]) == w.max_batch_size:
                    continue
                if fits(w, seed[w.worker_id] + [r]):
                    seed[w.worker_id].append(r)
                    admitted.append(r)
                    break
    if not admitted:
        return {}
    prediction_started = perf_counter_ns() if metrics is not None else 0
    times = input_times if input_times is not None else {
        r.slot: estimator.input_ready(view, 'target_prefill' if initial and stage == 'T' else stage, r, now)
        for r in admitted}
    if metrics is not None:
        metrics["placement_ns"] += perf_counter_ns() - prediction_started
    predictions = {}
    def predict(w, batch):
        key = w.worker_id, tuple(sorted(r.slot for r in batch))
        if key not in predictions:
            started = perf_counter_ns() if metrics is not None else 0
            prediction = estimator.stage_prediction(view, stage, w, batch, now,
                input_times={r.slot: times[r.slot] for r in batch}, initial=initial)
            # Reuse the measured duration already used in finish prediction.
            # Keep zero-weight callers compatible with their cost-only estimators.
            predictions[key] = dict(prediction, cost_s=prediction['cost_s'] +
                (service_weight * prediction['compute_s'] if service_weight else 0.0))
            if metrics is not None:
                metrics['placement_ns'] += perf_counter_ns() - started
                metrics['prediction_cells'] += 1
        return predictions[key]
    def allowed(w, batch):
        # Keep the service reward from recombining separated cohorts into a
        # single frozen batch. Release spread already hidden by compute is free.
        aligned = (worker_times is None or max(times[r.slot] for r in batch)
            <= max(worker_times[w.worker_id], min(times[r.slot] for r in batch)) + READY_SLACK_S)
        return aligned and fits(w, batch) and window.allow(now_ns=now,
            oldest_ready_ns=min(r.ready_ns or r.admitted_ns for r in batch),
            full=len(batch) == w.max_batch_size)
    # Seed is always available when the bounded batching window has expired.
    best = None
    menu = (tuple(tuple(b) for b in seed.values() if b),
            *partitions(admitted, workers, fits, times))
    seen = set()
    for batches in menu:
        identity = tuple(tuple(r.slot for r in b) for b in batches)
        if identity in seen:
            continue
        seen.add(identity)
        started = perf_counter_ns() if metrics is not None else 0
        prediction_before = 0 if metrics is None else metrics['placement_ns']
        result = assignment(batches, workers, allowed, predict)
        if metrics is not None:
            metrics['partitions'] += 1
            metrics['assignment_ns'] += perf_counter_ns() - started - (metrics['placement_ns'] - prediction_before)
        if result is None:
            continue
        cost, groups = result
        tie = tuple((wid, tuple(r.slot for r in batch)) for wid, batch in sorted(groups.items()))
        key = cost, tie
        if best is None or key < best[0]:
            best = key, groups
    if best is not None:
        return best[1]
    # A partial window must not prevent full/expired batches from proceeding.
    return {wid: batch for wid, batch in seed.items() if batch and
            allowed(next(w for w in workers if w.worker_id == wid), batch)}
