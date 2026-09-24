"""Stateless previous-Bank-batch locality before immutable prepare issuance."""

from nebulasd.core.enums import BankRole, BankState, WorkerRole, StateChangeBlockKind as K
from .eligibility import target_result
from .views import value as v


def related_batches(view, candidates):
    """Group only live, fenced results, never round numbers or cached members.

    A Target Bank incarnation holds one batch: generation/bank/epoch distinguish
    reuse and migrations. Missing identity fields conservatively form singletons.
    No cohort survives this call; terminal, stale and not-yet-eligible peers do
    not delay candidates. Draft readiness is still a Run fence, not a new wait
    before prepare, preserving the existing Draft/H2D overlap.
    """
    groups = {}
    for request in candidates:
        result = target_result(view, request)
        if result is None:
            continue
        identity = tuple(v(result, name) for name in
                         ('target_id', 'target_generation', 'bank_id', 'bank_epoch'))
        key = identity if None not in identity else ('single', request.slot, request.epoch)
        groups.setdefault(key, []).append(request)
    return tuple(groups.values())


def place_batches(policy, view, candidates, workers, occupied, now, decisions=()):
    """Work-conserving heuristic, not a calibrated latency/copy cost model.

    FIFO groups choose idle before busy destinations, then largest legal subset,
    then the estimator score, then fewer migrations. Legacy uses maximum member
    ready time; the optional table uses whole-batch finish time. Thus an
    idle partial fit beats a busy whole fit; within one availability class a
    split costs more than any predicted migration saving. The selected estimator only breaks equal-size ties. This deliberately has no ms-valued
    claim about fragmentation cost, and may lose useful parallelism.

    Every choice reuses cumulative token/block/shared-row fitting. Remainders
    immediately try all remaining capacity; oversized items cannot hide smaller
    work. Only one group list per Target is emitted by the caller, and frozen
    prepares are excluded before entry. No new persistent state or IPC.
    """
    destinations = []
    for worker in workers:
        if worker.role != WorkerRole.TARGET or worker.worker_id in occupied:
            continue
        bank = policy._bank(view, worker, BankRole.STANDBY)
        if bank is not None and v(bank, 'state') in (BankState.EMPTY, BankState.DRAINING):
            destinations.append((worker, bank, policy._other_rows(view, worker, bank)))
    placed = {}
    for cohort in related_batches(view, candidates):
        remaining = cohort
        while remaining:
            choices = []
            for worker, bank, other_rows in destinations:
                current = placed.get(worker.worker_id, [])
                fitted = policy._fit_target(current + remaining, worker, bank, other_rows=other_rows)
                addition = fitted[len(current):]
                if not addition:
                    continue
                busy = worker.worker_id in view.inflight or not policy._compute_idle(view, worker)
                score = policy._placement_score(view, worker, fitted, now, decisions)
                migrations = sum(v(view.row(K.REQUEST_TARGET_COMPUTE, r.slot), 'target_id')
                                 != worker.worker_id for r in addition)
                choices.append(((busy, -len(addition), score, migrations, worker.worker_id), worker, addition))
            if not choices:
                break
            _, worker, addition = min(choices, key=lambda choice: choice[0])
            placed.setdefault(worker.worker_id, []).extend(addition)
            chosen = {r.slot for r in addition}
            remaining = [r for r in remaining if r.slot not in chosen]
    return placed
