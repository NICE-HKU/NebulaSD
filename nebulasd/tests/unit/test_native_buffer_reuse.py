"""Reused native inputs must behave like a fresh bridge across transitions."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.engine.work_ledger import Record
from nebulasd.scheduler.native_completion import NativeCompletion
from nebulasd.workers.work import WorkKind
from test_native_completion import make
from test_wp08_scheduler import patch


def test_record_layout_reuse_and_replacement():
    view, scheduler = make('T', 8)
    native = NativeCompletion(scheduler._estimator)

    def work(seq, slots):
        return NS(worker_id=2, operation=WorkKind.TARGET_VERIFY, work_seq=seq,
                  rows=tuple(NS(slot=s, capacity_blocks=s+1, source=object() if s%2 else None)
                             for s in slots))

    a, b = Record(work(1, (0, 1, 2))), Record(work(2, (3, 4)))
    live = {1: a, 2: b}
    native.sync_records(live)
    records, members = native.records, native.members
    assert list(members) == [0, 1, 2, 3, 4]
    assert (records[0].h2d_blocks, records[0].h2d_rows, records[0].d2h_blocks) == (2, 1, 6)
    a.compute_done = True
    b.physical_done = True
    native.sync_records(live)
    assert native.records is records and native.members is members
    assert [(r.compute_done, r.physical_done) for r in records] == [(1, 0), (0, 1)]

    # Same count, new order; then same key/sequence, new frozen WORK.
    native.sync_records({2: b, 1: a})
    assert list(native.members) == [3, 4, 0, 1, 2]
    a.work = work(1, (7,))
    native.sync_records({2: b, 1: a})
    assert list(native.members) == [3, 4, 7]
    assert [r.count for r in native.records] == [2, 1]
    native.sync_records({1: a})
    assert len(native.records) == 1 and list(native.members) == [7]
    native.sync_records({})
    assert len(native.records) == len(native.members) == 0
    native.sync_records({1: a})
    assert list(native.members) == [7]


@pytest.mark.parametrize('service', [False, True])
def test_reused_bridge_matches_fresh_across_phase_and_capacity_changes(service, monkeypatch):
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL', 'check')
    view, scheduler = make('T', 16)
    view = replace(view, workers=tuple(replace(w, draft_banked=True) for w in view.workers))
    view.work_state.resources.specs = view.workers
    for w in view.workers:
        kind = K.WORKER_DRAFT_BANK if w.role.name == 'DRAFT' else K.WORKER_BANK
        for bank in (0, 1):
            patch(view, kind, w.worker_id*2+bank, bank_id=bank, bank_epoch=0,
                  capacity_blocks=128, state=0)
    for slot in view.requests:
        patch(view, K.REQUEST_DISPATCH, slot, target_run_seq=0)
    if service:
        from nebulasd.scheduler.service_interval import ServiceIntervalScheduler
        scheduler = ServiceIntervalScheduler(estimator=scheduler._estimator, clock=lambda: 1_000_000_000,
                                             draft_service_gap_ms=0, target_service_gap_ms=0)
    saved_plan = None

    def compare(phase, **kwargs):
        nonlocal saved_plan
        actual = scheduler.schedule(view, phase=phase, **kwargs)
        reused = scheduler._native
        prediction = dict(scheduler.native_predictions)
        blocked = set(scheduler.blocked_workers)
        scheduler._native = NativeCompletion(scheduler._estimator)
        try:
            expected = scheduler.schedule(view, phase=phase, **kwargs)
            assert actual == expected
            assert prediction == scheduler.native_predictions
            assert blocked == scheduler.blocked_workers
        finally:
            scheduler._native = reused
        if saved_plan is None and actual:
            saved_plan = (actual, repr(actual))
        if saved_plan:
            assert repr(saved_plan[0]) == saved_plan[1]
        return reused

    native = compare('T')
    assert saved_plan is not None
    workers, outputs, slots = native.workers, native.outputs, native.slots
    compare('D')
    compare('T', destinations=set())  # Must not expose prior outputs or capacity.
    assert native.workers is workers and native.outputs is outputs and native.slots is slots
    assert all(r.initial_rows == r.prepare_rows == 0 for r in native.worker_rows)
    for w in view.workers:
        patch(view, K.WORKER_COMMON, w.worker_id, status=0)
    compare('T')
    assert all(r.online == 0 for r in native.worker_rows)
    for w in view.workers:
        patch(view, K.WORKER_COMMON, w.worker_id, status=1)

    # Membership shrinks and grows; buffers retain capacity but candidate count does not.
    original = dict(view.requests)
    view.requests.clear()
    compare('D')
    compare('T')
    view.requests.update(list(original.items())[:2])
    compare('T')
    view.requests.update(original)
    compare('T')
    assert native.slots is slots
    scheduler.initial_batch_limit = 1
    for slot in view.requests:
        patch(view, K.REQUEST_DISPATCH, slot, target_run_seq=0)
    compare('T')
    assert all(r.initial_rows <= 1 for r in native.worker_rows)

    # Generation, configuration and ordering invalidate static Worker fields.
    view = replace(view, workers=tuple(replace(w, generation=w.generation+1, max_batch_size=2)
                                       for w in reversed(view.workers)))
    for w in view.workers:
        patch(view, K.WORKER_COMMON, w.worker_id, worker_generation=w.generation)
    compare('T')
    assert native.workers is not workers
    assert [(r.id, r.generation, r.max_batch) for r in native.worker_rows] == [
        (w.worker_id, w.generation, 2) for w in view.workers]

    view = replace(view, workers=view.workers[:1])
    compare('T')
    assert len(native.workers) == 1 and native.outputs is outputs

    view, _ = make('T', 32)
    compare('T')
    assert len(native.slots) >= 32 and native.slots is not slots
