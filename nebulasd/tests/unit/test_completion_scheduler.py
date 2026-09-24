"""Bounded planning, completion identity, stage isolation and joint matching."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from nebulasd.core.enums import StateChangeBlockKind as K, BankState, BankRole, ComputeStatus, CopyStatus, DraftStatus
from nebulasd.ipc.command_arena import CommandBackpressure
from nebulasd.observability.latency import LatencySamples
from nebulasd.scheduler.completion import CompletionScheduler
from nebulasd.scheduler.measured_placement import MeasuredPlacementEstimator
from nebulasd.scheduler.stage_planner import plan
from nebulasd.scheduler.batching import BatchWindow
from test_stage_planner import estimator, issued_target_view
from test_draft_placement import migration_view, prepared_view
from test_wp08_scheduler import patch, runnable


def policy(**kwargs):
    return CompletionScheduler(clock=lambda: 1_000_000_000, estimator=kwargs.pop('estimator', estimator()), **kwargs)




















@pytest.mark.parametrize('stage', ['D', 'T'])
def test_near_ready_requests_can_prepare_before_predecessor_finishes(stage):
    view = migration_view(2) if stage == 'D' else issued_target_view(2)
    if stage == 'D':
        patch(view, K.REQUEST_TARGET_COMPUTE, 0, round_id=0)
        patch(view, K.REQUEST_TARGET_COMPUTE, 1, round_id=0)
    else:
        from nebulasd.core.enums import DraftStatus
        for r in view.requests.values():
            patch(view, K.REQUEST_DRAFT, r.slot, status=DraftStatus.IN_DRAFT)
    commands = policy().schedule(view, phase=stage, destinations={0 if stage == 'D' else 2})
    assert commands and sum(len(c.requests) for c in commands) == 2


@pytest.mark.parametrize('stage', ['D', 'T'])
def test_frontier_and_full_prediction_work_do_not_grow_beyond_bound(stage, monkeypatch):
    from nebulasd.scheduler import completion
    original_source = completion.source
    sources = []
    def source_once(view, request):
        sources.append(request.slot)
        return original_source(view, request)
    monkeypatch.setattr(completion, 'source', source_once)
    counts = []
    for n in (64, 128, 256, 512):
        view = migration_view(n) if stage == 'D' else issued_target_view(n)
        scheduler = policy()
        sources.clear()
        commands = scheduler.schedule(view, phase=stage, destinations={0 if stage == 'D' else 2})
        m = scheduler.last_metrics
        assert m['scanned'] == n
        assert m['frontier'] <= m['frontier_limit']
        assert m['frontier_limit'] == 2 * (4 if stage == 'D' else 8)
        assert commands
        assert len(sources) == len(set(sources)) <= m['frontier_limit']
        counts.append((m['prediction_cells'], m['fit_checks']))
    assert len(set(counts)) == 1


def test_oversized_global_prefix_does_not_hide_legal_frontier_and_cache_is_fresh():
    view = migration_view(64)
    for slot in range(60):
        view.requests[slot] = replace(view.requests[slot], capacity_blocks=129)
    scheduler = policy()
    commands = scheduler.schedule(view, phase='D', destinations={0})
    assert [r.request_slot for c in commands for r in c.requests] == list(range(60, 64))
    assert scheduler.last_metrics['frontier'] == 4
    patch(view, K.WORKER_DRAFT_BANK, 1, capacity_blocks=1)
    assert not scheduler.schedule(view, phase='D', destinations={0})


def test_ready_frozen_run_never_scans_global_candidates(monkeypatch):
    scheduler = policy()
    monkeypatch.setattr(scheduler, '_cheap', lambda *a: pytest.fail('ready scanned candidates'))
    commands = scheduler.schedule(prepared_view(), phase='ready')
    assert commands and all(c.kind.name.startswith('RUN_') for c in commands)


class Curve:
    def __init__(self, expensive_large=False):
        self.expensive_large = expensive_large
        self.queries = 0

    def predict(self, stage, *, batch, **kwargs):
        self.queries += 1
        if stage in ('H2D', 'D2H'):
            seconds = .000001
        elif self.expensive_large and stage == 'target_verify':
            seconds = .009 if batch > 8 else .002
        else:
            seconds = .003 if batch > 8 else .002
        return SimpleNamespace(seconds=seconds)


def curve_plan(stage, table, weight):
    view = migration_view(16)
    requests = list(view.requests.values())
    workers = tuple(replace(w, max_batch_size=16) for w in view.workers[:2])
    return plan(view, stage, requests, workers, lambda w, b: len(b) <= 16,
        MeasuredPlacementEstimator(table, 1, 1), 1_000_000_000,
        window=BatchWindow(), service_weight=weight)


def test_service_rewards_saved_time_and_uses_existing_prediction():
    table = Curve()
    without = curve_plan('D', table, 0)
    calls = table.queries
    table.queries = 0
    with_service = curve_plan('D', table, 32)
    assert sorted(map(len, without.values())) == [8, 8]
    assert sorted(map(len, with_service.values())) == [16]
    assert table.queries == calls  # Weight adds no model queries.


def test_large_batch_not_always_preferred_and_stages_use_different_profiles():
    table = Curve(expensive_large=True)
    assert sorted(map(len, curve_plan('D', table, 32).values())) == [16]
    assert sorted(map(len, curve_plan('T', table, 32).values())) == [8, 8]


def test_configuration_is_explicit_and_legacy_defaults_stay_compatible():
    from nebulasd.config import NebulaSDConfig
    assert NebulaSDConfig().scheduler_execution == 'legacy'
    with pytest.raises(ValueError, match='completion requires'):
        NebulaSDConfig(scheduler_execution='completion')
    config = NebulaSDConfig(scheduler_execution='completion', draft_placement='stagewise',
        target_placement='stagewise', cost_model='table', backend_cost_table='explicit.json',
        draft_service_weight=24, target_service_weight=8)
    assert (config.draft_service_weight, config.target_service_weight) == (24, 8)
    for kwargs in ({'frontier_factor': 0}, {'draft_service_weight': float('nan')}, {'target_service_weight': -1}):
        with pytest.raises(ValueError):
            policy(**kwargs)


def next_cohort_view(stage):
    """Two frozen runnable members and four unrelated next-round candidates."""
    view = migration_view(6) if stage == 'D' else issued_target_view(6)
    ready = prepared_view() if stage == 'D' else runnable()
    view.rows.update(ready.rows)
    view.prepared.update(ready.prepared)
    command = policy().schedule(view, phase='ready')[0]
    for item in command.requests:
        if stage == 'D':
            patch(view, K.REQUEST_DISPATCH, item.request_slot, draft_round_id=item.round_id,
                  draft_issue_seq=item.run_seq)
        else:
            patch(view, K.REQUEST_DISPATCH, item.request_slot, target_round_id=item.round_id,
                  target_run_seq=item.run_seq)
    # Next cohort has an issued, unfinished predecessor.
    for slot in range(2, 6):
        if stage == 'D':
            patch(view, K.REQUEST_TARGET_COMPUTE, slot, round_id=0)
        else:
            patch(view, K.REQUEST_DRAFT, slot, status=DraftStatus.IN_DRAFT)
    return view, command


def publish_running_bank(view, command, stage, state=BankState.EMPTY):
    wid = command.worker_id
    kind = K.WORKER_DRAFT_BANK if stage == 'D' else K.WORKER_BANK
    patch(view, kind, wid * 2 + command.active_bank_id, role=BankRole.ACTIVE, state=BankState.COMPUTING,
          bank_epoch=command.active_bank_epoch, batch_seq=command.expected_batch_seq, alloc_rows=len(command.requests))
    patch(view, kind, wid * 2 + 1 - command.active_bank_id, role=BankRole.STANDBY, state=state, alloc_rows=0)
    runtime = K.WORKER_DRAFT_RUNTIME if stage == 'D' else K.WORKER_TARGET_COMPUTE_RUNTIME
    field = 'current_batch_seq' if stage == 'D' else 'compute_batch_seq'
    patch(view, runtime, wid, compute_status=ComputeStatus.RUNNING, compute_start_time_ns=1_000_000_000,
          **{field: command.expected_batch_seq})


@pytest.mark.parametrize('stage', ['D', 'T'])
@pytest.mark.parametrize('state', [BankState.EMPTY, BankState.DRAINING])
def test_running_destination_prepares_near_ready_cohort_with_shared_rows(stage, state):
    view, run = next_cohort_view(stage)
    view.prepared.clear()
    view.inflight[run.worker_id] = run
    publish_running_bank(view, run, stage, state)
    scheduler = policy()
    commands = scheduler.schedule(view, phase=stage, destinations={run.worker_id})
    assert len(commands) == 1
    assert commands[0].kind.name == ('PREPARE_DRAFT_BANK' if stage == 'D' else 'PREPARE_TARGET_BANK')
    assert {r.request_slot for r in commands[0].requests} == {2, 3, 4, 5}
    kind = K.WORKER_DRAFT_BANK if stage == 'D' else K.WORKER_BANK
    patch(view, kind, run.worker_id * 2 + run.active_bank_id, alloc_rows=8)
    assert not scheduler.schedule(view, phase=stage, destinations={run.worker_id})


@pytest.mark.parametrize('stage', ['D', 'T'])
def test_stale_ready_and_initial_requests_cannot_hide_busy_prepare_frontier(stage):
    view, run = next_cohort_view(stage)
    view = replace(view, workers=tuple(replace(w, max_batch_size=2) for w in view.workers))
    view.prepared.clear()
    view.inflight[run.worker_id] = run
    publish_running_bank(view, run, stage)
    for slot in (4, 5):
        patch(view, K.REQUEST_DISPATCH, slot, **{('draft_issue_seq' if stage == 'D' else 'target_run_seq'): 0})
    scheduler = policy()
    commands = scheduler.schedule(view, phase=stage, destinations={run.worker_id})
    assert len(commands) == 1 and {r.request_slot for r in commands[0].requests} == {2, 3}
    assert scheduler.last_metrics['frontier'] == 2




@pytest.mark.parametrize('stage', ['D', 'T'])
def test_running_worker_cannot_run_even_with_all_frozen_inputs_ready(stage):
    view = prepared_view() if stage == 'D' else runnable()
    run = policy().schedule(view, phase='ready')[0]
    kind = K.WORKER_DRAFT_RUNTIME if stage == 'D' else K.WORKER_TARGET_COMPUTE_RUNTIME
    patch(view, kind, run.worker_id, compute_status=ComputeStatus.RUNNING)
    assert not policy().schedule(view, phase='ready')
    view.inflight[run.worker_id] = run
    assert not policy().schedule(view, phase='ready')














def aligned_plan(view, times, free, calls=None):
    calls = calls if calls is not None else {'input': [], 'worker': []}
    class Timing(MeasuredPlacementEstimator):
        def input_ready(self, view, stage, request, now):
            calls['input'].append(request.slot)
            return times[request.slot]
        def _worker_free(self, view, worker, now):
            calls['worker'].append(worker.worker_id)
            return free[worker.worker_id]
    workers = [w for w in view.workers if w.worker_id in free]
    return plan(view, 'D', list(view.requests.values()), workers,
        lambda w, b: len(b) <= w.max_batch_size,
        Timing(Curve(), 1, 1).for_invocation(), 1_000_000_000,
        window=BatchWindow(), service_weight=32, time_aligned=True)


def test_frontier_time_admission_precedes_row_bound_not_fifo():
    view = migration_view(8)
    calls = {'input': [], 'worker': []}
    groups = aligned_plan(view, {i: 1.060 if i < 4 else 1.002 for i in range(8)}, {0: 1.0}, calls)
    assert [r.slot for r in groups[0]] == [4, 5, 6, 7]
    assert sorted(calls['input']) == list(range(8))
    assert calls['worker'] == [0]  # Includes all reconstruction prediction cells.


def test_admission_leaves_straggler_out_and_service_cannot_merge_cohorts():
    times = dict(enumerate((1.002, 1.0023, 1.003, 1.060)))
    view = migration_view(4)
    groups = aligned_plan(view, times, {0: 1.0})
    assert [r.slot for r in groups[0]] == [0, 1, 2]
    groups = aligned_plan(view, times, {0: 1.0, 1: 1.0})
    assert sorted(sorted(r.slot for r in b) for b in groups.values()) == [[0, 1, 2], [3]]


def test_time_admission_aligns_to_destination_and_hidden_spread_is_free():
    view = migration_view(6)
    times = {i: 1.0 if i < 3 else 1.050 for i in range(6)}
    groups = aligned_plan(view, times, {0: 1.050, 1: 1.0})
    assert [r.slot for r in groups[1]] == [0, 1, 2]
    assert [r.slot for r in groups[0]] == [3, 4, 5]
    view = migration_view(4)
    groups = aligned_plan(view, dict(enumerate((1.0, 1.01, 1.02, 1.03))), {0: 1.050})
    assert len(groups[0]) == 4  # Every input will be ready before compute frees.


def test_slack_never_waits_for_more_work_and_ready_age_prevents_starvation():
    view = migration_view(1)
    assert aligned_plan(view, {0: 1.060}, {0: 1.0})[0]  # Lone issued predecessor.
    assert aligned_plan(view, {0: 1.0}, {0: 1.0})[0]  # Idle worker sends immediately.
    view = migration_view(8)
    times = {i: 1.060 if i == 0 else 1.0 for i in range(8)}
    assert all(r.slot != 0 for r in aligned_plan(view, times, {0: 1.0})[0])
    times[0] = 1.0  # Only the currently issued predecessor, now completed.
    for slot in range(1, 8):
        view.requests[slot] = replace(view.requests[slot], ready_ns=999_999_999)
    assert aligned_plan(view, times, {0: 1.0})[0][0].slot == 0
