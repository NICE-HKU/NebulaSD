"""Stage symmetry, bounded reconstruction, true fences and measured costs."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from nebulasd.core.enums import StateChangeBlockKind as K, WorkerRole, ComputeStatus, DraftStatus, CopyStatus, H2DStatus, D2HStatus
from nebulasd.scheduler.policy import Scheduler
from nebulasd.scheduler.measured_placement import MeasuredPlacementEstimator
from nebulasd.scheduler.stage_planner import assignment, plan
from nebulasd.scheduler.batching import BatchWindow
from test_draft_placement import migration_view, prepared_view
from test_wp08_scheduler import patch, runnable
from test_grouped_placement import multi


class Table:
    def predict(self, stage, *, batch, kv=0, **kw):
        # Batched compute efficient; copies scale by bytes with stage-specific layouts.
        return SimpleNamespace(seconds=(0.010 + batch * .001 + kv * .00001
            if stage not in ('H2D', 'D2H') else kw['byte_count'] * .000001))


def estimator():
    return MeasuredPlacementEstimator(Table(), 16, draft_block_bytes=4)


def scheduler(**kw):
    return Scheduler(clock=lambda: 1_000_000_000, estimator=kw.pop('estimator', estimator()),
                     target_placement='stagewise', draft_placement='stagewise', **kw)


def drafts(view, e=None):
    return [c for c in scheduler(**({} if e is None else {'estimator': e})).schedule(view)
            if c.kind.name == 'PREPARE_DRAFT_BANK']


def targets(view, e=None):
    return [c for c in Scheduler(clock=lambda:1_000_000_000, target_placement='stagewise',
                                estimator=e or estimator()).schedule(view)
            if c.kind.name == 'PREPARE_TARGET_BANK']


def issued_target_view(count=4):
    view = multi(count)
    for r in view.requests.values():
        patch(view, K.REQUEST_DISPATCH, r.slot, draft_issue_seq=1, draft_round_id=1,
              draft_worker_id=0, draft_worker_generation=1)
        patch(view, K.REQUEST_DRAFT, r.slot, request_epoch=1, round_id=1, observed_issue_seq=1,
              worker_id=0, worker_generation=1, status=DraftStatus.READY_TARGET, result_code=0)
    return view


def test_draft_chooses_predicted_finish_and_merges_predecessors():
    view = migration_view(4)
    # Distinct predecessor Bank identities are deliberately irrelevant.
    for r in view.requests.values():
        patch(view, K.REQUEST_TARGET_COMPUTE, r.slot, bank_epoch=r.slot+1)
        patch(view, K.REQUEST_DRAFT_D2H, r.slot, request_epoch=1, snapshot_version=1,
              source_op_seq=1, owner_epoch=0, source_worker_generation=1,
              status=D2HStatus.HOST_READY, ready_version=1)
    patch(view, K.WORKER_DRAFT_COPY_RUNTIME, 0, copy_status=CopyStatus.H2D,
          copy_start_time_ns=1_000_000_000, copy_bytes=1_000_000)
    cmds = drafts(view)
    assert [(c.worker_id, len(c.requests)) for c in cmds] == [(1, 4)]
    assert drafts(view) == cmds


def test_busy_target_can_beat_idle_target_on_completion_cost():
    view = issued_target_view(1)
    # Explicit prediction isolates the planner's ordering from calibration noise.
    class Costs(MeasuredPlacementEstimator):
        def stage_prediction(self, view, stage, worker, requests, now_ns, **kw):
            return {'cost_s': len(requests) * (0.01 if worker.worker_id == 2 else 1.0)}
    patch(view, K.WORKER_TARGET_COMPUTE_RUNTIME, 2, compute_status=ComputeStatus.RUNNING)
    view.inflight[2] = object()
    assert targets(view, Costs(Table(), 16, 4))[0].worker_id == 2


def test_draft_batch_splits_to_target_capacity_and_oversize_skips():
    view = issued_target_view(6)
    view = replace(view, workers=tuple(replace(w, max_batch_size=2) for w in view.workers))
    cmds = targets(view)
    assert sorted(len(c.requests) for c in cmds) == [2, 2, 2]
    assert sorted(r.request_slot for c in cmds for r in c.requests) == list(range(6))
    view.requests[0] = replace(view.requests[0], capacity_blocks=129)
    assert sorted(r.request_slot for c in targets(view) for r in c.requests) == list(range(1, 6))


def test_late_member_split_avoids_head_of_line_wait():
    view = issued_target_view(2)
    class Late(MeasuredPlacementEstimator):
        def input_ready(self, view, stage, request, now):
            return 1.0 if request.slot == 0 else 2.0
    cmds = targets(view, Late(Table(), 16, 4))
    assert sorted(len(c.requests) for c in cmds) == [1, 1]


def test_predecessor_must_be_really_issued_both_directions():
    view = migration_view()
    patch(view, K.REQUEST_DISPATCH, 0, target_round_id=0)
    assert not drafts(view)
    patch(view, K.REQUEST_DISPATCH, 0, target_round_id=1)
    patch(view, K.REQUEST_TARGET_COMPUTE, 0, round_id=0)  # not completed
    assert drafts(view)
    view = issued_target_view(1)
    patch(view, K.REQUEST_DISPATCH, 0, draft_issue_seq=0)
    assert not targets(view)  # proposals selected in this call are not issued facts
    patch(view, K.REQUEST_DISPATCH, 0, draft_issue_seq=1)
    patch(view, K.REQUEST_DRAFT, 0, status=DraftStatus.IN_DRAFT)
    assert targets(view)


@pytest.mark.parametrize('stage', ['D', 'T'])
def test_immutable_prepares_and_real_run_fences(stage):
    view = prepared_view() if stage == 'D' else runnable()
    p = next(iter(view.prepared.values()))
    policy = scheduler() if stage == 'D' else Scheduler(target_placement='stagewise', estimator=estimator())
    run_kind = 'RUN_DRAFT_BATCH' if stage == 'D' else 'RUN_TARGET_BATCH'
    assert any(c.kind.name == run_kind for c in policy.schedule(view))
    patch(view, K.REQUEST_DRAFT_H2D if stage == 'D' else K.REQUEST_H2D, 1, status=H2DStatus.WAIT_HOST)
    cmds = policy.schedule(view)
    assert not any(c.kind.name == run_kind for c in cmds)
    assert not any(c.worker_id == p.worker_id and c.kind.name.startswith('PREPARE') for c in cmds)
    assert view.prepared[p.worker_id] == p


def test_stage_specific_layout_and_four_clock_equation():
    view = migration_view(2)
    e = estimator()
    rs = list(view.requests.values())
    for stage, w in [('D', view.workers[0]), ('T', view.workers[2])]:
        p = e.stage_prediction(view, stage, w, rs, 1_000_000_000,
                               input_times={0: 1.0, 1: 1.02})
        assert p['start_s'] == max(p[k] for k in ('input_ready_s', 'worker_ready_s', 'kv_ready_s'))
        assert p['finish_s'] == p['start_s'] + p['compute_s']
        assert p['cost_s'] == pytest.approx(2*p['finish_s'] - 2.02)
    other = replace(e, draft_block_bytes=400)
    assert other.stage_prediction(view, 'D', view.workers[0], rs, 1_000_000_000)['kv_ready_s'] > e.stage_prediction(view, 'D', view.workers[0], rs, 1_000_000_000)['kv_ready_s']
    assert other.stage_prediction(view, 'T', view.workers[2], rs, 1_000_000_000) == e.stage_prediction(view, 'T', view.workers[2], rs, 1_000_000_000)


def test_matching_is_not_greedy_and_is_deterministic():
    workers = [SimpleNamespace(worker_id=i) for i in range(2)]
    costs = ((1, 2), (1, 100))
    predict = lambda w, b: {'cost_s': costs[b[0]][w.worker_id]}
    result = assignment(((0,), (1,)), workers, lambda *a: True, predict)
    assert result == (3, {1: [0], 0: [1]})


def test_capacity_shared_rows_window_and_fair_admission():
    view = migration_view(2)
    patch(view, K.WORKER_DRAFT_BANK, 0, alloc_rows=8)
    assert all(c.worker_id == 1 for c in drafts(view))
    patch(view, K.WORKER_DRAFT_BANK, 2, alloc_rows=8)
    assert not drafts(view)
    view = migration_view(1)
    assert not scheduler(window=BatchWindow(2_000_000_000)).schedule(view)
    assert drafts(view)  # default window is work-conserving
    view = migration_view(2)
    view.requests[0] = replace(view.requests[0], ready_ns=999_999_999)
    groups = plan(view, 'D', list(view.requests.values()), [view.workers[0]],
                  lambda w, b: len(b) <= 1, estimator(), 1_000_000_000, window=BatchWindow())
    assert groups[0][0].slot == 1  # old waiting work before a newly-ready round


def test_banked_run_worker_clock_and_migrated_target_input():
    view = prepared_view()
    command = next(c for c in scheduler().schedule(view) if c.kind.name == 'RUN_DRAFT_BATCH')
    view.inflight[command.worker_id] = command
    patch(view, K.WORKER_DRAFT_RUNTIME, command.worker_id, compute_status=ComputeStatus.RUNNING,
          current_batch_seq=command.expected_batch_seq, compute_start_time_ns=1_000_000_000)
    assert estimator()._worker_free(view, view.workers[command.worker_id], 1_000_000_001) > 1.0
    # Issued Target destination can differ from the still-published previous source.
    patch(view, K.REQUEST_DISPATCH, 0, planned_target_id=99)
    class Free(MeasuredPlacementEstimator):
        def _worker_free(self, view, worker, now): return float(worker.worker_id)
    view = replace(view, workers=(*view.workers, replace(view.workers[2], worker_id=99)))
    patch(view, K.REQUEST_TARGET_COMPUTE, 0, round_id=0)
    assert Free(Table(), 16, 4).input_ready(view, 'D', view.requests[0], 1_000_000_000) == 99


def test_stagewise_configuration_and_independent_layout_setup(tmp_path, monkeypatch):
    from nebulasd.config import NebulaSDConfig, HostKVLayout
    from nebulasd.engine import cost_setup
    from test_cost_table import payload
    import json
    with pytest.raises(ValueError, match='cost_model=table'):
        NebulaSDConfig(draft_placement='stagewise')
    with pytest.raises(ValueError, match='stage-aware'):
        Scheduler(target_placement='stagewise')
    with pytest.raises(ValueError, match='draft_block_bytes'):
        Scheduler(draft_placement='stagewise', estimator=MeasuredPlacementEstimator(Table(), 16))
    path = tmp_path / 'cost.json'
    path.write_text(json.dumps(payload()))
    config = NebulaSDConfig(draft_placement='stagewise', target_placement='stagewise',
                             cost_model='table', backend_cost_table=str(path), max_proposal_depth=4)
    monkeypatch.setattr(cost_setup, 'cost_identity', lambda *a, **kw: {'device': 'test'})
    e = cost_setup.make_estimator(config, HostKVLayout(1024, 'torch.float16'), HostKVLayout(128, 'torch.float16'))
    assert e.block_bytes == 1024 and e.draft_block_bytes == 128


def test_draft_current_dirty_copy_and_queued_prepare_are_counted_once():
    view = migration_view(1)
    patch(view, K.REQUEST_DRAFT, 0, dirty_block_count=1)
    patch(view, K.REQUEST_DRAFT_D2H, 0, request_epoch=1, snapshot_version=1, source_op_seq=1,
          owner_epoch=0, source_worker_generation=1, status=D2HStatus.IN_D2H)
    patch(view, K.WORKER_DRAFT_COPY_RUNTIME, 0, copy_status=CopyStatus.D2H,
          copy_start_time_ns=1_000_000_000, copy_bytes=8)
    e = estimator()
    assert e._draft_kv_ready(view, view.workers[1], list(view.requests.values()), 1.0) == pytest.approx(1 + 16e-6)
    view = prepared_view()
    p = next(iter(view.prepared.values()))
    assert e._copy_free(view, p.worker_id, 1.0) == 1.0
    patch(view, K.REQUEST_DRAFT_H2D, 0, status=H2DStatus.WAIT_HOST)
    assert e._copy_free(view, p.worker_id, 1.0) == pytest.approx(1 + 8e-6)


def test_matching_relocation_fills_newly_available_legal_bank():
    view = migration_view(2)
    class Costs(MeasuredPlacementEstimator):
        def stage_prediction(self, view, stage, w, batch, now, **kw):
            return {'cost_s': 1 if w.worker_id == 1 else 2}
    # FIFO seed admits slot 0 on the large Bank, leaving no fit for slot 1.
    # Matching moves slot 0 to the small Bank; slot 1 must use the freed Bank.
    def fits(w, batch):
        return len(batch) == 1 and (batch[0].slot == 0 or w.worker_id == 0)
    groups = plan(view, 'D', list(view.requests.values()), view.workers[:2], fits,
                  Costs(Table(), 16, 4), 1_000_000_000, window=BatchWindow())
    assert {w: [r.slot for r in b] for w, b in groups.items()} == {1: [0], 0: [1]}


def test_stage_capacity_recomputed_after_bank_and_request_changes():
    view = migration_view(2)
    policy = scheduler()
    expected = policy.schedule(view)
    assert any(c.kind.name == 'PREPARE_DRAFT_BANK' for c in expected)
    saved = dict(view.rows)
    for worker in view.workers[:2]:
        patch(view, K.WORKER_DRAFT_BANK, worker.worker_id * 2, alloc_rows=8)
    assert not any(c.kind.name == 'PREPARE_DRAFT_BANK' for c in policy.schedule(view))
    view.rows.update(saved)
    assert policy.schedule(view) == expected
    for slot, request in tuple(view.requests.items()):
        view.requests[slot] = replace(request, capacity_blocks=129)
    assert not any(c.kind.name == 'PREPARE_DRAFT_BANK' for c in policy.schedule(view))
