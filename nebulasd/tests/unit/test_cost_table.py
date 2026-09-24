"""Measured estimates rank candidates without weakening scheduler fences."""

from dataclasses import replace
import copy
import pytest
from nebulasd.scheduler.cost_table import CostTable
from nebulasd.scheduler.measured_placement import MeasuredPlacementEstimator
from nebulasd.scheduler.policy import Scheduler
from nebulasd.core.enums import *
from nebulasd.ipc.protocol import (
    DraftBatchCommand,
    PrepareTargetBankCommand,
    RunTargetBatchCommand,
)
from test_wp08_scheduler import world, patch, runnable
from test_grouped_placement import multi

K = StateChangeBlockKind


def payload():
    rows = []
    for stage, depth, sync in [
        ("draft_first", 4, 0),
        ("draft_cached", 4, 1),
        ("draft_cached", 4, 2),
        ("target_verify", 4, 0),
        ("target_prefill", 0, 0),
        ("H2D", 0, 0),
        ("D2H", 0, 0),
    ]:
        for batch in (1, 8):
            for kv in (1, 128):
                rows.append(
                    dict(
                        stage=stage,
                        batch=batch,
                        kv=kv,
                        depth=depth,
                        sync=sync,
                        bytes=kv,
                        samples=20,
                        boundary="test completed wall",
                        p50_ms=batch + kv,
                        p95_ms=2 * (batch + kv),
                    )
                )
    return dict(schema_version=1, compatibility={"device": "test"}, rows=rows)


def estimator():
    return MeasuredPlacementEstimator(CostTable(payload(), expected={"device": "test"}), 1)


def test_validation_interpolation_and_explicit_fallbacks(tmp_path):
    data = payload()
    table = CostTable(data, expected=data["compatibility"])
    assert table.predict("target_verify", batch=4, kv=50, depth=4).seconds == pytest.approx(0.054)
    assert table.predict("target_verify", batch=1, kv=1, depth=4).method == "exact"
    assert "depth_upper_bound" in table.predict("target_verify", batch=1, kv=1, depth=2).method
    assert (
        table.predict("target_verify", batch=32, kv=2048, depth=4).method
        == "out_of_range_work_scaling"
    )
    for args in [("unknown", 4, 0), ("target_verify", 8, 0), ("draft_cached", 4, 3)]:
        with pytest.raises(ValueError, match="uncalibrated"):
            table.predict(args[0], batch=1, depth=args[1], sync=args[2])
    with pytest.raises(ValueError, match="compatibility"):
        CostTable(data, expected={"device": "other"})
    for key, value in [("samples", 19), ("p50_ms", float("nan")), ("p95_ms", 0.1), ("batch", 0)]:
        bad = copy.deepcopy(data)
        bad["rows"][0][key] = value
        with pytest.raises(ValueError):
            CostTable(bad, expected=data["compatibility"])
    data["rows"] = [
        r
        for r in data["rows"]
        if not (r["stage"] == "target_verify" and r["batch"] == 1 and r["kv"] == 128)
    ]
    assert (
        CostTable(data, expected=data["compatibility"])
        .predict("target_verify", batch=4, kv=50, depth=4)
        .method
        == "sparse_upper_shape"
    )


def test_remaining_worker_time_uses_matching_worker_start_and_clamps():
    view = world()
    e = estimator()
    command = next(c for c in Scheduler().schedule(view) if isinstance(c, DraftBatchCommand))
    view.inflight[0] = command
    worker = view.workers[0]
    duration = e._draft_duration(view, list(view.requests.values()), {0, 1})
    patch(
        view,
        K.WORKER_DRAFT_RUNTIME,
        0,
        compute_status=ComputeStatus.RUNNING,
        current_batch_seq=command.command_seq,
        compute_start_time_ns=1_000_000_000,
    )
    assert e._worker_free(view, worker, 1_001_000_000) == pytest.approx(1 + duration)
    assert e._worker_free(view, worker, 2_000_000_000) == 2
    patch(view, K.WORKER_DRAFT_RUNTIME, 0, current_batch_seq=command.command_seq + 1)
    assert e._worker_free(view, worker, 2_000_000_000) == pytest.approx(2 + duration)


def test_whole_batch_sums_copy_bytes_and_overlaps_draft():
    view = world()
    e = estimator()
    w = view.workers[2]
    for slot in view.requests:
        patch(view, K.REQUEST_D2H, slot, status=D2HStatus.HOST_READY, ready_version=1)
    one = e.batch_prediction(view, w, [view.requests[0]], 1_000_000_000)
    both = e.batch_prediction(view, w, list(view.requests.values()), 1_000_000_000)
    assert both["h2d_bytes"] == 2 * one["h2d_bytes"] == 4
    assert both["kv_ready_s"] > one["kv_ready_s"]
    assert both["start_s"] == max(both[k] for k in ("draft_ready_s", "kv_ready_s", "target_free_s"))
    # A long member must govern the batch shape, never the mean KV length.
    view.requests[1] = replace(view.requests[1], prompt_count=100)
    assert e._duration("target_verify", list(view.requests.values())) == pytest.approx(0.102)


@pytest.mark.parametrize("policy", ["fixed", "adaptive", "batch_adaptive"])
def test_measured_estimator_keeps_frozen_order_fences_and_recycle(policy):
    view = runnable()
    scheduler = Scheduler(target_placement=policy, estimator=estimator())
    run = next(c for c in scheduler.schedule(view) if isinstance(c, RunTargetBatchCommand))
    assert [i.request_slot for i in run.requests] == [
        i.request_slot for i in view.prepared[2].requests
    ]
    patch(view, K.REQUEST_DRAFT, 1, status=DraftStatus.IN_DRAFT)
    assert not any(isinstance(c, RunTargetBatchCommand) for c in scheduler.schedule(view))
    patch(view, K.REQUEST_DRAFT, 1, status=DraftStatus.READY_TARGET, request_epoch=2)
    assert not any(isinstance(c, RunTargetBatchCommand) for c in scheduler.schedule(view))


@pytest.mark.parametrize("policy", ["fixed", "adaptive", "batch_adaptive"])
def test_measured_capacity_and_fixed_ownership(policy):
    view = multi(6)
    view = replace(view, workers=tuple(replace(w, max_batch_size=2) for w in view.workers))
    commands = [
        c
        for c in Scheduler(target_placement=policy, estimator=estimator()).schedule(view)
        if isinstance(c, PrepareTargetBankCommand)
    ]
    assert commands and all(len(c.requests) <= 2 for c in commands)
    if policy == "fixed":
        assert all(c.worker_id == 2 + i.request_slot % 3 for c in commands for i in c.requests)
    else:
        assert sum(len(c.requests) for c in commands) == 6


def test_known_fifo_queue_adds_service_before_later_candidate():
    view = world(4)
    e = estimator()
    w = replace(view.workers[0], max_batch_size=2)
    early = e._draft_queue_duration(view, w, [view.requests[0]], set())
    late = e._draft_queue_duration(view, w, [view.requests[3]], set())
    assert late == pytest.approx(2 * early)
    patch(view, K.REQUEST_ENGINE, 0, lifecycle=Lifecycle.FINISHED)
    patch(view, K.REQUEST_ENGINE, 1, lifecycle=Lifecycle.FINISHED)
    assert e._draft_queue_duration(view, w, [view.requests[3]], set()) == pytest.approx(early)


def test_pending_copy_and_completed_busy_clock_are_distinct():
    view = runnable()
    e = estimator()
    w = view.workers[2]
    assert e._copy_free(view, 2, 1.0) == 1.0
    patch(view, K.REQUEST_H2D, 1, status=H2DStatus.WAIT_HOST)
    assert e._copy_free(view, 2, 1.0) > 1.0
    command = next(c for c in Scheduler().schedule(world()) if isinstance(c, DraftBatchCommand))
    view.inflight[0] = command
    patch(
        view,
        K.WORKER_DRAFT_RUNTIME,
        0,
        compute_status=ComputeStatus.IDLE,
        current_batch_seq=command.command_seq,
        compute_start_time_ns=999_000_000,
    )
    assert e._worker_free(view, view.workers[0], 1_000_000_000) == 1.0


def test_cold_setup_rejects_incompatible_dtype_and_missing_family(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from nebulasd.engine import cost_setup
    from nebulasd.config import NebulaSDConfig
    from nebulasd.scheduler.placement import PlacementEstimator

    assert NebulaSDConfig().cost_model == "legacy"
    with pytest.raises(ValueError):
        NebulaSDConfig(cost_model="table")
    monkeypatch.setattr(cost_setup, "cost_identity", lambda *a, **kw: {"device": "test"})
    path = tmp_path / "cost.json"
    data = payload()
    path.write_text(json.dumps(data))
    config = SimpleNamespace(
        cost_model="table",
        backend_cost_table=str(path),
        draft_model_path="",
        target_model_path="",
        devices=(0,),
        block_size=16,
        gpu_memory_fraction=0.90,
        max_proposal_depth=4,
    )
    layout = SimpleNamespace(dtype="torch.float16", block_bytes=1)
    assert isinstance(cost_setup.make_estimator(config, layout), MeasuredPlacementEstimator)
    with pytest.raises(ValueError, match="FP16"):
        cost_setup.make_estimator(config, SimpleNamespace(dtype="torch.bfloat16", block_bytes=1))
    data["rows"] = [r for r in data["rows"] if r["stage"] != "draft_cached"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="uncalibrated"):
        cost_setup.make_estimator(config, layout)
    config.cost_model = "legacy"
    assert isinstance(cost_setup.make_estimator(config, layout), PlacementEstimator)


@pytest.mark.parametrize("limit", ["tokens", "blocks", "rows"])
def test_measured_batch_split_keeps_other_capacity_limits(limit):
    view = multi(6)
    if limit == "tokens":
        view = replace(
            view, workers=tuple(replace(w, verify_max_batch_tokens=6) for w in view.workers)
        )
    for w in view.workers[2:]:
        if limit == "blocks":
            patch(view, K.WORKER_BANK, w.worker_id * 2 + 1, capacity_blocks=8)
        if limit == "rows":
            patch(view, K.WORKER_BANK, w.worker_id * 2 + 1, capacity_rows=8)
            patch(view, K.WORKER_BANK, w.worker_id * 2, alloc_rows=6)
    commands = [
        c
        for c in Scheduler(target_placement="batch_adaptive", estimator=estimator()).schedule(view)
        if isinstance(c, PrepareTargetBankCommand)
    ]
    assert sorted(len(c.requests) for c in commands) == [2, 2, 2]
    assert sorted(i.request_slot for c in commands for i in c.requests) == list(range(6))


def test_current_dirty_copy_is_not_charged_twice():
    view = world(1)
    e = estimator()
    worker = view.workers[2]
    patch(view, K.REQUEST_TARGET_COMPUTE, 0, bank_id=0, bank_epoch=1, dirty_block_count=4)
    patch(
        view,
        K.REQUEST_D2H,
        0,
        request_epoch=1,
        round_id=0,
        d2h_op_seq=1,
        source_bank_id=0,
        source_bank_epoch=1,
        status=D2HStatus.IN_D2H,
    )
    patch(
        view,
        K.WORKER_TARGET_COPY_RUNTIME,
        2,
        copy_status=CopyStatus.D2H,
        copy_start_time_ns=1_000_000_000,
        copy_bytes=8,
    )
    prediction = e.batch_prediction(view, worker, [view.requests[0]], 1_000_000_000)
    assert prediction["source_dirty_bytes"] == {2: 0}
    assert prediction["kv_ready_s"] == pytest.approx(1 + e._copy("D2H", 8) + e._copy("H2D", 2))
