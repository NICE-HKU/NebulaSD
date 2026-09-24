"""Validate timing semantics and sample identity without CUDA or model imports."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from nebulasd.kv.transfer import (
    CopyExecutor,
    CopyPlan,
    CopyRegion,
    CopyReceipt,
    _h2d_chunk_bytes_from_env,
    _plan_chunks,
)
from support.copy_measurements import summarize_copy_receipts
from support.gpu_timeline import GPUTimeline
from support.wp07_config import parse_args, args_from_env


def test_pytest_and_cli_share_complete_defaults():
    config = args_from_env({"STARSD_NEXT_TARGET_MODEL_PATH": "/model"}, "/output")
    cli = parse_args(["--target-model", "/model", "--output", "/output", "--rounds", "4",
                      "--warmup-rounds", "1", "--prompt-tokens", "31"])
    assert vars(config) == vars(cli)
    assert (config.request_capacity_blocks, config.draft_workers, config.copy_poll_us) == (8, 1, 100)


def test_h2d_chunking_is_opt_in_by_default(monkeypatch):
    monkeypatch.delenv("STARSD_H2D_CHUNK_BYTES", raising=False)
    assert _h2d_chunk_bytes_from_env() == 0


@pytest.mark.parametrize("flags", [["--copy-poll-us", "nan"], ["--owner-idle-us", "-1"],
    ["--require-overlap"], ["--rounds", "128"], ["--draft-workers", "0"]])
def test_invalid_experiment_settings_fail_before_loading_cuda(flags):
    with pytest.raises(ValueError):
        parse_args(["--target-model", "/model", "--output", "/output", *flags])


def make_plan(round_id=2):
    extent = SimpleNamespace(capacity_blocks=8, request_slot=0)
    return CopyPlan("D2H", (CopyRegion(extent, 0, 0, 1),), round_ids=(round_id,))


def wait_for(predicate):
    end = monotonic() + 2
    while not predicate():
        if monotonic() > end:
            raise AssertionError("condition did not become true")
        sleep(0.001)


def test_receipt_separates_launch_observation_owner_and_physical_estimate():
    cold, plan = make_plan(0), make_plan(2)
    receipt = CopyReceipt(2000, 5000, 0.2, ready_publish_started_ns=7000,
        ready_fact_published_ns=8000, ready_published_ns=8500, enqueued_ns=1000,
        launch_returned_ns=3000, last_pending_ns=3500, retired_ns=9000)
    target = SimpleNamespace(world=SimpleNamespace(arena=SimpleNamespace(descriptor=SimpleNamespace(block_bytes=16))),
        receipts=[(cold, replace(receipt, duration_ms=999)), (plan, receipt)])
    gpu = dict(copy_id=id(plan), kind="D2H", device=0, slots=[0], start_ns=3100, end_ns=4000)
    peer = dict(kind="target_verify", device=0, slots=[1], start_ns=6000, end_ns=16000)
    report = summarize_copy_receipts([target], [gpu, peer], {0: 20}, 1)
    summary = report["summary"]["D2H"]
    assert summary["samples"] == 1 and summary["cuda_dma"]["p50_ms"] == 0.2
    assert summary["observed_to_owner"]["p50_ms"] == 0.002
    assert summary["owner_to_ready_fact"]["p50_ms"] == 0.001
    assert summary["gpu_end_to_observed_estimate"]["p50_ms"] == 0.001
    assert summary["post_ready_cleanup"]["p50_ms"] == 0.001
    assert report["rows"][1]["clock_estimate_consistent"]
    assert report["rows"][1]["enqueue_to_next_independent_verify_gpu_estimate_ms"] == 0.005
    assert not report["rows"][0]["measured"]


def test_executor_records_enqueue_launch_return_and_observation_in_order():
    queried, complete = Event(), Event()
    class Ticket:
        def query(self):
            queried.set()
            return complete.is_set()
        def duration_ms(self):
            return 0.1
    backend = SimpleNamespace(launch=lambda plan: Ticket(), close=lambda: None)
    executor = CopyExecutor(backend)
    try:
        future = executor.submit(make_plan())
        assert queried.wait(2)
        complete.set()
        receipt = future.result(2)
        assert receipt.enqueued_ns <= receipt.submitted_ns <= receipt.launch_returned_ns <= receipt.completed_ns
        assert receipt.submitted_ns <= receipt.last_pending_ns <= receipt.completed_ns
    finally:
        complete.set()
        executor.close()


def test_h2d_chunking_groups_regions_by_kv_bytes_and_waits_between_launches():
    gates = [Event() for _ in range(3)]
    extent = SimpleNamespace(capacity_blocks=16, request_slot=0)
    regions = (
        CopyRegion(extent, 0, 0, 2),  # 40 bytes including K/V.
        CopyRegion(extent, 2, 2, 1),  # Starts a new chunk.
        CopyRegion(extent, 3, 3, 0),  # Zero-byte region stays in order.
        CopyRegion(extent, 3, 3, 3),  # Oversized single-region chunk.
    )
    dependency = SimpleNamespace(query=lambda: True)
    plan = CopyPlan("H2D", regions, (dependency,), (10, 11, 12, 13))
    launches = []

    class Ticket:
        def __init__(self, index):
            self.index = index
        def query(self):
            return gates[self.index].is_set()
        def duration_ms(self):
            return self.index + 0.25

    class Backend:
        arena = SimpleNamespace(descriptor=SimpleNamespace(block_bytes=10))
        def launch(self, chunk):
            launches.append(chunk)
            return Ticket(len(launches) - 1)
        def close(self):
            pass

    executor = CopyExecutor(Backend(), poll_interval_s=0.001, h2d_chunk_bytes=50, h2d_group_size=1)
    try:
        future = executor.submit(plan)
        wait_for(lambda: len(launches) == 1)
        assert launches[0].regions == regions[:1]
        assert launches[0].dependencies == (dependency,)
        assert launches[0].round_ids == (10,)
        assert not future.done()

        gates[0].set()
        wait_for(lambda: len(launches) == 2)
        assert launches[1].regions == regions[1:3]
        assert launches[1].dependencies == ()
        assert launches[1].round_ids == (11, 12)
        assert not future.done()

        gates[1].set()
        wait_for(lambda: len(launches) == 3)
        assert launches[2].regions == regions[3:]
        assert launches[2].dependencies == ()
        assert launches[2].round_ids == (13,)
        assert not future.done()

        gates[2].set()
        receipt = future.result(2)
        chunks = _plan_chunks(plan, 10, 50)
        assert tuple(c.bytes for c in chunks) == (40, 20, 60)
        assert tuple(c.oversized for c in chunks) == (False, False, True)
        assert receipt.duration_ms == 0.25 + 1.25 + 2.25
    finally:
        for gate in gates:
            gate.set()
        executor.close()


def test_h2d_chunk_failure_stops_unsubmitted_regions():
    extent = SimpleNamespace(capacity_blocks=8, request_slot=0)
    plan = CopyPlan("H2D", (
        CopyRegion(extent, 0, 0, 1),
        CopyRegion(extent, 1, 1, 1),
        CopyRegion(extent, 2, 2, 1),
    ))
    gate = Event()
    launches = []

    class Ticket:
        def query(self):
            return gate.is_set()
        def duration_ms(self):
            return 0.0

    class Backend:
        arena = SimpleNamespace(descriptor=SimpleNamespace(block_bytes=16))
        def launch(self, chunk):
            launches.append(chunk)
            if len(launches) == 2:
                raise RuntimeError("chunk submit failed")
            return Ticket()
        def close(self):
            pass

    executor = CopyExecutor(Backend(), poll_interval_s=0.001, h2d_chunk_bytes=32, h2d_group_size=1)
    try:
        future = executor.submit(plan)
        wait_for(lambda: len(launches) == 1)
        gate.set()
        with pytest.raises(RuntimeError, match="chunk submit failed"):
            future.result(2)
        assert len(launches) == 2
        assert launches[0].regions == plan.regions[:1]
        assert launches[1].regions == plan.regions[1:2]
    finally:
        gate.set()
        executor.close()


def test_d2h_and_disabled_budget_keep_single_launch():
    extent = SimpleNamespace(capacity_blocks=8, request_slot=0)
    regions = (CopyRegion(extent, 0, 0, 1), CopyRegion(extent, 1, 1, 1))
    launches = []

    class Ticket:
        def query(self):
            return True
        def duration_ms(self):
            return 0.1

    class Backend:
        arena = SimpleNamespace(descriptor=SimpleNamespace(block_bytes=16))
        def launch(self, chunk):
            launches.append(chunk)
            return Ticket()
        def close(self):
            pass

    executor = CopyExecutor(Backend(), h2d_chunk_bytes=32)
    disabled = CopyExecutor(Backend(), h2d_chunk_bytes=0)
    try:
        d2h = CopyPlan("D2H", regions)
        h2d = CopyPlan("H2D", regions)
        executor.submit(d2h).result(2)
        disabled.submit(h2d).result(2)
        assert launches == [d2h, h2d]
    finally:
        executor.close()
        disabled.close()


def test_two_draft_owners_on_one_device_keep_their_request_identity():
    barrier = Barrier(2)
    class FakeEvent:
        def record(self, stream):
            pass
    cuda = SimpleNamespace(Event=lambda **kwargs: FakeEvent(), current_stream=lambda device: None)
    timeline = GPUTimeline([], torch_module=SimpleNamespace(cuda=cuda))
    def make_model():
        return SimpleNamespace(_forward=lambda: barrier.wait(timeout=2))
    models = [make_model(), make_model()]
    for slot, model in enumerate(models):
        owner = f"draft:{slot}"
        timeline.set_context(owner, "draft", (slot,), slot + 1)
        timeline.instrument(model, 0, owner)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model._forward) for model in models]
        for future in futures:
            future.result(3)
    assert sorted((row[0]["owner"], row[0]["slots"], row[0]["round_ids"]) for row in timeline.events) == [
        ("draft:0", [0], [1]), ("draft:1", [1], [2])]
