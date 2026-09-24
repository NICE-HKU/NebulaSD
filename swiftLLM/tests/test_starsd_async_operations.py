from __future__ import annotations

from dataclasses import replace

import pytest

from swiftllm.server.starsd_async_operations import (
    StarsDAsyncOperationRegistry,
    StarsDCopyDirection,
    StarsDCopyHandle,
    StarsDCopyStatus,
)


class FakeEvent:
    def __init__(self, ready: bool = False, elapsed_ms: float | None = None) -> None:
        self.ready = bool(ready)
        self.synchronized = False
        self.elapsed_ms = elapsed_ms

    def query(self) -> bool:
        return self.ready

    def complete(self) -> None:
        self.ready = True

    def synchronize(self) -> None:
        self.synchronized = True
        self.ready = True

    def elapsed_time(self, _other: object) -> float:
        if self.elapsed_ms is None:
            raise RuntimeError("fake elapsed unavailable")
        return self.elapsed_ms


def handle(
    operation_id: str,
    *,
    direction: StarsDCopyDirection = StarsDCopyDirection.H2D,
    batch_id: str = "batch-a",
    bank_id: int = 1,
    bank_epoch: int = 2,
    target_id: str = "target-0",
    generation: int = 7,
    request_id: str | None = None,
    host_kv_version: int = 3,
) -> StarsDCopyHandle:
    suffix = operation_id.rsplit("-", 1)[-1]
    return StarsDCopyHandle(
        operation_id,
        direction,
        target_id,
        generation,
        request_id or f"req-{suffix}",
        1,
        0,
        batch_id,
        bank_id,
        bank_epoch,
        host_kv_version,
    )


def reserve_and_commit(
    registry: StarsDAsyncOperationRegistry,
    handles: tuple[StarsDCopyHandle, ...],
    *,
    event: FakeEvent | None,
    start_event: FakeEvent | None = None,
    results: tuple[object, ...] | None = None,
) -> None:
    admission = registry.reserve_batch(handles, tuple(f"fp:{item!r}" for item in handles))
    assert not admission.replayed
    registry.commit_batch(
        handles,
        event=event,
        start_event=start_event,
        results=results or tuple(f"result:{item.operation_id}" for item in handles),
    )


def test_stage_pending_progress_ready_finalize_and_replay() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    event = FakeEvent()
    op = handle("op-1")

    reserve_and_commit(registry, (op,), event=event)
    assert registry.active_operation_count == 1
    assert registry.progress(op).status == StarsDCopyStatus.PENDING

    event.complete()
    assert registry.progress(op).status == StarsDCopyStatus.READY
    assert registry.finalize_batch((op,), expected_direction=StarsDCopyDirection.H2D) == ("result:op-1",)
    assert registry.active_operation_count == 0
    assert registry.replay_tombstone_count == 1
    assert registry.finalize_batch((op,), expected_direction=StarsDCopyDirection.H2D) == ("result:op-1",)


def test_ready_progress_reports_cuda_event_elapsed_seconds_without_synchronizing() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    start = FakeEvent(True, elapsed_ms=2.5)
    end = FakeEvent(False)
    op = handle("op-1")

    reserve_and_commit(registry, (op,), event=end, start_event=start)
    assert registry.progress(op).status == StarsDCopyStatus.PENDING
    end.complete()
    progress = registry.progress(op)

    assert progress.status == StarsDCopyStatus.READY
    assert progress.device_duration_s == pytest.approx(0.0025)
    assert not start.synchronized
    assert not end.synchronized
    registry.finalize_batch((op,), expected_direction=StarsDCopyDirection.H2D)
    assert registry.progress(op).device_duration_s == pytest.approx(0.0025)


def test_activate_before_event_complete_is_rejected_without_retiring_active() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    op = handle("op-1")
    reserve_and_commit(registry, (op,), event=FakeEvent())

    with pytest.raises(RuntimeError, match="not ready"):
        registry.finalize_batch((op,), expected_direction=StarsDCopyDirection.H2D)

    assert registry.active_operation_count == 1
    assert registry.replay_tombstone_count == 0


def test_terminal_retirement_tombstone_window_and_active_capacity_are_separate() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7, capacity=1, replay_window=1)
    first = handle("op-1")
    reserve_and_commit(registry, (first,), event=FakeEvent(True))
    registry.finalize_batch((first,), expected_direction=StarsDCopyDirection.H2D)

    second = handle("op-2")
    reserve_and_commit(registry, (second,), event=FakeEvent(False))
    assert registry.active_operation_count == 1
    assert registry.replay_tombstone_count == 1

    third = handle("op-3")
    with pytest.raises(RuntimeError, match="capacity"):
        registry.reserve_batch((third,), ("fp:third",))
    assert registry.active_operation_count == 1
    assert registry.replay_tombstone_count == 1

    event = registry.progress(second)
    assert event.status == StarsDCopyStatus.PENDING


def test_exact_replay_and_different_payload_collision() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    op = handle("op-1")
    admission = registry.reserve_batch((op,), ("fp",))
    assert not admission.replayed
    replay = registry.reserve_batch((op,), ("fp",))
    assert replay.replayed

    with pytest.raises(RuntimeError, match="different payload"):
        registry.reserve_batch((replace(op, host_kv_version=99),), ("fp",))
    with pytest.raises(RuntimeError, match="different payload"):
        registry.reserve_batch((op,), ("other",))


def test_all_new_all_replay_and_mixed_batches_are_explicit() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    batch = (handle("op-1"), handle("op-2"))

    admission = registry.reserve_batch(batch, ("fp1", "fp2"))
    assert not admission.replayed
    replay = registry.reserve_batch(batch, ("fp1", "fp2"))
    assert replay.replayed

    with pytest.raises(RuntimeError, match="mixed replay/new"):
        registry.reserve_batch((batch[0], handle("op-3")), ("fp1", "fp3"))


def test_batch_second_item_collision_or_capacity_failure_has_zero_new_side_effect() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7, capacity=2)
    live = handle("op-live")
    reserve_and_commit(registry, (live,), event=FakeEvent(False))

    with pytest.raises(RuntimeError, match="different payload"):
        registry.reserve_batch((handle("op-new"), replace(live, host_kv_version=8)), ("fp-new", "fp-live"))
    assert registry.active_operation_count == 1

    with pytest.raises(RuntimeError, match="capacity"):
        registry.reserve_batch((handle("op-a"), handle("op-b")), ("fp-a", "fp-b"))
    assert registry.active_operation_count == 1


def test_finalize_requires_complete_ordered_single_batch() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    batch = (handle("op-1"), handle("op-2"))
    reserve_and_commit(registry, batch, event=FakeEvent(True))

    with pytest.raises(RuntimeError, match="complete and in stage order"):
        registry.finalize_batch((batch[0],), expected_direction=StarsDCopyDirection.H2D)
    with pytest.raises(RuntimeError, match="complete and in stage order"):
        registry.finalize_batch((batch[1], batch[0]), expected_direction=StarsDCopyDirection.H2D)

    other = handle("op-3", batch_id="batch-b")
    reserve_and_commit(registry, (other,), event=FakeEvent(True))
    with pytest.raises(RuntimeError, match="batch id"):
        registry.finalize_batch((batch[0], other), expected_direction=StarsDCopyDirection.H2D)


def test_finalize_callback_runs_exactly_once_for_batch() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    batch = (handle("op-1"), handle("op-2"))
    reserve_and_commit(registry, batch, event=FakeEvent(True), results=("staged-1", "staged-2"))
    calls: list[str] = []

    def finalize() -> tuple[str, str]:
        calls.append("activate")
        return ("activated-1", "activated-2")

    assert registry.finalize_batch(batch, expected_direction=StarsDCopyDirection.H2D, finalize=finalize) == ("activated-1", "activated-2")
    assert registry.finalize_batch(batch, expected_direction=StarsDCopyDirection.H2D, finalize=finalize) == ("activated-1", "activated-2")
    assert calls == ["activate"]


def test_pending_event_ready_abort_and_late_progress_converge() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    event = FakeEvent()
    pending = handle("op-1")
    reserve_and_commit(registry, (pending,), event=event)

    assert registry.abort(pending).status == StarsDCopyStatus.ABORT_REQUESTED
    assert registry.abort_pending_count == 1
    assert registry.progress(pending).status == StarsDCopyStatus.ABORT_REQUESTED

    event.complete()
    assert registry.progress(pending).status == StarsDCopyStatus.ABORTED
    assert registry.active_operation_count == 0
    assert registry.abort(pending).status == StarsDCopyStatus.ABORTED

    ready = handle("op-2")
    reserve_and_commit(registry, (ready,), event=FakeEvent(True))
    assert registry.abort(ready).status == StarsDCopyStatus.ABORTED
    assert registry.progress(ready).status == StarsDCopyStatus.ABORTED


def test_wrong_fences_are_rejected_without_consuming_correct_operation() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    op = handle("op-1")
    reserve_and_commit(registry, (op,), event=FakeEvent(False))

    with pytest.raises(RuntimeError, match="target generation"):
        registry.progress(replace(op, target_id="target-1"))
    with pytest.raises(RuntimeError, match="target generation"):
        registry.progress(replace(op, process_generation=8))
    with pytest.raises(RuntimeError):
        registry.progress(replace(op, bank_epoch=99))
    with pytest.raises(RuntimeError):
        registry.progress(replace(op, host_kv_version=99))
    assert registry.active_operation_count == 1


def test_d2h_result_ready_is_not_hostkv_published() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    op = handle("op-1", direction=StarsDCopyDirection.D2H)
    reserve_and_commit(registry, (op,), event=FakeEvent(True), results=("dirty-result",))

    assert registry.finalize_batch((op,), expected_direction=StarsDCopyDirection.D2H) == ("dirty-result",)
    progress = registry.progress(op)
    assert progress.status == StarsDCopyStatus.RESULT_READY
    assert progress.status.value != "published"


def test_shutdown_drains_or_aborts_active_operations() -> None:
    registry = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=7)
    events = (FakeEvent(False), FakeEvent(False))
    ops = (handle("op-1"), handle("op-2"))
    for op, event in zip(ops, events, strict=True):
        reserve_and_commit(registry, (op,), event=event)

    registry.drain()

    assert all(event.synchronized for event in events)
    assert registry.active_operation_count == 0
    assert registry.replay_tombstone_count == 2
    assert tuple(registry.progress(op).status for op in ops) == (
        StarsDCopyStatus.ABORTED,
        StarsDCopyStatus.ABORTED,
    )
