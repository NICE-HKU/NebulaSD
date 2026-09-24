from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from swiftllm.server.starsd_async_operations import StarsDCopyStatus, StarsDAsyncFatalError
from swiftllm.server.starsd_target_facade import (
    SwiftLLMStarsDTargetFacade,
    StarsDReservation,
    _BatchRecord,
)


class FakeEvent:
    def __init__(self, ready: bool = False) -> None:
        self.ready = bool(ready)

    def query(self) -> bool:
        return self.ready

    def complete(self) -> None:
        self.ready = True


@dataclass
class FakeLocation:
    bank_id: int
    bank_epoch: int
    bank_base_block: int
    request_start_block: int
    num_blocks: int
    logical_kv_len: int
    kv_version: int


@dataclass
class FakeBank:
    bank_id: int
    epoch: int
    num_blocks: int
    base_block: int = 0
    alloc_ptr: int = 0
    role: str = "STANDBY"
    batch_id: str | None = None
    request_ranges: dict[int, FakeLocation] | None = None
    ready_event: object | None = None

    @property
    def remaining_blocks(self) -> int:
        return self.num_blocks - self.alloc_ptr


class FakeRowManager:
    def __init__(self) -> None:
        self.available_ids = [1, 2, 3]
        self.free_calls: list[int] = []

    def free_id(self, row: int) -> None:
        self.free_calls.append(int(row))
        if int(row) not in self.available_ids:
            self.available_ids.append(int(row))


class FakeManager:
    def __init__(self) -> None:
        self.double_bank_enabled = True
        self.active_bank_id = 0
        self.standby_bank_id = 1
        self.banks = {
            0: FakeBank(0, 0, 8, role="ACTIVE", request_ranges={}),
            1: FakeBank(1, 2, 8, base_block=8, role="STANDBY", batch_id="batch-h2d", request_ranges={}),
        }
        self.num_seq_allocated_blocks = torch.zeros((4,), dtype=torch.int32)
        self.release_calls: list[tuple] = []
        self.version_updates = 0

    def get_bank_descriptor(self, bank_id: int) -> FakeBank:
        bank = self.banks[int(bank_id)]
        return FakeBank(
            bank.bank_id,
            bank.epoch,
            bank.num_blocks,
            bank.base_block,
            bank.alloc_ptr,
            bank.role,
            bank.batch_id,
            dict(bank.request_ranges or {}),
            bank.ready_event,
        )

    def set_bank_location_kv_version(self, row: int, *, bank_id: int, bank_epoch: int, kv_version: int) -> FakeLocation:
        bank = self.banks[int(bank_id)]
        location = bank.request_ranges[int(row)]
        if bank.epoch != int(bank_epoch):
            raise RuntimeError("bank epoch mismatch")
        updated = FakeLocation(
            location.bank_id,
            location.bank_epoch,
            location.bank_base_block,
            location.request_start_block,
            location.num_blocks,
            location.logical_kv_len,
            int(kv_version),
        )
        bank.request_ranges[int(row)] = updated
        self.version_updates += 1
        return updated

    def _set_valid_blocks_for_location(self, row: int, _location: FakeLocation, copied_blocks: int) -> None:
        self.num_seq_allocated_blocks[int(row)] = int(copied_blocks)

    def mark_bank_prepared(self, bank_id: int, *, batch_id: str | None = None, ready_event=None) -> FakeBank:
        bank = self.banks[int(bank_id)]
        bank.role = "PREPARED"
        bank.batch_id = batch_id
        bank.ready_event = ready_event
        return self.get_bank_descriptor(bank_id)

    def swap_active_standby(self):
        old_active = self.banks[self.active_bank_id]
        old_standby = self.banks[self.standby_bank_id]
        old_active.role = "STANDBY"
        old_standby.role = "ACTIVE"
        self.active_bank_id, self.standby_bank_id = self.standby_bank_id, self.active_bank_id
        return self.get_bank_descriptor(self.active_bank_id), self.get_bank_descriptor(self.standby_bank_id)

    def release_bank_ranges_exact_batch(self, ranges, apply: bool = True):
        self.release_calls.append((tuple(ranges), bool(apply)))
        if not apply:
            return
        for bank_id, _bank_epoch, row, _start, _capacity, _batch_id in ranges:
            self.banks[int(bank_id)].request_ranges.pop(int(row), None)


async def _release_bank_adapter_sessions(**_kwargs):
    return {"status": "ok", "released_count": 0, "released": []}


async def _release_exact_sessions(_keys):
    return {"status": "ok", "released_count": 0, "released": []}


def facade_with_one_reservation() -> tuple[SwiftLLMStarsDTargetFacade, FakeManager, StarsDReservation]:
    facade = SwiftLLMStarsDTargetFacade.__new__(SwiftLLMStarsDTargetFacade)
    manager = FakeManager()
    row_manager = FakeRowManager()
    model = SimpleNamespace(
        gpu_block_manager=manager,
        k_cache=torch.zeros((16, 4), dtype=torch.uint8),
        v_cache=torch.zeros((16, 4), dtype=torch.uint8),
    )
    facade.worker = SimpleNamespace(
        model=model,
        request_id_manager=row_manager,
        sessions={},
        release_bank_adapter_sessions=_release_bank_adapter_sessions,
        release_exact_sessions=_release_exact_sessions,
    )
    facade.engine_config = SimpleNamespace(max_seqs_in_block_table=4)
    facade._resource_rows = {}
    facade._reserve_batches = {}
    facade._released = {}
    facade._released_order = __import__("collections").deque()
    facade._release_replay_window = 4096
    facade._bank_protections = {}
    facade._protection_index = {}
    facade.configure_starsd_identity = SwiftLLMStarsDTargetFacade.configure_starsd_identity.__get__(facade)
    facade.starsd_async_operation_counts = SwiftLLMStarsDTargetFacade.starsd_async_operation_counts.__get__(facade)
    facade.abort_starsd_async_request = SwiftLLMStarsDTargetFacade.abort_starsd_async_request.__get__(facade)
    from swiftllm.server.starsd_async_operations import StarsDAsyncOperationRegistry

    facade._async_ops = StarsDAsyncOperationRegistry(target_id="target-0", process_generation=3)
    facade._async_identity = ("target-0", 3)
    reservation = StarsDReservation("batch-h2d:bank1:row0", "req-1", 1, 0, 0, 1, 2, 0, 4, 0, 0)
    key = (reservation.request_id, reservation.request_epoch, reservation.round_id, reservation.reservation_id)
    facade._resource_rows[key] = reservation
    facade._reserve_batches["batch-h2d"] = _BatchRecord("RESERVED", (key,))
    manager.banks[1].request_ranges[0] = FakeLocation(1, 2, 8, 0, 4, 0, 0)
    manager.banks[1].alloc_ptr = 4
    return facade, manager, reservation


def h2d_item(reservation: StarsDReservation, *, operation_id: str = "op-h2d") -> dict[str, object]:
    return {
        "operation_id": operation_id,
        "plan_id": operation_id,
        "target_id": "target-0",
        "target_process_generation": 3,
        "request_id": reservation.request_id,
        "request_epoch": reservation.request_epoch,
        "round_id": reservation.round_id,
        "reservation_id": reservation.reservation_id,
        "bank_id": reservation.bank_id,
        "bank_epoch": reservation.bank_epoch,
        "row": reservation.row,
        "start_block": reservation.start_block,
        "capacity_blocks": reservation.capacity_blocks,
        "copy_block_count": 2,
        "expected_host_version": 5,
    }


def d2h_item(*, operation_id: str = "op-d2h") -> dict[str, object]:
    return {
        "operation_id": operation_id,
        "plan_id": operation_id,
        "target_id": "target-0",
        "target_process_generation": 3,
        "request_id": "req-1",
        "request_epoch": 1,
        "round_id": 0,
        "batch_id": "batch-d2h",
        "bank_id": 0,
        "bank_epoch": 0,
        "row": 0,
        "start_block": 0,
        "capacity_blocks": 4,
        "kv_version": 7,
        "dirty_begin_block": 1,
        "dirty_block_count": 1,
        "post_crop_logical_kv_len": 4,
    }


def views(blocks: int = 2) -> tuple[memoryview, memoryview]:
    return memoryview(bytearray(blocks * 4)), memoryview(bytearray(blocks * 4))


def active_operation(facade: SwiftLLMStarsDTargetFacade, operation_id: str):
    return facade._async_ops._active[operation_id]


def test_pending_h2d_abort_defers_release_until_event_drains() -> None:
    facade, manager, reservation = facade_with_one_reservation()
    k, v = views()
    handles = facade.stage_h2d_from_host((h2d_item(reservation),), (k,), (v,))
    event = FakeEvent(False)
    active_operation(facade, handles[0].operation_id).event = event

    progress = facade.abort_starsd_async_request(request_id="req-1", request_epoch=1, round_id=0)

    assert progress[0].status == StarsDCopyStatus.ABORT_REQUESTED
    assert facade.starsd_async_operation_counts()["abort_pending"] == 1
    assert manager.banks[1].request_ranges

    retry = facade.abort_starsd_async_request(request_id="req-1", request_epoch=1, round_id=0)
    assert retry[0].status == StarsDCopyStatus.ABORT_REQUESTED

    event.complete()
    assert facade.abort_starsd_async_request(request_id="req-1", request_epoch=1, round_id=0)[0].status == StarsDCopyStatus.ABORTED
    assert facade.starsd_async_operation_counts()["active"] == 0
    __import__("asyncio").run(
        facade.release(({
            "request_id": "req-1",
            "request_epoch": 1,
            "round_id": 0,
            "reservation_id": reservation.reservation_id,
        },))
    )
    assert manager.banks[1].request_ranges == {}
    release_calls = len(manager.release_calls)
    __import__("asyncio").run(
        facade.release(({
            "request_id": "req-1",
            "request_epoch": 1,
            "round_id": 0,
            "reservation_id": reservation.reservation_id,
        },))
    )
    assert len(manager.release_calls) == release_calls


def test_abort_wrong_request_fence_has_zero_side_effect() -> None:
    facade, manager, reservation = facade_with_one_reservation()
    k, v = views()
    handles = facade.stage_h2d_from_host((h2d_item(reservation),), (k,), (v,))
    active_operation(facade, handles[0].operation_id).event = FakeEvent(False)

    assert facade.abort_starsd_async_request(request_id="req-1", request_epoch=2, round_id=0) == ()
    assert facade.abort_starsd_async_request(request_id="req-1", request_epoch=1, round_id=1) == ()
    assert facade.abort_starsd_async_request(request_id="req-1", request_epoch=1, round_id=0, process_generation=4) == ()
    assert facade.starsd_async_operation_counts()["active"] == 1
    assert manager.banks[1].request_ranges


def test_stage_h2d_pre_mutation_collision_has_zero_metadata_side_effect() -> None:
    facade, manager, reservation = facade_with_one_reservation()
    k, v = views()
    facade._async_ops.reserve_batch(
        (facade._h2d_handles_for_items((h2d_item(reservation, operation_id="op-collision"),))[0],),
        ("old",),
    )

    with pytest.raises(RuntimeError, match="different payload"):
        facade.stage_h2d_from_host((h2d_item(reservation, operation_id="op-collision"),), (k,), (v,))

    assert manager.version_updates == 0
    assert facade.starsd_async_operation_counts()["active"] == 1
    assert manager.banks[1].request_ranges[0].kv_version == 0


def test_h2d_post_mutation_commit_failure_is_fatal_and_keeps_state_for_fail_stop() -> None:
    facade, manager, reservation = facade_with_one_reservation()
    k, v = views()
    original_commit = facade._async_ops.commit_batch

    def fail_commit(*args, **kwargs):
        raise RuntimeError("commit injected")

    facade._async_ops.commit_batch = fail_commit

    with pytest.raises(StarsDAsyncFatalError):
        facade.stage_h2d_from_host((h2d_item(reservation),), (k,), (v,))

    assert manager.version_updates == 1
    assert manager.banks[1].request_ranges[0].kv_version == 5
    facade._async_ops.commit_batch = original_commit


def test_pending_d2h_abort_keeps_protection_and_result_ready_is_not_publish() -> None:
    facade, manager, _reservation = facade_with_one_reservation()
    manager.banks[0].request_ranges[0] = FakeLocation(0, 0, 0, 0, 4, 4, 7)
    facade.protect_bank_epoch("op-d2h", 0, 0)
    k, v = views(1)
    handles = facade.stage_dirty_to_host((d2h_item(),), (k,), (v,))
    event = FakeEvent(False)
    active_operation(facade, handles[0].operation_id).event = event

    assert facade.abort_starsd_async_request(request_id="req-1", request_epoch=1, round_id=0)[0].status == StarsDCopyStatus.ABORT_REQUESTED
    assert facade.starsd_async_operation_counts()["bank_protections"] == 1

    event.complete()
    assert facade.progress_dirty_to_host(handles)[0].status == StarsDCopyStatus.ABORTED
    assert facade.starsd_async_operation_counts()["active"] == 0
    facade.release_bank_epoch("op-d2h", 0, 0)
    assert facade.starsd_async_operation_counts()["bank_protections"] == 0


def test_d2h_post_submit_failure_is_fatal_and_keeps_protection() -> None:
    facade, manager, _reservation = facade_with_one_reservation()
    manager.banks[0].request_ranges[0] = FakeLocation(0, 0, 0, 0, 4, 4, 7)
    facade.protect_bank_epoch("op-d2h", 0, 0)
    original_commit = facade._async_ops.commit_batch

    def fail_commit(*args, **kwargs):
        raise RuntimeError("commit injected")

    facade._async_ops.commit_batch = fail_commit
    k, v = views(1)

    with pytest.raises(StarsDAsyncFatalError):
        facade.stage_dirty_to_host((d2h_item(),), (k,), (v,))

    assert facade.starsd_async_operation_counts()["bank_protections"] == 1
    facade._async_ops.commit_batch = original_commit


def test_shutdown_drains_async_operations_and_protections() -> None:
    facade, _manager, reservation = facade_with_one_reservation()
    k, v = views()
    handles = facade.stage_h2d_from_host((h2d_item(reservation),), (k,), (v,))
    active_operation(facade, handles[0].operation_id).event = FakeEvent(False)
    facade.protect_bank_epoch("manual", 1, 2)

    __import__("asyncio").run(facade.shutdown())

    counts = facade.starsd_async_operation_counts()
    assert counts["active"] == 0
    assert counts["abort_pending"] == 0
    assert counts["bank_protections"] == 0


def test_h2d_copy_supports_non_contiguous_bank_slice_on_cpu() -> None:
    location = FakeLocation(1, 2, 1, 1, 2, 0, 0)
    cache = torch.zeros((8, 2, 3), dtype=torch.uint8)
    payload = bytes(range(12))
    view = memoryview(bytearray(payload))

    SwiftLLMStarsDTargetFacade._copy_host_to_bank(cache, view, location, 2)

    start = location.bank_base_block + location.request_start_block
    copied = cache[start : start + 2].contiguous().numpy().tobytes()
    assert copied == payload
    assert cache[:start].sum().item() == 0
    assert cache[start + 2 :].sum().item() == 0
