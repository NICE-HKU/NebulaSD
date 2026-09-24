from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_process_local_facade_module():
    # A package import preserves relative imports of starsd_copy_support.
    # Do not manufacture a standalone module name or a replacement package.
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import swiftllm
    import swiftllm.server.starsd_target_facade as target_facade
    for name, module in (("swiftllm", swiftllm), ("swiftllm.server.starsd_target_facade", target_facade)):
        resolved = Path(module.__file__).resolve()
        print(f"{name}.__file__={resolved}", flush=True)
        assert resolved.is_relative_to(root)
    return importlib.import_module("swiftllm.server.starsd_process_local_facade")


process_facade = load_process_local_facade_module()
LocalCudaResult = process_facade.LocalCudaResult
H2DCompletion = process_facade.H2DCompletion
DirectVerifyBatchPlan = process_facade.DirectVerifyBatchPlan
DirectVerifyRequest = process_facade.DirectVerifyRequest
DirectPrefillBatchPlan = process_facade.DirectPrefillBatchPlan
DirectPrefillRequest = process_facade.DirectPrefillRequest
ExactBankRangeRelease = process_facade.ExactBankRangeRelease
ExactSessionKey = process_facade.ExactSessionKey
StandbyPrepareItem = process_facade.StandbyPrepareItem
SwiftLLMProcessLocalTargetFacade = process_facade.SwiftLLMProcessLocalTargetFacade


class FakeBank:
    def __init__(self, bank_id: int, base_block: int, num_blocks: int, role: str) -> None:
        self.bank_id = bank_id
        self.base_block = base_block
        self.num_blocks = num_blocks
        self.alloc_ptr = 0
        self.epoch = 0
        self.role = role
        self.batch_id = None
        self.request_ranges = {}


class FakeLocation:
    def __init__(self, *, row: int, bank: FakeBank, start: int, blocks: int, logical: int, version: int) -> None:
        self.request_id = row
        self.bank_id = bank.bank_id
        self.bank_epoch = bank.epoch
        self.request_start_block = start
        self.num_blocks = blocks
        self.logical_kv_len = logical
        self.kv_version = version


class FakeBlockManager:
    double_bank_enabled = True

    def __init__(self) -> None:
        self._active_bank_id = 0
        self._standby_bank_id = 1
        self.banks = {
            0: FakeBank(0, 0, 8, "ACTIVE"),
            1: FakeBank(1, 8, 8, "STANDBY"),
        }

    @property
    def active_bank_id(self) -> int:
        return self._active_bank_id

    @property
    def standby_bank_id(self) -> int:
        return self._standby_bank_id

    def get_bank_descriptor(self, bank_id: int) -> FakeBank:
        return self.banks[int(bank_id)]

    def reserve_in_bank_batch_atomic(self, bank_id, requests, *, reset_bank=False):
        bank = self.banks[int(bank_id)]
        if reset_bank:
            self.reset_bank(bank_id)
        out = []
        for row, blocks, logical, version, batch_id in requests:
            start = bank.alloc_ptr
            bank.alloc_ptr += int(blocks)
            bank.batch_id = batch_id
            location = FakeLocation(
                row=int(row),
                bank=bank,
                start=start,
                blocks=int(blocks),
                logical=int(logical),
                version=int(version),
            )
            bank.request_ranges[int(row)] = location
            out.append(location)
        return out

    def mark_bank_prepared(self, bank_id, *, batch_id=None, ready_event=None):
        bank = self.banks[int(bank_id)]
        bank.role = "PREPARED"
        bank.batch_id = batch_id
        return bank

    def swap_active_standby(self):
        old_active = self.banks[self._active_bank_id]
        old_standby = self.banks[self._standby_bank_id]
        old_active.role = "STANDBY"
        old_standby.role = "ACTIVE"
        self._active_bank_id, self._standby_bank_id = self._standby_bank_id, self._active_bank_id
        return self.banks[self._active_bank_id], self.banks[self._standby_bank_id]

    def reset_bank(self, bank_id):
        bank = self.banks[int(bank_id)]
        bank.alloc_ptr = 0
        bank.epoch += 1
        bank.request_ranges = {}
        bank.batch_id = None
        if bank.role == "PREPARED":
            bank.role = "STANDBY"
        return bank

    def release_bank_ranges_exact_batch(self, ranges, apply=True):
        for bank_id, bank_epoch, row, start_block, capacity_blocks, batch_id in ranges:
            bank = self.banks[int(bank_id)]
            location = bank.request_ranges.get(int(row))
            if location is None:
                raise RuntimeError("exact bank release range is not live")
            if int(bank.epoch) != int(bank_epoch):
                raise RuntimeError("exact bank release epoch mismatch")
            if int(location.request_start_block) != int(start_block):
                raise RuntimeError("exact bank release start mismatch")
            if int(location.num_blocks) != int(capacity_blocks):
                raise RuntimeError("exact bank release capacity mismatch")
            if batch_id is not None and str(bank.batch_id) != str(batch_id):
                raise RuntimeError("exact bank release batch_id mismatch")
        if not apply:
            return
        for bank_id, _bank_epoch, row, _start_block, _capacity_blocks, _batch_id in ranges:
            self.banks[int(bank_id)].request_ranges.pop(int(row), None)


class FakeCopyBackend:
    def __init__(self) -> None:
        self.calls = []

    def launch_h2d_on_stream(self, *args, **kwargs) -> LocalCudaResult:
        self.calls.append(("h2d", args, kwargs))
        return LocalCudaResult(ok=True, event="h2d-event")

    def launch_dirty_d2h_on_stream(self, *args, **kwargs) -> LocalCudaResult:
        self.calls.append(("d2h", args, kwargs))
        return LocalCudaResult(ok=True, event="d2h-event")


def make_facade() -> SwiftLLMProcessLocalTargetFacade:
    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=FakeBlockManager()),
        request_id_manager=SimpleNamespace(max_id=16, available_ids=list(range(16))),
        sessions={},
    )
    return SwiftLLMProcessLocalTargetFacade(worker=worker, copy_backend=FakeCopyBackend())


def test_process_local_facade_prepares_and_switches_standby_bank() -> None:
    facade = make_facade()
    active, standby = facade.describe_banks()
    assert active.bank_id == 0
    assert standby.bank_id == 1
    assert standby.free_blocks == 8
    assert active.capacity_rows == 16
    assert active.alloc_rows == 0

    prepared = facade.prepare_standby_batch(
        [StandbyPrepareItem(row=3, required_blocks=2, logical_kv_len=32, kv_version=7)],
        bank_id=standby.bank_id,
        batch_seq=42,
    )
    assert prepared[0].row == 3
    assert prepared[0].bank_id == 1
    assert prepared[0].bank_epoch == 1
    assert prepared[0].start_block == 0
    assert prepared[0].block_count == 2
    _, prepared_standby = facade.describe_banks()
    assert prepared_standby.capacity_rows == 16
    assert prepared_standby.alloc_rows == 1

    new_active, new_standby = facade.activate_or_switch_bank(
        bank_id=1,
        bank_epoch=1,
        batch_seq=42,
        h2d_completion=H2DCompletion(bank_id=1, bank_epoch=1, batch_seq=42, ok=True),
    )
    assert new_active.bank_id == 1
    assert new_standby.bank_id == 0


def test_process_local_facade_reset_and_copy_delegation() -> None:
    facade = make_facade()
    h2d = facade.launch_h2d_on_stream("host", "gpu", stream="copy")
    d2h = facade.launch_dirty_d2h_on_stream("gpu", "host", stream="copy")
    assert h2d.event == "h2d-event"
    assert d2h.event == "d2h-event"
    assert facade.copy_backend.calls[0][0] == "h2d"
    assert facade.copy_backend.calls[1][0] == "d2h"

    reset = facade.reset_bank(1)
    assert reset.bank_id == 1
    assert reset.bank_epoch == 1
    assert reset.alloc_ptr_blocks == 0


def test_process_local_facade_exact_release_api() -> None:
    async def release_exact_sessions(keys):
        return {"status": "ok", "released_count": len(keys), "released": list(keys)}

    manager = FakeBlockManager()
    manager.reserve_in_bank_batch_atomic(1, [(3, 2, 32, 7, "42")], reset_bank=True)
    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=manager),
        sessions={},
        release_exact_sessions=release_exact_sessions,
    )
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)

    sessions = asyncio.run(facade.release_exact_prefill_sessions((ExactSessionKey("client", "req"),)))
    assert sessions.ok
    assert sessions.value["released_count"] == 1

    ranges = asyncio.run(
        facade.release_exact_bank_ranges(
            (
                ExactBankRangeRelease(
                    bank_id=1,
                    bank_epoch=1,
                    row=3,
                    start_block=0,
                    capacity_blocks=2,
                    batch_seq=42,
                ),
            )
        )
    )
    assert ranges.ok
    assert ranges.value["released_count"] == 1
    assert manager.banks[1].request_ranges == {}


def test_process_local_facade_fails_without_copy_backend() -> None:
    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=FakeBlockManager()),
        sessions={},
    )
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    with pytest.raises(RuntimeError, match="copy backend is not configured"):
        facade.launch_h2d_on_stream()
    with pytest.raises(RuntimeError, match="copy backend is not configured"):
        facade.launch_dirty_d2h_on_stream()


def test_process_local_facade_rejects_unsafe_bank_operations() -> None:
    facade = make_facade()
    active, standby = facade.describe_banks()
    with pytest.raises(RuntimeError, match="current standby"):
        facade.prepare_standby_batch(
            [StandbyPrepareItem(row=1, required_blocks=1, logical_kv_len=16, kv_version=0)],
            bank_id=active.bank_id,
            batch_seq=1,
        )
    with pytest.raises(RuntimeError, match="active bank"):
        facade.reset_bank(active.bank_id)

    with pytest.raises(RuntimeError, match="not prepared"):
        facade.activate_or_switch_bank(
            bank_id=standby.bank_id,
            bank_epoch=standby.bank_epoch,
            batch_seq=None,
            h2d_completion=H2DCompletion(bank_id=standby.bank_id, bank_epoch=standby.bank_epoch, batch_seq=None, ok=True),
        )

    prepared = facade.prepare_standby_batch(
        [StandbyPrepareItem(row=1, required_blocks=1, logical_kv_len=16, kv_version=0)],
        bank_id=standby.bank_id,
        batch_seq=7,
    )
    prepared_epoch = prepared[0].bank_epoch
    with pytest.raises(RuntimeError, match="not prepareable"):
        facade.prepare_standby_batch(
            [StandbyPrepareItem(row=2, required_blocks=1, logical_kv_len=16, kv_version=0)],
            bank_id=standby.bank_id,
            batch_seq=8,
        )
    with pytest.raises(RuntimeError, match="H2D completion"):
        facade.activate_or_switch_bank(
            bank_id=standby.bank_id,
            bank_epoch=prepared_epoch,
            batch_seq=7,
            h2d_completion=H2DCompletion(bank_id=standby.bank_id, bank_epoch=prepared_epoch, batch_seq=7, ok=False),
        )
    with pytest.raises(RuntimeError, match="activation fence"):
        facade.activate_or_switch_bank(
            bank_id=standby.bank_id,
            bank_epoch=prepared_epoch,
            batch_seq=7,
            h2d_completion=H2DCompletion(bank_id=standby.bank_id, bank_epoch=prepared_epoch, batch_seq=8, ok=True),
        )
    with pytest.raises(RuntimeError, match="batch sequence"):
        facade.activate_or_switch_bank(
            bank_id=standby.bank_id,
            bank_epoch=prepared_epoch,
            batch_seq=8,
            h2d_completion=H2DCompletion(bank_id=standby.bank_id, bank_epoch=prepared_epoch, batch_seq=8, ok=True),
        )


def test_process_local_facade_release_session_and_verify_delegate() -> None:
    async def submit_verify_bank_batch(run_plan):
        return [{"ok": True, "plan": run_plan}]

    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=FakeBlockManager()),
        sessions={"session": object()},
        submit_verify_bank_batch=submit_verify_bank_batch,
    )
    worker.model.gpu_block_manager.banks[0].batch_id = "3"
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    facade.release_session("session")
    assert worker.sessions == {}

    plan = DirectVerifyBatchPlan(
        active_bank_id=0,
        active_bank_epoch=0,
        batch_seq=3,
        requests=(
            DirectVerifyRequest(
                request_id="r0",
                request_row=0,
                client_tag="client",
                prompt_len=1,
                output_token_ids=(11,),
                draft_token_ids=(12,),
            ),
        ),
    )
    result = asyncio.run(facade.verify_batch_direct(plan))
    assert result.ok
    assert result.value[0]["plan"]["request_ids"] == ["r0"]
    assert result.value[0]["plan"]["request_rows"] == {"r0": 0}
    assert result.value[0]["plan"]["verify_payloads"]["r0"]["prompt_len"] == 1


def test_process_local_facade_rejects_stale_direct_verify_plan() -> None:
    async def submit_verify_bank_batch(run_plan):
        raise AssertionError("stale plan must not reach worker")

    manager = FakeBlockManager()
    manager.banks[0].epoch = 5
    manager.banks[0].batch_id = "9"
    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=manager),
        sessions={},
        submit_verify_bank_batch=submit_verify_bank_batch,
    )
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    request = DirectVerifyRequest(
        request_id="r0",
        request_row=0,
        client_tag="client",
        prompt_len=1,
        output_token_ids=(11,),
        draft_token_ids=(12,),
    )
    with pytest.raises(RuntimeError, match="bank id mismatch"):
        asyncio.run(facade.verify_batch_direct(DirectVerifyBatchPlan(1, 5, 9, (request,))))
    with pytest.raises(RuntimeError, match="epoch mismatch"):
        asyncio.run(facade.verify_batch_direct(DirectVerifyBatchPlan(0, 4, 9, (request,))))
    with pytest.raises(RuntimeError, match="batch sequence mismatch"):
        asyncio.run(facade.verify_batch_direct(DirectVerifyBatchPlan(0, 5, 8, (request,))))


def test_process_local_facade_initialize_uses_direct_mode() -> None:
    class Worker:
        initialized = False

        def __init__(self) -> None:
            self.start_background_loop = None
            self.model = SimpleNamespace(gpu_block_manager=FakeBlockManager())

        async def initialize(self, *, start_background_loop: bool = True):
            self.start_background_loop = start_background_loop
            self.initialized = True

    worker = Worker()
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    asyncio.run(facade.initialize())
    assert worker.initialized
    assert worker.start_background_loop is False


def test_process_local_facade_prefill_direct_delegates_typed_batch() -> None:
    calls = []

    async def submit_prefill_batch(items):
        calls.append(items)
        return [
            SimpleNamespace(
                request_id=item["request_id"],
                client_tag=item["client_tag"],
                error=None,
                payload={
                    "accepted_token_ids": [99],
                    "num_accepted_draft_tokens": 0,
                    "output_token_ids": [99],
                    "logical_kv_len": len(item["input_ids"]),
                    "prefill_bank_location": {
                        "request_row": 3,
                        "bank_id": item["prefill_bank_plan"]["bank_id"],
                        "bank_epoch": item["prefill_bank_plan"]["bank_epoch"],
                        "bank_offset_blocks": item["prefill_bank_plan"]["bank_offset_blocks"],
                        "capacity_blocks": item["prefill_bank_plan"]["block_count"],
                        "num_blocks": item["prefill_bank_plan"]["block_count"],
                        "kv_version": 0,
                        "batch_id": item["prefill_bank_plan"]["batch_id"],
                    },
                },
            )
            for item in items
        ]

    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=FakeBlockManager()),
        sessions={},
        submit_prefill_batch=submit_prefill_batch,
    )
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    result = asyncio.run(
        facade.prefill_batch_direct(
            DirectPrefillBatchPlan(
                bank_id=1,
                bank_epoch=0,
                batch_seq=42,
                requests=(
                    DirectPrefillRequest(
                        request_id="req",
                        client_tag="target:1:g2",
                        prompt_token_ids=(1, 2, 3),
                        max_output_len=8,
                        bank_offset_blocks=0,
                        block_count=2,
                        stop_token_ids=(4,),
                        return_hidden=False,
                    ),
                ),
            )
        )
    )

    assert result.ok
    assert result.value[0].payload["accepted_token_ids"] == [99]
    assert calls == [
        [
            {
                "task_id": "target:1:g2:req:prefill",
                "client_tag": "target:1:g2",
                "request_id": "req",
                "input_ids": [1, 2, 3],
                "max_output_len": 8,
                "stop_token_ids": (4,),
                "return_hidden": False,
                "keep_bank_range_for_export": True,
                "prefill_bank_plan": {
                    "bank_id": 1,
                    "bank_epoch": 0,
                    "batch_id": "42",
                    "bank_offset_blocks": 0,
                    "block_count": 2,
                },
            }
        ]
    ]


def test_process_local_facade_prefill_direct_does_not_require_background_queue() -> None:
    calls = []

    async def run_prefill_batch_direct(items):
        calls.append(("direct", items))
        return [
            SimpleNamespace(
                request_id=item["request_id"],
                client_tag=item["client_tag"],
                error=None,
                payload={
                    "accepted_token_ids": [99],
                    "num_accepted_draft_tokens": 0,
                    "output_token_ids": [99],
                    "logical_kv_len": len(item["input_ids"]),
                    "prefill_bank_location": {
                        "request_row": 3,
                        "bank_id": item["prefill_bank_plan"]["bank_id"],
                        "bank_epoch": item["prefill_bank_plan"]["bank_epoch"],
                        "bank_offset_blocks": item["prefill_bank_plan"]["bank_offset_blocks"],
                        "capacity_blocks": item["prefill_bank_plan"]["block_count"],
                        "num_blocks": 1,
                        "kv_version": 0,
                        "batch_id": item["prefill_bank_plan"]["batch_id"],
                    },
                },
            )
            for item in items
        ]

    async def submit_prefill_batch(_items):
        await asyncio.Future()

    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=FakeBlockManager()),
        sessions={},
        run_prefill_batch_direct=run_prefill_batch_direct,
        submit_prefill_batch=submit_prefill_batch,
    )
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)

    result = asyncio.run(
        asyncio.wait_for(
            facade.prefill_batch_direct(
                DirectPrefillBatchPlan(
                    bank_id=1,
                    bank_epoch=0,
                    batch_seq=42,
                    requests=(
                        DirectPrefillRequest(
                            request_id="req",
                            client_tag="target:1:g2",
                            prompt_token_ids=(1, 2, 3),
                            max_output_len=8,
                            bank_offset_blocks=0,
                            block_count=2,
                        ),
                    ),
                )
            ),
            timeout=1.0,
        )
    )

    assert result.ok
    assert calls[0][0] == "direct"


def test_process_local_facade_prefill_direct_rejects_bank_location_mismatch() -> None:
    async def submit_prefill_batch(items):
        return [
            SimpleNamespace(
                request_id=items[0]["request_id"],
                client_tag=items[0]["client_tag"],
                error=None,
                payload={
                    "accepted_token_ids": [99],
                    "logical_kv_len": len(items[0]["input_ids"]),
                    "prefill_bank_location": {
                        "request_row": 3,
                        "bank_id": 0,
                        "bank_epoch": 0,
                        "bank_offset_blocks": 0,
                        "capacity_blocks": 2,
                        "num_blocks": 2,
                        "kv_version": 0,
                        "batch_id": "42",
                    },
                },
            )
        ]

    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=FakeBlockManager()),
        sessions={},
        submit_prefill_batch=submit_prefill_batch,
    )
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    with pytest.raises(RuntimeError, match="bank_id mismatch"):
        asyncio.run(
            facade.prefill_batch_direct(
                DirectPrefillBatchPlan(
                    bank_id=1,
                    bank_epoch=0,
                    batch_seq=42,
                    requests=(
                        DirectPrefillRequest(
                            request_id="req",
                            client_tag="target",
                            prompt_token_ids=(1,),
                            max_output_len=8,
                            bank_offset_blocks=0,
                            block_count=2,
                        ),
                    ),
                )
            )
        )


def test_process_local_facade_prefill_direct_validates_inputs() -> None:
    async def submit_prefill_batch(_items):
        raise AssertionError("invalid prefill must not reach worker")

    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=FakeBlockManager()),
        sessions={},
        submit_prefill_batch=submit_prefill_batch,
    )
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    with pytest.raises(ValueError, match="prompt_token_ids"):
        asyncio.run(
            facade.prefill_batch_direct(
                (
                    DirectPrefillRequest(
                        request_id="req",
                        client_tag="target",
                        prompt_token_ids=(),
                        max_output_len=8,
                    ),
                )
            )
        )
    with pytest.raises(ValueError, match="max_output_len"):
        asyncio.run(
            facade.prefill_batch_direct(
                (
                    DirectPrefillRequest(
                        request_id="req",
                        client_tag="target",
                        prompt_token_ids=(1,),
                        max_output_len=0,
                    ),
                )
            )
        )


def test_process_local_facade_rejects_non_bank_manager() -> None:
    worker = SimpleNamespace(
        initialized=True,
        model=SimpleNamespace(gpu_block_manager=SimpleNamespace(double_bank_enabled=False)),
    )
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    with pytest.raises(RuntimeError, match="double-bank"):
        facade.describe_banks()


def test_direct_facade_and_target_worker_do_not_import_old_engine_scheduler_helpers() -> None:
    root = Path(__file__).resolve().parents[1]
    facade_source = (root / "swiftllm" / "server" / "starsd_process_local_facade.py").read_text(encoding="utf-8")
    target_source = (root / "swiftllm" / "server" / "target_worker.py").read_text(encoding="utf-8")
    forbidden = (
        "from ." + "engine import build_batch_plan",
        "from ." + "scheduler import RequestIdManager",
        "from swiftllm.server." + "engine import build_batch_plan",
        "from swiftllm.server." + "scheduler import RequestIdManager",
    )
    for fragment in forbidden:
        assert fragment not in facade_source
        assert fragment not in target_source
        assert fragment not in (root / "swiftllm" / "server" / "draft_session.py").read_text(encoding="utf-8")
