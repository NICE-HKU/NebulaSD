"""Process-local narrow StarSD facade for SwiftLLM target workers.

This facade is intentionally not a cross-process protocol. It returns local
Python/CUDA objects to the owning worker adapter, which is responsible for
publishing numeric facts to StarSD Next shared tables.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, Sequence


@dataclasses.dataclass(frozen=True)
class LocalBankDescriptor:
    bank_id: int
    bank_epoch: int
    base_block: int
    total_blocks: int
    alloc_ptr_blocks: int
    role: str
    batch_seq: int | None = None
    capacity_rows: int = 0
    alloc_rows: int = 0

    @property
    def free_blocks(self) -> int:
        return self.total_blocks - self.alloc_ptr_blocks


@dataclasses.dataclass(frozen=True)
class StandbyPrepareItem:
    row: int
    required_blocks: int
    logical_kv_len: int
    kv_version: int


@dataclasses.dataclass(frozen=True)
class PreparedBankRange:
    row: int
    bank_id: int
    bank_epoch: int
    start_block: int
    block_count: int
    logical_kv_len: int
    kv_version: int
    batch_seq: int | None = None


@dataclasses.dataclass(frozen=True)
class LocalCudaResult:
    ok: bool
    event: Any | None = None
    value: Any | None = None


@dataclasses.dataclass(frozen=True)
class H2DCompletion:
    bank_id: int
    bank_epoch: int
    batch_seq: int | None
    ok: bool
    event: Any | None = None


@dataclasses.dataclass(frozen=True)
class DirectPrefillRequest:
    request_id: str
    client_tag: str
    prompt_token_ids: tuple[int, ...]
    max_output_len: int
    bank_offset_blocks: int = 0
    block_count: int = 0
    stop_token_ids: tuple[int, ...] = ()
    return_hidden: bool = False
    task_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DirectPrefillBatchPlan:
    bank_id: int
    bank_epoch: int
    batch_seq: int | None
    requests: tuple[DirectPrefillRequest, ...]


@dataclasses.dataclass(frozen=True)
class DirectVerifyRequest:
    request_id: str
    request_row: int
    client_tag: str
    prompt_len: int
    output_token_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]
    max_output_len: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    proposal_kind: str = "dflash_block"
    return_hidden: bool = True


@dataclasses.dataclass(frozen=True)
class DirectVerifyBatchPlan:
    active_bank_id: int
    active_bank_epoch: int
    batch_seq: int | None
    requests: tuple[DirectVerifyRequest, ...]


@dataclasses.dataclass(frozen=True)
class ExactSessionKey:
    client_tag: str
    request_id: str


@dataclasses.dataclass(frozen=True)
class ExactBankRangeRelease:
    bank_id: int
    bank_epoch: int
    row: int
    start_block: int
    capacity_blocks: int
    batch_seq: int | None


class _CopyBackend(Protocol):
    def launch_h2d_on_stream(self, *args: Any, **kwargs: Any) -> LocalCudaResult: ...

    def launch_dirty_d2h_on_stream(self, *args: Any, **kwargs: Any) -> LocalCudaResult: ...


class SwiftLLMProcessLocalTargetFacade:
    """Small backend API used by StarSD Next target worker adapters."""

    def __init__(self, engine_config: Any | None = None, *, worker: Any | None = None, copy_backend: _CopyBackend | None = None) -> None:
        if worker is None:
            if engine_config is None:
                raise ValueError("engine_config is required when worker is not provided")
            from swiftllm.server.target_worker import SwiftLLMTargetWorker

            worker = SwiftLLMTargetWorker(engine_config)
        self.worker = worker
        self.engine_config = engine_config if engine_config is not None else getattr(worker, "engine_config", None)
        self.copy_backend = copy_backend

    async def initialize(self) -> None:
        if not bool(getattr(self.worker, "initialized", False)) and hasattr(self.worker, "initialize"):
            await self.worker.initialize(start_background_loop=False)

    def describe_banks(self) -> tuple[LocalBankDescriptor, ...]:
        manager = self._require_block_manager()
        row_capacity = _row_capacity(self.worker, manager)
        descriptors = []
        for bank_id in (int(manager.active_bank_id), int(manager.standby_bank_id)):
            descriptors.append(_describe_bank(manager.get_bank_descriptor(bank_id), row_capacity=row_capacity))
        return tuple(descriptors)

    def prepare_standby_batch(
        self,
        items: Sequence[StandbyPrepareItem],
        *,
        bank_id: int | None = None,
        batch_seq: int | None = None,
    ) -> tuple[PreparedBankRange, ...]:
        manager = self._require_block_manager()
        target_bank_id = int(manager.standby_bank_id) if bank_id is None else _non_negative(bank_id, "bank_id")
        batch_seq = None if batch_seq is None else _non_negative(batch_seq, "batch_seq")
        if target_bank_id != int(manager.standby_bank_id):
            raise RuntimeError("prepare_standby_batch can only prepare the current standby bank")
        standby = manager.get_bank_descriptor(target_bank_id)
        if _bank_role(standby) not in {"STANDBY", "FREE"}:
            raise RuntimeError(f"standby bank is not prepareable: role={_bank_role(standby)}")
        old_bank_epoch = int(standby.epoch)
        normalized = tuple(_validate_prepare_item(item) for item in items)
        locations = manager.reserve_in_bank_batch_atomic(
            target_bank_id,
            [
                (item.row, item.required_blocks, item.logical_kv_len, item.kv_version, None if batch_seq is None else str(batch_seq))
                for item in normalized
            ],
            reset_bank=True,
        )
        ready = manager.mark_bank_prepared(target_bank_id, batch_id=None if batch_seq is None else str(batch_seq))
        if int(ready.epoch) != old_bank_epoch + 1:
            raise RuntimeError("prepared bank epoch did not advance exactly once")
        return tuple(
            PreparedBankRange(
                row=int(location.request_id),
                bank_id=int(location.bank_id),
                bank_epoch=int(location.bank_epoch),
                start_block=int(location.request_start_block),
                block_count=int(location.num_blocks),
                logical_kv_len=int(location.logical_kv_len),
                kv_version=int(location.kv_version),
                batch_seq=batch_seq,
            )
            for location in locations
        )

    def launch_h2d_on_stream(self, *args: Any, **kwargs: Any) -> LocalCudaResult:
        if self.copy_backend is None:
            raise RuntimeError("copy backend is not configured")
        return self.copy_backend.launch_h2d_on_stream(*args, **kwargs)

    async def prefill_batch_direct(
        self,
        requests_or_plan: Sequence[DirectPrefillRequest] | DirectPrefillBatchPlan,
    ) -> LocalCudaResult:
        """Run typed initial Target prefill through the canonical local worker.

        SwiftLLM owns worker-local row allocation. When a DirectPrefillBatchPlan
        is supplied, SwiftLLM must reserve the exact bank/ranges before forward
        and the facade validates returned locations against that plan.
        """

        plan = _validate_direct_prefill_plan(requests_or_plan)
        normalized = plan.requests
        if not normalized:
            return LocalCudaResult(ok=True, value=[])
        if not bool(getattr(self.worker, "initialized", False)):
            raise RuntimeError("SwiftLLMTargetWorker is not initialized")
        items = [_prefill_item(plan, request) for request in normalized]
        prefill = getattr(self.worker, "run_prefill_batch_direct", None)
        if prefill is None:
            prefill = getattr(self.worker, "submit_prefill_batch", None)
        if prefill is None:
            raise RuntimeError("worker does not support direct prefill")
        result = await prefill(items)
        _validate_direct_prefill_results(plan, result)
        from .starsd_copy_support import record_compute_ready
        return LocalCudaResult(ok=True, value=result, event=await record_compute_ready(self.worker))

    def activate_or_switch_bank(
        self,
        *,
        bank_id: int,
        bank_epoch: int,
        batch_seq: int | None,
        h2d_completion: H2DCompletion,
    ) -> tuple[LocalBankDescriptor, LocalBankDescriptor]:
        manager = self._require_block_manager()
        bank_id = _non_negative(bank_id, "bank_id")
        bank_epoch = _non_negative(bank_epoch, "bank_epoch")
        batch_seq = None if batch_seq is None else _non_negative(batch_seq, "batch_seq")
        if not h2d_completion.ok:
            raise RuntimeError("H2D completion is not successful")
        if (
            _non_negative(h2d_completion.bank_id, "h2d_completion.bank_id") != bank_id
            or _non_negative(h2d_completion.bank_epoch, "h2d_completion.bank_epoch") != bank_epoch
            or h2d_completion.batch_seq != batch_seq
        ):
            raise RuntimeError("H2D completion does not match bank activation fence")
        if bank_id == int(manager.active_bank_id):
            raise RuntimeError("activate_or_switch_bank requires the current standby bank")
        if bank_id != int(manager.standby_bank_id):
            raise RuntimeError("activate_or_switch_bank can only switch the current standby bank")
        standby = manager.get_bank_descriptor(bank_id)
        if int(standby.epoch) != bank_epoch:
            raise RuntimeError("standby bank epoch mismatch")
        if _bank_role(standby) not in {"PREPARED", "READY"}:
            raise RuntimeError(f"standby bank is not prepared: role={_bank_role(standby)}")
        if _parse_optional_int(getattr(standby, "batch_id", None)) != batch_seq:
            raise RuntimeError("standby bank batch sequence mismatch")
        active, new_standby = manager.swap_active_standby()
        if int(active.bank_id) != bank_id:
            raise RuntimeError("bank switch did not activate requested bank")
        row_capacity = _row_capacity(self.worker, manager)
        return (_describe_bank(active, row_capacity=row_capacity), _describe_bank(new_standby, row_capacity=row_capacity))

    async def verify_batch_direct(self, run_plan: DirectVerifyBatchPlan) -> LocalCudaResult:
        run_plan = _validate_direct_plan(run_plan)
        manager = self._require_block_manager()
        if run_plan.active_bank_id != int(manager.active_bank_id):
            raise RuntimeError("direct verify active bank id mismatch")
        active = manager.get_bank_descriptor(run_plan.active_bank_id)
        if int(active.epoch) != run_plan.active_bank_epoch:
            raise RuntimeError("direct verify active bank epoch mismatch")
        if _parse_optional_int(getattr(active, "batch_id", None)) != run_plan.batch_seq:
            raise RuntimeError("direct verify active bank batch sequence mismatch")
        if not hasattr(self.worker, "submit_verify_bank_batch"):
            raise RuntimeError("worker does not support direct bank verification")
        result = await self.worker.submit_verify_bank_batch(_legacy_verify_plan(run_plan))
        from .starsd_copy_support import record_compute_ready
        return LocalCudaResult(ok=True, value=result, event=await record_compute_ready(self.worker))

    async def verify_batch_compact(self, run_plan) -> LocalCudaResult:
        """Worker-private execution plan; no full output history is required."""
        from .starsd_compact_target import verify_compact
        return await verify_compact(self, run_plan)

    def launch_dirty_d2h_on_stream(self, *args: Any, **kwargs: Any) -> LocalCudaResult:
        if self.copy_backend is None:
            raise RuntimeError("copy backend is not configured")
        return self.copy_backend.launch_dirty_d2h_on_stream(*args, **kwargs)

    def reset_bank(self, bank_id: int) -> LocalBankDescriptor:
        manager = self._require_block_manager()
        bank_id = _non_negative(bank_id, "bank_id")
        if bank_id == int(manager.active_bank_id):
            raise RuntimeError("reset_bank cannot reset the active bank")
        return _describe_bank(manager.reset_bank(bank_id), row_capacity=_row_capacity(self.worker, manager))

    def release_session(self, session_key: Any) -> None:
        release = getattr(self.worker, "release_session", None)
        if release is not None:
            release(session_key)
            return
        sessions = getattr(self.worker, "sessions", None)
        if isinstance(sessions, dict):
            sessions.pop(session_key, None)

    async def release_exact_prefill_sessions(self, keys: Sequence[ExactSessionKey | tuple[str, str]]) -> LocalCudaResult:
        release = getattr(self.worker, "release_exact_sessions", None)
        if release is None:
            raise RuntimeError("worker does not support exact prefill session release")
        normalized = tuple(_validate_exact_session_key(key) for key in keys)
        result = release(tuple((key.client_tag, key.request_id) for key in normalized))
        if hasattr(result, "__await__"):
            result = await result
        return LocalCudaResult(ok=True, value=result)

    async def release_exact_bank_ranges(self, ranges: Sequence[ExactBankRangeRelease]) -> LocalCudaResult:
        manager = self._require_block_manager()
        normalized = tuple(_validate_exact_bank_range(item) for item in ranges)
        release_items = [
            (
                item.bank_id,
                item.bank_epoch,
                item.row,
                item.start_block,
                item.capacity_blocks,
                None if item.batch_seq is None else str(item.batch_seq),
            )
            for item in normalized
        ]
        manager.release_bank_ranges_exact_batch(release_items, apply=False)
        manager.release_bank_ranges_exact_batch(release_items)
        return LocalCudaResult(ok=True, value={"released_count": len(release_items)})

    async def shutdown(self) -> None:
        shutdown = getattr(self.worker, "shutdown", None)
        if shutdown is not None:
            result = shutdown()
            if hasattr(result, "__await__"):
                await result

    def _require_block_manager(self) -> Any:
        model = getattr(self.worker, "model", None)
        manager = getattr(model, "gpu_block_manager", None)
        if manager is None:
            raise RuntimeError("SwiftLLM block manager is not initialized")
        if not bool(getattr(manager, "double_bank_enabled", False)):
            raise RuntimeError("SwiftLLM process-local facade requires double-bank block manager")
        return manager


def _describe_bank(bank: Any, *, row_capacity: int) -> LocalBankDescriptor:
    return LocalBankDescriptor(
        bank_id=int(bank.bank_id),
        bank_epoch=int(bank.epoch),
        base_block=int(bank.base_block),
        total_blocks=int(bank.num_blocks),
        alloc_ptr_blocks=int(bank.alloc_ptr),
        role=_bank_role(bank),
        batch_seq=_parse_optional_int(getattr(bank, "batch_id", None)),
        capacity_rows=row_capacity,
        alloc_rows=len(getattr(bank, "request_ranges", {}) or {}),
    )


def _row_capacity(worker: Any, manager: Any) -> int:
    request_manager = getattr(worker, "request_id_manager", None)
    max_id = int(getattr(request_manager, "max_id", 0) or 0)
    if max_id > 0:
        return max_id
    table = getattr(manager, "num_seq_allocated_blocks", None)
    shape = getattr(table, "shape", None)
    if shape:
        return int(shape[0])
    config = getattr(worker, "engine_config", None)
    return int(getattr(config, "max_seqs_in_block_table", 0) or 0)


def _validate_prepare_item(item: StandbyPrepareItem) -> StandbyPrepareItem:
    row = _non_negative(item.row, "row")
    required_blocks = _non_negative(item.required_blocks, "required_blocks")
    logical_kv_len = _non_negative(item.logical_kv_len, "logical_kv_len")
    kv_version = _non_negative(item.kv_version, "kv_version")
    return StandbyPrepareItem(row, required_blocks, logical_kv_len, kv_version)


def _parse_optional_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bank_role(bank: Any) -> str:
    return str(getattr(getattr(bank, "role", ""), "value", getattr(bank, "role", ""))).upper()


def _validate_direct_plan(plan: DirectVerifyBatchPlan) -> DirectVerifyBatchPlan:
    active_bank_id = _non_negative(plan.active_bank_id, "active_bank_id")
    active_bank_epoch = _non_negative(plan.active_bank_epoch, "active_bank_epoch")
    batch_seq = None if plan.batch_seq is None else _non_negative(plan.batch_seq, "batch_seq")
    requests = tuple(_validate_direct_request(request) for request in plan.requests)
    return DirectVerifyBatchPlan(active_bank_id, active_bank_epoch, batch_seq, requests)


def _validate_direct_prefill_plan(
    requests_or_plan: Sequence[DirectPrefillRequest] | DirectPrefillBatchPlan,
) -> DirectPrefillBatchPlan:
    if isinstance(requests_or_plan, DirectPrefillBatchPlan):
        bank_id = _non_negative(requests_or_plan.bank_id, "bank_id")
        bank_epoch = _non_negative(requests_or_plan.bank_epoch, "bank_epoch")
        batch_seq = None if requests_or_plan.batch_seq is None else _non_negative(requests_or_plan.batch_seq, "batch_seq")
        requests = tuple(_validate_direct_prefill_request(request, require_bank_range=True) for request in requests_or_plan.requests)
        return DirectPrefillBatchPlan(bank_id, bank_epoch, batch_seq, requests)
    requests = tuple(_validate_direct_prefill_request(request, require_bank_range=False) for request in requests_or_plan)
    return DirectPrefillBatchPlan(0, 0, None, requests)


def _validate_direct_prefill_request(request: DirectPrefillRequest, *, require_bank_range: bool) -> DirectPrefillRequest:
    if not isinstance(request, DirectPrefillRequest):
        raise TypeError("prefill request must be DirectPrefillRequest")
    request_id = str(request.request_id)
    client_tag = str(request.client_tag)
    if not request_id or not client_tag:
        raise ValueError("request_id and client_tag must be non-empty")
    prompt = tuple(_non_negative(token, "prompt_token_id") for token in request.prompt_token_ids)
    if not prompt:
        raise ValueError("prompt_token_ids must be non-empty")
    max_output_len = _non_negative(request.max_output_len, "max_output_len")
    if max_output_len <= 0:
        raise ValueError("max_output_len must be positive")
    bank_offset_blocks = _non_negative(request.bank_offset_blocks, "bank_offset_blocks")
    block_count = _non_negative(request.block_count, "block_count")
    if require_bank_range and block_count <= 0:
        raise ValueError("block_count must be positive for planned prefill")
    stops = tuple(_non_negative(token, "stop_token_id") for token in request.stop_token_ids)
    task_id = None if request.task_id is None else str(request.task_id)
    if task_id == "":
        raise ValueError("task_id must be non-empty when provided")
    return DirectPrefillRequest(
        request_id=request_id,
        client_tag=client_tag,
        prompt_token_ids=prompt,
        max_output_len=max_output_len,
        bank_offset_blocks=bank_offset_blocks,
        block_count=block_count,
        stop_token_ids=stops,
        return_hidden=bool(request.return_hidden),
        task_id=task_id,
    )


def _validate_direct_prefill_results(plan: DirectPrefillBatchPlan, results: Sequence[Any]) -> None:
    if len(results) != len(plan.requests):
        raise RuntimeError("SwiftLLM prefill result count mismatch")
    for request, result in zip(plan.requests, results, strict=True):
        if getattr(result, "error", None):
            raise RuntimeError(f"SwiftLLM prefill failed: {result.error}")
        if str(getattr(result, "request_id", "")) != request.request_id:
            raise RuntimeError("SwiftLLM prefill result request_id mismatch")
        if str(getattr(result, "client_tag", "")) != request.client_tag:
            raise RuntimeError("SwiftLLM prefill result client_tag mismatch")
        payload = dict(getattr(result, "payload", {}) or {})
        if "logical_kv_len" not in payload:
            raise RuntimeError("SwiftLLM prefill result missing logical_kv_len")
        if int(payload["logical_kv_len"]) != len(request.prompt_token_ids):
            raise RuntimeError("SwiftLLM prefill result logical_kv_len mismatch")
        location = payload.get("prefill_bank_location")
        if request.block_count <= 0:
            continue
        if not isinstance(location, dict):
            raise RuntimeError("SwiftLLM prefill result missing planned bank location")
        checks = {
            "bank_id": plan.bank_id,
            "bank_epoch": plan.bank_epoch,
            "bank_offset_blocks": request.bank_offset_blocks,
            "capacity_blocks": request.block_count,
            "batch_id": None if plan.batch_seq is None else str(plan.batch_seq),
        }
        for field, expected in checks.items():
            actual = location.get(field)
            if field == "batch_id":
                actual = None if actual is None else str(actual)
            else:
                actual = int(actual)
            if actual != expected:
                raise RuntimeError(f"SwiftLLM prefill result {field} mismatch")
        if "kv_version" not in location:
            raise RuntimeError("SwiftLLM prefill result missing kv_version")
        if int(location["kv_version"]) < 0:
            raise RuntimeError("SwiftLLM prefill result kv_version is negative")
        if int(location.get("num_blocks", 0)) > request.block_count:
            raise RuntimeError("SwiftLLM prefill result valid blocks exceed planned capacity")


def _prefill_item(plan: DirectPrefillBatchPlan, request: DirectPrefillRequest) -> dict[str, Any]:
    item = {
        "task_id": request.task_id or f"{request.client_tag}:{request.request_id}:prefill",
        "client_tag": request.client_tag,
        "request_id": request.request_id,
        "input_ids": list(request.prompt_token_ids),
        "max_output_len": request.max_output_len,
        "stop_token_ids": tuple(request.stop_token_ids),
        "return_hidden": request.return_hidden,
        "keep_bank_range_for_export": True,
    }
    if request.block_count > 0:
        item["prefill_bank_plan"] = {
            "bank_id": plan.bank_id,
            "bank_epoch": plan.bank_epoch,
            "batch_id": None if plan.batch_seq is None else str(plan.batch_seq),
            "bank_offset_blocks": request.bank_offset_blocks,
            "block_count": request.block_count,
        }
    return item


def _validate_direct_request(request: DirectVerifyRequest) -> DirectVerifyRequest:
    request_id = str(request.request_id)
    client_tag = str(request.client_tag)
    if not request_id or not client_tag:
        raise ValueError("request_id and client_tag must be non-empty")
    output = tuple(_non_negative(token, "output_token_id") for token in request.output_token_ids)
    draft = tuple(_non_negative(token, "draft_token_id") for token in request.draft_token_ids)
    stops = tuple(_non_negative(token, "stop_token_id") for token in request.stop_token_ids)
    max_output_len = None if request.max_output_len is None else _non_negative(request.max_output_len, "max_output_len")
    return DirectVerifyRequest(
        request_id=request_id,
        request_row=_non_negative(request.request_row, "request_row"),
        client_tag=client_tag,
        prompt_len=_non_negative(request.prompt_len, "prompt_len"),
        output_token_ids=output,
        draft_token_ids=draft,
        max_output_len=max_output_len,
        stop_token_ids=stops,
        proposal_kind=str(request.proposal_kind),
        return_hidden=bool(request.return_hidden),
    )


def _validate_exact_session_key(key: ExactSessionKey | tuple[str, str]) -> ExactSessionKey:
    if isinstance(key, ExactSessionKey):
        client_tag = str(key.client_tag)
        request_id = str(key.request_id)
    else:
        client_tag, request_id = (str(key[0]), str(key[1]))
    if not client_tag or not request_id:
        raise ValueError("exact session release key fields must be non-empty")
    return ExactSessionKey(client_tag=client_tag, request_id=request_id)


def _validate_exact_bank_range(item: ExactBankRangeRelease) -> ExactBankRangeRelease:
    if not isinstance(item, ExactBankRangeRelease):
        raise TypeError("bank range release must be ExactBankRangeRelease")
    capacity_blocks = _non_negative(item.capacity_blocks, "capacity_blocks")
    if capacity_blocks <= 0:
        raise ValueError("capacity_blocks must be positive")
    batch_seq = None if item.batch_seq is None else _non_negative(item.batch_seq, "batch_seq")
    return ExactBankRangeRelease(
        bank_id=_non_negative(item.bank_id, "bank_id"),
        bank_epoch=_non_negative(item.bank_epoch, "bank_epoch"),
        row=_non_negative(item.row, "row"),
        start_block=_non_negative(item.start_block, "start_block"),
        capacity_blocks=capacity_blocks,
        batch_seq=batch_seq,
    )


def _legacy_verify_plan(plan: DirectVerifyBatchPlan) -> dict[str, Any]:
    return {
        "active_bank_id": plan.active_bank_id,
        "active_bank_epoch": plan.active_bank_epoch,
        "batch_id": None if plan.batch_seq is None else str(plan.batch_seq),
        "request_ids": [request.request_id for request in plan.requests],
        "request_rows": {request.request_id: request.request_row for request in plan.requests},
        "proposal_kind": "dflash_block",
        "return_hidden": any(request.return_hidden for request in plan.requests),
        "verify_payloads": {
            request.request_id: {
                "client_tag": request.client_tag,
                "request_row": request.request_row,
                "prompt_len": request.prompt_len,
                "output_token_ids": list(request.output_token_ids),
                "draft_token_ids": list(request.draft_token_ids),
                "max_output_len": request.max_output_len,
                "stop_token_ids": tuple(request.stop_token_ids),
                "proposal_kind": request.proposal_kind,
                "return_hidden": request.return_hidden,
            }
            for request in plan.requests
        },
    }


def _non_negative(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
