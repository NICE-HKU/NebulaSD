from __future__ import annotations

import asyncio
import dataclasses
import contextlib
import os
from collections import deque
from time import perf_counter
from typing import Any, Sequence

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.server.starsd_async_operations import (
    StarsDAsyncFatalError,
    StarsDAsyncOperationRegistry,
    StarsDCopyDirection,
    StarsDCopyHandle,
    StarsDCopyProgress,
)
from swiftllm.server.target_worker import SwiftLLMTargetWorker, TargetResult


def _fine_profile_enabled() -> bool:
    return os.environ.get("STARSD_PR08_FINE_PROFILE") == "1"


def _profile_start() -> float:
    return perf_counter()


def _profile_mark(profile: list[tuple[str, float]] | None, name: str, started: float) -> float:
    now = perf_counter()
    if profile is not None:
        profile.append((name, (now - started) * 1000.0))
    return now


def _attach_fine_profile(obj: object, scope: str, segments: Sequence[tuple[str, float]]) -> None:
    if not segments:
        return
    existing = tuple(getattr(obj, "starsd_fine_profile", ()))
    object.__setattr__(
        obj,
        "starsd_fine_profile",
        existing + ({"scope": str(scope), "segments_ms": tuple(segments)},),
    )


def _attach_fine_profile_many(
    objects: Sequence[object], scope: str, segments: Sequence[tuple[str, float]]
) -> None:
    for obj in objects:
        _attach_fine_profile(obj, scope, segments)


@dataclasses.dataclass(frozen=True)
class StarsDBankDescriptor:
    bank_id: int
    bank_epoch: int
    total_blocks: int
    free_blocks: int
    total_rows: int
    free_rows: int
    role: str


@dataclasses.dataclass(frozen=True)
class StarsDReservation:
    reservation_id: str
    request_id: str
    request_epoch: int
    round_id: int
    row: int
    bank_id: int
    bank_epoch: int
    start_block: int
    capacity_blocks: int
    logical_kv_len: int
    kv_version: int


@dataclasses.dataclass(frozen=True)
class StarsDPreparedBank:
    plan_id: str
    request_id: str
    request_epoch: int
    round_id: int
    bank_id: int
    bank_epoch: int
    row: int
    start_block: int
    capacity_blocks: int
    copied_blocks: int
    kv_version: int


@dataclasses.dataclass(frozen=True)
class StarsDDirtyWriteback:
    plan_id: str
    request_id: str
    copied_blocks: int
    logical_kv_len_after: int


@dataclasses.dataclass(frozen=True)
class StarsDPrefillWriteback:
    plan_id: str
    request_id: str
    request_epoch: int
    round_id: int
    copied_blocks: int
    logical_kv_len: int
    anchor_token_id: int
    output_token_ids: tuple[int, ...]
    finished: bool
    target_hidden: Any | None = None
    target_hidden_ready_event: Any | None = None


@dataclasses.dataclass(frozen=True)
class StarsDReserveRequest:
    request_id: str
    request_epoch: int
    round_id: int
    required_blocks: int
    batch_id: str


@dataclasses.dataclass(frozen=True)
class StarsDReleaseRequest:
    request_id: str
    request_epoch: int
    round_id: int
    reservation_id: str | None


@dataclasses.dataclass(frozen=True)
class StarsDResourceStats:
    active: StarsDBankDescriptor
    standby: StarsDBankDescriptor
    session_count: int
    reservation_count: int
    allocated_range_count: int
    free_row_count: int


@dataclasses.dataclass(frozen=True)
class StarsDResidentRange:
    request_id: str
    request_epoch: int
    round_id: int
    bank_id: int
    bank_epoch: int
    row: int
    start_block: int
    capacity_blocks: int
    valid_blocks: int
    kv_version: int


@dataclasses.dataclass(frozen=True)
class _BatchRecord:
    state: str
    keys: tuple[tuple[str, int, int, str], ...]
    snapshot: tuple[StarsDBankDescriptor, StarsDBankDescriptor] | None = None


class SwiftLLMStarsDTargetFacade:
    """Narrow StarSD-facing facade owned by SwiftLLM.

    The facade is the only place that touches SwiftLLM worker internals such as
    the model block manager. StarSD adapters call these methods and never read
    SwiftLLM private sessions or block tables directly.
    """

    def __init__(self, engine_config: EngineConfig, *, worker: SwiftLLMTargetWorker | None = None) -> None:
        self.worker = worker or SwiftLLMTargetWorker(engine_config)
        self.engine_config = engine_config
        row_capacity = int(getattr(engine_config, "max_seqs_in_block_table", 0) or 0)
        if row_capacity <= 0:
            raise RuntimeError("max_seqs_in_block_table must be positive")
        self._resource_rows: dict[tuple[str, int, int, str], StarsDReservation] = {}
        self._reserve_batches: dict[str, _BatchRecord] = {}
        self._released: dict[tuple[str, int, int, str | None], dict[str, Any]] = {}
        self._released_order: deque[tuple[str, int, int, str | None]] = deque()
        self._release_replay_window = 4096
        self._bank_protections: dict[tuple[int, int], set[str]] = {}
        self._protection_index: dict[str, tuple[int, int]] = {}
        self._async_ops = StarsDAsyncOperationRegistry(
            target_id="swiftllm-target-unconfigured",
            process_generation=0,
        )
        self._async_identity = ("swiftllm-target-unconfigured", 0)

    def configure_starsd_identity(self, *, target_id: str, process_generation: int) -> None:
        target_id = str(target_id)
        generation = int(process_generation)
        if not target_id or generation < 0:
            raise RuntimeError("invalid StarSD target identity")
        if self._async_identity == (target_id, generation):
            return
        if self._async_identity != ("swiftllm-target-unconfigured", 0):
            raise RuntimeError("StarSD target identity cannot change")
        self._async_ops = StarsDAsyncOperationRegistry(
            target_id=target_id,
            process_generation=generation,
        )
        self._async_identity = (target_id, generation)

    def starsd_async_operation_counts(self) -> dict[str, int]:
        return {
            "active": self._async_ops.active_operation_count,
            "replay_tombstones": self._async_ops.replay_tombstone_count,
            "abort_pending": self._async_ops.abort_pending_count,
            "bank_protections": len(self._protection_index),
        }

    def abort_starsd_async_request(
        self,
        *,
        request_id: str,
        request_epoch: int,
        round_id: int,
        target_id: str | None = None,
        process_generation: int | None = None,
    ) -> tuple[StarsDCopyProgress, ...]:
        if target_id is not None and str(target_id) != self._async_identity[0]:
            return ()
        if process_generation is not None and int(process_generation) != self._async_identity[1]:
            return ()
        return self._async_ops.abort_request(
            request_id=str(request_id),
            request_epoch=int(request_epoch),
            round_id=int(round_id),
        )

    async def initialize(self) -> None:
        if not getattr(self.worker, "initialized", False):
            await self.worker.initialize()

    async def prefill(
        self,
        *,
        client_tag: str,
        request_id: str,
        input_ids: Sequence[int],
        max_output_len: int,
        return_hidden: bool,
    ) -> TargetResult:
        return await self.worker.submit_prefill(
            client_tag=client_tag,
            request_id=request_id,
            input_ids=[int(token) for token in input_ids],
            max_output_len=int(max_output_len),
            return_hidden=bool(return_hidden),
        )

    async def decode(
        self,
        *,
        client_tag: str,
        request_id: str,
        return_hidden: bool = False,
    ) -> TargetResult:
        return await self.worker.submit_decode(
            client_tag=client_tag,
            request_id=request_id,
            return_hidden=bool(return_hidden),
        )

    def bank_snapshot(self) -> tuple[StarsDBankDescriptor, StarsDBankDescriptor]:
        manager = self._manager()
        active = manager.get_bank_descriptor(manager.active_bank_id)
        standby = manager.get_bank_descriptor(manager.standby_bank_id)
        return self._descriptor(active), self._descriptor(standby)

    def reserve_standby_batch(
        self,
        requests: Sequence[StarsDReserveRequest | dict[str, Any]],
        *,
        expected_bank_id: int | None = None,
        expected_bank_epoch: int | None = None,
    ) -> tuple[StarsDReservation, ...]:
        manager = self._manager()
        _ensure_mutable_bank_tensors(manager)
        old_standby = manager.get_bank_descriptor(manager.standby_bank_id)
        if expected_bank_id is not None and int(expected_bank_id) != int(old_standby.bank_id):
            raise RuntimeError("standby bank id mismatch")
        if expected_bank_epoch is not None and int(expected_bank_epoch) != int(old_standby.epoch):
            raise RuntimeError("standby bank epoch mismatch")
        items = tuple(_coerce_reserve_request(item) for item in requests)
        if not items:
            return ()
        seen: set[tuple[str, int, int]] = set()
        total_required = 0
        for item in items:
            key = (item.request_id, item.request_epoch, item.round_id)
            if key in seen:
                raise RuntimeError("duplicate StarSD reserve request")
            seen.add(key)
            if item.required_blocks <= 0:
                raise RuntimeError("required_blocks must be positive")
            total_required += int(item.required_blocks)
        if old_standby.request_ranges:
            active = manager.get_bank_descriptor(manager.active_bank_id)
            residual = []
            for row, location in sorted(old_standby.request_ranges.items()):
                residual.append(
                    {
                        "row": int(row),
                        "bank_id": int(getattr(location, "bank_id", old_standby.bank_id)),
                        "bank_epoch": int(getattr(location, "bank_epoch", old_standby.epoch)),
                        "request_start_block": int(getattr(location, "request_start_block", 0)),
                        "capacity_blocks": int(getattr(location, "num_blocks", 0)),
                        "logical_kv_len": int(getattr(location, "logical_kv_len", 0)),
                        "kv_version": int(getattr(location, "kv_version", 0)),
                    }
                )
            raise RuntimeError(
                "standby bank has unfinished StarSD reservations: "
                f"active_bank={{'bank_id': {int(active.bank_id)}, 'epoch': {int(active.epoch)}, 'batch_id': {active.batch_id!r}}} "
                f"standby_bank={{'bank_id': {int(old_standby.bank_id)}, 'epoch': {int(old_standby.epoch)}, 'batch_id': {old_standby.batch_id!r}}} "
                f"residual_ranges={residual}"
            )
        self._require_bank_unprotected(int(old_standby.bank_id), int(old_standby.epoch))
        if total_required > int(old_standby.num_blocks):
            raise RuntimeError("standby bank capacity exceeded")
        row_manager = self._row_manager()
        if len(items) > len(row_manager.available_ids):
            raise RuntimeError("standby row capacity exceeded")
        batch_ids = {item.batch_id for item in items}
        if len(batch_ids) != 1:
            raise RuntimeError("reserve batch must use one batch_id")
        batch_id = next(iter(batch_ids))
        if batch_id in self._reserve_batches:
            raise RuntimeError("duplicate live StarSD reserve batch_id")

        # Reset is part of the SwiftLLM-owned allocation protocol. Prevalidation
        # above makes the following bump allocations deterministic and whole-batch.
        before_rows = dict(self._resource_rows)
        before_batches = dict(self._reserve_batches)
        before_available_ids = list(row_manager.available_ids)
        allocated_keys: list[tuple[str, int, int, str]] = []
        rows: list[int] = []
        standby_bank_id = int(old_standby.bank_id)
        next_epoch = int(old_standby.epoch) + 1
        out: list[StarsDReservation] = []
        try:
            for _ in items:
                rows.append(int(row_manager.get_id()))
            with torch.inference_mode(False):
                locations = manager.reserve_in_bank_batch_atomic(
                    standby_bank_id,
                    [(row, int(item.required_blocks), 0, 0, item.batch_id) for item, row in zip(items, rows)],
                    reset_bank=True,
                )
            for item, row, location in zip(items, rows, locations):
                reservation_id = _reservation_id(item, standby_bank_id, next_epoch, row)
                reservation = StarsDReservation(
                    reservation_id=reservation_id,
                    request_id=str(item.request_id),
                    request_epoch=int(item.request_epoch),
                    round_id=int(item.round_id),
                    row=int(row),
                    bank_id=int(location.bank_id),
                    bank_epoch=int(location.bank_epoch),
                    start_block=int(location.request_start_block),
                    capacity_blocks=int(location.num_blocks),
                    logical_kv_len=int(location.logical_kv_len),
                    kv_version=int(location.kv_version),
                )
                key = _release_key_for_reservation(reservation)
                self._resource_rows[key] = reservation
                allocated_keys.append(key)
                out.append(reservation)
            self._reserve_batches[batch_id] = _BatchRecord("RESERVED", tuple(allocated_keys))
        except Exception:
            self._resource_rows = before_rows
            self._reserve_batches = before_batches
            row_manager.available_ids = before_available_ids
            raise
        return tuple(out)

    def mark_prepared_and_swap(self, *, bank_id: int, bank_epoch: int, batch_id: str) -> tuple[StarsDBankDescriptor, StarsDBankDescriptor]:
        manager = self._manager()
        descriptor = manager.get_bank_descriptor(int(bank_id))
        if int(descriptor.epoch) != int(bank_epoch):
            raise RuntimeError("prepared bank epoch is stale")
        if str(descriptor.batch_id) != str(batch_id):
            raise RuntimeError("prepared bank batch_id mismatch")
        record = self._reserve_batches.get(str(batch_id))
        if record is None:
            raise RuntimeError("unknown StarSD reserve batch_id")
        if record.state == "ACTIVATED":
            if record.snapshot is None:
                raise RuntimeError("activated batch is missing snapshot")
            return record.snapshot
        if record.state != "RESERVED":
            raise RuntimeError("StarSD reserve batch is not prepare-ready")
        old_active = manager.get_bank_descriptor(manager.active_bank_id)
        self._require_bank_unprotected(int(old_active.bank_id), int(old_active.epoch))
        with torch.inference_mode(False):
            manager.mark_bank_prepared(int(bank_id), batch_id=str(batch_id))
            active, standby = manager.swap_active_standby()
        snapshot = (self._descriptor(active), self._descriptor(standby))
        self._reserve_batches[str(batch_id)] = _BatchRecord("ACTIVATED", record.keys, snapshot)
        return snapshot

    def prepare_bank_from_host(
        self,
        items: Sequence[dict[str, Any]],
        k_views: Sequence[memoryview],
        v_views: Sequence[memoryview],
    ) -> tuple[StarsDPreparedBank, ...]:
        manager = self._manager()
        model = self.worker.model
        if model is None or model.k_cache is None or model.v_cache is None:
            raise RuntimeError("SwiftLLM KV cache is not initialized")
        _ensure_mutable_kv_cache(model)
        items = tuple(dict(item) for item in items)
        if len(items) != len(k_views) or len(items) != len(v_views):
            raise RuntimeError("prepare_bank_from_host view count mismatch")
        if not items:
            return ()
        records = self._preflight_prepare_bank_from_host(manager, model, items, tuple(k_views), tuple(v_views))
        batch_id = records[0][5]
        record = self._reserve_batches[batch_id]
        prepared: list[StarsDPreparedBank] = []
        stream = torch.cuda.Stream(device=model.k_cache.device) if model.k_cache.is_cuda else None
        try:
            with _maybe_cuda_stream(stream):
                for item, k_view, v_view, reservation, location, _batch_id, copied_blocks in records:
                    self._copy_host_to_bank(model.k_cache, k_view, location, copied_blocks, host_address=item.get("k_address"), expected_nbytes=item.get("k_nbytes"))
                    self._copy_host_to_bank(model.v_cache, v_view, location, copied_blocks, host_address=item.get("v_address"), expected_nbytes=item.get("v_nbytes"))
                    location = manager.set_bank_location_kv_version(
                        int(item["row"]),
                        bank_id=int(item["bank_id"]),
                        bank_epoch=int(item["bank_epoch"]),
                        kv_version=int(item["expected_host_version"]),
                    )
                    manager._set_valid_blocks_for_location(int(item["row"]), location, copied_blocks)
                    prepared.append(
                        StarsDPreparedBank(
                            plan_id=str(item["plan_id"]),
                            request_id=str(item["request_id"]),
                            request_epoch=int(item["request_epoch"]),
                            round_id=int(item["round_id"]),
                            bank_id=int(item["bank_id"]),
                            bank_epoch=int(item["bank_epoch"]),
                            row=int(item["row"]),
                            start_block=int(item["start_block"]),
                            capacity_blocks=int(item["capacity_blocks"]),
                            copied_blocks=copied_blocks,
                            kv_version=int(location.kv_version),
                        )
                    )
                ready_event = None
                if stream is not None:
                    ready_event = torch.cuda.Event()
                    ready_event.record(stream)
                    stream.synchronize()
        except Exception:
            if stream is not None:
                stream.synchronize()
            raise
        first = prepared[0]
        manager.mark_bank_prepared(int(first.bank_id), batch_id=batch_id, ready_event=ready_event)
        active, standby = manager.swap_active_standby()
        self._reserve_batches[batch_id] = dataclasses.replace(record, state="ACTIVATED", snapshot=(self._descriptor(active), self._descriptor(standby)))
        return tuple(prepared)

    def stage_h2d_from_host(
        self,
        items: Sequence[dict[str, Any]],
        k_views: Sequence[memoryview],
        v_views: Sequence[memoryview],
    ) -> tuple[StarsDCopyHandle, ...]:
        profile: list[tuple[str, float]] | None = [] if _fine_profile_enabled() else None
        marker = _profile_start()
        manager = self._manager()
        model = self.worker.model
        if model is None or model.k_cache is None or model.v_cache is None:
            raise RuntimeError("SwiftLLM KV cache is not initialized")
        _ensure_mutable_kv_cache(model)
        items = tuple(dict(item) for item in items)
        marker = _profile_mark(profile, "coerce_items_and_model", marker)
        if len(items) != len(k_views) or len(items) != len(v_views):
            raise RuntimeError("stage_h2d_from_host view count mismatch")
        if not items:
            return ()
        handles = self._h2d_handles_for_items(items)
        prepared_items = self._prepared_items_from_h2d_items(items)
        fingerprints = tuple(repr((item, prepared)) for item, prepared in zip(items, prepared_items, strict=True))
        marker = _profile_mark(profile, "build_handles_fingerprints", marker)
        admission = self._async_ops.reserve_batch(handles, fingerprints)
        marker = _profile_mark(profile, "async_reserve_batch", marker)
        if admission.replayed:
            return admission.handles
        records = ()
        prepared: list[StarsDPreparedBank] = []
        stream_start = _profile_start()
        stream = torch.cuda.Stream(device=model.k_cache.device) if model.k_cache.is_cuda else None
        _profile_mark(profile, "create_cuda_stream", stream_start)
        mutation_started = False
        try:
            records = self._preflight_prepare_bank_from_host(manager, model, items, tuple(k_views), tuple(v_views))
            marker = _profile_mark(profile, "preflight_prepare_bank_from_host", marker)
            batch_id = records[0][5]
            record = self._reserve_batches[batch_id]
            start_event = None
            ready_event = None
            with _maybe_cuda_stream(stream):
                if stream is not None:
                    event_marker = _profile_start()
                    start_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(stream)
                    _profile_mark(profile, "record_start_event", event_marker)
                for item, k_view, v_view, reservation, location, _batch_id, copied_blocks in records:
                    mutation_started = True
                    self._copy_host_to_bank(
                        model.k_cache,
                        k_view,
                        location,
                        copied_blocks,
                        host_address=item.get("k_address"),
                        expected_nbytes=item.get("k_nbytes"),
                        profile=profile,
                        label="h2d_k",
                    )
                    self._copy_host_to_bank(
                        model.v_cache,
                        v_view,
                        location,
                        copied_blocks,
                        host_address=item.get("v_address"),
                        expected_nbytes=item.get("v_nbytes"),
                        profile=profile,
                        label="h2d_v",
                    )
                    location = manager.set_bank_location_kv_version(
                        int(item["row"]),
                        bank_id=int(item["bank_id"]),
                        bank_epoch=int(item["bank_epoch"]),
                        kv_version=int(item["expected_host_version"]),
                    )
                    manager._set_valid_blocks_for_location(int(item["row"]), location, copied_blocks)
                    prepared.append(
                        StarsDPreparedBank(
                            plan_id=str(item["plan_id"]),
                            request_id=str(item["request_id"]),
                            request_epoch=int(item["request_epoch"]),
                            round_id=int(item["round_id"]),
                            bank_id=int(item["bank_id"]),
                            bank_epoch=int(item["bank_epoch"]),
                            row=int(item["row"]),
                            start_block=int(item["start_block"]),
                            capacity_blocks=int(item["capacity_blocks"]),
                            copied_blocks=copied_blocks,
                            kv_version=int(location.kv_version),
                        )
                    )
                marker = _profile_mark(profile, "h2d_copy_loop_and_bank_metadata", marker)
                if stream is not None:
                    event_marker = _profile_start()
                    ready_event = torch.cuda.Event(enable_timing=True)
                    ready_event.record(stream)
                    _profile_mark(profile, "record_ready_event", event_marker)
            prepared_items = tuple(prepared)
            self._reserve_batches[batch_id] = dataclasses.replace(record, state="STAGING")
            self._async_ops.commit_batch(handles, event=ready_event, start_event=start_event, results=prepared_items)
            _profile_mark(profile, "async_commit_batch", marker)
            _attach_fine_profile_many(handles, "facade_stage_h2d", tuple(profile or ()))
        except Exception:
            if stream is not None:
                stream.synchronize()
            if not mutation_started:
                self._async_ops.rollback_admission(handles)
                raise
            raise StarsDAsyncFatalError("H2D stage failed after target bank mutation started") from None
        return handles

    def progress_h2d(self, handles: Sequence[StarsDCopyHandle]) -> tuple[StarsDCopyProgress, ...]:
        profile: list[tuple[str, float]] | None = [] if _fine_profile_enabled() else None
        marker = _profile_start()
        progress = tuple(self._async_ops.progress(handle) for handle in handles)
        marker = _profile_mark(profile, "async_progress_batch", marker)
        if progress and all(item.status.value == "ready" for item in progress):
            batch_id = progress[0].handle.batch_id
            record = self._reserve_batches.get(batch_id)
            if record is not None and record.state == "STAGING":
                self._reserve_batches[batch_id] = dataclasses.replace(record, state="EVENT_READY")
        _profile_mark(profile, "reserve_batch_state_update", marker)
        _attach_fine_profile_many(progress, "facade_progress_h2d", tuple(profile or ()))
        return progress

    def activate_h2d_batch(self, handles: Sequence[StarsDCopyHandle]) -> tuple[StarsDPreparedBank, ...]:
        handles = tuple(handles)
        if not handles:
            return ()
        progress = self.progress_h2d(handles)
        if any(item.status.value != "ready" for item in progress):
            raise RuntimeError("H2D event is not ready")
        batch_id = handles[0].batch_id
        return tuple(
            self._async_ops.finalize_batch(
                handles,
                expected_direction=StarsDCopyDirection.H2D,
                finalize=lambda batch_id=batch_id, handles=handles: self._activate_h2d_batch_once(batch_id, handles),
            )
        )

    def _activate_h2d_batch_once(
        self,
        batch_id: str,
        handles: tuple[StarsDCopyHandle, ...],
    ) -> tuple[StarsDPreparedBank, ...]:
        record = self._reserve_batches.get(batch_id)
        if record is None:
            raise RuntimeError("unknown StarSD reserve batch_id")
        if record.state == "ACTIVATED":
            return tuple(self._prepared_items_from_h2d_handles(handles))
        if record.state not in {"STAGING", "EVENT_READY"}:
            raise RuntimeError("StarSD H2D batch is not activation-ready")
        manager = self._manager()
        prepared_items = tuple(self._prepared_items_from_h2d_handles(handles))
        first = prepared_items[0]
        descriptor = manager.get_bank_descriptor(int(first.bank_id))
        if int(descriptor.epoch) != int(first.bank_epoch):
            raise RuntimeError("prepared bank epoch is stale")
        old_active = manager.get_bank_descriptor(manager.active_bank_id)
        self._require_bank_unprotected(int(old_active.bank_id), int(old_active.epoch))
        with torch.inference_mode(False):
            manager.mark_bank_prepared(int(first.bank_id), batch_id=batch_id)
            active, standby = manager.swap_active_standby()
        self._reserve_batches[batch_id] = dataclasses.replace(record, state="ACTIVATED", snapshot=(self._descriptor(active), self._descriptor(standby)))
        return prepared_items

    async def prefill_to_host(
        self,
        items: Sequence[dict[str, Any]],
        k_views: Sequence[memoryview],
        v_views: Sequence[memoryview],
        *,
        output_spec: Any,
    ) -> tuple[StarsDPrefillWriteback, ...]:
        model = self.worker.model
        if model is None or model.k_cache is None or model.v_cache is None:
            raise RuntimeError("SwiftLLM KV cache is not initialized")
        return_hidden = self._return_hidden_for_output_spec(output_spec, phase="prefill")
        items = tuple(dict(item) for item in items)
        if len(items) != len(k_views) or len(items) != len(v_views):
            raise RuntimeError("prefill_to_host view count mismatch")
        self._preflight_prefill_to_host(model, items, tuple(k_views), tuple(v_views))
        tasks = tuple(
            {
                "task_id": str(item["plan_id"]),
                "client_tag": _prefill_client_tag(item),
                "request_id": str(item["request_id"]),
                "input_ids": tuple(int(token) for token in item["input_ids"]),
                "max_output_len": int(item["max_output_len"]),
                "stop_token_ids": tuple(int(token) for token in item.get("stop_token_ids", ())),
                "return_hidden": return_hidden,
                "keep_bank_range_for_export": True,
            }
            for item in items
        )
        results = await self.worker.submit_prefill_batch(tasks)
        try:
            records = self._validate_prefill_to_host_results(model, items, tuple(k_views), tuple(v_views), tuple(results))
        except Exception:
            await self._release_prefill_sessions_for_items(items)
            raise

        out: list[StarsDPrefillWriteback] = []
        stream = torch.cuda.Stream(device=model.k_cache.device) if model.k_cache.is_cuda else None
        try:
            with _maybe_cuda_stream(stream):
                for (
                    item,
                    k_view,
                    v_view,
                    location,
                    copied_blocks,
                    logical_kv_len,
                    output_ids,
                    finished,
                    target_hidden,
                    target_hidden_ready_event,
                ) in records:
                    self._copy_bank_to_host(model.k_cache, k_view, location, 0, copied_blocks)
                    self._copy_bank_to_host(model.v_cache, v_view, location, 0, copied_blocks)
                    out.append(
                        StarsDPrefillWriteback(
                            str(item["plan_id"]),
                            str(item["request_id"]),
                            int(item["request_epoch"]),
                            int(item["round_id"]),
                            copied_blocks,
                            logical_kv_len,
                            int(output_ids[-1]),
                            output_ids,
                            finished,
                            target_hidden,
                            target_hidden_ready_event,
                        )
                    )
        except Exception:
            if stream is not None:
                stream.synchronize()
            await self._release_prefill_sessions_for_items(items)
            raise
        if stream is not None:
            stream.synchronize()
        return tuple(out)

    def export_dirty_to_host(
        self,
        items: Sequence[dict[str, Any]],
        k_views: Sequence[memoryview],
        v_views: Sequence[memoryview],
    ) -> tuple[StarsDDirtyWriteback, ...]:
        manager = self._manager()
        model = self.worker.model
        if model is None or model.k_cache is None or model.v_cache is None:
            raise RuntimeError("SwiftLLM KV cache is not initialized")
        _ensure_mutable_kv_cache(model)
        items = tuple(dict(item) for item in items)
        if len(items) != len(k_views) or len(items) != len(v_views):
            raise RuntimeError("export_dirty_to_host view count mismatch")
        records = self._preflight_export_dirty_to_host(manager, model, items, tuple(k_views), tuple(v_views))
        out: list[StarsDDirtyWriteback] = []
        stream = torch.cuda.Stream(device=model.k_cache.device) if model.k_cache.is_cuda else None
        try:
            with _maybe_cuda_stream(stream):
                for item, k_view, v_view, location, dirty_begin, dirty_count in records:
                    self._copy_bank_to_host(model.k_cache, k_view, location, dirty_begin, dirty_count)
                    self._copy_bank_to_host(model.v_cache, v_view, location, dirty_begin, dirty_count)
                    out.append(
                        StarsDDirtyWriteback(
                            plan_id=str(item["plan_id"]),
                            request_id=str(item["request_id"]),
                            copied_blocks=dirty_count,
                            logical_kv_len_after=int(item["post_crop_logical_kv_len"]),
                        )
                    )
        except Exception:
            if stream is not None:
                stream.synchronize()
            raise
        if stream is not None:
            stream.synchronize()
        return tuple(out)

    def stage_dirty_to_host(
        self,
        items: Sequence[dict[str, Any]],
        k_views: Sequence[memoryview],
        v_views: Sequence[memoryview],
    ) -> tuple[StarsDCopyHandle, ...]:
        profile: list[tuple[str, float]] | None = [] if _fine_profile_enabled() else None
        marker = _profile_start()
        manager = self._manager()
        model = self.worker.model
        if model is None or model.k_cache is None or model.v_cache is None:
            raise RuntimeError("SwiftLLM KV cache is not initialized")
        _ensure_mutable_kv_cache(model)
        items = tuple(dict(item) for item in items)
        marker = _profile_mark(profile, "coerce_items_and_model", marker)
        if len(items) != len(k_views) or len(items) != len(v_views):
            raise RuntimeError("stage_dirty_to_host view count mismatch")
        handles = self._d2h_handles_for_items(items)
        expected_results = tuple(self._dirty_result_from_item(item) for item in items)
        fingerprints = tuple(repr((item, result)) for item, result in zip(items, expected_results, strict=True))
        marker = _profile_mark(profile, "build_handles_fingerprints", marker)
        admission = self._async_ops.reserve_batch(handles, fingerprints)
        marker = _profile_mark(profile, "async_reserve_batch", marker)
        if admission.replayed:
            return admission.handles
        records = ()
        out: list[StarsDDirtyWriteback] = []
        stream_start = _profile_start()
        stream = torch.cuda.Stream(device=model.k_cache.device) if model.k_cache.is_cuda else None
        _profile_mark(profile, "create_cuda_stream", stream_start)
        mutation_started = False
        try:
            records = self._preflight_export_dirty_to_host(manager, model, items, tuple(k_views), tuple(v_views))
            marker = _profile_mark(profile, "preflight_export_dirty_to_host", marker)
            start_event = None
            ready_event = None
            with _maybe_cuda_stream(stream):
                if stream is not None:
                    event_marker = _profile_start()
                    start_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(stream)
                    _profile_mark(profile, "record_start_event", event_marker)
                for item, k_view, v_view, location, dirty_begin, dirty_count in records:
                    mutation_started = True
                    self._copy_bank_to_host(
                        model.k_cache,
                        k_view,
                        location,
                        dirty_begin,
                        dirty_count,
                        profile=profile,
                        label="d2h_k",
                    )
                    self._copy_bank_to_host(
                        model.v_cache,
                        v_view,
                        location,
                        dirty_begin,
                        dirty_count,
                        profile=profile,
                        label="d2h_v",
                    )
                    out.append(
                        StarsDDirtyWriteback(
                            plan_id=str(item["plan_id"]),
                            request_id=str(item["request_id"]),
                            copied_blocks=dirty_count,
                            logical_kv_len_after=int(item["post_crop_logical_kv_len"]),
                        )
                    )
                marker = _profile_mark(profile, "d2h_copy_loop_and_results", marker)
                if stream is not None:
                    event_marker = _profile_start()
                    ready_event = torch.cuda.Event(enable_timing=True)
                    ready_event.record(stream)
                    _profile_mark(profile, "record_ready_event", event_marker)
            self._async_ops.commit_batch(handles, event=ready_event, start_event=start_event, results=tuple(out))
            _profile_mark(profile, "async_commit_batch", marker)
            _attach_fine_profile_many(handles, "facade_stage_d2h", tuple(profile or ()))
        except Exception:
            if stream is not None:
                stream.synchronize()
            if not mutation_started:
                self._async_ops.rollback_admission(handles)
                raise
            raise StarsDAsyncFatalError("D2H stage failed after target bank copy started") from None
        return handles

    def progress_dirty_to_host(self, handles: Sequence[StarsDCopyHandle]) -> tuple[StarsDCopyProgress, ...]:
        profile: list[tuple[str, float]] | None = [] if _fine_profile_enabled() else None
        marker = _profile_start()
        progress = tuple(self._async_ops.progress(handle) for handle in handles)
        _profile_mark(profile, "async_progress_batch", marker)
        _attach_fine_profile_many(progress, "facade_progress_d2h", tuple(profile or ()))
        return progress

    def finish_dirty_to_host(self, handles: Sequence[StarsDCopyHandle]) -> tuple[StarsDDirtyWriteback, ...]:
        profile: list[tuple[str, float]] | None = [] if _fine_profile_enabled() else None
        marker = _profile_start()
        handles = tuple(handles)
        marker = _profile_mark(profile, "coerce_handles", marker)
        progress = self.progress_dirty_to_host(handles)
        marker = _profile_mark(profile, "progress_dirty_to_host_recheck", marker)
        if any(item.status.value != "ready" for item in progress):
            raise RuntimeError("D2H event is not ready")
        results = tuple(
            self._async_ops.finalize_batch(handles, expected_direction=StarsDCopyDirection.D2H)
        )
        _profile_mark(profile, "async_finalize_batch", marker)
        _attach_fine_profile_many(handles, "facade_finish_d2h", tuple(profile or ()))
        return results

    def _validate_prefill_to_host_results(
        self,
        model: Any,
        items: tuple[dict[str, Any], ...],
        k_views: tuple[memoryview, ...],
        v_views: tuple[memoryview, ...],
        results: tuple[TargetResult, ...],
    ):
        if len(results) != len(items):
            raise RuntimeError("SwiftLLM prefill result count mismatch")
        manager = self._manager()
        records = []
        for item, k_view, v_view, result in zip(items, k_views, v_views, results, strict=True):
            if result.error:
                raise RuntimeError(f"SwiftLLM prefill failed: {result.error}")
            if str(result.request_id) != str(item["request_id"]):
                raise RuntimeError("SwiftLLM prefill result request_id mismatch")
            client_tag = _prefill_client_tag(item)
            if str(result.client_tag) != client_tag:
                raise RuntimeError("SwiftLLM prefill result client_tag mismatch")
            session_key = (client_tag, str(item["request_id"]))
            session = getattr(self.worker, "sessions", {}).get(session_key)
            if session is None:
                raise RuntimeError("SwiftLLM prefill exact session is missing")
            if getattr(session, "bank_adapter", False):
                raise RuntimeError("SwiftLLM prefill session unexpectedly uses bank adapter")

            payload = dict(result.payload)
            location_payload = payload.get("prefill_bank_location")
            if not isinstance(location_payload, dict):
                raise RuntimeError("SwiftLLM prefill result missing bank location")
            required_fields = (
                "request_row",
                "bank_id",
                "bank_epoch",
                "bank_base_block",
                "bank_offset_blocks",
                "num_blocks",
                "logical_kv_len",
                "prompt_len",
            )
            for field in required_fields:
                if field not in location_payload:
                    raise RuntimeError(f"SwiftLLM prefill bank location missing {field}")
            row = int(location_payload["request_row"])
            if int(session.request.request_id) != row:
                raise RuntimeError("SwiftLLM prefill session row/location row mismatch")
            descriptor = manager.get_bank_descriptor(int(location_payload["bank_id"]))
            if int(descriptor.epoch) != int(location_payload["bank_epoch"]):
                raise RuntimeError("SwiftLLM prefill bank descriptor epoch mismatch")
            location = descriptor.request_ranges.get(row)
            if location is None:
                raise RuntimeError("SwiftLLM prefill location is not live")
            actual = (
                int(location.bank_id),
                int(location.bank_epoch),
                int(location.bank_base_block),
                int(location.request_start_block),
                int(location.num_blocks),
            )
            expected = (
                int(location_payload["bank_id"]),
                int(location_payload["bank_epoch"]),
                int(location_payload["bank_base_block"]),
                int(location_payload["bank_offset_blocks"]),
                int(location_payload["num_blocks"]),
            )
            if actual != expected:
                raise RuntimeError("SwiftLLM prefill live bank location fence mismatch")
            copied_blocks = int(location_payload["num_blocks"])
            logical_kv_len = int(location_payload["logical_kv_len"])
            if int(location_payload["prompt_len"]) != int(item["expected_logical_kv_len"]):
                raise RuntimeError("SwiftLLM prefill prompt length mismatch")
            if copied_blocks != int(item["expected_committed_blocks"]) or logical_kv_len != int(item["expected_logical_kv_len"]):
                raise RuntimeError("SwiftLLM prefill copied block/logical length mismatch")
            expected_bytes = self._plane_bytes(model.k_cache, location, 0, copied_blocks)
            if len(k_view) != expected_bytes or len(v_view) != expected_bytes:
                raise RuntimeError("prefill D2H K/V view byte length mismatch")
            output_ids = tuple(int(token) for token in payload.get("output_token_ids", ()))
            if not output_ids:
                raise RuntimeError("SwiftLLM prefill result missing output tokens")
            target_hidden = payload.get("target_hidden")
            if target_hidden is not None:
                expected_shape = (1, logical_kv_len, int(target_hidden.shape[-1]))
                if tuple(int(dim) for dim in target_hidden.shape) != expected_shape:
                    raise RuntimeError("SwiftLLM prefill hidden shape mismatch")
            target_hidden_ready_event = payload.get("target_hidden_ready_event")
            if target_hidden is None and target_hidden_ready_event is not None:
                raise RuntimeError("SwiftLLM prefill hidden event without hidden tensor")
            records.append((
                item,
                k_view,
                v_view,
                location,
                copied_blocks,
                logical_kv_len,
                output_ids,
                bool(payload.get("finished", False)),
                target_hidden,
                target_hidden_ready_event,
            ))
        return tuple(records)

    def protect_bank_epoch(self, protection_id: str, bank_id: int, bank_epoch: int) -> None:
        self.protect_bank_epochs(((protection_id, bank_id, bank_epoch),))

    def protect_bank_epochs(self, protections: Sequence[tuple[str, int, int]]) -> None:
        items = tuple((str(item[0]), int(item[1]), int(item[2])) for item in protections)
        if not items:
            return
        seen: set[str] = set()
        manager = self._manager()
        for protection_id, bank_id, bank_epoch in items:
            if not protection_id:
                raise RuntimeError("protection_id must be non-empty")
            if protection_id in seen:
                raise RuntimeError("duplicate protection_id in batch")
            seen.add(protection_id)
            old = self._protection_index.get(protection_id)
            if old is not None and old != (bank_id, bank_epoch):
                raise RuntimeError("protection_id already bound to another bank epoch")
            descriptor = manager.get_bank_descriptor(bank_id)
            if int(descriptor.epoch) != bank_epoch:
                raise RuntimeError("bank protection epoch is stale")
        for protection_id, bank_id, bank_epoch in items:
            key = (bank_id, bank_epoch)
            self._bank_protections.setdefault(key, set()).add(protection_id)
            self._protection_index[protection_id] = key

    def release_bank_epoch(self, protection_id: str, bank_id: int, bank_epoch: int) -> None:
        self.release_bank_epochs(((protection_id, bank_id, bank_epoch),))

    def release_bank_epochs(self, protections: Sequence[tuple[str, int, int]]) -> None:
        items = tuple((str(item[0]), int(item[1]), int(item[2])) for item in protections)
        if not items:
            return
        seen: set[str] = set()
        for protection_id, bank_id, bank_epoch in items:
            if protection_id in seen:
                raise RuntimeError("duplicate protection_id in release batch")
            seen.add(protection_id)
            key = self._protection_index.get(protection_id)
            if key != (bank_id, bank_epoch):
                raise RuntimeError("unknown bank protection")
        for protection_id, bank_id, bank_epoch in items:
            key = (bank_id, bank_epoch)
            holders = self._bank_protections.get(key)
            if holders is None or protection_id not in holders:
                raise RuntimeError("unknown bank protection")
            holders.remove(protection_id)
            if not holders:
                self._bank_protections.pop(key, None)
            self._protection_index.pop(protection_id, None)

    def reuse_resident_batch(self, items: Sequence[dict[str, Any]]) -> tuple[StarsDResidentRange, ...]:
        manager = self._manager()
        active = manager.get_bank_descriptor(manager.active_bank_id)
        records = self._preflight_reuse_resident(manager, active, tuple(dict(item) for item in items))
        out: list[StarsDResidentRange] = []
        for item, old_key, new_key, reservation, location in records:
            current = dataclasses.replace(reservation, round_id=int(item["round_id"]))
            if old_key is not None:
                self._resource_rows.pop(old_key)
                self._resource_rows[new_key] = current
                self._replace_batch_key(old_key, new_key)
            out.append(
                StarsDResidentRange(
                    current.request_id,
                    current.request_epoch,
                    current.round_id,
                    current.bank_id,
                    current.bank_epoch,
                    current.row,
                    current.start_block,
                    current.capacity_blocks,
                    int(item["valid_blocks"]),
                    int(location.kv_version),
                )
            )
        return tuple(out)

    async def verify_bank_batch(self, run_plan: dict[str, Any]) -> list[TargetResult]:
        manager = self._manager()
        active = manager.get_bank_descriptor(manager.active_bank_id)
        batch_id = run_plan.get("batch_id")
        if batch_id is None or str(active.batch_id) != str(batch_id):
            raise RuntimeError("verify run plan batch_id does not match active bank")
        record = self._reserve_batches.get(str(batch_id))
        if record is None or record.state != "ACTIVATED":
            raise RuntimeError("verify requires an activated StarSD reserve batch")
        return await self.worker.submit_verify_bank_batch(dict(run_plan))

    def validate_target_output_spec(self, output_spec: Any, *, phase: str) -> None:
        self._return_hidden_for_output_spec(output_spec, phase=phase)

    async def release(self, requests: Sequence[StarsDReleaseRequest | dict[str, Any]]) -> dict[str, Any]:
        items = tuple(_coerce_release_request(item) for item in requests)
        if len(items) == 1:
            replay = self._released.get((items[0].request_id, items[0].request_epoch, items[0].round_id, items[0].reservation_id))
            if replay is not None:
                return dict(replay["ack"])
        request_ids: list[str] = []
        client_tags: dict[str, str] = {}
        exact_session_keys: list[tuple[str, str]] = []
        exact_prefill_keys: list[tuple[str, str]] = []
        exact_prefill_live: list[tuple[StarsDReleaseRequest, tuple[str, int, int, None], tuple[str, str]]] = []
        live: list[tuple[StarsDReleaseRequest, tuple[str, int, int, str], StarsDReservation, bool, tuple[str, str]]] = []
        passthrough: list[StarsDReleaseRequest] = []
        replayed_rows: list[dict[str, Any]] = []
        replayed_prefill = False
        for item in items:
            if item.reservation_id is not None:
                key = (item.request_id, item.request_epoch, item.round_id, item.reservation_id)
                replay = self._released.get(key)
                if replay is not None:
                    replayed_rows.extend(replay["ack"].get("released_rows", ()))
                    continue
                reservation = self._resource_rows.get(key)
                if reservation is None:
                    raise RuntimeError(f"unknown exact reservation release: {key}")
                session_key = (str(item.reservation_id), str(item.request_id))
                had_session = session_key in getattr(self.worker, "sessions", {})
                exact_session_keys.append(session_key)
                live.append((item, key, reservation, had_session, session_key))
                continue
            key = (item.request_id, item.request_epoch, item.round_id, None)
            replay = self._released.get(key)
            if replay is not None:
                replayed_prefill = True
                continue
            passthrough.append(item)
            prefill_key = (_prefill_client_tag_for_release(item), str(item.request_id))
            exact_prefill_keys.append(prefill_key)
            exact_prefill_live.append((item, key, prefill_key))
        if not request_ids and not exact_session_keys and not exact_prefill_keys:
            out = {"status": "ok", "backend": {"status": "ok", "replayed": bool(replayed_rows)}, "released_rows": replayed_rows}
            if replayed_prefill:
                out["prefill_backend"] = {"status": "ok", "released_count": 0, "released": [], "replayed": True}
            return out
        manager = self._manager() if live else None
        release_ranges = []
        if manager is not None:
            _ensure_mutable_bank_tensors(manager)
            release_ranges = [
                (
                    reservation.bank_id,
                    reservation.bank_epoch,
                    reservation.row,
                    reservation.start_block,
                    reservation.capacity_blocks,
                    _batch_id_from_reservation_id(reservation.reservation_id),
                )
                for _item, _key, reservation, _had_session, _session_key in live
            ]
            manager.release_bank_ranges_exact_batch(release_ranges, apply=False)
        backend_result = {"status": "ok", "released_count": 0, "released": []}
        if request_ids or exact_session_keys:
            backend_result = await self.worker.release_bank_adapter_sessions(
                request_ids=request_ids or None,
                client_tags=client_tags or None,
                exact_session_keys=tuple(exact_session_keys) or None,
            )
        prefill_result = {"status": "ok", "released_count": 0, "released": []}
        if exact_prefill_keys:
            prefill_result = await self.worker.release_exact_sessions(tuple(exact_prefill_keys))
            released_prefill = {
                (str(item.get("client_tag")), str(item.get("request_id")))
                for item in prefill_result.get("released", ())
            }
            missing_prefill = [key for key in exact_prefill_keys if key not in released_prefill]
            if missing_prefill:
                raise RuntimeError(f"SwiftLLM release did not clear exact prefill sessions: {missing_prefill}")
        remaining_sessions = getattr(self.worker, "sessions", {})
        still_live = [session_key for *_prefix, had_session, session_key in live if had_session and session_key in remaining_sessions]
        if still_live:
            raise RuntimeError(f"SwiftLLM release did not clear exact bank-adapter sessions: {still_live}")
        if manager is not None:
            with torch.inference_mode():
                manager.release_bank_ranges_exact_batch(release_ranges)
            residual = []
            for bank_id, bank_epoch, row, start_block, capacity_blocks, batch_id in release_ranges:
                bank = manager.get_bank_descriptor(int(bank_id))
                location = bank.request_ranges.get(int(row))
                if location is not None:
                    residual.append((int(bank_id), int(bank_epoch), int(row), int(start_block), int(capacity_blocks), batch_id))
            if residual:
                raise RuntimeError(f"SwiftLLM release did not clear exact reservation ranges: {residual}")
        released_rows = list(replayed_rows)
        for item, key, reservation, had_session, _session_key in live:
            if not had_session:
                self._free_row(reservation.row)
            self._resource_rows.pop(key, None)
            self._remove_from_batches(key)
            released_rows.append({"request_id": item.request_id, "reservation_id": item.reservation_id, "row": reservation.row})
        ack = {"status": "ok", "backend": backend_result, "released_rows": released_rows}
        if exact_prefill_keys:
            ack["prefill_backend"] = prefill_result
        for _item, key, _prefill_key in exact_prefill_live:
            self._remember_release(key, {
                "ack": {
                    "status": "ok",
                    "backend": backend_result,
                    "released_rows": [],
                    "prefill_backend": {"status": "ok", "released_count": 0, "released": [], "replayed": True},
                }
            })
        for _item, key, reservation, _had_session, _session_key in live:
            row_ack = {
                "status": "ok",
                "backend": backend_result,
                "released_rows": [{"request_id": reservation.request_id, "reservation_id": reservation.reservation_id, "row": reservation.row}],
            }
            self._remember_release(key, {"ack": row_ack})
        if passthrough and not live and not replayed_rows:
            out = {"status": "ok", "backend": backend_result, "released_rows": []}
            if exact_prefill_keys:
                out["prefill_backend"] = prefill_result
            elif replayed_prefill:
                out["prefill_backend"] = {"status": "ok", "released_count": 0, "released": [], "replayed": True}
            return out
        if replayed_prefill and "prefill_backend" not in ack:
            ack["prefill_backend"] = {"status": "ok", "released_count": 0, "released": [], "replayed": True}
        return ack

    def resource_stats(self) -> StarsDResourceStats:
        active, standby = self.bank_snapshot()
        manager = self._manager()
        allocated = sum(len(manager.get_bank_descriptor(bank_id).request_ranges) for bank_id in (active.bank_id, standby.bank_id))
        session_count = sum(
            1
            for (client_tag, _request_id), session in getattr(self.worker, "sessions", {}).items()
            if getattr(session, "bank_adapter", False) or str(client_tag).startswith("starsd-prefill:")
        )
        return StarsDResourceStats(active, standby, int(session_count), len(self._resource_rows), int(allocated), len(self._available_rows()))

    def resident_ranges(self) -> tuple[StarsDResidentRange, ...]:
        manager = self._manager()
        active = manager.get_bank_descriptor(manager.active_bank_id)
        out: list[StarsDResidentRange] = []
        for reservation in self._resource_rows.values():
            if int(reservation.bank_id) != int(active.bank_id) or int(reservation.bank_epoch) != int(active.epoch):
                continue
            location = active.request_ranges.get(int(reservation.row))
            if location is None:
                continue
            if int(location.num_blocks) != int(reservation.capacity_blocks):
                raise RuntimeError("resident range capacity fence mismatch")
            valid_blocks = int(manager.num_seq_allocated_blocks[int(reservation.row)].item())
            if valid_blocks < 0 or valid_blocks > int(reservation.capacity_blocks):
                raise RuntimeError("resident range valid blocks exceed reservation capacity")
            out.append(
                StarsDResidentRange(
                    reservation.request_id,
                    reservation.request_epoch,
                    reservation.round_id,
                    reservation.bank_id,
                    reservation.bank_epoch,
                    reservation.row,
                    int(location.request_start_block),
                    reservation.capacity_blocks,
                    valid_blocks,
                    int(location.kv_version),
                )
            )
        return tuple(out)

    async def shutdown(self) -> None:
        errors: list[str] = []
        try:
            self._async_ops.drain()
        except Exception as exc:
            errors.append(f"async operation drain failed: {exc}")
        exact_release = getattr(self.worker, "release_exact_sessions", None)
        if callable(exact_release):
            keys = tuple(
                (str(client_tag), str(request_id))
                for client_tag, request_id in getattr(self.worker, "sessions", {})
                if str(client_tag).startswith("starsd-prefill:")
            )
            if keys:
                try:
                    await exact_release(keys)
                except Exception as exc:
                    errors.append(f"prefill session release failed: {exc}")
        release = getattr(self.worker, "release_bank_adapter_sessions", None)
        if callable(release):
            try:
                await release()
            except Exception as exc:
                errors.append(f"bank adapter release failed: {exc}")
        task = getattr(self.worker, "_worker_task", None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._bank_protections.clear()
        self._protection_index.clear()
        if errors:
            raise RuntimeError("; ".join(errors))

    async def _release_prefill_sessions_for_items(self, items: Sequence[dict[str, Any]]) -> None:
        release = getattr(self.worker, "release_exact_sessions", None)
        if not callable(release):
            return
        keys = tuple((_prefill_client_tag(item), str(item["request_id"])) for item in items)
        if keys:
            await release(keys)

    def _manager(self):
        model = getattr(self.worker, "model", None)
        manager = getattr(model, "gpu_block_manager", None)
        if manager is None:
            raise RuntimeError("SwiftLLM model/block manager is not initialized")
        if not getattr(manager, "double_bank_enabled", False):
            raise RuntimeError("SwiftLLM double bank is not enabled")
        return manager

    def _descriptor(self, descriptor: Any) -> StarsDBankDescriptor:
        return StarsDBankDescriptor(
            bank_id=int(descriptor.bank_id),
            bank_epoch=int(descriptor.epoch),
            total_blocks=int(descriptor.num_blocks),
            free_blocks=int(descriptor.remaining_blocks),
            total_rows=int(getattr(self.engine_config, "max_seqs_in_block_table", 0)),
            free_rows=len(self._available_rows()),
            role=str(getattr(descriptor.role, "value", descriptor.role)),
        )

    def _row_manager(self):
        manager = getattr(self.worker, "request_id_manager", None)
        if manager is None:
            raise RuntimeError("SwiftLLM request_id_manager is not initialized")
        return manager

    def _available_rows(self) -> list[int]:
        manager = getattr(self.worker, "request_id_manager", None)
        if manager is None:
            return []
        return list(getattr(manager, "available_ids", ()))

    def _free_row(self, row: int) -> None:
        manager = self._row_manager()
        if int(row) not in manager.available_ids:
            manager.free_id(int(row))

    def _require_bank_unprotected(self, bank_id: int, bank_epoch: int) -> None:
        holders = self._bank_protections.get((int(bank_id), int(bank_epoch)))
        if holders:
            raise RuntimeError("bank epoch is protected by pending dirty export")

    def _return_hidden_for_output_spec(self, output_spec: Any, *, phase: str) -> bool:
        if getattr(output_spec, "need_logits", False):
            raise RuntimeError("SwiftLLM logits export is unsupported")
        hidden_mode = str(getattr(output_spec, "hidden_mode", "none"))
        if hidden_mode == "none":
            if getattr(output_spec, "need_accepted_hidden", False):
                raise RuntimeError("none hidden_mode cannot request accepted hidden")
            return False
        if phase == "verify" and not getattr(output_spec, "need_accepted_hidden", False):
            return False
        configured = self.worker._target_hidden_layer_ids()  # pylint: disable=protected-access
        requested = tuple(int(layer) for layer in getattr(output_spec, "hidden_layers", ()) or ())
        if hidden_mode == "selected_layers":
            if configured is None or tuple(configured) != requested:
                raise RuntimeError("selected hidden layers must match SwiftLLM target layer config")
            return True
        if hidden_mode == "last":
            if configured is not None:
                raise RuntimeError("last hidden mode requires SwiftLLM target layer config to be empty")
            return True
        raise RuntimeError("unsupported hidden_mode")

    def _replace_batch_key(self, old_key: tuple[str, int, int, str], new_key: tuple[str, int, int, str]) -> None:
        updates = {}
        for batch_id, record in self._reserve_batches.items():
            if old_key not in record.keys:
                updates[batch_id] = record
                continue
            if new_key in record.keys and new_key != old_key:
                raise RuntimeError("resident reuse destination key already exists in batch")
            updates[batch_id] = dataclasses.replace(
                record,
                keys=tuple(new_key if item == old_key else item for item in record.keys),
            )
        self._reserve_batches = updates

    def _preflight_reuse_resident(self, manager: Any, active: Any, items: tuple[dict[str, Any], ...]):
        if not items:
            raise RuntimeError("resident reuse batch requires items")
        seen_old: set[tuple[str, int, int, str]] = set()
        seen_new: set[tuple[str, int, int, str]] = set()
        records = []
        for item in items:
            request_id = str(item["request_id"])
            request_epoch = int(item["request_epoch"])
            old_round = int(item["previous_round_id"])
            new_round = int(item["round_id"])
            if new_round != old_round + 1:
                raise RuntimeError("resident reuse requires next consecutive round")
            reservation_id = str(item["reservation_id"])
            old_key = (request_id, request_epoch, old_round, reservation_id)
            new_key = (request_id, request_epoch, new_round, reservation_id)
            if old_key in seen_old or new_key in seen_new:
                raise RuntimeError("duplicate resident reuse key")
            seen_old.add(old_key)
            seen_new.add(new_key)
            reservation = self._resource_rows.get(old_key)
            replay = False
            if reservation is None:
                reservation = self._resource_rows.get(new_key)
                replay = True
            elif new_key in self._resource_rows and new_key != old_key:
                raise RuntimeError("resident reuse destination already exists")
            if reservation is None:
                raise RuntimeError("resident reuse source reservation is missing")
            expected_round = new_round if replay else old_round
            expected = (
                request_id,
                request_epoch,
                expected_round,
                reservation_id,
                int(item["bank_id"]),
                int(item["bank_epoch"]),
                int(item["row"]),
                int(item["start_block"]),
                int(item["capacity_blocks"]),
            )
            actual = (
                reservation.request_id,
                reservation.request_epoch,
                reservation.round_id,
                reservation.reservation_id,
                reservation.bank_id,
                reservation.bank_epoch,
                reservation.row,
                reservation.start_block,
                reservation.capacity_blocks,
            )
            if actual != expected:
                raise RuntimeError("resident reuse reservation fence mismatch")
            if int(active.bank_id) != reservation.bank_id or int(active.epoch) != reservation.bank_epoch:
                raise RuntimeError("resident reuse requires the source bank to be active")
            location = active.request_ranges.get(int(reservation.row))
            if location is None:
                raise RuntimeError("resident reuse source row is not live")
            self._validate_item_location(
                {
                    "bank_id": reservation.bank_id,
                    "bank_epoch": reservation.bank_epoch,
                    "row": reservation.row,
                    "start_block": reservation.start_block,
                    "capacity_blocks": reservation.capacity_blocks,
                },
                reservation,
                location,
            )
            valid_blocks = int(item["valid_blocks"])
            authoritative_valid_blocks = int(manager.num_seq_allocated_blocks[int(reservation.row)].item())
            if valid_blocks <= 0 or valid_blocks > reservation.capacity_blocks:
                raise RuntimeError("resident reuse valid_blocks exceeds reservation capacity")
            if valid_blocks != authoritative_valid_blocks:
                raise RuntimeError("resident reuse valid_blocks fence mismatch")
            if int(location.kv_version) != int(item["kv_version"]):
                raise RuntimeError("resident reuse kv_version fence mismatch")
            session_key = (reservation.reservation_id, reservation.request_id)
            session = getattr(self.worker, "sessions", {}).get(session_key)
            if session is None or not getattr(session, "bank_adapter", False):
                raise RuntimeError("resident reuse requires exact live bank-adapter session")
            if int(session.request.request_id) != reservation.row:
                raise RuntimeError("resident reuse session row fence mismatch")
            batch_id = _batch_id_from_reservation_id(reservation.reservation_id)
            record = self._reserve_batches.get(batch_id)
            if record is None or record.state != "ACTIVATED" or (old_key not in record.keys and new_key not in record.keys):
                raise RuntimeError("resident reuse requires an activated batch record")
            records.append((item, None if replay else old_key, new_key, reservation, location))
        replay_flags = tuple(old_key is None for _item, old_key, _new_key, _reservation, _location in records)
        if any(replay_flags) and not all(replay_flags):
            raise RuntimeError("resident reuse mixed replay/new batch is not supported")
        return tuple(records)

    def _remove_from_batches(self, key: tuple[str, int, int, str]) -> None:
        self._reserve_batches = {
            batch_id: dataclasses.replace(record, keys=tuple(item for item in record.keys if item != key))
            for batch_id, record in self._reserve_batches.items()
            if any(item != key for item in record.keys)
        }

    def _reservation_for_item(self, item: dict[str, Any]) -> StarsDReservation:
        key = (str(item["request_id"]), int(item["request_epoch"]), int(item["round_id"]), str(item["reservation_id"]))
        reservation = self._resource_rows.get(key)
        if reservation is None:
            raise RuntimeError("unknown StarSD reservation")
        return reservation

    def _h2d_handles_for_items(self, items: tuple[dict[str, Any], ...]) -> tuple[StarsDCopyHandle, ...]:
        handles: list[StarsDCopyHandle] = []
        for item in items:
            reservation = self._reservation_for_item(item)
            batch_id = _batch_id_from_reservation_id(reservation.reservation_id)
            handles.append(
                StarsDCopyHandle(
                    operation_id=str(item.get("operation_id") or item["plan_id"]),
                    direction=StarsDCopyDirection.H2D,
                    target_id=str(item["target_id"]),
                    process_generation=int(item["target_process_generation"]),
                    request_id=str(item["request_id"]),
                    request_epoch=int(item["request_epoch"]),
                    round_id=int(item["round_id"]),
                    batch_id=batch_id,
                    bank_id=int(item["bank_id"]),
                    bank_epoch=int(item["bank_epoch"]),
                    host_kv_version=int(item["expected_host_version"]),
                )
            )
        return tuple(handles)

    def _prepared_items_from_h2d_items(self, items: tuple[dict[str, Any], ...]) -> tuple[StarsDPreparedBank, ...]:
        prepared: list[StarsDPreparedBank] = []
        for item in items:
            copied_blocks = int(item["copy_block_count"])
            prepared.append(
                StarsDPreparedBank(
                    plan_id=str(item["plan_id"]),
                    request_id=str(item["request_id"]),
                    request_epoch=int(item["request_epoch"]),
                    round_id=int(item["round_id"]),
                    bank_id=int(item["bank_id"]),
                    bank_epoch=int(item["bank_epoch"]),
                    row=int(item["row"]),
                    start_block=int(item["start_block"]),
                    capacity_blocks=int(item["capacity_blocks"]),
                    copied_blocks=copied_blocks,
                    kv_version=int(item["expected_host_version"]),
                )
            )
        return tuple(prepared)

    def _prepared_items_from_h2d_handles(self, handles: tuple[StarsDCopyHandle, ...]) -> tuple[StarsDPreparedBank, ...]:
        manager = self._manager()
        prepared: list[StarsDPreparedBank] = []
        for handle in handles:
            reservation = None
            for candidate in self._resource_rows.values():
                if (
                    candidate.request_id == handle.request_id
                    and candidate.request_epoch == handle.request_epoch
                    and candidate.round_id == handle.round_id
                    and candidate.bank_id == handle.bank_id
                    and candidate.bank_epoch == handle.bank_epoch
                ):
                    reservation = candidate
                    break
            if reservation is None:
                raise RuntimeError("H2D activation requires exact live reservation")
            descriptor = manager.get_bank_descriptor(handle.bank_id)
            if int(descriptor.epoch) != handle.bank_epoch:
                raise RuntimeError("H2D activation bank epoch is stale")
            location = descriptor.request_ranges.get(int(reservation.row))
            if location is None:
                raise RuntimeError("H2D activation requires exact live bank range")
            copied_blocks = int(manager.num_seq_allocated_blocks[int(reservation.row)].item())
            prepared.append(
                StarsDPreparedBank(
                    plan_id=handle.operation_id,
                    request_id=handle.request_id,
                    request_epoch=handle.request_epoch,
                    round_id=handle.round_id,
                    bank_id=handle.bank_id,
                    bank_epoch=handle.bank_epoch,
                    row=reservation.row,
                    start_block=reservation.start_block,
                    capacity_blocks=reservation.capacity_blocks,
                    copied_blocks=copied_blocks,
                    kv_version=int(location.kv_version),
                )
            )
        return tuple(prepared)

    def _d2h_handles_for_items(self, items: tuple[dict[str, Any], ...]) -> tuple[StarsDCopyHandle, ...]:
        return tuple(
            StarsDCopyHandle(
                operation_id=str(item.get("operation_id") or item["plan_id"]),
                direction=StarsDCopyDirection.D2H,
                target_id=str(item["target_id"]),
                process_generation=int(item["target_process_generation"]),
                request_id=str(item["request_id"]),
                request_epoch=int(item["request_epoch"]),
                round_id=int(item["round_id"]),
                batch_id=str(item.get("batch_id") or item["plan_id"]),
                bank_id=int(item["bank_id"]),
                bank_epoch=int(item["bank_epoch"]),
                host_kv_version=int(item.get("expected_host_version", item["kv_version"])),
            )
            for item in items
        )

    @staticmethod
    def _dirty_result_from_item(item: dict[str, Any]) -> StarsDDirtyWriteback:
        return StarsDDirtyWriteback(
            plan_id=str(item["plan_id"]),
            request_id=str(item["request_id"]),
            copied_blocks=int(item["dirty_block_count"]),
            logical_kv_len_after=int(item["post_crop_logical_kv_len"]),
        )

    def _preflight_prepare_bank_from_host(self, manager: Any, model: Any, items: tuple[dict[str, Any], ...], k_views: tuple[memoryview, ...], v_views: tuple[memoryview, ...]):
        seen_plan: set[str] = set()
        seen_request: set[tuple[str, int, int]] = set()
        seen_row: set[tuple[int, int, int]] = set()
        seen_range: set[tuple[int, int, int, int]] = set()
        records = []
        for item, k_view, v_view in zip(items, k_views, v_views):
            plan_id = str(item["plan_id"])
            if plan_id in seen_plan:
                raise RuntimeError("duplicate H2D plan_id")
            seen_plan.add(plan_id)
            request_key = (str(item["request_id"]), int(item["request_epoch"]), int(item["round_id"]))
            if request_key in seen_request:
                raise RuntimeError("duplicate H2D request")
            seen_request.add(request_key)
            reservation = self._reservation_for_item(item)
            batch_id = _batch_id_from_reservation_id(reservation.reservation_id)
            record = self._reserve_batches.get(batch_id)
            if record is None or record.state != "RESERVED":
                raise RuntimeError("prepare requires a live RESERVED batch")
            descriptor = manager.get_bank_descriptor(int(item["bank_id"]))
            if int(descriptor.epoch) != int(item["bank_epoch"]):
                raise RuntimeError("prepare bank epoch is stale")
            location = descriptor.request_ranges.get(int(item["row"]))
            if location is None:
                raise RuntimeError("prepare references unknown reserved bank row")
            self._validate_item_location(item, reservation, location)
            copied_blocks = int(item["copy_block_count"])
            if copied_blocks <= 0 or copied_blocks > int(reservation.capacity_blocks):
                raise RuntimeError("copy_block_count exceeds reservation capacity")
            row_key = (int(item["bank_id"]), int(item["bank_epoch"]), int(item["row"]))
            range_key = (int(item["bank_id"]), int(item["bank_epoch"]), int(item["start_block"]), int(item["capacity_blocks"]))
            if row_key in seen_row or range_key in seen_range:
                raise RuntimeError("duplicate H2D row/range")
            seen_row.add(row_key)
            seen_range.add(range_key)
            expected_bytes = self._plane_bytes(model.k_cache, location, 0, copied_blocks)
            if len(k_view) != expected_bytes or len(v_view) != expected_bytes:
                raise RuntimeError("H2D K/V view byte length mismatch")
            if model.k_cache.is_cuda:
                if int(item.get("k_nbytes", -1)) != expected_bytes or int(item.get("v_nbytes", -1)) != expected_bytes:
                    raise RuntimeError("H2D registered K/V byte size mismatch")
                if int(item.get("k_address", 0) or 0) <= 0 or int(item.get("v_address", 0) or 0) <= 0:
                    raise RuntimeError("H2D CUDA path requires registered HostKV addresses")
            records.append((item, k_view, v_view, reservation, location, batch_id, copied_blocks))
        batch_ids = {record[5] for record in records}
        if len(batch_ids) != 1:
            raise RuntimeError("prepare batch spans multiple reserve batch ids")
        old_active = manager.get_bank_descriptor(manager.active_bank_id)
        prepared_bank = manager.get_bank_descriptor(int(records[0][0]["bank_id"]))
        self._require_bank_unprotected(int(old_active.bank_id), int(old_active.epoch))
        self._require_bank_unprotected(int(prepared_bank.bank_id), int(prepared_bank.epoch))
        return tuple(records)

    def _preflight_export_dirty_to_host(self, manager: Any, model: Any, items: tuple[dict[str, Any], ...], k_views: tuple[memoryview, ...], v_views: tuple[memoryview, ...]):
        seen_plan: set[str] = set()
        seen_request: set[tuple[str, int, int]] = set()
        seen_range: set[tuple[int, int, int, int, int]] = set()
        records = []
        for item, k_view, v_view in zip(items, k_views, v_views):
            plan_id = str(item["plan_id"])
            if plan_id in seen_plan:
                raise RuntimeError("duplicate D2H plan_id")
            seen_plan.add(plan_id)
            request_key = (str(item["request_id"]), int(item["request_epoch"]), int(item["round_id"]))
            if request_key in seen_request:
                raise RuntimeError("duplicate D2H request")
            seen_request.add(request_key)
            descriptor = manager.get_bank_descriptor(int(item["bank_id"]))
            if int(descriptor.epoch) != int(item["bank_epoch"]):
                raise RuntimeError("dirty export bank epoch is stale")
            location = descriptor.request_ranges.get(int(item["row"]))
            if location is None:
                raise RuntimeError("dirty export references unknown bank row")
            self._validate_item_location(item, None, location, allow_cropped=True)
            dirty_begin = int(item["dirty_begin_block"])
            dirty_count = int(item["dirty_block_count"])
            if dirty_count <= 0 or dirty_begin < 0 or dirty_begin + dirty_count > int(location.num_blocks):
                raise RuntimeError("dirty export range exceeds bank location")
            range_key = (int(item["bank_id"]), int(item["bank_epoch"]), int(item["row"]), dirty_begin, dirty_count)
            if range_key in seen_range:
                raise RuntimeError("duplicate D2H bank range")
            seen_range.add(range_key)
            expected_bytes = self._plane_bytes(model.k_cache, location, dirty_begin, dirty_count)
            if len(k_view) != expected_bytes or len(v_view) != expected_bytes:
                raise RuntimeError("D2H K/V view byte length mismatch")
            records.append((item, k_view, v_view, location, dirty_begin, dirty_count))
        return tuple(records)

    @staticmethod
    def _preflight_prefill_to_host(model: Any, items: tuple[dict[str, Any], ...], k_views: tuple[memoryview, ...], v_views: tuple[memoryview, ...]) -> None:
        seen_plan: set[str] = set()
        seen_request: set[tuple[str, int, int]] = set()
        block_bytes = int(model.k_cache[0:1].numel() * model.k_cache.element_size())
        for item, k_view, v_view in zip(items, k_views, v_views):
            plan_id = str(item["plan_id"])
            if plan_id in seen_plan:
                raise RuntimeError("duplicate prefill plan_id")
            seen_plan.add(plan_id)
            request_key = (str(item["request_id"]), int(item["request_epoch"]), int(item["round_id"]))
            if request_key in seen_request:
                raise RuntimeError("duplicate prefill request")
            seen_request.add(request_key)
            committed = int(item["expected_committed_blocks"])
            if committed <= 0 or int(item["expected_logical_kv_len"]) <= 0:
                raise RuntimeError("prefill expected committed/logical length must be positive")
            expected_bytes = committed * block_bytes
            if len(k_view) != expected_bytes or len(v_view) != expected_bytes:
                raise RuntimeError("prefill D2H K/V view byte length mismatch")

    @staticmethod
    def _validate_item_location(item: dict[str, Any], reservation: StarsDReservation | None, location: Any, *, allow_cropped: bool = False) -> None:
        expected = (
            int(item["bank_id"]),
            int(item["bank_epoch"]),
            int(item["row"]),
            int(item["start_block"]),
            int(item["capacity_blocks"]),
        )
        actual = (
            int(location.bank_id),
            int(location.bank_epoch),
            int(item["row"]),
            int(location.request_start_block),
            int(location.num_blocks),
        )
        if allow_cropped:
            same_base = actual[:4] == expected[:4]
            capacity_ok = int(actual[4]) <= int(expected[4])
            if not same_base or not capacity_ok:
                raise RuntimeError("bank location fence mismatch")
        elif actual != expected:
            raise RuntimeError("bank location fence mismatch")
        if reservation is not None and (
            int(reservation.bank_id),
            int(reservation.bank_epoch),
            int(reservation.row),
            int(reservation.start_block),
            int(reservation.capacity_blocks),
        ) != expected:
            raise RuntimeError("reservation fence mismatch")
        if "kv_version" in item and int(location.kv_version) != int(item["kv_version"]):
            raise RuntimeError("bank location kv_version fence mismatch")

    @staticmethod
    def _copy_host_to_bank(
        cache: torch.Tensor,
        host_view: memoryview,
        location: Any,
        block_count: int,
        *,
        host_address: int | None = None,
        expected_nbytes: int | None = None,
        profile: list[tuple[str, float]] | None = None,
        label: str = "h2d",
    ) -> None:
        marker = _profile_start()
        start = int(location.bank_base_block) + int(location.request_start_block)
        dst = cache[start : start + int(block_count)]
        nbytes = int(dst.numel() * dst.element_size())
        marker = _profile_mark(profile, f"{label}_gpu_slice_and_validate", marker)
        if len(host_view) != nbytes:
            raise RuntimeError("H2D host view byte length mismatch")
        if expected_nbytes is not None and int(expected_nbytes) != nbytes:
            raise RuntimeError("H2D registered byte size mismatch")
        if dst.is_cuda:
            if host_address is None or int(host_address) <= 0:
                raise RuntimeError("H2D CUDA copy requires a registered HostKV address")
            src = torch.frombuffer(host_view, dtype=dst.dtype, count=dst.numel()).reshape(dst.shape).pin_memory()
            marker = _profile_mark(profile, f"{label}_frombuffer_pin_memory", marker)
            dst.copy_(src.to(device=dst.device, non_blocking=True), non_blocking=True)
            _profile_mark(profile, f"{label}_copy_call", marker)
            return
        src = torch.frombuffer(host_view, dtype=dst.dtype, count=dst.numel()).reshape(dst.shape)
        marker = _profile_mark(profile, f"{label}_frombuffer", marker)
        dst.copy_(src, non_blocking=False)
        _profile_mark(profile, f"{label}_copy_call", marker)

    @staticmethod
    def _copy_bank_to_host(
        cache: torch.Tensor,
        host_view: memoryview,
        location: Any,
        dirty_begin: int,
        dirty_count: int,
        *,
        profile: list[tuple[str, float]] | None = None,
        label: str = "d2h",
    ) -> None:
        marker = _profile_start()
        start = int(location.bank_base_block) + int(location.request_start_block) + int(dirty_begin)
        src = cache[start : start + int(dirty_count)]
        marker = _profile_mark(profile, f"{label}_gpu_slice", marker)
        dst = torch.frombuffer(host_view, dtype=src.dtype, count=src.numel()).reshape(src.shape)
        marker = _profile_mark(profile, f"{label}_frombuffer", marker)
        dst.copy_(src.detach(), non_blocking=True)
        _profile_mark(profile, f"{label}_copy_call", marker)

    @staticmethod
    def _plane_bytes(cache: torch.Tensor, location: Any, begin_block: int, block_count: int) -> int:
        start = int(location.bank_base_block) + int(location.request_start_block) + int(begin_block)
        tensor = cache[start : start + int(block_count)]
        return int(tensor.numel() * tensor.element_size())

    def _remember_release(self, key: tuple[str, int, int, str | None], result: dict[str, Any]) -> None:
        self._released[key] = result
        self._released_order.append(key)
        while len(self._released_order) > self._release_replay_window:
            old = self._released_order.popleft()
            self._released.pop(old, None)


def _coerce_reserve_request(item: StarsDReserveRequest | dict[str, Any]) -> StarsDReserveRequest:
    if isinstance(item, StarsDReserveRequest):
        return item
    return StarsDReserveRequest(
        request_id=str(item["request_id"]),
        request_epoch=int(item["request_epoch"]),
        round_id=int(item["round_id"]),
        required_blocks=int(item["required_blocks"]),
        batch_id=str(item["batch_id"]),
    )


def _coerce_release_request(item: StarsDReleaseRequest | dict[str, Any]) -> StarsDReleaseRequest:
    if isinstance(item, StarsDReleaseRequest):
        return item
    return StarsDReleaseRequest(
        request_id=str(item["request_id"]),
        request_epoch=int(item["request_epoch"]),
        round_id=int(item["round_id"]),
        reservation_id=None if item.get("reservation_id") is None else str(item["reservation_id"]),
    )


def _prefill_client_tag(item: dict[str, Any]) -> str:
    return f"starsd-prefill:{item['request_id']}:epoch{int(item['request_epoch'])}:round{int(item['round_id'])}"


def _prefill_client_tag_for_release(item: StarsDReleaseRequest) -> str:
    return f"starsd-prefill:{item.request_id}:epoch{int(item.request_epoch)}:round{int(item.round_id)}"


def _reservation_id(item: StarsDReserveRequest, bank_id: int, bank_epoch: int, row: int) -> str:
    return f"{item.batch_id}:bank{int(bank_id)}:epoch{int(bank_epoch)}:row{int(row)}"


def _batch_id_from_reservation_id(reservation_id: str) -> str:
    marker = ":bank"
    if marker not in str(reservation_id):
        raise RuntimeError("reservation_id is missing batch fence")
    return str(reservation_id).rsplit(marker, 1)[0]


def _release_key_for_reservation(reservation: StarsDReservation) -> tuple[str, int, int, str]:
    return (reservation.request_id, reservation.request_epoch, reservation.round_id, reservation.reservation_id)


def _ensure_mutable_bank_tensors(manager: Any) -> None:
    for name in ("block_table", "num_seq_allocated_blocks", "is_block_free"):
        tensor = getattr(manager, name, None)
        if tensor is not None and getattr(torch, "is_inference", lambda _: False)(tensor):
            setattr(manager, name, tensor.clone())


def _ensure_mutable_kv_cache(model: Any) -> None:
    for name in ("k_cache", "v_cache"):
        tensor = getattr(model, name, None)
        if tensor is not None and getattr(torch, "is_inference", lambda _: False)(tensor):
            setattr(model, name, tensor.clone())


def _maybe_cuda_stream(stream: torch.cuda.Stream | None):
    if stream is None:
        return contextlib.nullcontext()
    return torch.cuda.stream(stream)


__all__ = (
    "StarsDBankDescriptor",
    "StarsDDirtyWriteback",
    "StarsDPreparedBank",
    "StarsDPrefillWriteback",
    "StarsDReleaseRequest",
    "StarsDReservation",
    "StarsDResidentRange",
    "StarsDReserveRequest",
    "StarsDResourceStats",
    "SwiftLLMStarsDTargetFacade",
)
