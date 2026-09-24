import asyncio
import dataclasses
import functools
import json
import math
import os
import time
from typing import Any, Sequence

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.speculative import DraftProposal, compute_acceptance_for_plan
from swiftllm.utils import GB
from swiftllm.worker.model import LlamaModel, ModelForwardOutput

from .backend_local import RequestIdManager, build_batch_plan
from .structs import RawRequest, Request

_DIVERGENCE_TRACE_DIR = os.environ.get("STARSD_TARGET_DIVERGENCE_TRACE_DIR")


@dataclasses.dataclass
class TargetTask:
    task_id: str
    client_tag: str
    request_id: str
    phase: str
    payload: dict[str, Any]
    future: asyncio.Future | None = None


@dataclasses.dataclass
class TargetResult:
    task_id: str
    client_tag: str
    request_id: str
    phase: str
    payload: dict[str, Any]
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class _SessionBankFence:
    bank_id: int
    bank_epoch: int
    row: int
    start_block: int
    capacity_blocks: int
    batch_id: str | None


@dataclasses.dataclass
class _TargetSession:
    request: Request
    client_tag: str
    request_id: str
    bank_adapter: bool = False
    bank_fence: _SessionBankFence | None = None


class SwiftLLMTargetWorker:
    """In-process SwiftLLM target/base worker for StarSD orchestration.

    StarSD remains responsible for draft generation. This worker receives
    prefill/decode/verify tasks, continuous-batches pending target rows, and
    returns target-side greedy verification results. A network server can wrap
    this class without adding a hard SwiftLLM dependency on StarSD.
    """

    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)
        self.model: LlamaModel | None = None
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.request_id_manager: RequestIdManager | None = None
        self.pending_target_tasks: asyncio.Queue[TargetTask] = asyncio.Queue()
        self.sessions: dict[tuple[str, str], _TargetSession] = {}
        self._worker_task: asyncio.Task | None = None
        self._batch_lock: asyncio.Lock | None = None
        self.initialized = False

    async def _run_on_model_async(self, func, *args, **kwargs):
        func_partial = functools.partial(func, *args, **kwargs)

        def run_with_model_device():
            device_index = self._model_cuda_device_index()
            if device_index is not None:
                torch.cuda.set_device(device_index)
            return func_partial()

        return await self.event_loop.run_in_executor(None, run_with_model_device)

    async def shutdown(self) -> None:
        """Release a quiescent direct worker after its compute/copy lanes join.

        The caller must first join executor-backed model work and external KV
        DMA. Cancelling a listener is not a substitute for that physical fence.
        """
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        self.sessions.clear()
        self.model = None
        self.request_id_manager = None
        self.initialized = False

    def _model_cuda_device_index(self) -> int | None:
        if self.model is None or self.model.k_cache is None:
            return None
        device = getattr(self.model.k_cache, "device", None)
        if getattr(device, "type", None) != "cuda":
            return None
        return device.index

    async def initialize(self, *, start_background_loop: bool = True):
        self.event_loop = asyncio.get_event_loop()
        self._batch_lock = asyncio.Lock()

        print("[SwiftLLMTargetWorker] Initializing model...")
        self.model = LlamaModel(self.engine_config)

        print("[SwiftLLMTargetWorker] Loading weights...")
        self.model.load_weights()

        print("[SwiftLLMTargetWorker] Profiling kv blocks...")
        num_gpu_blocks = self.model.profile_num_blocks()
        block_size_bytes = self.engine_config.block_size*self.model_config.get_kvslot_size()
        print(f"[SwiftLLMTargetWorker] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")

        print("[SwiftLLMTargetWorker] Allocating kv cache and swap...")
        self.model.init_kvcache_and_swap(num_gpu_blocks)
        self.request_id_manager = RequestIdManager(self.engine_config.max_seqs_in_block_table)
        self.initialized = True
        if start_background_loop:
            self._worker_task = asyncio.create_task(self._target_task_event_loop())

    async def submit_prefill(
        self,
        client_tag: str,
        request_id: str,
        input_ids: list[int],
        max_output_len: int,
        **kwargs,
    ) -> TargetResult:
        return await self._submit(TargetTask(
            task_id=kwargs.pop("task_id", f"{client_tag}:{request_id}:prefill"),
            client_tag=client_tag,
            request_id=request_id,
            phase="prefill",
            payload={"input_ids": input_ids, "max_output_len": max_output_len, **kwargs},
        ))

    async def submit_prefill_batch(self, items: Sequence[dict[str, Any]]) -> list[TargetResult]:
        if not self.initialized:
            raise RuntimeError("SwiftLLMTargetWorker is not initialized")
        tasks = self._prefill_tasks_from_items(items)
        for task in tasks:
            await self.pending_target_tasks.put(task)
        return [await task.future for task in tasks]

    async def run_prefill_batch_direct(self, items: Sequence[dict[str, Any]]) -> list[TargetResult]:
        """Run a process-local prefill batch without the internal task queue."""

        if not self.initialized:
            raise RuntimeError("SwiftLLMTargetWorker is not initialized")
        tasks = self._prefill_tasks_from_items(items)
        try:
            lock = self._batch_lock
            if lock is None:
                await self._run_task_batch(tasks)
            else:
                async with lock:
                    await self._run_task_batch(tasks)
        except Exception as exc:  # keep direct callers from hanging forever
            for task in tasks:
                if task.future is not None and not task.future.done():
                    task.future.set_result(self._make_result(task, {}, error=str(exc)))
        return [await task.future for task in tasks]

    def _prefill_tasks_from_items(self, items: Sequence[dict[str, Any]]) -> list[TargetTask]:
        loop = asyncio.get_running_loop()
        if self.event_loop is not None and self.event_loop is not loop:
            raise RuntimeError("SwiftLLMTargetWorker direct prefill called from a different event loop")
        if self.event_loop is None:
            self.event_loop = loop
        tasks: list[TargetTask] = []
        for raw in items:
            item = dict(raw)
            client_tag = str(item.pop("client_tag"))
            request_id = str(item.pop("request_id"))
            input_ids = [int(token) for token in list(item.pop("input_ids"))]
            max_output_len = int(item.pop("max_output_len"))
            task = TargetTask(
                task_id=str(item.pop("task_id", f"{client_tag}:{request_id}:prefill")),
                client_tag=client_tag,
                request_id=request_id,
                phase="prefill",
                payload={"input_ids": input_ids, "max_output_len": max_output_len, **item},
                future=loop.create_future(),
            )
            tasks.append(task)
        return tasks

    async def submit_verify(
        self,
        client_tag: str,
        request_id: str,
        draft_token_ids: list[int],
        **kwargs,
    ) -> TargetResult:
        return await self._submit(TargetTask(
            task_id=kwargs.pop("task_id", f"{client_tag}:{request_id}:verify"),
            client_tag=client_tag,
            request_id=request_id,
            phase="verify",
            payload={"draft_token_ids": draft_token_ids, **kwargs},
        ))

    async def submit_verify_bank_batch(self, run_plan: dict[str, Any]) -> list[TargetResult]:
        """Run verify for an already-prepared active GPU bank batch.

        This entrypoint uses worker-local SwiftLLM session metadata and active
        bank block-table rows. If a request has no local session, the run plan
        may provide the minimum compute metadata needed to create a bank adapter
        session: worker-local row, prompt length, output-token anchor, output
        limit, and draft tokens. It does not import HostKV, allocate via the
        legacy bitmap path, restore sessions from migration payloads, or copy
        block tables from another worker.
        """
        if not self.initialized:
            raise RuntimeError("SwiftLLMTargetWorker is not initialized")
        if self.model is None or self.model.gpu_block_manager is None:
            raise RuntimeError("SwiftLLM model/block manager is not initialized")
        manager = self.model.gpu_block_manager
        if not getattr(manager, "double_bank_enabled", False):
            raise RuntimeError("bank-aware verify forward requires double-bank block manager")
        plan = dict(run_plan or {})
        active_bank_id = int(plan.get("active_bank_id", manager.active_bank_id))
        if active_bank_id != int(manager.active_bank_id):
            raise RuntimeError(f"active bank mismatch: manager={manager.active_bank_id}, plan={active_bank_id}")
        descriptor = manager.get_bank_descriptor(active_bank_id)
        if plan.get("batch_id") is not None and descriptor.batch_id not in {None, str(plan.get("batch_id"))}:
            raise RuntimeError(f"active bank batch mismatch: {descriptor.batch_id} != {plan.get('batch_id')}")
        ready_event = getattr(descriptor, "ready_event", None)
        if ready_event is not None:
            await self._run_on_model_async(_wait_on_ready_event, ready_event)

        request_ids = [str(item) for item in list(plan.get("request_ids") or [])]
        rows = _bank_request_rows(plan, request_ids)
        payloads = _bank_verify_payloads(plan, request_ids)
        tasks: list[TargetTask] = []
        for request_id in request_ids:
            payload = dict(payloads.get(request_id) or {})
            client_tag = str(payload.get("client_tag") or payload.get("tag") or request_id)
            row = int(rows.get(request_id, payload.get("request_row", payload.get("row", -1))))
            session = self.sessions.get((client_tag, request_id))
            if session is not None and getattr(session, "bank_adapter", False) and row >= 0 and int(session.request.request_id) != row:
                self._release_bank_adapter_session(client_tag, request_id, session)
                session = None
            if session is None:
                session = self._create_bank_adapter_session(
                    client_tag=client_tag,
                    request_id=request_id,
                    row=row,
                    payload=payload,
                )
            if row < 0:
                row = int(session.request.request_id)
            if int(session.request.request_id) != row:
                raise RuntimeError(
                    f"request row mismatch for {request_id}: session={session.request.request_id}, plan={row}"
                )
            location = descriptor.request_ranges.get(row)
            if location is None:
                raise RuntimeError(f"request row {row} is not present in active bank {active_bank_id}")
            manager.validate_bank_location(location)
            draft_token_ids = payload.get("draft_token_ids", payload.get("draft_tokens"))
            if draft_token_ids is None:
                draft_token_ids = plan.get("draft_token_ids", [])
            session_output_ids = list(getattr(session.request, "output_token_ids", []))
            payload_output_ids = payload.get("output_token_ids")
            if payload_output_ids is not None:
                expected_output_ids = [int(token) for token in session_output_ids]
                actual_output_ids = [int(token) for token in list(payload_output_ids)]
            else:
                expected_output_ids = []
                actual_output_ids = []
            if payload_output_ids is not None and actual_output_ids != expected_output_ids:
                raise RuntimeError(f"bank adapter session output_token_ids mismatch for {request_id}")
            explicit_output_len = payload.get("max_output_len", payload.get("output_len"))
            if explicit_output_len is None:
                required_output_len = len(session_output_ids) + len(list(draft_token_ids or [])) + 1
            else:
                required_output_len = int(explicit_output_len)
            if int(session.request.output_len) < required_output_len:
                session.request.output_len = required_output_len
            stop_token_ids = frozenset(
                int(token) for token in payload.get("stop_token_ids", ())
            )
            if session.request.stop_token_ids != stop_token_ids:
                raise RuntimeError(
                    f"bank adapter session stop_token_ids mismatch for {request_id}"
                )
            task = TargetTask(
                task_id=str(payload.get("task_id") or f"{client_tag}:{request_id}:bank_verify"),
                client_tag=client_tag,
                request_id=request_id,
                phase="verify",
                payload={
                    "draft_token_ids": [int(token) for token in list(draft_token_ids or [])],
                    "proposal_kind": payload.get("proposal_kind") or plan.get("proposal_kind") or "dflash_block",
                    "return_hidden": bool(payload.get("return_hidden", plan.get("return_hidden", True))),
                    "keep_bank_range_for_export": bool(payload.get("keep_bank_range_for_export", plan.get("keep_bank_range_for_export", True))),
                },
                future=self.event_loop.create_future(),
            )
            tasks.append(task)
        if not tasks:
            return []
        lock = self._batch_lock
        if lock is None:
            await self._run_task_batch(tasks)
        else:
            async with lock:
                await self._run_task_batch(tasks)
        results = [task.future.result() for task in tasks]
        self._attach_bank_verify_metadata(
            tasks=tasks,
            results=results,
            bank_id=active_bank_id,
            bank_epoch=int(getattr(descriptor, "epoch", 0)),
        )
        return results

    def _create_bank_adapter_session(
        self,
        *,
        client_tag: str,
        request_id: str,
        row: int,
        payload: dict[str, Any],
    ) -> _TargetSession:
        if row < 0:
            raise RuntimeError(f"bank adapter session for {request_id} requires request_row/request_rows")
        if "prompt_len" not in payload or "output_token_ids" not in payload:
            raise RuntimeError(
                "bank adapter session requires prompt_len and output_token_ids; "
                "KV payload must already be prepared in the active bank"
            )
        output_token_ids = [int(token) for token in list(payload.get("output_token_ids") or [])]
        if not output_token_ids:
            raise RuntimeError("bank adapter session requires an unstored anchor in output_token_ids")
        draft_token_ids = list(payload.get("draft_token_ids", payload.get("draft_tokens")) or [])
        default_output_len = max(len(output_token_ids) + len(draft_token_ids) + 1, 1)
        explicit_output_len = payload.get("max_output_len", payload.get("output_len"))
        if explicit_output_len is None:
            max_output_len = default_output_len
        else:
            max_output_len = max(int(explicit_output_len), len(output_token_ids))
        req = Request(
            RawRequest(
                "",
                max_output_len,
                stop_token_ids=payload.get("stop_token_ids", ()),
            )
        )
        req.prompt_len = int(payload.get("prompt_len") or 0)
        req.prompt_token_ids = []
        req.output_token_ids = output_token_ids
        req.request_id = int(row)
        req.spec_enabled = True
        adapter_logical_kv_len = int(req.logical_kv_len_after_current_state())
        engine_config = getattr(self, "engine_config", None)
        block_size = getattr(engine_config, "block_size", None)
        if block_size is None:
            manager = getattr(getattr(self, "model", None), "gpu_block_manager", None)
            block_size = getattr(manager, "block_size", 1)
        block_size = max(int(block_size), 1)
        adapter_valid_blocks = int(math.ceil(adapter_logical_kv_len / block_size))
        imported_committed = int(
            payload.get(
                "imported_committed_blocks",
                payload.get("host_committed_blocks", payload.get("min_valid_blocks", 0)),
            )
            or 0
        )
        imported_logical = int(
            payload.get("imported_logical_kv_len", payload.get("host_logical_kv_len", payload.get("logical_kv_len", 0)))
            or 0
        )
        if imported_committed > 0 and adapter_valid_blocks < imported_committed:
            raise RuntimeError(
                "bank adapter session logical state is shorter than imported HostKV blocks: "
                f"request_id={request_id} row={row} prompt_len={req.prompt_len} "
                f"output_token_count={len(output_token_ids)} adapter_logical_kv_len={adapter_logical_kv_len} "
                f"imported_logical_kv_len={imported_logical} adapter_valid_blocks={adapter_valid_blocks} "
                f"imported_committed_blocks={imported_committed} output_token_ids={output_token_ids}"
            )
        if imported_logical > 0 and adapter_logical_kv_len < imported_logical:
            raise RuntimeError(
                "bank adapter session logical KV length is shorter than imported HostKV logical length: "
                f"request_id={request_id} row={row} prompt_len={req.prompt_len} "
                f"output_token_count={len(output_token_ids)} adapter_logical_kv_len={adapter_logical_kv_len} "
                f"imported_logical_kv_len={imported_logical} adapter_valid_blocks={adapter_valid_blocks} "
                f"imported_committed_blocks={imported_committed} output_token_ids={output_token_ids}"
            )
        if self.request_id_manager is not None and row in self.request_id_manager.available_ids:
            self.request_id_manager.available_ids.remove(row)
        session = _TargetSession(req, client_tag, request_id, bank_adapter=True)
        self.sessions[(client_tag, request_id)] = session
        return session

    def _release_bank_adapter_session(self, client_tag: str, request_id: str, session: _TargetSession) -> None:
        if not getattr(session, "bank_adapter", False):
            return
        self.sessions.pop((client_tag, request_id), None)
        row = int(session.request.request_id)
        available_ids = getattr(self.request_id_manager, "available_ids", None) if self.request_id_manager is not None else None
        already_available = available_ids is not None and row in available_ids
        if self.request_id_manager is not None and not already_available:
            self.request_id_manager.free_id(row)

    async def release_bank_adapter_sessions(
        self,
        request_ids: list[str] | None = None,
        client_tags: dict[str, str] | list[str] | None = None,
        exact_session_keys: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Release only temporary bank-adapter sessions and their worker-local rows."""

        request_filter = None if request_ids is None else {str(request_id) for request_id in request_ids}
        exact_filter = None if exact_session_keys is None else {(str(client_tag), str(request_id)) for client_tag, request_id in exact_session_keys}
        tag_filter: dict[str, str] = {}
        if isinstance(client_tags, dict):
            tag_filter = {str(request_id): str(client_tag) for request_id, client_tag in client_tags.items()}
        elif client_tags is not None and request_ids is not None:
            tag_filter = {str(request_id): str(tag) for request_id, tag in zip(request_ids, client_tags)}

        released: list[dict[str, Any]] = []
        for (client_tag, request_id), session in list(self.sessions.items()):
            if not getattr(session, "bank_adapter", False):
                continue
            exact_match = exact_filter is not None and (str(client_tag), str(request_id)) in exact_filter
            legacy_match = False
            if exact_filter is None and request_filter is None:
                legacy_match = True
            elif request_filter is not None:
                legacy_match = str(request_id) in request_filter
                expected_tag = tag_filter.get(str(request_id))
                if expected_tag is not None and str(client_tag) != expected_tag:
                    legacy_match = False
            if not exact_match and not legacy_match:
                continue
            row = int(session.request.request_id)
            self._release_bank_adapter_session(client_tag, request_id, session)
            released.append({"client_tag": str(client_tag), "request_id": str(request_id), "row": row})
        return {"status": "ok", "released_count": len(released), "released": released}

    async def release_exact_sessions(self, exact_session_keys: Sequence[tuple[str, str]]) -> dict[str, Any]:
        """Release exact non-bank prefill sessions without request-id wildcards."""

        keys = tuple((str(client_tag), str(request_id)) for client_tag, request_id in exact_session_keys)
        records: list[tuple[str, str, _TargetSession, int]] = []
        exact_ranges: list[tuple[int, int, int, int, int, str | None]] = []
        seen_keys: set[tuple[str, str]] = set()
        for client_tag, request_id in keys:
            key = (client_tag, request_id)
            if key in seen_keys:
                raise RuntimeError("duplicate exact prefill session release")
            seen_keys.add(key)
            session = self.sessions.get(key)
            if session is None:
                raise RuntimeError(f"unknown exact prefill session: {key}")
            if getattr(session, "bank_adapter", False):
                raise RuntimeError("exact prefill release does not accept bank-adapter sessions")
            row = int(session.request.request_id)
            fence = getattr(session, "bank_fence", None)
            if fence is None:
                raise RuntimeError("exact prefill session is missing bank fence")
            if int(fence.row) != row:
                raise RuntimeError("exact prefill session row/fence mismatch")
            if int(fence.capacity_blocks) <= 0:
                raise RuntimeError("exact prefill session capacity must be positive")
            records.append((client_tag, request_id, session, row))
            exact_ranges.append(
                (
                    int(fence.bank_id),
                    int(fence.bank_epoch),
                    int(fence.row),
                    int(fence.start_block),
                    int(fence.capacity_blocks),
                    None if fence.batch_id is None else str(fence.batch_id),
                )
            )

        released: list[dict[str, Any]] = []
        if exact_ranges:
            await self._run_on_model_async(self._release_exact_prefill_ranges, tuple(exact_ranges))
        for client_tag, request_id, _session, row in records:
            self.sessions.pop((client_tag, request_id), None)
            available_ids = getattr(self.request_id_manager, "available_ids", None) if self.request_id_manager is not None else None
            if self.request_id_manager is not None and (available_ids is None or row not in available_ids):
                self.request_id_manager.free_id(row)
            released.append({"client_tag": client_tag, "request_id": request_id, "row": row})
        return {"status": "ok", "released_count": len(released), "released": released}

    def _release_exact_prefill_ranges(self, ranges: tuple[tuple[int, int, int, int, int, str | None], ...]) -> None:
        manager = getattr(self.model, "gpu_block_manager", None) if self.model is not None else None
        if manager is None:
            raise RuntimeError("exact prefill release requires GPU block manager")
        with torch.inference_mode():
            manager.release_bank_ranges_exact_batch(list(ranges), apply=False)
            cpu_manager = getattr(self.model, "cpu_block_manager", None) if self.model is not None else None
            if cpu_manager is not None:
                rows = [int(item[2]) for item in ranges]
                device = getattr(manager.block_table, "device", torch.device("cuda"))
                seq_ids = torch.tensor(rows, dtype=torch.int32, device=device)
                cpu_manager.free_blocks_for_seqs(seq_ids)
            manager.release_bank_ranges_exact_batch(list(ranges))
            residual = []
            for bank_id, bank_epoch, row, start_block, capacity_blocks, batch_id in ranges:
                bank = manager.get_bank_descriptor(int(bank_id))
                if int(row) in bank.request_ranges:
                    residual.append((int(bank_id), int(bank_epoch), int(row), int(start_block), int(capacity_blocks), batch_id))
            if residual:
                raise RuntimeError(f"exact prefill release did not clear bank ranges: {residual}")

    def _attach_bank_verify_metadata(
        self,
        *,
        tasks: list[TargetTask],
        results: list[TargetResult],
        bank_id: int,
        bank_epoch: int,
    ) -> None:
        if self.model is None or self.model.gpu_block_manager is None:
            return
        manager = self.model.gpu_block_manager
        try:
            descriptor = manager.get_bank_descriptor(int(bank_id))
        except Exception:
            descriptor = None
        for task, result in zip(tasks, results):
            if getattr(result, "error", None):
                continue
            session = self.sessions.get((task.client_tag, task.request_id))
            if session is None:
                continue
            req = session.request
            row = int(req.request_id)
            location = None
            if descriptor is not None:
                location = getattr(descriptor, "request_ranges", {}).get(row)
            logical_kv_len = int(req.logical_kv_len_after_current_state())
            num_blocks = self._allocated_block_count_for_row(row, fallback=getattr(location, "num_blocks", 0))
            payload = result.payload
            payload.setdefault("client_tag", str(task.client_tag))
            payload.setdefault("request_id", str(task.request_id))
            payload["logical_kv_len"] = logical_kv_len
            payload["gpu_valid_blocks_after_crop"] = int(num_blocks)
            payload["num_blocks"] = int(num_blocks)
            if location is not None:
                next_kv_version = (int(getattr(location, "kv_version", 0)) + 1) % (1 << 64)
                location = manager.set_bank_location_kv_version(
                    row,
                    bank_id=int(getattr(location, "bank_id", bank_id)),
                    bank_epoch=int(getattr(location, "bank_epoch", bank_epoch)),
                    kv_version=next_kv_version,
                )
                payload["bank_offset_blocks"] = int(getattr(location, "request_start_block", 0))
                payload["bank_location"] = {
                    "worker_id": getattr(location, "worker_id", ""),
                    "device_id": getattr(location, "device_id", ""),
                    "model_kind": getattr(location, "model_kind", "target"),
                    "request_row": row,
                    "bank_id": int(getattr(location, "bank_id", bank_id)),
                    "bank_epoch": int(getattr(location, "bank_epoch", bank_epoch)),
                    "bank_base_block": int(getattr(location, "bank_base_block", 0)),
                    "request_start_block": int(getattr(location, "request_start_block", 0)),
                    "num_blocks": int(num_blocks),
                    "logical_kv_len": logical_kv_len,
                    "kv_version": int(getattr(location, "kv_version", 0)),
                }

    def _allocated_block_count_for_row(self, row: int, *, fallback: int = 0) -> int:
        manager = self.model.gpu_block_manager if self.model is not None else None
        if manager is None:
            return int(fallback or 0)
        table = getattr(manager, "num_seq_allocated_blocks", None)
        if table is not None:
            try:
                return int(table[int(row)].item())
            except Exception:
                try:
                    return int(table[int(row)])
                except Exception:
                    pass
        getter = getattr(manager, "get_num_allocated_blocks", None)
        if callable(getter):
            try:
                values = getter(torch.tensor([int(row)], dtype=torch.int32, device=table.device if table is not None else None))
                return int(values[0].item())
            except Exception:
                pass
        return int(fallback or 0)

    async def submit_decode(self, client_tag: str, request_id: str, **kwargs) -> TargetResult:
        return await self._submit(TargetTask(
            task_id=kwargs.pop("task_id", f"{client_tag}:{request_id}:decode"),
            client_tag=client_tag,
            request_id=request_id,
            phase="decode",
            payload=kwargs,
        ))

    async def submit_end(self, client_tag: str, request_id: str) -> TargetResult:
        return await self._submit(TargetTask(
            task_id=f"{client_tag}:{request_id}:end",
            client_tag=client_tag,
            request_id=request_id,
            phase="end",
            payload={},
        ))

    async def _submit(self, task: TargetTask) -> TargetResult:
        if not self.initialized:
            raise RuntimeError("SwiftLLMTargetWorker is not initialized")
        task.future = self.event_loop.create_future()
        await self.pending_target_tasks.put(task)
        return await task.future

    async def _target_task_event_loop(self):
        while True:
            first_task = await self.pending_target_tasks.get()
            tasks = [first_task]
            while True:
                try:
                    tasks.append(self.pending_target_tasks.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                lock = self._batch_lock
                if lock is None:
                    await self._run_task_batch(tasks)
                else:
                    async with lock:
                        await self._run_task_batch(tasks)
            except Exception as exc:  # keep submitters from hanging forever
                for task in tasks:
                    if task.future is not None and not task.future.done():
                        task.future.set_result(self._make_result(task, {}, error=str(exc)))
            finally:
                for _ in tasks:
                    self.pending_target_tasks.task_done()

    def _session_key(self, task: TargetTask) -> tuple[str, str]:
        return (task.client_tag, task.request_id)

    def _make_result(self, task: TargetTask, payload: dict[str, Any], error: str | None = None) -> TargetResult:
        return TargetResult(
            task_id=task.task_id,
            client_tag=task.client_tag,
            request_id=task.request_id,
            phase=task.phase,
            payload=payload,
            error=error,
        )

    @torch.inference_mode()
    def export_session_for_migration(
        self,
        client_tag: str,
        request_id: str,
        free_after_export: bool = False,
    ) -> dict[str, Any]:
        """Export one target session's request state and GPU KV blocks.

        Experimental in-process migration API for validating whether a
        SwiftLLM target session can be moved to another SwiftLLMTargetWorker.
        The returned dict intentionally carries CPU tensors and is not a wire
        protocol.
        """
        self._ensure_migration_ready()
        key = (client_tag, request_id)
        session = self.sessions.get(key)
        if session is None:
            raise RuntimeError(f"Unknown target session: {key}")

        req = session.request
        logical_kv_len = int(req.logical_kv_len_after_current_state())
        block_manager = self.model.gpu_block_manager

        self._sync_cache_device()
        export_t0 = time.perf_counter()
        block_ids = block_manager.get_allocated_block_ids(req.request_id)
        block_ids_cpu = block_ids.detach().cpu()
        if int(block_ids_cpu.numel()) > 0 and self._block_ids_are_contiguous(block_ids_cpu):
            start_block_id = int(block_ids_cpu[0].item())
            end_block_id = start_block_id + int(block_ids_cpu.numel())
            k_cache_blocks = self.model.k_cache[start_block_id:end_block_id].detach().cpu()
            v_cache_blocks = self.model.v_cache[start_block_id:end_block_id].detach().cpu()
            export_method = "contiguous_slice_cpu"
        else:
            cache_block_ids = block_ids.to(device=self.model.k_cache.device)
            k_cache_blocks = self.model.k_cache[cache_block_ids].detach().cpu()
            v_cache_blocks = self.model.v_cache[cache_block_ids].detach().cpu()
            export_method = "advanced_index_cpu"
        self._sync_cache_device()
        export_time_s = time.perf_counter() - export_t0

        kv_bytes = self._tensor_nbytes(k_cache_blocks) + self._tensor_nbytes(v_cache_blocks)
        state = {
            "client_tag": client_tag,
            "request_id": request_id,
            "prompt_len": int(req.prompt_len),
            "output_len": int(req.output_len),
            "stop_token_ids": tuple(sorted(req.stop_token_ids)),
            "output_token_ids": list(req.output_token_ids),
            "spec_stats": dict(req.spec_stats),
            "spec_enabled": bool(req.spec_enabled),
            "logical_kv_len": logical_kv_len,
            "old_internal_request_id": int(req.request_id),
            "block_size": int(self.engine_config.block_size),
            "num_blocks": int(block_ids.numel()),
            "source_block_ids": block_ids_cpu.clone(),
            "export_method": export_method,
            "kv_dtype": str(self.model.k_cache.dtype),
            "kv_cache_shape": tuple(self.model.k_cache.shape),
            "kv_block_shape": tuple(k_cache_blocks.shape[1:]),
            "k_cache_blocks": k_cache_blocks,
            "v_cache_blocks": v_cache_blocks,
            "kv_bytes": int(kv_bytes),
            "export_time_s": float(export_time_s),
            "effective_export_GBps": self._effective_gbps(kv_bytes, export_time_s),
        }

        if free_after_export:
            self.model.free_seqs_resources([req.request_id])
            self._remove_finished_sessions([req.request_id])

        return state

    @torch.inference_mode()
    def import_session_from_migration(
        self,
        state: dict[str, Any],
        client_tag: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Import a session exported by export_session_for_migration()."""
        self._ensure_migration_ready()
        import_client_tag = str(client_tag if client_tag is not None else state["client_tag"])
        import_request_id = str(request_id if request_id is not None else state["request_id"])
        key = (import_client_tag, import_request_id)
        if key in self.sessions:
            raise RuntimeError(f"Target session already exists: {key}")

        block_size = int(state["block_size"])
        if block_size != int(self.engine_config.block_size):
            raise RuntimeError(
                f"Cannot import KV blocks with block_size={block_size} into worker block_size={self.engine_config.block_size}"
            )

        k_cache_blocks = state["k_cache_blocks"]
        v_cache_blocks = state["v_cache_blocks"]
        expected_block_shape = tuple(self.model.k_cache.shape[1:])
        if tuple(k_cache_blocks.shape[1:]) != expected_block_shape or tuple(v_cache_blocks.shape[1:]) != expected_block_shape:
            raise RuntimeError(
                f"KV block shape mismatch: exported k={tuple(k_cache_blocks.shape)}, "
                f"v={tuple(v_cache_blocks.shape)}, target block shape={expected_block_shape}"
            )

        req = Request(
            RawRequest(
                "",
                int(state["output_len"]),
                stop_token_ids=state.get("stop_token_ids", ()),
            )
        )
        req.prompt_len = int(state["prompt_len"])
        req.output_token_ids = [int(token_id) for token_id in state["output_token_ids"]]
        req.spec_stats = {k: int(v) for k, v in dict(state["spec_stats"]).items()}
        req.spec_enabled = bool(state["spec_enabled"])
        req.spec_proposal = None
        req.request_id = self.request_id_manager.get_id()

        block_manager = self.model.gpu_block_manager
        device = block_manager.num_seq_allocated_blocks.device
        seq_ids = torch.tensor([req.request_id], dtype=torch.int32, device=device)
        target_lens = torch.tensor([int(state["logical_kv_len"])], dtype=torch.int32, device=device)
        allocated = False

        self._sync_cache_device()
        import_t0 = time.perf_counter()
        try:
            block_manager.allocate_blocks_for_seqs(seq_ids, target_lens)
            allocated = True
            dst_block_ids = block_manager.get_allocated_block_ids(req.request_id)
            if int(dst_block_ids.numel()) != int(state["num_blocks"]):
                raise RuntimeError(
                    f"Allocated {int(dst_block_ids.numel())} blocks, expected {int(state['num_blocks'])}"
                )
            if int(dst_block_ids.numel()) > 0:
                cache_dst_block_ids = dst_block_ids.to(device=self.model.k_cache.device)
                self.model.k_cache[cache_dst_block_ids] = k_cache_blocks.to(
                    device=self.model.k_cache.device,
                    dtype=self.model.k_cache.dtype,
                )
                self.model.v_cache[cache_dst_block_ids] = v_cache_blocks.to(
                    device=self.model.v_cache.device,
                    dtype=self.model.v_cache.dtype,
                )
            self._sync_cache_device()
        except Exception:
            if allocated:
                block_manager.free_blocks_for_seqs(seq_ids)
            self.request_id_manager.free_id(req.request_id)
            raise

        import_time_s = time.perf_counter() - import_t0
        self.sessions[key] = _TargetSession(req, import_client_tag, import_request_id)
        kv_bytes = int(state["kv_bytes"])
        return {
            "client_tag": import_client_tag,
            "request_id": import_request_id,
            "old_internal_request_id": int(state["old_internal_request_id"]),
            "new_internal_request_id": int(req.request_id),
            "logical_kv_len": int(state["logical_kv_len"]),
            "num_blocks": int(state["num_blocks"]),
            "kv_bytes": kv_bytes,
            "import_time_s": float(import_time_s),
            "effective_import_GBps": self._effective_gbps(kv_bytes, import_time_s),
        }

    def _ensure_migration_ready(self):
        if not self.initialized or self.model is None or self.request_id_manager is None:
            raise RuntimeError("SwiftLLMTargetWorker is not initialized")
        if self.model.gpu_block_manager is None or self.model.k_cache is None or self.model.v_cache is None:
            raise RuntimeError("SwiftLLMTargetWorker model KV cache is not initialized")

    def _sync_cache_device(self):
        device = getattr(self.model.k_cache, "device", None) if self.model is not None else None
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _block_ids_are_contiguous(block_ids_cpu: torch.Tensor) -> bool:
        if int(block_ids_cpu.numel()) <= 1:
            return True
        block_ids_cpu = block_ids_cpu.to(dtype=torch.int64)
        expected = torch.arange(
            int(block_ids_cpu[0].item()),
            int(block_ids_cpu[0].item()) + int(block_ids_cpu.numel()),
            dtype=torch.int64,
        )
        return bool(torch.equal(block_ids_cpu, expected))

    @staticmethod
    def _tensor_nbytes(tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    @staticmethod
    def _effective_gbps(num_bytes: int, elapsed_s: float) -> float:
        if elapsed_s <= 0:
            return 0.0
        return float(num_bytes / elapsed_s / GB)

    async def _run_task_batch(self, tasks: list[TargetTask]):
        profiling = bool(
            self.model is not None and getattr(self.model, "record_forward_timing", False)
        )
        worker_batch_started = time.perf_counter() if profiling else 0.0
        model_task_prepare_started = worker_batch_started
        model_tasks: list[TargetTask] = []
        requests: list[Request] = []
        task_by_swift_req_id: dict[int, TargetTask] = {}
        finished_req_ids: list[int] = []
        pending_results: list[tuple[TargetTask, TargetResult]] = []

        for task in tasks:
            try:
                if task.phase == "end":
                    await self._finish_session(task, finished_req_ids, pending_results)
                    continue

                req = self._prepare_model_task(task)
                if req.request_id in task_by_swift_req_id:
                    raise RuntimeError("Multiple target tasks for the same request in one batch are not supported")
                task_by_swift_req_id[req.request_id] = task
                requests.append(req)
                model_tasks.append(task)
            except Exception as exc:  # pylint: disable=broad-except
                task.future.set_result(self._make_result(task, {}, error=str(exc)))

        if not requests:
            if finished_req_ids:
                await self._run_on_model_async(self.model.free_seqs_resources, finished_req_ids)
                self._remove_finished_sessions(finished_req_ids)
            self._set_pending_results(pending_results)
            return

        try:
            self._reserve_planned_prefill_bank_ranges(model_tasks, requests)
        except Exception as exc:  # pylint: disable=broad-except
            for task, req in zip(model_tasks, requests, strict=True):
                if task.future is not None and not task.future.done():
                    task.future.set_result(self._make_result(task, {}, error=str(exc)))
                if task.phase == "prefill":
                    self.sessions.pop(self._session_key(task), None)
                    if self.request_id_manager is not None:
                        self.request_id_manager.free_id(req.request_id)
            return

        model_task_prepare_finished = time.perf_counter() if profiling else 0.0
        batch_plan_started = model_task_prepare_finished
        batch_plan = build_batch_plan(requests)
        batch_plan_finished = time.perf_counter() if profiling else 0.0
        return_hidden = any(task.payload.get("return_hidden", False) for task in model_tasks)
        hidden_layer_ids = self._target_hidden_layer_ids() if return_hidden else None
        forward_started = time.perf_counter() if profiling else 0.0
        forward_output = await self._run_on_model_async(
            self.model.forward,
            batch_plan.input_ids_list(),
            batch_plan.seq_ids_list(),
            batch_plan.decoding_seq_lens_list(),
            return_hidden=return_hidden,
            hidden_layer_ids=hidden_layer_ids,
            return_dict=return_hidden,
        )
        forward_finished = time.perf_counter() if profiling else 0.0
        if isinstance(forward_output, ModelForwardOutput):
            output_tokens = forward_output.token_ids
            row_hidden_states = self._select_row_hidden_states(
                forward_output.hidden_states,
                batch_plan.input_ids_list(),
                len(batch_plan.prefill_rows),
            )
            hidden_ready_event = forward_output.hidden_ready_event
            forward_timing = forward_output.forward_timing
        else:
            output_tokens = forward_output
            row_hidden_states = None
            hidden_ready_event = None
            forward_timing = getattr(self.model, "_last_forward_timing", None)

        acceptance_output_started = forward_finished
        spec_crop_req_ids: list[int] = []
        spec_crop_target_lens: list[int] = []

        for row_idx, row in enumerate(batch_plan.prefill_rows):
            task = task_by_swift_req_id[row.request.request_id]
            token_id = output_tokens[row_idx]
            row.request.output_token_ids.append(token_id)
            target_hidden, target_hidden_ready_event = self._hidden_payload_for_row(row_hidden_states, row_idx, hidden_ready_event)
            payload = {
                "posterior_token_ids": [token_id],
                "accepted_token_ids": [token_id],
                "accept_length": 1,
                "num_accepted_draft_tokens": 0,
                "target_hidden": target_hidden,
                "finished": row.request.is_finished(),
                "output_token_ids": list(row.request.output_token_ids),
                "logical_kv_len": int(row.request.logical_kv_len_after_current_state()),
            }
            if target_hidden_ready_event is not None:
                payload["target_hidden_ready_event"] = target_hidden_ready_event
            bank_location = self._prefill_bank_location_payload(row.request)
            if bank_location is not None:
                payload["prefill_bank_location"] = bank_location
                session = self.sessions.get(self._session_key(task))
                if session is not None:
                    session.bank_fence = _SessionBankFence(
                        bank_id=int(bank_location["bank_id"]),
                        bank_epoch=int(bank_location["bank_epoch"]),
                        row=int(bank_location["request_row"]),
                        start_block=int(bank_location["bank_offset_blocks"]),
                        capacity_blocks=int(bank_location["capacity_blocks"]),
                        batch_id=None if bank_location.get("batch_id") is None else str(bank_location["batch_id"]),
                    )
            pending_results.append((task, self._make_result(task, payload)))

        for row in batch_plan.normal_decode_rows:
            task = task_by_swift_req_id[row.request.request_id]
            token_id = output_tokens[row.output_row_start]
            row.request.output_token_ids.append(token_id)
            payload = self._single_token_payload(row.request, token_id, row_hidden_states, row.output_row_start)
            pending_results.append((task, self._make_result(task, payload)))

        for plan_item in batch_plan.verify_plan_items:
            req = plan_item.request
            task = task_by_swift_req_id[req.request_id]
            posterior = output_tokens[
                plan_item.output_row_start:plan_item.output_row_start + plan_item.output_row_count
            ]
            remaining = req.output_len - len(req.output_token_ids)
            accepted_token_ids, num_accepted_draft_tokens = compute_acceptance_for_plan(
                plan_item,
                posterior,
                remaining_output_len=remaining,
                stop_token_ids=req.stop_token_ids,
            )
            req.output_token_ids.extend(accepted_token_ids)
            req.spec_stats["num_draft_tokens"] += len(plan_item.draft_token_ids)
            req.spec_stats["num_accepted_tokens"] += num_accepted_draft_tokens
            req.spec_stats["num_spec_steps"] += 1
            req.spec_proposal = None

            target_hidden, target_hidden_ready_event = self._hidden_payload_for_rows(
                row_hidden_states,
                plan_item.output_row_start,
                len(accepted_token_ids),
                hidden_ready_event,
            )
            payload = {
                "posterior_token_ids": posterior,
                "accepted_token_ids": accepted_token_ids,
                "accept_length": len(accepted_token_ids),
                "num_accepted_draft_tokens": num_accepted_draft_tokens,
                "target_hidden": target_hidden,
                "finished": req.is_finished(),
                "output_token_ids": list(req.output_token_ids),
            }
            if target_hidden_ready_event is not None:
                payload["target_hidden_ready_event"] = target_hidden_ready_event
            pending_results.append((task, self._make_result(task, payload)))
            spec_crop_req_ids.append(req.request_id)
            spec_crop_target_lens.append(req.logical_kv_len_after_current_state())

        for req in requests:
            if not req.is_finished():
                continue
            task = task_by_swift_req_id.get(req.request_id)
            if task is not None and bool(task.payload.get("keep_bank_range_for_export", False)):
                continue
            finished_req_ids.append(req.request_id)

        acceptance_output_finished = time.perf_counter() if profiling else 0.0
        crop_started = acceptance_output_finished
        if spec_crop_req_ids:
            await self._run_on_model_async(self.model.crop_seqs_resources, spec_crop_req_ids, spec_crop_target_lens)
        crop_finished = time.perf_counter() if profiling else 0.0
        finalization_started = crop_finished
        if finished_req_ids:
            await self._run_on_model_async(self.model.free_seqs_resources, finished_req_ids)
            self._remove_finished_sessions(finished_req_ids)
        if profiling:
            finalization_finished = time.perf_counter()
            stage_timing = {
                "worker_batch_s": finalization_finished - worker_batch_started,
                "model_task_prepare_s": model_task_prepare_finished - model_task_prepare_started,
                "batch_plan_s": batch_plan_finished - batch_plan_started,
                "forward_wall_s": forward_finished - forward_started,
                "acceptance_output_s": acceptance_output_finished - acceptance_output_started,
                "crop_s": crop_finished - crop_started,
                "finalization_s": finalization_finished - finalization_started,
            }
            for _task, result in pending_results:
                result.payload["swiftllm_stage_timing"] = dict(stage_timing)
                if forward_timing is not None:
                    result.payload["swiftllm_forward_timing"] = dict(forward_timing)
        self._write_divergence_trace(
            model_tasks=model_tasks,
            batch_plan=batch_plan,
            output_tokens=output_tokens,
            pending_results=pending_results,
        )
        self._set_pending_results(pending_results)

    def _set_pending_results(self, pending_results: list[tuple[TargetTask, TargetResult]]):
        for task, result in pending_results:
            if task.future is not None and not task.future.done():
                task.future.set_result(result)

    def _target_hidden_layer_ids(self) -> list[int] | None:
        raw = getattr(self.engine_config, "speculative_target_layer_ids", None)
        if raw is None or raw == "":
            return None
        if isinstance(raw, str):
            return [int(item.strip()) for item in raw.split(",") if item.strip()]
        return [int(item) for item in raw]

    def _prepare_model_task(self, task: TargetTask) -> Request:
        key = self._session_key(task)
        if task.phase == "prefill":
            if key in self.sessions:
                raise RuntimeError(f"Target session already exists: {key}")
            req = Request(
                RawRequest(
                    "",
                    task.payload["max_output_len"],
                    stop_token_ids=task.payload.get("stop_token_ids", ()),
                )
            )
            req.prompt_token_ids = list(task.payload["input_ids"])
            req.prompt_len = len(req.prompt_token_ids)
            req.request_id = self.request_id_manager.get_id()
            req.spec_enabled = True
            self.sessions[key] = _TargetSession(req, task.client_tag, task.request_id)
            return req

        if key not in self.sessions:
            raise RuntimeError(f"Unknown target session: {key}")
        req = self.sessions[key].request
        if task.phase == "decode":
            if not req.has_unstored_anchor():
                raise RuntimeError("decode requires a prefilled request with an anchor token")
            return req
        if task.phase == "verify":
            if not req.has_unstored_anchor():
                raise RuntimeError("verify requires a prefilled request with an anchor token")
            draft_token_ids = list(task.payload["draft_token_ids"])
            max_draft_tokens = self.engine_config.speculative_max_draft_tokens
            if max_draft_tokens and len(draft_token_ids) > max_draft_tokens:
                raise RuntimeError(
                    f"draft_token_ids length {len(draft_token_ids)} exceeds speculative_max_draft_tokens {max_draft_tokens}"
                )
            proposal_kind = str(task.payload.get("proposal_kind") or task.payload.get("kind") or "linear")
            req.spec_proposal = DraftProposal(req.request_id, proposal_kind, draft_token_ids)
            return req
        raise RuntimeError(f"Unsupported target task phase: {task.phase}")

    def _reserve_planned_prefill_bank_ranges(self, tasks: list[TargetTask], requests: list[Request]) -> None:
        planned = tuple((task, req, task.payload.get("prefill_bank_plan")) for task, req in zip(tasks, requests, strict=True) if task.phase == "prefill" and task.payload.get("prefill_bank_plan"))
        if not planned:
            return
        if self.model is None or self.model.gpu_block_manager is None:
            raise RuntimeError("planned prefill requires GPU block manager")
        manager = self.model.gpu_block_manager
        by_bank: dict[tuple[int, int, str | None], list[tuple[TargetTask, Request, dict[str, Any]]]] = {}
        for task, req, raw_plan in planned:
            plan = dict(raw_plan)
            key = (int(plan["bank_id"]), int(plan["bank_epoch"]), None if plan.get("batch_id") is None else str(plan["batch_id"]))
            by_bank.setdefault(key, []).append((task, req, plan))
        for (bank_id, bank_epoch, batch_id), rows in by_bank.items():
            descriptor = manager.get_bank_descriptor(bank_id)
            if int(descriptor.epoch) != bank_epoch:
                raise RuntimeError("planned prefill bank_epoch mismatch")
            expected_offset = int(getattr(descriptor, "alloc_ptr", 0))
            reservations = []
            for _task, req, plan in rows:
                offset_blocks = int(plan["bank_offset_blocks"])
                block_count = int(plan["block_count"])
                if offset_blocks != expected_offset:
                    raise RuntimeError("planned prefill bank_offset does not match next allocation")
                expected_offset += block_count
                reservations.append((int(req.request_id), block_count, int(req.prompt_len), 0, batch_id))
            manager.reserve_in_bank_batch_atomic(bank_id, reservations, reset_bank=False)

    async def _finish_session(
        self,
        task: TargetTask,
        finished_req_ids: list[int],
        pending_results: list[tuple[TargetTask, TargetResult]],
    ):
        key = self._session_key(task)
        session = self.sessions.get(key)
        if session is not None:
            finished_req_ids.append(session.request.request_id)
        pending_results.append((task, self._make_result(task, {"finished": True})))

    def _remove_finished_sessions(self, finished_req_ids: list[int]):
        finished_req_id_set = set(finished_req_ids)
        for key, session in list(self.sessions.items()):
            if session.request.request_id in finished_req_id_set:
                self.sessions.pop(key)
                self.request_id_manager.free_id(session.request.request_id)

    def _single_token_payload(self, req: Request, token_id: int, row_hidden_states, row_idx: int) -> dict[str, Any]:
        return {
            "posterior_token_ids": [token_id],
            "accepted_token_ids": [token_id],
            "accept_length": 1,
            "num_accepted_draft_tokens": 0,
            "target_hidden": self._hidden_for_row(row_hidden_states, row_idx),
            "finished": req.is_finished(),
            "output_token_ids": list(req.output_token_ids),
        }

    def _prefill_bank_location_payload(self, req: Request) -> dict[str, Any] | None:
        """Return contiguous bank metadata while the prefill request is still live."""

        manager = getattr(self.model, "gpu_block_manager", None) if self.model is not None else None
        if manager is None or not getattr(manager, "double_bank_enabled", False):
            return None
        row = int(req.request_id)
        candidate_ids: list[int] = []
        for value in (getattr(manager, "active_bank_id", None), getattr(manager, "standby_bank_id", None), 0, 1):
            if value is None:
                continue
            bank_id = int(value)
            if bank_id not in candidate_ids:
                candidate_ids.append(bank_id)
        for bank_id in candidate_ids:
            try:
                descriptor = manager.get_bank_descriptor(bank_id)
            except Exception:
                continue
            location = (getattr(descriptor, "request_ranges", {}) or {}).get(row)
            if location is None:
                continue
            manager.validate_bank_location(location)
            logical_fn = getattr(req, "logical_kv_len_after_current_state", None)
            logical_kv_len = int(logical_fn() if callable(logical_fn) else getattr(location, "logical_kv_len", 0))
            num_blocks = int(getattr(location, "num_blocks", 0))
            allocated = getattr(manager, "num_seq_allocated_blocks", None)
            if allocated is not None:
                try:
                    num_blocks = int(allocated[row].item())
                except Exception:
                    pass
            return {
                "request_row": row,
                "bank_id": int(getattr(descriptor, "bank_id", getattr(location, "bank_id", 0))),
                "bank_epoch": int(getattr(location, "bank_epoch", getattr(descriptor, "epoch", 0))),
                "bank_base_block": int(getattr(location, "bank_base_block", getattr(descriptor, "base_block", 0))),
                "bank_offset_blocks": int(getattr(location, "request_start_block", 0)),
                "capacity_blocks": int(getattr(location, "num_blocks", num_blocks)),
                "num_blocks": int(num_blocks),
                "logical_kv_len": int(logical_kv_len),
                "kv_version": int(getattr(location, "kv_version", 0)),
                "batch_id": None if getattr(descriptor, "batch_id", None) is None else str(getattr(descriptor, "batch_id")),
                "prompt_len": int(getattr(req, "prompt_len", logical_kv_len)),
            }
        return None

    def _select_row_hidden_states(self, hidden_states, input_ids_list: list[list[int]], num_prefill_rows: int):
        if hidden_states is None:
            return None
        row_hidden_states = []
        offset = 0
        for row_idx, input_ids in enumerate(input_ids_list):
            row_len = len(input_ids)
            if row_idx < num_prefill_rows:
                row_hidden_states.append(hidden_states[offset:offset + row_len])
            else:
                row_hidden_states.append(hidden_states[offset])
            offset += row_len
        return row_hidden_states

    def _hidden_for_row(self, row_hidden_states, row_idx: int):
        if row_hidden_states is None or row_idx < 0:
            return None
        if row_idx >= len(row_hidden_states):
            return None
        hidden = row_hidden_states[row_idx]
        return self._ensure_hidden_batch_seq(hidden)

    def _hidden_payload_for_row(self, row_hidden_states, row_idx: int, producer_ready_event):
        if row_hidden_states is None or row_idx < 0 or row_idx >= len(row_hidden_states):
            return None, None
        source = row_hidden_states[row_idx]
        return self._hidden_payload_from_source(lambda: self._ensure_hidden_batch_seq(source), source, producer_ready_event)

    def _hidden_for_rows(self, row_hidden_states, row_start: int, row_count: int):
        if row_hidden_states is None or row_start < 0 or row_count <= 0:
            return None
        rows = row_hidden_states[row_start:row_start + row_count]
        if not rows:
            return None
        return self._hidden_from_rows(rows)

    def _hidden_from_rows(self, rows):
        first = rows[0]
        if hasattr(first, "dim") and first.dim() == 1:
            hidden = first.new_empty((len(rows), first.shape[-1]))
            for idx, row in enumerate(rows):
                hidden[idx, :] = row
            return hidden.unsqueeze(0)
        if len(rows) == 1:
            return self._ensure_hidden_batch_seq(rows[0])
        if hasattr(first, "dim"):
            return torch.cat([self._ensure_hidden_batch_seq(row) for row in rows], dim=1)
        return self._ensure_hidden_batch_seq(rows[0])

    def _hidden_payload_for_rows(self, row_hidden_states, row_start: int, row_count: int, producer_ready_event):
        if row_hidden_states is None or row_start < 0 or row_count <= 0:
            return None, None
        rows = row_hidden_states[row_start:row_start + row_count]
        if not rows:
            return None, None
        return self._hidden_payload_from_source(lambda: self._hidden_from_rows(rows), rows[0], producer_ready_event)

    def _hidden_payload_from_source(self, build_hidden, source, producer_ready_event):
        if producer_ready_event is None:
            raise RuntimeError("target hidden producer ready event is missing")
        stream = torch.cuda.Stream(device=source.device)
        with torch.cuda.stream(stream):
            stream.wait_event(producer_ready_event)
            hidden = build_hidden()
            if hasattr(hidden, "is_contiguous") and not hidden.is_contiguous():
                hidden = hidden.contiguous()
            ready_event = torch.cuda.Event()
            ready_event.record(stream)
        return hidden, ready_event

    def _ensure_hidden_batch_seq(self, hidden):
        if hasattr(hidden, "dim"):
            if hidden.dim() == 1:
                return hidden.view(1, 1, -1)
            if hidden.dim() == 2:
                return hidden.unsqueeze(0)
        return hidden

    def _write_divergence_trace(
        self,
        *,
        model_tasks: list[TargetTask],
        batch_plan,
        output_tokens: list[int],
        pending_results: list[tuple[TargetTask, TargetResult]],
    ) -> None:
        if not _DIVERGENCE_TRACE_DIR:
            return
        try:
            os.makedirs(_DIVERGENCE_TRACE_DIR, exist_ok=True)
            task_by_key = {(task.client_tag, task.request_id): task for task in model_tasks}
            task_by_row = {}
            for task in model_tasks:
                session = self.sessions.get((task.client_tag, task.request_id))
                if session is not None:
                    task_by_row[int(session.request.request_id)] = task
            result_by_key = {
                (task.client_tag, task.request_id): _json_safe_result(result.payload)
                for task, result in pending_results
            }
            rows = []
            for row_index, row in enumerate(batch_plan.rows):
                task = task_by_row.get(int(row.seq_id))
                rows.append({
                    "row_index": int(row_index),
                    "kind": str(row.kind),
                    "client_tag": None if task is None else task.client_tag,
                    "request_id": None if task is None else task.request_id,
                    "swift_row": int(row.seq_id),
                    "input_ids": [int(token) for token in row.input_ids],
                    "seq_len": None if row.seq_len is None else int(row.seq_len),
                    "output_token": int(output_tokens[row_index]),
                })
            verify_items = []
            for item in batch_plan.verify_plan_items:
                task = task_by_row.get(int(item.request.request_id))
                key = None if task is None else (task.client_tag, task.request_id)
                payload = {} if key is None else result_by_key.get(key, {})
                verify_items.append({
                    "client_tag": None if task is None else task.client_tag,
                    "request_id": None if task is None else task.request_id,
                    "swift_row": int(item.request.request_id),
                    "prompt_len": int(getattr(item.request, "prompt_len", 0)),
                    "output_len_limit": int(getattr(item.request, "output_len", 0)),
                    "output_tokens_after": [int(token) for token in getattr(item.request, "output_token_ids", [])],
                    "draft_token_ids": [int(token) for token in item.draft_token_ids],
                    "proposal_kind": str(item.proposal_kind),
                    "output_row_start": int(item.output_row_start),
                    "output_row_count": int(item.output_row_count),
                    "posterior_token_ids": [int(token) for token in payload.get("posterior_token_ids", [])],
                    "accepted_token_ids": [int(token) for token in payload.get("accepted_token_ids", [])],
                    "num_accepted_draft_tokens": int(payload.get("num_accepted_draft_tokens", -1)),
                    "logical_kv_len_after": int(payload.get("logical_kv_len", item.request.logical_kv_len_after_current_state())),
                    "bank_location": payload.get("bank_location"),
                })
            record = {
                "pid": int(os.getpid()),
                "time_ns": int(time.perf_counter_ns()),
                "cuda_device": None if self._model_cuda_device_index() is None else int(self._model_cuda_device_index()),
                "tasks": [
                    {
                        "phase": task.phase,
                        "client_tag": task.client_tag,
                        "request_id": task.request_id,
                        "payload": _json_safe_result(task.payload),
                    }
                    for task in model_tasks
                ],
                "rows": rows,
                "verify_items": verify_items,
                "forward_metadata": getattr(self.model, "_last_forward_metadata", None),
                "logits_probe": getattr(getattr(self.model, "post_layer", None), "_last_logits_probe", None),
            }
            path = os.path.join(_DIVERGENCE_TRACE_DIR, f"target-worker-{os.getpid()}.jsonl")
            with open(path, "a", encoding="utf-8") as out:
                out.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:  # diagnostics must not perturb inference
            path = os.path.join(_DIVERGENCE_TRACE_DIR, f"target-worker-{os.getpid()}.errors")
            with open(path, "a", encoding="utf-8") as out:
                out.write(f"{type(exc).__name__}: {exc}\n")



def _wait_on_ready_event(event):
    torch.cuda.current_stream().wait_event(event)


def _json_safe_result(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in {"target_hidden", "target_hidden_ready_event", "event"}:
                continue
            out[str(key)] = _json_safe_result(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_json_safe_result(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    return repr(value)


def _bank_request_rows(plan: dict[str, Any], request_ids: list[str]) -> dict[str, int]:
    rows = plan.get("request_rows") or plan.get("block_table_rows") or {}
    if isinstance(rows, dict):
        return {str(k): int(v) for k, v in rows.items()}
    rows_list = list(rows or [])
    return {request_id: int(rows_list[idx]) for idx, request_id in enumerate(request_ids) if idx < len(rows_list)}


def _bank_verify_payloads(plan: dict[str, Any], request_ids: list[str]) -> dict[str, dict[str, Any]]:
    payloads = plan.get("verification_tokens_or_draft_params") or plan.get("verify_payloads") or {}
    if isinstance(payloads, dict):
        return {str(k): dict(v or {}) for k, v in payloads.items()}
    items = list(payloads or [])
    return {request_id: dict(items[idx] or {}) for idx, request_id in enumerate(request_ids) if idx < len(items)}


class StarSDTargetWorkerAdapter:
    """Dictionary-packet adapter for in-process StarSD integration.

    This is intentionally not a network listener. StarSD can wrap it with its
    transport/RPC layer while keeping SwiftLLM free of a hard StarSD import.
    """

    def __init__(self, target_worker: SwiftLLMTargetWorker):
        self.target_worker = target_worker

    async def handle_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        msg = packet.get("msg")
        client_tag = packet.get("client_tag", packet.get("payload", {}).get("client_tag", ""))
        request_id = str(packet.get("request_id", packet.get("payload", {}).get("request_id", "")))
        payload = packet.get("payload", {})

        if msg == "initialize":
            if not self.target_worker.initialized:
                await self.target_worker.initialize()
            result = TargetResult("initialize", client_tag, request_id, "initialize", {"initialized": True})
        elif msg == "prefill_req":
            result = await self.target_worker.submit_prefill(
                client_tag,
                request_id,
                payload["input_ids"],
                payload["max_output_len"],
                **{k: v for k, v in payload.items() if k not in {"input_ids", "max_output_len"}},
            )
        elif msg == "verify_req":
            result = await self.target_worker.submit_verify(
                client_tag,
                request_id,
                payload["draft_token_ids"],
                **{k: v for k, v in payload.items() if k != "draft_token_ids"},
            )
        elif msg == "decode_req":
            result = await self.target_worker.submit_decode(client_tag, request_id, **payload)
        elif msg == "end_req":
            result = await self.target_worker.submit_end(client_tag, request_id)
        else:
            result = TargetResult(str(packet.get("task_id", "")), client_tag, request_id, "unknown", {}, error=f"Unsupported msg: {msg}")

        return {
            "proto": "starsd.v2",
            "algo": packet.get("algo", "dflash"),
            "role": "target",
            "phase": packet.get("phase", "inference"),
            "msg": self._response_msg(msg),
            "client_tag": result.client_tag,
            "request_id": result.request_id,
            "payload": result.payload,
            "error": result.error,
        }

    def _response_msg(self, msg: str | None) -> str:
        if msg is None:
            return "error_resp"
        if msg.endswith("_req"):
            return msg[:-4] + "_resp"
        return msg + "_resp"
