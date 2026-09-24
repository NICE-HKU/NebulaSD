"""Small process-local DMA contracts and a single-owner copy executor."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import os
from threading import Event
from time import perf_counter_ns
from typing import Protocol

from .arena import HostKVExtent


@dataclass(frozen=True)
class CopyRegion:
    extent: HostKVExtent
    gpu_begin_block: int
    host_begin_block: int
    block_count: int

    def __post_init__(self) -> None:
        if min(self.gpu_begin_block, self.host_begin_block, self.block_count) < 0:
            raise ValueError("negative copy range")
        if self.host_begin_block + self.block_count > self.extent.capacity_blocks:
            raise ValueError("copy exceeds HostKV extent")


@dataclass(frozen=True)
class HostCompletedFence:
    """Producer has already synchronized the dependency before publishing it."""


@dataclass(frozen=True)
class CopyPlan:
    direction: str
    regions: tuple[CopyRegion, ...]
    dependencies: tuple[object, ...] = ()
    round_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.direction not in ("D2H", "H2D") or not self.regions:
            raise ValueError("copy plan requires a direction and non-empty batch")
        if self.round_ids and len(self.round_ids) != len(self.regions):
            raise ValueError("copy rounds must match regions")


@dataclass(frozen=True)
class CopyReceipt:
    """Host timestamps; completed_ns means observation, not GPU completion.

    submitted_ns is the DMA thread's launch entry. The physical CUDA interval
    is measured separately by the backend's events. No clock conversion occurs
    on the production hot path. For chunked H2D, duration_ms is the sum of
    per-chunk event durations, and launch_returned_ns is the return time of the
    last submitted chunk, not a single CUDA launch cost. With event waiting,
    last_pending_ns remains the last launch return; no intermediate pending
    observations are collected.
    """
    submitted_ns: int
    completed_ns: int
    duration_ms: float
    ready_publish_started_ns: int = 0
    ready_fact_published_ns: int = 0
    ready_published_ns: int = 0
    enqueued_ns: int = 0
    launch_returned_ns: int = 0
    last_pending_ns: int = 0
    retired_ns: int = 0


class CopyTicket(Protocol):
    # CUDA tickets additionally expose synchronize() for a sleeping host wait.
    # Query-only backends and diagnostic wrappers keep their existing contract.
    def query(self) -> bool: ...
    def duration_ms(self) -> float: ...


class CopyBackend(Protocol):
    """Launches share one ordered stream: a completed ticket fences prior launches."""
    def launch(self, plan: CopyPlan) -> CopyTicket: ...
    def close(self) -> None: ...


class CopyExecutor:
    """One DMA thread; never publishes Tables or mutates Bank/session state.

Only one batch may be outstanding. Event polling does not occupy the Target
owner loop and never uses a device-wide synchronize. Shutdown joins DMA before
unregistering host memory, including after partial submission failure.

H2D defaults to waiting on the final event in this DMA thread, where supported.
With the default zero chunk budget, the whole plan is enqueued before waiting.
"""

    def __init__(self, backend: CopyBackend, *, poll_interval_s: float = 0.0001,
                 h2d_chunk_bytes: int | None = None,
                 h2d_group_size: int | None = None,
                 h2d_wait_mode: str | None = None) -> None:
        if poll_interval_s < 0:
            raise ValueError("copy poll interval must be non-negative")
        if h2d_chunk_bytes is None:
            h2d_chunk_bytes = _h2d_chunk_bytes_from_env()
        if h2d_chunk_bytes < 0:
            raise ValueError("H2D chunk byte budget must be non-negative")
        if h2d_group_size is None:
            h2d_group_size = int(os.environ.get("STARSD_H2D_GROUP_SIZE") or "4")
        if h2d_group_size < 1:
            raise ValueError("H2D group size must be positive")
        if h2d_wait_mode is None:
            h2d_wait_mode = os.environ.get("STARSD_H2D_WAIT_MODE") or "event"
        if h2d_wait_mode not in ("poll", "event"):
            raise ValueError("H2D wait mode must be poll or event")
        self.backend = backend
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="starsd-copy")
        self._future: Future[CopyReceipt] | None = None
        self._closed = False
        self._pause = Event()
        self._interval = poll_interval_s
        self._h2d_chunk_bytes = h2d_chunk_bytes
        self._h2d_group_size = h2d_group_size
        self._h2d_wait_mode = h2d_wait_mode
        # ThreadPoolExecutor starts lazily. Pay thread creation at setup, not
        # between the first enqueue timestamp and launch; no CUDA work here.
        self._pool.submit(lambda: None).result()

    def submit(self, plan: CopyPlan) -> Future[CopyReceipt]:
        if self._closed:
            raise RuntimeError("copy executor is closed")
        if self._future is not None and not self._future.done():
            raise RuntimeError("copy executor has an outstanding batch")
        if self._future is not None:
            self._future.result()  # A failed stream must never accept more work.
        self._future = self._pool.submit(self._run, plan, perf_counter_ns())
        return self._future

    def _run(self, plan: CopyPlan, enqueued_ns: int) -> CopyReceipt:
        started = perf_counter_ns()
        chunks = tuple(_plan_chunks(plan, self._plan_block_bytes(plan), self._h2d_chunk_bytes))
        chunk_durations = []
        launch_returned, last_pending, observed = started, started, started
        group_size = self._h2d_group_size if getattr(plan, "direction", None) == "H2D" else 1
        for begin in range(0, len(chunks), group_size):
            # Keep every ticket alive through completion and timing collection.
            # The last event on the backend's ordered stream fences this group.
            tickets = [self.backend.launch(chunk.plan)
                       for chunk in chunks[begin:begin + group_size]]
            launch_returned = perf_counter_ns()
            last_pending = launch_returned
            synchronize = (getattr(tickets[-1], "synchronize", None)
                           if getattr(plan, "direction", None) == "H2D"
                           and self._h2d_wait_mode == "event" else None)
            if synchronize is not None:
                # Only this DMA job thread waits; the whole-plan Future remains
                # pending. Query-only backends retain their completion contract.
                synchronize()
                observed = perf_counter_ns()
            else:
                while True:
                    query_started = perf_counter_ns()
                    if tickets[-1].query():
                        observed = perf_counter_ns()
                        break
                    last_pending = query_started
                    self._pause.wait(self._interval)
            chunk_durations.extend(ticket.duration_ms() for ticket in tickets)
        return CopyReceipt(started, observed, sum(chunk_durations), enqueued_ns=enqueued_ns,
                           launch_returned_ns=launch_returned, last_pending_ns=last_pending)

    def _plan_block_bytes(self, plan: CopyPlan) -> int:
        descriptor = getattr(getattr(self.backend, "arena", None), "descriptor", None)
        return int(getattr(descriptor, "block_bytes", 0) or 0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # FIFO submission places cleanup after the last copy, even if it raised.
        cleanup = self._pool.submit(self.backend.close)
        try:
            cleanup.result()
        finally:
            self._pool.shutdown(wait=True)


@dataclass(frozen=True)
class _PlanChunk:
    plan: CopyPlan
    bytes: int
    oversized: bool = False


def _h2d_chunk_bytes_from_env() -> int:
    value = os.environ.get("STARSD_H2D_CHUNK_BYTES")
    if value is None or value == "":
        return 0
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("STARSD_H2D_CHUNK_BYTES must be an integer byte count") from exc
    if parsed < 0:
        raise ValueError("STARSD_H2D_CHUNK_BYTES must be non-negative")
    return parsed


def _region_bytes(region: CopyRegion, block_bytes: int) -> int:
    return region.block_count * block_bytes * 2


def _plan_chunks(plan: CopyPlan, block_bytes: int, budget_bytes: int) -> tuple[_PlanChunk, ...]:
    if getattr(plan, "direction", None) != "H2D" or budget_bytes <= 0 or block_bytes <= 0:
        return (_PlanChunk(plan, sum(_region_bytes(r, block_bytes) for r in getattr(plan, "regions", ()))),)
    chunks, regions, round_ids = [], [], []
    total = 0
    dependencies = plan.dependencies

    def flush() -> None:
        nonlocal regions, round_ids, total, dependencies
        if not regions:
            return
        ids = tuple(round_ids) if plan.round_ids else ()
        chunk_plan = CopyPlan(plan.direction, tuple(regions), dependencies, ids)
        chunks.append(_PlanChunk(chunk_plan, total, total > budget_bytes))
        regions, round_ids, total = [], [], 0
        dependencies = ()

    for index, region in enumerate(plan.regions):
        size = _region_bytes(region, block_bytes)
        if regions and total + size > budget_bytes:
            flush()
        regions.append(region)
        if plan.round_ids:
            round_ids.append(plan.round_ids[index])
        total += size
        if size > budget_bytes:
            flush()
    flush()
    return tuple(chunks)
