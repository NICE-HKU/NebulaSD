"""Admission-time HostKV allocation and process-local lifetime pins.

The allocator never commits copy results. Ready versions remain Worker-owned
Table facts. Pins protect payload lifetime while native multiprocess IPC is
developed separately; all local Target lanes must share this allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from nebulasd.core.ids import HOST_SLOT_GENERATION, WRITER_LEASE_GENERATION
from .arena import HostKVCapacityError, HostKVExtent, SharedHostKVArena


@dataclass
class _Allocation:
    extent: HostKVExtent
    readers: int = 0
    writing: bool = False


class HostKVAllocator:
    def __init__(self, arena: SharedHostKVArena) -> None:
        self.arena = arena
        self._free = [(0, arena.descriptor.total_blocks)]
        self._allocations: dict[int, _Allocation] = {}
        self._generations: dict[int, int] = {}
        self._lock = RLock()

    def allocate(self, request_slot: int, request_epoch: int, capacity_blocks: int) -> HostKVExtent:
        with self._lock:
            if request_slot in self._allocations:
                raise ValueError("request already has a HostKV allocation")
            if capacity_blocks <= 0:
                raise ValueError("HostKV capacity must be positive")
            for index, (offset, available) in enumerate(self._free):
                if available < capacity_blocks:
                    continue
                generation = self._generations.get(request_slot, 1)
                extent = self.arena.make_extent(
                    request_slot=request_slot, request_epoch=request_epoch,
                    host_slot=request_slot, host_slot_generation=generation,
                    writer_lease_generation=generation, offset_blocks=offset,
                    capacity_blocks=capacity_blocks,
                )
                self._free[index:index + 1] = (
                    [(offset + capacity_blocks, available - capacity_blocks)]
                    if available > capacity_blocks else []
                )
                self._allocations[request_slot] = _Allocation(extent)
                return extent
            raise HostKVCapacityError("HostKV arena capacity exhausted")

    def require(self, request_slot: int, request_epoch: int) -> HostKVExtent:
        with self._lock:
            allocation = self._allocations.get(request_slot)
            if allocation is None or allocation.extent.request_epoch != request_epoch:
                raise ValueError("stale or missing HostKV request allocation")
            return allocation.extent

    def pin(self, extent: HostKVExtent, *, write: bool) -> bool:
        """Return False for a live conflicting copy, reject stale leases."""
        with self._lock:
            current = self.require(extent.request_slot, extent.request_epoch)
            if current != extent:
                raise ValueError("stale HostKV extent/slot/writer lease")
            allocation = self._allocations[extent.request_slot]
            if allocation.writing or (write and allocation.readers):
                return False
            if write:
                allocation.writing = True
            else:
                allocation.readers += 1
            return True

    def unpin(self, extent: HostKVExtent, *, write: bool) -> None:
        with self._lock:
            if self.require(extent.request_slot, extent.request_epoch) != extent:
                raise ValueError("stale HostKV unpin")
            allocation = self._allocations[extent.request_slot]
            if write:
                if not allocation.writing:
                    raise ValueError("HostKV writer pin is not held")
                allocation.writing = False
            else:
                if not allocation.readers:
                    raise ValueError("HostKV reader pin is not held")
                allocation.readers -= 1

    def recycle(self, request_slot: int, request_epoch: int, *, quiescent: bool) -> None:
        """Engine must first stop dispatch and acknowledge Worker retirement."""
        with self._lock:
            extent = self.require(request_slot, request_epoch)
            allocation = self._allocations[request_slot]
            if not quiescent or allocation.writing or allocation.readers:
                raise RuntimeError("cannot recycle HostKV with live request/copy ownership")
            self._generations[request_slot] = HOST_SLOT_GENERATION.next(extent.host_slot_generation)
            WRITER_LEASE_GENERATION.validate(self._generations[request_slot])
            del self._allocations[request_slot]
            self._free.append((extent.offset_blocks, extent.capacity_blocks))
            merged: list[tuple[int, int]] = []
            for offset, length in sorted(self._free):
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    previous, size = merged.pop()
                    merged.append((previous, size + length))
                else:
                    merged.append((offset, length))
            self._free = merged

    @property
    def allocated_count(self) -> int:
        with self._lock:
            return len(self._allocations)
