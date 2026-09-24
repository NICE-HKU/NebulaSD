"""Shared HostKV payload arena using work package 01 numeric handles."""

from __future__ import annotations

import ctypes
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory

from nebulasd.core.handles import HostKVArenaHandle
from nebulasd.core.ids import (
    HOST_SLOT,
    HOST_SLOT_GENERATION,
    KV_VERSION,
    REQUEST_EPOCH,
    REQUEST_SLOT,
    WRITER_LEASE_GENERATION,
)


class HostKVCapacityError(ValueError):
    """A HostKV extent or view exceeds the arena capacity."""


@dataclass(frozen=True, slots=True)
class HostKVExtent:
    """Process-local HostKV convenience object.

    This object is not canonical scheduler state. Writers publish allocation
    fields to HostKVAllocationBlock and completion fields to D2H/H2D blocks
    separately.
    """

    request_slot: int
    request_epoch: int
    host_slot: int
    host_slot_generation: int
    writer_lease_generation: int
    arena: HostKVArenaHandle
    capacity_blocks: int
    committed_blocks: int = 0
    kv_version: int = 0

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        HOST_SLOT.validate(self.host_slot)
        HOST_SLOT_GENERATION.validate(self.host_slot_generation)
        WRITER_LEASE_GENERATION.validate(self.writer_lease_generation)
        KV_VERSION.validate(self.kv_version)
        _non_negative(self.capacity_blocks, "capacity_blocks")
        _non_negative(self.committed_blocks, "committed_blocks")
        if self.capacity_blocks == 0:
            raise HostKVCapacityError("capacity_blocks must be positive")
        if self.committed_blocks > self.capacity_blocks:
            raise HostKVCapacityError("committed_blocks exceeds capacity_blocks")

    @property
    def offset_blocks(self) -> int:
        return self.arena.offset_blocks

    def next_version(self, committed_blocks: int) -> "HostKVExtent":
        _non_negative(committed_blocks, "committed_blocks")
        if committed_blocks > self.capacity_blocks:
            raise HostKVCapacityError("committed_blocks exceeds capacity_blocks")
        return HostKVExtent(
            request_slot=self.request_slot,
            request_epoch=self.request_epoch,
            host_slot=self.host_slot,
            host_slot_generation=self.host_slot_generation,
            writer_lease_generation=self.writer_lease_generation,
            arena=self.arena,
            capacity_blocks=self.capacity_blocks,
            committed_blocks=committed_blocks,
            kv_version=KV_VERSION.next(self.kv_version),
        )


@dataclass(frozen=True, slots=True)
class HostKVWriteLease:
    """Numeric writer lease for one dirty HostKV range."""

    request_slot: int
    request_epoch: int
    host_slot: int
    host_slot_generation: int
    writer_lease_generation: int
    expected_version: int
    dirty_begin_block: int
    dirty_block_count: int

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        HOST_SLOT.validate(self.host_slot)
        HOST_SLOT_GENERATION.validate(self.host_slot_generation)
        WRITER_LEASE_GENERATION.validate(self.writer_lease_generation)
        KV_VERSION.validate(self.expected_version)
        _non_negative(self.dirty_begin_block, "dirty_begin_block")
        _non_negative(self.dirty_block_count, "dirty_block_count")

    @classmethod
    def for_extent(
        cls,
        extent: HostKVExtent,
        *,
        dirty_begin_block: int,
        dirty_block_count: int,
    ) -> "HostKVWriteLease":
        return cls(
            request_slot=extent.request_slot,
            request_epoch=extent.request_epoch,
            host_slot=extent.host_slot,
            host_slot_generation=extent.host_slot_generation,
            writer_lease_generation=extent.writer_lease_generation,
            expected_version=extent.kv_version,
            dirty_begin_block=dirty_begin_block,
            dirty_block_count=dirty_block_count,
        )


@dataclass(frozen=True, slots=True)
class SharedHostKVArenaDescriptor:
    arena_id: str
    shm_name: str
    total_bytes: int
    block_bytes: int
    total_blocks: int
    k_plane_offset: int
    v_plane_offset: int
    plane_bytes: int
    dtype: str
    kv_block_shape: tuple[int, ...]
    descriptor_generation: int

    def __post_init__(self) -> None:
        if not str(self.arena_id) or not str(self.shm_name) or not str(self.dtype):
            raise ValueError("arena_id, shm_name, and dtype must be non-empty")
        object.__setattr__(self, "arena_id", str(self.arena_id))
        object.__setattr__(self, "shm_name", str(self.shm_name))
        object.__setattr__(self, "dtype", str(self.dtype))
        shape = tuple(_positive(dim, "kv_block_shape dimension") for dim in self.kv_block_shape)
        if any(dim <= 0 for dim in shape):
            raise ValueError("kv_block_shape dimensions must be positive")
        object.__setattr__(self, "kv_block_shape", shape)
        for name in ("total_bytes", "block_bytes", "total_blocks", "plane_bytes", "descriptor_generation"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("k_plane_offset", "v_plane_offset"):
            object.__setattr__(self, name, _non_negative(getattr(self, name), name))
        if self.k_plane_offset != 0:
            raise ValueError("K plane must start at offset 0")
        if self.plane_bytes != self.total_blocks * self.block_bytes:
            raise ValueError("plane_bytes must equal total_blocks * block_bytes")
        if self.v_plane_offset != self.plane_bytes:
            raise ValueError("V plane must start immediately after K plane")
        if self.v_plane_offset + self.plane_bytes != self.total_bytes:
            raise ValueError("V plane must fit exactly in total_bytes")


class HostKVSlotView:
    """Memory views for the K and V planes of one HostKV extent."""

    def __init__(self, *, k_view: memoryview, v_view: memoryview, begin_block: int, block_count: int) -> None:
        self.k = k_view
        self.v = v_view
        self.begin_block = int(begin_block)
        self.block_count = int(block_count)

    @property
    def nbytes_per_plane(self) -> int:
        return len(self.k)

    def release(self) -> None:
        self.k.release()
        self.v.release()


class SharedHostKVArena:
    """Payload-plane arena; scheduler-visible metadata stays numeric."""

    def __init__(self, descriptor: SharedHostKVArenaDescriptor, shm: shared_memory.SharedMemory, *, owner: bool) -> None:
        self.descriptor = descriptor
        self._shm = shm
        self._owner = bool(owner)
        self.numa_placement = None
        self._closed = False
        # macOS rounds POSIX shared-memory mappings up to its page size. The
        # descriptor defines usable payload bytes; padding is never exposed.
        if len(self._shm.buf) < descriptor.total_bytes:
            raise ValueError("shared memory size does not match descriptor")

    @classmethod
    def create(
        cls,
        *,
        total_blocks: int,
        block_bytes: int,
        dtype: str = "uint8",
        kv_block_shape: tuple[int, ...] = (),
        arena_id: str | None = None,
        descriptor_generation: int = 1,
        numa_policy: str = "default",
        numa_nodes: tuple[int, ...] = (),
    ) -> "SharedHostKVArena":
        total_blocks = _positive(total_blocks, "total_blocks")
        block_bytes = _positive(block_bytes, "block_bytes")
        from .numa import validate_policy, place_new_mapping
        validate_policy(numa_policy, numa_nodes)
        plane_bytes = total_blocks * block_bytes
        shm = shared_memory.SharedMemory(create=True, size=plane_bytes * 2)
        try:
            placement = place_new_mapping(shm, plane_bytes * 2, numa_policy, numa_nodes)
            descriptor = SharedHostKVArenaDescriptor(
                arena_id=arena_id or f"nebulasd_hostkv_{uuid.uuid4().hex}",
                shm_name=shm.name,
                total_bytes=plane_bytes * 2,
                block_bytes=block_bytes,
                total_blocks=total_blocks,
                k_plane_offset=0,
                v_plane_offset=plane_bytes,
                plane_bytes=plane_bytes,
                dtype=dtype,
                kv_block_shape=tuple(_positive(dim, "kv_block_shape dimension") for dim in kv_block_shape),
                descriptor_generation=descriptor_generation,
            )
            arena = cls(descriptor, shm, owner=True)
            arena.numa_placement = placement
            return arena
        except Exception:
            shm.close()
            shm.unlink()
            raise

    @classmethod
    def attach(cls, descriptor: SharedHostKVArenaDescriptor) -> "SharedHostKVArena":
        shm = shared_memory.SharedMemory(name=descriptor.shm_name, create=False)
        try:
            return cls(descriptor, shm, owner=False)
        except Exception:
            shm.close()
            raise

    def make_extent(
        self,
        *,
        request_slot: int,
        request_epoch: int,
        host_slot: int,
        host_slot_generation: int,
        writer_lease_generation: int,
        offset_blocks: int,
        capacity_blocks: int,
    ) -> HostKVExtent:
        extent = HostKVExtent(
            request_slot=request_slot,
            request_epoch=request_epoch,
            host_slot=host_slot,
            host_slot_generation=host_slot_generation,
            writer_lease_generation=writer_lease_generation,
            arena=HostKVArenaHandle(offset_blocks, capacity_blocks, self.descriptor.descriptor_generation),
            capacity_blocks=capacity_blocks,
        )
        self._validate_extent(extent)
        return extent

    def view(self, extent: HostKVExtent, *, begin_block: int = 0, block_count: int | None = None) -> HostKVSlotView:
        self._validate_extent(extent)
        begin = _non_negative(begin_block, "begin_block")
        count = extent.capacity_blocks if block_count is None else _non_negative(block_count, "block_count")
        if begin + count > extent.capacity_blocks:
            raise HostKVCapacityError("HostKV view range exceeds extent capacity")
        return self._slot_view(extent, begin_block=begin, block_count=count, readonly=True)

    def writer_view(self, extent: HostKVExtent, *, lease: HostKVWriteLease) -> HostKVSlotView:
        self._validate_write_lease(extent, lease)
        return self._slot_view(
            extent,
            begin_block=lease.dirty_begin_block,
            block_count=lease.dirty_block_count,
            readonly=False,
        )

    def write_kv(self, extent: HostKVExtent, *, lease: HostKVWriteLease, k_payload: bytes, v_payload: bytes) -> HostKVExtent:
        if len(k_payload) != len(v_payload):
            raise ValueError("K and V payloads must have equal byte length")
        expected = lease.dirty_block_count * self.descriptor.block_bytes
        if len(k_payload) != expected:
            raise ValueError("payload size must match lease dirty range")
        self._validate_write_lease(extent, lease)
        if lease.dirty_block_count == 0:
            return extent
        view = self._slot_view(
            extent,
            begin_block=lease.dirty_begin_block,
            block_count=lease.dirty_block_count,
            readonly=False,
        )
        try:
            view.k[:] = k_payload
            view.v[:] = v_payload
        finally:
            view.release()
        committed = max(extent.committed_blocks, lease.dirty_begin_block + lease.dirty_block_count)
        return extent.next_version(committed)

    def read_kv(self, extent: HostKVExtent, *, begin_block: int, block_count: int) -> tuple[bytes, bytes]:
        view = self.view(extent, begin_block=begin_block, block_count=block_count)
        try:
            return bytes(view.k), bytes(view.v)
        finally:
            view.release()

    def address(self) -> int:
        return ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))

    def close(self) -> None:
        if self._closed:
            return
        self._shm.close()
        self._closed = True

    def unlink(self) -> None:
        if self._owner:
            self._shm.unlink()
            self._owner = False

    def _slot_view(
        self,
        extent: HostKVExtent,
        *,
        begin_block: int,
        block_count: int,
        readonly: bool,
    ) -> HostKVSlotView:
        rel_begin = (extent.offset_blocks + begin_block) * self.descriptor.block_bytes
        nbytes = block_count * self.descriptor.block_bytes
        k_begin = self.descriptor.k_plane_offset + rel_begin
        v_begin = self.descriptor.v_plane_offset + rel_begin
        k_view = self._shm.buf[k_begin : k_begin + nbytes]
        v_view = self._shm.buf[v_begin : v_begin + nbytes]
        if readonly:
            k_view = k_view.toreadonly()
            v_view = v_view.toreadonly()
        return HostKVSlotView(k_view=k_view, v_view=v_view, begin_block=begin_block, block_count=block_count)

    def _validate_write_lease(self, extent: HostKVExtent, lease: HostKVWriteLease) -> None:
        self._validate_extent(extent)
        if (
            lease.request_slot != extent.request_slot
            or lease.request_epoch != extent.request_epoch
            or lease.host_slot != extent.host_slot
            or lease.host_slot_generation != extent.host_slot_generation
            or lease.writer_lease_generation != extent.writer_lease_generation
        ):
            raise ValueError("HostKV write lease does not match extent identity")
        if lease.expected_version != extent.kv_version:
            raise ValueError("HostKV write lease expected_version does not match extent version")
        if lease.dirty_begin_block > extent.committed_blocks:
            raise HostKVCapacityError("HostKV write lease cannot skip over uncommitted blocks")
        if lease.dirty_begin_block + lease.dirty_block_count > extent.capacity_blocks:
            raise HostKVCapacityError("HostKV write lease dirty range exceeds extent capacity")

    def _validate_extent(self, extent: HostKVExtent) -> None:
        if extent.arena.generation != self.descriptor.descriptor_generation:
            raise ValueError("HostKV extent generation does not match arena descriptor")
        if extent.arena.block_count != extent.capacity_blocks:
            raise HostKVCapacityError("HostKV handle block count must match extent capacity")
        if not extent.arena.within_blocks(self.descriptor.total_blocks):
            raise HostKVCapacityError("HostKV extent exceeds arena capacity")


def _non_negative(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive(value: int, name: str) -> int:
    value = _non_negative(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
