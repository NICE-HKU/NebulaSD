"""CPU characterization tests for HostKV arena and registration helpers."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from nebulasd.kv.arena import HostKVCapacityError, HostKVWriteLease, SharedHostKVArena
from nebulasd.kv.host_registration import CudaHostRegistrationAdapter, HostRegistrationError, HostRegistrationRecord


class FakeCudaRuntime:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []

    def cudaHostRegister(self, address: int, nbytes: int, flags: int) -> int:
        self.registered.append((address, nbytes, flags))
        return 0

    def cudaHostUnregister(self, address: int) -> int:
        self.unregistered.append(address)
        return 0


def test_shared_hostkv_arena_extent_write_read_and_attach() -> None:
    arena = SharedHostKVArena.create(total_blocks=8, block_bytes=4, kv_block_shape=(1, 4))
    attached = None
    try:
        extent = arena.make_extent(
            request_slot=7,
            request_epoch=3,
            host_slot=2,
            host_slot_generation=9,
            writer_lease_generation=11,
            offset_blocks=2,
            capacity_blocks=3,
        )
        lease = HostKVWriteLease.for_extent(extent, dirty_begin_block=0, dirty_block_count=2)
        updated = arena.write_kv(extent, lease=lease, k_payload=b"abcdefgh", v_payload=b"ABCDEFGH")
        assert updated.kv_version == 1
        assert updated.committed_blocks == 2
        assert arena.read_kv(updated, begin_block=0, block_count=2) == (b"abcdefgh", b"ABCDEFGH")

        attached = SharedHostKVArena.attach(arena.descriptor)
        assert attached.read_kv(updated, begin_block=0, block_count=2) == (b"abcdefgh", b"ABCDEFGH")
    finally:
        if attached is not None:
            attached.close()
        arena.close()
        arena.unlink()


def test_hostkv_rejects_extent_and_lease_capacity_errors() -> None:
    arena = SharedHostKVArena.create(total_blocks=4, block_bytes=8)
    try:
        with pytest.raises(HostKVCapacityError):
            arena.make_extent(
                request_slot=1,
                request_epoch=1,
                host_slot=1,
                host_slot_generation=1,
                writer_lease_generation=1,
                offset_blocks=3,
                capacity_blocks=2,
            )
        extent = arena.make_extent(
            request_slot=1,
            request_epoch=1,
            host_slot=1,
            host_slot_generation=1,
            writer_lease_generation=1,
            offset_blocks=0,
            capacity_blocks=2,
        )
        lease = HostKVWriteLease.for_extent(extent, dirty_begin_block=1, dirty_block_count=2)
        with pytest.raises(HostKVCapacityError):
            arena.writer_view(extent, lease=lease)
    finally:
        arena.close()
        arena.unlink()


def test_hostkv_committed_prefix_cannot_skip_holes_and_zero_write_is_noop() -> None:
    arena = SharedHostKVArena.create(total_blocks=8, block_bytes=4)
    try:
        extent = arena.make_extent(
            request_slot=1,
            request_epoch=1,
            host_slot=1,
            host_slot_generation=1,
            writer_lease_generation=1,
            offset_blocks=0,
            capacity_blocks=8,
        )
        gap_lease = HostKVWriteLease.for_extent(extent, dirty_begin_block=5, dirty_block_count=1)
        with pytest.raises(HostKVCapacityError):
            arena.write_kv(extent, lease=gap_lease, k_payload=b"aaaa", v_payload=b"bbbb")

        zero_lease = HostKVWriteLease.for_extent(extent, dirty_begin_block=0, dirty_block_count=0)
        assert arena.write_kv(extent, lease=zero_lease, k_payload=b"", v_payload=b"") is extent

        stale_zero = HostKVWriteLease(
            request_slot=extent.request_slot,
            request_epoch=extent.request_epoch + 1,
            host_slot=extent.host_slot,
            host_slot_generation=extent.host_slot_generation,
            writer_lease_generation=extent.writer_lease_generation,
            expected_version=extent.kv_version,
            dirty_begin_block=0,
            dirty_block_count=0,
        )
        with pytest.raises(ValueError, match="does not match"):
            arena.write_kv(extent, lease=stale_zero, k_payload=b"", v_payload=b"")

        append = HostKVWriteLease.for_extent(extent, dirty_begin_block=0, dirty_block_count=2)
        updated = arena.write_kv(extent, lease=append, k_payload=b"abcdefgh", v_payload=b"ABCDEFGH")
        assert updated.committed_blocks == 2
        assert updated.kv_version == 1
    finally:
        arena.close()
        arena.unlink()


def test_hostkv_strictly_rejects_float_block_inputs() -> None:
    arena = SharedHostKVArena.create(total_blocks=4, block_bytes=8)
    try:
        with pytest.raises(TypeError):
            arena.make_extent(
                request_slot=1,
                request_epoch=1,
                host_slot=1,
                host_slot_generation=1,
                writer_lease_generation=1,
                offset_blocks=1.5,  # type: ignore[arg-type]
                capacity_blocks=1,
            )
    finally:
        arena.close()
        arena.unlink()


def test_registration_record_is_process_local_and_adapter_is_idempotent() -> None:
    cudart = FakeCudaRuntime()
    arena = SimpleNamespace(
        descriptor=SimpleNamespace(arena_id="arena", total_bytes=4096),
        address=lambda: 123456,
    )
    adapter = CudaHostRegistrationAdapter(cudart)
    first = adapter.register(arena, executor_id="target-0", process_generation=5)
    second = adapter.register(arena, executor_id="target-0", process_generation=5)
    assert first == second
    assert adapter.registration_count == 1
    assert cudart.registered == [(123456, 4096, 1)]

    with pytest.raises(TypeError):
        pickle.dumps(first)

    adapter.unregister(first)
    assert adapter.registration_count == 0
    assert cudart.unregistered == [123456]


def test_registration_rejects_duplicate_key_with_different_mapping() -> None:
    adapter = CudaHostRegistrationAdapter(FakeCudaRuntime())
    arena = SimpleNamespace(descriptor=SimpleNamespace(arena_id="arena", total_bytes=4096), address=lambda: 123456)
    moved = SimpleNamespace(descriptor=SimpleNamespace(arena_id="arena", total_bytes=4096), address=lambda: 654321)
    adapter.register(arena, executor_id="target-0", process_generation=5)
    with pytest.raises(HostRegistrationError):
        adapter.register(moved, executor_id="target-0", process_generation=5)


def test_registration_record_validates_numeric_fields() -> None:
    with pytest.raises(ValueError):
        HostRegistrationRecord("arena", "executor", -1, 1, 1)
    with pytest.raises(ValueError):
        HostRegistrationRecord("arena", "executor", 1, 0, 1)
