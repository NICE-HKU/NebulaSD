"""Validate HostKV source/allocation facts before touching payload memory."""

from __future__ import annotations

from dataclasses import replace

from nebulasd.core.enums import D2HStatus, TargetStatus, StateChangeBlockKind as Kind
from nebulasd.core.errors import ResultCode
from nebulasd.core.ids import KV_VERSION, U64
from nebulasd.table.storage import RequestSchedulingTable
from .arena import HostKVExtent
from .host_allocator import HostKVAllocator


class HostKVFacts:
    def __init__(self, allocator: HostKVAllocator, table: RequestSchedulingTable) -> None:
        self.allocator = allocator
        self.table = table
        self._shared_allocation = (getattr(allocator, "table", None) is table
            and getattr(allocator, "allocation_kind", None) == Kind.REQUEST_HOSTKV)

    def allocation(self, slot: int, epoch: int) -> HostKVExtent:
        extent = self.allocator.require(slot, epoch)
        if not self._shared_allocation:
            row = self.table.partition(Kind.REQUEST_HOSTKV).read_stable(slot, field_names=("request_epoch", "host_slot", "host_slot_generation",
                "writer_lease_generation", "offset_blocks", "capacity_blocks"))
            expected = dict(request_epoch=epoch, host_slot=extent.host_slot,
                            host_slot_generation=extent.host_slot_generation,
                            writer_lease_generation=extent.writer_lease_generation,
                            offset_blocks=extent.offset_blocks, capacity_blocks=extent.capacity_blocks)
            for name, value in expected.items():
                if row.get(name) != value:
                    raise ValueError(f"HostKV allocation {name} mismatch")
        engine = self.table.partition(Kind.REQUEST_ENGINE).read_stable(slot, field_names=("request_epoch",))
        if engine.get("request_epoch") != epoch:
            raise ValueError("HostKV engine request_epoch mismatch")
        return extent

    def prepare_source(self, request) -> HostKVExtent:
        extent = self.allocation(request.request_slot, request.request_epoch)
        target = self.table.partition(Kind.REQUEST_TARGET_COMPUTE).read_stable(request.request_slot,
            field_names=("request_epoch", "round_id", "status", "target_kv_version"))
        if (target.get("request_epoch") != request.request_epoch
                or target.get("round_id") + 1 != request.round_id
                or target.get("status") != int(TargetStatus.READY_DRAFT)
                or target.get("target_kv_version") != request.source_host_version):
            raise ValueError("prepare source does not match completed Target round/version")
        for name, value in (
            ("host_slot", extent.host_slot),
            ("host_slot_generation", extent.host_slot_generation),
            ("host_writer_lease_generation", extent.writer_lease_generation),
            ("hostkv_handle", extent.arena),
        ):
            if getattr(request, name) != value:
                raise ValueError(f"prepare {name} does not match HostKV allocation")
        return extent

    def source(self, request) -> HostKVExtent | None:
        return self.source_ready(request, self.prepare_source(request))

    def source_ready(self, request, extent):
        """Mutable version check; caller holds the read pin for submission."""
        ready = self._ready(extent)
        if ready is None:
            return None
        if ready.get("ready_version") != request.source_host_version:
            if KV_VERSION.is_newer(ready.get("ready_version"), request.source_host_version):
                raise ValueError("prepare refers to an overwritten HostKV version")
            return None
        if ready.get("committed_blocks") != request.committed_blocks:
            raise ValueError("HostKV committed blocks mismatch")
        if ready.get("logical_kv_len") != request.logical_kv_len:
            raise ValueError("HostKV logical length mismatch")
        if request.valid_blocks != request.committed_blocks:
            raise ValueError("full-prefix H2D requires all valid blocks committed in HostKV")
        return extent

    def write_extent(self, dirty, *, block_size: int, allocation=None) -> HostKVExtent:
        extent = allocation if allocation is not None else self.allocation(dirty.request_slot, dirty.request_epoch)
        ready = self._ready(extent)
        version = 0 if ready is None else ready.get("ready_version")
        committed = 0 if ready is None else ready.get("committed_blocks")
        KV_VERSION.validate(dirty.target_kv_version)
        # Zero is a valid first backend version; only an existing committed
        # prefix supplies a version against which strict advancement is defined.
        if ready is not None and not KV_VERSION.is_newer(dirty.target_kv_version, version):
            raise ValueError("D2H KV version must advance")
        valid = (dirty.logical_kv_len + block_size - 1) // block_size
        if (dirty.dirty_begin_block != max(committed - 1, 0)
                or dirty.dirty_begin_block + dirty.dirty_block_count != valid
                or valid > extent.capacity_blocks):
            raise ValueError("D2H dirty range must cover the old tail and full new prefix")
        return replace(extent, committed_blocks=committed, kv_version=version)

    def _ready(self, extent: HostKVExtent):
        partition = self.table.partition(Kind.REQUEST_D2H)
        if partition.read_publish_seq(extent.request_slot) == U64.invalid:
            return None
        ready = partition.read_stable(extent.request_slot, field_names=("request_epoch", "status",
            "result_code", "host_slot_generation", "writer_version", "ready_version",
            "committed_blocks", "logical_kv_len"))
        if ready.get("request_epoch") != extent.request_epoch:
            return None
        if ready.get("status") != int(D2HStatus.HOST_READY):
            return None
        if (ready.get("result_code") != int(ResultCode.OK)
                or ready.get("host_slot_generation") != extent.host_slot_generation
                or ready.get("writer_version") != extent.writer_lease_generation):
            raise ValueError("HostKV completion fence mismatch")
        return ready
