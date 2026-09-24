"""Process-shared HostKV pins. Allocation rows are Engine-owned and never recycled live."""

from nebulasd.core.enums import StateChangeBlockKind as Kind
from nebulasd.ipc.native import library

WRITE = 1 << 63


class SharedHostKVAllocator:
    def __init__(self, arena, table, pin_segment, *, allocation_kind=Kind.REQUEST_HOSTKV, on_release=None):
        self.arena, self.table, self.pins = arena, table, pin_segment
        if allocation_kind not in (Kind.REQUEST_HOSTKV, Kind.REQUEST_DRAFT_HOSTKV):
            raise ValueError("expected a HostKV allocation partition")
        self.allocation_kind = allocation_kind
        self.native = library()
        self.on_release = on_release

    def require(self, slot, epoch):
        row = self.table.partition(self.allocation_kind).read_stable(slot, field_names=("request_epoch", "host_slot", "host_slot_generation",
            "writer_lease_generation", "offset_blocks", "capacity_blocks"))
        if row.get("request_epoch") != epoch:
            raise ValueError("stale HostKV allocation")
        return self.arena.make_extent(request_slot=slot, request_epoch=epoch,
            host_slot=row.get("host_slot"), host_slot_generation=row.get("host_slot_generation"),
            writer_lease_generation=row.get("writer_lease_generation"),
            offset_blocks=row.get("offset_blocks"), capacity_blocks=row.get("capacity_blocks"))

    def _address(self, extent):
        if self.require(extent.request_slot, extent.request_epoch) != extent:
            raise ValueError("stale HostKV pin extent")
        return self.pins.address + extent.request_slot * 64

    def pin(self, extent, *, write):
        self._address(extent)
        return self.pin_prepared(extent, write=write)

    def pin_prepared(self, extent, *, write):
        # Internal worker path only: command/dirty acceptance checked allocation.
        # Engine's all-owner recycle ACK barrier forbids replacing it while a
        # pending prepare, dirty descriptor or Bank remains on this owner.
        # Payload versions are NOT frozen by that barrier: check them under the
        # acquired pin before DMA, and still use the atomic reader/writer CAS.
        address = self.pins.address + extent.request_slot * 64
        while True:
            current = self.native.sd_load(address)
            if current & WRITE or (write and current):
                return False
            if self.native.sd_cas(address, current, WRITE if write else current + 1):
                return True

    def unpin(self, extent, *, write):
        self._address(extent)
        self.release_pinned(extent, write=write)

    def release_pinned(self, extent, *, write):
        # A successfully held pin prevents allocation recycling. Its captured
        # slot remains valid until this release; re-decoding allocation here
        # merely lengthens every physical completion's critical section.
        address = self.pins.address + extent.request_slot * 64
        while True:
            current = self.native.sd_load(address)
            if (write and current != WRITE) or (not write and (current == 0 or current & WRITE)):
                raise ValueError("HostKV pin not held")
            if self.native.sd_cas(address, current, 0 if write else current - 1):
                # A waiting writer becomes eligible only after the last reader.
                # Publish the hint AFTER the successful release, not at READY.
                if self.on_release is not None and (write or current == 1):
                    self.on_release(extent)
                return
