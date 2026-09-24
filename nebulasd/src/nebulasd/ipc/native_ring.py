"""SPSC notification rings: one producer process per ring, one Engine consumer."""

import ctypes as C
from struct import Struct

from .mapped_segment import MappedSegment
from .native import library
from .state_change_ring import StateChangeEntry, StateChangeBatch, StateChangePublishResult


class NativeRing:
    def __init__(self, capacity, width, descriptor=None):
        if capacity <= 0 or capacity & (capacity - 1) or width <= 0:
            raise ValueError("ring capacity must be a power of two and width positive")
        self.native = library()
        self.capacity, self.width = capacity, width
        schema = f"spsc:{capacity}:{width}"
        self.segment = (MappedSegment.create(192 + capacity * width, schema) if descriptor is None
                        else MappedSegment(descriptor))
        if self.segment.descriptor.schema != schema:
            self.segment.close()
            raise ValueError("ring layout mismatch")
        self.address = self.segment.address

    def head(self):
        return self.native.sd_load(self.address)

    def tail(self):
        return self.native.sd_load(self.address + 64)

    def can_push(self):
        return (self.tail() - self.head()) % (1 << 64) < self.capacity

    def push(self, raw):
        if len(raw) != self.width:
            raise ValueError("ring record width mismatch")
        return bool(self.native.sd_ring_push(self.address, self.capacity, self.width, raw))

    def peek(self):
        raw = C.create_string_buffer(self.width)
        if self.native.sd_ring_peek(self.address, self.capacity, self.width, raw):
            return raw.raw
        return None

    def ack(self):
        self.native.sd_ring_ack(self.address)

    def close(self):
        self.segment.close()


class NativeStateChangeRing(NativeRing):
    RECORD = Struct("<IIQ")

    def __init__(self, capacity, descriptor=None, doorbell=None):
        super().__init__(capacity, self.RECORD.size, descriptor)
        self.doorbell = doorbell
        self._drain_buffer = C.create_string_buffer(capacity * self.width)

    def push(self, entry):
        accepted = super().push(self.RECORD.pack(int(entry.block_kind), entry.row, entry.publish_seq))
        if not accepted:
            self.native.sd_store(self.address + 128, 1)
        if self.doorbell is not None and not self.native.sd_exchange(self.address + 136, 1):
            self.doorbell.ring()
        return StateChangePublishResult(accepted, not accepted)

    def drain(self, max_entries=None):
        if max_entries is not None and max_entries < 0:
            raise ValueError("negative drain limit")
        self.native.sd_store(self.address + 136, 0)
        # Clear overflow before draining: a racing overflow remains for next poll.
        overflow = bool(self.native.sd_exchange(self.address + 128, 0))
        limit = self.capacity if max_entries is None else min(max_entries, self.capacity)
        count = self.native.sd_ring_drain(self.address, self.capacity, self.width,
                                          limit, self._drain_buffer)
        entries = tuple(StateChangeEntry(*self.RECORD.unpack_from(self._drain_buffer, i*self.width))
                        for i in range(count))
        return StateChangeBatch(entries, overflow)

    def has_pending(self):
        return self.head() != self.tail() or bool(self.native.sd_load(self.address + 128))
