"""Fixed ABI commands with bounded slot-aligned payloads and consume-after-copy ACK."""

from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import COMMAND_SEQ
from .command_arena import CommandBackpressure
from .command_ring import CommandEnvelope, StaleWorkerGeneration
from .mapped_segment import MappedSegment
from .native_ring import NativeRing
from .protocol import CommandHeader


class NativeCommandRing(NativeRing):
    def __init__(self, capacity, descriptor=None, doorbell=None):
        super().__init__(capacity, CommandHeader.byte_size, descriptor)
        self.doorbell = doorbell
        self._last_produced = self._last_consumed = None

    def publish(self, header):
        if self.producer_closed():
            raise RuntimeError("command producer closed")
        if self._last_produced is not None and not COMMAND_SEQ.is_newer(header.command_seq, self._last_produced):
            raise ValueError("command sequence must advance")
        if not super().push(header.to_bytes()):
            raise CommandBackpressure("command ring full")
        self._last_produced = header.command_seq
        if self.doorbell is not None:
            self.doorbell.ring()

    def consume(self, *, expected_worker_generation, arena):
        raw = self.peek()
        if raw is None:
            return None
        header = CommandHeader.from_bytes(raw)
        if header.worker_generation != expected_worker_generation:
            raise StaleWorkerGeneration("stale native command generation")
        if self._last_consumed is not None and not COMMAND_SEQ.is_newer(header.command_seq, self._last_consumed):
            raise ValueError("consumed command sequence must advance")
        expected_offset = (self.head() % self.capacity) * arena.slot_bytes
        if header.payload_offset != expected_offset or header.payload_length > arena.slot_bytes:
            raise ValueError("command payload does not belong to consumer slot")
        payload = arena.read(ArenaHandle(header.payload_offset, header.payload_length, arena.generation))
        # ACK after copying payload, never after merely reading the header.
        self.ack()
        self._last_consumed = header.command_seq
        return CommandEnvelope(header, payload)

    def is_empty(self):
        return self.head() == self.tail()

    def close_producer(self):
        self.native.sd_store(self.address + 128, 1)
        if self.doorbell:
            self.doorbell.ring()

    def producer_closed(self):
        return bool(self.native.sd_load(self.address + 128))


class NativeCommandArena:
    """One payload slot per ring entry; no wrap padding or unbounded allocations."""

    def __init__(self, ring, slot_bytes=65536, descriptor=None, generation=1):
        if slot_bytes <= 0:
            raise ValueError("slot_bytes must be positive")
        self.ring, self.slot_bytes, self.generation = ring, slot_bytes, generation
        schema = f"command-payload:{ring.capacity}:{slot_bytes}:{generation}"
        self.segment = (MappedSegment.create(ring.capacity * slot_bytes, schema) if descriptor is None
                        else MappedSegment(descriptor))
        if self.segment.descriptor.schema != schema:
            self.segment.close()
            raise ValueError("command arena layout mismatch")
        self._pending = None

    def allocate(self, *, command_seq, payload):
        COMMAND_SEQ.validate(command_seq)
        if not payload or len(payload) > self.slot_bytes:
            raise ValueError("command payload exceeds fixed slot size")
        if not self.ring.can_push():
            raise CommandBackpressure("native command ring capacity exhausted")
        offset = self.ring.tail() % self.ring.capacity * self.slot_bytes
        self.segment.buffer[offset:offset + len(payload)] = payload
        self._pending = (command_seq, self.ring.tail())
        return ArenaHandle(offset, len(payload), self.generation)

    def discard_last(self, command_seq):
        if self._pending != (command_seq, self.ring.tail()):
            raise ValueError("cannot discard published command payload")
        self._pending = None

    def read(self, handle):
        if handle.generation != self.generation or not handle.within(len(self.segment.buffer)):
            raise ValueError("invalid native payload handle")
        return bytes(self.segment.buffer[handle.offset:handle.offset + handle.length])

    def close(self):
        self.segment.close()
