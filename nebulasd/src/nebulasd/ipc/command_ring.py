"""SPSC command ring for fixed-width command headers."""

from __future__ import annotations

from dataclasses import dataclass

from nebulasd.core.ids import COMMAND_SEQ, WORKER_GENERATION

from .command_arena import CommandArena
from .command_arena import CommandBackpressure
from .protocol import CommandHeader, HotCommand, decode_command_payload


@dataclass(frozen=True, slots=True)
class CommandEnvelope:
    header: CommandHeader
    payload: bytes

    def decode(self, *, worker_id: int) -> HotCommand:
        return decode_command_payload(
            self.header.command_kind,
            self.payload,
            worker_id=worker_id,
            worker_generation=self.header.worker_generation,
            command_seq=self.header.command_seq,
        )


class StaleWorkerGeneration(RuntimeError):
    pass


class CommandRing:
    """Single-producer/single-consumer ring storing command headers only."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an int")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._entries: list[CommandHeader | None] = [None] * capacity
        self._head = 0
        self._tail = 0
        self._size = 0
        self._last_produced: int | None = None
        self._last_consumed: int | None = None
        self._producer_closed = False

    @property
    def capacity(self) -> int:
        return len(self._entries)

    def can_push(self) -> bool:
        return self._size < self.capacity

    def publish(self, header: CommandHeader) -> None:
        if not isinstance(header, CommandHeader):
            raise TypeError("header must be a CommandHeader")
        if self._producer_closed:
            raise RuntimeError("command ring producer is closed")
        if self._size == self.capacity:
            raise CommandBackpressure("command ring is full")
        if self._last_produced is not None and not COMMAND_SEQ.is_newer(header.command_seq, self._last_produced):
            raise ValueError("command_seq must increase monotonically")

        self._entries[self._tail] = header
        self._tail = (self._tail + 1) % self.capacity
        self._size += 1
        self._last_produced = header.command_seq

    def consume(self, *, expected_worker_generation: int, arena: CommandArena) -> CommandEnvelope | None:
        WORKER_GENERATION.validate(expected_worker_generation)
        if self._size == 0:
            return None

        header = self._entries[self._head]
        if header is None:
            raise RuntimeError("command ring corruption")
        if header.worker_generation != expected_worker_generation:
            raise StaleWorkerGeneration("command worker_generation does not match consumer")
        if self._last_consumed is not None and not COMMAND_SEQ.is_newer(header.command_seq, self._last_consumed):
            raise ValueError("consumed command_seq must increase monotonically")

        self._entries[self._head] = None
        self._head = (self._head + 1) % self.capacity
        self._size -= 1
        self._last_consumed = header.command_seq
        payload = arena.read(header_payload_handle(header, arena_generation=arena.generation))
        arena.release_through(header.command_seq)
        return CommandEnvelope(header=header, payload=payload)

    def is_empty(self) -> bool:
        return self._size == 0

    def is_full(self) -> bool:
        return self._size == self.capacity

    def close_producer(self) -> None:
        self._producer_closed = True

    def producer_closed(self) -> bool:
        return self._producer_closed


def header_payload_handle(header: CommandHeader, *, arena_generation: int = 0):
    from nebulasd.core.handles import ArenaHandle

    return ArenaHandle(offset=header.payload_offset, length=header.payload_length, generation=arena_generation)
