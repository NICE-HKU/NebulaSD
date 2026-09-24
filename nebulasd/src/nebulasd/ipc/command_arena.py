"""Fixed-capacity command payload arena."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from nebulasd.core.errors import ControlPlaneError, ErrorSeverity, ResultCode
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import COMMAND_SEQ, U32


class CommandBackpressure(ControlPlaneError):
    def __init__(self, message: str) -> None:
        super().__init__(ResultCode.BACKPRESSURE, message, severity=ErrorSeverity.RETRYABLE)


@dataclass(frozen=True, slots=True)
class _Allocation:
    command_seq: int
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


class CommandArena:
    """Byte arena reclaimed by monotonically consumed command sequence."""

    def __init__(self, capacity_bytes: int, *, generation: int = 0) -> None:
        U32.validate(capacity_bytes)
        U32.validate(generation)
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self._bytes = bytearray(capacity_bytes)
        self._generation = generation
        self._head = 0
        self._tail = 0
        self._allocations: deque[_Allocation] = deque()

    @property
    def capacity_bytes(self) -> int:
        return len(self._bytes)

    @property
    def generation(self) -> int:
        return self._generation

    def allocate(self, *, command_seq: int, payload: bytes) -> ArenaHandle:
        COMMAND_SEQ.validate(command_seq)
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        length = len(payload)
        U32.validate(length)
        if length <= 0:
            raise ValueError("payload must be non-empty")
        if length > self.capacity_bytes:
            raise CommandBackpressure("command payload is larger than arena capacity")
        if self._allocations and not COMMAND_SEQ.is_newer(command_seq, self._allocations[-1].command_seq):
            raise ValueError("command arena allocation sequence must increase monotonically")

        offset = self._reserve(length)
        self._bytes[offset : offset + length] = payload
        self._allocations.append(_Allocation(command_seq, offset, length))
        return ArenaHandle(offset=offset, length=length, generation=self._generation)

    def read(self, handle: ArenaHandle) -> bytes:
        if handle.generation != self._generation:
            raise ValueError("stale command arena generation")
        if not handle.within(self.capacity_bytes):
            raise ValueError("command arena handle outside capacity")
        return bytes(self._bytes[handle.offset : handle.offset + handle.length])

    def release_through(self, consumer_seq: int) -> None:
        COMMAND_SEQ.validate(consumer_seq)
        while self._allocations:
            allocation = self._allocations[0]
            if allocation.command_seq != consumer_seq and not COMMAND_SEQ.is_newer_or_equal(consumer_seq, allocation.command_seq):
                break
            allocation = self._allocations.popleft()
            self._tail = allocation.end % self.capacity_bytes
        if not self._allocations:
            self._head = 0
            self._tail = 0

    def discard_last(self, command_seq: int) -> None:
        COMMAND_SEQ.validate(command_seq)
        if not self._allocations or self._allocations[-1].command_seq != command_seq:
            raise ValueError("only the most recent allocation can be discarded")
        self._allocations.pop()
        self._head = self._allocations[-1].end % self.capacity_bytes if self._allocations else 0
        if not self._allocations:
            self._tail = 0

    def _reserve(self, length: int) -> int:
        if not self._allocations:
            self._head = 0
            self._tail = 0
            self._head = length % self.capacity_bytes
            return 0
        if self._head == self._tail:
            raise CommandBackpressure("command arena has insufficient free space")

        if self._head >= self._tail:
            if length <= self.capacity_bytes - self._head:
                offset = self._head
                self._head = (self._head + length) % self.capacity_bytes
                return offset
            if length <= self._tail:
                self._head = length
                return 0
        else:
            if length <= self._tail - self._head:
                offset = self._head
                self._head += length
                return offset
        raise CommandBackpressure("command arena has insufficient free space")
