"""Fixed-width token payload arena."""

from __future__ import annotations

from dataclasses import dataclass
from struct import unpack_from

from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import ARENA_GENERATION, ARENA_LENGTH, ARENA_OFFSET, U32


class TokenArenaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TokenArenaSnapshot:
    capacity_bytes: int
    generation: int
    bytes_used: int


class TokenArena:
    """Append-only u32 token arena used by worker input/output payloads."""

    def __init__(self, capacity_bytes: int, *, generation: int = 1) -> None:
        U32.validate(capacity_bytes)
        ARENA_GENERATION.validate(generation)
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self._bytes = bytearray(capacity_bytes)
        self._generation = generation
        self._head = 0

    @property
    def capacity_bytes(self) -> int:
        return len(self._bytes)

    @property
    def generation(self) -> int:
        return self._generation

    def write_tokens(self, token_ids: tuple[int, ...]) -> ArenaHandle:
        tokens = tuple(_token(token) for token in token_ids)
        length = len(tokens) * 4
        ARENA_LENGTH.validate(length)
        if self._head + length > self.capacity_bytes:
            raise TokenArenaError("token arena capacity exhausted")
        offset = self._head
        for index, token in enumerate(tokens):
            start = offset + index * 4
            self._bytes[start : start + 4] = token.to_bytes(4, "little")
        self._head += length
        return ArenaHandle(offset=offset, length=length, generation=self._generation)

    def write_token_batches(self, batches: tuple[tuple[int, ...], ...]) -> tuple[ArenaHandle, ...]:
        normalized = tuple(tuple(_token(token) for token in batch) for batch in batches)
        lengths = tuple(len(batch) * 4 for batch in normalized)
        for length in lengths:
            ARENA_LENGTH.validate(length)
        total = sum(lengths)
        if self._head + total > self.capacity_bytes:
            raise TokenArenaError("token arena capacity exhausted")
        handles: list[ArenaHandle] = []
        for tokens, length in zip(normalized, lengths, strict=True):
            offset = self._head
            for index, token in enumerate(tokens):
                start = offset + index * 4
                self._bytes[start : start + 4] = token.to_bytes(4, "little")
            self._head += length
            handles.append(ArenaHandle(offset=offset, length=length, generation=self._generation))
        return tuple(handles)

    def read_tokens(self, handle: ArenaHandle) -> tuple[int, ...]:
        self._validate_handle(handle)
        if handle.length % 4 != 0:
            raise TokenArenaError("token arena handle length must be a multiple of 4")
        return unpack_from(f"<{handle.length // 4}I", self._bytes, handle.offset)

    def snapshot(self) -> TokenArenaSnapshot:
        return TokenArenaSnapshot(self.capacity_bytes, self._generation, self._head)

    def reset_quiescent(self) -> None:
        self._head = 0
        self._generation = ARENA_GENERATION.next(self._generation)

    def _validate_handle(self, handle: ArenaHandle) -> None:
        if not isinstance(handle, ArenaHandle):
            raise TypeError("handle must be ArenaHandle")
        if handle.generation != self._generation:
            raise TokenArenaError("stale token arena generation")
        if not handle.within(self.capacity_bytes):
            raise TokenArenaError("token arena handle outside capacity")


def _token(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("token id must be an int, not bool")
    if not isinstance(value, int):
        raise TypeError("token id must be an int")
    U32.validate(value)
    return value
