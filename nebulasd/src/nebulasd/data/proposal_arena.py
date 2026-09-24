"""Proposal payload arena."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from struct import Struct
from typing import Sequence

from nebulasd.core.enums import ProposalKind, validate_enum
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import ARENA_GENERATION, U32


_HEADER = Struct("<II")


class ProposalArenaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProposalPayload:
    kind: ProposalKind
    draft_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", validate_enum(ProposalKind, self.kind))
        tokens = tuple(_token(token) for token in self.draft_token_ids)
        object.__setattr__(self, "draft_token_ids", tokens)


@dataclass(frozen=True, slots=True)
class _Allocation:
    offset: int
    length: int
    generation: int

    @property
    def end(self) -> int:
        return self.offset + self.length


class ProposalArena:
    """FIFO proposal arena for linear Draft proposals."""

    def __init__(self, capacity_bytes: int, *, generation: int = 1) -> None:
        U32.validate(capacity_bytes)
        ARENA_GENERATION.validate(generation)
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self._bytes = bytearray(capacity_bytes)
        self._generation = generation
        self._head = 0
        self._tail = 0
        self._allocations: deque[_Allocation] = deque()
        self._live_handles: set[ArenaHandle] = set()

    @property
    def capacity_bytes(self) -> int:
        return len(self._bytes)

    @property
    def generation(self) -> int:
        return self._generation

    def write_proposal(self, payload: ProposalPayload) -> ArenaHandle:
        return self.write_proposals((payload,))[0]

    def write_proposals(self, payloads: Sequence[ProposalPayload]) -> tuple[ArenaHandle, ...]:
        payloads = tuple(payloads)
        if not payloads:
            return ()
        raws = tuple(_encode_payload(payload) for payload in payloads)
        for raw in raws:
            if len(raw) > self.capacity_bytes:
                raise ProposalArenaError("proposal payload is larger than arena capacity")
        offsets = self._reserve_many(tuple(len(raw) for raw in raws))
        handles = []
        for offset, raw in zip(offsets, raws, strict=True):
            handle = ArenaHandle(offset=offset, length=len(raw), generation=self._generation)
            self._bytes[offset : offset + len(raw)] = raw
            self._allocations.append(_Allocation(offset, len(raw), self._generation))
            self._live_handles.add(handle)
            handles.append(handle)
        return tuple(handles)

    def release(self, handle: ArenaHandle) -> None:
        self.release_many((handle,))

    def release_many(self, handles: Sequence[ArenaHandle]) -> None:
        handles = tuple(handles)
        if not handles:
            return
        if len(handles) > len(self._allocations):
            raise ProposalArenaError("proposal arena has fewer live allocations than requested releases")
        for handle, allocation in zip(handles, self._allocations, strict=False):
            self._validate_handle(handle)
            if not _matches(handle, allocation):
                raise ProposalArenaError("proposal arena releases must follow FIFO order")
        for handle in handles:
            allocation = self._allocations.popleft()
            self._live_handles.remove(handle)
            self._tail = allocation.end
        if not self._allocations:
            self._tail = self._head

    def reset_quiescent(self) -> None:
        if self._allocations:
            raise ProposalArenaError("cannot reset proposal arena with live allocations")
        self._head = 0
        self._tail = 0
        self._live_handles.clear()
        self._generation = ARENA_GENERATION.next(self._generation)

    def _reserve_many(self, lengths: tuple[int, ...]) -> tuple[int, ...]:
        for length in lengths:
            if length <= 0:
                raise ProposalArenaError("proposal payload must be non-empty")
        total = sum(lengths)
        if self._head + total > self.capacity_bytes:
            raise ProposalArenaError("proposal arena capacity exhausted; reset_quiescent is required before reuse")
        offset = self._head
        self._head += total
        return tuple(_prefix_offsets(offset, lengths))

    def _reserve(self, length: int) -> int:
        offset = self._reserve_many((length,))[0]
        return offset

    def read_proposal(self, handle: ArenaHandle) -> ProposalPayload:
        self._validate_handle(handle)
        self._validate_live_allocation(handle)
        if handle.length < _HEADER.size:
            raise ProposalArenaError("proposal handle is too short")
        start = handle.offset
        kind, count = _HEADER.unpack(self._bytes[start : start + _HEADER.size])
        expected = _HEADER.size + count * 4
        if handle.length != expected:
            raise ProposalArenaError("proposal handle length mismatch")
        tokens_start = start + _HEADER.size
        tokens = tuple(
            int.from_bytes(self._bytes[offset : offset + 4], "little")
            for offset in range(tokens_start, tokens_start + count * 4, 4)
        )
        return ProposalPayload(ProposalKind(kind), tokens)

    def _validate_handle(self, handle: ArenaHandle) -> None:
        if not isinstance(handle, ArenaHandle):
            raise TypeError("handle must be ArenaHandle")
        if handle.generation != self._generation:
            raise ProposalArenaError("stale proposal arena generation")
        if not handle.within(self.capacity_bytes):
            raise ProposalArenaError("proposal handle outside capacity")

    def _validate_live_allocation(self, handle: ArenaHandle) -> None:
        if handle not in self._live_handles:
            raise ProposalArenaError("proposal handle does not match a live allocation")


def _encode_payload(payload: ProposalPayload) -> bytes:
    if not isinstance(payload, ProposalPayload):
        raise TypeError("payload must be ProposalPayload")
    raw = bytearray(_HEADER.pack(int(payload.kind), len(payload.draft_token_ids)))
    for token in payload.draft_token_ids:
        raw += token.to_bytes(4, "little")
    return bytes(raw)


def _prefix_offsets(start: int, lengths: tuple[int, ...]) -> tuple[int, ...]:
    offsets = []
    offset = start
    for length in lengths:
        offsets.append(offset)
        offset += length
    return tuple(offsets)


def _matches(handle: ArenaHandle, allocation: _Allocation) -> bool:
    return (
        handle.offset == allocation.offset
        and handle.length == allocation.length
        and handle.generation == allocation.generation
    )


def _token(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("draft token id must be an int, not bool")
    if not isinstance(value, int):
        raise TypeError("draft token id must be an int")
    U32.validate(value)
    return value
