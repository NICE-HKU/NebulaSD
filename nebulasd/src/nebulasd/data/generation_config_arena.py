"""Generation configuration payload arena for Draft workers."""

from __future__ import annotations

from dataclasses import dataclass
from struct import Struct
from typing import Sequence

from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import ARENA_GENERATION, U32


_HEADER = Struct("<IIII")
_U32_INVALID = U32.invalid


class GenerationConfigArenaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DraftGenerationConfig:
    max_new_tokens: int
    proposal_depth: int
    stop_token_ids: tuple[int, ...] = ()
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        U32.validate(self.max_new_tokens)
        U32.validate(self.proposal_depth)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.proposal_depth <= 0:
            raise ValueError("proposal_depth must be positive")
        stops = tuple(_token(token, "stop_token_id") for token in self.stop_token_ids)
        object.__setattr__(self, "stop_token_ids", stops)
        if self.eos_token_id is not None:
            object.__setattr__(self, "eos_token_id", _token(self.eos_token_id, "eos_token_id"))

    @property
    def all_stop_token_ids(self) -> frozenset[int]:
        eos = () if self.eos_token_id is None else (self.eos_token_id,)
        return frozenset((*self.stop_token_ids, *eos))


class GenerationConfigArena:
    """Append-only fixed schema generation config arena.

    Schema:
      max_new_tokens:u32, proposal_depth:u32, eos_token_id:u32-or-invalid,
      stop_count:u32, stop_token_ids:u32[stop_count]
    """

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

    def write_config(self, config: DraftGenerationConfig) -> ArenaHandle:
        if not isinstance(config, DraftGenerationConfig):
            raise TypeError("config must be DraftGenerationConfig")
        eos = _U32_INVALID if config.eos_token_id is None else config.eos_token_id
        raw = bytearray(_HEADER.pack(config.max_new_tokens, config.proposal_depth, eos, len(config.stop_token_ids)))
        for token in config.stop_token_ids:
            raw += token.to_bytes(4, "little")
        if self._head + len(raw) > self.capacity_bytes:
            raise GenerationConfigArenaError("generation config arena capacity exhausted")
        offset = self._head
        self._bytes[offset : offset + len(raw)] = raw
        self._head += len(raw)
        return ArenaHandle(offset=offset, length=len(raw), generation=self._generation)

    def read_config(self, handle: ArenaHandle) -> DraftGenerationConfig:
        self._validate_handle(handle)
        if handle.length < _HEADER.size:
            raise GenerationConfigArenaError("generation config handle is too short")
        start = handle.offset
        max_new_tokens, proposal_depth, eos, stop_count = _HEADER.unpack(
            self._bytes[start : start + _HEADER.size]
        )
        expected = _HEADER.size + stop_count * 4
        if handle.length != expected:
            raise GenerationConfigArenaError("generation config handle length mismatch")
        stops_start = start + _HEADER.size
        stops = tuple(
            int.from_bytes(self._bytes[offset : offset + 4], "little")
            for offset in range(stops_start, stops_start + stop_count * 4, 4)
        )
        return DraftGenerationConfig(
            max_new_tokens=max_new_tokens,
            proposal_depth=proposal_depth,
            stop_token_ids=stops,
            eos_token_id=None if eos == _U32_INVALID else eos,
        )

    def reset_quiescent(self) -> None:
        self._head = 0
        self._generation = ARENA_GENERATION.next(self._generation)

    def _validate_handle(self, handle: ArenaHandle) -> None:
        if not isinstance(handle, ArenaHandle):
            raise TypeError("handle must be ArenaHandle")
        if handle.generation != self._generation:
            raise GenerationConfigArenaError("stale generation config arena generation")
        if not handle.within(self.capacity_bytes):
            raise GenerationConfigArenaError("generation config handle outside capacity")


def _token(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    U32.validate(value)
    return value
