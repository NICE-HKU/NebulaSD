"""Arena handle ABI helpers.

Generic arena handles are represented in native layouts as fixed-width numeric
triples: ``offset:u64, length:u32, generation:u32``. The unit of offset/length
is owned by the arena type; byte arenas use bytes, HostKV arenas use blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .ids import ARENA_GENERATION, ARENA_LENGTH, ARENA_OFFSET


@dataclass(frozen=True, slots=True)
class ArenaHandle:
    offset: int
    length: int
    generation: int

    def __post_init__(self) -> None:
        ARENA_OFFSET.validate(self.offset)
        ARENA_LENGTH.validate(self.length)
        ARENA_GENERATION.validate(self.generation)

    @classmethod
    def null(cls) -> "ArenaHandle":
        return cls(offset=0, length=0, generation=0)

    @property
    def end_offset(self) -> int:
        return self.offset + self.length

    def is_empty(self) -> bool:
        return self.length == 0

    def within(self, capacity_bytes: int) -> bool:
        ARENA_OFFSET.validate(capacity_bytes)
        return self.end_offset <= capacity_bytes


@dataclass(frozen=True, slots=True)
class HostKVArenaHandle:
    """HostKV arena handle whose offset and length are measured in KV blocks."""

    offset_blocks: int
    block_count: int
    generation: int

    def __post_init__(self) -> None:
        ARENA_OFFSET.validate(self.offset_blocks)
        ARENA_LENGTH.validate(self.block_count)
        ARENA_GENERATION.validate(self.generation)

    @property
    def end_block(self) -> int:
        return self.offset_blocks + self.block_count

    def within_blocks(self, total_blocks: int) -> bool:
        ARENA_OFFSET.validate(total_blocks)
        return self.end_block <= total_blocks


TokenArenaHandle: TypeAlias = ArenaHandle
ProposalArenaHandle: TypeAlias = ArenaHandle
OutputArenaHandle: TypeAlias = ArenaHandle
CommandArenaHandle: TypeAlias = ArenaHandle
GenerationConfigHandle: TypeAlias = ArenaHandle
DraftStateHandle: TypeAlias = ArenaHandle
