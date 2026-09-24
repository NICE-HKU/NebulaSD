"""Fixed-size state-change notification ring for the observation plane."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from nebulasd.core.enums import StateChangeBlockKind, validate_enum
from nebulasd.core.ids import REQUEST_SLOT, U64


@dataclass(frozen=True, slots=True)
class StateChangeEntry:
    """Ring hint pointing to the canonical table fact."""

    block_kind: StateChangeBlockKind
    row: int
    publish_seq: int

    def __post_init__(self) -> None:
        if isinstance(self.block_kind, bool):
            raise TypeError("block_kind must not be bool")
        object.__setattr__(self, "block_kind", validate_enum(StateChangeBlockKind, int(self.block_kind)))
        REQUEST_SLOT.validate(self.row)
        U64.validate(self.publish_seq)


@dataclass(frozen=True, slots=True)
class StateChangePublishResult:
    accepted: bool
    overflowed: bool


@dataclass(frozen=True, slots=True)
class StateChangeBatch:
    entries: tuple[StateChangeEntry, ...]
    overflowed: bool


class StateChangeRing:
    """Single-consumer ring whose entries are hints, not source of truth."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an int")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._entries: list[StateChangeEntry | None] = [None] * capacity
        self._head = 0
        self._tail = 0
        self._size = 0
        self._overflowed = False
        self._lock = Lock()

    @property
    def capacity(self) -> int:
        return len(self._entries)

    def push(self, entry: StateChangeEntry) -> StateChangePublishResult:
        if not isinstance(entry, StateChangeEntry):
            raise TypeError("entry must be a StateChangeEntry")
        with self._lock:
            if self._size == self.capacity:
                self._overflowed = True
                return StateChangePublishResult(accepted=False, overflowed=True)

            self._entries[self._tail] = entry
            self._tail = (self._tail + 1) % self.capacity
            self._size += 1
            return StateChangePublishResult(accepted=True, overflowed=self._overflowed)

    def drain(self, max_entries: int | None = None) -> StateChangeBatch:
        if max_entries is not None:
            if isinstance(max_entries, bool) or not isinstance(max_entries, int):
                raise TypeError("max_entries must be an int")
            if max_entries < 0:
                raise ValueError("max_entries must be non-negative")

        with self._lock:
            limit = self._size if max_entries is None else min(self._size, max_entries)
            drained: list[StateChangeEntry] = []
            for _ in range(limit):
                entry = self._entries[self._head]
                if entry is None:
                    raise RuntimeError("state-change ring corruption")
                drained.append(entry)
                self._entries[self._head] = None
                self._head = (self._head + 1) % self.capacity
                self._size -= 1

            overflowed = self._overflowed
            self._overflowed = False
            return StateChangeBatch(entries=tuple(drained), overflowed=overflowed)

    def has_pending(self) -> bool:
        with self._lock:
            return self._size > 0 or self._overflowed
