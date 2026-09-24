"""Numeric identity, epoch, version, and sequence rules for the wire ABI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


RequestSlot: TypeAlias = int
RequestEpoch: TypeAlias = int
RoundId: TypeAlias = int
OpSeq: TypeAlias = int
CommandSeq: TypeAlias = int
BatchSeq: TypeAlias = int
WorkerId: TypeAlias = int
WorkerGeneration: TypeAlias = int
BankId: TypeAlias = int
BankEpoch: TypeAlias = int
HostSlot: TypeAlias = int
HostSlotGeneration: TypeAlias = int
WriterLeaseGeneration: TypeAlias = int
KVVersion: TypeAlias = int


@dataclass(frozen=True, slots=True)
class UIntDomain:
    """Unsigned integer domain with one all-ones invalid sentinel."""

    name: str
    bits: int

    @property
    def modulus(self) -> int:
        return 1 << self.bits

    @property
    def max_value(self) -> int:
        return self.modulus - 1

    @property
    def invalid(self) -> int:
        return self.max_value

    @property
    def max_valid(self) -> int:
        return self.max_value - 1

    @property
    def half_range(self) -> int:
        return self.max_value // 2

    def validate(self, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{self.name} must be an int, not bool")
        if not isinstance(value, int):
            raise TypeError(f"{self.name} must be an int")
        if value < 0 or value > self.max_valid:
            raise ValueError(f"{self.name} must be in [0, {self.max_valid}]")
        return value

    def is_valid(self, value: int) -> bool:
        return isinstance(value, int) and 0 <= value <= self.max_valid

    def next(self, value: int) -> int:
        self.validate(value)
        return (value + 1) % self.max_value

    def distance(self, newer: int, older: int) -> int:
        self.validate(newer)
        self.validate(older)
        return (newer - older) % self.max_value

    def is_newer(self, candidate: int, baseline: int) -> bool:
        distance = self.distance(candidate, baseline)
        return 0 < distance <= self.half_range

    def is_newer_or_equal(self, candidate: int, baseline: int) -> bool:
        return candidate == baseline or self.is_newer(candidate, baseline)


U8 = UIntDomain("u8", 8)
U16 = UIntDomain("u16", 16)
U32 = UIntDomain("u32", 32)
U64 = UIntDomain("u64", 64)

REQUEST_SLOT = U32
REQUEST_EPOCH = U64
ROUND_ID = U64
OP_SEQ = U64
COMMAND_SEQ = U64
BATCH_SEQ = U64
WORKER_ID = U32
WORKER_GENERATION = U64
BANK_ID = U8
BANK_EPOCH = U64
HOST_SLOT = U32
HOST_SLOT_GENERATION = U64
WRITER_LEASE_GENERATION = U64
KV_VERSION = U64
ARENA_OFFSET = U64
ARENA_LENGTH = U32
ARENA_GENERATION = U32


@dataclass(frozen=True, slots=True)
class RequestFence:
    """Fence carried by worker facts to prevent slot reuse ABA bugs."""

    request_slot: RequestSlot
    request_epoch: RequestEpoch

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)

    def matches(self, other: "RequestFence") -> bool:
        return self.request_slot == other.request_slot and self.request_epoch == other.request_epoch


@dataclass(frozen=True, slots=True)
class OperationFence:
    """Fence for one request operation within a specific worker generation."""

    request: RequestFence
    round_id: RoundId
    op_seq: OpSeq
    worker_id: WorkerId
    worker_generation: WorkerGeneration

    def __post_init__(self) -> None:
        ROUND_ID.validate(self.round_id)
        OP_SEQ.validate(self.op_seq)
        WORKER_ID.validate(self.worker_id)
        WORKER_GENERATION.validate(self.worker_generation)

    def applies_to(self, intent: "OperationFence") -> bool:
        return (
            self.request.matches(intent.request)
            and self.round_id == intent.round_id
            and self.op_seq == intent.op_seq
            and self.worker_id == intent.worker_id
            and self.worker_generation == intent.worker_generation
        )


def is_stale_request_fact(current: RequestFence, fact: RequestFence) -> bool:
    """Return true when a fact belongs to an old occupant of the same slot."""

    return current.request_slot == fact.request_slot and current.request_epoch != fact.request_epoch
