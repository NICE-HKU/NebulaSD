"""Versioned, bounded WORK wire format. Future inputs name table facts, never handles."""
from dataclasses import dataclass
from enum import IntEnum
from struct import Struct

from nebulasd.core.handles import ArenaHandle
from nebulasd.ipc.command_kinds import CommandKind

WORK_VERSION = 4
MAX_ROWS = 256
MAX_OUTSTANDING = 4
_HEADER = Struct('<4sHHIQQB7xQQIII')
_ROW = Struct('<IQQQIIQQIQIQQIIQIIQII')
_DEP = Struct('<IIQQI4x')


class WorkKind(IntEnum):
    TARGET_PREFILL = 1
    DRAFT_INITIAL = 2
    TARGET_VERIFY = 3
    DRAFT_DECODE = 4


class Selector(IntEnum):
    TARGET_HOST = 1
    DRAFT_HOST = 2
    PROPOSAL = 3
    DELTA = 4
    CLASSIFIED = 5
    TARGET_DECISION = 6


class Outcome(IntEnum):
    EXECUTED = 1
    SKIPPED_FINISHED = 2
    SKIPPED_SHUTDOWN = 3


@dataclass(frozen=True, slots=True)
class TableDependency:
    kind: int
    slot: int
    request_epoch: int
    expected_ticket: int
    selector: Selector

    def to_bytes(self):
        return _DEP.pack(self.kind, self.slot, self.request_epoch,
                         self.expected_ticket, int(self.selector))


@dataclass(frozen=True, slots=True)
class RowWork:
    slot: int
    epoch: int
    round_id: int
    run_seq: int
    destination_offset: int
    capacity_blocks: int
    host_offset: int
    host_capacity: int
    host_arena: int
    host_generation: int
    host_slot: int
    writer_generation: int
    prompt: ArenaHandle
    config: ArenaHandle
    output: ArenaHandle
    token_budget: int
    max_new_tokens: int
    prompt_count: int
    source: TableDependency | None = None
    predecessor: TableDependency | None = None
    classified: TableDependency | None = None

    owner_epoch: int | None = None
    layout_id: int | None = None

    def to_bytes(self):
        raw = _ROW.pack(self.slot, self.epoch, self.round_id, self.run_seq,
            self.destination_offset, self.capacity_blocks, self.host_offset,
            self.host_capacity, self.host_arena, self.host_generation, self.host_slot, self.writer_generation,
            self.prompt.offset, self.prompt.length, self.prompt.generation,
            self.config.offset, self.config.length, self.config.generation,
            self.output.offset, self.output.length, self.output.generation)
        raw += Struct('<III').pack(self.token_budget, self.max_new_tokens, self.prompt_count)
        for dep in (self.source, self.predecessor, self.classified):
            raw += bytes(_DEP.size) if dep is None else dep.to_bytes()
        raw += Struct('<QQ').pack(*((1 << 64)-1 if v is None else v for v in (self.owner_epoch, self.layout_id)))
        return raw


ROW_BYTES = _ROW.size + 12 + 3 * _DEP.size + 16


@dataclass(frozen=True, slots=True)
class Work:
    worker_id: int
    worker_generation: int
    work_seq: int
    operation: WorkKind
    bank_id: int
    bank_epoch: int
    completion_offset: int
    result_bytes: int
    rows: tuple[RowWork, ...]
    kind = CommandKind.WORK

    @property
    def command_seq(self):
        return self.work_seq

    def __post_init__(self):
        object.__setattr__(self, 'operation', WorkKind(self.operation))
        if self.bank_id not in (0, 1) or not 0 < len(self.rows) <= MAX_ROWS:
            raise ValueError('WORK requires Bank 0/1 and 1..256 rows')
        if len({r.slot for r in self.rows}) != len(self.rows):
            raise ValueError('duplicate WORK request')
        end = 0
        for r in self.rows:
            if r.destination_offset != end or r.capacity_blocks <= 0:
                raise ValueError('WORK layout must be positive contiguous frozen ranges')
            if self.operation in (WorkKind.DRAFT_INITIAL, WorkKind.DRAFT_DECODE):
                if r.owner_epoch is None or r.layout_id is None or r.predecessor is None:
                    raise ValueError('Draft WORK requires explicit owner/layout and Target delta')
                if r.predecessor.selector != Selector.DELTA or r.classified is None or r.classified.selector not in (Selector.CLASSIFIED, Selector.TARGET_DECISION):
                    raise ValueError('Draft requires Target delta and a completion decision')
                if r.source is not None and r.source.selector != Selector.DRAFT_HOST:
                    raise ValueError('Draft source must be a Draft HOST_READY selector')
                if self.operation == WorkKind.DRAFT_INITIAL and r.source is not None:
                    raise ValueError('Draft initial cannot import a source')
            end += r.capacity_blocks
            if r.host_capacity < r.capacity_blocks:
                raise ValueError('HostKV reservation smaller than layout')
            for d in (r.source, r.predecessor, r.classified):
                if d is not None and (d.slot, d.request_epoch) != (r.slot, r.epoch):
                    raise ValueError('dependency request mismatch')
            if self.operation != WorkKind.TARGET_PREFILL and r.classified is None:
                raise ValueError('compute requires a completion decision')
            if self.operation in (WorkKind.TARGET_VERIFY, WorkKind.DRAFT_DECODE) and r.source is None:
                raise ValueError('continuation requires HostKV source')
        if self.result_bytes <= 0:
            raise ValueError('WORK requires pre-reserved result storage')

    def to_bytes(self):
        return _HEADER.pack(b'SDWO', WORK_VERSION, int(self.operation), self.worker_id,
            self.worker_generation, self.work_seq, self.bank_id, self.bank_epoch,
            self.completion_offset, self.result_bytes, len(self.rows), ROW_BYTES) + b''.join(r.to_bytes() for r in self.rows)

    @classmethod
    def from_bytes(cls, raw):
        if len(raw) < _HEADER.size:
            raise ValueError('truncated WORK')
        magic, version, op, wid, gen, seq, bank, epoch, completion, budget, count, stride = _HEADER.unpack_from(raw)
        if magic != b'SDWO' or version != WORK_VERSION or stride != ROW_BYTES:
            raise ValueError('unsupported WORK ABI')
        if not 0 < count <= MAX_ROWS or len(raw) != _HEADER.size + count * stride:
            raise ValueError('invalid WORK byte count')
        rows = []
        for i in range(count):
            pos = _HEADER.size + i * stride
            v = _ROW.unpack_from(raw, pos)
            budgets = Struct('<III').unpack_from(raw, pos + _ROW.size)
            deps = []
            for j in range(3):
                values = _DEP.unpack_from(raw, pos + _ROW.size + 12 + j * _DEP.size)
                if values == (0, 0, 0, 0, 0):
                    deps.append(None)
                else:
                    deps.append(TableDependency(*values[:4], Selector(values[4])))
            rows.append(RowWork(*v[:12], ArenaHandle(*v[12:15]), ArenaHandle(*v[15:18]),
                                ArenaHandle(*v[18:21]), *budgets, *deps,
                                *(None if x == (1 << 64)-1 else x for x in Struct('<QQ').unpack_from(raw, pos + stride - 16))))
        return cls(wid, gen, seq, WorkKind(op), bank, epoch, completion, budget, tuple(rows))
