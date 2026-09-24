"""Draft prepare/run wire contracts. No export/commit/release round trips."""
from dataclasses import dataclass
from struct import Struct
from typing import ClassVar
from nebulasd.core.draft_contracts import DraftSnapshotIdentity, require_handle, require_snapshot_handle, pack_handle
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import U32, U64, ROUND_ID, OP_SEQ
from .command_kinds import CommandKind

_PREPARE = Struct('<QQQII')
_RUN = Struct('<IQQQQQI')
_BATCH = Struct('<QBQI')


def bank_id(value):
    U32.validate(value)
    if value not in (0, 1):
        raise ValueError('Draft requires bank 0 or 1')


@dataclass(frozen=True, slots=True)
class DraftPrepareRequest:
    source: DraftSnapshotIdentity
    snapshot_handle: ArenaHandle
    prepare_seq: int
    next_round_id: int
    next_owner_epoch: int
    destination_offset_blocks: int
    destination_capacity_blocks: int
    growth_token_budget: int

    def __post_init__(self):
        if not isinstance(self.source, DraftSnapshotIdentity):
            raise TypeError('expected snapshot identity')
        require_snapshot_handle(self.snapshot_handle)
        for n in ('prepare_seq', 'next_round_id', 'next_owner_epoch'):
            U64.validate(getattr(self, n))
        for n in ('destination_offset_blocks', 'destination_capacity_blocks', 'growth_token_budget'):
            U32.validate(getattr(self, n))
        if self.next_round_id != ROUND_ID.next(self.source.round_id):
            raise ValueError('prepare must name the round after snapshot')
        if self.next_owner_epoch != U64.next(self.source.owner_epoch):
            raise ValueError('prepare must advance owner epoch exactly once')
        required = (self.source.logical_kv_len + self.growth_token_budget + self.source.allocation.block_size - 1) // self.source.allocation.block_size
        # Near the output/context limit, reconcile may crop the imported
        # proposal and need zero or one extra KV token. The scheduler supplies
        # the bounded growth; the worker still preflights the actual next run.
        if self.destination_capacity_blocks < required:
            raise ValueError('Draft destination must reserve reconcile/proposal growth')

    @property
    def request_slot(self): return self.source.request_slot
    @property
    def request_epoch(self): return self.source.request_epoch
    @property
    def op_seq(self): return self.prepare_seq

    def to_bytes(self):
        return (self.source.to_bytes() + pack_handle(self.snapshot_handle) + _PREPARE.pack(
            self.prepare_seq, self.next_round_id, self.next_owner_epoch,
            self.destination_offset_blocks, self.destination_capacity_blocks)
            + self.growth_token_budget.to_bytes(4, 'little'))


@dataclass(frozen=True, slots=True)
class DraftRunRequest:
    request_slot: int
    request_epoch: int
    round_id: int
    run_seq: int
    owner_epoch: int
    snapshot_version: int
    scheduled_token_count: int
    token_delta_handle: ArenaHandle

    def __post_init__(self):
        for n in ('request_slot', 'scheduled_token_count'):
            U32.validate(getattr(self, n))
        for n in ('request_epoch', 'round_id', 'run_seq', 'owner_epoch', 'snapshot_version'):
            U64.validate(getattr(self, n))
        require_handle(self.token_delta_handle)
        if not self.scheduled_token_count:
            raise ValueError('Draft run requires a positive token budget')

    def to_bytes(self):
        return _RUN.pack(*(getattr(self, n) for n in self.__dataclass_fields__ if n != 'token_delta_handle')) + pack_handle(self.token_delta_handle)


def validate_batch(command):
    for n in ('worker_id',): U32.validate(getattr(command, n))
    for n in ('worker_generation', 'command_seq'): U64.validate(getattr(command, n))
    if not isinstance(command.requests, tuple) or not command.requests:
        raise ValueError('Draft batch requires immutable non-empty membership')
    slots = [r.request_slot for r in command.requests]
    if len(set(slots)) != len(slots):
        raise ValueError('duplicate Draft request slot')


@dataclass(frozen=True, slots=True)
class PrepareDraftBankCommand:
    worker_id: int
    worker_generation: int
    command_seq: int
    batch_seq: int
    standby_bank_id: int
    next_bank_epoch: int
    requests: tuple[DraftPrepareRequest, ...]
    kind: ClassVar[CommandKind] = CommandKind.PREPARE_DRAFT_BANK

    def __post_init__(self):
        validate_batch(self)
        U64.validate(self.batch_seq)
        U64.validate(self.next_bank_epoch)
        bank_id(self.standby_bank_id)
        offset = 0
        for row in self.requests:
            if not isinstance(row, DraftPrepareRequest): raise TypeError('expected DraftPrepareRequest')
            if row.destination_offset_blocks != offset:
                raise ValueError('Draft prepare ranges must be contiguous in member order')
            offset += row.destination_capacity_blocks
        U32.validate(offset)


@dataclass(frozen=True, slots=True)
class RunDraftBatchCommand:
    worker_id: int
    worker_generation: int
    command_seq: int
    expected_batch_seq: int
    active_bank_id: int
    active_bank_epoch: int
    requests: tuple[DraftRunRequest, ...]
    kind: ClassVar[CommandKind] = CommandKind.RUN_DRAFT_BATCH

    def __post_init__(self):
        validate_batch(self)
        U64.validate(self.expected_batch_seq)
        U64.validate(self.active_bank_epoch)
        bank_id(self.active_bank_id)
        if any(not isinstance(r, DraftRunRequest) for r in self.requests):
            raise TypeError('expected DraftRunRequest')


def validate_prepared_run(prepare, run):
    if (prepare.worker_id, prepare.worker_generation, prepare.batch_seq, prepare.standby_bank_id, prepare.next_bank_epoch) != (
        run.worker_id, run.worker_generation, run.expected_batch_seq, run.active_bank_id, run.active_bank_epoch):
        raise ValueError('Draft run Bank/batch/worker fence mismatch')
    if len(prepare.requests) != len(run.requests):
        raise ValueError('Draft run membership mismatch')
    for p, r in zip(prepare.requests, run.requests, strict=True):
        if (p.request_slot, p.request_epoch, p.next_round_id, p.next_owner_epoch, p.source.snapshot_version) != (
            r.request_slot, r.request_epoch, r.round_id, r.owner_epoch, r.snapshot_version):
            raise ValueError('Draft run ordered member/owner/snapshot fence mismatch')
        if not OP_SEQ.is_newer(r.run_seq, p.source.op_seq):
            raise ValueError('Draft run sequence must advance source compute sequence')
        # Proposal depth is not net KV growth: reconcile may crop, and the
        # remaining output budget may truncate this proposal. The destination
        # checks the actual bounded logical length before consuming this Bank.


class DraftPreparedContract:
    """One frozen next batch. Physical reservations remain a runtime concern."""
    def __init__(self):
        self.pending = None

    def discard(self, bank_id, bank_epoch, batch_seq):
        command = self.pending
        if command is None or (command.standby_bank_id, command.next_bank_epoch, command.batch_seq) != (bank_id, bank_epoch, batch_seq):
            return False
        self.pending = None
        return True

    def accept(self, command):
        if isinstance(command, PrepareDraftBankCommand):
            if self.pending is not None:
                raise ValueError('unconsumed Draft prepare cannot be overwritten')
            self.pending = command
        else:
            if self.pending is None:
                raise ValueError('Draft run requires an unconsumed prepare')
            validate_prepared_run(self.pending, command)
            self.pending = None


def encode_draft_command(command):
    if isinstance(command, PrepareDraftBankCommand):
        head = _BATCH.pack(command.batch_seq, command.standby_bank_id, command.next_bank_epoch, len(command.requests))
    else:
        head = _BATCH.pack(command.expected_batch_seq, command.active_bank_id, command.active_bank_epoch, len(command.requests))
    return head + b''.join(r.to_bytes() for r in command.requests)


def decode_draft_command(kind, reader, worker_id, generation, seq):
    batch, bank, epoch, count = reader.u64(), reader.u8(), reader.u64(), reader.u32()
    rows = []
    for _ in range(count):
        if kind == CommandKind.PREPARE_DRAFT_BANK:
            source = DraftSnapshotIdentity.from_bytes(reader.raw(DraftSnapshotIdentity.byte_size))
            rows.append(DraftPrepareRequest(source, reader.handle(), reader.u64(), reader.u64(), reader.u64(),
                                            reader.u32(), reader.u32(), reader.u32()))
        else:
            rows.append(DraftRunRequest(reader.u32(), reader.u64(), reader.u64(), reader.u64(), reader.u64(),
                                        reader.u64(), reader.u32(), reader.handle()))
    cls = PrepareDraftBankCommand if kind == CommandKind.PREPARE_DRAFT_BANK else RunDraftBatchCommand
    return cls(worker_id, generation, seq, batch, bank, epoch, tuple(rows))


@dataclass(frozen=True, slots=True)
class DraftInitialBank:
    """Initial batch reservation intent; owner epoch and source version are zero."""
    bank_id: int
    bank_epoch: int
    batch_seq: int
    block_size: int
    capacity_blocks: tuple[int, ...]

    def __post_init__(self):
        bank_id(self.bank_id)
        U64.validate(self.bank_epoch)
        U64.validate(self.batch_seq)
        U32.validate(self.block_size)
        if not self.block_size or not isinstance(self.capacity_blocks, tuple) or not self.capacity_blocks:
            raise ValueError('initial Draft Bank requires block size and frozen capacities')
        for count in self.capacity_blocks:
            U32.validate(count)
            if not count: raise ValueError('initial Draft reservation must be positive')
        U32.validate(sum(self.capacity_blocks))

    def validate_requests(self, new_requests, deltas, removed):
        if deltas or removed or len(new_requests) != len(self.capacity_blocks):
            raise ValueError('initial Draft Bank must exactly match new-request membership')
        for request, capacity in zip(new_requests, self.capacity_blocks, strict=True):
            tokens = (request.input_tokens_handle.length + request.initial_output_tokens_handle.length) // 4
            if not request.scheduled_token_count or capacity * self.block_size < tokens + request.scheduled_token_count - 1:
                raise ValueError('initial Draft reservation does not cover proposal growth')

    def to_bytes(self):
        return (Struct('<BQQII').pack(self.bank_id, self.bank_epoch, self.batch_seq, self.block_size, len(self.capacity_blocks))
                + b''.join(n.to_bytes(4, 'little') for n in self.capacity_blocks))

    @classmethod
    def read(cls, reader):
        bank, epoch, batch, block_size, count = reader.u8(), reader.u64(), reader.u64(), reader.u32(), reader.u32()
        return cls(bank, epoch, batch, block_size, tuple(reader.u32() for _ in range(count)))
