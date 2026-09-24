"""Fixed-width command protocol for the dispatch plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from struct import Struct
from typing import ClassVar

from nebulasd.core.enums import validate_enum
from nebulasd.core.handles import ArenaHandle, HostKVArenaHandle
from nebulasd.core.ids import (
    ARENA_LENGTH,
    ARENA_OFFSET,
    BANK_EPOCH,
    BANK_ID,
    BATCH_SEQ,
    COMMAND_SEQ,
    HOST_SLOT,
    HOST_SLOT_GENERATION,
    KV_VERSION,
    OP_SEQ,
    REQUEST_EPOCH,
    REQUEST_SLOT,
    ROUND_ID,
    U32,
    U64,
    WORKER_GENERATION,
    WRITER_LEASE_GENERATION,
    WORKER_ID,
)


COMMAND_PROTOCOL_VERSION = 5
COMMAND_PAYLOAD_MAGIC = 0x5344_434D
COMMAND_HEADER_STRUCT = Struct("<QQIQII")


from .command_kinds import CommandKind
from .draft_protocol import (PrepareDraftBankCommand, RunDraftBatchCommand, DraftPrepareRequest,
                             DraftRunRequest, DraftInitialBank, DraftPreparedContract, encode_draft_command, decode_draft_command)

class ManagementCommandKind(IntEnum):
    CANCEL_REQUEST = 1
    RELEASE_REQUEST = 2
    SNAPSHOT_WORKER = 3
    SHUTDOWN_WORKER = 4


@dataclass(frozen=True, slots=True)
class CommandHeader:
    command_seq: int
    worker_generation: int
    command_kind: CommandKind
    payload_offset: int
    payload_length: int
    flags: int = 0

    byte_size: ClassVar[int] = COMMAND_HEADER_STRUCT.size

    def __post_init__(self) -> None:
        COMMAND_SEQ.validate(self.command_seq)
        WORKER_GENERATION.validate(self.worker_generation)
        object.__setattr__(self, "command_kind", validate_enum(CommandKind, self.command_kind))
        ARENA_OFFSET.validate(self.payload_offset)
        ARENA_LENGTH.validate(self.payload_length)
        U32.validate(self.flags)

    def to_bytes(self) -> bytes:
        return COMMAND_HEADER_STRUCT.pack(
            self.command_seq,
            self.worker_generation,
            int(self.command_kind),
            self.payload_offset,
            self.payload_length,
            self.flags,
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CommandHeader":
        if len(raw) != COMMAND_HEADER_STRUCT.size:
            raise ValueError(f"command header must be {COMMAND_HEADER_STRUCT.size} bytes")
        command_seq, worker_generation, command_kind, payload_offset, payload_length, flags = COMMAND_HEADER_STRUCT.unpack(raw)
        return cls(command_seq, worker_generation, CommandKind(command_kind), payload_offset, payload_length, flags)


@dataclass(frozen=True, slots=True)
class RemovedRequest:
    request_slot: int
    request_epoch: int

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)


@dataclass(frozen=True, slots=True)
class OwnedArenaHandle:
    owner_worker_id: int
    owner_generation: int
    handle: ArenaHandle

    def __post_init__(self) -> None:
        WORKER_ID.validate(self.owner_worker_id)
        WORKER_GENERATION.validate(self.owner_generation)
        _require_handle_type(self.handle, ArenaHandle, "handle")


@dataclass(frozen=True, slots=True)
class DraftNewRequestData:
    """Shared wire row for new requests.

    Draft commands require a non-empty initial Target output anchor. Target
    prepare commands may carry a null anchor before initial Target prefill has
    produced that output.
    """

    request_slot: int
    request_epoch: int
    round_id: int
    op_seq: int
    scheduled_token_count: int
    input_tokens_handle: ArenaHandle
    initial_output_tokens_handle: ArenaHandle
    generation_config_handle: ArenaHandle
    initial_kv_handle: ArenaHandle

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        ROUND_ID.validate(self.round_id)
        OP_SEQ.validate(self.op_seq)
        U32.validate(self.scheduled_token_count)
        _require_handle_type(self.input_tokens_handle, ArenaHandle, "input_tokens_handle")
        _require_handle_type(self.initial_output_tokens_handle, ArenaHandle, "initial_output_tokens_handle")
        _require_handle_type(self.generation_config_handle, ArenaHandle, "generation_config_handle")
        _require_handle_type(self.initial_kv_handle, ArenaHandle, "initial_kv_handle")


NewRequestData = DraftNewRequestData


@dataclass(frozen=True, slots=True)
class CachedRequestDelta:
    request_slot: int
    request_epoch: int
    round_id: int
    op_seq: int
    token_delta_handle: ArenaHandle
    proposal_handle: ArenaHandle
    hostkv_handle: HostKVArenaHandle
    target_bank_mapping_handle: ArenaHandle
    scheduled_token_count: int

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        ROUND_ID.validate(self.round_id)
        OP_SEQ.validate(self.op_seq)
        U32.validate(self.scheduled_token_count)
        _require_handle_type(self.token_delta_handle, ArenaHandle, "token_delta_handle")
        _require_handle_type(self.proposal_handle, ArenaHandle, "proposal_handle")
        _require_handle_type(self.hostkv_handle, HostKVArenaHandle, "hostkv_handle")
        _require_handle_type(self.target_bank_mapping_handle, ArenaHandle, "target_bank_mapping_handle")


@dataclass(frozen=True, slots=True)
class TargetPrefillRequest:
    request_slot: int
    request_epoch: int
    round_id: int
    run_seq: int
    input_tokens_handle: ArenaHandle
    generation_config_handle: ArenaHandle
    scheduled_token_count: int
    max_output_len: int
    bank_id: int
    bank_epoch: int
    bank_offset_blocks: int
    block_count: int

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        ROUND_ID.validate(self.round_id)
        OP_SEQ.validate(self.run_seq)
        _require_handle_type(self.input_tokens_handle, ArenaHandle, "input_tokens_handle")
        _require_handle_type(self.generation_config_handle, ArenaHandle, "generation_config_handle")
        U32.validate(self.scheduled_token_count)
        U32.validate(self.max_output_len)
        BANK_ID.validate(self.bank_id)
        BANK_EPOCH.validate(self.bank_epoch)
        U32.validate(self.bank_offset_blocks)
        U32.validate(self.block_count)
        if self.block_count == 0:
            raise ValueError("TargetPrefillRequest.block_count must be positive")
        if self.max_output_len == 0:
            raise ValueError("TargetPrefillRequest.max_output_len must be positive")


@dataclass(frozen=True, slots=True)
class BankAllocation:
    request_slot: int
    request_epoch: int
    bank_id: int
    bank_epoch: int
    offset_blocks: int
    block_count: int

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        BANK_ID.validate(self.bank_id)
        BANK_EPOCH.validate(self.bank_epoch)
        U32.validate(self.offset_blocks)
        U32.validate(self.block_count)


@dataclass(frozen=True, slots=True)
class HostKVSource:
    request_slot: int
    request_epoch: int
    host_slot: int
    host_slot_generation: int
    source_host_version: int

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        HOST_SLOT.validate(self.host_slot)
        HOST_SLOT_GENERATION.validate(self.host_slot_generation)
        KV_VERSION.validate(self.source_host_version)


@dataclass(frozen=True, slots=True)
class RunTargetRequest:
    request_slot: int
    request_epoch: int
    round_id: int
    run_seq: int
    proposal_handle: ArenaHandle
    output_handle: ArenaHandle

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        ROUND_ID.validate(self.round_id)
        OP_SEQ.validate(self.run_seq)
        _require_handle_type(self.proposal_handle, ArenaHandle, "proposal_handle")
        _require_handle_type(self.output_handle, ArenaHandle, "output_handle")


@dataclass(frozen=True, slots=True)
class TargetVerifyRequest:
    request_slot: int
    request_epoch: int
    round_id: int
    run_seq: int
    proposal_handle: OwnedArenaHandle
    committed_output_handle: ArenaHandle
    committed_output_count: int
    prompt_token_count: int
    generation_config_handle: ArenaHandle
    bank_id: int
    bank_epoch: int
    bank_offset_blocks: int
    block_count: int

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        ROUND_ID.validate(self.round_id)
        OP_SEQ.validate(self.run_seq)
        _require_handle_type(self.proposal_handle, OwnedArenaHandle, "proposal_handle")
        _require_handle_type(self.committed_output_handle, ArenaHandle, "committed_output_handle")
        U32.validate(self.committed_output_count)
        U32.validate(self.prompt_token_count)
        _require_handle_type(self.generation_config_handle, ArenaHandle, "generation_config_handle")
        BANK_ID.validate(self.bank_id)
        BANK_EPOCH.validate(self.bank_epoch)
        U32.validate(self.bank_offset_blocks)
        U32.validate(self.block_count)
        if self.block_count == 0:
            raise ValueError("TargetVerifyRequest.block_count must be positive")


@dataclass(frozen=True, slots=True)
class TargetPrepareRequest:
    request_slot: int
    request_epoch: int
    round_id: int
    op_seq: int
    committed_output_handle: ArenaHandle
    committed_output_count: int
    prompt_token_count: int
    generation_config_handle: ArenaHandle
    hostkv_handle: HostKVArenaHandle
    host_slot: int
    host_slot_generation: int
    host_writer_lease_generation: int
    source_host_version: int
    logical_kv_len: int
    committed_blocks: int
    valid_blocks: int
    destination_bank_id: int
    destination_bank_epoch: int
    destination_bank_offset_blocks: int
    destination_capacity_blocks: int

    def __post_init__(self) -> None:
        REQUEST_SLOT.validate(self.request_slot)
        REQUEST_EPOCH.validate(self.request_epoch)
        ROUND_ID.validate(self.round_id)
        OP_SEQ.validate(self.op_seq)
        _require_handle_type(self.committed_output_handle, ArenaHandle, "committed_output_handle")
        U32.validate(self.committed_output_count)
        U32.validate(self.prompt_token_count)
        _require_handle_type(self.generation_config_handle, ArenaHandle, "generation_config_handle")
        _require_handle_type(self.hostkv_handle, HostKVArenaHandle, "hostkv_handle")
        HOST_SLOT.validate(self.host_slot)
        HOST_SLOT_GENERATION.validate(self.host_slot_generation)
        WRITER_LEASE_GENERATION.validate(self.host_writer_lease_generation)
        KV_VERSION.validate(self.source_host_version)
        U32.validate(self.logical_kv_len)
        U32.validate(self.committed_blocks)
        U32.validate(self.valid_blocks)
        BANK_ID.validate(self.destination_bank_id)
        BANK_EPOCH.validate(self.destination_bank_epoch)
        U32.validate(self.destination_bank_offset_blocks)
        U32.validate(self.destination_capacity_blocks)
        if self.destination_capacity_blocks == 0:
            raise ValueError("TargetPrepareRequest.destination_capacity_blocks must be positive")
        if self.valid_blocks > self.destination_capacity_blocks:
            raise ValueError("TargetPrepareRequest.valid_blocks cannot exceed destination capacity")
        if self.committed_blocks > self.valid_blocks:
            raise ValueError("TargetPrepareRequest.committed_blocks cannot exceed valid_blocks")
        if self.hostkv_handle.block_count < self.committed_blocks:
            raise ValueError("TargetPrepareRequest hostkv_handle is shorter than committed blocks")


@dataclass(frozen=True, slots=True)
class DraftBatchCommand:
    worker_id: int
    worker_generation: int
    command_seq: int
    new_requests: tuple[NewRequestData, ...]
    cached_request_deltas: tuple[CachedRequestDelta, ...]
    removed_requests: tuple[RemovedRequest, ...] = ()
    bank: DraftInitialBank | None = None

    kind: ClassVar[CommandKind] = CommandKind.DRAFT_BATCH

    def __post_init__(self) -> None:
        WORKER_ID.validate(self.worker_id)
        WORKER_GENERATION.validate(self.worker_generation)
        COMMAND_SEQ.validate(self.command_seq)
        _validate_request_lists(self.new_requests, self.cached_request_deltas, self.removed_requests)
        if self.bank is not None:
            if not isinstance(self.bank, DraftInitialBank):
                raise TypeError("bank must be DraftInitialBank")
            self.bank.validate_requests(self.new_requests, self.cached_request_deltas, self.removed_requests)
        for row in self.new_requests:
            if row.initial_output_tokens_handle.is_empty():
                raise ValueError("DraftBatchCommand new_requests require a non-empty initial_output_tokens_handle")


@dataclass(frozen=True, slots=True)
class TargetPrefillBatchCommand:
    worker_id: int
    target_generation: int
    command_seq: int
    batch_seq: int
    bank_id: int
    bank_epoch: int
    requests: tuple[TargetPrefillRequest, ...]

    kind: ClassVar[CommandKind] = CommandKind.TARGET_PREFILL_BATCH

    def __post_init__(self) -> None:
        WORKER_ID.validate(self.worker_id)
        WORKER_GENERATION.validate(self.target_generation)
        COMMAND_SEQ.validate(self.command_seq)
        BATCH_SEQ.validate(self.batch_seq)
        BANK_ID.validate(self.bank_id)
        BANK_EPOCH.validate(self.bank_epoch)
        _ensure_unique_request_keys(self.requests, "requests")
        for request in self.requests:
            if request.bank_id != self.bank_id:
                raise ValueError("TargetPrefillRequest.bank_id must match command bank_id")
            if request.bank_epoch != self.bank_epoch:
                raise ValueError("TargetPrefillRequest.bank_epoch must match command bank_epoch")


@dataclass(frozen=True, slots=True)
class PrepareTargetBankCommand:
    worker_id: int
    target_generation: int
    command_seq: int
    batch_seq: int
    standby_bank_id: int
    next_bank_epoch: int
    requests: tuple[TargetPrepareRequest, ...]

    kind: ClassVar[CommandKind] = CommandKind.PREPARE_TARGET_BANK

    def __post_init__(self) -> None:
        WORKER_ID.validate(self.worker_id)
        WORKER_GENERATION.validate(self.target_generation)
        COMMAND_SEQ.validate(self.command_seq)
        BATCH_SEQ.validate(self.batch_seq)
        BANK_ID.validate(self.standby_bank_id)
        BANK_EPOCH.validate(self.next_bank_epoch)
        _ensure_unique_request_keys(self.requests, "requests")
        expected_offset = 0
        for request in self.requests:
            if request.destination_bank_id != self.standby_bank_id:
                raise ValueError("TargetPrepareRequest.destination_bank_id must match standby_bank_id")
            if request.destination_bank_epoch != self.next_bank_epoch:
                raise ValueError("TargetPrepareRequest.destination_bank_epoch must match next_bank_epoch")
            if request.destination_bank_offset_blocks != expected_offset:
                raise ValueError("TargetPrepareRequest destination ranges must be contiguous in request order")
            expected_offset += request.destination_capacity_blocks


@dataclass(frozen=True, slots=True)
class RunTargetBatchCommand:
    worker_id: int
    target_generation: int
    command_seq: int
    expected_batch_seq: int
    active_bank_id: int
    active_bank_epoch: int
    standby_bank_epoch: int
    requests: tuple[RunTargetRequest | TargetVerifyRequest, ...]

    kind: ClassVar[CommandKind] = CommandKind.RUN_TARGET_BATCH

    def __post_init__(self) -> None:
        WORKER_ID.validate(self.worker_id)
        WORKER_GENERATION.validate(self.target_generation)
        COMMAND_SEQ.validate(self.command_seq)
        BATCH_SEQ.validate(self.expected_batch_seq)
        BANK_ID.validate(self.active_bank_id)
        BANK_EPOCH.validate(self.active_bank_epoch)
        BANK_EPOCH.validate(self.standby_bank_epoch)
        _ensure_unique_request_keys(self.requests, "requests")


from nebulasd.workers.work import Work


HotCommand = Work | PrepareDraftBankCommand | RunDraftBatchCommand | DraftBatchCommand | TargetPrefillBatchCommand | PrepareTargetBankCommand | RunTargetBatchCommand


@dataclass(frozen=True, slots=True)
class WorkerRequestCacheEntry:
    request_slot: int
    request_epoch: int
    last_round_id: int
    last_op_seq: int
    input_tokens_handle: ArenaHandle
    initial_output_tokens_handle: ArenaHandle
    generation_config_handle: ArenaHandle
    token_delta_handle: ArenaHandle
    proposal_handle: ArenaHandle
    hostkv_handle: HostKVArenaHandle
    target_bank_mapping_handle: ArenaHandle
    prepared_batch_seq: int | None = None
    prepared_bank_id: int | None = None
    prepared_bank_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class PreparedBatchState:
    batch_seq: int
    bank_id: int
    bank_epoch: int
    ordered_request_keys: tuple[tuple[int, int], ...]
    consumed: bool = False

    def __post_init__(self) -> None:
        BATCH_SEQ.validate(self.batch_seq)
        BANK_ID.validate(self.bank_id)
        BANK_EPOCH.validate(self.bank_epoch)
        if not isinstance(self.consumed, bool):
            raise TypeError("consumed must be a bool")
        seen_slots: set[int] = set()
        for slot, epoch in self.ordered_request_keys:
            REQUEST_SLOT.validate(slot)
            REQUEST_EPOCH.validate(epoch)
            if slot in seen_slots:
                raise ValueError("prepared batch contains duplicate request slot")
            seen_slots.add(slot)

    def mark_consumed(self) -> "PreparedBatchState":
        return PreparedBatchState(self.batch_seq, self.bank_id, self.bank_epoch, self.ordered_request_keys, consumed=True)


class WorkerCommandCache:
    """Worker-local request cache with generation and sequence validation."""

    def __init__(self, *, worker_id: int, worker_generation: int) -> None:
        WORKER_ID.validate(worker_id)
        WORKER_GENERATION.validate(worker_generation)
        self.worker_id = worker_id
        self.worker_generation = worker_generation
        self._draft_prepared = DraftPreparedContract()
        self._last_command_seq: int | None = None
        self._entries: dict[int, WorkerRequestCacheEntry] = {}
        self._prepared_batches: dict[int, PreparedBatchState] = {}

    def discard_draft_prepare(self, bank_id, bank_epoch, batch_seq):
        return self._draft_prepared.discard(bank_id, bank_epoch, batch_seq)

    def apply(self, command: HotCommand) -> None:
        self._validate_command(command)
        if isinstance(command, (PrepareDraftBankCommand, RunDraftBatchCommand)):
            self._draft_prepared.accept(command)
            self._last_command_seq = command.command_seq
            return
        next_entries = dict(self._entries)
        next_batches = dict(self._prepared_batches)
        if isinstance(command, DraftBatchCommand):
            _validate_request_lists(command.new_requests, command.cached_request_deltas, command.removed_requests)
            for row in command.new_requests:
                self._apply_new(next_entries, row)
            for row in command.cached_request_deltas:
                self._apply_delta(next_entries, row)
            for row in command.removed_requests:
                self._remove(next_entries, next_batches, row)
        elif isinstance(command, TargetPrefillBatchCommand):
            current_batch = next_batches.get(command.bank_id)
            if current_batch is not None and not current_batch.consumed:
                raise ValueError("Target prefill bank already has an unconsumed batch")
            ordered_request_keys = tuple(_request_key(row) for row in command.requests)
            for row in command.requests:
                self._apply_target_prefill(next_entries, command, row)
            next_batches[command.bank_id] = PreparedBatchState(
                batch_seq=command.batch_seq,
                bank_id=command.bank_id,
                bank_epoch=command.bank_epoch,
                ordered_request_keys=ordered_request_keys,
                consumed=True,
            )
        elif isinstance(command, PrepareTargetBankCommand):
            current_batch = next_batches.get(command.standby_bank_id)
            if current_batch is not None and not current_batch.consumed:
                raise ValueError("prepared Target bank already has an unconsumed batch")
            ordered_request_keys = tuple(_request_key(row) for row in command.requests)
            for row in command.requests:
                self._apply_target_prepare(next_entries, command, row)
            next_batches[command.standby_bank_id] = PreparedBatchState(
                batch_seq=command.batch_seq,
                bank_id=command.standby_bank_id,
                bank_epoch=command.next_bank_epoch,
                ordered_request_keys=ordered_request_keys,
            )
        else:
            prepared_batch = next_batches.get(command.active_bank_id)
            if prepared_batch is None:
                raise ValueError("RunTargetBatchCommand requires a prepared batch")
            if prepared_batch.batch_seq != command.expected_batch_seq:
                raise ValueError("RunTargetBatchCommand batch_seq does not match prepared batch")
            if prepared_batch.bank_epoch != command.active_bank_epoch:
                raise ValueError("RunTargetBatchCommand bank_epoch does not match prepared batch")
            if prepared_batch.consumed:
                raise ValueError("RunTargetBatchCommand cannot execute a consumed prepared batch")
            if tuple(_request_key(row) for row in command.requests) != prepared_batch.ordered_request_keys:
                raise ValueError("RunTargetBatchCommand requests must exactly match prepared batch order")
            for row in command.requests:
                self._apply_run(next_entries, command, row)
            next_batches[command.active_bank_id] = prepared_batch.mark_consumed()
        self._entries = next_entries
        self._prepared_batches = next_batches
        self._last_command_seq = command.command_seq

    def get(self, request_slot: int) -> WorkerRequestCacheEntry | None:
        REQUEST_SLOT.validate(request_slot)
        return self._entries.get(request_slot)

    def discard_prepared(self, *, bank_id: int, bank_epoch: int, batch_seq: int) -> None:
        """Management ACK after the whole batch's physical resources are retired."""
        batch = self._prepared_batches.get(bank_id)
        if batch is None or (batch.bank_epoch, batch.batch_seq) != (bank_epoch, batch_seq) or batch.consumed:
            raise ValueError("discard does not match an unconsumed prepared batch")
        self._prepared_batches[bank_id] = batch.mark_consumed()

    def _validate_command(self, command: HotCommand) -> None:
        if command.worker_id != self.worker_id:
            raise ValueError("command worker_id does not match cache owner")
        generation = command.worker_generation if isinstance(command, (DraftBatchCommand, PrepareDraftBankCommand, RunDraftBatchCommand)) else command.target_generation
        if generation != self.worker_generation:
            raise ValueError("command worker_generation does not match cache owner")
        if self._last_command_seq is not None and not COMMAND_SEQ.is_newer(command.command_seq, self._last_command_seq):
            raise ValueError("command_seq must increase monotonically")

    def _apply_new(
        self,
        entries: dict[int, WorkerRequestCacheEntry],
        row: NewRequestData,
        *,
        prepared_batch_seq: int | None = None,
        prepared_bank_id: int | None = None,
        prepared_bank_epoch: int | None = None,
    ) -> None:
        if row.request_slot in entries:
            raise ValueError("NewRequestData cannot overwrite an initialized worker cache entry")
        entries[row.request_slot] = WorkerRequestCacheEntry(
            request_slot=row.request_slot,
            request_epoch=row.request_epoch,
            last_round_id=row.round_id,
            last_op_seq=row.op_seq,
            input_tokens_handle=row.input_tokens_handle,
            initial_output_tokens_handle=row.initial_output_tokens_handle,
            generation_config_handle=row.generation_config_handle,
            token_delta_handle=ArenaHandle.null(),
            proposal_handle=ArenaHandle.null(),
            hostkv_handle=HostKVArenaHandle(0, 0, 0),
            target_bank_mapping_handle=ArenaHandle.null(),
            prepared_batch_seq=prepared_batch_seq,
            prepared_bank_id=prepared_bank_id,
            prepared_bank_epoch=prepared_bank_epoch,
        )

    def _apply_delta(
        self,
        entries: dict[int, WorkerRequestCacheEntry],
        row: CachedRequestDelta,
        *,
        prepared_batch_seq: int | None = None,
        prepared_bank_id: int | None = None,
        prepared_bank_epoch: int | None = None,
    ) -> None:
        current = entries.get(row.request_slot)
        if current is None:
            raise ValueError("CachedRequestDelta requires an initialized worker cache entry")
        if current.request_epoch != row.request_epoch:
            raise ValueError("CachedRequestDelta request_epoch does not match worker cache")
        if not _operation_is_newer(row.round_id, row.op_seq, current.last_round_id, current.last_op_seq):
            raise ValueError("CachedRequestDelta round/op must move forward")
        entries[row.request_slot] = WorkerRequestCacheEntry(
            request_slot=row.request_slot,
            request_epoch=row.request_epoch,
            last_round_id=row.round_id,
            last_op_seq=row.op_seq,
            input_tokens_handle=current.input_tokens_handle,
            initial_output_tokens_handle=current.initial_output_tokens_handle,
            generation_config_handle=current.generation_config_handle,
            token_delta_handle=row.token_delta_handle,
            proposal_handle=row.proposal_handle,
            hostkv_handle=row.hostkv_handle,
            target_bank_mapping_handle=row.target_bank_mapping_handle,
            prepared_batch_seq=prepared_batch_seq if prepared_batch_seq is not None else current.prepared_batch_seq,
            prepared_bank_id=prepared_bank_id if prepared_bank_id is not None else current.prepared_bank_id,
            prepared_bank_epoch=prepared_bank_epoch if prepared_bank_epoch is not None else current.prepared_bank_epoch,
        )

    def _apply_target_prefill(
        self,
        entries: dict[int, WorkerRequestCacheEntry],
        command: TargetPrefillBatchCommand,
        row: TargetPrefillRequest,
    ) -> None:
        if row.request_slot in entries:
            raise ValueError("TargetPrefillRequest cannot overwrite an initialized worker cache entry")
        entries[row.request_slot] = WorkerRequestCacheEntry(
            request_slot=row.request_slot,
            request_epoch=row.request_epoch,
            last_round_id=row.round_id,
            last_op_seq=row.run_seq,
            input_tokens_handle=row.input_tokens_handle,
            initial_output_tokens_handle=ArenaHandle.null(),
            generation_config_handle=row.generation_config_handle,
            token_delta_handle=ArenaHandle.null(),
            proposal_handle=ArenaHandle.null(),
            hostkv_handle=HostKVArenaHandle(0, 0, 0),
            target_bank_mapping_handle=ArenaHandle.null(),
            prepared_batch_seq=command.batch_seq,
            prepared_bank_id=command.bank_id,
            prepared_bank_epoch=command.bank_epoch,
        )

    def _apply_target_prepare(
        self,
        entries: dict[int, WorkerRequestCacheEntry],
        command: PrepareTargetBankCommand,
        row: TargetPrepareRequest,
    ) -> None:
        current = entries.get(row.request_slot)
        if current is None:
            # First arrival at a Target carries its own authoritative restore
            # fields. Backend-local row allocation belongs to that Target.
            entries[row.request_slot] = WorkerRequestCacheEntry(
                request_slot=row.request_slot, request_epoch=row.request_epoch,
                last_round_id=row.round_id, last_op_seq=row.op_seq,
                input_tokens_handle=ArenaHandle.null(),
                initial_output_tokens_handle=row.committed_output_handle,
                generation_config_handle=row.generation_config_handle,
                token_delta_handle=ArenaHandle.null(), proposal_handle=ArenaHandle.null(),
                hostkv_handle=row.hostkv_handle,
                target_bank_mapping_handle=ArenaHandle.null(),
                prepared_batch_seq=command.batch_seq,
                prepared_bank_id=command.standby_bank_id,
                prepared_bank_epoch=command.next_bank_epoch,
            )
            return
        if current.request_epoch != row.request_epoch:
            raise ValueError("TargetPrepareRequest request_epoch does not match worker cache")
        if not _operation_is_newer(row.round_id, row.op_seq, current.last_round_id, current.last_op_seq):
            raise ValueError("TargetPrepareRequest round/op must move forward")
        entries[row.request_slot] = WorkerRequestCacheEntry(
            request_slot=row.request_slot,
            request_epoch=row.request_epoch,
            last_round_id=row.round_id,
            last_op_seq=row.op_seq,
            input_tokens_handle=current.input_tokens_handle,
            initial_output_tokens_handle=current.initial_output_tokens_handle,
            generation_config_handle=row.generation_config_handle,
            token_delta_handle=current.token_delta_handle,
            proposal_handle=current.proposal_handle,
            hostkv_handle=row.hostkv_handle,
            target_bank_mapping_handle=current.target_bank_mapping_handle,
            prepared_batch_seq=command.batch_seq,
            prepared_bank_id=command.standby_bank_id,
            prepared_bank_epoch=command.next_bank_epoch,
        )

    def _apply_run(
        self,
        entries: dict[int, WorkerRequestCacheEntry],
        command: RunTargetBatchCommand,
        row: RunTargetRequest | TargetVerifyRequest,
    ) -> None:
        current = entries.get(row.request_slot)
        if current is None:
            raise ValueError("RunTargetRequest requires prepared worker cache entry")
        if current.request_epoch != row.request_epoch:
            raise ValueError("RunTargetRequest request_epoch does not match worker cache")
        if current.prepared_batch_seq != command.expected_batch_seq:
            raise ValueError("RunTargetRequest batch_seq does not match prepared batch")
        if current.prepared_bank_id != command.active_bank_id:
            raise ValueError("RunTargetRequest bank_id does not match prepared bank")
        if current.prepared_bank_epoch != command.active_bank_epoch:
            raise ValueError("RunTargetRequest bank_epoch does not match prepared bank")
        if not _operation_is_newer(row.round_id, row.run_seq, current.last_round_id, current.last_op_seq):
            raise ValueError("RunTargetRequest round/run_seq must move forward")
        entries[row.request_slot] = WorkerRequestCacheEntry(
            request_slot=row.request_slot,
            request_epoch=row.request_epoch,
            last_round_id=row.round_id,
            last_op_seq=row.run_seq,
            input_tokens_handle=current.input_tokens_handle,
            initial_output_tokens_handle=current.initial_output_tokens_handle,
            generation_config_handle=current.generation_config_handle,
            token_delta_handle=current.token_delta_handle,
            proposal_handle=row.proposal_handle.handle if isinstance(row, TargetVerifyRequest) else row.proposal_handle,
            hostkv_handle=current.hostkv_handle,
            target_bank_mapping_handle=current.target_bank_mapping_handle,
            prepared_batch_seq=current.prepared_batch_seq,
            prepared_bank_id=current.prepared_bank_id,
            prepared_bank_epoch=current.prepared_bank_epoch,
        )

    def _remove(
        self,
        entries: dict[int, WorkerRequestCacheEntry],
        prepared_batches: dict[int, PreparedBatchState],
        row: RemovedRequest,
    ) -> None:
        current = entries.get(row.request_slot)
        if current is not None and current.request_epoch != row.request_epoch:
            raise ValueError("RemovedRequest request_epoch does not match worker cache")
        for batch_key, prepared_batch in tuple(prepared_batches.items()):
            if (row.request_slot, row.request_epoch) in prepared_batch.ordered_request_keys and not prepared_batch.consumed:
                raise ValueError("RemovedRequest cannot remove a request from an unconsumed prepared batch")
        entries.pop(row.request_slot, None)


def _require_handle_type(handle: object, handle_type: type[object], label: str) -> None:
    if not isinstance(handle, handle_type):
        raise TypeError(f"{label} must be {handle_type.__name__}")


def _request_key(
    row: NewRequestData
    | CachedRequestDelta
    | TargetPrefillRequest
    | TargetVerifyRequest
    | TargetPrepareRequest
    | RemovedRequest
    | BankAllocation
    | HostKVSource
    | RunTargetRequest,
) -> tuple[int, int]:
    return (row.request_slot, row.request_epoch)


def _unique_keys(rows: tuple[object, ...], label: str) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    slots: set[int] = set()
    for row in rows:
        key = _request_key(row)  # type: ignore[arg-type]
        if key in keys:
            raise ValueError(f"{label} contains duplicate request slot/epoch")
        if key[0] in slots:
            raise ValueError(f"{label} contains duplicate request slot")
        keys.add(key)
        slots.add(key[0])
    return keys


def _ensure_unique_request_keys(rows: tuple[object, ...], label: str) -> None:
    _unique_keys(rows, label)


def _validate_request_lists(
    new_requests: tuple[NewRequestData, ...],
    deltas: tuple[CachedRequestDelta, ...],
    removed_requests: tuple[RemovedRequest, ...],
) -> None:
    new_keys = _unique_keys(new_requests, "new_requests")
    delta_keys = _unique_keys(deltas, "cached_request_deltas")
    removed_keys = _unique_keys(removed_requests, "removed_requests")
    new_slots = {slot for slot, _epoch in new_keys}
    delta_slots = {slot for slot, _epoch in delta_keys}
    removed_slots = {slot for slot, _epoch in removed_keys}
    if new_keys & delta_keys:
        raise ValueError("new_requests and cached_request_deltas cannot contain the same request")
    if new_slots & delta_slots:
        raise ValueError("new_requests and cached_request_deltas cannot contain the same request slot")
    if (new_keys | delta_keys) & removed_keys:
        raise ValueError("removed_requests cannot overlap active request payloads")
    if (new_slots | delta_slots) & removed_slots:
        raise ValueError("removed_requests cannot overlap active request slots")


def _operation_is_newer(round_id: int, op_seq: int, previous_round_id: int, previous_op_seq: int) -> bool:
    if ROUND_ID.is_newer(round_id, previous_round_id):
        return True
    if round_id == previous_round_id and OP_SEQ.is_newer(op_seq, previous_op_seq):
        return True
    return False


def _u8(value: int) -> bytes:
    BANK_ID.validate(value)
    return value.to_bytes(1, "little")


def _u32(value: int) -> bytes:
    U32.validate(value)
    return value.to_bytes(4, "little")


def _u64(value: int) -> bytes:
    U64.validate(value)
    return value.to_bytes(8, "little")


def _handle(handle: ArenaHandle) -> bytes:
    return _u64(handle.offset) + _u32(handle.length) + _u32(handle.generation)


def _owned_handle(handle: OwnedArenaHandle) -> bytes:
    return _u32(handle.owner_worker_id) + _u64(handle.owner_generation) + _handle(handle.handle)


def _hostkv_handle(handle: HostKVArenaHandle) -> bytes:
    return _u64(handle.offset_blocks) + _u32(handle.block_count) + _u32(handle.generation)


class _PayloadReader:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self._offset = 0

    def raw(self, size: int) -> bytes:
        self._require(size)
        value = self._raw[self._offset:self._offset + size]
        self._offset += size
        return value

    def u8(self) -> int:
        self._require(1)
        value = self._raw[self._offset]
        self._offset += 1
        return value

    def u32(self) -> int:
        self._require(4)
        value = int.from_bytes(self._raw[self._offset : self._offset + 4], "little")
        self._offset += 4
        return value

    def u64(self) -> int:
        self._require(8)
        value = int.from_bytes(self._raw[self._offset : self._offset + 8], "little")
        self._offset += 8
        return value

    def handle(self) -> ArenaHandle:
        return ArenaHandle(self.u64(), self.u32(), self.u32())

    def hostkv_handle(self) -> HostKVArenaHandle:
        return HostKVArenaHandle(self.u64(), self.u32(), self.u32())

    def finish(self) -> None:
        if self._offset != len(self._raw):
            raise ValueError("payload has trailing bytes")

    def _require(self, size: int) -> None:
        if self._offset + size > len(self._raw):
            raise ValueError("truncated command payload")


def encode_command_payload(command: HotCommand) -> bytes:
    if command.kind is CommandKind.WORK:
        return command.to_bytes()
    payload = bytearray()
    payload += _u32(COMMAND_PAYLOAD_MAGIC)
    payload += _u32(COMMAND_PROTOCOL_VERSION)
    payload += _u32(int(command.kind))

    if isinstance(command, (PrepareDraftBankCommand, RunDraftBatchCommand)):
        payload += encode_draft_command(command)
    elif isinstance(command, DraftBatchCommand):
        payload += _u32(len(command.new_requests))
        payload += _u32(len(command.cached_request_deltas))
        payload += _u32(len(command.removed_requests))
        for row in command.new_requests:
            payload += _new_request(row)
        for row in command.cached_request_deltas:
            payload += _cached_delta(row)
        for row in command.removed_requests:
            payload += _removed_request(row)
        payload += _u8(int(command.bank is not None))
        if command.bank is not None:
            payload += command.bank.to_bytes()
    elif isinstance(command, TargetPrefillBatchCommand):
        payload += _u64(command.batch_seq)
        payload += _u8(command.bank_id)
        payload += _u64(command.bank_epoch)
        payload += _u32(len(command.requests))
        for row in command.requests:
            payload += _target_prefill_request(row)
    elif isinstance(command, PrepareTargetBankCommand):
        payload += _u64(command.batch_seq)
        payload += _u8(command.standby_bank_id)
        payload += _u64(command.next_bank_epoch)
        payload += _u32(len(command.requests))
        for row in command.requests:
            payload += _target_prepare_request(row)
    elif isinstance(command, RunTargetBatchCommand):
        payload += _u64(command.expected_batch_seq)
        payload += _u8(command.active_bank_id)
        payload += _u64(command.active_bank_epoch)
        payload += _u64(command.standby_bank_epoch)
        payload += _u32(len(command.requests))
        for row in command.requests:
            payload += _run_target_request(row)
    else:
        raise TypeError("unsupported command")
    return bytes(payload)


def decode_command_payload(command_kind: CommandKind, raw: bytes, *, worker_id: int, worker_generation: int, command_seq: int) -> HotCommand:
    if command_kind == CommandKind.WORK:
        from nebulasd.workers.work import Work
        work = Work.from_bytes(raw)
        if (work.worker_id, work.worker_generation, work.work_seq) != (worker_id, worker_generation, command_seq):
            raise ValueError("WORK header identity mismatch")
        return work
    reader = _PayloadReader(raw)
    if reader.u32() != COMMAND_PAYLOAD_MAGIC:
        raise ValueError("invalid command payload magic")
    if reader.u32() != COMMAND_PROTOCOL_VERSION:
        raise ValueError("invalid command payload version")
    encoded_kind = validate_enum(CommandKind, reader.u32())
    expected_kind = validate_enum(CommandKind, command_kind)
    if encoded_kind is not expected_kind:
        raise ValueError("command payload kind does not match header")

    if expected_kind in (CommandKind.PREPARE_DRAFT_BANK, CommandKind.RUN_DRAFT_BATCH):
        command = decode_draft_command(expected_kind, reader, worker_id, worker_generation, command_seq)
    elif expected_kind is CommandKind.DRAFT_BATCH:
        new_count = reader.u32()
        delta_count = reader.u32()
        removed_count = reader.u32()
        command: HotCommand = DraftBatchCommand(
            worker_id=worker_id,
            worker_generation=worker_generation,
            command_seq=command_seq,
            new_requests=tuple(_read_new_request(reader) for _ in range(new_count)),
            cached_request_deltas=tuple(_read_cached_delta(reader) for _ in range(delta_count)),
            removed_requests=tuple(_read_removed_request(reader) for _ in range(removed_count)),
        )
        variant = reader.u8()
        if variant not in (0, 1):
            raise ValueError("invalid initial Draft Bank variant")
        if variant:
            from dataclasses import replace
            command = replace(command, bank=DraftInitialBank.read(reader))
    elif expected_kind is CommandKind.TARGET_PREFILL_BATCH:
        batch_seq = reader.u64()
        bank_id = reader.u8()
        bank_epoch = reader.u64()
        request_count = reader.u32()
        command = TargetPrefillBatchCommand(
            worker_id=worker_id,
            target_generation=worker_generation,
            command_seq=command_seq,
            batch_seq=batch_seq,
            bank_id=bank_id,
            bank_epoch=bank_epoch,
            requests=tuple(_read_target_prefill_request(reader) for _ in range(request_count)),
        )
    elif expected_kind is CommandKind.PREPARE_TARGET_BANK:
        batch_seq = reader.u64()
        standby_bank_id = reader.u8()
        next_bank_epoch = reader.u64()
        request_count = reader.u32()
        command = PrepareTargetBankCommand(
            worker_id=worker_id,
            target_generation=worker_generation,
            command_seq=command_seq,
            batch_seq=batch_seq,
            standby_bank_id=standby_bank_id,
            next_bank_epoch=next_bank_epoch,
            requests=tuple(_read_target_prepare_request(reader) for _ in range(request_count)),
        )
    else:
        expected_batch_seq = reader.u64()
        active_bank_id = reader.u8()
        active_bank_epoch = reader.u64()
        standby_bank_epoch = reader.u64()
        request_count = reader.u32()
        command = RunTargetBatchCommand(
            worker_id=worker_id,
            target_generation=worker_generation,
            command_seq=command_seq,
            expected_batch_seq=expected_batch_seq,
            active_bank_id=active_bank_id,
            active_bank_epoch=active_bank_epoch,
            standby_bank_epoch=standby_bank_epoch,
            requests=tuple(_read_run_target_request(reader) for _ in range(request_count)),
        )
    reader.finish()
    return command


def _new_request(row: NewRequestData) -> bytes:
    return (
        _u32(row.request_slot)
        + _u64(row.request_epoch)
        + _u64(row.round_id)
        + _u64(row.op_seq)
        + _u32(row.scheduled_token_count)
        + _handle(row.input_tokens_handle)
        + _handle(row.initial_output_tokens_handle)
        + _handle(row.generation_config_handle)
        + _handle(row.initial_kv_handle)
    )


def _read_new_request(reader: _PayloadReader) -> NewRequestData:
    return DraftNewRequestData(
        reader.u32(),
        reader.u64(),
        reader.u64(),
        reader.u64(),
        reader.u32(),
        reader.handle(),
        reader.handle(),
        reader.handle(),
        reader.handle(),
    )


def _target_prefill_request(row: TargetPrefillRequest) -> bytes:
    return (
        _u32(row.request_slot)
        + _u64(row.request_epoch)
        + _u64(row.round_id)
        + _u64(row.run_seq)
        + _handle(row.input_tokens_handle)
        + _handle(row.generation_config_handle)
        + _u32(row.scheduled_token_count)
        + _u32(row.max_output_len)
        + _u8(row.bank_id)
        + _u64(row.bank_epoch)
        + _u32(row.bank_offset_blocks)
        + _u32(row.block_count)
    )


def _read_target_prefill_request(reader: _PayloadReader) -> TargetPrefillRequest:
    return TargetPrefillRequest(
        reader.u32(),
        reader.u64(),
        reader.u64(),
        reader.u64(),
        reader.handle(),
        reader.handle(),
        reader.u32(),
        reader.u32(),
        reader.u8(),
        reader.u64(),
        reader.u32(),
        reader.u32(),
    )


def _target_prepare_request(row: TargetPrepareRequest) -> bytes:
    return (
        _u32(row.request_slot)
        + _u64(row.request_epoch)
        + _u64(row.round_id)
        + _u64(row.op_seq)
        + _handle(row.committed_output_handle)
        + _u32(row.committed_output_count)
        + _u32(row.prompt_token_count)
        + _handle(row.generation_config_handle)
        + _hostkv_handle(row.hostkv_handle)
        + _u32(row.host_slot)
        + _u64(row.host_slot_generation)
        + _u64(row.host_writer_lease_generation)
        + _u64(row.source_host_version)
        + _u32(row.logical_kv_len)
        + _u32(row.committed_blocks)
        + _u32(row.valid_blocks)
        + _u8(row.destination_bank_id)
        + _u64(row.destination_bank_epoch)
        + _u32(row.destination_bank_offset_blocks)
        + _u32(row.destination_capacity_blocks)
    )


def _read_target_prepare_request(reader: _PayloadReader) -> TargetPrepareRequest:
    return TargetPrepareRequest(
        reader.u32(),
        reader.u64(),
        reader.u64(),
        reader.u64(),
        reader.handle(),
        reader.u32(),
        reader.u32(),
        reader.handle(),
        reader.hostkv_handle(),
        reader.u32(),
        reader.u64(),
        reader.u64(),
        reader.u64(),
        reader.u32(),
        reader.u32(),
        reader.u32(),
        reader.u8(),
        reader.u64(),
        reader.u32(),
        reader.u32(),
    )


def _cached_delta(row: CachedRequestDelta) -> bytes:
    return (
        _u32(row.request_slot)
        + _u64(row.request_epoch)
        + _u64(row.round_id)
        + _u64(row.op_seq)
        + _handle(row.token_delta_handle)
        + _handle(row.proposal_handle)
        + _hostkv_handle(row.hostkv_handle)
        + _handle(row.target_bank_mapping_handle)
        + _u32(row.scheduled_token_count)
    )


def _read_cached_delta(reader: _PayloadReader) -> CachedRequestDelta:
    return CachedRequestDelta(
        reader.u32(),
        reader.u64(),
        reader.u64(),
        reader.u64(),
        reader.handle(),
        reader.handle(),
        reader.hostkv_handle(),
        reader.handle(),
        reader.u32(),
    )


def _removed_request(row: RemovedRequest) -> bytes:
    return _u32(row.request_slot) + _u64(row.request_epoch)


def _read_removed_request(reader: _PayloadReader) -> RemovedRequest:
    return RemovedRequest(reader.u32(), reader.u64())


def _bank_allocation(row: BankAllocation) -> bytes:
    return _u32(row.request_slot) + _u64(row.request_epoch) + _u8(row.bank_id) + _u64(row.bank_epoch) + _u32(row.offset_blocks) + _u32(row.block_count)


def _read_bank_allocation(reader: _PayloadReader) -> BankAllocation:
    return BankAllocation(reader.u32(), reader.u64(), reader.u8(), reader.u64(), reader.u32(), reader.u32())


def _hostkv_source(row: HostKVSource) -> bytes:
    return _u32(row.request_slot) + _u64(row.request_epoch) + _u32(row.host_slot) + _u64(row.host_slot_generation) + _u64(row.source_host_version)


def _read_hostkv_source(reader: _PayloadReader) -> HostKVSource:
    return HostKVSource(reader.u32(), reader.u64(), reader.u32(), reader.u64(), reader.u64())


def _run_target_request(row: RunTargetRequest | TargetVerifyRequest) -> bytes:
    if isinstance(row, TargetVerifyRequest):
        return (
            _u8(1)
            + _u32(row.request_slot)
            + _u64(row.request_epoch)
            + _u64(row.round_id)
            + _u64(row.run_seq)
            + _owned_handle(row.proposal_handle)
            + _handle(row.committed_output_handle)
            + _u32(row.committed_output_count)
            + _u32(row.prompt_token_count)
            + _handle(row.generation_config_handle)
            + _u8(row.bank_id)
            + _u64(row.bank_epoch)
            + _u32(row.bank_offset_blocks)
            + _u32(row.block_count)
        )
    return (
        _u8(0)
        + _u32(row.request_slot)
        + _u64(row.request_epoch)
        + _u64(row.round_id)
        + _u64(row.run_seq)
        + _handle(row.proposal_handle)
        + _handle(row.output_handle)
    )


def _read_run_target_request(reader: _PayloadReader) -> RunTargetRequest | TargetVerifyRequest:
    variant = reader.u8()
    if variant == 0:
        return RunTargetRequest(reader.u32(), reader.u64(), reader.u64(), reader.u64(), reader.handle(), reader.handle())
    if variant == 1:
        return TargetVerifyRequest(
            reader.u32(),
            reader.u64(),
            reader.u64(),
            reader.u64(),
            _read_owned_handle(reader),
            reader.handle(),
            reader.u32(),
            reader.u32(),
            reader.handle(),
            reader.u8(),
            reader.u64(),
            reader.u32(),
            reader.u32(),
        )
    raise ValueError("unknown RunTargetRequest variant")


def _read_owned_handle(reader: _PayloadReader) -> OwnedArenaHandle:
    return OwnedArenaHandle(reader.u32(), reader.u64(), reader.handle())
