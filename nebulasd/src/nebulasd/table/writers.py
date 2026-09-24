"""Owner-specific typed writers for scheduling observation tables."""

from __future__ import annotations

from nebulasd.core.enums import (
    BankRole,
    BankState,
    ComputeStatus,
    CopyStatus,
    D2HStatus,
    DraftStatus,
    H2DStatus,
    Lifecycle,
    StateChangeBlockKind,
    TargetStatus,
    WorkerRole,
    WorkerStatus,
    validate_enum,
)
from nebulasd.core.errors import ResultCode
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import BANK_ID, OperationFence, RequestFence

from .storage import FieldValue, RequestSchedulingTable, TableProtocolError, WorkerSchedulingRegistry


def _enum_value(enum_type: type, value: int) -> int:
    return int(validate_enum(enum_type, value))


def _expect(snapshot_value: int | ArenaHandle, expected: int, message: str) -> None:
    if not isinstance(snapshot_value, int) or snapshot_value != expected:
        raise TableProtocolError(message)


class EngineTableWriter:
    def __init__(self, table: RequestSchedulingTable) -> None:
        self._table = table

    def publish_active(
        self,
        *,
        slot: int,
        publish_seq: int,
        request_epoch: int,
        current_round_id: int,
        arrival_seq: int,
        prompt_token_count: int,
        max_new_tokens: int,
        spec_token_limit: int,
        input_tokens_handle: ArenaHandle | None = None,
        generation_config_handle: ArenaHandle | None = None,
        lifecycle: Lifecycle = Lifecycle.ACTIVE,
        classified_result_ticket: int = 0,
    ) -> None:
        self._table._publish_owned(
            owner="engine",
            block_kind=StateChangeBlockKind.REQUEST_ENGINE,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", request_epoch),
                FieldValue("current_round_id", current_round_id),
                FieldValue("classified_result_ticket", classified_result_ticket),
                FieldValue("arrival_seq", arrival_seq),
                FieldValue("lifecycle", _enum_value(Lifecycle, lifecycle)),
                FieldValue("prompt_token_count", prompt_token_count),
                FieldValue("max_new_tokens", max_new_tokens),
                FieldValue("spec_token_limit", spec_token_limit),
                FieldValue("input_tokens_handle", input_tokens_handle or ArenaHandle.null()),
                FieldValue("generation_config_handle", generation_config_handle or ArenaHandle.null()),
            ),
        )

    def publish_lifecycle(self, *, slot: int, publish_seq: int, request_epoch: int, lifecycle: Lifecycle) -> None:
        current = self._table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(slot, include_cold=False)
        _expect(current.get("request_epoch"), request_epoch, "stale request_epoch for lifecycle update")
        self._table._publish_owned(
            owner="engine",
            block_kind=StateChangeBlockKind.REQUEST_ENGINE,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", request_epoch),
                FieldValue("lifecycle", _enum_value(Lifecycle, lifecycle)),
            ),
        )


class HostKVAllocatorWriter:
    def __init__(self, table: RequestSchedulingTable) -> None:
        self._table = table

    def publish_allocation(
        self,
        *,
        slot: int,
        publish_seq: int,
        request_epoch: int,
        host_slot_generation: int,
        writer_lease_generation: int,
        host_slot: int,
        capacity_blocks: int,
        offset_blocks: int,
    ) -> None:
        engine = self._table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(slot, include_cold=False)
        _expect(engine.get("request_epoch"), request_epoch, "stale request_epoch for HostKV allocation")
        self._table._publish_owned(
            owner="engine_hostkv_allocator",
            block_kind=StateChangeBlockKind.REQUEST_HOSTKV,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", request_epoch),
                FieldValue("host_slot_generation", host_slot_generation),
                FieldValue("writer_lease_generation", writer_lease_generation),
                FieldValue("host_slot", host_slot),
                FieldValue("capacity_blocks", capacity_blocks),
                FieldValue("offset_blocks", offset_blocks),
            ),
        )


class DispatcherTableWriter:
    def next_publish_seq(self, slot):
        from .draft_fences import next_seq
        return next_seq(self._table, StateChangeBlockKind.REQUEST_DISPATCH, slot)

    def __init__(self, table: RequestSchedulingTable) -> None:
        self._table = table

    def publish_dispatch(
        self,
        *,
        slot: int,
        publish_seq: int,
        request_epoch: int,
        draft_issue_seq: int,
        draft_worker_generation: int,
        draft_round_id: int,
        target_prepare_seq: int,
        planned_target_generation: int,
        planned_bank_epoch: int,
        target_run_seq: int,
        target_round_id: int,
        draft_worker_id: int,
        planned_target_id: int,
        planned_bank_id: int,
    ) -> None:
        BANK_ID.validate(planned_bank_id)
        engine = self._table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(slot, include_cold=False)
        _expect(engine.get("request_epoch"), request_epoch, "stale request_epoch for dispatch")
        self._publish_owned(
            owner="dispatcher",
            block_kind=StateChangeBlockKind.REQUEST_DISPATCH,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", request_epoch),
                FieldValue("draft_issue_seq", draft_issue_seq),
                FieldValue("draft_worker_generation", draft_worker_generation),
                FieldValue("draft_round_id", draft_round_id),
                FieldValue("target_prepare_seq", target_prepare_seq),
                FieldValue("planned_target_generation", planned_target_generation),
                FieldValue("planned_bank_epoch", planned_bank_epoch),
                FieldValue("target_run_seq", target_run_seq),
                FieldValue("target_round_id", target_round_id),
                FieldValue("draft_worker_id", draft_worker_id),
                FieldValue("planned_target_id", planned_target_id),
                FieldValue("planned_bank_id", planned_bank_id),
            ),
        )

    def publish_draft_command_sent(
        self,
        *,
        slot: int,
        publish_seq: int,
        request_epoch: int,
        draft_issue_seq: int,
        draft_worker_generation: int,
        draft_round_id: int,
        draft_worker_id: int,
        bank=None,
    ) -> None:
        self._validate_request(slot=slot, request_epoch=request_epoch)
        self._publish_owned(
            owner="dispatcher",
            block_kind=StateChangeBlockKind.REQUEST_DISPATCH,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", request_epoch),
                FieldValue("draft_issue_seq", draft_issue_seq),
                FieldValue("draft_worker_generation", draft_worker_generation),
                FieldValue("draft_round_id", draft_round_id),
                FieldValue("draft_worker_id", draft_worker_id),
                *((FieldValue("draft_owner_epoch", 0),
                   FieldValue("draft_source_snapshot_version", 0),
                   FieldValue("draft_run_bank_id", bank.bank_id),
                   FieldValue("draft_run_bank_epoch", bank.bank_epoch),
                   FieldValue("draft_run_batch_seq", bank.batch_seq)) if bank is not None else ()),
            ),
        )

    def publish_prepare_command_sent(
        self,
        *,
        slot: int,
        publish_seq: int,
        request_epoch: int,
        target_prepare_seq: int,
        planned_target_generation: int,
        planned_bank_epoch: int,
        planned_target_id: int,
        planned_bank_id: int,
    ) -> None:
        BANK_ID.validate(planned_bank_id)
        self._validate_request(slot=slot, request_epoch=request_epoch)
        self._publish_owned(
            owner="dispatcher",
            block_kind=StateChangeBlockKind.REQUEST_DISPATCH,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", request_epoch),
                FieldValue("target_prepare_seq", target_prepare_seq),
                FieldValue("planned_target_generation", planned_target_generation),
                FieldValue("planned_bank_epoch", planned_bank_epoch),
                FieldValue("planned_target_id", planned_target_id),
                FieldValue("planned_bank_id", planned_bank_id),
            ),
        )

    def publish_run_command_sent(
        self,
        *,
        slot: int,
        publish_seq: int,
        request_epoch: int,
        target_run_seq: int,
        target_round_id: int,
        planned_target_generation: int,
        planned_bank_epoch: int,
        planned_target_id: int,
        planned_bank_id: int,
    ) -> None:
        BANK_ID.validate(planned_bank_id)
        self._validate_request(slot=slot, request_epoch=request_epoch)
        self._publish_owned(
            owner="dispatcher",
            block_kind=StateChangeBlockKind.REQUEST_DISPATCH,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", request_epoch),
                FieldValue("target_run_seq", target_run_seq),
                FieldValue("target_round_id", target_round_id),
                FieldValue("planned_target_generation", planned_target_generation),
                FieldValue("planned_bank_epoch", planned_bank_epoch),
                FieldValue("planned_target_id", planned_target_id),
                FieldValue("planned_bank_id", planned_bank_id),
            ),
        )

    def _publish_owned(self, **kwargs):
        from .draft_fences import reset_dispatch_fields_for_epoch
        kwargs['fields'] = reset_dispatch_fields_for_epoch(self._table, kwargs['row'], kwargs['fields'])
        self._table._publish_owned(**kwargs)

    def _validate_request(self, *, slot: int, request_epoch: int) -> None:
        engine = self._table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(slot, include_cold=False)
        _expect(engine.get("request_epoch"), request_epoch, "stale request_epoch for dispatch")


class DraftWorkerTableWriter:
    def next_publish_seq(self, slot):
        from .draft_fences import next_seq
        return next_seq(self._table, StateChangeBlockKind.REQUEST_DRAFT, slot)

    def __init__(self, table: RequestSchedulingTable) -> None:
        self._table = table

    def publish_in_draft(
        self,
        *,
        fence: OperationFence,
        publish_seq: int,
        result_code: ResultCode = ResultCode.OK,
    ) -> None:
        slot = fence.request.request_slot
        self._validate_fence(fence)
        self._table._publish_owned(
            owner="draft_worker",
            block_kind=StateChangeBlockKind.REQUEST_DRAFT,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", fence.request.request_epoch),
                FieldValue("round_id", fence.round_id),
                FieldValue("observed_issue_seq", fence.op_seq),
                FieldValue("worker_generation", fence.worker_generation),
                FieldValue("worker_id", fence.worker_id),
                FieldValue("status", _enum_value(DraftStatus, DraftStatus.IN_DRAFT)),
                FieldValue("result_code", _enum_value(ResultCode, result_code)),
                FieldValue("proposal_token_count", 0),
                FieldValue("proposal_handle", ArenaHandle.null()),
                FieldValue("draft_state_handle", ArenaHandle.null()),
            ),
        )

    def validate_fence(self, fence: OperationFence) -> None:
        self._validate_fence(fence)

    def publish_ready_target(
        self,
        *,
        fence: OperationFence,
        publish_seq: int,
        proposal_token_count: int,
        proposal_handle: ArenaHandle | None = None,
        draft_state_handle: ArenaHandle | None = None,
        result_code: ResultCode = ResultCode.OK,
    ) -> None:
        slot = fence.request.request_slot
        self._validate_fence(fence)
        self._table._publish_owned(
            owner="draft_worker",
            block_kind=StateChangeBlockKind.REQUEST_DRAFT,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", fence.request.request_epoch),
                FieldValue("round_id", fence.round_id),
                FieldValue("observed_issue_seq", fence.op_seq),
                FieldValue("worker_generation", fence.worker_generation),
                FieldValue("worker_id", fence.worker_id),
                FieldValue("status", _enum_value(DraftStatus, DraftStatus.READY_TARGET)),
                FieldValue("result_code", _enum_value(ResultCode, result_code)),
                FieldValue("proposal_token_count", proposal_token_count),
                FieldValue("proposal_handle", proposal_handle or ArenaHandle.null()),
                FieldValue("draft_state_handle", draft_state_handle or ArenaHandle.null()),
            ),
        )

    def _validate_fence(self, fence: OperationFence) -> None:
        engine = self._table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(
            fence.request.request_slot,
            include_cold=False,
        )
        dispatch = self._table.partition(StateChangeBlockKind.REQUEST_DISPATCH).read_stable(
            fence.request.request_slot,
            include_cold=False,
        )
        _expect(engine.get("request_epoch"), fence.request.request_epoch, "stale request_epoch for Draft fact")
        _expect(dispatch.get("draft_owner_epoch"), 0, "legacy Draft writer cannot publish a migrated owner")
        _expect(dispatch.get("draft_round_id"), fence.round_id, "stale round_id for Draft fact")
        _expect(dispatch.get("draft_issue_seq"), fence.op_seq, "stale draft_issue_seq for Draft fact")
        _expect(dispatch.get("draft_worker_id"), fence.worker_id, "wrong Draft worker_id for Draft fact")
        _expect(dispatch.get("draft_worker_generation"), fence.worker_generation, "stale Draft worker_generation")


class TargetComputeTableWriter:
    def next_publish_seq(self, slot: int) -> int:
        """A migrated request keeps its publication order across Target writers."""
        from nebulasd.core.ids import OP_SEQ, U64
        current = self._table.partition(StateChangeBlockKind.REQUEST_TARGET_COMPUTE).read_publish_seq(slot)
        return 0 if current == U64.invalid else OP_SEQ.next(current)

    def __init__(self, table: RequestSchedulingTable) -> None:
        self._table = table

    def publish_ready_draft(
        self,
        *,
        fence: OperationFence,
        publish_seq: int,
        bank_id: int,
        bank_epoch: int,
        target_kv_version: int,
        accepted_draft_count: int | None = None,
        committed_delta_count: int | None = None,
        last_committed_token: int | None = None,
        logical_kv_len: int,
        dirty_begin_block: int,
        dirty_block_count: int,
        committed_delta_handle: ArenaHandle | None = None,
        result_code: ResultCode = ResultCode.OK,
        accepted_token_count: int | None = None,
        sampled_token: int | None = None,
        accepted_tokens_handle: ArenaHandle | None = None,
    ) -> None:
        slot = fence.request.request_slot
        self._validate_fence(fence, bank_id=bank_id, bank_epoch=bank_epoch)
        if accepted_draft_count is None:
            accepted_draft_count = 0 if accepted_token_count is None else accepted_token_count
        if committed_delta_count is None:
            committed_delta_count = accepted_draft_count
        if last_committed_token is None:
            if sampled_token is None:
                raise TypeError("last_committed_token is required")
            last_committed_token = sampled_token
        if committed_delta_handle is None:
            committed_delta_handle = accepted_tokens_handle
        self._table._publish_owned(
            owner="target_compute_lane",
            block_kind=StateChangeBlockKind.REQUEST_TARGET_COMPUTE,
            row=slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", fence.request.request_epoch),
                FieldValue("round_id", fence.round_id),
                FieldValue("observed_run_seq", fence.op_seq),
                FieldValue("target_generation", fence.worker_generation),
                FieldValue("bank_epoch", bank_epoch),
                FieldValue("target_kv_version", target_kv_version),
                FieldValue("target_id", fence.worker_id),
                FieldValue("status", _enum_value(TargetStatus, TargetStatus.READY_DRAFT)),
                FieldValue("result_code", _enum_value(ResultCode, result_code)),
                FieldValue("bank_id", bank_id),
                FieldValue("accepted_draft_count", accepted_draft_count),
                FieldValue("committed_delta_count", committed_delta_count),
                FieldValue("last_committed_token", last_committed_token),
                FieldValue("logical_kv_len", logical_kv_len),
                FieldValue("dirty_begin_block", dirty_begin_block),
                FieldValue("dirty_block_count", dirty_block_count),
                FieldValue("committed_delta_handle", committed_delta_handle or ArenaHandle.null()),
            ),
        )

    def _validate_fence(self, fence: OperationFence, *, bank_id: int, bank_epoch: int) -> None:
        BANK_ID.validate(bank_id)
        engine = self._table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(
            fence.request.request_slot,
            include_cold=False,
        )
        dispatch = self._table.partition(StateChangeBlockKind.REQUEST_DISPATCH).read_stable(
            fence.request.request_slot,
            include_cold=False,
        )
        _expect(engine.get("request_epoch"), fence.request.request_epoch, "stale request_epoch for Target fact")
        _expect(dispatch.get("target_round_id"), fence.round_id, "stale round_id for Target fact")
        _expect(dispatch.get("target_run_seq"), fence.op_seq, "stale target_run_seq for Target fact")
        _expect(dispatch.get("planned_target_id"), fence.worker_id, "wrong Target worker_id for Target fact")
        _expect(dispatch.get("planned_target_generation"), fence.worker_generation, "stale Target worker_generation")
        _expect(dispatch.get("planned_bank_id"), bank_id, "wrong Target bank_id for Target fact")
        _expect(dispatch.get("planned_bank_epoch"), bank_epoch, "stale Target bank_epoch")



def request_fence(slot: int, request_epoch: int) -> RequestFence:
    return RequestFence(request_slot=slot, request_epoch=request_epoch)


class WorkerRegistryWriter:
    def __init__(self, registry: WorkerSchedulingRegistry) -> None:
        self._registry = registry

    def publish_common(
        self,
        *,
        worker_row: int,
        publish_seq: int,
        worker_id: int,
        role: WorkerRole,
        worker_generation: int,
        status: WorkerStatus,
        command_consumer_seq: int,
        max_batch_size: int,
        max_batch_tokens: int,
    ) -> None:
        self._registry._publish_owned(
            owner="worker",
            block_kind=StateChangeBlockKind.WORKER_COMMON,
            row=worker_row,
            publish_seq=publish_seq,
            fields=(
                FieldValue("worker_id", worker_id),
                FieldValue("role", _enum_value(WorkerRole, role)),
                FieldValue("worker_generation", worker_generation),
                FieldValue("status", _enum_value(WorkerStatus, status)),
                FieldValue("command_consumer_seq", command_consumer_seq),
                FieldValue("max_batch_size", max_batch_size),
                FieldValue("max_batch_tokens", max_batch_tokens),
            ),
        )

    def publish_draft_runtime(
        self,
        *,
        worker_row: int,
        publish_seq: int,
        worker_generation: int,
        current_batch_seq: int,
        compute_status: ComputeStatus,
        compute_start_time_ns: int,
        batch_request_count: int,
        batch_token_count: int,
    ) -> None:
        self._validate_worker(worker_row, role=WorkerRole.DRAFT, worker_generation=worker_generation)
        self._registry._publish_owned(
            owner="draft_worker",
            block_kind=StateChangeBlockKind.WORKER_DRAFT_RUNTIME,
            row=worker_row,
            publish_seq=publish_seq,
            fields=(
                FieldValue("current_batch_seq", current_batch_seq),
                FieldValue("compute_status", _enum_value(ComputeStatus, compute_status)),
                FieldValue("compute_start_time_ns", compute_start_time_ns),
                FieldValue("batch_request_count", batch_request_count),
                FieldValue("batch_token_count", batch_token_count),
            ),
        )

    def publish_target_compute_runtime(
        self,
        *,
        worker_row: int,
        publish_seq: int,
        worker_generation: int,
        compute_batch_seq: int,
        compute_status: ComputeStatus,
        compute_start_time_ns: int,
        compute_request_count: int,
        compute_token_count: int,
    ) -> None:
        self._validate_worker(worker_row, role=WorkerRole.TARGET, worker_generation=worker_generation)
        self._registry._publish_owned(
            owner="target_compute_lane",
            block_kind=StateChangeBlockKind.WORKER_TARGET_COMPUTE_RUNTIME,
            row=worker_row,
            publish_seq=publish_seq,
            fields=(
                FieldValue("compute_batch_seq", compute_batch_seq),
                FieldValue("compute_status", _enum_value(ComputeStatus, compute_status)),
                FieldValue("compute_start_time_ns", compute_start_time_ns),
                FieldValue("compute_request_count", compute_request_count),
                FieldValue("compute_token_count", compute_token_count),
            ),
        )

    def publish_target_copy_runtime(
        self,
        *,
        worker_row: int,
        publish_seq: int,
        worker_generation: int,
        copy_op_seq: int,
        copy_status: CopyStatus,
        copy_start_time_ns: int,
        copy_bytes: int,
    ) -> None:
        self._validate_worker(worker_row, role=WorkerRole.TARGET, worker_generation=worker_generation)
        self._registry._publish_owned(
            owner="target_copy_lane",
            block_kind=StateChangeBlockKind.WORKER_TARGET_COPY_RUNTIME,
            row=worker_row,
            publish_seq=publish_seq,
            fields=(
                FieldValue("copy_op_seq", copy_op_seq),
                FieldValue("copy_status", _enum_value(CopyStatus, copy_status)),
                FieldValue("copy_start_time_ns", copy_start_time_ns),
                FieldValue("copy_bytes", copy_bytes),
            ),
        )

    def publish_bank(
        self,
        *,
        worker_row: int,
        bank_index: int,
        publish_seq: int,
        worker_generation: int,
        bank_id: int,
        bank_epoch: int,
        role: BankRole,
        state: BankState,
        batch_seq: int,
        capacity_blocks: int,
        alloc_ptr_blocks: int,
        capacity_rows: int,
        alloc_rows: int,
    ) -> None:
        self._validate_worker(worker_row, role=WorkerRole.TARGET, worker_generation=worker_generation)
        BANK_ID.validate(bank_index)
        BANK_ID.validate(bank_id)
        if bank_index >= self._registry.bank_rows_per_worker:
            raise TableProtocolError("bank_index outside worker bank segment")
        if bank_id != bank_index:
            raise TableProtocolError("bank_id must match fixed bank_index")
        bank_row = worker_row * self._registry.bank_rows_per_worker + bank_index
        self._registry._publish_owned(
            owner="target_copy_lane",
            block_kind=StateChangeBlockKind.WORKER_BANK,
            row=bank_row,
            publish_seq=publish_seq,
            fields=(
                FieldValue("bank_id", bank_id),
                FieldValue("bank_epoch", bank_epoch),
                FieldValue("role", _enum_value(BankRole, role)),
                FieldValue("state", _enum_value(BankState, state)),
                FieldValue("batch_seq", batch_seq),
                FieldValue("capacity_blocks", capacity_blocks),
                FieldValue("alloc_ptr_blocks", alloc_ptr_blocks),
                FieldValue("capacity_rows", capacity_rows),
                FieldValue("alloc_rows", alloc_rows),
            ),
        )

    def _validate_worker(self, worker_row: int, *, role: WorkerRole, worker_generation: int) -> None:
        common = self._registry.partition(StateChangeBlockKind.WORKER_COMMON).read_stable(worker_row, include_cold=False)
        _expect(common.get("role"), _enum_value(WorkerRole, role), "wrong worker role for runtime publication")
        _expect(common.get("worker_generation"), worker_generation, "stale worker_generation for runtime publication")

# Public compatibility import; copy publication has its own owner module.
from .copy_writers import TargetCopyTableWriter

from .draft_writers import (DraftHostKVAllocatorWriter, DraftMigrationComputeWriter,
                            DraftSourceCopyWriter, DraftDestinationCopyWriter)
from .draft_registry import DraftRegistryWriter
