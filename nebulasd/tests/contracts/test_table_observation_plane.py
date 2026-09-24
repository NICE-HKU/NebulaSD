"""Contracts for the scheduling observation plane."""

from __future__ import annotations

import pytest

from nebulasd.core.enums import (
    BankRole,
    BankState,
    ComputeStatus,
    Lifecycle,
    StateChangeBlockKind,
    WorkerRole,
    WorkerStatus,
)
from nebulasd.core.ids import OperationFence, RequestFence
from nebulasd.core.ids import U64
from nebulasd.ipc.doorbell import Doorbell
from nebulasd.ipc.state_change_ring import StateChangeRing
from nebulasd.table.reader import IncrementalTableReader
from nebulasd.table.layout import ENDIANNESS
from nebulasd.table.storage import FieldValue, RequestSchedulingTable, StableReadConflict, TableProtocolError
from nebulasd.table.storage import TableSegmentHeader
from nebulasd.table.storage import WorkerSchedulingRegistry
from nebulasd.table.writers import (
    DispatcherTableWriter,
    DraftWorkerTableWriter,
    EngineTableWriter,
    HostKVAllocatorWriter,
    TargetComputeTableWriter,
    TargetCopyTableWriter,
    WorkerRegistryWriter,
)


def test_request_table_header_locks_abi_layout_records() -> None:
    table = RequestSchedulingTable(8)

    assert table.header.table_kind == "request"
    assert table.header.capacity_rows == 8
    assert table.header.byte_size == table.byte_size
    table.header.validate(table.header.layouts)

    bad_header = TableSegmentHeader(
        magic=table.header.magic,
        abi_version=table.header.abi_version,
        table_kind=table.header.table_kind,
        capacity_rows=table.header.capacity_rows,
        byte_size=table.header.byte_size,
        layouts=table.header.layouts[:-1],
    )
    with pytest.raises(TableProtocolError):
        bad_header.validate(table.header.layouts)


def _seed_request(table: RequestSchedulingTable) -> OperationFence:
    EngineTableWriter(table).publish_active(
        slot=0,
        publish_seq=1,
        request_epoch=7,
        current_round_id=11,
        arrival_seq=3,
        prompt_token_count=5,
        max_new_tokens=16,
        spec_token_limit=4,
    )
    DispatcherTableWriter(table).publish_dispatch(
        slot=0,
        publish_seq=2,
        request_epoch=7,
        draft_issue_seq=101,
        draft_worker_generation=17,
        draft_round_id=11,
        target_prepare_seq=201,
        planned_target_generation=19,
        planned_bank_epoch=23,
        target_run_seq=301,
        target_round_id=11,
        draft_worker_id=2,
        planned_target_id=4,
        planned_bank_id=1,
    )
    HostKVAllocatorWriter(table).publish_allocation(
        slot=0,
        publish_seq=3,
        request_epoch=7,
        host_slot_generation=29,
        writer_lease_generation=31,
        host_slot=6,
        capacity_blocks=128,
        offset_blocks=4096,
    )
    return OperationFence(
        request=RequestFence(request_slot=0, request_epoch=7),
        round_id=11,
        op_seq=101,
        worker_id=2,
        worker_generation=17,
    )


def test_owner_api_cannot_write_another_owner_block() -> None:
    table = RequestSchedulingTable(1)

    assert not hasattr(table, "publish")
    with pytest.raises(TableProtocolError):
        table._publish_owned(
            owner="engine",
            block_kind=StateChangeBlockKind.REQUEST_DRAFT,
            row=0,
            publish_seq=1,
            fields=(FieldValue("status", 2),),
        )


def test_publish_seq_zero_is_valid_but_same_or_older_sequence_is_rejected() -> None:
    table = RequestSchedulingTable(1)
    writer = EngineTableWriter(table)

    writer.publish_active(
        slot=0,
        publish_seq=0,
        request_epoch=1,
        current_round_id=1,
        arrival_seq=1,
        prompt_token_count=1,
        max_new_tokens=8,
        spec_token_limit=2,
    )
    writer.publish_lifecycle(slot=0, publish_seq=1, request_epoch=1, lifecycle=Lifecycle.FINISHED)

    with pytest.raises(TableProtocolError):
        writer.publish_lifecycle(slot=0, publish_seq=1, request_epoch=1, lifecycle=Lifecycle.CANCELLED)
    with pytest.raises(TableProtocolError):
        writer.publish_lifecycle(slot=0, publish_seq=0, request_epoch=1, lifecycle=Lifecycle.CANCELLED)
    with pytest.raises(ValueError):
        writer.publish_lifecycle(slot=0, publish_seq=U64.invalid, request_epoch=1, lifecycle=Lifecycle.CANCELLED)


def test_stable_reader_returns_published_fields_and_doorbell_rings() -> None:
    ring = StateChangeRing(capacity=8)
    doorbell = Doorbell()
    table = RequestSchedulingTable(2, ring=ring, doorbell=doorbell)
    start_generation = doorbell.generation

    EngineTableWriter(table).publish_active(
        slot=1,
        publish_seq=1,
        request_epoch=9,
        current_round_id=1,
        arrival_seq=1,
        prompt_token_count=12,
        max_new_tokens=32,
        spec_token_limit=4,
    )

    assert doorbell.generation != start_generation
    view = table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(1)
    assert view.get("request_epoch") == 9
    assert view.get("lifecycle") == int(Lifecycle.ACTIVE)

    update = IncrementalTableReader(request_table=table, rings=(ring,)).poll()
    assert len(update.views) == 1
    assert update.views[0].block_kind is StateChangeBlockKind.REQUEST_ENGINE
    assert update.views[0].row == 1
    assert update.views[0].get("prompt_token_count") == 12


def test_duplicate_ring_notifications_are_merged_to_latest_table_fact() -> None:
    ring = StateChangeRing(capacity=8)
    table = RequestSchedulingTable(1, ring=ring)
    writer = EngineTableWriter(table)

    writer.publish_active(
        slot=0,
        publish_seq=1,
        request_epoch=1,
        current_round_id=1,
        arrival_seq=1,
        prompt_token_count=1,
        max_new_tokens=8,
        spec_token_limit=2,
    )
    writer.publish_lifecycle(slot=0, publish_seq=2, request_epoch=1, lifecycle=Lifecycle.FINISHED)

    update = IncrementalTableReader(request_table=table, rings=(ring,)).poll()
    assert len(update.views) == 1
    assert update.views[0].publish_seq == 2
    assert update.views[0].get("lifecycle") == int(Lifecycle.FINISHED)


def test_stale_request_epoch_round_issue_seq_and_worker_generation_are_rejected() -> None:
    table = RequestSchedulingTable(1)
    good_fence = _seed_request(table)
    writer = DraftWorkerTableWriter(table)

    writer.publish_ready_target(fence=good_fence, publish_seq=4, proposal_token_count=2)

    stale_epoch = OperationFence(RequestFence(0, 8), 11, 101, 2, 17)
    stale_round = OperationFence(RequestFence(0, 7), 12, 101, 2, 17)
    stale_issue = OperationFence(RequestFence(0, 7), 11, 102, 2, 17)
    stale_generation = OperationFence(RequestFence(0, 7), 11, 101, 2, 18)
    for fence in (stale_epoch, stale_round, stale_issue, stale_generation):
        with pytest.raises(TableProtocolError):
            writer.publish_ready_target(fence=fence, publish_seq=5, proposal_token_count=2)


def test_stale_target_run_bank_epoch_and_hostkv_version_are_rejected() -> None:
    table = RequestSchedulingTable(1)
    _seed_request(table)
    target_fence = OperationFence(RequestFence(0, 7), 11, 301, 4, 19)
    copy_writer = TargetCopyTableWriter(table)

    TargetComputeTableWriter(table).publish_ready_draft(
        fence=target_fence,
        publish_seq=4,
        bank_id=1,
        bank_epoch=23,
        target_kv_version=41,
        accepted_draft_count=1,
        committed_delta_count=1,
        last_committed_token=99,
        logical_kv_len=17,
        dirty_begin_block=0,
        dirty_block_count=1,
    )
    copy_writer.publish_host_ready(
        fence=target_fence,
        publish_seq=5,
        source_bank_id=1,
        source_bank_epoch=23,
        host_slot_generation=29,
        writer_version=31,
        ready_version=43,
        committed_blocks=1,
        logical_kv_len=17,
    )

    stale_run = OperationFence(RequestFence(0, 7), 11, 302, 4, 19)
    with pytest.raises(TableProtocolError):
        TargetComputeTableWriter(table).publish_ready_draft(
            fence=stale_run,
            publish_seq=6,
            bank_id=1,
            bank_epoch=23,
            target_kv_version=42,
            accepted_draft_count=1,
        committed_delta_count=1,
        last_committed_token=99,
            logical_kv_len=17,
            dirty_begin_block=0,
            dirty_block_count=1,
        )

    with pytest.raises(TableProtocolError):
        TargetComputeTableWriter(table).publish_ready_draft(
            fence=target_fence,
            publish_seq=6,
            bank_id=1,
            bank_epoch=24,
            target_kv_version=42,
            accepted_draft_count=1,
        committed_delta_count=1,
        last_committed_token=99,
            logical_kv_len=17,
            dirty_begin_block=0,
            dirty_block_count=1,
        )

    h2d_fence = OperationFence(RequestFence(0, 7), 11, 201, 4, 19)
    with pytest.raises(TableProtocolError):
        copy_writer.publish_gpu_ready(
            fence=h2d_fence,
            publish_seq=6,
            destination_bank_id=1,
            destination_bank_epoch=23,
            source_host_version=42,
            gpu_ready_version=44,
            copied_blocks=1,
        )


def test_d2h_completion_uses_target_compute_fact_after_next_prepare_overwrites_dispatch() -> None:
    table = RequestSchedulingTable(1)
    _seed_request(table)
    target_fence = OperationFence(RequestFence(0, 7), 11, 301, 4, 19)

    TargetComputeTableWriter(table).publish_ready_draft(
        fence=target_fence,
        publish_seq=4,
        bank_id=1,
        bank_epoch=23,
        target_kv_version=41,
        accepted_draft_count=1,
        committed_delta_count=1,
        last_committed_token=99,
        logical_kv_len=17,
        dirty_begin_block=0,
        dirty_block_count=1,
    )
    DispatcherTableWriter(table).publish_dispatch(
        slot=0,
        publish_seq=4,
        request_epoch=7,
        draft_issue_seq=102,
        draft_worker_generation=17,
        draft_round_id=12,
        target_prepare_seq=202,
        planned_target_generation=99,
        planned_bank_epoch=24,
        target_run_seq=302,
        target_round_id=12,
        draft_worker_id=2,
        planned_target_id=9,
        planned_bank_id=0,
    )

    TargetCopyTableWriter(table).publish_host_ready(
        fence=target_fence,
        publish_seq=5,
        source_bank_id=1,
        source_bank_epoch=23,
        host_slot_generation=29,
        writer_version=31,
        ready_version=43,
        committed_blocks=1,
        logical_kv_len=17,
    )
    d2h = table.partition(StateChangeBlockKind.REQUEST_D2H).read_stable(0)
    assert d2h.get("target_id") == 4
    assert d2h.get("source_bank_epoch") == 23


def test_h2d_next_round_can_consume_previous_round_d2h_hostkv_version() -> None:
    table = RequestSchedulingTable(1)
    _seed_request(table)
    target_fence = OperationFence(RequestFence(0, 7), 11, 301, 4, 19)
    copy_writer = TargetCopyTableWriter(table)

    TargetComputeTableWriter(table).publish_ready_draft(
        fence=target_fence,
        publish_seq=4,
        bank_id=1,
        bank_epoch=23,
        target_kv_version=41,
        accepted_draft_count=1,
        committed_delta_count=1,
        last_committed_token=99,
        logical_kv_len=17,
        dirty_begin_block=0,
        dirty_block_count=1,
    )
    copy_writer.publish_host_ready(
        fence=target_fence,
        publish_seq=5,
        source_bank_id=1,
        source_bank_epoch=23,
        host_slot_generation=29,
        writer_version=31,
        ready_version=43,
        committed_blocks=1,
        logical_kv_len=17,
    )
    DispatcherTableWriter(table).publish_dispatch(
        slot=0,
        publish_seq=4,
        request_epoch=7,
        draft_issue_seq=102,
        draft_worker_generation=17,
        draft_round_id=12,
        target_prepare_seq=202,
        planned_target_generation=19,
        planned_bank_epoch=24,
        target_run_seq=302,
        target_round_id=12,
        draft_worker_id=2,
        planned_target_id=4,
        planned_bank_id=0,
    )

    h2d_fence = OperationFence(RequestFence(0, 7), 12, 202, 4, 19)
    copy_writer.publish_gpu_ready(
        fence=h2d_fence,
        publish_seq=6,
        destination_bank_id=0,
        destination_bank_epoch=24,
        source_host_version=43,
        gpu_ready_version=44,
        copied_blocks=1,
    )

    h2d = table.partition(StateChangeBlockKind.REQUEST_H2D).read_stable(0)
    assert h2d.get("round_id") == 12
    assert h2d.get("source_host_version") == 43


def test_h2d_completion_rejects_recycled_slot_and_changed_hostkv_lease() -> None:
    table = RequestSchedulingTable(1)
    _seed_request(table)
    target_fence = OperationFence(RequestFence(0, 7), 11, 301, 4, 19)
    copy_writer = TargetCopyTableWriter(table)
    TargetComputeTableWriter(table).publish_ready_draft(
        fence=target_fence,
        publish_seq=4,
        bank_id=1,
        bank_epoch=23,
        target_kv_version=41,
        accepted_draft_count=1,
        committed_delta_count=1,
        last_committed_token=99,
        logical_kv_len=17,
        dirty_begin_block=0,
        dirty_block_count=1,
    )
    copy_writer.publish_host_ready(
        fence=target_fence,
        publish_seq=5,
        source_bank_id=1,
        source_bank_epoch=23,
        host_slot_generation=29,
        writer_version=31,
        ready_version=43,
        committed_blocks=1,
        logical_kv_len=17,
    )
    h2d_fence = OperationFence(RequestFence(0, 7), 11, 201, 4, 19)

    HostKVAllocatorWriter(table).publish_allocation(
        slot=0,
        publish_seq=4,
        request_epoch=7,
        host_slot_generation=29,
        writer_lease_generation=32,
        host_slot=6,
        capacity_blocks=128,
        offset_blocks=4096,
    )
    with pytest.raises(TableProtocolError):
        copy_writer.publish_gpu_ready(
            fence=h2d_fence,
            publish_seq=6,
            destination_bank_id=1,
            destination_bank_epoch=23,
            source_host_version=43,
            gpu_ready_version=44,
            copied_blocks=1,
        )

    EngineTableWriter(table).publish_active(
        slot=0,
        publish_seq=2,
        request_epoch=8,
        current_round_id=11,
        arrival_seq=4,
        prompt_token_count=5,
        max_new_tokens=16,
        spec_token_limit=4,
    )
    with pytest.raises(TableProtocolError):
        copy_writer.publish_gpu_ready(
            fence=h2d_fence,
            publish_seq=6,
            destination_bank_id=1,
            destination_bank_epoch=23,
            source_host_version=43,
            gpu_ready_version=44,
            copied_blocks=1,
        )


def test_stable_reader_rejects_in_progress_wire_marker() -> None:
    table = RequestSchedulingTable(1)
    EngineTableWriter(table).publish_active(
        slot=0,
        publish_seq=0,
        request_epoch=1,
        current_round_id=1,
        arrival_seq=1,
        prompt_token_count=1,
        max_new_tokens=8,
        spec_token_limit=2,
    )
    partition = table.partition(StateChangeBlockKind.REQUEST_ENGINE)
    base = partition._row_base(0)
    partition._bytes[base : base + 8] = U64.invalid.to_bytes(8, ENDIANNESS)
    try:
        with pytest.raises(StableReadConflict):
            partition.read_stable(0, max_retries=2)
    finally:
        partition._bytes[base : base + 8] = (0).to_bytes(8, ENDIANNESS)


def test_fake_producers_publish_ready_host_ready_and_gpu_ready_incrementally() -> None:
    ring = StateChangeRing(capacity=16)
    doorbell = Doorbell()
    table = RequestSchedulingTable(1, ring=ring, doorbell=doorbell)
    draft_fence = _seed_request(table)
    reader = IncrementalTableReader(request_table=table, rings=(ring,))
    reader.poll()
    start_generation = doorbell.generation

    DraftWorkerTableWriter(table).publish_ready_target(
        fence=draft_fence,
        publish_seq=4,
        proposal_token_count=2,
    )
    target_fence = OperationFence(RequestFence(0, 7), 11, 301, 4, 19)
    TargetComputeTableWriter(table).publish_ready_draft(
        fence=target_fence,
        publish_seq=5,
        bank_id=1,
        bank_epoch=23,
        target_kv_version=41,
        accepted_draft_count=1,
        committed_delta_count=1,
        last_committed_token=99,
        logical_kv_len=17,
        dirty_begin_block=0,
        dirty_block_count=1,
    )
    TargetCopyTableWriter(table).publish_host_ready(
        fence=target_fence,
        publish_seq=6,
        source_bank_id=1,
        source_bank_epoch=23,
        host_slot_generation=29,
        writer_version=31,
        ready_version=43,
        committed_blocks=1,
        logical_kv_len=17,
    )
    h2d_fence = OperationFence(RequestFence(0, 7), 11, 201, 4, 19)
    TargetCopyTableWriter(table).publish_gpu_ready(
        fence=h2d_fence,
        publish_seq=7,
        destination_bank_id=1,
        destination_bank_epoch=23,
        source_host_version=43,
        gpu_ready_version=44,
        copied_blocks=1,
    )

    assert doorbell.generation != start_generation
    update = reader.poll()
    assert {view.block_kind for view in update.views} == {
        StateChangeBlockKind.REQUEST_DRAFT,
        StateChangeBlockKind.REQUEST_TARGET_COMPUTE,
        StateChangeBlockKind.REQUEST_D2H,
        StateChangeBlockKind.REQUEST_H2D,
    }
    assert reader.cached_view(StateChangeBlockKind.REQUEST_DRAFT, 0).get("status") == 2  # type: ignore[union-attr]
    assert reader.cached_view(StateChangeBlockKind.REQUEST_D2H, 0).get("ready_version") == 43  # type: ignore[union-attr]
    assert reader.cached_view(StateChangeBlockKind.REQUEST_H2D, 0).get("gpu_ready_version") == 44  # type: ignore[union-attr]


def test_worker_registry_publishes_common_runtime_and_bank_views() -> None:
    ring = StateChangeRing(capacity=8)
    registry = WorkerSchedulingRegistry(2, ring=ring)
    writer = WorkerRegistryWriter(registry)

    writer.publish_common(
        worker_row=1,
        publish_seq=1,
        worker_id=10,
        role=WorkerRole.TARGET,
        worker_generation=12,
        status=WorkerStatus.ONLINE,
        command_consumer_seq=33,
        max_batch_size=8,
        max_batch_tokens=512,
    )
    writer.publish_target_compute_runtime(
        worker_row=1,
        publish_seq=2,
        worker_generation=12,
        compute_batch_seq=44,
        compute_status=ComputeStatus.RUNNING,
        compute_start_time_ns=1000,
        compute_request_count=4,
        compute_token_count=128,
    )
    writer.publish_bank(
        worker_row=1,
        bank_index=0,
        publish_seq=3,
        worker_generation=12,
        bank_id=0,
        bank_epoch=55,
        role=BankRole.ACTIVE,
        state=BankState.COMPUTING,
        batch_seq=44,
        capacity_blocks=1024,
        alloc_ptr_blocks=64,
        capacity_rows=16,
        alloc_rows=4,
    )

    update = IncrementalTableReader(worker_registry=registry, rings=(ring,)).poll()
    assert {view.block_kind for view in update.views} == {
        StateChangeBlockKind.WORKER_COMMON,
        StateChangeBlockKind.WORKER_TARGET_COMPUTE_RUNTIME,
        StateChangeBlockKind.WORKER_BANK,
    }
    assert registry.header.table_kind == "worker"
    assert registry.header.capacity_rows == 2


def test_worker_runtime_validates_role_generation_bank_mapping_and_enums() -> None:
    registry = WorkerSchedulingRegistry(2)
    writer = WorkerRegistryWriter(registry)
    assert not hasattr(registry, "publish")

    writer.publish_common(
        worker_row=0,
        publish_seq=0,
        worker_id=10,
        role=WorkerRole.DRAFT,
        worker_generation=12,
        status=WorkerStatus.ONLINE,
        command_consumer_seq=33,
        max_batch_size=8,
        max_batch_tokens=512,
    )
    with pytest.raises(TableProtocolError):
        writer.publish_target_copy_runtime(
            worker_row=0,
            publish_seq=0,
            worker_generation=12,
            copy_op_seq=1,
            copy_status=1,
            copy_start_time_ns=1000,
            copy_bytes=4096,
        )
    with pytest.raises(TableProtocolError):
        writer.publish_draft_runtime(
            worker_row=0,
            publish_seq=0,
            worker_generation=13,
            current_batch_seq=1,
            compute_status=ComputeStatus.RUNNING,
            compute_start_time_ns=1000,
            batch_request_count=1,
            batch_token_count=4,
        )
    with pytest.raises(ValueError):
        writer.publish_draft_runtime(
            worker_row=0,
            publish_seq=0,
            worker_generation=12,
            current_batch_seq=1,
            compute_status=999,  # type: ignore[arg-type]
            compute_start_time_ns=1000,
            batch_request_count=1,
            batch_token_count=4,
        )
    with pytest.raises(TableProtocolError):
        registry._publish_owned(
            owner="worker",
            block_kind=StateChangeBlockKind.WORKER_TARGET_COPY_RUNTIME,
            row=0,
            publish_seq=0,
            fields=(FieldValue("copy_status", 999),),
        )

    writer.publish_common(
        worker_row=1,
        publish_seq=0,
        worker_id=11,
        role=WorkerRole.TARGET,
        worker_generation=20,
        status=WorkerStatus.ONLINE,
        command_consumer_seq=34,
        max_batch_size=8,
        max_batch_tokens=512,
    )
    with pytest.raises(ValueError):
        writer.publish_target_copy_runtime(
            worker_row=1,
            publish_seq=0,
            worker_generation=20,
            copy_op_seq=1,
            copy_status=999,  # type: ignore[arg-type]
            copy_start_time_ns=1000,
            copy_bytes=4096,
        )
    with pytest.raises(TableProtocolError):
        writer.publish_bank(
            worker_row=1,
            bank_index=2,
            publish_seq=0,
            worker_generation=20,
            bank_id=0,
            bank_epoch=1,
            role=BankRole.ACTIVE,
            state=BankState.READY,
            batch_seq=1,
            capacity_blocks=1024,
            alloc_ptr_blocks=0,
            capacity_rows=16,
            alloc_rows=0,
        )
    with pytest.raises(TableProtocolError):
        writer.publish_bank(
            worker_row=1,
            bank_index=0,
            publish_seq=0,
            worker_generation=20,
            bank_id=1,
            bank_epoch=1,
            role=BankRole.ACTIVE,
            state=BankState.READY,
            batch_seq=1,
            capacity_blocks=1024,
            alloc_ptr_blocks=0,
            capacity_rows=16,
            alloc_rows=0,
        )
    with pytest.raises(ValueError):
        writer.publish_bank(
            worker_row=1,
            bank_index=1,
            publish_seq=0,
            worker_generation=20,
            bank_id=255,
            bank_epoch=1,
            role=BankRole.STANDBY,
            state=BankState.READY,
            batch_seq=1,
            capacity_blocks=1024,
            alloc_ptr_blocks=0,
            capacity_rows=16,
            alloc_rows=0,
        )


def test_typed_bank_id_fields_reject_invalid_sentinel() -> None:
    table = RequestSchedulingTable(1)
    EngineTableWriter(table).publish_active(
        slot=0,
        publish_seq=0,
        request_epoch=7,
        current_round_id=11,
        arrival_seq=3,
        prompt_token_count=5,
        max_new_tokens=16,
        spec_token_limit=4,
    )

    with pytest.raises(ValueError):
        DispatcherTableWriter(table).publish_dispatch(
            slot=0,
            publish_seq=0,
            request_epoch=7,
            draft_issue_seq=101,
            draft_worker_generation=17,
            draft_round_id=11,
            target_prepare_seq=201,
            planned_target_generation=19,
            planned_bank_epoch=23,
            target_run_seq=301,
            target_round_id=11,
            draft_worker_id=2,
            planned_target_id=4,
            planned_bank_id=255,
        )
