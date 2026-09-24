"""Contracts for mechanical Scheduler decision dispatch."""

from __future__ import annotations

import pytest

from nebulasd.core.handles import ArenaHandle, HostKVArenaHandle
from nebulasd.ipc.command_arena import CommandArena, CommandBackpressure
from nebulasd.ipc.command_ring import CommandRing
from nebulasd.ipc.protocol import (
    BankAllocation,
    CachedRequestDelta,
    DraftBatchCommand,
    HostKVSource,
    NewRequestData,
    PrepareTargetBankCommand,
    RunTargetBatchCommand,
    RunTargetRequest,
    TargetPrefillBatchCommand,
    TargetPrefillRequest,
    TargetPrepareRequest,
)
from nebulasd.scheduler.commands import DispatchPlane, DispatchPlanePoisoned, WorkerCommandEndpoint
from nebulasd.table.storage import RequestSchedulingTable, StableReadConflict, TableProtocolError
from nebulasd.table.writers import EngineTableWriter, TargetComputeTableWriter
from nebulasd.core.enums import StateChangeBlockKind
from nebulasd.core.ids import OperationFence, RequestFence


def _handle(offset: int, length: int = 4) -> ArenaHandle:
    return ArenaHandle(offset=offset, length=length, generation=1)


def _hostkv_handle(offset: int, block_count: int = 4) -> HostKVArenaHandle:
    return HostKVArenaHandle(offset_blocks=offset, block_count=block_count, generation=1)


def _delta(slot: int, *, round_id: int = 5, op_seq: int = 7) -> CachedRequestDelta:
    return CachedRequestDelta(
        request_slot=slot,
        request_epoch=11,
        round_id=round_id,
        op_seq=op_seq,
        token_delta_handle=_handle(10),
        proposal_handle=_handle(20),
        hostkv_handle=_hostkv_handle(30),
        target_bank_mapping_handle=_handle(40),
        scheduled_token_count=3,
    )


def _new(slot: int, *, round_id: int = 5, op_seq: int = 7) -> NewRequestData:
    return NewRequestData(
        request_slot=slot,
        request_epoch=11,
        round_id=round_id,
        op_seq=op_seq,
        scheduled_token_count=3,
        input_tokens_handle=_handle(1),
        initial_output_tokens_handle=_handle(2),
        generation_config_handle=_handle(3),
        initial_kv_handle=_handle(4),
    )


def _target_prefill(slot: int, *, round_id: int = 0, run_seq: int = 6) -> TargetPrefillRequest:
    return TargetPrefillRequest(
        request_slot=slot,
        request_epoch=11,
        round_id=round_id,
        run_seq=run_seq,
        input_tokens_handle=_handle(100),
        generation_config_handle=_handle(120),
        scheduled_token_count=1,
        max_output_len=16,
        bank_id=1,
        bank_epoch=21,
        bank_offset_blocks=0,
        block_count=4,
    )


def _target_prepare(slot: int, *, round_id: int = 6, op_seq: int = 8) -> TargetPrepareRequest:
    return TargetPrepareRequest(
        request_slot=slot,
        request_epoch=11,
        round_id=round_id,
        op_seq=op_seq,
        committed_output_handle=_handle(60),
        committed_output_count=1,
        prompt_token_count=8,
        generation_config_handle=_handle(70),
        hostkv_handle=_hostkv_handle(30),
        host_slot=5,
        host_slot_generation=23,
        host_writer_lease_generation=33,
        source_host_version=31,
        logical_kv_len=8,
        committed_blocks=1,
        valid_blocks=1,
        destination_bank_id=1,
        destination_bank_epoch=21,
        destination_bank_offset_blocks=0,
        destination_capacity_blocks=4,
    )


def _table() -> RequestSchedulingTable:
    table = RequestSchedulingTable(1)
    EngineTableWriter(table).publish_active(
        slot=0,
        publish_seq=0,
        request_epoch=11,
        current_round_id=5,
        arrival_seq=1,
        prompt_token_count=8,
        max_new_tokens=16,
        spec_token_limit=4,
    )
    return table


def _two_slot_table() -> RequestSchedulingTable:
    table = RequestSchedulingTable(2)
    writer = EngineTableWriter(table)
    writer.publish_active(
        slot=0,
        publish_seq=0,
        request_epoch=11,
        current_round_id=5,
        arrival_seq=1,
        prompt_token_count=8,
        max_new_tokens=16,
        spec_token_limit=4,
    )
    writer.publish_active(
        slot=1,
        publish_seq=0,
        request_epoch=12,
        current_round_id=5,
        arrival_seq=2,
        prompt_token_count=8,
        max_new_tokens=16,
        spec_token_limit=4,
    )
    return table


def test_dispatch_notifies_after_facts_even_if_ring_notification_was_consumed():
    from types import SimpleNamespace
    table = _table()
    ring = CommandRing(4)
    observed = []
    def wake():
        observed.append(table.partition(StateChangeBlockKind.REQUEST_DISPATCH).read_stable(0).get('draft_issue_seq'))
    ring.doorbell = SimpleNamespace(ring=wake)
    dispatcher = DispatchPlane(endpoints=(WorkerCommandEndpoint(0, 1, ring, CommandArena(4096)),), request_table=table)
    dispatcher.dispatch(DraftBatchCommand(0, 1, 1, (_new(0),), ()))
    assert observed == [7]  # A wake issued only before facts would fail this read.


def test_dispatcher_sends_three_hot_commands_and_publishes_matching_dispatch_facts() -> None:
    table = _table()
    draft_ring = CommandRing(4)
    target_ring = CommandRing(4)
    draft_arena = CommandArena(4096)
    target_arena = CommandArena(4096)
    dispatcher = DispatchPlane(
        endpoints=(
            WorkerCommandEndpoint(2, 9, draft_ring, draft_arena),
            WorkerCommandEndpoint(4, 10, target_ring, target_arena),
        ),
        request_table=table,
    )

    dispatcher.dispatch(
        DraftBatchCommand(
            worker_id=2,
            worker_generation=9,
            command_seq=0,
            new_requests=(_new(0, round_id=5, op_seq=7),),
            cached_request_deltas=(),
        )
    )
    dispatcher.dispatch(
        PrepareTargetBankCommand(
            worker_id=4,
            target_generation=10,
            command_seq=0,
            batch_seq=17,
            standby_bank_id=1,
            next_bank_epoch=21,
            requests=(_target_prepare(0, round_id=6, op_seq=8),),
        )
    )
    dispatcher.dispatch(
        TargetPrefillBatchCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=18,
            bank_id=1,
            bank_epoch=21,
            requests=(_target_prefill(0, round_id=0, run_seq=6),),
        )
    )
    dispatcher.dispatch(
        RunTargetBatchCommand(
            worker_id=4,
            target_generation=10,
            command_seq=2,
            expected_batch_seq=17,
            active_bank_id=1,
            active_bank_epoch=21,
            standby_bank_epoch=22,
            requests=(RunTargetRequest(0, 11, 6, 9, _handle(50), _handle(60)),),
        )
    )

    draft_command = draft_ring.consume(expected_worker_generation=9, arena=draft_arena).decode(worker_id=2)  # type: ignore[union-attr]
    prepare_command = target_ring.consume(expected_worker_generation=10, arena=target_arena).decode(worker_id=4)  # type: ignore[union-attr]
    prefill_command = target_ring.consume(expected_worker_generation=10, arena=target_arena).decode(worker_id=4)  # type: ignore[union-attr]
    run_command = target_ring.consume(expected_worker_generation=10, arena=target_arena).decode(worker_id=4)  # type: ignore[union-attr]
    assert isinstance(draft_command, DraftBatchCommand)
    assert isinstance(prepare_command, PrepareTargetBankCommand)
    assert isinstance(prefill_command, TargetPrefillBatchCommand)
    assert isinstance(run_command, RunTargetBatchCommand)

    dispatch = table.partition(StateChangeBlockKind.REQUEST_DISPATCH).read_stable(0)
    assert dispatch.get("draft_issue_seq") == 7
    assert dispatch.get("target_prepare_seq") == 8
    assert dispatch.get("target_run_seq") == 9
    assert dispatch.get("planned_target_id") == 4
    latency = dispatcher.latency_summary()
    assert latency.samples == 4
    assert latency.p50_ns > 0
    assert latency.p95_ns >= latency.p50_ns
    assert latency.p99_ns >= latency.p95_ns


def test_dispatch_failure_does_not_publish_dispatch_fact() -> None:
    table = _table()
    ring = CommandRing(1)
    arena = CommandArena(4096)
    dispatcher = DispatchPlane(endpoints=(WorkerCommandEndpoint(2, 9, ring, arena),), request_table=table)

    dispatcher.dispatch(
        DraftBatchCommand(
            worker_id=2,
            worker_generation=9,
            command_seq=0,
            new_requests=(),
            cached_request_deltas=(_delta(0),),
        )
    )
    with pytest.raises(CommandBackpressure):
        dispatcher.dispatch(
            DraftBatchCommand(
                worker_id=2,
                worker_generation=9,
                command_seq=1,
                new_requests=(),
                cached_request_deltas=(_delta(0),),
            )
        )

    dispatch = table.partition(StateChangeBlockKind.REQUEST_DISPATCH).read_stable(0)
    assert dispatch.get("draft_issue_seq") == 7


def test_arena_backpressure_does_not_publish_dispatch_fact() -> None:
    table = _table()
    ring = CommandRing(4)
    dispatcher = DispatchPlane(endpoints=(WorkerCommandEndpoint(2, 9, ring, CommandArena(32)),), request_table=table)

    with pytest.raises(CommandBackpressure):
        dispatcher.dispatch(
            DraftBatchCommand(
                worker_id=2,
                worker_generation=9,
                command_seq=0,
                new_requests=(_new(0),),
                cached_request_deltas=(),
            )
        )
    with pytest.raises(StableReadConflict):
        table.partition(StateChangeBlockKind.REQUEST_DISPATCH).read_stable(0, max_retries=2)


def test_dispatch_preflight_rejects_stale_epoch_before_ring_publish() -> None:
    table = _two_slot_table()
    ring = CommandRing(4)
    arena = CommandArena(4096)
    dispatcher = DispatchPlane(endpoints=(WorkerCommandEndpoint(2, 9, ring, arena),), request_table=table)

    with pytest.raises(TableProtocolError):
        dispatcher.dispatch(
            DraftBatchCommand(
                worker_id=2,
                worker_generation=9,
                command_seq=0,
                new_requests=(
                    _new(0),
                    NewRequestData(1, 13, 5, 7, 3, _handle(1), _handle(2), _handle(3), _handle(4)),
                ),
                cached_request_deltas=(),
            )
        )

    assert ring.is_empty()
    with pytest.raises(StableReadConflict):
        table.partition(StateChangeBlockKind.REQUEST_DISPATCH).read_stable(0, max_retries=2)


def test_post_ring_dispatch_fact_failure_poisons_dispatch_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    table = _table()
    ring = CommandRing(4)
    arena = CommandArena(4096)
    dispatcher = DispatchPlane(endpoints=(WorkerCommandEndpoint(2, 9, ring, arena),), request_table=table)
    original = dispatcher._dispatch_writer.publish_draft_command_sent  # type: ignore[attr-defined]

    def fail_after_publish(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        raise RuntimeError("simulated post-ring failure")

    monkeypatch.setattr(dispatcher._dispatch_writer, "publish_draft_command_sent", fail_after_publish)  # type: ignore[attr-defined]

    with pytest.raises(DispatchPlanePoisoned):
        dispatcher.dispatch(
            DraftBatchCommand(
                worker_id=2,
                worker_generation=9,
                command_seq=0,
                new_requests=(),
                cached_request_deltas=(_delta(0),),
            )
        )

    assert not ring.is_empty()
    with pytest.raises(DispatchPlanePoisoned):
        dispatcher.dispatch(
            DraftBatchCommand(
                worker_id=2,
                worker_generation=9,
                command_seq=1,
                new_requests=(),
                cached_request_deltas=(_delta(0, op_seq=8),),
            )
        )


def test_dispatch_uses_request_run_seq_so_03_completion_accepts_worker_fact() -> None:
    table = _table()
    ring = CommandRing(4)
    arena = CommandArena(4096)
    dispatcher = DispatchPlane(endpoints=(WorkerCommandEndpoint(4, 10, ring, arena),), request_table=table)
    command = RunTargetBatchCommand(
        worker_id=4,
        target_generation=10,
        command_seq=1,
        expected_batch_seq=17,
        active_bank_id=1,
        active_bank_epoch=21,
        standby_bank_epoch=22,
        requests=(RunTargetRequest(0, 11, 6, 9, _handle(50), _handle(60)),),
    )

    dispatcher.dispatch(command)
    TargetComputeTableWriter(table).publish_ready_draft(
        fence=OperationFence(RequestFence(0, 11), 6, 9, 4, 10),
        publish_seq=0,
        bank_id=1,
        bank_epoch=21,
        target_kv_version=41,
        accepted_draft_count=1,
        committed_delta_count=1,
        last_committed_token=99,
        logical_kv_len=17,
        dirty_begin_block=0,
        dirty_block_count=1,
    )

    target = table.partition(StateChangeBlockKind.REQUEST_TARGET_COMPUTE).read_stable(0)
    assert target.get("observed_run_seq") == 9
