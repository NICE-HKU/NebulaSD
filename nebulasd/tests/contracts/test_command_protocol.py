"""Contracts for dispatch-plane command ABI encoding."""

from __future__ import annotations

import inspect

import pytest

from nebulasd.core.handles import ArenaHandle, HostKVArenaHandle
from nebulasd.ipc.management import ManagementCommand, ManagementQueue
from nebulasd.ipc.protocol import (
    COMMAND_PAYLOAD_MAGIC,
    COMMAND_HEADER_STRUCT,
    COMMAND_PROTOCOL_VERSION,
    BankAllocation,
    CachedRequestDelta,
    CommandHeader,
    CommandKind,
    DraftBatchCommand,
    ManagementCommandKind,
    NewRequestData,
    OwnedArenaHandle,
    PrepareTargetBankCommand,
    HostKVSource,
    RemovedRequest,
    RunTargetBatchCommand,
    RunTargetRequest,
    TargetPrefillBatchCommand,
    TargetPrefillRequest,
    TargetPrepareRequest,
    TargetVerifyRequest,
    WorkerCommandCache,
    decode_command_payload,
    encode_command_payload,
)


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


def _target_prefill(slot: int, *, round_id: int = 0, run_seq: int = 1, bank_id: int = 1, bank_epoch: int = 21) -> TargetPrefillRequest:
    return TargetPrefillRequest(
        request_slot=slot,
        request_epoch=11,
        round_id=round_id,
        run_seq=run_seq,
        input_tokens_handle=_handle(100),
        generation_config_handle=_handle(120),
        scheduled_token_count=1,
        max_output_len=16,
        bank_id=bank_id,
        bank_epoch=bank_epoch,
        bank_offset_blocks=0,
        block_count=4,
    )


def _owned_handle(owner: int = 2, generation: int = 9, offset: int = 50) -> OwnedArenaHandle:
    return OwnedArenaHandle(owner, generation, _handle(offset))


def _target_verify(slot: int, *, round_id: int = 6, run_seq: int = 9) -> TargetVerifyRequest:
    return TargetVerifyRequest(
        request_slot=slot,
        request_epoch=11,
        round_id=round_id,
        run_seq=run_seq,
        proposal_handle=_owned_handle(),
        committed_output_handle=_handle(60),
        committed_output_count=1,
        prompt_token_count=8,
        generation_config_handle=_handle(70),
        bank_id=1,
        bank_epoch=21,
        bank_offset_blocks=0,
        block_count=4,
    )


def _target_prepare(
    slot: int,
    *,
    round_id: int = 6,
    op_seq: int = 8,
    bank_id: int = 1,
    bank_epoch: int = 21,
    offset_blocks: int = 0,
    capacity_blocks: int = 4,
    request_epoch: int = 11,
) -> TargetPrepareRequest:
    return TargetPrepareRequest(
        request_slot=slot,
        request_epoch=request_epoch,
        round_id=round_id,
        op_seq=op_seq,
        committed_output_handle=_handle(60 + slot),
        committed_output_count=1,
        prompt_token_count=8,
        generation_config_handle=_handle(70 + slot),
        hostkv_handle=_hostkv_handle(30 + slot, block_count=capacity_blocks),
        host_slot=5 + slot,
        host_slot_generation=23 + slot,
        host_writer_lease_generation=33 + slot,
        source_host_version=31 + slot,
        logical_kv_len=8,
        committed_blocks=1,
        valid_blocks=1,
        destination_bank_id=bank_id,
        destination_bank_epoch=bank_epoch,
        destination_bank_offset_blocks=offset_blocks,
        destination_capacity_blocks=capacity_blocks,
    )


def test_command_header_round_trip_is_fixed_width() -> None:
    assert COMMAND_PROTOCOL_VERSION == 5
    header = CommandHeader(
        command_seq=2,
        worker_generation=9,
        command_kind=CommandKind.DRAFT_BATCH,
        payload_offset=128,
        payload_length=64,
        flags=0,
    )

    raw = header.to_bytes()
    assert len(raw) == COMMAND_HEADER_STRUCT.size
    assert CommandHeader.from_bytes(raw) == header
    assert raw == (
        b"\x02\x00\x00\x00\x00\x00\x00\x00"
        b"\x09\x00\x00\x00\x00\x00\x00\x00"
        b"\x01\x00\x00\x00"
        b"\x80\x00\x00\x00\x00\x00\x00\x00"
        b"\x40\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    assert COMMAND_HEADER_STRUCT.size == 36

    large_offset = CommandHeader(3, 9, CommandKind.DRAFT_BATCH, 1 << 40, 8)
    assert CommandHeader.from_bytes(large_offset.to_bytes()).payload_offset == 1 << 40
    with pytest.raises(TypeError):
        CommandHeader(3, 9, True, 0, 8)  # type: ignore[arg-type]


def test_hot_commands_binary_round_trip_without_pickle_or_dict() -> None:
    commands = (
        DraftBatchCommand(
            worker_id=2,
            worker_generation=9,
            command_seq=0,
            new_requests=(_new(0),),
            cached_request_deltas=(),
            removed_requests=(RemovedRequest(3, 12),),
        ),
        PrepareTargetBankCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=17,
            standby_bank_id=1,
            next_bank_epoch=21,
            requests=(_target_prepare(0, round_id=6, op_seq=8),),
        ),
        TargetPrefillBatchCommand(
            worker_id=4,
            target_generation=10,
            command_seq=2,
            batch_seq=18,
            bank_id=1,
            bank_epoch=21,
            requests=(_target_prefill(0),),
        ),
        RunTargetBatchCommand(
            worker_id=4,
            target_generation=10,
            command_seq=3,
            expected_batch_seq=17,
            active_bank_id=1,
            active_bank_epoch=21,
            standby_bank_epoch=22,
            requests=(RunTargetRequest(0, 11, 6, 9, _handle(50), _handle(60)),),
        ),
        RunTargetBatchCommand(
            worker_id=4,
            target_generation=10,
            command_seq=4,
            expected_batch_seq=17,
            active_bank_id=1,
            active_bank_epoch=21,
            standby_bank_epoch=22,
            requests=(_target_verify(0),),
        ),
    )

    for command in commands:
        raw = encode_command_payload(command)
        decoded = decode_command_payload(
            command.kind,
            raw,
            worker_id=command.worker_id,
            worker_generation=command.worker_generation if isinstance(command, DraftBatchCommand) else command.target_generation,
            command_seq=command.command_seq,
        )
        assert decoded == command

    source = inspect.getsource(encode_command_payload)
    assert "pickle" not in source
    assert "dict" not in source


def test_invalid_bank_id_and_management_kind_are_rejected() -> None:
    with pytest.raises(ValueError):
        PrepareTargetBankCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=17,
            standby_bank_id=255,
            next_bank_epoch=21,
            requests=(),
        )

    queue = ManagementQueue()
    command = ManagementCommand(ManagementCommandKind.CANCEL_REQUEST, worker_id=4, request_slot=0, request_epoch=11)
    queue.submit(command)
    assert queue.pop() == command
    assert queue.pop() is None
    with pytest.raises(TypeError):
        ManagementCommand(True, worker_id=4)  # type: ignore[arg-type]


def test_worker_command_cache_validates_generation_sequence_and_request_epoch() -> None:
    cache = WorkerCommandCache(worker_id=2, worker_generation=9)
    command = DraftBatchCommand(
        worker_id=2,
        worker_generation=9,
        command_seq=0,
        new_requests=(_new(0),),
        cached_request_deltas=(),
        removed_requests=(),
    )
    cache.apply(command)
    assert cache.get(0).request_epoch == 11  # type: ignore[union-attr]
    assert cache.get(0).last_op_seq == 7  # type: ignore[union-attr]

    with pytest.raises(ValueError):
        cache.apply(command)
    with pytest.raises(ValueError):
        cache.apply(
            DraftBatchCommand(
                worker_id=2,
                worker_generation=10,
                command_seq=1,
                new_requests=(),
                cached_request_deltas=(),
            )
        )
    with pytest.raises(ValueError):
        cache.apply(
            DraftBatchCommand(
                worker_id=2,
                worker_generation=9,
                command_seq=1,
                new_requests=(),
                cached_request_deltas=(CachedRequestDelta(0, 12, 5, 8, _handle(10), _handle(20), _hostkv_handle(30), _handle(40), 3),),
            )
        )

    cache.apply(DraftBatchCommand(2, 9, 1, (), (), (RemovedRequest(0, 11),)))
    assert cache.get(0) is None


def test_worker_command_cache_applies_batch_atomically_and_rejects_op_rollback() -> None:
    cache = WorkerCommandCache(worker_id=2, worker_generation=9)
    cache.apply(DraftBatchCommand(2, 9, 0, (_new(0, round_id=5, op_seq=5),), (), ()))

    with pytest.raises(ValueError):
        cache.apply(
            DraftBatchCommand(
                worker_id=2,
                worker_generation=9,
                command_seq=1,
                new_requests=(_new(1, round_id=1, op_seq=1),),
                cached_request_deltas=(CachedRequestDelta(0, 12, 6, 6, _handle(10), _handle(20), _hostkv_handle(30), _handle(40), 3),),
            )
        )
    assert cache.get(1) is None
    assert cache.get(0).last_round_id == 5  # type: ignore[union-attr]

    with pytest.raises(ValueError):
        cache.apply(DraftBatchCommand(2, 9, 1, (), (_delta(0, round_id=4, op_seq=4),), ()))
    assert cache.get(0).last_op_seq == 5  # type: ignore[union-attr]

    with pytest.raises(ValueError):
        DraftBatchCommand(2, 9, 1, (_new(2),), (_delta(2),), ())

    with pytest.raises(ValueError):
        DraftBatchCommand(
            worker_id=2,
            worker_generation=9,
            command_seq=1,
            new_requests=(_new(2), NewRequestData(2, 12, 5, 7, 3, _handle(1), _handle(2), _handle(3), _handle(4))),
            cached_request_deltas=(),
        )

    with pytest.raises(TypeError):
        CachedRequestDelta(0, 11, 5, 7, _handle(10), _handle(20), _handle(30), _handle(40), 3)  # type: ignore[arg-type]


def test_old_protocol_version_is_rejected_by_version_4_decoder() -> None:
    command = DraftBatchCommand(2, 9, 0, (_new(0),), ())
    raw = bytearray(encode_command_payload(command))
    raw[4:8] = (COMMAND_PROTOCOL_VERSION - 1).to_bytes(4, "little")
    assert int.from_bytes(raw[0:4], "little") == COMMAND_PAYLOAD_MAGIC

    with pytest.raises(ValueError, match="payload version"):
        decode_command_payload(command.kind, bytes(raw), worker_id=2, worker_generation=9, command_seq=0)


def test_target_prefill_batch_command_validates_bank_and_cache_entry() -> None:
    command = TargetPrefillBatchCommand(
        worker_id=4,
        target_generation=10,
        command_seq=0,
        batch_seq=18,
        bank_id=1,
        bank_epoch=21,
        requests=(_target_prefill(0),),
    )
    decoded = decode_command_payload(
        command.kind,
        encode_command_payload(command),
        worker_id=4,
        worker_generation=10,
        command_seq=0,
    )
    assert decoded == command

    cache = WorkerCommandCache(worker_id=4, worker_generation=10)
    cache.apply(command)
    entry = cache.get(0)
    assert entry is not None
    assert entry.last_op_seq == 1
    assert entry.prepared_batch_seq == 18
    assert entry.prepared_bank_id == 1

    with pytest.raises(ValueError, match="bank_id"):
        TargetPrefillBatchCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=19,
            bank_id=0,
            bank_epoch=21,
            requests=(_target_prefill(1),),
        )


def test_run_target_batch_command_accepts_typed_verify_request() -> None:
    cache = WorkerCommandCache(worker_id=4, worker_generation=10)
    prefill = TargetPrefillBatchCommand(4, 10, 0, 18, 1, 21, (_target_prefill(0, round_id=0, run_seq=1),))
    prepare = PrepareTargetBankCommand(
        worker_id=4,
        target_generation=10,
        command_seq=1,
        batch_seq=19,
        standby_bank_id=1,
        next_bank_epoch=21,
        requests=(_target_prepare(0, round_id=1, op_seq=2),),
    )
    run = RunTargetBatchCommand(
        worker_id=4,
        target_generation=10,
        command_seq=2,
        expected_batch_seq=19,
        active_bank_id=1,
        active_bank_epoch=21,
        standby_bank_epoch=22,
        requests=(_target_verify(0, round_id=1, run_seq=3),),
    )

    decoded = decode_command_payload(run.kind, encode_command_payload(run), worker_id=4, worker_generation=10, command_seq=2)
    assert decoded == run
    cache.apply(prefill)
    cache.apply(prepare)
    cache.apply(run)
    entry = cache.get(0)
    assert entry is not None
    assert entry.last_op_seq == 3
    assert entry.proposal_handle == _handle(50)


def test_draft_new_request_rejects_empty_initial_anchor_before_dispatch() -> None:
    with pytest.raises(ValueError, match="initial_output_tokens_handle"):
        DraftBatchCommand(
            worker_id=2,
            worker_generation=9,
            command_seq=0,
            new_requests=(
                NewRequestData(0, 11, 5, 7, 3, _handle(1), ArenaHandle.null(), _handle(3), _handle(4)),
            ),
            cached_request_deltas=(),
        )


def test_prepare_command_validates_bank_and_request_sets() -> None:
    prepare = PrepareTargetBankCommand(
        worker_id=4,
        target_generation=10,
        command_seq=1,
        batch_seq=17,
        standby_bank_id=1,
        next_bank_epoch=21,
        requests=(_target_prepare(0),),
    )
    decoded = decode_command_payload(
        prepare.kind,
        encode_command_payload(prepare),
        worker_id=4,
        worker_generation=10,
        command_seq=1,
    )
    assert decoded == prepare

    with pytest.raises(ValueError):
        PrepareTargetBankCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=17,
            standby_bank_id=1,
            next_bank_epoch=21,
            requests=(_target_prepare(0, bank_id=0),),
        )
    with pytest.raises(ValueError):
        PrepareTargetBankCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=17,
            standby_bank_id=1,
            next_bank_epoch=21,
            requests=(_target_prepare(0, bank_epoch=20),),
        )
    with pytest.raises(ValueError):
        PrepareTargetBankCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=17,
            standby_bank_id=1,
            next_bank_epoch=21,
            requests=(
                _target_prepare(0, offset_blocks=0),
                _target_prepare(1, offset_blocks=5),
            ),
        )


def test_target_cache_requires_prepare_before_run_and_batch_identity_matches() -> None:
    cache = WorkerCommandCache(worker_id=4, worker_generation=10)
    with pytest.raises(ValueError):
        cache.apply(
            RunTargetBatchCommand(
                worker_id=4,
                target_generation=10,
                command_seq=0,
                expected_batch_seq=17,
                active_bank_id=1,
                active_bank_epoch=21,
                standby_bank_epoch=22,
                requests=(RunTargetRequest(0, 11, 6, 9, _handle(50), _handle(60)),),
            )
        )

    cache.apply(
        TargetPrefillBatchCommand(4, 10, 0, 16, 1, 20, (_target_prefill(0, round_id=0, run_seq=1, bank_epoch=20),))
    )
    cache.apply(
        PrepareTargetBankCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=17,
            standby_bank_id=1,
            next_bank_epoch=21,
            requests=(_target_prepare(0, round_id=6, op_seq=8),),
        )
    )
    with pytest.raises(ValueError):
        cache.apply(
            RunTargetBatchCommand(
                worker_id=4,
                target_generation=10,
                command_seq=2,
                expected_batch_seq=18,
                active_bank_id=1,
                active_bank_epoch=21,
                standby_bank_epoch=22,
                requests=(RunTargetRequest(0, 11, 6, 9, _handle(50), _handle(60)),),
            )
        )
    cache.apply(
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
    assert cache.get(0).last_op_seq == 9  # type: ignore[union-attr]


def test_target_cache_requires_full_ordered_batch_and_single_execution() -> None:
    cache = WorkerCommandCache(worker_id=4, worker_generation=10)
    cache.apply(
        TargetPrefillBatchCommand(
            4,
            10,
            0,
            16,
            1,
            20,
            (
                _target_prefill(0, round_id=0, run_seq=1, bank_epoch=20),
                _target_prefill(1, round_id=0, run_seq=1, bank_epoch=20),
            ),
        )
    )
    cache.apply(
        PrepareTargetBankCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=17,
            standby_bank_id=1,
            next_bank_epoch=21,
            requests=(
                _target_prepare(0, round_id=6, op_seq=8, offset_blocks=0),
                _target_prepare(1, round_id=6, op_seq=8, offset_blocks=4),
            ),
        )
    )

    with pytest.raises(ValueError):
        cache.apply(
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
    assert cache.get(0).last_op_seq == 8  # type: ignore[union-attr]

    with pytest.raises(ValueError):
        cache.apply(
            RunTargetBatchCommand(
                worker_id=4,
                target_generation=10,
                command_seq=2,
                expected_batch_seq=17,
                active_bank_id=1,
                active_bank_epoch=21,
                standby_bank_epoch=22,
                requests=(
                    RunTargetRequest(1, 11, 6, 9, _handle(51), _handle(61)),
                    RunTargetRequest(0, 11, 6, 9, _handle(50), _handle(60)),
                ),
            )
        )


def test_target_cache_keeps_only_latest_prepared_state_per_bank() -> None:
    cache = WorkerCommandCache(worker_id=4, worker_generation=10)
    batch_count = 3000
    cache.apply(TargetPrefillBatchCommand(4, 10, 0, 16, 1, 20, (_target_prefill(0, round_id=0, run_seq=1, bank_epoch=20),)))

    for batch_seq in range(batch_count):
        bank_id = 1 + (batch_seq % 2)
        bank_epoch = 21 + batch_seq
        round_id = 6 + batch_seq
        prepare_op_seq = 8 + batch_seq * 2
        run_seq = prepare_op_seq + 1

        cache.apply(
            PrepareTargetBankCommand(
                worker_id=4,
                target_generation=10,
                command_seq=batch_seq * 2 + 1,
                batch_seq=batch_seq,
                standby_bank_id=bank_id,
                next_bank_epoch=bank_epoch,
                requests=(
                    _target_prepare(0, round_id=round_id, op_seq=prepare_op_seq, bank_id=bank_id, bank_epoch=bank_epoch),
                ),
            )
        )
        assert len(cache._prepared_batches) <= 2  # type: ignore[attr-defined]

        cache.apply(
            RunTargetBatchCommand(
                worker_id=4,
                target_generation=10,
                command_seq=batch_seq * 2 + 2,
                expected_batch_seq=batch_seq,
                active_bank_id=bank_id,
                active_bank_epoch=bank_epoch,
                standby_bank_epoch=bank_epoch + 1,
                requests=(RunTargetRequest(0, 11, round_id, run_seq, _handle(50), _handle(60)),),
            )
        )
        assert len(cache._prepared_batches) <= 2  # type: ignore[attr-defined]

    assert len(cache._prepared_batches) == 2  # type: ignore[attr-defined]


def test_target_cache_rejects_prepare_over_unconsumed_bank_state() -> None:
    cache = WorkerCommandCache(worker_id=4, worker_generation=10)
    cache.apply(TargetPrefillBatchCommand(4, 10, 0, 16, 1, 20, (_target_prefill(0, round_id=0, run_seq=1, bank_epoch=20),)))
    cache.apply(
        PrepareTargetBankCommand(
            worker_id=4,
            target_generation=10,
            command_seq=1,
            batch_seq=17,
            standby_bank_id=1,
            next_bank_epoch=21,
            requests=(_target_prepare(0, round_id=6, op_seq=8),),
        )
    )

    with pytest.raises(ValueError):
        cache.apply(
            PrepareTargetBankCommand(
                worker_id=4,
                target_generation=10,
                command_seq=2,
                batch_seq=18,
                standby_bank_id=1,
                next_bank_epoch=22,
                requests=(_target_prepare(0, round_id=7, op_seq=10, bank_epoch=22),),
            )
        )
    assert cache._prepared_batches[1].batch_seq == 17  # type: ignore[attr-defined]
    assert cache._prepared_batches[1].bank_epoch == 21  # type: ignore[attr-defined]
