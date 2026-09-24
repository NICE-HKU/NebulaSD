"""Unit tests for core ABI numeric domains, handles, enums, and errors."""

from __future__ import annotations

import pytest

from nebulasd.core.enums import (
    BankRole,
    BankState,
    ComputeStatus,
    CopyStatus,
    D2HStatus,
    DraftStatus,
    H2DStatus,
    Lifecycle,
    ProposalKind,
    StateChangeBlockKind,
    TargetStatus,
    WorkerRole,
    WorkerStatus,
    validate_enum,
)
from nebulasd.core.errors import ErrorSeverity, ResultCode
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.handles import HostKVArenaHandle
from nebulasd.core.ids import (
    ARENA_GENERATION,
    ARENA_LENGTH,
    ARENA_OFFSET,
    COMMAND_SEQ,
    HOST_SLOT,
    OP_SEQ,
    REQUEST_EPOCH,
    REQUEST_SLOT,
    U8,
    U32,
    U64,
    OperationFence,
    RequestFence,
    is_stale_request_fact,
)


def test_numeric_domains_reserve_all_ones_as_invalid() -> None:
    assert U32.invalid == 0xFFFF_FFFF
    assert U32.max_valid == 0xFFFF_FFFE
    assert U64.invalid == 0xFFFF_FFFF_FFFF_FFFF
    assert U64.max_valid == 0xFFFF_FFFF_FFFF_FFFE

    REQUEST_SLOT.validate(37)
    REQUEST_EPOCH.validate(5)
    HOST_SLOT.validate(HOST_SLOT.max_valid)

    with pytest.raises(ValueError):
        REQUEST_SLOT.validate(REQUEST_SLOT.invalid)
    with pytest.raises(ValueError):
        REQUEST_SLOT.validate(-1)
    with pytest.raises(TypeError):
        REQUEST_SLOT.validate("37")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        REQUEST_SLOT.validate(True)  # type: ignore[arg-type]


def test_sequence_wraparound_skips_invalid_sentinel() -> None:
    assert COMMAND_SEQ.next(41) == 42
    assert COMMAND_SEQ.next(COMMAND_SEQ.max_valid) == 0

    assert COMMAND_SEQ.is_newer(3, 2)
    assert not COMMAND_SEQ.is_newer(2, 2)
    assert COMMAND_SEQ.is_newer(0, COMMAND_SEQ.max_valid)
    assert not COMMAND_SEQ.is_newer(COMMAND_SEQ.max_valid, 0)

    assert U8.distance(126, 254) == 127
    assert U8.is_newer(126, 254)


def test_operation_fence_rejects_stale_aba_result() -> None:
    request_a = RequestFence(request_slot=37, request_epoch=5)
    request_b = RequestFence(request_slot=37, request_epoch=6)
    assert is_stale_request_fact(request_b, request_a)

    old_fact = OperationFence(request_a, round_id=9, op_seq=101, worker_id=3, worker_generation=8)
    current_intent = OperationFence(request_b, round_id=9, op_seq=101, worker_id=3, worker_generation=8)
    assert not old_fact.applies_to(current_intent)


def test_arena_handle_boundaries() -> None:
    handle = ArenaHandle(offset=128, length=64, generation=7)
    assert handle.end_offset == 192
    assert handle.within(192)
    assert not handle.within(191)
    assert ArenaHandle.null().is_empty()

    with pytest.raises(ValueError):
        ArenaHandle(offset=ARENA_OFFSET.invalid, length=0, generation=0)
    with pytest.raises(ValueError):
        ArenaHandle(offset=0, length=ARENA_LENGTH.invalid, generation=0)
    with pytest.raises(ValueError):
        ArenaHandle(offset=0, length=0, generation=ARENA_GENERATION.invalid)


def test_enum_values_are_stable() -> None:
    expected = {
        Lifecycle: {"FREE": 0, "ACTIVE": 1, "FINISHED": 2, "CANCELLED": 3},
        DraftStatus: {"IDLE": 0, "IN_DRAFT": 1, "READY_TARGET": 2, "FAILED": 3},
        TargetStatus: {"IDLE": 0, "IN_TARGET": 1, "READY_DRAFT": 2, "FAILED": 3},
        D2HStatus: {"IDLE": 0, "IN_D2H": 1, "HOST_READY": 2, "FAILED": 3},
        H2DStatus: {"IDLE": 0, "WAIT_HOST": 1, "IN_H2D": 2, "GPU_READY": 3, "FAILED": 4},
        BankRole: {"ACTIVE": 1, "STANDBY": 2},
        BankState: {"EMPTY": 0, "DRAINING": 1, "PREPARING": 2, "READY": 3, "COMPUTING": 4},
        WorkerRole: {"DRAFT": 1, "TARGET": 2},
        WorkerStatus: {"STARTING": 0, "ONLINE": 1, "FAILED": 2, "STOPPED": 3},
        ComputeStatus: {"IDLE": 0, "RUNNING": 1},
        CopyStatus: {"IDLE": 0, "D2H": 1, "H2D": 2},
        ResultCode: {
            "OK": 0,
            "STALE_FENCE": 1,
            "INVALID_REQUEST": 2,
            "INVALID_WORKER_GENERATION": 3,
            "INVALID_BANK_EPOCH": 4,
            "ARENA_BOUNDS": 5,
            "BACKPRESSURE": 6,
            "WORKER_FAILED": 7,
            "CUDA_ERROR": 8,
            "SWIFTLLM_ERROR": 9,
            "INTERNAL_ERROR": 255,
        },
        ErrorSeverity: {"RETRYABLE": 1, "FAIL_FAST": 2},
        ProposalKind: {"LINEAR": 1, "DFLASH_BLOCK": 2, "TREE": 3},
    }
    for enum_type, values in expected.items():
        assert {member.name: int(member) for member in enum_type} == values


def test_invalid_enum_values_are_rejected() -> None:
    assert validate_enum(Lifecycle, 1) is Lifecycle.ACTIVE
    with pytest.raises(ValueError):
        validate_enum(Lifecycle, 99)


def test_state_change_block_kind_has_unique_numeric_values() -> None:
    values = [int(member) for member in StateChangeBlockKind]
    assert len(values) == len(set(values))
    assert min(values) > 0


def test_hostkv_arena_handle_uses_block_units() -> None:
    handle = HostKVArenaHandle(offset_blocks=5, block_count=3, generation=2)
    assert handle.end_block == 8
    assert handle.within_blocks(8)
    assert not handle.within_blocks(7)

    with pytest.raises(TypeError):
        HostKVArenaHandle(offset_blocks=1.5, block_count=3, generation=2)  # type: ignore[arg-type]
