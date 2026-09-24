"""Stable fixed-width enum values used by the control-plane ABI."""

from __future__ import annotations

from enum import IntEnum


ENUM_WIRE_BITS = 32


class Lifecycle(IntEnum):
    FREE = 0
    ACTIVE = 1
    FINISHED = 2
    CANCELLED = 3


class DraftStatus(IntEnum):
    IDLE = 0
    IN_DRAFT = 1
    READY_TARGET = 2
    FAILED = 3


class TargetStatus(IntEnum):
    IDLE = 0
    IN_TARGET = 1
    READY_DRAFT = 2
    FAILED = 3


class D2HStatus(IntEnum):
    IDLE = 0
    IN_D2H = 1
    HOST_READY = 2
    FAILED = 3


class H2DStatus(IntEnum):
    IDLE = 0
    WAIT_HOST = 1
    IN_H2D = 2
    GPU_READY = 3
    FAILED = 4


class BankRole(IntEnum):
    ACTIVE = 1
    STANDBY = 2


class BankState(IntEnum):
    EMPTY = 0
    DRAINING = 1
    PREPARING = 2
    READY = 3
    COMPUTING = 4


class WorkerRole(IntEnum):
    DRAFT = 1
    TARGET = 2


class WorkerStatus(IntEnum):
    STARTING = 0
    ONLINE = 1
    FAILED = 2
    STOPPED = 3


class ComputeStatus(IntEnum):
    IDLE = 0
    RUNNING = 1


class CopyStatus(IntEnum):
    IDLE = 0
    D2H = 1
    H2D = 2


class StateChangeBlockKind(IntEnum):
    REQUEST_ENGINE = 1
    REQUEST_DISPATCH = 2
    REQUEST_DRAFT = 3
    REQUEST_TARGET_COMPUTE = 4
    REQUEST_D2H = 5
    REQUEST_H2D = 6
    REQUEST_HOSTKV = 7
    WORKER_COMMON = 8
    WORKER_DRAFT_RUNTIME = 9
    WORKER_TARGET_COMPUTE_RUNTIME = 10
    WORKER_TARGET_COPY_RUNTIME = 11
    WORKER_BANK = 12
    REQUEST_DRAFT_HOSTKV = 13
    REQUEST_DRAFT_D2H = 14
    REQUEST_DRAFT_H2D = 15
    WORKER_DRAFT_COPY_RUNTIME = 16
    WORKER_DRAFT_BANK = 17


class ProposalKind(IntEnum):
    LINEAR = 1
    DFLASH_BLOCK = 2
    TREE = 3


def validate_enum(enum_type: type[IntEnum], value: int) -> IntEnum:
    """Reject integer values outside the stable ABI enum set."""

    if isinstance(value, bool):
        raise TypeError(f"{enum_type.__name__} value must be an int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{enum_type.__name__} value must be an int")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not valid for {enum_type.__name__}") from exc
