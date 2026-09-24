"""Fixed-width result codes and management-path exceptions."""

from __future__ import annotations

from enum import IntEnum


class ResultCode(IntEnum):
    OK = 0
    STALE_FENCE = 1
    INVALID_REQUEST = 2
    INVALID_WORKER_GENERATION = 3
    INVALID_BANK_EPOCH = 4
    ARENA_BOUNDS = 5
    BACKPRESSURE = 6
    WORKER_FAILED = 7
    CUDA_ERROR = 8
    SWIFTLLM_ERROR = 9
    INTERNAL_ERROR = 255


class ErrorSeverity(IntEnum):
    RETRYABLE = 1
    FAIL_FAST = 2


class ControlPlaneError(RuntimeError):
    """Management-path exception; never serialized through hot IPC paths."""

    def __init__(self, code: ResultCode, message: str, *, severity: ErrorSeverity = ErrorSeverity.FAIL_FAST) -> None:
        super().__init__(message)
        self.code = code
        self.severity = severity
