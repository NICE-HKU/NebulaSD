"""Process-local async copy operation lifecycle for the StarSD target facade.

Only pickle-safe handles and results cross the StarSD child boundary. CUDA
streams and events remain private to this registry and are never serialized.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence


class StarsDCopyDirection(str, Enum):
    H2D = "h2d"
    D2H = "d2h"


class StarsDCopyStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    ABORT_REQUESTED = "abort_requested"
    ACTIVATED = "activated"
    RESULT_READY = "result_ready"
    FAILED = "failed"
    ABORTED = "aborted"


class StarsDAsyncFatalError(RuntimeError):
    """Raised when a staged copy may have mutated target-local state."""


@dataclass(frozen=True)
class StarsDCopyHandle:
    operation_id: str
    direction: StarsDCopyDirection
    target_id: str
    process_generation: int
    request_id: str
    request_epoch: int
    round_id: int
    batch_id: str
    bank_id: int
    bank_epoch: int
    host_kv_version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", str(self.operation_id))
        object.__setattr__(self, "direction", StarsDCopyDirection(self.direction))
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "batch_id", str(self.batch_id))
        for name in (
            "process_generation",
            "request_epoch",
            "round_id",
            "bank_id",
            "bank_epoch",
            "host_kv_version",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))


@dataclass(frozen=True)
class StarsDCopyProgress:
    handle: StarsDCopyHandle
    status: StarsDCopyStatus
    error: str = ""
    device_duration_s: float | None = None


@dataclass(frozen=True)
class StarsDStageAdmission:
    handles: tuple[StarsDCopyHandle, ...]
    replayed: bool


@dataclass
class _Operation:
    handle: StarsDCopyHandle
    fingerprint: str
    batch_operation_ids: tuple[str, ...]
    status: StarsDCopyStatus
    event: Any | None = None
    start_event: Any | None = None
    result: Any = None
    committed: bool = False
    error: str = ""
    device_duration_s: float | None = None


@dataclass(frozen=True)
class _Tombstone:
    handle: StarsDCopyHandle
    fingerprint: str
    status: StarsDCopyStatus
    result: Any
    error: str = ""
    device_duration_s: float | None = None


class StarsDAsyncOperationRegistry:
    """Bounded target-process registry for staged H2D/D2H copies.

    Active capacity tracks only operations that may still progress, finalize, or
    abort. Terminal operations retire into bounded exact-replay tombstones so a
    caller can safely replay a completion without pinning the active registry.
    """

    def __init__(
        self,
        *,
        target_id: str,
        process_generation: int,
        capacity: int = 4096,
        replay_window: int = 4096,
    ) -> None:
        if not target_id or process_generation < 0 or capacity <= 0 or replay_window <= 0:
            raise ValueError("invalid async operation registry configuration")
        self._target_id = target_id
        self._process_generation = int(process_generation)
        self._capacity = int(capacity)
        self._replay_window = int(replay_window)
        self._active: OrderedDict[str, _Operation] = OrderedDict()
        self._tombstones: OrderedDict[str, _Tombstone] = OrderedDict()

    @property
    def active_operation_count(self) -> int:
        return len(self._active)

    @property
    def replay_tombstone_count(self) -> int:
        return len(self._tombstones)

    @property
    def abort_pending_count(self) -> int:
        return sum(1 for item in self._active.values() if item.status == StarsDCopyStatus.ABORT_REQUESTED)

    def reserve_batch(
        self,
        handles: Sequence[StarsDCopyHandle],
        fingerprints: Sequence[str],
    ) -> StarsDStageAdmission:
        handles = tuple(handles)
        fingerprints = tuple(str(item) for item in fingerprints)
        self._validate_batch_shape(handles, fingerprints)

        replay_count = 0
        new_count = 0
        for handle, fingerprint in zip(handles, fingerprints, strict=True):
            existing = self._active.get(handle.operation_id)
            if existing is not None:
                self._require_same_payload(existing.handle, existing.fingerprint, handle, fingerprint)
                replay_count += 1
                continue
            tombstone = self._tombstones.get(handle.operation_id)
            if tombstone is not None:
                self._require_same_payload(tombstone.handle, tombstone.fingerprint, handle, fingerprint)
                replay_count += 1
                continue
            new_count += 1

        if replay_count and new_count:
            raise RuntimeError("mixed replay/new async copy batch must be split by caller")
        if replay_count:
            return StarsDStageAdmission(handles, replayed=True)
        if len(self._active) + new_count > self._capacity:
            raise RuntimeError("async operation registry capacity exceeded")

        batch_operation_ids = tuple(handle.operation_id for handle in handles)
        for handle, fingerprint in zip(handles, fingerprints, strict=True):
            self._active[handle.operation_id] = _Operation(
                handle=handle,
                fingerprint=fingerprint,
                batch_operation_ids=batch_operation_ids,
                status=StarsDCopyStatus.PENDING,
            )
        return StarsDStageAdmission(handles, replayed=False)

    def commit_batch(
        self,
        handles: Sequence[StarsDCopyHandle],
        *,
        event: Any | None,
        start_event: Any | None = None,
        results: Sequence[Any],
    ) -> tuple[StarsDCopyHandle, ...]:
        handles = tuple(handles)
        results = tuple(results)
        if len(handles) != len(results):
            raise RuntimeError("async operation result count mismatch")
        operations = self._require_active_batch(handles)
        for operation, result in zip(operations, results, strict=True):
            if operation.committed:
                raise RuntimeError("async operation batch is already committed")
            operation.event = event
            operation.start_event = start_event
            operation.result = result
            operation.committed = True
        return handles

    def rollback_admission(self, handles: Sequence[StarsDCopyHandle]) -> None:
        handles = tuple(handles)
        for handle in handles:
            operation = self._active.get(handle.operation_id)
            if operation is None:
                continue
            if operation.handle != handle:
                raise RuntimeError("async operation rollback handle mismatch")
            if operation.committed:
                raise RuntimeError("committed async operation cannot be rolled back")
        for handle in handles:
            self._active.pop(handle.operation_id, None)

    def stage(
        self,
        handle: StarsDCopyHandle,
        *,
        event: Any | None,
        result: Any,
        fingerprint: str,
        finalize: Callable[[], Any] | None = None,
    ) -> StarsDCopyHandle:
        del finalize
        admission = self.reserve_batch((handle,), (fingerprint,))
        if not admission.replayed:
            self.commit_batch((handle,), event=event, results=(result,))
        return handle

    def progress(self, handle: StarsDCopyHandle) -> StarsDCopyProgress:
        self._validate_handle(handle)
        active = self._active.get(handle.operation_id)
        if active is not None:
            self._require_same_payload(active.handle, active.fingerprint, handle, active.fingerprint)
            if not active.committed:
                raise RuntimeError("async operation has not been committed")
            if active.status == StarsDCopyStatus.PENDING and self._event_ready(active.event):
                active.status = StarsDCopyStatus.READY
                active.device_duration_s = self._event_elapsed_seconds(active.start_event, active.event)
            elif active.status == StarsDCopyStatus.ABORT_REQUESTED and self._event_ready(active.event):
                tombstone = self._retire(active, StarsDCopyStatus.ABORTED, None, active.error)
                return StarsDCopyProgress(handle, tombstone.status, tombstone.error, tombstone.device_duration_s)
            return StarsDCopyProgress(handle, active.status, active.error, active.device_duration_s)

        tombstone = self._require_tombstone(handle)
        return StarsDCopyProgress(handle, tombstone.status, tombstone.error, tombstone.device_duration_s)

    def finalize_batch(
        self,
        handles: Sequence[StarsDCopyHandle],
        *,
        expected_direction: StarsDCopyDirection,
        finalize: Callable[[], Sequence[Any]] | None = None,
    ) -> tuple[Any, ...]:
        handles = tuple(handles)
        if not handles:
            return ()
        self._validate_finalize_handles(handles, expected_direction)
        terminal = (
            StarsDCopyStatus.ACTIVATED
            if expected_direction is StarsDCopyDirection.H2D
            else StarsDCopyStatus.RESULT_READY
        )

        active = [self._active.get(handle.operation_id) for handle in handles]
        tombstones = [self._tombstones.get(handle.operation_id) for handle in handles]
        if all(item is not None for item in tombstones):
            checked = tuple(self._require_tombstone(handle) for handle in handles)
            if any(item.status != terminal for item in checked):
                raise RuntimeError("async operation replay terminal status mismatch")
            return tuple(item.result for item in checked)
        if any(item is not None for item in tombstones) or any(item is None for item in active):
            raise RuntimeError("partial async operation batch replay is not supported")

        operations = self._require_active_batch(handles)
        if tuple(handles) != tuple(operation.handle for operation in operations):
            raise RuntimeError("async operation batch handle order mismatch")
        if any(operation.handle.direction is not expected_direction for operation in operations):
            raise RuntimeError("async operation direction mismatch")
        for handle in handles:
            progress = self.progress(handle)
            if progress.status != StarsDCopyStatus.READY:
                raise RuntimeError("async copy event is not ready")

        results = tuple(operation.result for operation in operations)
        if finalize is not None:
            results = tuple(finalize())
            if len(results) != len(handles):
                raise RuntimeError("async operation finalize result count mismatch")
        retired = []
        for operation, result in zip(operations, results, strict=True):
            operation.result = result
            retired.append(self._retire(operation, terminal, result, ""))
        return tuple(item.result for item in retired)

    def finalize(self, handle: StarsDCopyHandle, *, expected_direction: StarsDCopyDirection) -> Any:
        result = self.finalize_batch((handle,), expected_direction=expected_direction)
        return result[0]

    def abort(self, handle: StarsDCopyHandle) -> StarsDCopyProgress:
        self._validate_handle(handle)
        active = self._active.get(handle.operation_id)
        if active is None:
            tombstone = self._require_tombstone(handle)
            if tombstone.status == StarsDCopyStatus.ABORTED:
                return StarsDCopyProgress(handle, tombstone.status, tombstone.error)
            raise RuntimeError("completed async operation cannot be aborted")
        self._require_same_payload(active.handle, active.fingerprint, handle, active.fingerprint)
        if not active.committed:
            tombstone = self._retire(active, StarsDCopyStatus.ABORTED, None, "aborted before copy commit")
            return StarsDCopyProgress(handle, tombstone.status, tombstone.error)
        if self._event_ready(active.event):
            tombstone = self._retire(active, StarsDCopyStatus.ABORTED, None, active.error)
            return StarsDCopyProgress(handle, tombstone.status, tombstone.error)
        active.status = StarsDCopyStatus.ABORT_REQUESTED
        return StarsDCopyProgress(handle, active.status, active.error)

    def abort_request(self, *, request_id: str, request_epoch: int, round_id: int) -> tuple[StarsDCopyProgress, ...]:
        matches = tuple(
            operation.handle
            for operation in self._active.values()
            if operation.handle.request_id == str(request_id)
            and operation.handle.request_epoch == int(request_epoch)
            and operation.handle.round_id == int(round_id)
        )
        return tuple(self.abort(handle) for handle in matches)

    def drain(self) -> None:
        for operation in tuple(self._active.values()):
            if operation.event is not None and hasattr(operation.event, "synchronize"):
                operation.event.synchronize()
            if operation.status == StarsDCopyStatus.ABORT_REQUESTED:
                self._retire(operation, StarsDCopyStatus.ABORTED, None, operation.error)
            elif operation.status in {StarsDCopyStatus.PENDING, StarsDCopyStatus.READY}:
                self._retire(operation, StarsDCopyStatus.ABORTED, None, "aborted during shutdown")

    def _require_active_batch(self, handles: Sequence[StarsDCopyHandle]) -> tuple[_Operation, ...]:
        handles = tuple(handles)
        if not handles:
            return ()
        operations = []
        for handle in handles:
            self._validate_handle(handle)
            operation = self._active.get(handle.operation_id)
            if operation is None or operation.handle != handle:
                raise RuntimeError("unknown or forged async operation handle")
            operations.append(operation)
        expected_ids = operations[0].batch_operation_ids
        actual_ids = tuple(handle.operation_id for handle in handles)
        if actual_ids != expected_ids:
            raise RuntimeError("async operation batch must be complete and in stage order")
        return tuple(operations)

    def _require_tombstone(self, handle: StarsDCopyHandle) -> _Tombstone:
        self._validate_handle(handle)
        tombstone = self._tombstones.get(handle.operation_id)
        if tombstone is None or tombstone.handle != handle:
            raise RuntimeError("unknown or forged async operation handle")
        self._tombstones.move_to_end(handle.operation_id)
        return tombstone

    def _retire(
        self,
        operation: _Operation,
        status: StarsDCopyStatus,
        result: Any,
        error: str,
    ) -> _Tombstone:
        self._active.pop(operation.handle.operation_id, None)
        tombstone = _Tombstone(
            operation.handle,
            operation.fingerprint,
            status,
            result,
            error,
            operation.device_duration_s,
        )
        self._tombstones[operation.handle.operation_id] = tombstone
        self._tombstones.move_to_end(operation.handle.operation_id)
        while len(self._tombstones) > self._replay_window:
            self._tombstones.popitem(last=False)
        return tombstone

    def _validate_batch_shape(
        self,
        handles: tuple[StarsDCopyHandle, ...],
        fingerprints: tuple[str, ...],
    ) -> None:
        if len(handles) != len(fingerprints):
            raise RuntimeError("async operation fingerprint count mismatch")
        if not handles:
            return
        ids = tuple(handle.operation_id for handle in handles)
        if len(set(ids)) != len(ids):
            raise RuntimeError("duplicate async operation id in batch")
        first = handles[0]
        for handle in handles:
            self._validate_handle(handle)
            if handle.direction is not first.direction:
                raise RuntimeError("async operation batch direction mismatch")
            if handle.batch_id != first.batch_id:
                raise RuntimeError("async operation batch id mismatch")
            if handle.bank_id != first.bank_id or handle.bank_epoch != first.bank_epoch:
                raise RuntimeError("async operation batch bank fence mismatch")
            if handle.target_id != first.target_id or handle.process_generation != first.process_generation:
                raise RuntimeError("async operation batch target generation mismatch")

    def _validate_finalize_handles(
        self,
        handles: tuple[StarsDCopyHandle, ...],
        expected_direction: StarsDCopyDirection,
    ) -> None:
        self._validate_batch_shape(handles, tuple("" for _ in handles))
        if any(handle.direction is not expected_direction for handle in handles):
            raise RuntimeError("async operation direction mismatch")

    def _validate_handle(self, handle: StarsDCopyHandle) -> None:
        if handle.target_id != self._target_id or handle.process_generation != self._process_generation:
            raise RuntimeError("async operation target generation mismatch")
        if not handle.operation_id or not handle.request_id or not handle.batch_id:
            raise RuntimeError("async operation handle has empty identity")
        if handle.request_epoch < 0 or handle.round_id < 0 or handle.bank_id < 0 or handle.bank_epoch < 0:
            raise RuntimeError("async operation handle has negative fence")
        if handle.host_kv_version < 0:
            raise RuntimeError("async operation HostKV version is negative")

    @staticmethod
    def _event_ready(event: Any | None) -> bool:
        return event is None or bool(event.query())

    @staticmethod
    def _event_elapsed_seconds(start_event: Any | None, end_event: Any | None) -> float | None:
        if start_event is None or end_event is None:
            return None
        elapsed = getattr(start_event, "elapsed_time", None)
        if elapsed is None:
            return None
        try:
            return max(float(elapsed(end_event)) / 1000.0, 0.0)
        except Exception:
            return None

    @staticmethod
    def _require_same_payload(
        old_handle: StarsDCopyHandle,
        old_fingerprint: str,
        handle: StarsDCopyHandle,
        fingerprint: str,
    ) -> None:
        if old_handle != handle or old_fingerprint != fingerprint:
            raise RuntimeError("async operation replay used a different payload")
