"""Process-local CUDA host registration for HostKV arenas."""

from __future__ import annotations

from typing import Any, Protocol


class HostRegisterableDescriptor(Protocol):
    arena_id: str
    total_bytes: int


class HostRegisterableArena(Protocol):
    descriptor: HostRegisterableDescriptor

    def address(self) -> int: ...


class HostRegistrationError(RuntimeError):
    """Raised when host registration fails."""


class HostRegistrationRecord:
    """Process-local registration receipt; never send through hot IPC paths."""

    __slots__ = ("arena_id", "executor_id", "process_generation", "address", "nbytes")

    def __init__(self, arena_id: str, executor_id: str, process_generation: int, address: int, nbytes: int) -> None:
        if not str(arena_id) or not str(executor_id):
            raise ValueError("arena_id and executor_id must be non-empty")
        self.arena_id = str(arena_id)
        self.executor_id = str(executor_id)
        self.process_generation = _non_negative(process_generation, "process_generation")
        self.address = _positive(address, "address")
        self.nbytes = _positive(nbytes, "nbytes")

    def as_tuple(self) -> tuple[str, str, int, int, int]:
        return (self.arena_id, self.executor_id, self.process_generation, self.address, self.nbytes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HostRegistrationRecord) and self.as_tuple() == other.as_tuple()

    def __repr__(self) -> str:
        return (
            "HostRegistrationRecord("
            f"arena_id={self.arena_id!r}, executor_id={self.executor_id!r}, "
            f"process_generation={self.process_generation!r}, address={self.address!r}, "
            f"nbytes={self.nbytes!r})"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("HostRegistrationRecord is process-local and cannot be pickled")


class CudaHostRegistrationAdapter:
    """Register one attached shared-memory mapping per executor process."""

    def __init__(self, cudart: Any | None = None) -> None:
        self._records: dict[tuple[str, str, int], HostRegistrationRecord] = {}
        self._cudart = cudart

    def register(
        self,
        arena: HostRegisterableArena,
        *,
        executor_id: str,
        process_generation: int,
    ) -> HostRegistrationRecord:
        record = HostRegistrationRecord(
            arena.descriptor.arena_id,
            executor_id,
            process_generation,
            arena.address(),
            arena.descriptor.total_bytes,
        )
        key = (record.arena_id, record.executor_id, record.process_generation)
        existing = self._records.get(key)
        if existing is not None:
            if existing.address != record.address or existing.nbytes != record.nbytes:
                raise HostRegistrationError("duplicate registration key has different address or nbytes")
            return existing
        cudart = self._load_cudart()
        _check_cuda_status(cudart.cudaHostRegister(record.address, record.nbytes, 1), "cudaHostRegister")
        self._records[key] = record
        return record

    def unregister(self, record: HostRegistrationRecord) -> None:
        key = (record.arena_id, record.executor_id, record.process_generation)
        current = self._records.get(key)
        if current is None:
            return
        if current != record:
            raise HostRegistrationError("unregister record does not match adapter-owned registration")
        cudart = self._load_cudart()
        _check_cuda_status(cudart.cudaHostUnregister(current.address), "cudaHostUnregister")
        self._records.pop(key, None)

    def unregister_all(self) -> None:
        errors: list[str] = []
        for record in tuple(self._records.values()):
            try:
                self.unregister(record)
            except HostRegistrationError as exc:
                errors.append(str(exc))
        if errors:
            raise HostRegistrationError("; ".join(errors))

    @property
    def registration_count(self) -> int:
        return len(self._records)

    def _load_cudart(self) -> Any:
        if self._cudart is not None:
            return self._cudart
        try:
            import torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise HostRegistrationError("torch is required for cudaHostRegister") from exc
        if not torch.cuda.is_available():
            raise HostRegistrationError("CUDA is not available for cudaHostRegister")
        return torch.cuda.cudart()


def _check_cuda_status(status: Any, op: str) -> None:
    code = int(status[0] if isinstance(status, tuple) else status)
    if code != 0:
        raise HostRegistrationError(f"{op} failed with cuda status {code}")


def _non_negative(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive(value: int, name: str) -> int:
    value = _non_negative(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
