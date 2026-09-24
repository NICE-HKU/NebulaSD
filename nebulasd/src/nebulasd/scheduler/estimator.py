"""Pure scheduler cost and overlap estimators."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


GIB = 1024.0**3


@dataclass(frozen=True, slots=True)
class ProfilingSnapshot:
    """Immutable cost snapshot. Durations are seconds, bandwidth is GiB/s."""

    h2d_gib_per_s: float = 40.0
    d2h_gib_per_s: float = 40.0
    prepare_control_s: float = 0.0005
    dispatch_control_s: float = 0.0001
    ema_alpha: float = 0.2

    def __post_init__(self) -> None:
        for name in ("h2d_gib_per_s", "d2h_gib_per_s", "prepare_control_s", "dispatch_control_s", "ema_alpha"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.h2d_gib_per_s <= 0.0:
            raise ValueError("h2d_gib_per_s must be positive")
        if self.d2h_gib_per_s <= 0.0:
            raise ValueError("d2h_gib_per_s must be positive")
        if self.ema_alpha > 1.0:
            raise ValueError("ema_alpha must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ProfilingObservation:
    h2d_bytes: int = 0
    h2d_duration_s: float | None = None
    d2h_bytes: int = 0
    d2h_duration_s: float | None = None
    prepare_control_s: float | None = None
    dispatch_control_s: float | None = None


def update_profiling(snapshot: ProfilingSnapshot, observation: ProfilingObservation) -> ProfilingSnapshot:
    """Return a new profiling snapshot; scheduling decisions never mutate EMA."""

    alpha = snapshot.ema_alpha
    updates: dict[str, float] = {}
    if observation.h2d_duration_s is not None and observation.h2d_bytes > 0 and _finite_positive(observation.h2d_duration_s, "h2d_duration_s"):
        updates["h2d_gib_per_s"] = _ema(
            snapshot.h2d_gib_per_s,
            _gib_per_second(observation.h2d_bytes, observation.h2d_duration_s),
            alpha,
        )
    if observation.d2h_duration_s is not None and observation.d2h_bytes > 0 and _finite_positive(observation.d2h_duration_s, "d2h_duration_s"):
        updates["d2h_gib_per_s"] = _ema(
            snapshot.d2h_gib_per_s,
            _gib_per_second(observation.d2h_bytes, observation.d2h_duration_s),
            alpha,
        )
    if observation.prepare_control_s is not None:
        _finite_non_negative(observation.prepare_control_s, "prepare_control_s")
        updates["prepare_control_s"] = _ema(snapshot.prepare_control_s, observation.prepare_control_s, alpha)
    if observation.dispatch_control_s is not None:
        _finite_non_negative(observation.dispatch_control_s, "dispatch_control_s")
        updates["dispatch_control_s"] = _ema(snapshot.dispatch_control_s, observation.dispatch_control_s, alpha)
    return replace(snapshot, **updates)


@dataclass(frozen=True, slots=True)
class CostModel:
    profiling: ProfilingSnapshot = ProfilingSnapshot()

    def h2d_time(self, byte_count: int) -> float:
        return _copy_time(byte_count, self.profiling.h2d_gib_per_s)

    def d2h_time(self, byte_count: int) -> float:
        return _copy_time(byte_count, self.profiling.d2h_gib_per_s)

    def prepare_control_time(self, request_count: int = 1) -> float:
        return self.profiling.prepare_control_s * max(int(request_count), 1)

    def dispatch_control_time(self, command_count: int = 1) -> float:
        return self.profiling.dispatch_control_s * max(int(command_count), 1)

    def host_migration_ready_time(
        self,
        *,
        host_ready_time: float,
        copy_stream_available_time: float,
        h2d_bytes: int,
    ) -> float:
        return (
            max(float(host_ready_time), float(copy_stream_available_time))
            + self.h2d_time(h2d_bytes)
            + self.prepare_control_time(1)
        )

    def dirty_writeback_done_time(
        self,
        *,
        compute_done_time: float,
        copy_stream_available_time: float,
        d2h_bytes: int,
    ) -> float:
        return max(float(compute_done_time), float(copy_stream_available_time)) + self.d2h_time(d2h_bytes)

    def verify_start_time(self, *, draft_done_time: float, target_free_time: float, kv_ready_time: float) -> float:
        return max(float(draft_done_time), float(target_free_time), float(kv_ready_time))

    def visible_kv_wait(
        self,
        *,
        proposal_ready_time: float,
        target_compute_available_time: float,
        kv_ready_time: float,
    ) -> float:
        return max(0.0, float(kv_ready_time) - max(float(proposal_ready_time), float(target_compute_available_time)))


def _copy_time(byte_count: int, gib_per_s: float) -> float:
    return (max(int(byte_count), 0) / GIB) / float(gib_per_s)


def _gib_per_second(byte_count: int, duration_s: float) -> float:
    return (max(int(byte_count), 0) / GIB) / float(duration_s)


def _ema(old: float, new: float, alpha: float) -> float:
    return (1.0 - float(alpha)) * float(old) + float(alpha) * max(float(new), 0.0)


def _finite_positive(value: float, name: str) -> bool:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return True


def _finite_non_negative(value: float, name: str) -> None:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
