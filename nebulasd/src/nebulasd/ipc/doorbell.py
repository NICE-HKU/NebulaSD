"""Thread-level doorbell used by tests and process-local schedulers."""

from __future__ import annotations

from threading import Condition

from nebulasd.core.ids import U64


class Doorbell:
    """Wake-up counter paired with state-change rings."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._generation = 0

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def ring(self) -> int:
        with self._condition:
            self._generation = U64.next(self._generation)
            self._condition.notify_all()
            return self._generation

    def wait_after(self, last_seen: int, timeout: float | None = None) -> int:
        U64.validate(last_seen)
        with self._condition:
            if self._generation == last_seen:
                self._condition.wait(timeout=timeout)
            return self._generation

