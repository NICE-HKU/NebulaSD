"""Independent management path for slow worker commands."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from nebulasd.core.enums import validate_enum
from nebulasd.core.ids import REQUEST_EPOCH, REQUEST_SLOT, WORKER_ID

from .protocol import ManagementCommandKind


@dataclass(frozen=True, slots=True)
class ManagementCommand:
    kind: ManagementCommandKind
    worker_id: int
    request_slot: int | None = None
    request_epoch: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", validate_enum(ManagementCommandKind, self.kind))
        WORKER_ID.validate(self.worker_id)
        if self.request_slot is not None:
            REQUEST_SLOT.validate(self.request_slot)
        if self.request_epoch is not None:
            REQUEST_EPOCH.validate(self.request_epoch)


class ManagementQueue:
    """Slow-path queue kept separate from hot command rings."""

    def __init__(self) -> None:
        self._queue: deque[ManagementCommand] = deque()

    def submit(self, command: ManagementCommand) -> None:
        if not isinstance(command, ManagementCommand):
            raise TypeError("command must be a ManagementCommand")
        self._queue.append(command)

    def pop(self) -> ManagementCommand | None:
        if not self._queue:
            return None
        return self._queue.popleft()
