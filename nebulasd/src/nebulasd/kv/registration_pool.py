"""Reference-counted portable host registration shared by local GPU workers."""

from threading import Lock

from .host_registration import CudaHostRegistrationAdapter


class HostRegistrationPool:
    def __init__(self, adapter=None) -> None:
        self.adapter = adapter or CudaHostRegistrationAdapter()
        self._entries = {}
        self._lock = Lock()

    def acquire(self, arena):
        key = (arena.address(), arena.descriptor.total_bytes)
        with self._lock:
            if key in self._entries:
                record, count = self._entries[key]
                self._entries[key] = (record, count + 1)
            else:
                record = self.adapter.register(arena, executor_id=f"portable-copy-pool:{key[0]}",
                                               process_generation=1)
                self._entries[key] = (record, 1)
            return record

    def release(self, record) -> None:
        key = (record.address, record.nbytes)
        with self._lock:
            actual, count = self._entries[key]
            if actual is not record:
                raise ValueError("registration pool receipt mismatch")
            if count == 1:
                self.adapter.unregister(record)
                del self._entries[key]
            else:
                self._entries[key] = (record, count - 1)
