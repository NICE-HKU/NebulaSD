"""Named shared segments with a checked cold header and aligned native payload."""

import ctypes as C
from dataclasses import dataclass
import json
from multiprocessing.shared_memory import SharedMemory

HEADER_BYTES = 4096


@dataclass(frozen=True)
class SegmentDescriptor:
    name: str
    size: int
    schema: str


class MappedSegment:
    def __init__(self, descriptor, *, create=False):
        self.descriptor = descriptor
        self.shm = SharedMemory(name=descriptor.name, create=create,
                                size=HEADER_BYTES + descriptor.size if create else 0)
        expected = json.dumps(dict(magic="STARSD_NATIVE", abi=1, size=descriptor.size,
                                   schema=descriptor.schema), sort_keys=True).encode()
        try:
            if len(expected) >= HEADER_BYTES or self.shm.size < HEADER_BYTES + descriptor.size:
                raise ValueError("invalid shared segment size")
            if create:
                self.shm.buf[:len(expected)] = expected
            if bytes(self.shm.buf[:HEADER_BYTES]).rstrip(b"\0") != expected:
                raise ValueError("shared segment header/schema mismatch")
            self.buffer = self.shm.buf[HEADER_BYTES:HEADER_BYTES + descriptor.size]
            self.address = C.addressof(C.c_char.from_buffer(self.buffer))
        except BaseException:
            self.shm.close()
            if create:
                self.shm.unlink()
            raise

    @classmethod
    def create(cls, size, schema):
        from uuid import uuid4
        if size <= 0:
            raise ValueError("segment size must be positive")
        return cls(SegmentDescriptor("sd_" + uuid4().hex[:24], size, schema), create=True)

    def close(self):
        self.buffer.release()
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
