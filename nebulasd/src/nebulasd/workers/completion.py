"""Append-only cohort completions, published with native release/acquire."""
from dataclasses import dataclass
from struct import Struct
from nebulasd.ipc.mapped_segment import MappedSegment
from nebulasd.ipc.native import library
from .work import Outcome

_HEADER = Struct('<QQQQII')
_MEMBER = Struct('<IQQI')


def completion_bytes(count):
    return (8 + _HEADER.size + count * _MEMBER.size + 7) & ~7


def cohort_completion_budget(max_new_tokens):
    # Every batch contains >=1 member: charge a full header per member.
    return sum((2 * n + 4) * completion_bytes(1) for n in max_new_tokens)


@dataclass(frozen=True, slots=True)
class MemberCompletion:
    slot: int
    epoch: int
    round_id: int
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class WorkCompletion:
    worker_generation: int
    work_seq: int
    bank_epoch: int
    physical_done_ns: int
    bank_id: int
    members: tuple[MemberCompletion, ...]


class CompletionArena:
    """Engine reserves offsets; one assigned worker writes each offset once.

    No per-stage receipt, consumption ACK or GPU resource state lives here.
    Reset requires the caller's completed cohort barrier.
    """
    def __init__(self, capacity=None, *, descriptor=None):
        self.segment = (MappedSegment.create(capacity, 'work-completion:1') if descriptor is None
                        else MappedSegment(descriptor))
        if self.segment.descriptor.schema != 'work-completion:1':
            self.segment.close()
            raise ValueError('completion ABI mismatch')
        self.native = library()
        self.next_offset = 0

    def reserve(self, count):
        size = completion_bytes(count)
        if count <= 0 or self.next_offset + size > self.segment.descriptor.size:
            return None
        offset = self.next_offset
        self.next_offset += size
        return offset

    def _check(self, offset, size):
        if offset < 0 or offset % 8 or offset + size > self.segment.descriptor.size:
            raise ValueError('completion range outside arena')

    def publish(self, offset, record):
        size = completion_bytes(len(record.members))
        self._check(offset, size)
        if self.native.sd_load(self.segment.address + offset):
            raise RuntimeError('completion overwrite')
        raw = _HEADER.pack(record.worker_generation, record.work_seq, record.bank_epoch,
            record.physical_done_ns, record.bank_id, len(record.members))
        raw += b''.join(_MEMBER.pack(m.slot, m.epoch, m.round_id, int(m.outcome)) for m in record.members)
        self.segment.buffer[offset + 8:offset + 8 + len(raw)] = raw
        self.native.sd_store(self.segment.address + offset, 1)

    def read(self, offset):
        self._check(offset, 8 + _HEADER.size)
        if not self.native.sd_load(self.segment.address + offset):
            return None
        gen, seq, epoch, done, bank, count = _HEADER.unpack_from(self.segment.buffer, offset + 8)
        self._check(offset, completion_bytes(count))
        members = []
        for i in range(count):
            slot, req_epoch, round_id, outcome = _MEMBER.unpack_from(
                self.segment.buffer, offset + 8 + _HEADER.size + i * _MEMBER.size)
            members.append(MemberCompletion(slot, req_epoch, round_id, Outcome(outcome)))
        return WorkCompletion(gen, seq, epoch, done, bank, tuple(members))

    def close(self, *, unlink=False):
        self.segment.close()
        if unlink:
            self.segment.unlink()
