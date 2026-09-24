"""Fixed Draft snapshot identity; no session objects or GPU addresses cross IPC.

Session identity is (request_slot, request_epoch). Arena generations are unique
within each payload kind, as in ArenaRouter. All referenced payloads remain
immutable through the all-owner retirement barrier, including imported readers.
"""
from dataclasses import dataclass
from struct import Struct
from .handles import ArenaHandle
from .ids import U32, U64

_HANDLE = Struct('<QII')
_ALLOCATION = Struct('<IIQQQQIIQ')
_IDENTITY = Struct('<IQQQIQQQII')


def require_handle(handle):
    if not isinstance(handle, ArenaHandle):
        raise TypeError('expected shared ArenaHandle')
    if handle.is_empty():
        raise ValueError('shared handle must not be empty')
    return handle


def pack_handle(handle):
    require_handle(handle)
    return _HANDLE.pack(handle.offset, handle.length, handle.generation)


@dataclass(frozen=True, slots=True)
class DraftHostAllocation:
    arena_id: int
    arena_generation: int
    layout_id: int
    host_slot_generation: int
    writer_lease_generation: int
    offset_blocks: int
    host_slot: int
    capacity_blocks: int
    block_size: int

    byte_size = _ALLOCATION.size

    def __post_init__(self):
        for name in ('arena_id', 'arena_generation', 'host_slot', 'capacity_blocks'):
            U32.validate(getattr(self, name))
        for name in ('layout_id', 'host_slot_generation', 'writer_lease_generation', 'offset_blocks', 'block_size'):
            U64.validate(getattr(self, name))
        if not self.capacity_blocks or not self.block_size:
            raise ValueError('HostKV capacity and block size must be positive')

    def to_bytes(self):
        return _ALLOCATION.pack(*(getattr(self, n) for n in self.__dataclass_fields__))

    @classmethod
    def from_bytes(cls, raw):
        return cls(*_ALLOCATION.unpack(raw))


@dataclass(frozen=True, slots=True)
class DraftSnapshotIdentity:
    request_slot: int
    request_epoch: int
    round_id: int
    op_seq: int
    worker_id: int
    worker_generation: int
    owner_epoch: int
    snapshot_version: int
    logical_kv_len: int
    valid_blocks: int
    allocation: DraftHostAllocation

    byte_size = _IDENTITY.size + DraftHostAllocation.byte_size

    def __post_init__(self):
        for name in ('request_slot', 'worker_id', 'logical_kv_len', 'valid_blocks'):
            U32.validate(getattr(self, name))
        for name in ('request_epoch', 'round_id', 'op_seq', 'worker_generation', 'owner_epoch', 'snapshot_version'):
            U64.validate(getattr(self, name))
        if not isinstance(self.allocation, DraftHostAllocation):
            raise TypeError('expected DraftHostAllocation')
        if self.valid_blocks != (self.logical_kv_len + self.allocation.block_size - 1) // self.allocation.block_size:
            raise ValueError('snapshot valid blocks do not match actual KV length')
        if self.valid_blocks > self.allocation.capacity_blocks:
            raise ValueError('snapshot exceeds HostKV allocation')

    def validate_expected(self, expected):
        if self != expected:
            raise ValueError('stale Draft snapshot/request/owner/allocation identity')

    def to_bytes(self):
        return _IDENTITY.pack(*(getattr(self, n) for n in self.__dataclass_fields__ if n != 'allocation')) + self.allocation.to_bytes()

    @classmethod
    def from_bytes(cls, raw):
        if len(raw) != cls.byte_size:
            raise ValueError('invalid snapshot identity size')
        return cls(*_IDENTITY.unpack(raw[:_IDENTITY.size]), DraftHostAllocation.from_bytes(raw[_IDENTITY.size:]))


@dataclass(frozen=True, slots=True)
class DraftSnapshot:
    identity: DraftSnapshotIdentity
    prompt_handle: ArenaHandle
    generation_config_handle: ArenaHandle
    committed_output_handle: ArenaHandle
    proposal_handle: ArenaHandle
    prompt_count: int
    committed_output_count: int
    proposal_count: int

    # Independent payload version, covered by the shared arena schema as well.
    version = 1
    _header = Struct('<IIII')
    byte_size = _header.size + DraftSnapshotIdentity.byte_size + 4 * _HANDLE.size

    def __post_init__(self):
        if not isinstance(self.identity, DraftSnapshotIdentity):
            raise TypeError('expected DraftSnapshotIdentity')
        for n in ('prompt_count', 'committed_output_count', 'proposal_count'):
            U32.validate(getattr(self, n))
            if not getattr(self, n):
                raise ValueError('completed Draft snapshot requires prompt, output anchor and proposal')
        for n in ('prompt_handle', 'generation_config_handle', 'committed_output_handle', 'proposal_handle'):
            require_handle(getattr(self, n))
        if self.prompt_handle.length != self.prompt_count * 4 or self.committed_output_handle.length != self.committed_output_count * 4:
            raise ValueError('snapshot token handle/count mismatch')
        if self.identity.logical_kv_len != self.prompt_count + self.committed_output_count + self.proposal_count - 1:
            raise ValueError('snapshot must describe P + C + Q[:-1]')

    def to_bytes(self):
        return (self._header.pack(self.version, self.prompt_count, self.committed_output_count, self.proposal_count)
                + self.identity.to_bytes() + b''.join(pack_handle(getattr(self, n)) for n in
                    ('prompt_handle', 'generation_config_handle', 'committed_output_handle', 'proposal_handle')))

    @classmethod
    def from_bytes(cls, raw):
        if len(raw) != cls.byte_size:
            raise ValueError('invalid Draft snapshot size')
        version, *counts = cls._header.unpack(raw[:cls._header.size])
        if version != cls.version:
            raise ValueError('invalid Draft snapshot version')
        begin = cls._header.size
        identity = DraftSnapshotIdentity.from_bytes(raw[begin:begin + DraftSnapshotIdentity.byte_size])
        begin += DraftSnapshotIdentity.byte_size
        handles = [ArenaHandle(*_HANDLE.unpack(raw[i:i + 16])) for i in range(begin, len(raw), 16)]
        return cls(identity, *handles, *counts)


def require_snapshot_handle(handle):
    require_handle(handle)
    if handle.length != DraftSnapshot.byte_size:
        raise ValueError('snapshot handle must reference exactly one fixed metadata record')
    return handle
