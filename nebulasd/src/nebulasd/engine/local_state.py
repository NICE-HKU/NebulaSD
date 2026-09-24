"""Immutable owner-local rows with constant-time field lookup.

Keep the original stable payload for the native scheduler's numeric projection.
This is a scheduling representation, never a second shared publication plane.
"""
from dataclasses import dataclass
from types import MappingProxyType
from struct import Struct

from nebulasd.table.storage import FieldRead


@dataclass(frozen=True, slots=True)
class SchedulingRow:
    block_kind: object
    row: int
    publish_seq: int
    fields: tuple
    payload: bytes | None
    _values: object

    @classmethod
    def observed(cls, snapshot):
        return cls(snapshot.block_kind, snapshot.row, snapshot.publish_seq,
                   snapshot.fields, snapshot.payload,
                   MappingProxyType({f.name: f.value for f in snapshot.fields}))

    @classmethod
    def local(cls, kind, slot, seq, values, payload=None):
        values = dict(values)
        return cls(kind, slot, seq, LocalFields(values),
                   payload, MappingProxyType(values))

    def get(self, name):
        return self._values[name]


class LocalFields:
    """Only diagnostics/observer iteration materializes field wrappers."""
    __slots__ = ('values',)

    def __init__(self, values):
        self.values = MappingProxyType(values)

    def __iter__(self):
        return (FieldRead(k, v) for k, v in self.values.items())


class PayloadDecoder:
    """Shared Worker row field offsets; Engine-local state never uses this."""
    def __init__(self, partition):
        self.decode = {}
        for name, field in partition._field_by_name.items():
            if name == 'publish_seq':
                continue
            typ = field.type
            handle = typ.name == 'arena_handle'
            fmt = '<QII' if handle else '<' + {
                (1, False): 'B', (4, False): 'I', (4, True): 'i', (8, False): 'Q'
            }[typ.size, typ.signed]
            self.decode[name] = (Struct(fmt).unpack_from, partition._offset_by_name[name] - 8, handle)


class PayloadFields:
    """Compatibility iteration is cold; native projection reads payload directly."""
    __slots__ = ('payload', '_codec', '_values')

    def __init__(self, row):
        self.payload, self._codec, self._values = row.payload, row._codec, row._values

    def __iter__(self):
        return (FieldRead(name, PayloadRow.get(self, name)) for name in self._codec)


@dataclass(frozen=True, slots=True)
class PayloadRow:
    block_kind: object
    row: int
    publish_seq: int
    payload: bytes
    _codec: object
    _values: dict
    fields: object = None

    def __post_init__(self):
        object.__setattr__(self, 'fields', PayloadFields(self))

    def get(self, name):
        try:
            return self._values[name]
        except KeyError:
            unpack, offset, handle = self._codec[name]
            values = unpack(self.payload, offset)
            if handle:
                from nebulasd.core.handles import ArenaHandle
                result = ArenaHandle(*values)
            else:
                result = values[0]
            self._values[name] = result
            return result


class SchedulingSnapshotReader:
    """Owner-private read buffers, immutable payloads, decode only accessed fields.

    Uses the identical native stable-read primitive and retry bound. Workers,
    publishers and the native scheduler ABI do not change.
    """
    def __init__(self):
        self.partitions = {}

    def __call__(self, partition, row):
        import ctypes as C
        from nebulasd.table.native_storage import NativeTablePartition
        from nebulasd.table.storage import StableReadConflict
        if not isinstance(partition, NativeTablePartition):
            return partition.read_stable(row)
        partition._validate_row(row)
        cached = self.partitions.get(partition)
        if cached is None:
            codec = {}
            for field in partition._decode_fields:
                typ = field.field.type
                handle = typ.name == 'arena_handle'
                fmt = '<QII' if handle else '<' + {
                    (1, False): 'B', (4, False): 'I', (4, True): 'i', (8, False): 'Q'
                }[typ.size, typ.signed]
                codec[field.field.name] = (Struct(fmt).unpack_from, field.offset-8, handle)
            cached = (C.create_string_buffer(partition._payload_size-8), C.c_uint64(), codec)
            self.partitions[partition] = cached
        output, seq, codec = cached
        for _ in range(16):
            if partition.native.sd_table_read(partition.segment.address + partition._row_base(row),
                                             partition._payload_size, output, C.byref(seq)):
                return PayloadRow(partition.block_kind, row, seq.value, output.raw, codec, {})
        raise StableReadConflict('native row changed during stable read')
