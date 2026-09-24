"""Internal single-owner commits for an already authorized batch lifetime.

These are row patches, not an atomic batch or a new scheduling protocol. The
caller validates command/source/Bank identity before preparing them. Exact
publication sequences fence replacement and duplicate completion without
decoding shared rows again. All checks precede all stores.
"""
from dataclasses import dataclass
import ctypes as C
from struct import Struct

from nebulasd.core.ids import OP_SEQ, U64
from .storage import FieldValue, TableProtocolError


@dataclass(frozen=True)
class PreparedPublication:
    partition: object
    rows: tuple

    @classmethod
    def capture(cls, table, kind, updates, *, expected_sequences=None, initial=False):
        partition = table.partition(kind)
        rows = []
        for slot, values in updates:
            current = (partition.read_publish_seq(slot) if expected_sequences is None
                       else expected_sequences[slot])
            if current == U64.invalid and not initial:
                raise TableProtocolError('prepared commit requires an authorized row')
            encoded = partition._prepare_fields(tuple(FieldValue(k, v) for k, v in values.items()))
            rows.append((slot, current, 0 if current == U64.invalid else OP_SEQ.next(current), encoded))
        if len({r[0] for r in rows}) != len(rows):
            raise TableProtocolError('prepared commit has duplicate rows')
        return cls(partition, tuple(rows))

    def commit(self):
        partition = self.partition
        for slot, expected, _, _ in self.rows:
            if partition.read_publish_seq(slot) != expected:
                raise TableProtocolError('stale or duplicate prepared publication')
        for slot, _, seq, encoded in self.rows:
            partition._publish_encoded(slot, seq, encoded)
        return {slot: seq for slot, _, seq, _ in self.rows}



class PreparedRow:
    def __init__(self, partition, row, constants, dynamic):
        partition._validate_row(row)
        self.partition, self.row = partition, row
        self.address = partition.segment.address + partition._row_base(row) if hasattr(partition, "segment") else None
        fields = partition._field_by_name
        names = tuple(constants) + tuple(dynamic)
        if len(set(names)) != len(names):
            raise ValueError('duplicate prepared field')
        self.lengths = tuple(fields[n].type.size for n in names)
        self.offsets = tuple(partition._offset_by_name[n] for n in names)
        self.buffer = C.create_string_buffer(sum(self.lengths))
        self.native_offsets = (C.c_uint * len(names))(*self.offsets)
        self.native_lengths = (C.c_uint * len(names))(*self.lengths)
        position, patches, chunks = 0, [], []
        for name, length in zip(names, self.lengths):
            if name in constants:
                raw = partition._encode_field(FieldValue(name, constants[name]))
                chunks.append(raw)
            else:
                typ = fields[name].type
                fmt = '<QII' if typ.name == 'arena_handle' else '<' + {
                    (1,False):'B',(4,False):'I',(4,True):'i',(8,False):'Q'}[length,typ.signed]
                patches.append((Struct(fmt),position,typ.name == 'arena_handle'))
                chunks.append(bytes(length))
            position += length
        self.buffer.raw = b''.join(chunks)
        self.patches = tuple(patches)

    def publish(self, values):
        if len(values) != len(self.patches):
            raise ValueError('prepared publication value count mismatch')
        for (codec,offset,handle),value in zip(self.patches,values):
            if handle:
                codec.pack_into(self.buffer,offset,value.offset,value.length,value.generation)
            else:
                codec.pack_into(self.buffer,offset,value)
        previous = (self.partition.native.sd_load(self.address) if self.address is not None
                    else self.partition.read_publish_seq(self.row))
        if previous == (1 << 64) - 2:
            raise TableProtocolError('publication sequence exhausted')
        sequence = 0 if previous == (1 << 64) - 1 else previous + 1
        self.partition._publish_packed(self.row,sequence,self)
        self.sequence = sequence

    def publication_snapshot(self):
        """Snapshot a fully specified row from its private publication buffer.

        Used by WORK HostReady receipts: the shared row may advance before the
        Engine consumes the receipt. Never read back another writer's version.
        """
        if len(self.offsets) != len(self.partition._field_by_name) - 1:
            raise ValueError('publication snapshot requires every payload field')
        payload = bytearray(self.partition._payload_size - 8)
        raw, position = self.buffer.raw, 0
        for offset, length in zip(self.offsets, self.lengths):
            payload[offset-8:offset-8+length] = raw[position:position+length]
            position += length
        return self.sequence, bytes(payload)
