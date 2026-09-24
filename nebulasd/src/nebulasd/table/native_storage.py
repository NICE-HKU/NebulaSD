"""Native partitions preserve typed owner writers and the existing observation ABI."""

import ctypes as C

from nebulasd.core.ids import OP_SEQ, U64
from nebulasd.ipc.mapped_segment import MappedSegment
from nebulasd.ipc.native import library
from nebulasd.ipc.state_change_ring import StateChangeEntry
from .storage import (TablePartition, BlockSnapshot, TableProtocolError, StableReadConflict,
                      RequestSchedulingTable, WorkerSchedulingRegistry, layout_fingerprint)


class NativeTablePartition(TablePartition):
    def __init__(self, *, layout, block_kind, capacity_rows, ring=None, descriptor=None):
        self.native = library()
        # Reuse scalar validation/decoding, but not Python's bytearray transport.
        super().__init__(layout=layout, block_kind=block_kind, capacity_rows=capacity_rows)
        self._ring = ring
        self._encoded_layouts = {}
        schema = f"table:{int(block_kind)}:{capacity_rows}:{layout_fingerprint(layout)}"
        self.segment = (MappedSegment.create(self.byte_size, schema) if descriptor is None
                        else MappedSegment(descriptor))
        if self.segment.descriptor.schema != schema:
            self.segment.close()
            raise TableProtocolError("native table layout mismatch")
        if descriptor is None:
            self.segment.buffer[:] = self._bytes
        self._bytes = self.segment.buffer

    def read_publish_seq(self, row):
        self._validate_row(row)
        return self.native.sd_load(self.segment.address + self._row_base(row))

    def _publish(self, row, publish_seq, fields):
        self._validate_row(row)
        OP_SEQ.validate(publish_seq)
        encoded = tuple((self._offset_by_name[f.name], self._encode_field(f)) for f in fields)
        current = self.read_publish_seq(row)
        if current != U64.invalid and not OP_SEQ.is_newer(publish_seq, current):
            raise TableProtocolError("native publish_seq must advance")
        self._publish_encoded(row, publish_seq, self._pack_fields(encoded))

    def _prepare_fields(self, fields):
        return self._pack_fields(tuple((self._offset_by_name[f.name], self._encode_field(f)) for f in fields))

    def _pack_fields(self, encoded):
        count = len(encoded)
        key = tuple((offset, len(raw)) for offset, raw in encoded)
        layout = self._encoded_layouts.get(key)
        if layout is None:
            layout = ((C.c_uint * count)(*(offset for offset, length in key)),
                      (C.c_uint * count)(*(length for offset, length in key)))
            self._encoded_layouts[key] = layout
        offsets, lengths = layout
        data = b"".join(raw for offset, raw in encoded)
        return offsets, lengths, data, count

    def _publish_encoded(self, row, publish_seq, encoded):
        self.native.sd_table_publish(self.segment.address + self._row_base(row), publish_seq, *encoded)
        if self._ring is not None:
            self._ring.push(StateChangeEntry(self.block_kind, row, publish_seq))

    def _publish_packed(self, row, publish_seq, prepared):
        ring = self._ring
        if ring is None or not hasattr(ring, 'address'):
            self.native.sd_table_publish(prepared.address, publish_seq,
                prepared.native_offsets, prepared.native_lengths, prepared.buffer, len(prepared.offsets))
            if ring is not None:
                ring.push(StateChangeEntry(self.block_kind, row, publish_seq))
        else:
            if self.native.sd_table_publish_notify(prepared.address, publish_seq,
                    prepared.native_offsets, prepared.native_lengths, prepared.buffer, len(prepared.offsets),
                    ring.address, ring.capacity, int(self.block_kind), row, ring.doorbell is not None):
                ring.doorbell.ring()

    def read_stable(self, row, *, include_cold=True, max_retries=16, field_names=None):
        self._validate_row(row)
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        output, seq = C.create_string_buffer(self._payload_size - 8), C.c_uint64()
        for _ in range(max_retries):
            if self.native.sd_table_read(self.segment.address + self._row_base(row),
                                         self._payload_size, output, C.byref(seq)):
                payload = output.raw
                return BlockSnapshot(self.block_kind, row, seq.value,
                    self._decode_payload(payload, include_cold=include_cold, field_names=field_names),
                    payload if include_cold and field_names is None else None)
        raise StableReadConflict("native row changed during stable read")


def mapped_table(reference, *, descriptors=None, ring=None):
    """Create/attach partitions from the frozen layout manifest of a typed table."""
    partitions = {}
    try:
        for kind, old in reference._partitions.items():
            partitions[kind] = NativeTablePartition(layout=old.layout, block_kind=kind,
                capacity_rows=old.capacity_rows, ring=ring,
                descriptor=None if descriptors is None else descriptors[kind])
    except BaseException:
        close_table_partitions(partitions, unlink=descriptors is None)
        raise
    reference._partitions = partitions
    return reference


def table_descriptors(table):
    return {kind: p.segment.descriptor for kind, p in table._partitions.items()}


def close_table_partitions(partitions, *, unlink=False):
    for partition in partitions.values():
        partition.segment.close()
        if unlink:
            partition.segment.unlink()


def request_table(capacity, *, descriptors=None, ring=None):
    return mapped_table(RequestSchedulingTable(capacity), descriptors=descriptors, ring=ring)


def worker_table(capacity, *, descriptors=None, ring=None):
    return mapped_table(WorkerSchedulingRegistry(capacity), descriptors=descriptors, ring=ring)
