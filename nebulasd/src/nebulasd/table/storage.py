"""Bytearray-backed scheduling observation tables."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from hashlib import blake2b
from time import sleep
from typing import Iterable

from nebulasd.core.enums import StateChangeBlockKind
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import OP_SEQ, REQUEST_SLOT, U8 as CORE_U8, U32, U64
from nebulasd.ipc.doorbell import Doorbell
from nebulasd.ipc.state_change_ring import StateChangeEntry, StateChangeRing

from .layout import (
    DRAFT_HOSTKV_BLOCK, DRAFT_D2H_BLOCK, DRAFT_H2D_BLOCK,
    DRAFT_COPY_RUNTIME_BLOCK, DRAFT_BANK_BLOCK,
    ABI_VERSION,
    ARENA_HANDLE,
    ENDIANNESS,
    I32,
    U8 as LAYOUT_U8,
    U32 as LAYOUT_U32,
    U64 as LAYOUT_U64,
    BANK_BLOCK,
    D2H_BLOCK,
    DISPATCH_BLOCK,
    DRAFT_RUNTIME_BLOCK,
    DRAFT_WORKER_BLOCK,
    ENGINE_BLOCK,
    H2D_BLOCK,
    HOSTKV_ALLOCATION_BLOCK,
    REQUEST_BLOCKS,
    TARGET_COMPUTE_BLOCK,
    TARGET_COMPUTE_RUNTIME_BLOCK,
    TARGET_COPY_RUNTIME_BLOCK,
    WORKER_BLOCKS,
    WORKER_COMMON_BLOCK,
    FieldDef,
    ScalarType,
    StructLayout,
)


SEGMENT_MAGIC = 0x5354_4152_5344_4E58

REQUEST_BLOCK_KIND_TO_LAYOUT: dict[StateChangeBlockKind, StructLayout] = {
    StateChangeBlockKind.REQUEST_ENGINE: ENGINE_BLOCK,
    StateChangeBlockKind.REQUEST_DISPATCH: DISPATCH_BLOCK,
    StateChangeBlockKind.REQUEST_DRAFT: DRAFT_WORKER_BLOCK,
    StateChangeBlockKind.REQUEST_TARGET_COMPUTE: TARGET_COMPUTE_BLOCK,
    StateChangeBlockKind.REQUEST_D2H: D2H_BLOCK,
    StateChangeBlockKind.REQUEST_H2D: H2D_BLOCK,
    StateChangeBlockKind.REQUEST_HOSTKV: HOSTKV_ALLOCATION_BLOCK,
    StateChangeBlockKind.REQUEST_DRAFT_HOSTKV: DRAFT_HOSTKV_BLOCK,
    StateChangeBlockKind.REQUEST_DRAFT_D2H: DRAFT_D2H_BLOCK,
    StateChangeBlockKind.REQUEST_DRAFT_H2D: DRAFT_H2D_BLOCK,
}

WORKER_BLOCK_KIND_TO_LAYOUT: dict[StateChangeBlockKind, StructLayout] = {
    StateChangeBlockKind.WORKER_COMMON: WORKER_COMMON_BLOCK,
    StateChangeBlockKind.WORKER_DRAFT_RUNTIME: DRAFT_RUNTIME_BLOCK,
    StateChangeBlockKind.WORKER_TARGET_COMPUTE_RUNTIME: TARGET_COMPUTE_RUNTIME_BLOCK,
    StateChangeBlockKind.WORKER_TARGET_COPY_RUNTIME: TARGET_COPY_RUNTIME_BLOCK,
    StateChangeBlockKind.WORKER_BANK: BANK_BLOCK,
    StateChangeBlockKind.WORKER_DRAFT_COPY_RUNTIME: DRAFT_COPY_RUNTIME_BLOCK,
    StateChangeBlockKind.WORKER_DRAFT_BANK: DRAFT_BANK_BLOCK,
}


class TableProtocolError(RuntimeError):
    """Raised when an observation-plane publication violates ownership/fences."""


class StableReadConflict(RuntimeError):
    """Raised when a row cannot be read without concurrent mutation."""


@dataclass(frozen=True, slots=True)
class TableLayoutRecord:
    block_kind: StateChangeBlockKind
    name: str
    size: int
    row_stride: int
    fingerprint: int


@dataclass(frozen=True, slots=True)
class TableSegmentHeader:
    magic: int
    abi_version: int
    table_kind: str
    capacity_rows: int
    byte_size: int
    layouts: tuple[TableLayoutRecord, ...]

    def validate(self, expected_layouts: Iterable[TableLayoutRecord]) -> None:
        if self.magic != SEGMENT_MAGIC:
            raise TableProtocolError("invalid table segment magic")
        if self.abi_version != ABI_VERSION:
            raise TableProtocolError("invalid table segment ABI version")
        expected = tuple(expected_layouts)
        if self.layouts != expected:
            raise TableProtocolError("table segment layout mismatch")


@dataclass(frozen=True, slots=True)
class FieldValue:
    name: str
    value: int | ArenaHandle


@dataclass(frozen=True, slots=True)
class FieldRead:
    name: str
    value: int | ArenaHandle


@dataclass(frozen=True, slots=True)
class BlockSnapshot:
    block_kind: StateChangeBlockKind
    row: int
    publish_seq: int
    fields: tuple[FieldRead, ...]
    # Optional bytes from the SAME successful native stable read, never a live view.
    # Partial-field snapshots omit this payload to preserve their visibility contract.
    payload: bytes | None = dataclass_field(default=None, compare=False, repr=False)

    def get(self, name: str) -> int | ArenaHandle:
        for field in self.fields:
            if field.name == name:
                return field.value
        raise KeyError(name)


def _strict_int(name: str, value: int, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _encode_scalar(field: FieldDef, value: int | ArenaHandle) -> bytes:
    scalar = field.type
    if scalar is ARENA_HANDLE:
        if not isinstance(value, ArenaHandle):
            raise TypeError(f"{field.name} must be an ArenaHandle")
        return (
            value.offset.to_bytes(8, ENDIANNESS)
            + value.length.to_bytes(4, ENDIANNESS)
            + value.generation.to_bytes(4, ENDIANNESS)
        )

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field.name} must be an int")
    if scalar is I32:
        _strict_int(field.name, value, minimum=-(1 << 31), maximum=(1 << 31) - 1)
        return value.to_bytes(scalar.size, ENDIANNESS, signed=True)
    # Same UIntDomain range (all-ones is invalid), without recomputing its
    # property chain for every field of every row. Type checks remain above.
    maximum = _UNSIGNED_MAXIMUM[scalar.size]
    if value < 0 or value > maximum:
        raise ValueError(f"{field.name} must be in [0, {maximum}]")
    return value.to_bytes(scalar.size, ENDIANNESS, signed=False)


_UNSIGNED_MAXIMUM = {1: CORE_U8.max_valid, 4: U32.max_valid, 8: U64.max_valid}


def _decode_scalar(scalar: ScalarType, raw: bytes) -> int | ArenaHandle:
    if scalar is ARENA_HANDLE:
        return ArenaHandle(
            offset=int.from_bytes(raw[0:8], ENDIANNESS),
            length=int.from_bytes(raw[8:12], ENDIANNESS),
            generation=int.from_bytes(raw[12:16], ENDIANNESS),
        )
    return int.from_bytes(raw, ENDIANNESS, signed=scalar.signed)


def layout_fingerprint(layout: StructLayout) -> int:
    digest = blake2b(digest_size=8)
    digest.update(layout.name.encode("ascii"))
    digest.update(layout.owner.encode("ascii"))
    digest.update(layout.version.to_bytes(4, ENDIANNESS))
    digest.update(layout.size.to_bytes(4, ENDIANNESS))
    digest.update(layout.row_stride.to_bytes(4, ENDIANNESS))
    for field_layout in layout.field_layouts:
        field = field_layout.field
        digest.update(field.name.encode("ascii"))
        digest.update(field.type.name.encode("ascii"))
        digest.update(field_layout.offset.to_bytes(4, ENDIANNESS))
        digest.update(field.type.size.to_bytes(4, ENDIANNESS))
        digest.update(bytes((1 if field.hot else 0, 1 if field.required else 0)))
    return int.from_bytes(digest.digest(), ENDIANNESS)


def layout_record(block_kind: StateChangeBlockKind, layout: StructLayout) -> TableLayoutRecord:
    return TableLayoutRecord(
        block_kind=block_kind,
        name=layout.name,
        size=layout.size,
        row_stride=layout.row_stride,
        fingerprint=layout_fingerprint(layout),
    )


class TablePartition:
    """Fixed-layout partition where every row has one owner and publish marker."""

    def __init__(
        self,
        *,
        layout: StructLayout,
        block_kind: StateChangeBlockKind,
        capacity_rows: int,
        ring: StateChangeRing | None = None,
        doorbell: Doorbell | None = None,
    ) -> None:
        REQUEST_SLOT.validate(capacity_rows)
        self.layout = layout
        self.block_kind = block_kind
        self.capacity_rows = capacity_rows
        self._ring = ring
        self._doorbell = doorbell
        # StructLayout is immutable, but its properties compute offsets anew.
        # Compile those once; never cache mutable row values or fence results.
        self._row_stride = layout.row_stride
        self._payload_size = layout.size
        field_layouts = layout.field_layouts
        self._decode_fields = tuple(f for f in field_layouts if f.field.name != "publish_seq")
        self._hot_decode_fields = tuple(f for f in self._decode_fields if f.field.hot)
        self._selected_fields = {}
        self._bytes = bytearray(self._row_stride * capacity_rows)
        self._field_by_name = {field.name: field for field in layout.fields}
        self._offset_by_name = {field.field.name: field.offset for field in field_layouts}
        invalid = U64.invalid.to_bytes(8, ENDIANNESS)
        for row in range(capacity_rows):
            base = self._row_base(row)
            self._bytes[base : base + 8] = invalid

    @property
    def byte_size(self) -> int:
        return len(self._bytes)

    def _publish(self, row: int, publish_seq: int, fields: Iterable[FieldValue]) -> None:
        self._validate_row(row)
        OP_SEQ.validate(publish_seq)
        encoded = tuple((field_value.name, self._encode_field(field_value)) for field_value in fields)

        current = self.read_publish_seq(row)
        if current != U64.invalid and not OP_SEQ.is_newer(publish_seq, current):
            raise TableProtocolError(f"{self.layout.name}[{row}] publish_seq must increase monotonically")

        self._publish_encoded(row, publish_seq, encoded)

    def _prepare_fields(self, fields):
        return tuple((f.name, self._encode_field(f)) for f in fields)

    def _publish_encoded(self, row, publish_seq, encoded):
        # Internal prepared publication: row, fields and sequence were checked
        # before entering this store-only portion of the seqlock protocol.
        base = self._row_base(row)
        self._bytes[base : base + 8] = U64.invalid.to_bytes(8, ENDIANNESS)
        for name, raw in encoded:
            offset = self._offset_by_name[name]
            self._bytes[base + offset : base + offset + len(raw)] = raw
        publish_raw = publish_seq.to_bytes(8, ENDIANNESS)
        self._bytes[base : base + 8] = publish_raw

        if self._ring is not None:
            self._ring.push(StateChangeEntry(self.block_kind, row, publish_seq))
        if self._doorbell is not None:
            self._doorbell.ring()

    def _publish_packed(self, row, publish_seq, prepared):
        base = self._row_base(row)
        self._bytes[base:base+8] = U64.invalid.to_bytes(8, ENDIANNESS)
        position = 0
        for offset, length in zip(prepared.offsets, prepared.lengths):
            self._bytes[base+offset:base+offset+length] = prepared.buffer[position:position+length]
            position += length
        self._bytes[base:base+8] = publish_seq.to_bytes(8, ENDIANNESS)
        if self._ring is not None:
            self._ring.push(StateChangeEntry(self.block_kind,row,publish_seq))
        # Prepared publishers coalesce doorbells after their bounded batch.

    def read_stable(self, row: int, *, include_cold: bool = True, max_retries: int = 16, field_names=None) -> BlockSnapshot:
        self._validate_row(row)
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")

        for _ in range(max_retries):
            seq_a = self.read_publish_seq(row)
            if seq_a == U64.invalid:
                sleep(0)
                continue
            payload = self._copy_payload(row)
            seq_b = self.read_publish_seq(row)
            if seq_a == seq_b and seq_b != U64.invalid:
                fields = self._decode_payload(payload, include_cold=include_cold, field_names=field_names)
                return BlockSnapshot(self.block_kind, row, seq_b, fields)
            sleep(0)
        raise StableReadConflict(f"{self.layout.name}[{row}] changed during stable read")

    def read_publish_seq(self, row: int) -> int:
        self._validate_row(row)
        base = self._row_base(row)
        return int.from_bytes(self._bytes[base : base + 8], ENDIANNESS)

    def publish_sequences(self) -> tuple[int, ...]:
        return tuple(self.read_publish_seq(row) for row in range(self.capacity_rows))

    def _encode_field(self, field_value: FieldValue) -> bytes:
        if field_value.name == "publish_seq":
            raise TableProtocolError("publish_seq is controlled by the publication protocol")
        field = self._field_by_name.get(field_value.name)
        if field is None:
            raise KeyError(f"{self.layout.name} has no field {field_value.name!r}")
        return _encode_scalar(field, field_value.value)

    def _copy_payload(self, row: int) -> bytes:
        base = self._row_base(row)
        return bytes(self._bytes[base + 8 : base + self._payload_size])

    def _decode_payload(self, payload: bytes, *, include_cold: bool, field_names=None) -> tuple[FieldRead, ...]:
        reads: list[FieldRead] = []
        fields = self._decode_fields if include_cold else self._hot_decode_fields
        if field_names is not None:
            fields = self._selected_fields.get(field_names)
            if fields is None:
                by_name = {f.field.name: f for f in self._decode_fields}
                fields = tuple(by_name[name] for name in field_names)
                self._selected_fields[field_names] = fields
        for field_layout in fields:
            field = field_layout.field
            start = field_layout.offset - 8
            end = start + field.type.size
            reads.append(FieldRead(field.name, _decode_scalar(field.type, payload[start:end])))
        return tuple(reads)

    def _row_base(self, row: int) -> int:
        return row * self._row_stride

    def _validate_row(self, row: int) -> None:
        REQUEST_SLOT.validate(row)
        if row >= self.capacity_rows:
            raise IndexError(f"row {row} outside {self.layout.name} capacity {self.capacity_rows}")


class RequestSchedulingTable:
    """Request-indexed observation table, partitioned by writer owner."""

    def __init__(
        self,
        max_active_requests: int,
        *,
        ring: StateChangeRing | None = None,
        doorbell: Doorbell | None = None,
    ) -> None:
        REQUEST_SLOT.validate(max_active_requests)
        if max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")
        self.max_active_requests = max_active_requests
        self._partitions = {
            kind: TablePartition(
                layout=layout,
                block_kind=kind,
                capacity_rows=max_active_requests,
                ring=ring,
                doorbell=doorbell,
            )
            for kind, layout in REQUEST_BLOCK_KIND_TO_LAYOUT.items()
        }
        self.header = TableSegmentHeader(
            magic=SEGMENT_MAGIC,
            abi_version=ABI_VERSION,
            table_kind="request",
            capacity_rows=max_active_requests,
            byte_size=self.byte_size,
            layouts=tuple(layout_record(kind, layout) for kind, layout in REQUEST_BLOCK_KIND_TO_LAYOUT.items()),
        )
        self.header.validate(self.header.layouts)

    @property
    def byte_size(self) -> int:
        return sum(partition.byte_size for partition in self._partitions.values())

    def partition(self, block_kind: StateChangeBlockKind) -> TablePartition:
        try:
            return self._partitions[block_kind]
        except KeyError as exc:
            raise KeyError(f"{block_kind!r} is not a request block") from exc

    def _publish_owned(
        self,
        *,
        owner: str,
        block_kind: StateChangeBlockKind,
        row: int,
        publish_seq: int,
        fields: Iterable[FieldValue],
    ) -> None:
        partition = self.partition(block_kind)
        if partition.layout.owner != owner:
            raise TableProtocolError(f"{owner!r} cannot publish {partition.layout.name} owned by {partition.layout.owner!r}")
        partition._publish(row, publish_seq, fields)

    def request_partitions(self) -> tuple[TablePartition, ...]:
        return tuple(self._partitions[kind] for kind in REQUEST_BLOCK_KIND_TO_LAYOUT)


class WorkerSchedulingRegistry:
    """Worker-indexed observation table with isolated runtime partitions."""

    def __init__(
        self,
        max_workers: int,
        *,
        ring: StateChangeRing | None = None,
        doorbell: Doorbell | None = None,
        bank_rows_per_worker: int = 2,
    ) -> None:
        REQUEST_SLOT.validate(max_workers)
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if bank_rows_per_worker <= 0:
            raise ValueError("bank_rows_per_worker must be positive")
        self.max_workers = max_workers
        self.bank_rows_per_worker = bank_rows_per_worker
        self._partitions = {
            kind: TablePartition(
                layout=layout,
                block_kind=kind,
                capacity_rows=(max_workers * bank_rows_per_worker if kind in (StateChangeBlockKind.WORKER_BANK, StateChangeBlockKind.WORKER_DRAFT_BANK) else max_workers),
                ring=ring,
                doorbell=doorbell,
            )
            for kind, layout in WORKER_BLOCK_KIND_TO_LAYOUT.items()
        }
        self.header = TableSegmentHeader(
            magic=SEGMENT_MAGIC,
            abi_version=ABI_VERSION,
            table_kind="worker",
            capacity_rows=max_workers,
            byte_size=sum(partition.byte_size for partition in self._partitions.values()),
            layouts=tuple(layout_record(kind, layout) for kind, layout in WORKER_BLOCK_KIND_TO_LAYOUT.items()),
        )
        self.header.validate(self.header.layouts)

    def partition(self, block_kind: StateChangeBlockKind) -> TablePartition:
        try:
            return self._partitions[block_kind]
        except KeyError as exc:
            raise KeyError(f"{block_kind!r} is not a worker block") from exc

    def _publish_owned(
        self,
        *,
        owner: str,
        block_kind: StateChangeBlockKind,
        row: int,
        publish_seq: int,
        fields: Iterable[FieldValue],
    ) -> None:
        partition = self.partition(block_kind)
        if partition.layout.owner != owner:
            raise TableProtocolError(f"{owner!r} cannot publish {partition.layout.name} owned by {partition.layout.owner!r}")
        partition._publish(row, publish_seq, fields)

    def worker_partitions(self) -> tuple[TablePartition, ...]:
        return tuple(self._partitions[kind] for kind in WORKER_BLOCK_KIND_TO_LAYOUT)
