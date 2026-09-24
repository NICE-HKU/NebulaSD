"""Canonical Python declaration of the NebulaSD control-plane ABI.

This module describes field widths, offsets, alignment, enum storage, and
struct versions. It does not implement shared memory or rely on Python object
layout as the wire representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


ABI_VERSION = 6
ENDIANNESS = "little"
CACHE_LINE_BYTES = 64


@dataclass(frozen=True, slots=True)
class ScalarType:
    name: str
    size: int
    alignment: int
    signed: bool = False


U8 = ScalarType("u8", 1, 1)
U32 = ScalarType("u32", 4, 4)
I32 = ScalarType("i32", 4, 4, signed=True)
U64 = ScalarType("u64", 8, 8)
ENUM32 = ScalarType("enum32", 4, 4)
RESULT32 = ScalarType("result32", 4, 4)
ARENA_HANDLE = ScalarType("arena_handle", 16, 8)


def _align(offset: int, alignment: int) -> int:
    remainder = offset % alignment
    return offset if remainder == 0 else offset + alignment - remainder


@dataclass(frozen=True, slots=True)
class FieldDef:
    name: str
    type: ScalarType
    owner: str
    hot: bool
    required: bool
    description: str


@dataclass(frozen=True, slots=True)
class FieldLayout:
    field: FieldDef
    offset: int

    @property
    def size(self) -> int:
        return self.field.type.size


@dataclass(frozen=True, slots=True)
class StructLayout:
    name: str
    owner: str
    version: int
    fields: tuple[FieldDef, ...]
    partition_alignment: int = CACHE_LINE_BYTES
    row_alignment: int = 8
    multi_writer: bool = False

    @property
    def alignment(self) -> int:
        return max(field.type.alignment for field in self.fields)

    @property
    def field_layouts(self) -> tuple[FieldLayout, ...]:
        offset = 0
        layouts: list[FieldLayout] = []
        for field in self.fields:
            offset = _align(offset, field.type.alignment)
            layouts.append(FieldLayout(field, offset))
            offset += field.type.size
        return tuple(layouts)

    @property
    def size(self) -> int:
        layouts = self.field_layouts
        if not layouts:
            return 0
        last = layouts[-1]
        return _align(last.offset + last.size, self.alignment)

    @property
    def row_stride(self) -> int:
        return _align(self.size, self.row_alignment)

    @property
    def hot_prefix_size(self) -> int:
        layouts = [layout for layout in self.field_layouts if layout.field.hot]
        if not layouts:
            return 0
        last = layouts[-1]
        return last.offset + last.size

    def field_offset(self, name: str) -> int:
        for layout in self.field_layouts:
            if layout.field.name == name:
                return layout.offset
        raise KeyError(name)

    def field(self, name: str) -> FieldDef:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)


def field(name: str, type_: ScalarType, owner: str, description: str, *, hot: bool, required: bool = True) -> FieldDef:
    return FieldDef(name=name, type=type_, owner=owner, hot=hot, required=required, description=description)


def publish_seq(owner: str) -> FieldDef:
    return field("publish_seq", U64, owner, "release/acquire publication sequence", hot=True)


