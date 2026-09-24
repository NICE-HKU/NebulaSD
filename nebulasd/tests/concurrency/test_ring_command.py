"""Concurrency-facing tests for command rings and command arena reuse."""

from __future__ import annotations

import pytest

from nebulasd.ipc.command_arena import CommandArena, CommandBackpressure
from nebulasd.ipc.command_ring import CommandRing, StaleWorkerGeneration
from nebulasd.ipc.protocol import CommandHeader, CommandKind


def _header(seq: int, offset: int = 0, length: int = 1, generation: int = 9) -> CommandHeader:
    return CommandHeader(seq, generation, CommandKind.DRAFT_BATCH, offset, length, 0)


def test_command_ring_empty_full_and_wraparound() -> None:
    ring = CommandRing(2)
    assert ring.consume(expected_worker_generation=9, arena=CommandArena(16)) is None

    arena = CommandArena(16)
    h0 = arena.allocate(command_seq=0, payload=b"a")
    h1 = arena.allocate(command_seq=1, payload=b"b")
    ring.publish(_header(0, h0.offset, h0.length))
    ring.publish(_header(1, h1.offset, h1.length))
    assert ring.is_full()
    with pytest.raises(CommandBackpressure):
        ring.publish(_header(2))

    assert ring.consume(expected_worker_generation=9, arena=arena).payload == b"a"  # type: ignore[union-attr]
    h2 = arena.allocate(command_seq=2, payload=b"c")
    ring.publish(_header(2, h2.offset, h2.length))
    assert ring.consume(expected_worker_generation=9, arena=arena).payload == b"b"  # type: ignore[union-attr]
    assert ring.consume(expected_worker_generation=9, arena=arena).payload == b"c"  # type: ignore[union-attr]
    assert ring.is_empty()


def test_stale_worker_generation_is_rejected_without_consuming_header() -> None:
    arena = CommandArena(16)
    ring = CommandRing(1)
    h0 = arena.allocate(command_seq=0, payload=b"a")
    ring.publish(_header(0, h0.offset, h0.length, generation=9))

    with pytest.raises(StaleWorkerGeneration):
        ring.consume(expected_worker_generation=10, arena=arena)
    assert ring.consume(expected_worker_generation=9, arena=arena).payload == b"a"  # type: ignore[union-attr]


def test_producer_exit_state_is_visible_after_consumer_drains_ring() -> None:
    arena = CommandArena(16)
    ring = CommandRing(1)
    h0 = arena.allocate(command_seq=0, payload=b"a")
    ring.publish(_header(0, h0.offset, h0.length))
    ring.close_producer()

    with pytest.raises(RuntimeError):
        ring.publish(_header(1))
    assert ring.consume(expected_worker_generation=9, arena=arena).payload == b"a"  # type: ignore[union-attr]
    assert ring.consume(expected_worker_generation=9, arena=arena) is None
    assert ring.producer_closed()


def test_command_arena_reclaims_consumed_payloads_and_wraps() -> None:
    arena = CommandArena(8)
    first = arena.allocate(command_seq=0, payload=b"abcd")
    second = arena.allocate(command_seq=1, payload=b"efgh")
    assert first.offset == 0
    assert second.offset == 4

    with pytest.raises(CommandBackpressure):
        arena.allocate(command_seq=2, payload=b"zz")

    arena.release_through(0)
    with pytest.raises(CommandBackpressure):
        arena.allocate(command_seq=2, payload=b"zzzzz")
    third = arena.allocate(command_seq=2, payload=b"zz")
    assert third.offset == 0
