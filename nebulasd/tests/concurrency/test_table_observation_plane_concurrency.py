"""Concurrency-facing tests for table notification behavior."""

from __future__ import annotations

from nebulasd.core.enums import Lifecycle, StateChangeBlockKind
from nebulasd.ipc.state_change_ring import StateChangeRing
from nebulasd.table.reader import IncrementalTableReader
from nebulasd.table.storage import RequestSchedulingTable
from nebulasd.table.writers import EngineTableWriter


def test_ring_overflow_scans_table_and_recovers_final_facts() -> None:
    ring = StateChangeRing(capacity=2)
    table = RequestSchedulingTable(4, ring=ring)
    writer = EngineTableWriter(table)

    for slot in range(4):
        writer.publish_active(
            slot=slot,
            publish_seq=slot,
            request_epoch=slot + 10,
            current_round_id=1,
            arrival_seq=slot,
            prompt_token_count=4,
            max_new_tokens=8,
            spec_token_limit=2,
        )

    update = IncrementalTableReader(request_table=table, rings=(ring,)).poll()
    assert update.overflow_recovered
    assert update.scanned_rows == table.header.capacity_rows * len(table.header.layouts)
    assert {(view.block_kind, view.row) for view in update.views} == {
        (StateChangeBlockKind.REQUEST_ENGINE, 0),
        (StateChangeBlockKind.REQUEST_ENGINE, 1),
        (StateChangeBlockKind.REQUEST_ENGINE, 2),
        (StateChangeBlockKind.REQUEST_ENGINE, 3),
    }
    assert {view.get("request_epoch") for view in update.views} == {10, 11, 12, 13}


def test_single_row_update_does_not_rebuild_large_table_snapshot(monkeypatch) -> None:
    ring = StateChangeRing(capacity=8)
    table = RequestSchedulingTable(100_000, ring=ring)
    writer = EngineTableWriter(table)
    writer.publish_active(
        slot=99_999,
        publish_seq=1,
        request_epoch=1,
        current_round_id=1,
        arrival_seq=1,
        prompt_token_count=1,
        max_new_tokens=8,
        spec_token_limit=2,
    )

    calls = 0
    partition = table.partition(StateChangeBlockKind.REQUEST_ENGINE)
    original = partition.read_stable

    def counted_read_stable(row: int, *, include_cold: bool = True, max_retries: int = 16):
        nonlocal calls
        calls += 1
        return original(row, include_cold=include_cold, max_retries=max_retries)

    monkeypatch.setattr(partition, "read_stable", counted_read_stable)
    update = IncrementalTableReader(request_table=table, rings=(ring,)).poll()

    assert len(update.views) == 1
    assert update.views[0].row == 99_999
    assert calls == 1
    assert update.scanned_rows == 0

    writer.publish_lifecycle(slot=99_999, publish_seq=2, request_epoch=1, lifecycle=Lifecycle.FINISHED)
    update = IncrementalTableReader(request_table=table, rings=(ring,)).poll()
    assert len(update.views) == 1
    assert update.scanned_rows == 0
