"""Stable incremental readers for scheduling observation tables."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from itertools import islice

from nebulasd.core.enums import StateChangeBlockKind
from nebulasd.core.ids import OP_SEQ, U64
from nebulasd.ipc.state_change_ring import StateChangeEntry, StateChangeRing

from .storage import BlockSnapshot, RequestSchedulingTable, TablePartition, WorkerSchedulingRegistry, StableReadConflict

# H2D completion advances Worker execution, not Engine scheduling. Keep the
# shared rows for Worker consumers, but do not materialize them in the owner.
ENGINE_IGNORED_KINDS = frozenset((StateChangeBlockKind.REQUEST_ENGINE,
    StateChangeBlockKind.REQUEST_DISPATCH, StateChangeBlockKind.REQUEST_H2D,
    StateChangeBlockKind.REQUEST_DRAFT_H2D))


@dataclass(frozen=True, slots=True)
class TableUpdateBatch:
    # Latest stable view per (block_kind, row), unique within one poll.
    views: tuple[BlockSnapshot, ...]
    overflow_recovered: bool
    scanned_rows: int


class IncrementalTableReader:
    """Read changed rows only, falling back to scans when notification rings overflow."""

    def __init__(
        self,
        *,
        request_table: RequestSchedulingTable | None = None,
        worker_registry: WorkerSchedulingRegistry | None = None,
        rings: tuple[StateChangeRing, ...] = (),
    ) -> None:
        if request_table is None and worker_registry is None:
            raise ValueError("at least one table is required")
        self._request_table = request_table
        self._worker_registry = worker_registry
        self._rings = rings
        self._seen: dict[tuple[StateChangeBlockKind, int], int] = {}
        self._views: dict[tuple[StateChangeBlockKind, int], BlockSnapshot] = {}
        self._pending: dict[tuple[StateChangeBlockKind, int], StateChangeEntry] = {}
        self.read_snapshot = lambda partition, row: partition.read_stable(row)
        self.max_entries: int | None = None
        self.ignored_kinds = set()
        self.priority_rows = ()
        self._ring_cursor = 0
        self._scan = deque()
        self._rescan = False

    def poll(self) -> TableUpdateBatch:
        if self.max_entries is not None:
            return self._poll_bounded(self.max_entries)
        merged, self._pending = self._pending, {}
        overflowed = False
        for ring in self._rings:
            batch = ring.drain()
            overflowed = overflowed or batch.overflowed
            for entry in batch.entries:
                if entry.block_kind in self.ignored_kinds:
                    continue
                key = (entry.block_kind, entry.row)
                previous = merged.get(key)
                if previous is None or OP_SEQ.is_newer(entry.publish_seq, previous.publish_seq):
                    merged[key] = entry

        scanned_rows = 0
        if overflowed:
            changed, scanned_rows = self._scan_changed()
            # An INVALID row can be mid-publication during recovery. Retain
            # drained hints until a stable read succeeds, even without a new hint.
            recovered = {(row.block_kind, row.row) for row in changed}
            for key, entry in merged.items():
                if key not in recovered:
                    self._pending.setdefault(key, entry)
        else:
            changed = []
            for entry in merged.values():
                try:
                    changed.append(self._read_entry(entry))
                except StableReadConflict:
                    self._pending[(entry.block_kind, entry.row)] = entry

        final_views: dict[tuple[StateChangeBlockKind, int], BlockSnapshot] = {}
        for view in changed:
            key = (view.block_kind, view.row)
            previous_seq = self._seen.get(key)
            if previous_seq is None or view.publish_seq != previous_seq:
                self._seen[key] = view.publish_seq
                self._views[key] = view
                final_views[key] = view
        return TableUpdateBatch(
            views=tuple(final_views.values()),
            overflow_recovered=overflowed,
            scanned_rows=scanned_rows,
        )

    def has_pending(self):
        return bool(self._pending or self._scan or self._rescan or
                    any(ring.has_pending() for ring in self._rings))

    def reset_pending(self):
        """Only at the all-owner retirement barrier, after arenas are reset."""
        self._pending.clear()
        self._scan.clear()
        self._rescan = False
        # All producers are stopped. A bounded last turn may have left hints
        # for rows which the barrier has just invalidated; do not retry them
        # forever in the new cohort.
        for ring in self._rings:
            ring.drain()

    def reset_requests(self, kinds):
        for cache in (self._seen, self._views, self._pending):
            for key in tuple(cache):
                if key[0] in kinds:
                    del cache[key]

    def _poll_bounded(self, budget):
        """Bound hint drain, stable reads AND overflow recovery per owner turn.

        Rotate the first ring so a noisy worker cannot starve another. Hints
        retained across turns are still coalesced; a failed stable read is put
        at the tail, not allowed to monopolize the next turn.
        """
        if budget <= 0:
            raise ValueError('observation budget must be positive')
        count = len(self._rings)
        remaining = budget
        overflowed = False
        for offset in range(count):
            ring = self._rings[(self._ring_cursor + offset) % count]
            share = remaining // (count - offset)
            batch = ring.drain(share)
            remaining -= len(batch.entries)
            overflowed |= batch.overflowed
            for entry in batch.entries:
                if entry.block_kind in self.ignored_kinds:
                    continue
                key = (entry.block_kind, entry.row)
                previous = self._pending.get(key)
                if previous is None or OP_SEQ.is_newer(entry.publish_seq, previous.publish_seq):
                    self._pending[key] = entry
        if count:
            self._ring_cursor = (self._ring_cursor + 1) % count
        if overflowed:
            # A later overflow may affect already-scanned rows. Finish this pass
            # before restarting, so continuous overflow cannot pin the cursor.
            self._rescan = True
        if not self._scan and self._rescan:
            self._scan.extend((p, 0) for p in self._partitions()
                              if p.block_kind not in self.ignored_kinds)
            self._rescan = False
        changed = {}
        def retain(view):
            key = (view.block_kind, view.row)
            if self._seen.get(key) != view.publish_seq:
                self._seen[key] = view.publish_seq
                self._views[key] = view
                changed[key] = view

        # Fixed-size worker state bypasses request hint backlog. The shared
        # version cache also suppresses decoding when its ring hint arrives.
        for kind, row in self.priority_rows:
            partition = self._partition(kind)
            seq = partition.read_publish_seq(row)
            if seq == U64.invalid or self._seen.get((kind, row)) == seq:
                continue
            try:
                retain(self.read_snapshot(partition, row))
            except StableReadConflict:
                pass  # Revisited next poll, independently of notification loss.

        # Reserve half the read budget for recovery when necessary. Reading the
        # publish sequence one row at a time avoids an unbounded full-table scan.
        hint_budget = budget if not self._scan else budget // 2
        keys = tuple(islice(self._pending, hint_budget))
        for key in keys:
            entry = self._pending.pop(key)
            try:
                retain(self._read_entry(entry))
            except StableReadConflict:
                self._pending[key] = entry
        scanned = 0
        while self._scan and scanned < budget - len(keys):
            partition, row = self._scan.popleft()
            seq = partition.read_publish_seq(row)
            key = (partition.block_kind, row)
            if seq == U64.invalid:
                # A racing publication sends a hint after its release commit;
                # if that hint overflows it requests a subsequent recovery pass.
                pass
            elif self._seen.get(key) != seq:
                try:
                    retain(self.read_snapshot(partition, row))
                except StableReadConflict:
                    self._pending[key] = StateChangeEntry(partition.block_kind, row, seq)
            scanned += 1
            if row + 1 < partition.capacity_rows:
                self._scan.appendleft((partition, row + 1))
        return TableUpdateBatch(tuple(changed.values()), overflowed or bool(scanned), scanned)

    def cached_view(self, block_kind: StateChangeBlockKind, row: int) -> BlockSnapshot | None:
        return self._views.get((block_kind, row))

    def _read_entry(self, entry: StateChangeEntry) -> BlockSnapshot:
        key = (entry.block_kind, entry.row)
        seen = self._seen.get(key)
        if seen is not None and not OP_SEQ.is_newer(entry.publish_seq, seen):
            return self._views[key]
        return self.read_snapshot(self._partition(entry.block_kind), entry.row)

    def _scan_changed(self) -> tuple[list[BlockSnapshot], int]:
        changed: list[BlockSnapshot] = []
        scanned_rows = 0
        for partition in self._partitions():
            if partition.block_kind in self.ignored_kinds:
                continue
            for row, publish_seq in enumerate(partition.publish_sequences()):
                scanned_rows += 1
                if publish_seq == U64.invalid:
                    continue
                key = (partition.block_kind, row)
                if self._seen.get(key) != publish_seq:
                    try:
                        changed.append(partition.read_stable(row))
                    except StableReadConflict:
                        self._pending[key] = StateChangeEntry(partition.block_kind, row, publish_seq)
        return changed, scanned_rows

    def _partition(self, block_kind: StateChangeBlockKind) -> TablePartition:
        if self._request_table is not None:
            try:
                return self._request_table.partition(block_kind)
            except KeyError:
                pass
        if self._worker_registry is not None:
            return self._worker_registry.partition(block_kind)
        raise KeyError(block_kind)

    def _partitions(self) -> tuple[TablePartition, ...]:
        partitions: list[TablePartition] = []
        if self._request_table is not None:
            partitions.extend(self._request_table.request_partitions())
        if self._worker_registry is not None:
            partitions.extend(self._worker_registry.worker_partitions())
        return tuple(partitions)
