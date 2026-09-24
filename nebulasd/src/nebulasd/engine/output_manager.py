"""Deliver the unread cumulative prefix; Target alone decides output termination."""

from dataclasses import replace
from collections import deque
from time import perf_counter_ns

from nebulasd.core.enums import Lifecycle, TargetStatus, StateChangeBlockKind as K
from nebulasd.scheduler.views import value as v


class OutputManager:
    def __init__(self, registry, *, on_tokens=None):
        self.registry = registry
        self.on_tokens = on_tokens
        self.defer_delivery = False
        self.pending = deque()

    def flush(self):
        while self.pending:
            record, tokens, lifecycle = self.pending.popleft()
            record.output_chunks.append(tokens)
            if self.on_tokens is not None:
                self.on_tokens(record.request_id, tokens, lifecycle)

    def consume(self, updates):
        changed = {}
        for row in updates:
            record = self.consume_row(row)
            if record is not None:
                changed[record.input.slot] = record
        if not self.defer_delivery:
            self.flush()
        return tuple(changed.values())

    def consume_row(self, row):
        if row.block_kind != K.REQUEST_TARGET_COMPUTE or v(row, "status") != TargetStatus.READY_DRAFT:
            return None
        record = self.registry.records.get(row.row)
        if record is None or v(row, "request_epoch") != record.input.epoch:
            raise ValueError("Target result has stale request identity")
        round_id = v(row, "round_id")
        if round_id <= record.current_round:
            return None
        count = v(row, 'output_count')
        prefix = v(row, 'output_handle')
        if (count < record.input.output_count or count > record.config.max_new_tokens
                or prefix.offset != record.reservation.offset
                or prefix.generation != record.reservation.generation or prefix.length != 4*count):
            raise ValueError('invalid cumulative output prefix')
        from nebulasd.core.handles import ArenaHandle
        cursor = record.input.output_count
        accepted = self.registry.resources.token_router.read_tokens(
            ArenaHandle(prefix.offset + 4*cursor, 4*(count-cursor), prefix.generation)) if count > cursor else ()
        record.current_round = round_id
        if record.lifecycle != Lifecycle.ACTIVE:
            return None
        record.input = replace(record.input, output=prefix, output_count=count, ready_ns=perf_counter_ns())
        if v(row, 'output_finished'):
            record.lifecycle = Lifecycle.FINISHED
        self.pending.append((record, tuple(accepted), record.lifecycle))
        return record
