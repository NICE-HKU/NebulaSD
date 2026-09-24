"""Bounded level-triggered shared-table observation; no peer notification handles."""
from dataclasses import dataclass
from collections import deque
from time import perf_counter_ns
from nebulasd.core.enums import StateChangeBlockKind as K, Lifecycle, D2HStatus, DraftStatus, TargetStatus
from nebulasd.table.storage import StableReadConflict
from .work import Selector

_RULES = {
    Selector.TARGET_HOST: (K.REQUEST_D2H, 'ready_version', D2HStatus.HOST_READY),
    Selector.DRAFT_HOST: (K.REQUEST_DRAFT_D2H, 'ready_version', D2HStatus.HOST_READY),
    Selector.PROPOSAL: (K.REQUEST_DRAFT, 'round_id', DraftStatus.READY_TARGET),
    Selector.DELTA: (K.REQUEST_TARGET_COMPUTE, 'round_id', TargetStatus.READY_DRAFT),
    Selector.CLASSIFIED: (K.REQUEST_ENGINE, 'classified_result_ticket', None),
    Selector.TARGET_DECISION: (K.REQUEST_TARGET_COMPUTE, 'round_id', TargetStatus.READY_DRAFT),
}


_FIELDS = {
    Selector.TARGET_HOST: ('request_epoch', 'ready_version', 'status', 'logical_kv_len'),
    Selector.DRAFT_HOST: ('request_epoch', 'ready_version', 'status', 'logical_kv_len',
        'snapshot_handle', 'snapshot_round_id', 'snapshot_version', 'source_op_seq',
        'source_worker_id', 'source_worker_generation', 'owner_epoch', 'valid_blocks',
        'arena_id', 'arena_generation', 'layout_id', 'host_slot_generation',
        'writer_lease_generation', 'offset_blocks', 'host_slot', 'capacity_blocks', 'block_size'),
    Selector.PROPOSAL: ('request_epoch', 'round_id', 'status', 'proposal_handle'),
    Selector.DELTA: ('request_epoch', 'round_id', 'status', 'committed_delta_handle',
                     'accepted_draft_count', 'logical_kv_len', 'last_committed_token'),
    Selector.CLASSIFIED: ('request_epoch', 'classified_result_ticket', 'lifecycle'),
    Selector.TARGET_DECISION: ('request_epoch', 'round_id', 'status', 'result_code',
                               'logical_kv_len', 'last_committed_token', 'output_count', 'output_finished'),
}


@dataclass(slots=True)
class Watch:
    key: tuple
    dependency: object
    last_seq: int | None = None
    decision: tuple | None = None


@dataclass(frozen=True, slots=True)
class Captured:
    key: tuple
    snapshot: object
    observed_ns: int


class Dependencies:
    def __init__(self, table, *, period_s=0.0001, profile=False, configs=None):
        if period_s <= 0:
            raise ValueError('dependency polling requires a positive period')
        self.table, self.period_s = table, period_s
        self.profile = profile
        self.configs = configs
        self.pending = {}
        self.order = deque()
        self.checks = 0
        self.scanner = None
        if all(hasattr(p, 'segment') for p in table._partitions.values()):
            from .native_dependencies import NativeDependencies
            self.scanner = NativeDependencies(table, _FIELDS, _RULES)

    def register(self, work):
        # Classification is checked before missing source/proposal, including
        # terminal requests which will never publish their expected tickets.
        for i, row in enumerate(work.rows):
            for name in ('classified', 'source', 'predecessor'):
                dep = getattr(row, name)
                if dep is not None:
                    key = (work.work_seq, i, name)
                    self.add(key, dep)
                    if dep.selector == Selector.TARGET_DECISION:
                        config = self.configs.read_config(row.config)
                        self.pending[key].decision = (row.prompt_count, row.max_new_tokens,
                                                      tuple(config.all_stop_token_ids))

    def retire(self, seq):
        for key in tuple(self.pending):
            if key[0] == seq:
                self.discard(key)
        self.order = deque(k for k in self.order if k in self.pending)

    def clear(self):
        self.pending.clear()
        self.order.clear()

    def add(self, key, dependency):
        if key in self.pending:
            raise ValueError('duplicate dependency')
        if int(_RULES[dependency.selector][0]) != dependency.kind:
            raise ValueError('selector/table mismatch')
        self.pending[key] = Watch(key, dependency)
        self.order.append(key)
        if self.scanner is not None:
            self.scanner.dirty = True

    def discard(self, key):
        self.pending.pop(key, None)

    def _changed(self, budget):
        if self.scanner is not None:
            rows = self.scanner.scan(self.pending)
            self.checks += len(self.scanner.keys)
            for key, row in rows.items():
                watch = self.pending.get(key)
                if watch is not None:
                    yield key, watch, row
            return
        for _ in range(min(budget, len(self.order))):
            key = self.order.popleft()
            watch = self.pending.get(key)
            if watch is None:
                continue
            dep = watch.dependency
            partition = self.table.partition(dep.kind)
            self.checks += 1
            seq = partition.read_publish_seq(dep.slot)
            if seq != watch.last_seq:
                try:
                    row = partition.read_stable(dep.slot, max_retries=1, field_names=_FIELDS[dep.selector])
                except StableReadConflict:
                    pass
                else:
                    yield key, watch, row
            if key in self.pending:
                self.order.append(key)

    def poll(self, budget=64):
        captured = []
        for key, watch, row in self._changed(budget):
            dep = watch.dependency
            watch.last_seq = row.publish_seq
            epoch = row.get('request_epoch')
            _, ticket_field, status = _RULES[dep.selector]
            ticket = row.get(ticket_field)
            # Unpublished rows contain invalid sentinels; their sequence is U64_MAX.
            if row.publish_seq != (1 << 64) - 1:
                if epoch > dep.request_epoch:
                    raise RuntimeError('dependency crossed cohort lifetime')
                if epoch == dep.request_epoch:
                    # FINISHED is terminal for this request epoch. No later
                    # classification/proposal will be produced for skipped
                    # members retained in an advancing frozen batch.
                    if dep.selector == Selector.CLASSIFIED and row.get('lifecycle') == Lifecycle.FINISHED:
                        for name in ('source', 'predecessor'):
                            self.discard((key[0], key[1], name))
                        captured.append(Captured(key, row, perf_counter_ns() if self.profile else 0))
                        del self.pending[key]
                        continue
                    if dep.selector == Selector.TARGET_DECISION and ticket == dep.expected_ticket and row.get('status') == status:
                        if row.get('result_code') != 0:
                            raise RuntimeError('failed Target decision')
                        prompt, maximum, stops = watch.decision
                        count = row.get('output_count')
                        if not 0 < count <= maximum:
                            raise RuntimeError('Target decision output count outside request budget')
                        finished = bool(row.get('output_finished'))
                        snapshot = dict(lifecycle=Lifecycle.FINISHED if finished else Lifecycle.ACTIVE,
                                        last_committed_token=row.get('last_committed_token'),
                                        logical_kv_len=row.get('logical_kv_len'))
                        if finished:
                            for name in ('source', 'predecessor'):
                                self.discard((key[0], key[1], name))
                        captured.append(Captured(key, snapshot, perf_counter_ns() if self.profile else 0))
                        del self.pending[key]
                        continue
                    if ticket > dep.expected_ticket:
                        raise RuntimeError('dependency overwritten before capture')
                    if ticket == dep.expected_ticket and (status is None or row.get('status') == status):
                        if dep.selector == Selector.CLASSIFIED and row.get('lifecycle') not in (Lifecycle.ACTIVE, Lifecycle.FINISHED):
                            raise RuntimeError('unsupported lifecycle in WORK')
                        captured.append(Captured(key, row, perf_counter_ns() if self.profile else 0))
                        del self.pending[key]
                        continue
        return captured
