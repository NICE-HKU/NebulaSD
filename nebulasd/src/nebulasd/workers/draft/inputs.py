"""Frozen Draft input boundary; imports do not depend on Target classification."""
from threading import Lock
from nebulasd.kv.transfer import CopyRegion
from nebulasd.workers.target.inputs import TargetInputs, Imports
from typing import NamedTuple
from dataclasses import dataclass
from nebulasd.core.draft_contracts import DraftHostAllocation, DraftSnapshotIdentity
from nebulasd.workers.work import WorkKind


def allocation(row, arena_generation, block_size):
    return DraftHostAllocation(row.host_arena, arena_generation, row.layout_id,
        row.host_generation, row.writer_generation, row.host_offset, row.host_slot,
        row.host_capacity, block_size)


def read_source(row, source, snapshots, arena_generation, block_size):
    alloc = allocation(row, arena_generation, block_size)
    for name in alloc.__dataclass_fields__:
        if source.get(name) != getattr(alloc, name):
            raise ValueError(f'Draft source allocation mismatch: {name}')
    identity = DraftSnapshotIdentity(row.slot, row.epoch, source.get('snapshot_round_id'),
        source.get('source_op_seq'), source.get('source_worker_id'), source.get('source_worker_generation'),
        source.get('owner_epoch'), source.get('snapshot_version'), source.get('logical_kv_len'),
        source.get('valid_blocks'), alloc)
    if identity.snapshot_version != row.source.expected_ticket or source.get('ready_version') != identity.snapshot_version:
        raise ValueError('Draft source ticket mismatch')
    if identity.round_id >= row.round_id or identity.op_seq >= row.run_seq or row.owner_epoch <= identity.owner_epoch:
        raise ValueError('Draft source must precede new ownership')
    handle = source.get('snapshot_handle')
    snapshot = snapshots.arenas[handle.generation].read_snapshot(handle, expected=identity,
                                                               expected_layout_id=row.layout_id)
    if (snapshot.prompt_handle, snapshot.generation_config_handle, snapshot.prompt_count) != (row.prompt, row.config, row.prompt_count):
        raise ValueError('Draft snapshot static input mismatch')
    if (snapshot.committed_output_handle.offset, snapshot.committed_output_handle.generation) != (row.output.offset, row.output.generation) or snapshot.committed_output_handle.length > row.output.length:
        raise ValueError("Draft snapshot committed output reservation mismatch")
    return snapshot


class RowPlan(NamedTuple):
    index: int
    local_row: int
    retained: int
    suffix: tuple
    limit: int
    stops: tuple
    committed_count: int
    version: int
    extent: object
    gpu_begin: int


@dataclass(frozen=True, slots=True)
class Plan:
    spec: object
    rows: tuple


def reconcile(row, config, delta, accepted, snapshot, prompt, block_size=16):
    """Accepted count is captured once at the Target boundary.

    Q[-1] has no KV. Full acceptance therefore replays that token plus bonus.
    Correction/bonus always provides a suffix; no whole-history LCP is needed.
    """
    delta = tuple(delta)
    if not delta:
        raise ValueError('Draft needs committed Target output')
    if snapshot is None:
        if len(prompt) != row.prompt_count:
            raise ValueError("Draft prompt count mismatch")
        retained, suffix, count, version = 0, tuple(prompt)+delta, len(delta), 1
    else:
        if not 0 <= accepted <= snapshot.proposal_count or len(delta) != accepted + 1:
            raise ValueError('invalid live Target acceptance/delta counts')
        reused = min(accepted, snapshot.proposal_count-1)
        retained = snapshot.prompt_count + snapshot.committed_output_count + reused
        suffix = delta[reused:]
        count = snapshot.committed_output_count + len(delta)
        version = snapshot.identity.snapshot_version + 1
    limit = min(row.token_budget, config.proposal_depth, row.max_new_tokens-count)
    if config.max_new_tokens != row.max_new_tokens or limit <= 0:
        raise ValueError('Draft compute without output budget')
    if retained+len(suffix)+limit-1 > row.capacity_blocks*block_size:
        raise ValueError('Draft forward exceeds frozen capacity')
    return retained, suffix, count, version, limit


@dataclass(frozen=True, slots=True)
class Imported:
    index: int
    version: int
    blocks: int
    snapshot_round: int
    logical: int
    local_row: int
    snapshot_handle: object


class DraftInputs(TargetInputs):
    def __init__(self, *, snapshots, layout_id, host_arena_id, **kwargs):
        super().__init__(**kwargs)
        self.snapshots, self.layout_id, self.host_arena_id = snapshots, layout_id, host_arena_id
        self.sources, self.source_lock = {}, Lock()

    def _source(self, spec, i, captured):
        row = spec.rows[i]
        if row.layout_id != self.layout_id or row.host_arena != self.host_arena_id:
            raise ValueError('Draft WORK model layout mismatch')
        if row.source is None:
            return None
        key = (spec.work_seq, i)
        with self.source_lock:
            if key not in self.sources:
                self.sources[key] = read_source(row, captured[i]['source'], self.snapshots,
                    self.host.descriptor.descriptor_generation, self.block_size)
            return self.sources[key]

    def compile_import(self, spec, captured, layout, live):
        def compile():
            if getattr(self.input_pool, 'inline', False):
                self._prepare_static(spec, live)
            regions, rows = [], []
            for i in live:
                snapshot = self._source(spec, i, captured)
                if snapshot is not None:
                    blocks = snapshot.identity.valid_blocks
                    regions.append(CopyRegion(self._extent(spec.rows[i]), layout.offsets[i], 0, blocks))
                    rows.append(Imported(i, snapshot.identity.snapshot_version, blocks, snapshot.identity.round_id,
                        snapshot.identity.logical_kv_len, layout.rows[i], captured[i]['source'].get('snapshot_handle')))
            return Imports(tuple(regions), tuple(rows))
        return self.input_pool.submit(compile)

    def compile_compute(self, spec, captured, layout, live):
        def compile():
            if spec.operation not in (WorkKind.DRAFT_INITIAL, WorkKind.DRAFT_DECODE):
                raise ValueError('Draft operation required')
            static = self._prepare_static(spec, live)
            rows = []
            for i in live:
                row = spec.rows[i]
                snapshot = self._source(spec, i, captured)
                config, extent = static[i]
                target = captured[i]['predecessor']
                delta = self.tokens.read_tokens(target.get('committed_delta_handle'))
                prompt = self.tokens.read_tokens(row.prompt) if snapshot is None else ()
                retained, suffix, count, version, limit = reconcile(row, config, delta,
                    target.get('accepted_draft_count'), snapshot, prompt, self.block_size)
                rows.append(RowPlan(i, layout.rows[i], retained, suffix, limit,
                    tuple(config.all_stop_token_ids), count, version, extent, layout.offsets[i]))
            if sum(len(r.suffix) for r in rows) > self.max_batch_tokens:
                raise ValueError('Draft suffix exceeds model batch token capacity')
            return Plan(spec, tuple(rows))
        return self.input_pool.submit(compile)

    def retire(self, work_seq):
        super().retire(work_seq)
        with self.source_lock:
            for key in tuple(self.sources):
                if key[0] == work_seq:
                    del self.sources[key]
