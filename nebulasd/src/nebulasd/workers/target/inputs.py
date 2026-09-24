"""CPU-only Target input compilation; no model or CUDA ownership."""
from typing import NamedTuple
from dataclasses import dataclass
from nebulasd.core.handles import ArenaHandle
from nebulasd.kv.transfer import CopyPlan, CopyRegion
from nebulasd.workers.work import WorkKind


@dataclass(frozen=True)
class ImportedRow:
    index: int
    version: int
    blocks: int


@dataclass(frozen=True)
class Imports:
    regions: tuple
    rows: tuple[ImportedRow, ...]
    def copy_plan(self, metadata):
        return CopyPlan('H2D', self.regions, (metadata,)) if self.regions else None


class Input(NamedTuple):
    index: int
    local_row: int
    logical: int
    version: int
    prompt: tuple
    anchor: int
    proposal: tuple
    proposal_kind: str
    remaining: int
    stops: tuple
    extent: object
    gpu_begin: int
    capacity: int


@dataclass(frozen=True)
class Plan:
    spec: object
    rows: tuple[Input, ...]


class TargetInputs:
    incremental_inputs = True

    def __init__(self, *, host, tokens, configs, proposals, input_pool,
                 block_size=16, max_batch_tokens=4096):
        self.host, self.tokens, self.configs, self.proposals = host, tokens, configs, proposals
        self.input_pool = input_pool
        self.block_size, self.max_batch_tokens = block_size, max_batch_tokens
        self.prepared_inputs = {}

    def _extent(self, row):
        return self.host.make_extent(request_slot=row.slot, request_epoch=row.epoch,
            host_slot=row.host_slot, host_slot_generation=row.host_generation, writer_lease_generation=row.writer_generation,
            offset_blocks=row.host_offset, capacity_blocks=row.host_capacity)

    def _prepare_static(self, spec, live):
        cache = self.prepared_inputs
        if spec.work_seq not in cache:
            cache[spec.work_seq] = {i: (self.configs.read_config(spec.rows[i].config),
                self._extent(spec.rows[i])) for i in live}
        return cache[spec.work_seq]

    def compile_import(self, spec, captured, layout, live):
        def compile():
            if getattr(self.input_pool, 'inline', False):
                self._prepare_static(spec, live)
            regions, imported = [], []
            for i in live:
                row = spec.rows[i]
                if row.source is not None:
                    source = captured[i]['source']
                    blocks = (source.get('logical_kv_len') + self.block_size - 1) // self.block_size
                    if blocks > row.capacity_blocks:
                        raise ValueError('source exceeds WORK capacity')
                    regions.append(CopyRegion(self._extent(row), layout.offsets[i], 0, blocks))
                    imported.append(ImportedRow(i, source.get('ready_version'), blocks))
            return Imports(tuple(regions), tuple(imported))
        return self.input_pool.submit(compile)

    def compile_compute(self, spec, captured, layout, live):
        def compile():
            static = self._prepare_static(spec, live)
            rows = []
            for i in live:
                row = spec.rows[i]
                config, extent = static[i]
                prefill = spec.operation == WorkKind.TARGET_PREFILL
                if spec.operation not in (WorkKind.TARGET_PREFILL, WorkKind.TARGET_VERIFY):
                    raise ValueError('Target operation required')
                if prefill:
                    prompt = self.tokens.read_tokens(row.prompt)
                    logical, version, count, anchor, proposal, kind = 0, 0, 0, 0, (), 'linear'
                else:
                    source = captured[i]['source']
                    logical, version = source.get('logical_kv_len'), source.get('ready_version')
                    count = logical + 1 - row.prompt_count
                    decision = captured[i]['classified']
                    if 'last_committed_token' in decision:
                        if decision['logical_kv_len'] != logical:
                            raise ValueError('Target anchor and imported KV prefix disagree')
                        anchor = decision['last_committed_token']
                    else:
                        # Legacy diagnostic WORKs retain their Engine-classified input.
                        anchor_handle = ArenaHandle(row.output.offset + (count - 1) * 4, 4, row.output.generation)
                        anchor = self.tokens.read_tokens(anchor_handle)[0]
                    payload = self.proposals.read_proposal(captured[i]['predecessor'].get('proposal_handle'))
                    proposal, kind = payload.draft_token_ids, payload.kind.name.lower()
                    if kind not in ('linear', 'dflash_block'):
                        raise ValueError('unsupported Target proposal kind')
                    if len(proposal) > min(row.token_budget, config.proposal_depth):
                        raise ValueError('proposal exceeds authorized WORK depth')
                    prompt = ()
                if count >= row.max_new_tokens or count < 0:
                    raise ValueError('compute without live output budget')
                needed = len(prompt) if prefill else logical + 1 + len(proposal)
                if needed > row.capacity_blocks * self.block_size:
                    raise ValueError('forward exceeds frozen WORK range')
                rows.append(Input(i, layout.rows[i], logical, version, prompt, anchor, proposal, kind,
                    row.max_new_tokens - count, tuple(config.all_stop_token_ids), extent,
                    layout.offsets[i], row.capacity_blocks))
            total = sum(len(r.prompt) if spec.operation == WorkKind.TARGET_PREFILL else len(r.proposal)+1 for r in rows)
            if total > self.max_batch_tokens:
                raise ValueError('WORK exceeds model token capacity')
            return Plan(spec, tuple(rows))
        return self.input_pool.submit(compile)

    def retire(self, work_seq):
        self.prepared_inputs.pop(work_seq, None)
