"""Draft migration ABI partitions and extensions (Table ABI v4)."""
from dataclasses import replace
from .layout_types import (StructLayout, field, publish_seq, U8, U32, U64, ENUM32,
                     RESULT32, ARENA_HANDLE, ABI_VERSION, CACHE_LINE_BYTES)


def fields(owner, specs, *, hot=True):
    return tuple(field(name, typ, owner, description, hot=hot) for name, typ, description in specs)


DISPATCH_FIELDS = fields('dispatcher', (
    ('request_epoch', U64, 'request slot identity'),
    ('draft_owner_epoch', U64, 'executing owner generation, advanced only by run'),
    ('draft_prepare_seq', U64, 'request prepare sequence, not ring command sequence'),
    ('planned_draft_id', U32, 'prepare destination'),
    ('planned_draft_generation', U64, 'prepare destination process generation'),
    ('planned_draft_bank_id', U8, 'prepare destination bank'),
    ('planned_draft_bank_epoch', U64, 'prepare bank epoch'),
    ('draft_prepared_batch_seq', U64, 'frozen prepare batch'),
    ('draft_source_snapshot_version', U64, 'snapshot required by prepare'),
    ('draft_snapshot_round_id', U64, 'completed snapshot round, not next run round'),
    ('draft_snapshot_handle', ARENA_HANDLE, 'shared immutable snapshot metadata'),
    ('draft_next_owner_epoch', U64, 'owner epoch authorized by next run'),
    ('draft_run_bank_id', U8, 'actual run bank'),
    ('draft_run_bank_epoch', U64, 'actual run bank epoch'),
    ('draft_run_batch_seq', U64, 'actual run batch'),
))

COMPUTE_FIELDS = fields('draft_worker', (
    ('owner_epoch', U64, 'executing owner fence'),
    ('snapshot_version', U64, 'actual KV and metadata version'),
    ('logical_kv_len', U32, 'actual unverified prefix length'),
    ('valid_blocks', U32, 'valid blocks, not Target committed blocks'),
    ('bank_id', U8, 'source bank'),
    ('bank_epoch', U64, 'source bank epoch'),
    ('batch_seq', U64, 'source batch'),
    ('dirty_begin_block', U32, 'first modified block'),
    ('dirty_block_count', U32, 'modified block count'),
), hot=False)

ALLOCATION_SPECS = (
    ('arena_id', U32, 'Draft HostKV arena identity'),
    ('arena_generation', U32, 'Draft HostKV arena generation'),
    ('layout_id', U64, 'compatible model/weights/KV layout identity'),
    ('host_slot_generation', U64, 'allocation slot generation'),
    ('writer_lease_generation', U64, 'allocation writer lease'),
    ('offset_blocks', U64, 'allocation block offset'),
    ('host_slot', U32, 'allocation slot'),
    ('capacity_blocks', U32, 'allocation capacity'),
    ('block_size', U64, 'tokens per KV block'),
)


def block(name, owner, specs, *, multi=True):
    return StructLayout(name, owner, ABI_VERSION, (publish_seq(owner), *fields(owner, specs)),
                        row_alignment=CACHE_LINE_BYTES if multi else 8, multi_writer=multi)


DRAFT_HOSTKV_BLOCK = block('DraftHostKVAllocationBlock', 'engine_draft_hostkv_allocator',
    (('request_epoch', U64, 'request identity'), *ALLOCATION_SPECS), multi=False)

_COPY_COMMON = (
    ('request_epoch', U64, 'request identity'),
    ('snapshot_round_id', U64, 'completed source snapshot round'),
    ('snapshot_version', U64, 'required snapshot version'),
    ('snapshot_handle', ARENA_HANDLE, 'shared snapshot metadata'),
    ('logical_kv_len', U32, 'actual KV length'),
    ('status', ENUM32, 'direction-specific copy status'),
    ('result_code', RESULT32, 'copy result'),
)
_COPY_COLD = (('copy_start_time_ns', U64, 'copy submission time'), ('copy_bytes', U64, 'copy workload'))

DRAFT_D2H_BLOCK = block('DraftD2HBlock', 'draft_source_copy_lane', (*_COPY_COMMON,
    ('source_worker_id', U32, 'source owner'), ('source_worker_generation', U64, 'source process'),
    ('source_op_seq', U64, 'completed compute operation'), ('owner_epoch', U64, 'source owner epoch'),
    ('source_bank_id', U8, 'source bank'), ('source_bank_epoch', U64, 'source bank epoch'),
    ('source_batch_seq', U64, 'source batch'), ('ready_version', U64, 'complete HostKV prefix version'),
    ('valid_blocks', U32, 'valid unverified KV blocks'), *ALLOCATION_SPECS))
DRAFT_H2D_BLOCK = block('DraftH2DBlock', 'draft_destination_copy_lane', (*_COPY_COMMON,
    ('next_round_id', U64, 'next compute round'), ('observed_prepare_seq', U64, 'prepare sequence'),
    ('destination_worker_id', U32, 'destination'), ('destination_worker_generation', U64, 'destination process'),
    ('next_owner_epoch', U64, 'planned next owner epoch'),
    ('destination_bank_id', U8, 'destination bank'), ('destination_bank_epoch', U64, 'destination bank epoch'),
    ('prepared_batch_seq', U64, 'frozen batch'), ('gpu_ready_version', U64, 'restored snapshot version'),
    ('copied_blocks', U32, 'imported valid blocks'), ('local_row', U32, 'destination local block table row'),
    *ALLOCATION_SPECS))
# Timings are cold; all fence fields precede them.
DRAFT_D2H_BLOCK = replace(DRAFT_D2H_BLOCK, fields=DRAFT_D2H_BLOCK.fields + fields('draft_source_copy_lane', _COPY_COLD, hot=False))
DRAFT_H2D_BLOCK = replace(DRAFT_H2D_BLOCK, fields=DRAFT_H2D_BLOCK.fields + fields('draft_destination_copy_lane', _COPY_COLD, hot=False))


def worker_layouts(copy_layout, bank_layout):
    def owned(layout, name):
        return replace(layout, name=name, owner='draft_worker',
                       fields=tuple(replace(f, owner='draft_worker') for f in layout.fields))
    return owned(copy_layout, 'DraftCopyRuntimeBlock'), owned(bank_layout, 'DraftBankBlock')
