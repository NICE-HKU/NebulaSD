"""Fixed-layout Table ABI; scalar definitions live in layout_types."""
from __future__ import annotations
from typing import Iterable
from .layout_types import (ABI_VERSION, ENDIANNESS, CACHE_LINE_BYTES, ScalarType, U8, U32,
                           I32, U64, ENUM32, RESULT32, ARENA_HANDLE, FieldDef, FieldLayout,
                           StructLayout, field, publish_seq)

ENGINE_BLOCK = StructLayout(
    "EngineBlock",
    "engine",
    ABI_VERSION,
    (
        publish_seq("engine"),
        field("request_epoch", U64, "engine", "slot generation for request identity", hot=True),
        field("current_round_id", U64, "engine", "latest accepted logical round", hot=True),
        field("arrival_seq", U64, "engine", "request arrival sequence for stable fairness", hot=True),
        field("lifecycle", ENUM32, "engine", "Lifecycle enum", hot=True),
        field("prompt_token_count", U32, "engine", "initial prompt token count", hot=False),
        field("max_new_tokens", U32, "engine", "maximum generated token count", hot=False),
        field("spec_token_limit", U32, "engine", "maximum speculative tokens per round", hot=False),
        field("input_tokens_handle", ARENA_HANDLE, "engine", "token arena handle for input tokens", hot=False),
        field("generation_config_handle", ARENA_HANDLE, "engine", "arena handle for generation configuration", hot=False),
        field("classified_result_ticket", U64, "engine", "0 before classification; Target round + 1 after output consumption", hot=False),
    ),
)

DISPATCH_BLOCK = StructLayout(
    "DispatchBlock",
    "dispatcher",
    ABI_VERSION,
    (
        publish_seq("dispatcher"),
        field("draft_issue_seq", U64, "dispatcher", "last written Draft command sequence", hot=True),
        field("draft_worker_generation", U64, "dispatcher", "Draft worker generation fence", hot=True),
        field("draft_round_id", U64, "dispatcher", "Draft command round fence", hot=True),
        field("target_prepare_seq", U64, "dispatcher", "last written Target prepare command sequence", hot=True),
        field("planned_target_generation", U64, "dispatcher", "Target prepare worker generation fence", hot=True),
        field("planned_bank_epoch", U64, "dispatcher", "standby bank epoch fence", hot=True),
        field("target_run_seq", U64, "dispatcher", "last written Target run command sequence", hot=True),
        field("target_round_id", U64, "dispatcher", "Target verification round fence", hot=True),
        field("draft_worker_id", U32, "dispatcher", "Draft worker command target", hot=True),
        field("planned_target_id", U32, "dispatcher", "Target prepare command target", hot=True),
        field("planned_bank_id", U8, "dispatcher", "standby bank selected by prepare command", hot=True),
    ),
)

DRAFT_WORKER_BLOCK = StructLayout(
    "DraftWorkerBlock",
    "draft_worker",
    ABI_VERSION,
    (
        publish_seq("draft_worker"),
        field("request_epoch", U64, "draft_worker", "request generation fence", hot=True),
        field("round_id", U64, "draft_worker", "Draft round fence", hot=True),
        field("observed_issue_seq", U64, "draft_worker", "consumed Draft command sequence", hot=True),
        field("worker_generation", U64, "draft_worker", "executing Draft worker generation", hot=True),
        field("worker_id", U32, "draft_worker", "executing Draft worker id", hot=True),
        field("status", ENUM32, "draft_worker", "DraftStatus enum", hot=True),
        field("result_code", RESULT32, "draft_worker", "fixed-width ResultCode", hot=True),
        field("compute_start_ns", U64, "draft_worker", "actual compute start", hot=False),
        field("compute_end_ns", U64, "draft_worker", "actual compute end", hot=False),
        field("proposal_token_count", U32, "draft_worker", "proposal token count", hot=False),
        field("proposal_handle", ARENA_HANDLE, "draft_worker", "proposal arena handle", hot=False),
        field("draft_state_handle", ARENA_HANDLE, "draft_worker", "shared immutable Draft snapshot metadata handle", hot=False),
    ),
    row_alignment=CACHE_LINE_BYTES,
    multi_writer=True,
)

TARGET_COMPUTE_BLOCK = StructLayout(
    "TargetComputeBlock",
    "target_compute_lane",
    ABI_VERSION,
    (
        publish_seq("target_compute_lane"),
        field("request_epoch", U64, "target_compute_lane", "request generation fence", hot=True),
        field("round_id", U64, "target_compute_lane", "Target round fence", hot=True),
        field("observed_run_seq", U64, "target_compute_lane", "consumed Target run command sequence", hot=True),
        field("target_generation", U64, "target_compute_lane", "executing Target worker generation", hot=True),
        field("bank_epoch", U64, "target_compute_lane", "active bank epoch fence", hot=True),
        field("target_kv_version", U64, "target_compute_lane", "Target KV version after verification", hot=True),
        field("target_id", U32, "target_compute_lane", "executing Target worker id", hot=True),
        field("status", ENUM32, "target_compute_lane", "TargetStatus enum", hot=True),
        field("result_code", RESULT32, "target_compute_lane", "fixed-width ResultCode", hot=True),
        field("bank_id", U8, "target_compute_lane", "active bank id used for verification", hot=True),
        field("compute_start_ns", U64, "target_compute_lane", "actual compute start", hot=False),
        field("compute_end_ns", U64, "target_compute_lane", "actual compute end", hot=False),
        field("output_count", U32, "target_compute_lane", "cumulative immutable output length", hot=False),
        field("output_finished", U8, "target_compute_lane", "authoritative generation termination", hot=False),
        field("output_handle", ARENA_HANDLE, "target_compute_lane", "request cumulative output prefix", hot=False),
        field("accepted_draft_count", U32, "target_compute_lane", "accepted Draft prefix token count", hot=False),
        field("committed_delta_count", U32, "target_compute_lane", "committed token delta count", hot=False),
        field("last_committed_token", U32, "target_compute_lane", "last committed token for next-round anchor", hot=False),
        field("logical_kv_len", U32, "target_compute_lane", "valid Target KV length", hot=False),
        field("dirty_begin_block", U32, "target_compute_lane", "first dirty KV block for D2H", hot=False),
        field("dirty_block_count", U32, "target_compute_lane", "dirty KV block count for D2H", hot=False),
        field("committed_delta_handle", ARENA_HANDLE, "target_compute_lane", "committed delta token arena handle", hot=False),
    ),
    row_alignment=CACHE_LINE_BYTES,
    multi_writer=True,
)

D2H_BLOCK = StructLayout(
    "D2HBlock",
    "target_copy_lane",
    ABI_VERSION,
    (
        publish_seq("target_copy_lane"),
        field("request_epoch", U64, "target_copy_lane", "request generation fence", hot=True),
        field("round_id", U64, "target_copy_lane", "Target KV round fence", hot=True),
        field("d2h_op_seq", U64, "target_copy_lane", "D2H operation sequence", hot=True),
        field("target_generation", U64, "target_copy_lane", "D2H source Target generation", hot=True),
        field("source_bank_epoch", U64, "target_copy_lane", "D2H source bank epoch", hot=True),
        field("host_slot_generation", U64, "target_copy_lane", "HostKV slot generation fence", hot=True),
        field("writer_version", U64, "target_copy_lane", "HostKV writer version lease", hot=True),
        field("ready_version", U64, "target_copy_lane", "HostKV version published after D2H", hot=True),
        field("target_id", U32, "target_copy_lane", "D2H source Target worker id", hot=True),
        field("status", ENUM32, "target_copy_lane", "D2HStatus enum", hot=True),
        field("result_code", RESULT32, "target_copy_lane", "fixed-width ResultCode", hot=True),
        field("source_bank_id", U8, "target_copy_lane", "D2H source bank id", hot=True),
        field("committed_blocks", U32, "target_copy_lane", "blocks committed to HostKV", hot=False),
        field("logical_kv_len", U32, "target_copy_lane", "valid HostKV logical length", hot=False),
        field("copy_start_time_ns", U64, "target_copy_lane", "D2H submission timestamp", hot=False),
        field("copy_bytes", U64, "target_copy_lane", "D2H byte workload", hot=False),
    ),
    row_alignment=CACHE_LINE_BYTES,
    multi_writer=True,
)

H2D_BLOCK = StructLayout(
    "H2DBlock",
    "target_copy_lane",
    ABI_VERSION,
    (
        publish_seq("target_copy_lane"),
        field("request_epoch", U64, "target_copy_lane", "request generation fence", hot=True),
        field("round_id", U64, "target_copy_lane", "Target round being prepared", hot=True),
        field("observed_prepare_seq", U64, "target_copy_lane", "consumed Target prepare command sequence", hot=True),
        field("target_generation", U64, "target_copy_lane", "H2D destination Target generation", hot=True),
        field("source_host_version", U64, "target_copy_lane", "required HostKV source version", hot=True),
        field("destination_bank_epoch", U64, "target_copy_lane", "destination bank epoch fence", hot=True),
        field("gpu_ready_version", U64, "target_copy_lane", "Target KV version loaded to GPU", hot=True),
        field("target_id", U32, "target_copy_lane", "H2D destination Target worker id", hot=True),
        field("status", ENUM32, "target_copy_lane", "H2DStatus enum", hot=True),
        field("result_code", RESULT32, "target_copy_lane", "fixed-width ResultCode", hot=True),
        field("destination_bank_id", U8, "target_copy_lane", "destination bank id", hot=True),
        field("copied_blocks", U32, "target_copy_lane", "copied block count", hot=False),
        field("copy_start_time_ns", U64, "target_copy_lane", "H2D submission timestamp", hot=False),
        field("copy_bytes", U64, "target_copy_lane", "H2D byte workload", hot=False),
    ),
    row_alignment=CACHE_LINE_BYTES,
    multi_writer=True,
)

HOSTKV_ALLOCATION_BLOCK = StructLayout(
    "HostKVAllocationBlock",
    "engine_hostkv_allocator",
    ABI_VERSION,
    (
        publish_seq("engine_hostkv_allocator"),
        field("request_epoch", U64, "engine_hostkv_allocator", "request generation fence", hot=True),
        field("host_slot_generation", U64, "engine_hostkv_allocator", "HostKV slot generation", hot=True),
        field("writer_lease_generation", U64, "engine_hostkv_allocator", "long-lived writer lease generation", hot=True),
        field("host_slot", U32, "engine_hostkv_allocator", "HostKV slot id", hot=True),
        field("capacity_blocks", U32, "engine_hostkv_allocator", "HostKV block capacity", hot=False),
        field("offset_blocks", U64, "engine_hostkv_allocator", "HostKV arena block offset", hot=False),
    ),
)

WORKER_COMMON_BLOCK = StructLayout(
    "WorkerCommonBlock",
    "worker",
    ABI_VERSION,
    (
        publish_seq("worker"),
        field("worker_id", U32, "worker", "numeric worker id", hot=True),
        field("role", ENUM32, "worker", "WorkerRole enum", hot=True),
        field("worker_generation", U64, "worker", "worker process generation", hot=True),
        field("status", ENUM32, "worker", "WorkerStatus enum", hot=True),
        field("command_consumer_seq", U64, "worker", "last consumed command sequence", hot=True, required=False),
        field("max_batch_size", U32, "worker", "maximum batch request count", hot=False),
        field("max_batch_tokens", U32, "worker", "maximum batch token count", hot=False),
    ),
)

DRAFT_RUNTIME_BLOCK = StructLayout(
    "DraftRuntimeBlock",
    "draft_worker",
    ABI_VERSION,
    (
        publish_seq("draft_worker"),
        field("current_batch_seq", U64, "draft_worker", "current Draft batch sequence", hot=True),
        field("compute_status", ENUM32, "draft_worker", "ComputeStatus enum", hot=True),
        field("compute_start_time_ns", U64, "draft_worker", "current batch start timestamp", hot=True),
        field("batch_request_count", U32, "draft_worker", "current batch request count", hot=True),
        field("batch_token_count", U32, "draft_worker", "current batch token workload", hot=True),
    ),
)

TARGET_COMPUTE_RUNTIME_BLOCK = StructLayout(
    "TargetComputeRuntimeBlock",
    "target_compute_lane",
    ABI_VERSION,
    (
        publish_seq("target_compute_lane"),
        field("compute_batch_seq", U64, "target_compute_lane", "current Target compute batch sequence", hot=True),
        field("compute_status", ENUM32, "target_compute_lane", "ComputeStatus enum", hot=True),
        field("compute_start_time_ns", U64, "target_compute_lane", "verification start timestamp", hot=True),
        field("compute_request_count", U32, "target_compute_lane", "verification request count", hot=True),
        field("compute_token_count", U32, "target_compute_lane", "verification token workload", hot=True),
    ),
)

TARGET_COPY_RUNTIME_BLOCK = StructLayout(
    "TargetCopyRuntimeBlock",
    "target_copy_lane",
    ABI_VERSION,
    (
        publish_seq("target_copy_lane"),
        field("copy_op_seq", U64, "target_copy_lane", "current CopyLane operation sequence", hot=True),
        field("copy_status", ENUM32, "target_copy_lane", "CopyStatus enum", hot=True),
        field("copy_start_time_ns", U64, "target_copy_lane", "copy start timestamp", hot=True),
        field("copy_bytes", U64, "target_copy_lane", "copy byte workload", hot=True),
    ),
)

BANK_BLOCK = StructLayout(
    "BankBlock",
    "target_copy_lane",
    ABI_VERSION,
    (
        publish_seq("target_copy_lane"),
        field("bank_id", U8, "target_copy_lane", "fixed bank id", hot=True),
        field("bank_epoch", U64, "target_copy_lane", "bank reset/reuse generation", hot=True),
        field("role", ENUM32, "target_copy_lane", "BankRole enum", hot=True),
        field("state", ENUM32, "target_copy_lane", "BankState enum", hot=True),
        field("batch_seq", U64, "target_copy_lane", "batch currently resident in bank", hot=True),
        field("capacity_blocks", U32, "target_copy_lane", "bank block capacity", hot=False),
        field("alloc_ptr_blocks", U32, "target_copy_lane", "bank bump allocation pointer", hot=False),
        field("capacity_rows", U32, "target_copy_lane", "bank request row capacity", hot=False),
        field("alloc_rows", U32, "target_copy_lane", "bank allocated request rows", hot=False),
    ),
)

from dataclasses import replace as _replace
from .draft_layout import (DISPATCH_FIELDS, COMPUTE_FIELDS, DRAFT_HOSTKV_BLOCK,
                           DRAFT_D2H_BLOCK, DRAFT_H2D_BLOCK, worker_layouts)
DISPATCH_BLOCK = _replace(DISPATCH_BLOCK, fields=DISPATCH_BLOCK.fields + DISPATCH_FIELDS)
DRAFT_WORKER_BLOCK = _replace(DRAFT_WORKER_BLOCK, fields=DRAFT_WORKER_BLOCK.fields + COMPUTE_FIELDS)
DRAFT_COPY_RUNTIME_BLOCK, DRAFT_BANK_BLOCK = worker_layouts(TARGET_COPY_RUNTIME_BLOCK, BANK_BLOCK)

REQUEST_BLOCKS = (
    ENGINE_BLOCK,
    DISPATCH_BLOCK,
    DRAFT_WORKER_BLOCK,
    TARGET_COMPUTE_BLOCK,
    D2H_BLOCK,
    H2D_BLOCK,
    HOSTKV_ALLOCATION_BLOCK,
    DRAFT_HOSTKV_BLOCK, DRAFT_D2H_BLOCK, DRAFT_H2D_BLOCK,
)

WORKER_BLOCKS = (
    WORKER_COMMON_BLOCK,
    DRAFT_RUNTIME_BLOCK,
    TARGET_COMPUTE_RUNTIME_BLOCK,
    TARGET_COPY_RUNTIME_BLOCK,
    BANK_BLOCK,
    DRAFT_COPY_RUNTIME_BLOCK, DRAFT_BANK_BLOCK,
)

ALL_STRUCT_LAYOUTS = REQUEST_BLOCKS + WORKER_BLOCKS


def request_row_size_bytes(blocks: Iterable[StructLayout] = REQUEST_BLOCKS) -> int:
    return sum(block.size for block in blocks)


def physical_request_row_size_bytes(blocks: Iterable[StructLayout] = REQUEST_BLOCKS) -> int:
    return sum(block.row_stride for block in blocks)


def active_request_table_bytes(active_requests: int) -> int:
    if active_requests < 0:
        raise ValueError("active_requests must be non-negative")
    return physical_request_row_size_bytes() * active_requests
