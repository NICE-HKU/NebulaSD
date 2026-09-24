"""Read-only scheduling inputs; no backend, IPC or registry access in policy."""

from dataclasses import dataclass
from typing import Mapping

from nebulasd.core.enums import WorkerRole, StateChangeBlockKind as Kind
from nebulasd.core.handles import ArenaHandle, HostKVArenaHandle


def value(row, field, default=None):
    if row is None:
        return default
    try:
        return row.get(field)
    except KeyError:
        return default


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    role: WorkerRole
    generation: int = 1
    max_batch_size: int = 4
    max_batch_tokens: int = 256
    prefill_max_batch_tokens: int | None = None
    verify_max_batch_tokens: int | None = None
    bank_blocks: int = 128
    bank_rows: int = 8
    block_size: int = 16
    draft_banked: bool = False  # Legacy explicit WorkerSpec users; public bootstrap always enables it.

    def __post_init__(self):
        prefill = self.max_batch_tokens if self.prefill_max_batch_tokens is None else self.prefill_max_batch_tokens
        verify = self.max_batch_tokens if self.verify_max_batch_tokens is None else self.verify_max_batch_tokens
        if self.worker_id < 0 or min(self.generation, self.max_batch_size, self.max_batch_tokens,
                                     prefill, verify, self.bank_blocks, self.bank_rows, self.block_size) <= 0:
            raise ValueError("invalid Worker capacities/identity")
        object.__setattr__(self, 'prefill_max_batch_tokens', prefill)
        object.__setattr__(self, 'verify_max_batch_tokens', verify)


@dataclass(frozen=True)
class RequestInput:
    slot: int
    epoch: int
    arrival_seq: int
    prompt_count: int
    max_new_tokens: int
    proposal_depth: int
    prompt: ArenaHandle
    config: ArenaHandle
    output: ArenaHandle
    output_count: int
    host: HostKVArenaHandle
    capacity_blocks: int
    admitted_ns: int
    ready_ns: int = 0


@dataclass(frozen=True)
class SchedulingView:
    requests: Mapping[int, RequestInput]
    workers: tuple[WorkerSpec, ...]
    rows: Mapping[tuple[Kind, int], object]
    prepared: Mapping[int, object]  # Exact issued Prepare commands, not Bank mirrors.
    inflight: Mapping[int, object]
    sequences: Mapping[int, int]

    work_state: object = None
    # Owner request records are read only during the synchronous invocation.
    # Standalone snapshot callers may instead supply REQUEST_ENGINE rows.
    request_states: Mapping | None = None

    def row(self, kind, index):
        return self.rows.get((kind, index))
