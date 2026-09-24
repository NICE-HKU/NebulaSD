"""Canonical GPU execution; the non-isolated runner composes CPU inputs."""
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple
from dataclasses import dataclass
from time import perf_counter_ns
from .inputs import TargetInputs
from nebulasd.kv.transfer import CopyPlan, CopyRegion, HostCompletedFence
from nebulasd.workers.work import WorkKind


def publish_clock(backend, seq, started, ended):
    backend.compute_clock = (seq, started, ended)
    slot = getattr(backend, 'clock_slot', None)
    if slot is not None:
        with slot.get_lock():
            slot[:] = backend.compute_clock
    if backend.clock_wakeup is not None:
        backend.clock_wakeup.set()


class Output(NamedTuple):
    index: int
    tokens: tuple
    accepted: int
    logical: int
    version: int
    dirty_begin: int
    dirty_blocks: int


@dataclass(frozen=True)
class Result:
    executed_rows: tuple
    rows: tuple[Output, ...]
    export_plan: object
    compute_start_ns: int
    compute_end_ns: int


class TargetBackend:
    input_type = TargetInputs
    def __init__(self, *, model_path, device, blocks_per_bank, capacity_rows,
                 host, tokens, configs, proposals, block_size=16, max_batch_tokens=4096, job_threads=True, input_options=None):
        from nebulasd.canonical import import_canonical
        import_canonical()
        import torch
        from swiftllm.engine_config import EngineConfig
        from swiftllm.worker.model import LlamaModel
        self.torch, self.device, self.host = torch, device, host
        self.compute_clock = (0,0,0)
        self.clock_wakeup = None
        self.block_size = block_size
        self.max_batch_tokens = max_batch_tokens
        torch.cuda.set_device(device)
        config = EngineConfig(model_path, False, block_size, 0.9, 0,
            capacity_rows, blocks_per_bank, capacity_rows, max_batch_tokens)
        self.model = LlamaModel(config)
        self.model.load_weights()
        self.model.init_kvcache_and_swap(2 * blocks_per_bank, external_layout=True)
        self.block_table = torch.empty((capacity_rows, blocks_per_bank), dtype=torch.int32, device=device)
        self.compute_stream = torch.cuda.Stream(device=device)
        self.metadata_streams = tuple(torch.cuda.Stream(device=device) for _ in range(2)) if job_threads else ()
        self.input_pool = ThreadPoolExecutor(2, thread_name_prefix='worker-input') if job_threads else None
        self.inputs = (self.input_type(host=host, tokens=tokens, configs=configs, proposals=proposals,
            input_pool=self.input_pool, block_size=block_size, max_batch_tokens=max_batch_tokens,
            **(input_options or {})) if job_threads else None)
        self.metadata_pools = tuple(ThreadPoolExecutor(1, thread_name_prefix=f'target-metadata-{b}') for b in range(2)) if job_threads else ()
        self.compute_pool = ThreadPoolExecutor(1, thread_name_prefix='target-compute') if job_threads else None
        torch.cuda.synchronize(device)  # Cold weights/cache initialization only.

    def write_metadata(self, layout, imports):
        def write():
            torch = self.torch
            torch.cuda.set_device(self.device)
            stream = self.metadata_streams[layout.bank_id]
            with torch.cuda.stream(stream):
                # Every reachable entry is overwritten; stale tail is inaccessible.
                for row, begin, count in zip(layout.rows, layout.offsets, layout.capacities):
                    self.block_table[row, :count] = torch.arange(begin, begin + count, dtype=torch.int32, device=self.device)
                done = torch.cuda.Event()
                done.record(stream)
            done.synchronize()  # Job fence, never owner/control; other Bank has its own job unit.
            return HostCompletedFence()
        return self.metadata_pools[layout.bank_id].submit(write)

    def execute(self, plan):
        return self.compute_pool.submit(self._execute, plan)

    def _execute(self, plan):
        from swiftllm.speculative import VerifyPlanItem, compute_acceptance_for_plan
        torch = self.torch
        torch.cuda.set_device(self.device)
        started = perf_counter_ns()
        publish_clock(self, plan.spec.work_seq, started, 0)
        with torch.inference_mode(), torch.cuda.stream(self.compute_stream):
            prefill = plan.spec.operation == WorkKind.TARGET_PREFILL
            if prefill:
                posterior = self.model.forward([list(r.prompt) for r in plan.rows],
                    [r.local_row for r in plan.rows], [], kv_block_table=self.block_table)
                batches = [(r, (int(token),), 0) for r, token in zip(plan.rows, posterior, strict=True)]
            else:
                inputs, ids, lengths, verifies = [], [], [], []
                for row in plan.rows:
                    tokens = [row.anchor, *row.proposal]
                    lens = list(range(row.logical + 1, row.logical + len(tokens) + 1))
                    verifies.append(VerifyPlanItem(None, tokens, lens, list(row.proposal), row.proposal_kind,
                        len(inputs), len(tokens)))
                    inputs.extend([t] for t in tokens)
                    ids.extend([row.local_row] * len(tokens))
                    lengths.extend(lens)
                posterior = self.model.forward(inputs, ids, lengths, kv_block_table=self.block_table)
                batches = []
                for row, verify in zip(plan.rows, verifies):
                    tokens, accepted = compute_acceptance_for_plan(verify,
                        posterior[verify.output_row_start:verify.output_row_start + verify.output_row_count],
                        remaining_output_len=row.remaining, stop_token_ids=row.stops)
                    batches.append((row, tuple(tokens), accepted))
            outputs, regions = [], []
            for row, tokens, accepted in batches:
                logical = len(row.prompt) if prefill else row.logical + len(tokens)
                # Logical crop retains the accepted prefix. Rejected physical tail
                # is unreachable via future sequence lengths, and is overwritten.
                blocks = (logical + self.block_size - 1) // self.block_size
                dirty = 0 if prefill else row.logical // self.block_size
                version = 1 if prefill else row.version + 2
                outputs.append(Output(row.index, tuple(tokens), accepted, logical, version, dirty, blocks-dirty))
                regions.append(CopyRegion(row.extent, row.gpu_begin + dirty, dirty, blocks-dirty))
            done = torch.cuda.Event()
            done.record(self.compute_stream)
        done.synchronize()
        ended = perf_counter_ns()
        publish_clock(self, plan.spec.work_seq, started, ended)
        return Result(tuple(r.index for r in plan.rows), tuple(outputs),
            CopyPlan('D2H', tuple(regions), (HostCompletedFence(),)), started, ended)

    def close(self):
        for pool in (*self.metadata_pools, self.input_pool, self.compute_pool):
            if pool is not None:
                pool.shutdown(wait=True)
        # DMA child has already closed/joined (ExitStack order). Drop exported
        # allocations explicitly before Python/CUDA module teardown.
        self.model.k_cache = self.model.v_cache = None
        import gc
        gc.collect()
        self.torch.cuda.ipc_collect()
