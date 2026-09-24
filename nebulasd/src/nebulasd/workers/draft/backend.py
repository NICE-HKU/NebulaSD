"""Real linear Draft decode on the shared explicit-layout model primitives."""
from typing import NamedTuple
from dataclasses import dataclass
from time import perf_counter_ns
from nebulasd.core.enums import ProposalKind
from nebulasd.kv.transfer import CopyPlan, CopyRegion, HostCompletedFence
from nebulasd.workers.target.backend import TargetBackend, publish_clock
from nebulasd.workers.work import WorkKind
from .inputs import DraftInputs


class Output(NamedTuple):
    index: int
    proposal: tuple
    proposal_kind: int
    logical: int
    version: int
    dirty_begin: int
    dirty_blocks: int
    committed_count: int


@dataclass(frozen=True, slots=True)
class Result:
    executed_rows: tuple
    rows: tuple
    export_plan: object
    compute_start_ns: int
    compute_end_ns: int


class DraftBackend(TargetBackend):
    input_type = DraftInputs

    def __init__(self, *, snapshots, layout_id, host_arena_id, **kwargs):
        super().__init__(input_options=dict(snapshots=snapshots, layout_id=layout_id,
                                           host_arena_id=host_arena_id), **kwargs)

    def _execute(self, plan):
        torch = self.torch
        torch.cuda.set_device(self.device)
        started = perf_counter_ns()
        publish_clock(self, plan.spec.work_seq, started, 0)
        proposals, lengths = {}, {}
        with torch.inference_mode(), torch.cuda.stream(self.compute_stream):
            initial = [r for r in plan.rows if r.retained == 0]
            if initial:
                seeds = self.model.forward([list(r.suffix) for r in initial],
                    [r.local_row for r in initial], [], kv_block_table=self.block_table)
                for r, seed in zip(initial, seeds, strict=True):
                    proposals[r.index], lengths[r.index] = [int(seed)], len(r.suffix)
            continuation = [r for r in plan.rows if r.retained]
            for offset in range(max((len(r.suffix) for r in continuation), default=0)):
                active = [r for r in continuation if offset < len(r.suffix)]
                seeds = self.model.forward([[r.suffix[offset]] for r in active],
                    [r.local_row for r in active], [r.retained+offset+1 for r in active],
                    kv_block_table=self.block_table)
                for r, seed in zip(active, seeds, strict=True):
                    if offset == len(r.suffix)-1:
                        proposals[r.index], lengths[r.index] = [int(seed)], r.retained+len(r.suffix)
            while True:
                active = [r for r in plan.rows if len(proposals[r.index]) < r.limit
                          and proposals[r.index][-1] not in r.stops]
                if not active:
                    break
                seeds = self.model.forward([[proposals[r.index][-1]] for r in active],
                    [r.local_row for r in active], [lengths[r.index]+1 for r in active],
                    kv_block_table=self.block_table)
                for r, seed in zip(active, seeds, strict=True):
                    proposals[r.index].append(int(seed))
                    lengths[r.index] += 1
            outputs, regions = [], []
            for r in plan.rows:
                logical = lengths[r.index]
                dirty, valid = r.retained//self.block_size, (logical+self.block_size-1)//self.block_size
                outputs.append(Output(r.index, tuple(proposals[r.index]), int(ProposalKind.LINEAR),
                    logical, r.version, dirty, valid-dirty, r.committed_count))
                regions.append(CopyRegion(r.extent, r.gpu_begin+dirty, dirty, valid-dirty))
            done = torch.cuda.Event()
            done.record(self.compute_stream)
        done.synchronize()
        ended = perf_counter_ns()
        publish_clock(self, plan.spec.work_seq, started, ended)
        return Result(tuple(r.index for r in plan.rows), tuple(outputs),
            CopyPlan('D2H', tuple(regions), (HostCompletedFence(),)), started, ended)
