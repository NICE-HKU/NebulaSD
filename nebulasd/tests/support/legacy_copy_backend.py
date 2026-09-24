"""Benchmark adapter calling the canonical old KV copy methods unchanged.

This measures the copy primitive, including its staging allocations. It does
not include the old Engine/RPC/replay control path. Streams are per batch, as
in the old staged facade; host registration is shared and amortized.
"""

from types import SimpleNamespace
from nebulasd.kv.arena import HostKVWriteLease
from nebulasd.kv.cuda_transfer import _CudaTicket


class LegacyTicket(_CudaTicket):
    def __init__(self, start, done, views):
        super().__init__(start, done)
        self.views = views

    def query(self):
        if not super().query():
            return False
        while self.views:
            self.views.pop().release()
        return True


class LegacyCopyBackend:
    def __init__(self, arena, k_cache, v_cache, pool):
        self.arena, self.k_cache, self.v_cache = arena, k_cache, v_cache
        self.pool, self.record, self.stream = pool, None, None
        self.views = []

    def launch(self, plan):
        import torch
        from swiftllm.server.starsd_target_facade import SwiftLLMStarsDTargetFacade as Old
        torch.cuda.set_device(self.k_cache.device)
        if self.record is None:
            self.record = self.pool.acquire(self.arena)
        self.stream = torch.cuda.Stream(device=self.k_cache.device)
        start, done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            for event in plan.dependencies:
                self.stream.wait_event(event)
            start.record(self.stream)
            for region in plan.regions:
                if plan.direction == "D2H":
                    lease = HostKVWriteLease.for_extent(region.extent, dirty_begin_block=region.host_begin_block,
                                                       dirty_block_count=region.block_count)
                    view = self.arena.writer_view(region.extent, lease=lease)
                else:
                    view = self.arena.view(region.extent, begin_block=region.host_begin_block,
                                           block_count=region.block_count)
                self.views.append(view)
                location = SimpleNamespace(bank_base_block=0, request_start_block=region.gpu_begin_block)
                for cache, host_view, plane in ((self.k_cache, view.k, self.arena.descriptor.k_plane_offset),
                                                (self.v_cache, view.v, self.arena.descriptor.v_plane_offset)):
                    if plan.direction == "D2H":
                        Old._copy_bank_to_host(cache, host_view, location, 0, region.block_count)
                    else:
                        address = self.arena.address() + plane + (region.extent.offset_blocks + region.host_begin_block) * self.arena.descriptor.block_bytes
                        Old._copy_host_to_bank(cache, host_view, location, region.block_count,
                                               host_address=address, expected_nbytes=len(host_view))
            done.record(self.stream)
        return LegacyTicket(start, done, self.views)

    def close(self):
        if self.stream is not None:
            self.stream.synchronize()
        while self.views:
            self.views.pop().release()
        if self.record is not None:
            self.pool.release(self.record)
            self.record = None
        self.stream = None
