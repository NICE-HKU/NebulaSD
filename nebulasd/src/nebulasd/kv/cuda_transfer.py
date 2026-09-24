"""Pinned HostKV DMA on an explicitly owned CUDA stream.

No SwiftLLM import is required: the canonical facade supplies cache tensors.
K/V are block-major contiguous arrays. Unsupported layouts fail before DMA.
"""

from __future__ import annotations

from .arena import SharedHostKVArena
from .host_registration import CudaHostRegistrationAdapter
from .registration_pool import HostRegistrationPool
from .cuda_memcpy import CudaMemcpy
from .transfer import CopyPlan, HostCompletedFence


class _CudaTicket:
    def __init__(self, start, done) -> None:
        self.start, self.done = start, done

    def query(self) -> bool:
        return bool(self.done.query())

    def synchronize(self) -> None:
        self.done.synchronize()

    def duration_ms(self) -> float:
        return float(self.start.elapsed_time(self.done))


class CudaCopyBackend:
    def __init__(self, *, arena: SharedHostKVArena, k_cache, v_cache,
                 registration: CudaHostRegistrationAdapter | None = None,
                 registration_pool: HostRegistrationPool | None = None) -> None:
        import torch

        if (not k_cache.is_cuda or not v_cache.is_cuda
                or k_cache.device != v_cache.device or k_cache.dtype != v_cache.dtype
                or k_cache.shape != v_cache.shape
                or not k_cache.is_contiguous() or not v_cache.is_contiguous()):
            raise ValueError("copy backend requires matching contiguous CUDA KV caches")
        block_bytes = k_cache[0].numel() * k_cache.element_size()
        if arena.descriptor.block_bytes != block_bytes:
            raise ValueError("HostKV block byte size does not match GPU layout")
        if (arena.descriptor.dtype != str(k_cache.dtype)
                or arena.descriptor.kv_block_shape != tuple(k_cache.shape[1:])):
            raise ValueError("HostKV dtype/block shape does not match GPU layout")
        self._torch = torch
        self.arena = arena
        self.k_cache, self.v_cache = k_cache, v_cache
        self._device = k_cache.device
        self._registration = registration_pool or HostRegistrationPool(registration)
        self._record = None
        self._stream = None
        self._stream_handle = None
        self._memcpy = None
        self._closed = False
        self._close_error = None

    def fork(self):
        """Own another stream/events, sharing only cache storage and registration."""
        return type(self)(arena=self.arena, k_cache=self.k_cache, v_cache=self.v_cache,
                          registration_pool=self._registration)

    def initialize(self) -> None:
        """Cold startup: own stream and register HostKV before accepting WORK."""
        torch = self._torch
        if self._closed:
            raise RuntimeError("CUDA copy backend is closed")
        torch.cuda.set_device(self._device)
        if self._stream is None:
            self._memcpy = CudaMemcpy()
            # ExternalStream borrows the handle; only this backend destroys it.
            # Retain the raw handle even if wrapping/registration fails.
            self._stream_handle = self._memcpy.create_stream()
            self._stream = torch.cuda.ExternalStream(self._stream_handle, device=self._device)
            self._record = self._registration.acquire(self.arena)

    def launch(self, plan: CopyPlan) -> _CudaTicket:
        self.initialize()
        torch = self._torch
        descriptor = self.arena.descriptor
        for region in plan.regions:
            self.arena._validate_extent(region.extent)
            if region.gpu_begin_block + region.block_count > self.k_cache.shape[0]:
                raise ValueError("copy exceeds GPU KV cache")
        with torch.cuda.stream(self._stream):
            for dependency in plan.dependencies:
                if not isinstance(dependency, HostCompletedFence):
                    self._stream.wait_event(dependency)
            start = torch.cuda.Event(enable_timing=True)
            # cudaEventBlockingSync lets an event waiter sleep instead of
            # spinning on the CPU. Query-based completion remains supported.
            done = torch.cuda.Event(enable_timing=True, blocking=True)
            start.record(self._stream)
            for region in plan.regions:
                count = region.block_count * descriptor.block_bytes
                if count == 0:
                    continue  # Metadata-only versions still honor events above.
                host_offset = (region.extent.offset_blocks + region.host_begin_block) * descriptor.block_bytes
                gpu_offset = region.gpu_begin_block * descriptor.block_bytes
                for cache, plane in ((self.k_cache, descriptor.k_plane_offset),
                                     (self.v_cache, descriptor.v_plane_offset)):
                    host = self.arena.address() + plane + host_offset
                    gpu = cache.data_ptr() + gpu_offset
                    dst, src, kind = (host, gpu, 2) if plan.direction == "D2H" else (gpu, host, 1)
                    self._memcpy.copy(dst, src, count, kind, self._stream.cuda_stream)
            done.record(self._stream)
        return _CudaTicket(start, done)

    def close(self) -> None:
        if self._close_error is not None:
            raise self._close_error
        if self._closed:
            return
        self._closed = True
        try:
            self._torch.cuda.set_device(self._device)
            if self._stream_handle is not None:
                # Partial launches may lack a done event. No memory release or
                # stream destruction unless physical completion is established.
                self._memcpy.synchronize_stream(self._stream_handle)
            if self._record is not None:
                self._registration.release(self._record)
                self._record = None
            if self._stream_handle is not None:
                self._memcpy.destroy_stream(self._stream_handle)
                self._stream_handle = None
        except BaseException as error:
            self._close_error = error
            raise
        self._stream = None
        self._memcpy = None
        self.k_cache = self.v_cache = None
