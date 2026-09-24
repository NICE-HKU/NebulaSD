"""Minimal cudart binding; no reliance on optional torch cudart Python methods."""

import ctypes
import ctypes.util
from pathlib import Path


class CudaMemcpy:
    def __init__(self, library: str | None = None) -> None:
        import torch

        if library is None:
            library = ctypes.util.find_library("cudart")
        if library is None:
            root = Path(torch.__file__).resolve().parent
            candidates = sorted((root / "lib").glob("libcudart.so*"))
            candidates += sorted((root.parent / "nvidia/cuda_runtime/lib").glob("libcudart.so*"))
            library = str(candidates[0]) if candidates else "libcudart.so"
        self._library = ctypes.CDLL(library)
        self._copy = self._library.cudaMemcpyAsync
        self._copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self._copy.restype = ctypes.c_int
        self._stream_create = self._library.cudaStreamCreateWithFlags
        self._stream_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self._stream_create.restype = ctypes.c_int
        self._stream_sync = self._library.cudaStreamSynchronize
        self._stream_sync.argtypes = [ctypes.c_void_p]
        self._stream_sync.restype = ctypes.c_int
        self._stream_destroy = self._library.cudaStreamDestroy
        self._stream_destroy.argtypes = [ctypes.c_void_p]
        self._stream_destroy.restype = ctypes.c_int

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        status = self._stream_create(ctypes.byref(stream), 1)  # cudaStreamNonBlocking
        if status:
            raise RuntimeError(f"cudaStreamCreateWithFlags failed with CUDA status {status}")
        if not stream.value:
            raise RuntimeError("cudaStreamCreateWithFlags returned a null stream")
        return stream.value

    def synchronize_stream(self, stream: int) -> None:
        status = self._stream_sync(stream)
        if status:
            raise RuntimeError(f"cudaStreamSynchronize failed with CUDA status {status}")

    def destroy_stream(self, stream: int) -> None:
        status = self._stream_destroy(stream)
        if status:
            raise RuntimeError(f"cudaStreamDestroy failed with CUDA status {status}")

    def copy(self, destination: int, source: int, nbytes: int, kind: int, stream: int) -> None:
        status = self._copy(destination, source, nbytes, kind, stream)
        if status:
            raise RuntimeError(f"cudaMemcpyAsync failed with CUDA status {status}")
