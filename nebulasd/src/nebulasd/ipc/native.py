"""Explicitly built native ABI. Missing native transport fails instead of falling back."""

import ctypes as C
import os
from pathlib import Path

_library = None


def library():
    global _library
    if _library is not None:
        return _library
    path = os.environ.get("STARSD_NEXT_NATIVE_LIBRARY")
    if not path:
        raise RuntimeError("set STARSD_NEXT_NATIVE_LIBRARY; build with nebulasd/tools/build_native.py")
    lib = C.CDLL(str(Path(path).resolve()))
    signatures = {
        "sd_table_publish_notify": (C.c_int, [C.c_void_p, C.c_uint64,
            C.POINTER(C.c_uint), C.POINTER(C.c_uint), C.c_void_p, C.c_uint,
            C.c_void_p, C.c_uint64, C.c_uint, C.c_uint, C.c_int]),
        "sd_watch_scan": (C.c_uint, [C.POINTER(C.c_size_t), C.POINTER(C.c_uint),
            C.POINTER(C.c_uint64), C.c_uint, C.c_void_p, C.c_uint,
            C.POINTER(C.c_uint64), C.POINTER(C.c_uint)]),
        "sd_native_abi": (C.c_uint, []),
        "sd_load": (C.c_uint64, [C.c_void_p]),
        "sd_store": (None, [C.c_void_p, C.c_uint64]),
        "sd_exchange": (C.c_uint64, [C.c_void_p, C.c_uint64]),
        "sd_cas": (C.c_int, [C.c_void_p, C.c_uint64, C.c_uint64]),
        "sd_table_publish": (None, [C.c_void_p, C.c_uint64, C.POINTER(C.c_uint),
                                  C.POINTER(C.c_uint), C.c_void_p, C.c_uint]),
        "sd_table_read": (C.c_int, [C.c_void_p, C.c_uint, C.c_void_p, C.POINTER(C.c_uint64)]),
        "sd_ring_push": (C.c_int, [C.c_void_p, C.c_uint64, C.c_uint64, C.c_void_p]),
        "sd_ring_peek": (C.c_int, [C.c_void_p, C.c_uint64, C.c_uint64, C.c_void_p]),
        "sd_ring_ack": (None, [C.c_void_p]),
        "sd_ring_drain": (C.c_uint64, [C.c_void_p, C.c_uint64, C.c_uint64,
                                      C.c_uint64, C.c_void_p]),
    }
    for name, (result, args) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, args
    if lib.sd_native_abi() != 1:
        raise RuntimeError("native control-plane ABI mismatch")
    _library = lib
    return lib
