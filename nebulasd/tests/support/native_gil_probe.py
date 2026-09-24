"""Experiment-only GIL retention for the bounded native control-plane ABI.

native.cpp functions perform finite lock-free atomics/memory copies and do not
wait or call CUDA/Python. Do not apply this to CUDA, blocking I/O or other DLLs.
The existing library object is retained because partitions/rings reference it.
"""
import ctypes as C
from nebulasd.ipc.native import library

SYMBOLS=('sd_native_abi','sd_load','sd_store','sd_exchange','sd_cas','sd_table_publish','sd_table_read','sd_ring_push','sd_ring_peek','sd_ring_ack')


def install():
    lib=library()
    held=C.PyDLL(lib._name)
    original={name:getattr(lib,name) for name in SYMBOLS}
    for name,old in original.items():
        new=getattr(held,name)
        new.restype,new.argtypes=old.restype,old.argtypes
        setattr(lib,name,new)
    lib._gil_probe_handle=held
    assert lib.sd_native_abi()==1
    return original
