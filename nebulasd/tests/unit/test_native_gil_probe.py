import ctypes as C
import os
import pytest
from nebulasd.ipc.native import library
from support.native_gil_probe import install,SYMBOLS


@pytest.mark.skipif(not os.environ.get('STARSD_NEXT_NATIVE_LIBRARY'),reason='native library required')
def test_native_gil_probe_preserves_atomic_and_table_abi():
    lib=library();old=install()
    try:
        assert all(getattr(lib,n)._flags_ & C._FUNCFLAG_PYTHONAPI for n in SYMBOLS)
        v=C.c_uint64(7);ptr=C.addressof(v)
        assert lib.sd_load(ptr)==7 and lib.sd_cas(ptr,7,9)==1
        assert lib.sd_exchange(ptr,10)==9 and v.value==10
        lib.sd_store(ptr,11);assert v.value==11
        row=(C.c_uint64*3)();address=C.addressof(row)
        offsets=(C.c_uint*1)(8);lengths=(C.c_uint*1)(8);payload=C.c_uint64(123)
        lib.sd_table_publish(address,4,offsets,lengths,C.byref(payload),1)
        out=C.create_string_buffer(16);seq=C.c_uint64()
        assert lib.sd_table_read(address,24,out,C.byref(seq))==1 and seq.value==4
        assert int.from_bytes(out.raw[:8],'little')==123
    finally:
        for name,fn in old.items():setattr(lib,name,fn)
    assert not (lib.sd_load._flags_ & C._FUNCFLAG_PYTHONAPI)
