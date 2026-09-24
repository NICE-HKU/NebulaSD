from contextlib import nullcontext
from types import SimpleNamespace as NS
from threading import Event
from concurrent.futures import ThreadPoolExecutor
import pytest
from nebulasd.kv.cuda_transfer import CudaCopyBackend
from nebulasd.kv.transfer import CopyPlan


def backend(monkeypatch, *, partial=False, wrap_fail=False, sync_fail=False):
    import nebulasd.kv.cuda_transfer as module
    calls=[]; gate=Event(); entered=Event()
    class API:
        def create_stream(self):calls.append('create');return 123
        def copy(self,*a):
            calls.append('copy')
            if partial:raise RuntimeError('partial submit')
        def synchronize_stream(self,h):
            assert h==123;calls.append('sync');entered.set();gate.wait(2)
            if sync_fail:raise RuntimeError('sync failed')
        def destroy_stream(self,h):assert h==123;calls.append('destroy')
    class CUDA:
        def set_device(self,d):calls.append('device')
        def ExternalStream(self,h,device):
            assert h==123
            calls.append('wrap')
            if wrap_fail:raise RuntimeError('wrap failed')
            return NS(cuda_stream=h,wait_event=lambda e: calls.append('wait'))
        def stream(self,s):return nullcontext()
        def Event(self,**kw):return NS(record=lambda s:calls.append('record'))
    monkeypatch.setattr(module,'CudaMemcpy',API)
    b=CudaCopyBackend.__new__(CudaCopyBackend)
    b._torch=NS(cuda=CUDA());b._device=0;b._closed=False;b._close_error=None
    b._stream=b._stream_handle=b._record=b._memcpy=None
    b.k_cache=b.v_cache=NS(shape=(10,),data_ptr=lambda:100)
    b.arena=NS(descriptor=NS(block_bytes=1,k_plane_offset=0,v_plane_offset=10),address=lambda:200,_validate_extent=lambda e:None)
    b._registration=NS(acquire=lambda a:calls.append('register') or object(),release=lambda r:calls.append('release'))
    plan=NS(dependencies=(object(),),regions=(NS(block_count=1,gpu_begin_block=0,host_begin_block=0,extent=NS(offset_blocks=0)),),direction='H2D')
    return b,plan,calls,gate,entered


def test_partial_submit_close_waits_before_release_and_destroy(monkeypatch):
    b,p,c,g,e=backend(monkeypatch,partial=True)
    with pytest.raises(RuntimeError,match='partial submit'):b.launch(p)
    with ThreadPoolExecutor(1) as pool:
        f=pool.submit(b.close);assert e.wait(1)
        assert 'release' not in c and 'destroy' not in c and b.k_cache is not None
        g.set();f.result(2)
    assert c.index('sync')<c.index('release')<c.index('destroy')
    assert b._stream_handle is None and b.k_cache is None
    before=list(c);b.close();assert c==before
    with pytest.raises(RuntimeError,match='closed'):b.launch(p)


def test_failed_sync_retains_memory_and_never_retries_close(monkeypatch):
    b,p,c,g,e=backend(monkeypatch,sync_fail=True);b.launch(p);g.set()
    for _ in range(2):
        with pytest.raises(RuntimeError,match='sync failed'):b.close()
    assert c.count('sync')==1 and 'release' not in c and 'destroy' not in c
    assert b._record is not None and b._stream_handle==123 and b.k_cache is not None


def test_wrapper_failure_still_closes_owned_raw_handle(monkeypatch):
    b,p,c,g,e=backend(monkeypatch,wrap_fail=True)
    with pytest.raises(RuntimeError,match='wrap failed'):b.launch(p)
    assert b._stream is None and b._stream_handle==123
    g.set();b.close();assert 'register' not in c and c[-2:]==['sync','destroy']


def test_cuda_binding_nonblocking_flag_and_fail_stop():
    import ctypes
    from nebulasd.kv.cuda_memcpy import CudaMemcpy
    a=CudaMemcpy.__new__(CudaMemcpy);calls=[]
    def create(ptr,flags):
        assert flags==1
        ctypes.cast(ptr,ctypes.POINTER(ctypes.c_void_p))[0]=456
        return 0
    a._stream_create=create;a._stream_sync=lambda h:calls.append(('sync',h)) or 0
    a._stream_destroy=lambda h:calls.append(('destroy',h)) or 0
    h=a.create_stream();a.synchronize_stream(h);a.destroy_stream(h)
    assert calls==[('sync',456),('destroy',456)]
    a._stream_create=lambda *args:2
    with pytest.raises(RuntimeError,match='CreateWithFlags'):a.create_stream()
    a._stream_sync=lambda h:3
    with pytest.raises(RuntimeError,match='Synchronize'):a.synchronize_stream(h)
    a._stream_destroy=lambda h:4
    with pytest.raises(RuntimeError,match='Destroy'):a.destroy_stream(h)


def test_copy_completion_event_supports_sleeping_host_wait(monkeypatch):
    b,p,c,g,e=backend(monkeypatch)
    flags=[]
    def event(**kw):
        flags.append(kw)
        return NS(record=lambda stream:None, synchronize=lambda:c.append('event_sync'))
    b._torch.cuda.Event=event
    ticket=b.launch(p)
    assert flags[-1] == dict(enable_timing=True,blocking=True)
    ticket.synchronize()
    assert c[-1]=='event_sync'
    g.set();b.close()
