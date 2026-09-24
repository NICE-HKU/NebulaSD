"""Placement must preserve the shared arena contract and fail without leaks."""
import errno
from multiprocessing import shared_memory
import pytest
from nebulasd import NebulaSDConfig
from nebulasd.kv import numa
from nebulasd.kv.arena import SharedHostKVArena

@pytest.mark.parametrize('policy,nodes', [('bad',()),('default',(1,)),('bind',()),('interleave',(1,1)),('bind',(True,)),('bind',(-1,)),('bind',[1])])
def test_invalid_config(policy,nodes):
    with pytest.raises(ValueError):
        NebulaSDConfig(draft_host_numa_policy=policy,draft_host_numa_nodes=nodes)
    with pytest.raises(ValueError):
        NebulaSDConfig(target_host_numa_policy=policy,target_host_numa_nodes=nodes)

def test_default_never_loads_or_touches_numa(monkeypatch):
    monkeypatch.setattr(numa,'available_nodes',lambda:pytest.fail('default queried NUMA'))
    a=SharedHostKVArena.create(total_blocks=2,block_bytes=32)
    try:assert a.numa_placement is None
    finally:a.close();a.unlink()

@pytest.mark.parametrize('failure',['node','mbind','reserve','verify'])
def test_failed_creation_unlinks_shared_memory(monkeypatch,failure):
    real=shared_memory.SharedMemory;names=[]
    def create(*args,**kwargs):
        s=real(*args,**kwargs)
        if kwargs.get('create'):names.append(s.name)
        return s
    monkeypatch.setattr(shared_memory,'SharedMemory',create)
    monkeypatch.setattr(numa,'available_nodes',lambda: (1,))
    monkeypatch.setattr(numa,'_mbind',lambda *args:None)
    if failure=='mbind':
        def fail(*args):raise PermissionError('denied')
        monkeypatch.setattr(numa,'_mbind',fail)
    if failure=='reserve':
        def fail(*args):raise OSError(errno.ENOSPC,'no space')
        monkeypatch.setattr(numa.os,'posix_fallocate',fail)
    if failure=='verify':monkeypatch.setattr(numa,'residency',lambda address:{2:16})
    with pytest.raises((ValueError,OSError,RuntimeError)):
        SharedHostKVArena.create(total_blocks=8,block_bytes=4096,numa_policy='bind',numa_nodes=(2,) if failure=='node' else (1,))
    assert len(names)==1
    with pytest.raises(FileNotFoundError):real(name=names[0])

@pytest.mark.parametrize('policy',['bind','interleave'])
def test_real_placement_preserves_offsets_and_shared_data(policy):
    if not numa.ctypes.util.find_library('numa'):pytest.skip('libnuma unavailable')
    try:nodes=numa.available_nodes()
    except (RuntimeError,OSError):pytest.skip('Linux NUMA unavailable')
    if not nodes:pytest.skip('no allowed memory nodes')
    nodes=nodes[:1] if policy=='bind' else nodes[:2]
    try:a=SharedHostKVArena.create(total_blocks=8,block_bytes=4096,numa_policy=policy,numa_nodes=nodes)
    except PermissionError:pytest.skip('mbind denied by execution environment')
    attached=None
    try:
        assert a.numa_placement['verified']
        assert a.descriptor.total_blocks==8 and a.descriptor.v_plane_offset==8*4096
        assert set(a.numa_placement['resident_pages'])<=set(nodes)
        attached=SharedHostKVArena.attach(a.descriptor)
        a._shm.buf[4096:4100]=b'kv01'
        assert bytes(attached._shm.buf[4096:4100])==b'kv01'
        assert attached.descriptor==a.descriptor
    finally:
        if attached:attached.close()
        a.close();a.unlink()

def test_stage_policies_reach_owned_resources():
    from nebulasd.engine.resources import ControlResources
    from nebulasd.core.enums import WorkerRole
    from nebulasd.scheduler.views import WorkerSpec
    if not numa.ctypes.util.find_library('numa'):pytest.skip('libnuma unavailable')
    try:nodes=numa.available_nodes()
    except (RuntimeError,OSError):pytest.skip('Linux NUMA unavailable')
    if not nodes:pytest.skip('no memory nodes')
    try:
        r=ControlResources((WorkerSpec(0,WorkerRole.DRAFT,draft_banked=True),WorkerSpec(1,WorkerRole.TARGET)),
            slots=2,host_blocks=8,block_bytes=4096,payload_capacity=65536,
            draft_host_numa_policy='bind',draft_host_numa_nodes=(nodes[0],),
            target_host_numa_policy='interleave',target_host_numa_nodes=nodes)
    except PermissionError:pytest.skip('mbind denied')
    try:
        assert r.draft_host.numa_placement['policy']=='bind'
        assert r.host.numa_placement['policy']=='interleave'
        assert r.draft_host.descriptor.total_blocks==r.host.descriptor.total_blocks==8
    finally:r.close()
