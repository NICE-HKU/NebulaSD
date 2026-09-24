"""Independent Draft allocations, shared descriptors and admission preflight."""
import os
import pytest
from nebulasd.config import HostKVLayout
from nebulasd.core.enums import WorkerRole, StateChangeBlockKind as K
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from nebulasd.engine.resources import ControlResources
from nebulasd.engine.request_registry import RequestRegistry
from nebulasd.engine.admission import AdmissionCapacityError, AdmissionRejected
from nebulasd.scheduler.views import WorkerSpec

pytestmark = pytest.mark.skipif(not os.environ.get('STARSD_NEXT_NATIVE_LIBRARY'), reason='explicit native build required')


def resources(**kwargs):
    specs = (WorkerSpec(0, WorkerRole.DRAFT, bank_blocks=2, draft_banked=True),
             WorkerSpec(1, WorkerRole.TARGET, bank_blocks=8))
    return ControlResources(specs, slots=4, host_blocks=16, block_bytes=16,
                            draft_layout=HostKVLayout(64, 'uint8', (64,)), **kwargs)


def test_independent_layout_and_pins_survive_attach_and_generation_recycle():
    r = resources()
    child = None
    try:
        child = ControlResources.attach(r.descriptor, 0, None)
        assert child.host.descriptor.block_bytes == 16
        assert child.draft_host.descriptor.block_bytes == 64
        assert child.draft_host.descriptor == r.draft_host.descriptor
        assert child.pins.descriptor != child.draft_pins.descriptor
        r.draft_pins.buffer[0] = 1
        assert child.draft_pins.buffer[0] == 1 and child.pins.buffer[0] == 0
        r.draft_pins.buffer[0] = 0
        routed = child.snapshot_arenas
        old = child.snapshots[0].generation
        # Emulate the established all-owner barrier; no live payloads exist.
        child.recycle_payloads()
        r.recycle_payloads()
        assert child.snapshot_arenas is routed
        assert old not in routed
        assert child.snapshots[0].generation == r.snapshots[0].generation
        assert routed[child.snapshots[0].generation] is child.snapshots[0]
    finally:
        if child is not None:
            child.close()
        r.close()




@pytest.mark.parametrize('limit', ['snapshots', 'bank'])
def test_rejection_is_atomic_before_target_or_draft_allocation(limit):
    r = resources(payload_capacity=1024)
    try:
        registry = RequestRegistry(r)
        config = DraftGenerationConfig(6 if limit == 'snapshots' else 40, 2)
        expected = AdmissionCapacityError if limit == 'snapshots' else AdmissionRejected
        with pytest.raises(expected, match='snapshots' if limit == 'snapshots' else 'Draft'):
            registry.admit('too-large', (1, 2), config)
        assert registry.records == {} and registry.identities == {}
        assert registry.allocator.allocated_count == registry.draft_allocator.allocated_count == 0
        assert r.token_router.writer._head == r.configs._head == 0
        assert all(not getattr(registry.budget.used, name) for name in registry.budget.used.__dataclass_fields__)
        registry.admit('fits', (1, 2), DraftGenerationConfig(4, 2))
        assert registry.allocator.allocated_count == registry.draft_allocator.allocated_count == 1
        allocation = r.table.partition(K.REQUEST_DRAFT_HOSTKV).read_stable(0)
        assert allocation.get('layout_id') == r.draft_layout_id
        assert allocation.get('capacity_blocks') == 1
    finally:
        r.close()
