from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS
import pytest
from support.copy_compute_probe import install


def test_serializes_copy_completion_and_compute_and_preserves_errors():
    entered,release,compute_started=Event(),Event(),Event()
    def copy():
        entered.set()
        assert release.wait(3)
        return 'retired'
    def compute():
        compute_started.set()
        return 'output'
    backend=NS(run_batch=compute)
    executor=NS(_run=copy)
    install(NS(worker=NS(backend=backend,copy_lane=NS(executor=executor))))
    with ThreadPoolExecutor(2) as pool:
        dma=pool.submit(executor._run)
        assert entered.wait(2)
        model=pool.submit(backend.run_batch)
        try:assert not compute_started.wait(.02)
        finally:release.set()
        assert dma.result()=='retired' and model.result()=='output'
    assert backend._profile_activity['compute_gate_wait_ns']>0
    def fail():raise ValueError('retained')
    broken=NS(run_batch=fail)
    lane=NS(_run=lambda:5)
    install(NS(worker=NS(backend=broken,copy_lane=NS(executor=lane))))
    with pytest.raises(ValueError,match='retained'):broken.run_batch()
    assert lane._run()==5
