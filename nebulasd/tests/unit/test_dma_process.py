from dataclasses import replace
import pytest
from nebulasd.kv.transfer import CopyPlan, CopyRegion, HostCompletedFence, _plan_chunks
from nebulasd.workers.dma_process import bounded_regions, transport_plan
from nebulasd.kv.arena import SharedHostKVArena


def test_host_fence_transport_rejects_unfinished_device_dependency():
    plan = CopyPlan('D2H', (object(),), (object(),))
    with pytest.raises(ValueError, match='host-completed'):
        transport_plan(plan)
    fenced = replace(plan, dependencies=(HostCompletedFence(),))
    assert transport_plan(fenced).dependencies == ()
    assert fenced.dependencies == (HostCompletedFence(),)


def test_window_splits_oversized_rows_without_losing_blocks_or_rounds():
    arena = SharedHostKVArena.create(total_blocks=32, block_bytes=8,
        dtype='torch.float16', kv_block_shape=(4,))
    try:
        extent = arena.make_extent(request_slot=0, request_epoch=1, host_slot=0,
            host_slot_generation=1, writer_lease_generation=1,
            offset_blocks=0, capacity_blocks=32)
        regions = (CopyRegion(extent, 10, 2, 13), CopyRegion(extent, 25, 20, 0),
                   CopyRegion(extent, 26, 21, 5))
        plan = CopyPlan('H2D', regions, (), (3, 4, 5))
        split = bounded_regions(plan, 8, 64)
        chunks = _plan_chunks(split, 8, 64)
        assert all(c.bytes <= 64 for c in chunks)
        def blocks(p):
            return [(r.gpu_begin_block+i, r.host_begin_block+i, r.extent, round_id)
                    for r, round_id in zip(p.regions, p.round_ids)
                    for i in range(r.block_count)]
        assert blocks(split) == blocks(plan)
        assert sum(r.block_count == 0 for r in split.regions) == 1
        assert bounded_regions(replace(plan, direction='D2H'), 8, 64).regions == regions
        assert bounded_regions(plan, 8, 0) is plan
        with pytest.raises(ValueError, match='one K/V block'):
            bounded_regions(plan, 8, 15)
    finally:
        arena.close()
        arena.unlink()


def _fake_dma(connections, gate, entered):
    from concurrent.futures import ThreadPoolExecutor
    from nebulasd.kv.transfer import CopyReceipt
    def serve(bank):
        while True:
            plan = connections[bank].recv()
            if plan is None:
                return
            if bank == 0:
                entered.set()
                if not gate.wait(10):
                    raise RuntimeError('test gate timed out')
            connections[bank].send(CopyReceipt(1, 2, 3.0))
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(serve, bank) for bank in range(2)]
        for job in jobs:
            job.result()


def test_process_replies_preserve_bank_credit_and_whole_plan_completion():
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor
    from threading import Thread, Event
    from nebulasd.workers.dma_process import ProcessDMA
    ctx = mp.get_context('spawn')
    pairs = [ctx.Pipe() for _ in range(2)]
    gate, entered = ctx.Event(), ctx.Event()
    dma = ProcessDMA.__new__(ProcessDMA)
    dma.connections = [pair[0] for pair in pairs]
    dma._pool = ThreadPoolExecutor(2)
    dma._futures = [None, None]
    dma._failure = None
    dma._closed = False
    dma._caches = (object(), object())
    dma.process = ctx.Process(target=_fake_dma,
        args=([pair[1] for pair in pairs], gate, entered))
    dma.process.start()
    for pair in pairs:
        pair[1].close()
    dma._monitor = Thread(target=dma._watch, args=(Event(),), daemon=True)
    dma._monitor.start()
    try:
        plan = CopyPlan('H2D', (1,))
        held = dma.submit(0, plan)
        assert entered.wait(5)
        assert not held.done()
        with pytest.raises(RuntimeError, match='outstanding'):
            dma.submit(0, plan)
        # Other bank can finish while bank 0 is still physically pending.
        assert dma.submit(1, replace(plan, direction='D2H')).result(5).duration_ms == 3
        assert not held.done()
        gate.set()
        assert held.result(5).completed_ns == 2
        assert dma.submit(0, plan).result(5).completed_ns == 2
    finally:
        gate.set()
        dma.close()
    assert dma._caches is None
    with pytest.raises(RuntimeError, match='closed'):
        dma.submit(0, plan)
