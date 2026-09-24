import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS
from support.defer_owner_probe import install


def test_defers_without_dropping_pending_progress():
    entered,release=Event(),Event();calls=[]
    def run():
        entered.set()
        assert release.wait(3)
        return 8
    async def step():calls.append('step');return True
    lane=NS(poll=lambda:calls.append('poll') or True,_progress=NS())
    adapter=NS(worker=NS(backend=NS(run_batch=run),copy_lane=lane),step=step)
    install(adapter)
    with ThreadPoolExecutor(1) as pool:
        task=pool.submit(adapter.worker.backend.run_batch)
        assert entered.wait(2)
        try:
            assert not asyncio.run(adapter.step())
            assert not lane._progress._progress()
            assert calls==[]
        finally:release.set()
        assert task.result()==8
    assert asyncio.run(adapter.step()) and lane.poll()
    assert calls==['step','poll']
