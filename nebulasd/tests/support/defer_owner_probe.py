"""Experiment-only defer owner steps/copy metadata while backend runs.

Already submitted GPU DMA is allowed to finish. Periodic steps resume pending
commands/facts after the backend returns; no readiness check is bypassed.
"""
import asyncio


def install(adapter):
    backend=adapter.worker.backend
    activity=getattr(backend,'_profile_activity',{})
    backend._profile_activity=activity
    activity.update(deferred_steps=0,deferred_polls=0)
    busy=False
    original_run=backend.run_batch
    def run(*args,**kwargs):
        nonlocal busy
        busy=True
        try:return original_run(*args,**kwargs)
        finally:busy=False
    backend.run_batch=run
    original_step=adapter.step
    async def step():
        if busy:
            activity['deferred_steps']+=1
            await asyncio.sleep(0)
            return False
        return await original_step()
    adapter.step=step
    lane=adapter.worker.copy_lane
    original_poll=lane.poll
    def poll():
        if busy:
            activity['deferred_polls']+=1
            return False
        return original_poll()
    lane.poll=poll
    lane._progress._progress=poll
