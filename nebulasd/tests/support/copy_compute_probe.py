"""Experiment-only serialization of complete DMA and Draft backend calls.

The copy executor already waits for its completion event. Holding one local lock
across that existing operation removes copy/compute overlap without adding CUDA
synchronization or bypassing readiness. Gate waits remain explicitly accounted.
"""
from threading import Lock
from time import perf_counter_ns


def install(adapter):
    lock=Lock()
    backend=adapter.worker.backend
    activity=getattr(backend,'_profile_activity',{})
    backend._profile_activity=activity
    activity.update(compute_gate_wait_ns=0,copy_gate_wait_ns=0)
    def guarded(obj,method,counter):
        original=getattr(obj,method)
        def call(*args,**kwargs):
            start=perf_counter_ns()
            with lock:
                activity[counter]+=perf_counter_ns()-start
                return original(*args,**kwargs)
        setattr(obj,method,call)
    guarded(backend,'run_batch','compute_gate_wait_ns')
    guarded(adapter.worker.copy_lane.executor,'_run','copy_gate_wait_ns')
