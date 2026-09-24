"""Symmetric Draft host timings for banked and ordinary session backends.

No CUDA events, synchronization, model re-execution or per-layer hooks. The
backend timer excludes metadata collection; nested timings include hook overhead.
"""
from dataclasses import asdict
from functools import wraps
import os
from pathlib import Path
import resource
import sys
from threading import get_native_id
from time import perf_counter_ns, thread_time_ns


def attach_draft_backend(backend, recorder):
    sessions = getattr(backend, '_adapter', None)
    native = getattr(sessions, '_adapter', None)
    model = getattr(native, 'model', None)
    if sessions is not None:
        for method in ('prefill_batch', 'decode_batch', 'crop_batch', 'preflight_batch'):
            recorder.wrap(sessions, method, 'host.draft.'+method, category='host_operation')
    recorder.wrap(backend, '_preflight_inputs', 'host.draft.inputs', category='host_operation')
    if model is not None:
        recorder.wrap(model, '_forward', 'host.forward', category='host_forward')
        original_forward = model.forward
        @wraps(original_forward)
        def forward(input_ids, rows, decode_lens, *args, **kwargs):
            start, cpu = perf_counter_ns(), thread_time_ns()
            failed = True
            try:
                result = original_forward(input_ids, rows, decode_lens, *args, **kwargs)
                failed = False
                return result
            finally:
                end, cpu_end = perf_counter_ns(), thread_time_ns()
                recorder.record('host.model_forward', start, end, category='host_operation',
                    cpu_ns=cpu_end-cpu, failed=failed, batch_size=len(rows),
                    input_tokens=sum(map(len,input_ids)), decode_lens=list(decode_lens))
        model.forward = forward
    original_run = backend.run_batch
    calls = 0
    @wraps(original_run)
    def run(items):
        nonlocal calls
        keys = [[i.request_slot,i.request_epoch,i.round_id] for i in items]
        token = recorder.keys.set(keys)
        calls += 1
        if calls == 1:
            import torch
            recorder.record('draft.environment',perf_counter_ns(),keys=[],
                pid=os.getpid(),native_thread=get_native_id(),
                native_control_gil_retained=getattr(backend,"_profile_native_gil",False),
                affinity=sorted(os.sched_getaffinity(0)),
                torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
                switch_interval=sys.getswitchinterval(),adapter=type(native).__name__,
                config=asdict(backend.config) if hasattr(backend,'config') else None,
                forward_cuda_timing=getattr(model,'record_forward_timing',None),
                env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')})
        if calls % 32 == 1:
            path=Path('/proc/thread-self/schedstat')
            recorder.record('draft.schedstat',perf_counter_ns(),keys=[],native_thread=get_native_id(),
                counters=[int(x) for x in path.read_text().split()] if path.exists() else None)
        activity = getattr(backend, '_profile_activity', {})
        activity_start = dict(activity)
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        start,cpu = perf_counter_ns(),thread_time_ns()
        failed=True
        try:
            result=original_run(items)
            failed=False
            return result
        finally:
            end,cpu_end=perf_counter_ns(),thread_time_ns()
            after=resource.getrusage(resource.RUSAGE_THREAD)
            recorder.record('backend.run_batch',start,end,category='backend_wall',cpu_ns=cpu_end-cpu,
                failed=failed,activity_delta={k:v-activity_start[k] for k,v in activity.items()},voluntary_switches=after.ru_nvcsw-usage.ru_nvcsw,
                involuntary_switches=after.ru_nivcsw-usage.ru_nivcsw,
                initial=[i.state is None for i in items],delta_lengths=[len(i.token_delta) for i in items],
                proposal_depths=[i.scheduled_token_count for i in items])
            recorder.keys.reset(token)
    backend.run_batch=run


def attach_copy_receipts(raw, recorder):
    lane=raw.worker.copy_lane
    previous=lane.on_receipt
    def receipt(plan,result):
        recorder.record('copy.receipt',result.enqueued_ns,result.retired_ns,
            keys=[[r.extent.request_slot,r.extent.request_epoch,n] for r,n in zip(plan.regions,plan.round_ids)],
            direction=plan.direction)
        if previous is not None:previous(plan,result)
    lane.on_receipt=receipt
