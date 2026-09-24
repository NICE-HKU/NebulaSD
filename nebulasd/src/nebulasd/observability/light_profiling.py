"""Host-only scheduling evidence, with no candidate scans or cost predictions.

States describe observed gates, not exclusive root causes. Full published fields
retain versions/fences for offline inspection. Only changed worker snapshots emit.
"""
from time import perf_counter_ns
from nebulasd.core.enums import StateChangeBlockKind as K, WorkerRole
from nebulasd.scheduler.views import value


def attach_scheduler(scheduler, recorder):
    original = scheduler.schedule
    previous = {}

    def schedule(view, **kwargs):
        if kwargs.get("phase") not in (None, "ready"):
            return original(view, **kwargs)  # Planning reservations are not issued facts.
        # Capture the input view; the command list is a proposal, not an issue.
        now = perf_counter_ns()
        for worker in view.workers:
            wid = worker.worker_id
            draft = worker.role == WorkerRole.DRAFT
            runtime = view.row(K.WORKER_DRAFT_RUNTIME if draft else K.WORKER_TARGET_COMPUTE_RUNTIME, wid)
            prepared = view.prepared.get(wid)
            inflight = view.inflight.get(wid)
            banks = tuple(tuple(value(view.row(K.WORKER_DRAFT_BANK if draft else K.WORKER_BANK, wid*2+i), f)
                          for f in ('state', 'bank_epoch', 'batch_seq', 'role', 'alloc_rows')) for i in range(2))
            state = (getattr(prepared, 'command_seq', None), getattr(inflight, 'command_seq', None),
                     value(runtime, 'compute_status'), banks)
            if state != previous.get(wid):
                recorder.record('scheduler.worker_state', now, keys=[], worker=wid,
                    stage='D' if draft else 'T', prepared_seq=state[0], inflight_seq=state[1],
                    compute_status=state[2], banks=banks)
                previous[wid] = state
        return original(view, **kwargs)
    scheduler.schedule = schedule


def attach_forward(raw, recorder):
    """Time existing host calls only; asynchronous GPU work is not synchronized."""
    backend = raw.worker.backend if getattr(raw, 'draft_banked', False) else raw.worker._backend
    if raw.target:
        facade = getattr(backend, 'bank_facade', None)
        model = getattr(getattr(facade, 'worker', None), 'model', None)
    else:
        sessions = getattr(backend, '_adapter', None)
        model = getattr(getattr(sessions, '_adapter', None), 'model', None)
        if sessions is not None:
            for name in ('prefill_batch', 'decode_batch', 'crop_batch', 'preflight_batch'):
                recorder.wrap(sessions, name, 'host.draft.'+name, category='host_operation')
    if model is not None:
        recorder.wrap(model, '_forward', 'host.forward', category='host_forward')
