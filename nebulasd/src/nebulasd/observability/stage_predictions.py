"""Observe existing predictions without extra estimation or changing decisions."""
from .profiling import request_keys


def attach(scheduler, recorder):
    estimator = scheduler._estimator
    if not hasattr(estimator, 'stage_prediction'):
        return
    import os
    turnaround = os.environ.get("STARSD_BANK_TURNAROUND_PROFILE") == "1"
    schedule = scheduler.schedule
    predictions = {}
    from time import perf_counter_ns, thread_time_ns
    costs = {}
    if turnaround and hasattr(scheduler, "_plan"):
        original_plan = scheduler._plan
        def plan(view, stage, *args, **kwargs):
            begin, cpu = perf_counter_ns(), thread_time_ns()
            try:
                return original_plan(view, stage, *args, **kwargs)
            finally:
                recorder.record("turn.plan", begin, perf_counter_ns(), keys=[], stage=stage,
                    cpu_ns=thread_time_ns()-cpu)
        scheduler._plan = plan

    def prediction(original, view, stage, worker, requests, now_ns, **kwargs):
        begin_cpu = thread_time_ns() if turnaround else 0
        result = original(view, stage, worker, requests, now_ns, **kwargs)
        if turnaround:
            row = costs.setdefault(stage, [0, 0])
            row[0] += 1
            row[1] += thread_time_ns()-begin_cpu
        key = (stage, worker.worker_id, tuple(sorted((r.slot, r.epoch) for r in requests)),
               kwargs.get('initial', False))
        predictions[key] = (now_ns, result)
        return result

    class ObservedEstimator:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def stage_prediction(self, *args, **kwargs):
            return prediction(self.original.stage_prediction, *args, **kwargs)

        def for_invocation(self):
            return ObservedEstimator(self.original.for_invocation())

    def observed(view, **kwargs):
        from time import perf_counter_ns, thread_time_ns
        entered, cpu = perf_counter_ns(), thread_time_ns()
        predictions.clear()
        costs.clear()
        original = scheduler._estimator
        scheduler._estimator = ObservedEstimator(original)
        try:
            commands = schedule(view, **kwargs)
            returned, used = perf_counter_ns(), thread_time_ns() - cpu
            if turnaround and commands:
                recorder.record("turn.schedule", entered, returned, keys=[], cpu_ns=used, estimator_costs=dict(costs),
                    commands=[(c.worker_id, c.command_seq, c.kind.name) for c in commands])
            for command in commands:
                kind = command.kind.name
                if kind not in ('PREPARE_DRAFT_BANK', 'PREPARE_TARGET_BANK', 'DRAFT_BATCH', 'TARGET_PREFILL_BATCH'):
                    continue
                stage = 'D' if kind in ('PREPARE_DRAFT_BANK', 'DRAFT_BATCH') else 'T'
                keys = request_keys(command)
                key = (stage, command.worker_id, tuple(sorted((k[0], k[1]) for k in keys)),
                       kind in ('DRAFT_BATCH', 'TARGET_PREFILL_BATCH'))
                saved = predictions.get(key) or getattr(scheduler, "native_predictions", {}).get((command.worker_id, command.command_seq))
                recorder.record('scheduler.stage_prediction', saved[0] if saved else scheduler._clock(),
                    keys=keys, worker=command.worker_id, command_seq=command.command_seq,
                    kind=kind, stage=stage, initial=key[-1], prediction=None if saved is None else saved[1])
            return commands
        finally:
            scheduler._estimator = original
            predictions.clear()

    scheduler.schedule = observed
