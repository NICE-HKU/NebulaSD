"""Opt-in host phase and search timing; never repeats an estimator call."""
from time import perf_counter_ns, thread_time_ns


def attach(engine, recorder):
    scheduler = engine.scheduler
    original = scheduler.schedule
    def schedule(view, **kwargs):
        start, cpu = perf_counter_ns(), thread_time_ns()
        commands = original(view, **kwargs)
        recorder.record('scheduler.phase', start, perf_counter_ns(), keys=[],
            phase=kwargs.get('phase') or 'legacy', cpu_ns=thread_time_ns()-cpu,
            commands=[(c.worker_id, c.command_seq, c.kind.name) for c in commands])
        if getattr(scheduler, 'execution_mode', None) == 'completion' and kwargs.get('phase') != 'ready':
            recorder.record('scheduler.bounded_frontier', start, perf_counter_ns(), keys=[],
                stage=kwargs.get('phase'), destinations=sorted(kwargs.get('destinations', ())),
                **scheduler.last_metrics)
        return commands
    scheduler.schedule = schedule
    original_plan = scheduler._plan
    def plan(view, stage, requests, workers, *args, **kwargs):
        start, cpu = perf_counter_ns(), thread_time_ns()
        result = original_plan(view, stage, requests, workers, *args, **kwargs)
        reason = ('no_candidates' if not requests else 'no_destinations' if not workers
                  else 'selected' if result else 'capacity_or_window')
        recorder.record('scheduler.planning', start, perf_counter_ns(), keys=[],
            stage=stage, initial=kwargs.get('initial', False), candidates=len(requests),
            destinations=len(workers), batches=len(result), reason=reason,
            cpu_ns=thread_time_ns()-cpu)
        return result
    scheduler._plan = plan
    recorder.wrap(engine.scheduling_progress,"advance","engine.work_progress")
