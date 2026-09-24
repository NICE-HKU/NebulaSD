"""Opt-in diagnostic spans; preserve notification, scheduling and publication behavior."""
import os
from time import perf_counter_ns, thread_time_ns
from .profiling import ProfileRecorder


def span(recorder, obj, method, name, identity=None):
    original = getattr(obj, method)
    def call(*args, **kwargs):
        start, cpu = perf_counter_ns(), thread_time_ns()
        fields = identity(args) if identity else {}
        failed = True
        try:
            result = original(*args, **kwargs)
            failed = False
            return result
        finally:
            recorder.record(name, start, perf_counter_ns(), keys=[],
                            cpu_ns=thread_time_ns()-cpu, failed=failed, **fields)
    setattr(obj, method, call)


def work_identity(args):
    w = args[0]
    return dict(worker=w.worker_id, seq=getattr(w, 'work_seq', getattr(w, 'command_seq', None)))


def attach_engine(engine, recorder):
    for obj, method, name in (
        (engine, 'step', 'detail.step'),
        (engine.supervisor, 'check', 'detail.check'),
        (engine.supervisor.bell, 'drain', 'detail.bell_drain'),
        (engine.supervisor.bell, 'wait', 'detail.bell_wait'),
        (engine, '_observe_facts', 'detail.observe_facts'),
        (engine.reader, 'poll', 'detail.table_poll'),
        (engine, 'apply_facts', 'detail.apply_facts'),
        (engine.ledger, 'refresh', 'detail.refresh'),
        (engine.recycler, 'progress', 'detail.recycle')):
        span(recorder, obj, method, name)
    for obj, method, name in (
        (engine.scheduling_progress, 'build', 'detail.build'),
        (engine.scheduling_progress, 'publish_dispatch', 'detail.publish'),
        (engine.ledger, 'sent', 'detail.ledger_sent')):
        span(recorder, obj, method, name, work_identity)
    original_start = engine.supervisor.start
    def start_workers():
        result = original_start()
        for pair in engine.supervisor.pairs.values():
            span(recorder, pair, 'submit', 'detail.submit', work_identity)
            if getattr(pair, 'direct_imports', False):
                span(recorder, pair, 'submit_import', 'detail.dma_send', work_identity)
        return result
    engine.supervisor.start = start_workers


def attach_control(options, publisher, event, stack):
    recorder = ProfileRecorder(os.environ['STARSD_ENGINE_DETAIL_DIR'],
                               f"control-{options['worker_id']}-{os.getpid()}", 1000000)
    stack.callback(recorder.flush)
    active = [None]
    original = publisher._advance
    def advance(record):
        active[0] = record['work'].work_seq
        before = record['result']
        start = perf_counter_ns()
        try:
            result = original(record)
            if record['result'] != before and record['result'] == len(record['work'].rows):
                recorder.record('detail.result_rows_ready', start, perf_counter_ns(), keys=[],
                                seq=active[0], worker=options['worker_id'])
            return result
        finally:
            active[0] = None
    publisher._advance = advance
    bells = list(publisher.doorbells)
    if event is not None and event.doorbell is not None:
        bells.append(event.doorbell)
    for bell in {id(b): b for b in bells}.values():
        span(recorder, bell, 'ring', 'detail.worker_bell',
             lambda a: dict(worker=options['worker_id'], seq=active[0]))
