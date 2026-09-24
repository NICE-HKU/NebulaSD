"""Opt-in host handoff trace; buffered until process teardown, no hot-path I/O.

Row publication is bracketed: the release commit occurs within [begin, end].
Identity is (worker, generation, WORK seq, member index); WORK supplies epoch/round.
"""
import json
import os
from pathlib import Path
from time import perf_counter_ns, process_time_ns

class HandoffTrace:
    def __init__(self, options, stack, role):
        directory = os.environ.get('STARSD_HANDOFF_TRACE')
        self.enabled = bool(directory)
        self.events = []
        self.worker = options.get('worker_id')
        self.generation = options.get('worker_generation')
        if directory:
            self.path = Path(directory) / f'{self.worker}-{role}-{os.getpid()}.json'
            self.start = (perf_counter_ns(), process_time_ns())
            stack.callback(self.close)
    def mark(self, kind, seq, index=-1, begin=0):
        if self.enabled:
            self.events.append((kind, seq, index, begin, perf_counter_ns()))
    def close(self):
        end = (perf_counter_ns(), process_time_ns())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(worker=self.worker, generation=self.generation,
            pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)), start=self.start,
            end=end, events=self.events)))
