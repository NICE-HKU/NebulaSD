"""Worker-scoped CUDA/host timeline; physical time conversion is report-only."""

from contextlib import contextmanager
from pathlib import Path
import json
from threading import Lock
from time import perf_counter_ns


def measured(round_ids, warmup_rounds):
    return bool(round_ids) and min(round_ids) > warmup_rounds


class GPUTimeline:
    def __init__(self, devices, *, warmup_rounds=0, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.warmup_rounds = warmup_rounds
        self.origins, self.calibration_ns = {}, {}
        self.events, self.host_events = [], []
        self.context = {}
        self._lock = Lock()
        for device in devices:
            with self.torch.cuda.device(device):
                # Warm the CUDA event machinery before clock alignment.
                warm = self.torch.cuda.Event(enable_timing=True)
                warm.record()
                warm.synchronize()
                candidates = []
                for _ in range(3):
                    origin = self.torch.cuda.Event(enable_timing=True)
                    before = perf_counter_ns()
                    origin.record()
                    origin.synchronize()
                    after = perf_counter_ns()
                    candidates.append((after - before, origin, (before + after) // 2))
                width, origin, midpoint = min(candidates, key=lambda item: item[0])
                self.origins[device] = (origin, midpoint)
                self.calibration_ns[device] = width

    def set_context(self, owner, kind, slots, round_id, *, round_ids=None):
        rounds = [round_id] * len(slots) if round_ids is None else list(round_ids)
        if len(rounds) != len(slots):
            raise ValueError("timeline slots/rounds must have identical lengths")
        with self._lock:
            self.context[owner] = dict(kind=kind, slots=list(slots),
                                       round_ids=rounds, owner=owner)

    def instrument(self, model, device, owner):
        original = model._forward

        def forward(*args, **kwargs):
            # Snapshot at entry. Another worker on this GPU cannot relabel us.
            with self._lock:
                context = dict(self.context[owner])
            start = self.torch.cuda.Event(enable_timing=True)
            done = self.torch.cuda.Event(enable_timing=True)
            host_start = perf_counter_ns()
            stream = self.torch.cuda.current_stream(device)
            start.record(stream)
            output = original(*args, **kwargs)
            done.record(stream)
            with self._lock:
                self.events.append((dict(context, device=device), start, done))
                self.host_events.append(dict(context, kind="forward_submit", device=device,
                                             start_ns=host_start, end_ns=perf_counter_ns()))
            return output
        model._forward = forward

    @contextmanager
    def span(self, kind, owner, device, slots, round_id):
        start = perf_counter_ns()
        try:
            yield
        finally:
            self._host(kind, owner, device, slots, round_id, start, perf_counter_ns())

    def mark(self, kind, owner, device, slots, round_id):
        now = perf_counter_ns()
        self._host(kind, owner, device, slots, round_id, now, now)

    def _host(self, kind, owner, device, slots, round_id, start, end):
        with self._lock:
            self.host_events.append(dict(kind=kind, owner=owner, device=device, slots=list(slots),
                round_ids=[round_id] * len(slots), start_ns=start, end_ns=end))

    def add_copy(self, plan, ticket, device):
        with self._lock:
            self.events.append((dict(kind=plan.direction, slots=[r.extent.request_slot for r in plan.regions],
                round_ids=list(plan.round_ids), device=device, owner=f"target:{device}", copy_id=id(plan)),
                ticket.start, ticket.done))

    def intervals(self):
        rows = []
        for context, start, done in self.events:
            done.synchronize()  # Report generation only; never the hot path.
            origin, host_ns = self.origins[context["device"]]
            rows.append(dict(context, start_ns=host_ns + int(origin.elapsed_time(start) * 1e6),
                             end_ns=host_ns + int(origin.elapsed_time(done) * 1e6)))
        return rows

    def write(self, directory, metadata, *, targets=()):
        from nebulasd.observability.copy_timing import summarize_intervals
        from support.copy_measurements import summarize_copy_receipts, receipt_trace
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        intervals = self.intervals()
        selected = [r for r in intervals if measured(r["round_ids"], self.warmup_rounds)]
        copies = summarize_copy_receipts(targets, intervals, self.calibration_ns, self.warmup_rounds)
        report = dict(metadata, measurement_schema=3, clock_alignment_uncertainty_ns=self.calibration_ns,
                      warmup_rule="all request round_ids > warmup_rounds; prefill excluded",
                      **summarize_intervals(selected, self.calibration_ns), copy_receipts=copies,
                      intervals=intervals, host_intervals=self.host_events)
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        trace = []
        for is_gpu, r in [(True, r) for r in intervals] + [(False, r) for r in self.host_events]:
            trace.append(dict(name=r["kind"], cat="CUDA" if is_gpu else "CPU", ph="X",
                pid=f"gpu:{r['device']}" if is_gpu else "host", tid=r["owner"] + ":" + r["kind"],
                ts=r["start_ns"] / 1000, dur=(r["end_ns"] - r["start_ns"]) / 1000,
                args=dict(requests=r["slots"], rounds=r["round_ids"],
                          measured=measured(r["round_ids"], self.warmup_rounds))))
        trace.extend(receipt_trace(copies["rows"]))
        (directory / "trace.json").write_text(json.dumps({"traceEvents": trace}))
        return report
