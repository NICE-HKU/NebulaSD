"""Offline copy timing: observation, physical estimate and owner work stay separate."""

from dataclasses import asdict
from nebulasd.observability.copy_timing import percentiles
from support.gpu_timeline import measured


def summarize_copy_receipts(targets, intervals, calibration_ns, warmup_rounds):
    gpu_copies = {r["copy_id"]: r for r in intervals if "copy_id" in r}
    rows = []
    for target in targets:
        block_bytes = target.world.arena.descriptor.block_bytes
        for plan, receipt in target.receipts:
            stamps = asdict(receipt)
            row = dict(copy_id=id(plan), direction=plan.direction,
                requests=[r.extent.request_slot for r in plan.regions], rounds=list(plan.round_ids),
                copy_bytes=sum(r.block_count for r in plan.regions) * 2 * block_bytes,
                measured=measured(plan.round_ids, warmup_rounds), timestamps=stamps,
                cuda_dma_ms=receipt.duration_ms)
            pairs = {
                "executor_queue": ("enqueued_ns", "submitted_ns"),
                "submit_to_last_launch_return": ("submitted_ns", "launch_returned_ns"),
                "launch_call": ("submitted_ns", "launch_returned_ns"),
                "launch_return_to_observed": ("launch_returned_ns", "completed_ns"),
                "completion_detection_window": ("last_pending_ns", "completed_ns"),
                "observed_to_owner": ("completed_ns", "ready_publish_started_ns"),
                "owner_to_ready_fact": ("ready_publish_started_ns", "ready_fact_published_ns"),
                "observed_to_ready_fact": ("completed_ns", "ready_fact_published_ns"),
                "post_ready_cleanup": ("ready_fact_published_ns", "retired_ns"),
                "enqueue_to_retired": ("enqueued_ns", "retired_ns"),
            }
            for name, (begin, end) in pairs.items():
                row[name + "_ms"] = ((stamps[end] - stamps[begin]) / 1e6
                                     if stamps[begin] and stamps[end] else None)
            gpu = gpu_copies.get(id(plan))
            if gpu is not None:
                error = calibration_ns[gpu["device"]]
                row.update(device=gpu["device"], clock_uncertainty_ns=error,
                    estimated_gpu_completed_ns=gpu["end_ns"],
                    gpu_end_to_observed_estimate_ms=(receipt.completed_ns - gpu["end_ns"]) / 1e6,
                    gpu_end_to_ready_estimate_ms=(receipt.ready_fact_published_ns - gpu["end_ns"]) / 1e6,
                    clock_estimate_consistent=(receipt.last_pending_ns - error <= gpu["end_ns"]
                                               <= receipt.completed_ns + error))
                peers = [r for r in intervals if r["kind"] == "target_verify"
                         and r["device"] == gpu["device"] and r["start_ns"] >= receipt.enqueued_ns]
                # The old ambiguous metric excluded all overlapping request
                # sets; it could skip many rounds of the same batch.
                groups = dict(any=peers,
                    same_request=[r for r in peers if set(r["slots"]) & set(row["requests"])],
                    independent=[r for r in peers if not set(r["slots"]) & set(row["requests"])])
                for name, candidates in groups.items():
                    row[f"enqueue_to_next_{name}_verify_gpu_estimate_ms"] = (
                        (min(r["start_ns"] for r in candidates) - receipt.enqueued_ns) / 1e6
                        if candidates else None)
            rows.append(row)
    summary = {}
    for direction in ("D2H", "H2D"):
        selected = [r for r in rows if r["direction"] == direction and r["measured"]]
        fields = sorted({name for r in selected for name in r if name.endswith("_ms")})
        summary[direction] = dict(samples=len(selected), copy_bytes=sorted({r["copy_bytes"] for r in selected}),
            **{name.removesuffix("_ms"): percentiles([r[name] for r in selected if r.get(name) is not None])
               for name in fields})
    return dict(summary=summary, rows=rows)


def receipt_trace(rows):
    events = []
    for row in rows:
        for name, stamp in row["timestamps"].items():
            if not name.endswith("_ns") or not stamp:
                continue
            events.append(dict(name=f"{row['direction']}:{name}", cat="copy_lifecycle", ph="i", s="t",
                pid="host", tid=f"copy:{row.get('device', 'cpu')}", ts=stamp / 1000,
                args=dict(copy_id=row["copy_id"], requests=row["requests"], rounds=row["rounds"],
                          measured=row["measured"])))
    return events
