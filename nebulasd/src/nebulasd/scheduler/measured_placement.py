"""Whole-batch finish estimates over existing dispatch and runtime facts."""

from dataclasses import dataclass, replace
from nebulasd.core.enums import (
    ComputeStatus,
    CopyStatus,
    D2HStatus,
    H2DStatus,
    WorkerRole,
    StateChangeBlockKind as K,
)
from .eligibility import draft_ready, target_result, active
from .batching import fit
from .views import value as v


@dataclass(frozen=True)
class MeasuredPlacementEstimator:
    """Unknown cached sync uses two forwards, a documented upper-shape estimate.

    Backend wall is measured separately from queueing. Remaining worker time is
    max(0, predicted whole call - elapsed since worker compute_start_time_ns).
    Dispatch-before-consume has no worker start clock: charge the whole call.
    Mixed first/cached Draft calls conservatively sum two measured calls; this
    overcounts shared generation and is exposed as a model limitation.
    All predictions rank only. Real eligibility and frozen prepare/run are intact.
    """

    table: object
    block_bytes: int
    draft_block_bytes: int | None = None
    draft_copy_table: object = None
    ignore_kv_time: bool = False  # Prediction-only experiment switch; never changes execution gates.

    def for_invocation(self):
        """Private caches for one immutable view; never retain runtime clocks."""
        result = replace(self, table=_InvocationCostTable(self.table),
            draft_copy_table=None if self.draft_copy_table is None else _InvocationCostTable(self.draft_copy_table))
        for name in ('_worker_free', '_copy_free'):
            original = getattr(result, name)
            def memo(view, worker, now, _original=original, _cache=None):
                # The cache belongs to this particular method and invocation.
                key = getattr(worker, 'worker_id', worker), now
                if key not in _cache:
                    _cache[key] = _original(view, worker, now)
                return _cache[key]
            from functools import partial
            object.__setattr__(result, name, partial(memo, _cache={}))
        return result

    @staticmethod
    def _items(command):
        return (*command.new_requests, *command.cached_request_deltas) if hasattr(command, "new_requests") else command.requests

    def _duration(self, stage, requests, *, sync=0):
        if not requests:
            return 0.0
        kv = max(r.prompt_count + r.output_count for r in requests)
        depth = max(min(r.proposal_depth, r.max_new_tokens - r.output_count) for r in requests)
        if stage == "target_prefill":
            kv = max(r.prompt_count for r in requests)
            depth = 0
        elif stage == "target_verify":
            kv = max(0, kv - 1)  # Calibration prefix precedes the anchor.
        return self.table.predict(
            stage,
            batch=len(requests),
            kv=kv,
            depth=max(1, depth) if stage != "target_prefill" else 0,
            sync=sync,
        ).seconds

    def _draft_duration(self, view, requests, first_slots=None):
        first_slots = (
            {
                r.slot
                for r in requests
                if not v(view.row(K.REQUEST_DISPATCH, r.slot), "draft_issue_seq", 0)
            }
            if first_slots is None
            else first_slots
        )
        return self._duration(
            "draft_first", [r for r in requests if r.slot in first_slots]
        ) + self._duration(
            "draft_cached", [r for r in requests if r.slot not in first_slots], sync=2
        )

    def _worker_free(self, view, worker, now_ns):
        if view.work_state is not None:
            return view.work_state.worker_free(self,view,worker,now_ns)
        now = now_ns / 1e9
        command = view.inflight.get(worker.worker_id)
        if command is None:
            return now
        draft = worker.role == WorkerRole.DRAFT
        items = self._items(command)
        requests = [
            view.requests[i.request_slot]
            for i in items
            if i.request_slot in view.requests
            and view.requests[i.request_slot].epoch == i.request_epoch
        ]
        if not requests:
            return now
        duration = (
            self._draft_duration(view, requests, {i.request_slot for i in getattr(command, "new_requests", ())})
            if draft
            else self._duration(
                (
                    "target_prefill"
                    if command.kind.name == "TARGET_PREFILL_BATCH"
                    else "target_verify"
                ),
                requests,
            )
        )
        row = view.row(
            K.WORKER_DRAFT_RUNTIME if draft else K.WORKER_TARGET_COMPUTE_RUNTIME, worker.worker_id
        )
        start = v(row, "compute_start_time_ns", 0)
        expected = (
            getattr(command, "expected_batch_seq", command.command_seq)
            if draft
            else getattr(command, "expected_batch_seq", getattr(command, "batch_seq", None))
        )
        if v(row, "current_batch_seq" if draft else "compute_batch_seq") != expected:
            start = 0
        elif v(row, "compute_status") == ComputeStatus.IDLE:
            return now  # Matching completed call; the sent ledger can lag observation.
        return (
            max(now, start / 1e9 + duration)
            if start and v(row, "compute_status") == ComputeStatus.RUNNING
            else now + duration
        )

    def draft_worker_score(self, view, worker, now_ns):
        return self._worker_free(view, worker, now_ns)

    def worker_ready(self, view, worker, now_ns):
        """Current compute-free clock for either stage; invocation-memoized."""
        return self._worker_free(view, worker, now_ns)

    def _copy(self, direction, byte_count, batch=1, *, draft=False):
        if not byte_count:
            return 0.0
        table = self.draft_copy_table if draft and self.draft_copy_table is not None else self.table
        return table.predict(direction, batch=batch, byte_count=byte_count).seconds

    def _copy_free(self, view, worker_id, now):
        if view.work_state is not None:
            return view.work_state.copy_free(self,view,worker_id,now)
        draft = any(w.worker_id == worker_id and w.role == WorkerRole.DRAFT for w in view.workers)
        row = view.row(K.WORKER_DRAFT_COPY_RUNTIME if draft else K.WORKER_TARGET_COPY_RUNTIME, worker_id)
        status = v(row, "copy_status")
        free = now
        # Runtime has total bytes but no region count. Use one batch upper-byte
        # query; cold table extrapolation method is explicit, never 1ms/token.
        if status in (CopyStatus.H2D, CopyStatus.D2H):
            duration = self._copy(
                "H2D" if status == CopyStatus.H2D else "D2H", v(row, "copy_bytes", 0), draft=draft
            )
            free = max(now, v(row, "copy_start_time_ns", 0) / 1e9 + duration)
        # A frozen queued prepare may be waiting for HostKV. Account for its
        # known service bytes, without inventing its unknown release time. The
        # currently running H2D already covers this sole prepared command.
        prepared = view.prepared.get(worker_id)
        if prepared is not None and status != CopyStatus.H2D:
            pending = [
                i
                for i in prepared.requests
                if not (
                    v(view.row(K.REQUEST_DRAFT_H2D if draft else K.REQUEST_H2D, i.request_slot), "status") == H2DStatus.GPU_READY
                    and v(view.row(K.REQUEST_DRAFT_H2D if draft else K.REQUEST_H2D, i.request_slot), "observed_prepare_seq")
                    == (i.prepare_seq if draft else i.op_seq)
                )
            ]
            if pending:
                free += self._copy(
                    "H2D", sum(i.source.valid_blocks if draft else i.valid_blocks for i in pending)
                    * (self.draft_block_bytes if draft else self.block_bytes) * 2, len(pending), draft=draft
                )
        return free

    def _draft_queue_duration(self, view, worker, requests, inflight_slots):
        """Replay currently eligible FIFO service, with the real Draft fit limits.

        No prediction of future Target completions, service preemption or arrival
        is made. This estimates waiting behind known work, not just execution
        of the candidate subset. Newly selected commands are injected by policy.
        """
        waiting = {r.slot for r in requests if r.slot not in inflight_slots}
        queue = []
        for r in sorted(view.requests.values(), key=lambda r: (r.arrival_seq, r.slot)):
            dispatch = view.row(K.REQUEST_DISPATCH, r.slot)
            result = target_result(view, r)
            issued = v(dispatch, "draft_issue_seq", 0)
            if (
                active(view, r)
                and result is not None
                and r.slot not in inflight_slots
                and (not issued or v(dispatch, "draft_worker_id") == worker.worker_id)
                and (not issued or v(dispatch, "draft_round_id", 0) <= v(result, "round_id"))
            ):
                queue.append(r)
        duration = 0.0
        while waiting and queue:
            batch = fit(
                queue,
                max_rows=worker.max_batch_size,
                max_tokens=worker.max_batch_tokens,
                max_blocks=1 << 60,
                token_cost=lambda r: r.proposal_depth
                + (
                    r.prompt_count + r.output_count
                    if not v(view.row(K.REQUEST_DISPATCH, r.slot), "draft_issue_seq", 0)
                    else 1
                ),
            )
            if not batch:
                break
            duration += self._draft_duration(view, batch)
            slots = {r.slot for r in batch}
            waiting -= slots
            queue = [r for r in queue if r.slot not in slots]
        return duration

    def batch_score(self, view, worker, requests, now_ns):
        return self.batch_prediction(view, worker, requests, now_ns)["finish_s"]

    def batch_prediction(self, view, worker, requests, now_ns, *, input_times=None):
        now = now_ns / 1e9
        draft_done = max(input_times.values(), default=now) if input_times is not None else now
        for dw in (w for w in view.workers if w.role == WorkerRole.DRAFT and input_times is None):
            relevant = [
                r
                for r in requests
                if (
                    not v(view.row(K.REQUEST_DISPATCH, r.slot), "draft_issue_seq", 0)
                    or v(view.row(K.REQUEST_DISPATCH, r.slot), "draft_worker_id") == dw.worker_id
                )
                and not draft_ready(
                    view, r, v(view.row(K.REQUEST_TARGET_COMPUTE, r.slot), "round_id", 0) + 1
                )
            ]
            if not relevant:
                continue
            command = view.inflight.get(dw.worker_id)
            inflight_slots = (
                set()
                if command is None
                else {
                    i.request_slot for i in self._items(command)
                }
            )
            draft_done = max(
                draft_done,
                self._worker_free(view, dw, now_ns)
                + self._draft_queue_duration(view, dw, relevant, inflight_slots),
            )
        # Source dirty D2H workloads serialize per source copy lane, destinations
        # serialize the SUM of prefix bytes. Different lanes overlap via max.
        source_bytes = {}
        h2d_bytes = 0
        for r in requests:
            target = view.row(K.REQUEST_TARGET_COMPUTE, r.slot)
            d2h = view.row(K.REQUEST_D2H, r.slot)
            source = v(target, "target_id")
            h2d_bytes += (
                ((v(target, "logical_kv_len", 0) + worker.block_size - 1) // worker.block_size)
                * self.block_bytes
                * 2
            )
            if v(d2h, "status") != D2HStatus.HOST_READY or v(d2h, "ready_version") != v(
                target, "target_kv_version"
            ):
                # Matching dirty work already on the copy lane is included in
                # its remaining time. Do not charge the same bytes twice.
                copying = (
                    v(d2h, "status") == D2HStatus.IN_D2H
                    and v(d2h, "request_epoch") == r.epoch
                    and v(d2h, "round_id") == v(target, "round_id")
                    and v(d2h, "d2h_op_seq") == v(target, "observed_run_seq")
                    and v(d2h, "source_bank_id") == v(target, "bank_id")
                    and v(d2h, "source_bank_epoch") == v(target, "bank_epoch")
                    and v(view.row(K.WORKER_TARGET_COPY_RUNTIME, source), "copy_status")
                    == CopyStatus.D2H
                )
                source_bytes.setdefault(source, 0)
                if not copying:
                    source_bytes[source] += v(target, "dirty_block_count", 0) * self.block_bytes * 2
        host_ready = max(
            [now]
            + [
                self._copy_free(view, w, now) + self._copy("D2H", n)
                for w, n in source_bytes.items()
            ]
        )
        kv_ready = max(host_ready, self._copy_free(view, worker.worker_id, now)) + self._copy(
            "H2D", h2d_bytes, len(requests)
        )
        free = self._worker_free(view, worker, now_ns)
        verify = self._duration("target_verify", requests)
        start = max(draft_done, kv_ready, free)
        return dict(
            finish_s=start + verify,
            start_s=start,
            draft_ready_s=draft_done,
            kv_ready_s=kv_ready,
            target_free_s=free,
            verify_s=verify,
            h2d_bytes=h2d_bytes,
            source_dirty_bytes=source_bytes,
        )

    def input_ready(self, view, stage, request, now_ns):
        """Only the issued predecessor is predicted; never replay a future queue."""
        now = now_ns / 1e9
        dispatch = view.row(K.REQUEST_DISPATCH, request.slot)
        if stage == "target_prefill":
            return now
        if stage == "D":
            if target_result(view, request) is not None:
                return now
            owner = v(dispatch, "planned_target_id", v(view.row(K.REQUEST_TARGET_COMPUTE, request.slot), "target_id"))
        else:
            if draft_ready(view, request, v(dispatch, "draft_round_id")):
                return now
            owner = v(dispatch, "draft_worker_id")
        worker = next((w for w in view.workers if w.worker_id == owner), None)
        return now if worker is None else self._worker_free(view, worker, now_ns)

    def stage_prediction(self, view, stage, worker, requests, now_ns, *, input_times=None, initial=False):
        """Stage-specific compute and KV models, common four-clock contract."""
        now = now_ns / 1e9
        times = input_times if input_times is not None else {
            r.slot: self.input_ready(view, stage, r, now_ns) for r in requests}
        input_ready = max(times.values(), default=now)
        free = self._worker_free(view, worker, now_ns)
        if stage == "T" and not initial:
            if self.ignore_kv_time:
                kv, compute = now, self._duration("target_verify", requests)
            else:
                p = self.batch_prediction(view, worker, requests, now_ns, input_times=times)
                kv, compute = p["kv_ready_s"], p["verify_s"]
        elif stage == "D":
            if self.draft_block_bytes is None:
                raise ValueError("stagewise Draft requires explicit draft_block_bytes")
            compute = self._draft_duration(view, requests)
            kv = now if initial or self.ignore_kv_time else self._draft_kv_ready(view, worker, requests, now)
        else:
            compute, kv = self._duration("target_prefill", requests), now
        start = max(input_ready, free, kv)
        finish = start + compute
        return dict(input_ready_s=input_ready, worker_ready_s=free, kv_ready_s=kv,
                    compute_s=compute, start_s=start, finish_s=finish,
                    cost_s=sum(finish - times[r.slot] for r in requests))

    def _draft_kv_ready(self, view, worker, requests, now):
        source_bytes, h2d_bytes = {}, 0
        for r in requests:
            source = view.row(K.REQUEST_DRAFT, r.slot)
            host = view.row(K.REQUEST_DRAFT_D2H, r.slot)
            owner = v(source, "worker_id")
            h2d_bytes += v(source, "valid_blocks", 0) * self.draft_block_bytes * 2
            matching = (v(host, "request_epoch") == r.epoch
                        and v(host, "snapshot_version") == v(source, "snapshot_version")
                        and v(host, "source_op_seq") == v(source, "observed_issue_seq")
                        and v(host, "owner_epoch") == v(source, "owner_epoch")
                        and v(host, "source_worker_generation") == v(source, "worker_generation"))
            if matching and v(host, "status") == D2HStatus.HOST_READY and v(host, "ready_version") == v(source, "snapshot_version"):
                continue
            copying = (matching and v(host, "status") == D2HStatus.IN_D2H
                       and v(view.row(K.WORKER_DRAFT_COPY_RUNTIME, owner), "copy_status") == CopyStatus.D2H)
            source_bytes.setdefault(owner, 0)
            if not copying:
                source_bytes[owner] += v(source, "dirty_block_count", v(source, "valid_blocks", 0)) * self.draft_block_bytes * 2
        host_ready = max([now] + [self._copy_free(view, owner, now) + self._copy("D2H", n, draft=True)
                                 for owner, n in source_bytes.items()])
        return max(host_ready, self._copy_free(view, worker.worker_id, now)) + self._copy("H2D", h2d_bytes, len(requests), draft=True)


class _InvocationCostTable:
    def __init__(self, table):
        self.table, self.cache = table, {}

    def predict(self, stage, **shape):
        key = stage, tuple(sorted(shape.items()))
        if key not in self.cache:
            self.cache[key] = self.table.predict(stage, **shape)
        return self.cache[key]
