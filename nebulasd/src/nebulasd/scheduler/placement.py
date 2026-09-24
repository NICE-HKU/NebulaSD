"""Finish-time ranking uses injected profiling; predictions never authorize execution."""

from dataclasses import dataclass, field
from math import isfinite
from nebulasd.core.enums import ComputeStatus, CopyStatus, D2HStatus, DraftStatus, WorkerRole, StateChangeBlockKind as K
from .estimator import CostModel
from .views import value as v


@dataclass(frozen=True)
class PlacementEstimator:
    cost: CostModel = field(default_factory=CostModel)
    seconds_per_target_token: float = 0.001
    seconds_per_draft_token: float = 0.001
    block_bytes: int = 16

    def __post_init__(self):
        if self.block_bytes <= 0 or any(not isfinite(x) or x < 0 for x in
                (self.seconds_per_target_token, self.seconds_per_draft_token)):
            raise ValueError("invalid placement profiling parameters")

    def draft_worker_score(self, view, worker, now_ns):
        row = view.row(K.WORKER_DRAFT_RUNTIME, worker.worker_id)
        now = now_ns / 1e9
        if v(row, "compute_status") != ComputeStatus.RUNNING:
            return now
        return max(now, v(row, "compute_start_time_ns", 0) / 1e9
                   + v(row, "batch_token_count", 0) * self.seconds_per_draft_token)

    def _copy_free(self, view, worker_id, now):
        row = view.row(K.WORKER_TARGET_COPY_RUNTIME, worker_id)
        status = v(row, "copy_status")
        if status not in (CopyStatus.D2H, CopyStatus.H2D):
            return now
        cost = self.cost.d2h_time if status == CopyStatus.D2H else self.cost.h2d_time
        return max(now, v(row, "copy_start_time_ns", 0) / 1e9 + cost(v(row, "copy_bytes", 0)))

    def score(self, view, worker, request, now_ns):
        now = now_ns / 1e9
        runtime = view.row(K.WORKER_TARGET_COMPUTE_RUNTIME, worker.worker_id)
        free = now
        if v(runtime, "compute_status") == ComputeStatus.RUNNING:
            free = max(now, v(runtime, "compute_start_time_ns", 0) / 1e9
                       + v(runtime, "compute_token_count", 0) * self.seconds_per_target_token)
        draft = view.row(K.REQUEST_DRAFT, request.slot)
        dispatch = view.row(K.REQUEST_DISPATCH, request.slot)
        affinity = v(dispatch, "draft_worker_id") if v(dispatch, "draft_issue_seq", 0) else None
        draft_workers = [w for w in view.workers if w.role == WorkerRole.DRAFT
                         and (affinity is None or w.worker_id == affinity)]
        draft_done = min((self.draft_worker_score(view, w, now_ns) for w in draft_workers), default=now)
        draft_done += request.proposal_depth * self.seconds_per_draft_token
        if (v(draft, "status") == DraftStatus.READY_TARGET
                and v(draft, "round_id") == v(dispatch, "draft_round_id")
                and v(draft, "observed_issue_seq") == v(dispatch, "draft_issue_seq")):
            draft_done = now
        target = view.row(K.REQUEST_TARGET_COMPUTE, request.slot)
        d2h = view.row(K.REQUEST_D2H, request.slot)
        host_ready = now
        if v(d2h, "status") != D2HStatus.HOST_READY or v(d2h, "ready_version") != v(target, "target_kv_version"):
            host_ready = self._copy_free(view, v(target, "target_id"), now) + self.cost.d2h_time(v(target, "dirty_block_count", 0) * self.block_bytes * 2)
        blocks = (v(target, "logical_kv_len", 0) + worker.block_size - 1) // worker.block_size
        kv_ready = self.cost.host_migration_ready_time(host_ready_time=host_ready,
            copy_stream_available_time=self._copy_free(view, worker.worker_id, now), h2d_bytes=blocks * self.block_bytes * 2)
        return self.cost.verify_start_time(draft_done_time=draft_done, target_free_time=free, kv_ready_time=kv_ready)
