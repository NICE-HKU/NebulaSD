"""Ready tags are joined with live identity/dispatch/worker/Bank fences."""

from nebulasd.core.enums import (Lifecycle, TargetStatus, DraftStatus, H2DStatus, D2HStatus,
                                    WorkerStatus, StateChangeBlockKind as K)
from .views import value as v


def online(view, worker):
    row = view.row(K.WORKER_COMMON, worker.worker_id)
    return (v(row, "status") == WorkerStatus.ONLINE and v(row, "worker_id") == worker.worker_id
            and v(row, "worker_generation") == worker.generation)


def active(view, request):
    if view.request_states is not None:
        state = view.request_states.get(request.slot)
        return state is not None and state.input.epoch == request.epoch and state.lifecycle == Lifecycle.ACTIVE
    row = view.row(K.REQUEST_ENGINE, request.slot)
    return v(row, "request_epoch") == request.epoch and v(row, "lifecycle") == Lifecycle.ACTIVE


def target_result(view, request):
    row, dispatch = view.row(K.REQUEST_TARGET_COMPUTE, request.slot), view.row(K.REQUEST_DISPATCH, request.slot)
    if not active(view, request) or row is None or dispatch is None:
        return None
    worker = next((w for w in view.workers if w.worker_id == v(row, "target_id")), None)
    if (worker is None or not online(view, worker) or v(row, "result_code") != 0
            or v(row, "status") != TargetStatus.READY_DRAFT
            or v(row, "request_epoch") != request.epoch
            or v(row, "target_generation") != worker.generation
            or v(row, "round_id") != v(dispatch, "target_round_id")
            or v(row, "observed_run_seq") != v(dispatch, "target_run_seq")):
        return None
    # Engine must have consumed/classified these tokens before scheduling them.
    round_id = (view.request_states[request.slot].current_round if view.request_states is not None
                else v(view.row(K.REQUEST_ENGINE, request.slot), "current_round_id"))
    return row if request.output_count and round_id == v(row, "round_id") else None


def draft_ready(view, request, round_id):
    row, dispatch = view.row(K.REQUEST_DRAFT, request.slot), view.row(K.REQUEST_DISPATCH, request.slot)
    worker = next((w for w in view.workers if w.worker_id == v(row, "worker_id")), None)
    if worker is None or not online(view, worker) or not active(view, request):
        return False
    return (v(row, "request_epoch") == request.epoch and v(row, "round_id") == round_id
            and v(row, "status") == DraftStatus.READY_TARGET and v(row, "result_code") == 0
            and v(row, "worker_generation") == worker.generation
            and v(dispatch, "draft_worker_id") == worker.worker_id
            and v(dispatch, "draft_worker_generation") == worker.generation
            and v(dispatch, "draft_round_id") == round_id
            and v(dispatch, "draft_issue_seq") == v(row, "observed_issue_seq"))


def restored(view, request, prepare, item):
    row, dispatch = view.row(K.REQUEST_H2D, request.slot), view.row(K.REQUEST_DISPATCH, request.slot)
    host, d2h = view.row(K.REQUEST_HOSTKV, request.slot), view.row(K.REQUEST_D2H, request.slot)
    return (item.request_epoch == request.epoch
            and v(row, "request_epoch") == request.epoch and v(row, "round_id") == item.round_id
            and v(row, "observed_prepare_seq") == item.op_seq
            and v(dispatch, "target_prepare_seq") == item.op_seq
            and v(dispatch, "planned_target_id") == prepare.worker_id
            and v(dispatch, "planned_target_generation") == prepare.target_generation
            and v(dispatch, "planned_bank_id") == prepare.standby_bank_id
            and v(dispatch, "planned_bank_epoch") == prepare.next_bank_epoch
            and v(row, "target_id") == prepare.worker_id
            and v(row, "target_generation") == prepare.target_generation
            and v(row, "destination_bank_id") == prepare.standby_bank_id
            and v(row, "destination_bank_epoch") == prepare.next_bank_epoch
            and v(row, "status") == H2DStatus.GPU_READY and v(row, "result_code") == 0
            and v(row, "source_host_version") == v(row, "gpu_ready_version") == item.source_host_version
            and v(row, "copied_blocks") == item.valid_blocks
            and v(host, "request_epoch") == v(d2h, "request_epoch") == request.epoch
            and v(host, "host_slot_generation") == v(d2h, "host_slot_generation") == item.host_slot_generation
            and v(host, "writer_lease_generation") == v(d2h, "writer_version") == item.host_writer_lease_generation
            and v(d2h, "status") == D2HStatus.HOST_READY and v(d2h, "result_code") == 0
            and v(d2h, "ready_version") == item.source_host_version)


def draft_issued(view, request):
    """Real dispatch fact, not a command merely proposed in this schedule call."""
    return _draft_issued(view, request, target_result(view, request))


def _draft_issued(view, request, result):
    dispatch = view.row(K.REQUEST_DISPATCH, request.slot)
    worker = next((w for w in view.workers if w.worker_id == v(dispatch, 'draft_worker_id')), None)
    return (result is not None and worker is not None and online(view, worker)
            and v(dispatch, 'draft_worker_generation') == worker.generation
            and bool(v(dispatch, 'draft_issue_seq', 0))
            and v(dispatch, 'draft_round_id') == v(result, 'round_id') + 1)


class ScheduleEligibility:
    """Lazy Target joins owned by one synchronous schedule invocation.

    The owner does not merge new rows or dispatch commands during schedule().
    RequestInput and BlockSnapshot values stay fixed for that invocation; the
    cache is discarded before dispatch and never carried into the next view.
    Both successful and missing results are reusable under that same boundary.
    """

    def __init__(self, view):
        self.view = view
        self._target_results = {}

    def target_result(self, request):
        key = request.slot, request.epoch
        if key not in self._target_results:
            self._target_results[key] = target_result(self.view, request)
        return self._target_results[key]

    def draft_issued(self, request):
        return _draft_issued(self.view, request, self.target_result(request))
