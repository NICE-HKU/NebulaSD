"""Bounded Draft initial/prepare/run decisions; fixed-ID placement by default.

Only stable numeric facts enter policy. A snapshot can be placed before Target
finishes, but ownership changes only in the dispatched run after both joins.
"""
from nebulasd.core.enums import BankRole, BankState, ComputeStatus, DraftStatus, H2DStatus, D2HStatus, StateChangeBlockKind as K
from nebulasd.core.ids import U64
from nebulasd.core.draft_contracts import DraftSnapshotIdentity, DraftHostAllocation
from nebulasd.ipc.protocol import DraftBatchCommand, NewRequestData
from nebulasd.ipc.draft_protocol import DraftInitialBank, DraftPrepareRequest, PrepareDraftBankCommand, DraftRunRequest, RunDraftBatchCommand
from nebulasd.core.handles import ArenaHandle
from nebulasd.table.draft_fences import h2d_values, allocation_values, prepare_values
from .eligibility import ScheduleEligibility, active
from .views import value as v


def bank(view, worker, role):
    return next((b for i in (0, 1) if (b := view.row(K.WORKER_DRAFT_BANK, worker.worker_id * 2 + i))
                 and v(b, 'role') == role), None)


def idle(view, worker):
    return v(view.row(K.WORKER_DRAFT_RUNTIME, worker.worker_id), 'compute_status') == ComputeStatus.IDLE


def matches(row, values):
    return row is not None and all(v(row, key) == value for key, value in values.items())


def source(view, request):
    row = view.row(K.REQUEST_DRAFT, request.slot)
    dispatch = view.row(K.REQUEST_DISPATCH, request.slot)
    if not matches(row, dict(request_epoch=request.epoch, status=DraftStatus.READY_TARGET, result_code=0,
        round_id=v(dispatch, 'draft_round_id'), observed_issue_seq=v(dispatch, 'draft_issue_seq'),
        worker_id=v(dispatch, 'draft_worker_id'), worker_generation=v(dispatch, 'draft_worker_generation'),
        owner_epoch=v(dispatch, 'draft_owner_epoch'))):
        return None
    allocation = view.row(K.REQUEST_DRAFT_HOSTKV, request.slot)
    if not allocation or v(allocation, 'request_epoch') != request.epoch:
        return None
    return DraftSnapshotIdentity(request.slot, request.epoch, v(row, 'round_id'), v(row, 'observed_issue_seq'),
        v(row, 'worker_id'), v(row, 'worker_generation'), v(row, 'owner_epoch'), v(row, 'snapshot_version'),
        v(row, 'logical_kv_len'), v(row, 'valid_blocks'),
        DraftHostAllocation(**{name: v(allocation, name) for name in DraftHostAllocation.__dataclass_fields__}))


def scheduled_depth(request):
    return min(request.proposal_depth, request.max_new_tokens - request.output_count)


def fits(view, worker, selected, destination, *, initial=False):
    other = view.row(K.WORKER_DRAFT_BANK, worker.worker_id * 2 + (1 - v(destination, 'bank_id')))
    rows = min(worker.max_batch_size, v(destination, 'capacity_rows', 0) - v(other, 'alloc_rows', 0))
    return (len(selected) <= rows and sum(r.capacity_blocks for r in selected) <= min(worker.bank_blocks, v(destination, 'capacity_blocks', 0))
        and sum(r.prompt_count + r.output_count + scheduled_depth(r) if initial else r.proposal_depth + 1
                for r in selected) <= worker.max_batch_tokens)


def schedule(view, candidates, workers, sequences, *, placement, select_workers=None, planner=None, target_result_for=None, phase=None):
    if target_result_for is None:
        target_result_for = ScheduleEligibility(view).target_result
    workers = sorted((w for w in workers if w.role.name == 'DRAFT' and w.draft_banked), key=lambda w: w.worker_id)
    decisions, issued, used = [], set(), set()
    def emit(command):
        decisions.append(command)
        sequences[command.worker_id] = U64.next(command.command_seq)
        issued.add(command.worker_id)
    if phase in (None, "ready"):
        for worker in workers:
            p = view.prepared.get(worker.worker_id)
            if (not isinstance(p, PrepareDraftBankCommand) or p.worker_generation != worker.generation
                    or worker.worker_id in view.inflight or not idle(view, worker)):
                continue
            b = view.row(K.WORKER_DRAFT_BANK, worker.worker_id * 2 + p.standby_bank_id)
            if not matches(b, dict(state=BankState.READY, bank_epoch=p.next_bank_epoch, batch_seq=p.batch_seq)):
                continue
            rows = []
            for item in p.requests:
                request = view.requests.get(item.request_slot)
                target = None if request is None or not active(view, request) else target_result_for(request)
                h2d = view.row(K.REQUEST_DRAFT_H2D, item.request_slot)
                host = view.row(K.REQUEST_DRAFT_D2H, item.request_slot)
                if (target is None or v(target, 'round_id') != item.source.round_id
                        or source(view, request) != item.source
                        or not matches(view.row(K.REQUEST_DISPATCH, item.request_slot), prepare_values(p, item))
                        or not matches(h2d, dict(**h2d_values(p, item), status=H2DStatus.GPU_READY,
                            result_code=0, gpu_ready_version=item.source.snapshot_version, copied_blocks=item.source.valid_blocks))
                        or not matches(host, dict(request_epoch=item.request_epoch, status=D2HStatus.HOST_READY,
                            ready_version=item.source.snapshot_version, snapshot_handle=item.snapshot_handle,
                            snapshot_version=item.source.snapshot_version, snapshot_round_id=item.source.round_id,
                            source_worker_id=item.source.worker_id, source_worker_generation=item.source.worker_generation,
                            source_op_seq=item.source.op_seq, owner_epoch=item.source.owner_epoch,
                            logical_kv_len=item.source.logical_kv_len, valid_blocks=item.source.valid_blocks, result_code=0,
                            **allocation_values(item.source.allocation)))):
                    break
                rows.append(DraftRunRequest(item.request_slot, item.request_epoch, item.next_round_id,
                    U64.next(item.source.op_seq), item.next_owner_epoch, item.source.snapshot_version,
                    scheduled_depth(request), v(target, 'committed_delta_handle')))
            if len(rows) == len(p.requests):
                emit(RunDraftBatchCommand(worker.worker_id, worker.generation, sequences[worker.worker_id],
                    p.batch_seq, p.standby_bank_id, p.next_bank_epoch, tuple(rows)))

    if phase == "ready":
        return decisions

    initial_groups = None
    if planner is not None:
        destinations = [w for w in workers if w.worker_id not in issued
            and w.worker_id not in view.inflight and w.worker_id not in view.prepared
            and idle(view, w) and v(bank(view, w, BankRole.ACTIVE), 'state') == BankState.EMPTY]
        initial_groups = planner([r for r in candidates
            if not v(view.row(K.REQUEST_DISPATCH, r.slot), 'draft_issue_seq', 0)
            and target_result_for(r) is not None], destinations,
            lambda w, batch: fits(view, w, batch, bank(view, w, BankRole.ACTIVE), initial=True), True, decisions)

    for worker in workers:
        if worker.worker_id in issued or worker.worker_id in view.inflight or worker.worker_id in view.prepared or not idle(view, worker):
            continue
        b = bank(view, worker, BankRole.ACTIVE)
        if v(b, 'state') != BankState.EMPTY:
            continue
        batch = []
        for request in (candidates if initial_groups is None else initial_groups.get(worker.worker_id, [])):
            dispatch = view.row(K.REQUEST_DISPATCH, request.slot)
            if request.slot in used or v(dispatch, 'draft_issue_seq', 0) or target_result_for(request) is None:
                continue
            if fits(view, worker, batch + [request], b, initial=True):
                batch.append(request)
        if batch:
            seq = sequences[worker.worker_id]
            rows = tuple(NewRequestData(r.slot, r.epoch, U64.next(v(target_result_for(r), 'round_id')), 1,
                scheduled_depth(r), r.prompt, r.output, r.config, ArenaHandle.null()) for r in batch)
            emit(DraftBatchCommand(worker.worker_id, worker.generation, seq, rows, (),
                bank=DraftInitialBank(v(b, 'bank_id'), U64.next(v(b, 'bank_epoch')), seq, worker.block_size,
                    tuple(r.capacity_blocks for r in batch))))
            used.update(r.slot for r in batch)

    occupied = {r.request_slot for p in view.prepared.values() if isinstance(p, PrepareDraftBankCommand) for r in p.requests}
    groups, sources = {}, {}
    for request in candidates:
        if request.slot in occupied or request.slot in used:
            continue
        identity = source(view, request)
        if identity is None:
            continue
        # A frozen next Draft batch must not occupy the only prepare slot
        # while Target still waits for another request's proposal. Requiring
        # verification issuance (not its result) breaks that dependency cycle
        # without a graph planner or an extra worker handshake.
        dispatch = view.row(K.REQUEST_DISPATCH, request.slot)
        if v(dispatch, 'target_round_id') != identity.round_id or not v(dispatch, 'target_run_seq', 0):
            continue
        if planner is not None:
            sources[request.slot] = identity
            continue
        choices = workers
        if placement == 'round_robin':
            choices = sorted(workers, key=lambda w: (w.worker_id <= identity.worker_id, w.worker_id))
        if select_workers is not None:
            choices = select_workers(identity, choices)
        for worker in choices:
            if worker.worker_id in issued or worker.worker_id in view.prepared:
                continue
            b = bank(view, worker, BankRole.STANDBY)
            batch = groups.get(worker.worker_id, [])
            if v(b, 'state') == BankState.EMPTY and fits(view, worker, batch + [request], b):
                groups[worker.worker_id] = batch + [request]
                sources[request.slot] = identity
                break
    if planner is not None:
        destinations = [w for w in workers if w.worker_id not in issued and w.worker_id not in view.prepared
            and v(bank(view, w, BankRole.STANDBY), 'state') == BankState.EMPTY]
        groups = planner([r for r in candidates if r.slot in sources], destinations,
            lambda w, batch: fits(view, w, batch, bank(view, w, BankRole.STANDBY)), False, decisions)
    for worker in workers:
        batch = groups.get(worker.worker_id)
        if not batch:
            continue
        b = bank(view, worker, BankRole.STANDBY)
        offset, rows = 0, []
        for request in batch:
            i = sources[request.slot]
            rows.append(DraftPrepareRequest(i, v(view.row(K.REQUEST_DRAFT, request.slot), 'draft_state_handle'),
                U64.next(v(view.row(K.REQUEST_DISPATCH, request.slot), 'draft_prepare_seq', 0)),
                U64.next(i.round_id), U64.next(i.owner_epoch), offset, request.capacity_blocks,
                max(0, request.prompt_count + request.max_new_tokens - 1 - i.logical_kv_len)))
            offset += request.capacity_blocks
        seq = sequences[worker.worker_id]
        emit(PrepareDraftBankCommand(worker.worker_id, worker.generation, seq, seq,
            v(b, 'bank_id'), U64.next(v(b, 'bank_epoch')), tuple(rows)))
    return decisions
