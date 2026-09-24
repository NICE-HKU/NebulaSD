"""Shared validation for an unsent immutable plan; never trims a batch."""
from nebulasd.core.enums import StateChangeBlockKind as K, BankRole, BankState
from .commands import _dispatch_fact_requests
from .eligibility import active, online, ScheduleEligibility
from .views import value as v


def valid_pending(command, view, *, allow_draining_draft=False):
    worker = next((w for w in view.workers if w.worker_id == command.worker_id), None)
    generation = getattr(command, 'worker_generation', getattr(command, 'target_generation', None))
    identities_valid = (worker is not None and worker.generation == generation and online(view, worker)
            and command.command_seq == view.sequences[worker.worker_id]
            and all((r := view.requests.get(i.request_slot)) is not None
                    and r.epoch == i.request_epoch and active(view, r)
                    for i in _dispatch_fact_requests(command)))

    if not identities_valid:
        return False
    kind = command.kind.name
    if kind.startswith('PREPARE_'):
        draft = kind == 'PREPARE_DRAFT_BANK'
        bank = view.row(K.WORKER_DRAFT_BANK if draft else K.WORKER_BANK,
                        worker.worker_id * 2 + command.standby_bank_id)
        if (worker.worker_id in view.prepared or v(bank, 'role') != BankRole.STANDBY
                or v(bank, 'bank_epoch') + 1 != command.next_bank_epoch
                or v(bank, 'state') not in ((BankState.EMPTY, BankState.DRAINING) if draft and allow_draining_draft else
                                           (BankState.EMPTY,) if draft else
                                            (BankState.EMPTY, BankState.DRAINING))):
            return False
        eligibility = ScheduleEligibility(view)
        if draft:
            from nebulasd.scheduler.draft_placement import source
            return all(source(view, view.requests[i.request_slot]) == i.source
                       and v(view.row(K.REQUEST_DISPATCH, i.request_slot), 'target_round_id') == i.source.round_id
                       for i in command.requests)
        return all(eligibility.draft_issued(view.requests[i.request_slot])
                   and v(view.row(K.REQUEST_DISPATCH, i.request_slot), 'draft_round_id') == i.round_id
                   for i in command.requests)
    # Initial commands have no existing model session to overwrite.
    field = 'draft_issue_seq' if kind == 'DRAFT_BATCH' else 'target_run_seq'
    return all(not v(view.row(K.REQUEST_DISPATCH, i.request_slot), field, 0)
               for i in _dispatch_fact_requests(command))

