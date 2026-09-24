"""Selected batch only. The autonomous Engine builds the one real WORK from it.

Legacy Prepare/Run protocol objects are materialized only by differential checks,
not built and immediately discarded on the production scheduler path.
"""
from dataclasses import dataclass
from nebulasd.ipc.command_kinds import CommandKind


@dataclass(frozen=True, slots=True)
class PlanMember:
    request_slot: int
    request_epoch: int
    round_id: int

    @property
    def next_round_id(self):
        return self.round_id


@dataclass(frozen=True, slots=True)
class NativePlan:
    worker_id: int
    worker_generation: int
    command_seq: int
    kind: CommandKind
    bank_id: int
    requests: tuple[PlanMember, ...]

    @property
    def new_requests(self):
        return self.requests

    @property
    def bank(self):
        return self

    @property
    def standby_bank_id(self):
        return self.bank_id

    def materialize(self, scheduler, view, bank):
        from . import builders
        from .draft_placement import source
        worker=next(w for w in view.workers if w.worker_id==self.worker_id)
        requests=[view.requests[r.request_slot] for r in self.requests]
        initial=self.kind in (CommandKind.DRAFT_BATCH,CommandKind.TARGET_PREFILL_BATCH)
        if self.kind in (CommandKind.DRAFT_BATCH,CommandKind.PREPARE_DRAFT_BANK):
            sources={} if initial else {r.slot:source(view,r) for r in requests}
            return scheduler._draft_command(view,worker,self.command_seq,requests,bank,sources,initial)
        return (builders.prefill(worker,self.command_seq,requests,bank) if initial else
                builders.prepare(view,worker,self.command_seq,requests,bank))
