"""Conservative cohort budgets: reject new work before any shared mutation."""
from dataclasses import dataclass
from nebulasd.core.draft_contracts import DraftSnapshot


class AdmissionRejected(ValueError):
    """Rejected without changing request, arena or Worker state."""


class AdmissionCapacityError(AdmissionRejected):
    """Session cannot guarantee completion of another request within capacity."""


@dataclass(frozen=True)
class AdmissionCost:
    engine_tokens: int
    config_bytes: int
    proposals: int
    host_blocks: int
    snapshots: int = 0
    draft_host_blocks: int = 0
    completions: int = 0


class SessionBudget:
    """Worst-case per-worker budgeting permits Target migration.

    Worker output arenas are append-only. Merely checking their current write
    position would let later rounds of an accepted request exhaust the arena.
    Reserve each request's entire worst case against every possible owner. This
    sacrifices capacity for a simple guarantee without placement reservations.
    """
    def __init__(self, resources):
        self.resources = resources
        self.used = AdmissionCost(0,0,0,0)

    def check(self, prompt_count, config, host_blocks):
        from nebulasd.workers.completion import cohort_completion_budget
        rounds, depth = config.max_new_tokens,config.proposal_depth
        cost = AdmissionCost(4*(prompt_count+rounds),16+4*len(config.stop_token_ids),
                             (rounds+1)*(8+4*depth),host_blocks,
                             (rounds+1)*DraftSnapshot.byte_size if any(w.draft_banked for w in self.resources.specs) else 0,
                             host_blocks if any(w.draft_banked for w in self.resources.specs) else 0,
                             cohort_completion_budget([rounds]))
        r, used = self.resources,self.used
        limits = (('engine_tokens',r.token_router.writer.capacity_bytes),
                  ('config_bytes',r.configs.capacity_bytes),
                  ('proposals',min(a.capacity_bytes for a in r.proposals.values())),
                  ('host_blocks',r.host.descriptor.total_blocks),
                  ('snapshots',min(a.capacity_bytes for a in r.snapshots.values())),
                  ('draft_host_blocks',r.draft_host.descriptor.total_blocks))
        if getattr(r,'completions',None) is not None:
            limits += (('completions',r.completions.segment.descriptor.size),)
        for name, capacity in limits:
            if getattr(used,name)+getattr(cost,name)>capacity:
                raise AdmissionCapacityError(f'{name} cohort budget exhausted; drain completed work then retry')
        return cost

    def commit(self, cost):
        self.used = AdmissionCost(*(getattr(self.used,name)+getattr(cost,name)
                                    for name in AdmissionCost.__dataclass_fields__))
