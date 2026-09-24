"""Engine-owned identities, lifecycle and accepted output; no Worker resource mirrors."""

from dataclasses import dataclass, field
from time import perf_counter_ns

from nebulasd.core.enums import Lifecycle
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from .admission import AdmissionRejected, AdmissionCapacityError, SessionBudget
from nebulasd.core.handles import ArenaHandle
from nebulasd.kv.host_allocator import HostKVAllocator
from nebulasd.scheduler.views import RequestInput
from nebulasd.table.writers import EngineTableWriter, HostKVAllocatorWriter


@dataclass
class RequestRecord:
    request_id: str
    input: RequestInput
    config: object
    reservation: ArenaHandle
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    current_round: int = -1
    output_chunks: list[tuple[int, ...]] = field(default_factory=list)

    @property
    def output(self):
        return tuple(token for chunk in self.output_chunks for token in chunk)


class RequestRegistry:
    def __init__(self, resources):
        self.resources = resources
        self.records, self.identities = {}, {}
        self.allocator = HostKVAllocator(resources.host)
        self.draft_allocator = HostKVAllocator(resources.draft_host)
        self.writer = EngineTableWriter(resources.table)
        self._publish_seq = 0
        self.epoch = 1
        self._arrival = 0
        self.budget = SessionBudget(resources)

    def admit(self, request_id, prompt, config):
        if request_id in self.identities:
            raise AdmissionRejected("duplicate request id")
        slot = len(self.records)
        if slot >= self.resources.slots:
            raise AdmissionCapacityError("admission slot capacity exhausted; wait for cohort retirement and retry")
        if not isinstance(config, DraftGenerationConfig):
            raise AdmissionRejected("config must be DraftGenerationConfig")
        if not prompt or any(type(t) is not int or not 0 <= t < (1 << 32) - 1 for t in prompt):
            raise AdmissionRejected("prompt must contain valid unsigned token ids")
        block_size = self.resources.specs[0].block_size
        capacity = (len(prompt) + config.max_new_tokens + block_size - 1) // block_size
        targets = [w for w in self.resources.specs if w.role.name == "TARGET"]
        drafts = [w for w in self.resources.specs if w.role.name == "DRAFT"]
        if not any(capacity <= w.bank_blocks and len(prompt) <= w.prefill_max_batch_tokens for w in targets):
            raise AdmissionRejected("request cannot fit any Target")
        if not any(config.proposal_depth + 1 <= w.verify_max_batch_tokens for w in targets):
            raise AdmissionRejected("request verify step cannot fit any Target")
        if not any(len(prompt) + config.proposal_depth + 1 <= w.max_batch_tokens
                   and (not w.draft_banked or capacity <= w.bank_blocks) for w in drafts):
            raise AdmissionRejected("request cannot fit any Draft")
        budget = self.budget.check(len(prompt),config,capacity)
        arena = self.resources.token_router.writer
        prompt_handle = arena.write_tokens(tuple(prompt))
        config_handle = self.resources.configs.write_config(config)
        output = arena.reserve_output(config.max_new_tokens)
        extent = self.allocator.allocate(slot, self.epoch, capacity)
        request = RequestInput(slot, self.epoch, self._arrival, len(prompt), config.max_new_tokens, config.proposal_depth,
            prompt_handle, config_handle, ArenaHandle(output.offset, 0, output.generation), 0,
            extent.arena, capacity, perf_counter_ns())
        record = RequestRecord(request_id, request, config, output)
        self.records[slot], self.identities[request_id] = record, slot
        self.publish(record)
        HostKVAllocatorWriter(self.resources.table).publish_allocation(slot=slot, publish_seq=0,
            request_epoch=self.epoch, host_slot_generation=extent.host_slot_generation,
            writer_lease_generation=extent.writer_lease_generation, host_slot=extent.host_slot,
            capacity_blocks=extent.capacity_blocks, offset_blocks=extent.offset_blocks)
        if any(w.draft_banked for w in drafts):
            from nebulasd.core.draft_contracts import DraftHostAllocation
            from nebulasd.core.ids import RequestFence
            from nebulasd.table.draft_writers import DraftHostKVAllocatorWriter
            d = self.draft_allocator.allocate(slot, self.epoch, capacity)
            host = self.resources.draft_host.descriptor
            allocation = DraftHostAllocation(1, host.descriptor_generation, self.resources.draft_layout_id,
                d.host_slot_generation, d.writer_lease_generation, d.offset_blocks, d.host_slot, capacity, block_size)
            DraftHostKVAllocatorWriter(self.resources.table).publish_allocation(
                request=RequestFence(slot, self.epoch), allocation=allocation)
        self.budget.commit(budget)
        self._arrival += 1
        return slot

    def publish(self, record):
        r = record.input
        self.writer.publish_active(slot=r.slot, publish_seq=self._next(), request_epoch=r.epoch,
            current_round_id=max(record.current_round, 0), arrival_seq=r.arrival_seq,
            classified_result_ticket=record.current_round + 1,
            prompt_token_count=r.prompt_count, max_new_tokens=r.max_new_tokens, spec_token_limit=r.proposal_depth,
            input_tokens_handle=r.prompt, generation_config_handle=r.config, lifecycle=record.lifecycle)

    def cancel(self, request_id):
        record = self.records[self.identities[request_id]]
        if record.lifecycle == Lifecycle.ACTIVE:
            record.lifecycle = Lifecycle.CANCELLED
            self.writer.publish_lifecycle(slot=record.input.slot, publish_seq=self._next(),
                request_epoch=record.input.epoch, lifecycle=record.lifecycle)

    def _next(self):
        result = self._publish_seq
        self._publish_seq += 1
        return result
