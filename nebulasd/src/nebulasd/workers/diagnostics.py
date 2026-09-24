"""Opt-in DMA completion observation delay for lifecycle acceptance.

Real CUDA must complete too. The extra deadline delays only the selected DMA
thread's completion, never an owner tick or another Bank's submission. Device
elapsed time remains the unmodified CUDA event duration.
"""
from time import monotonic


class DelayedTicket:
    def __init__(self, ticket, delay_s):
        self.ticket = ticket
        self.deadline = monotonic() + delay_s

    def query(self):
        # Always observe the real ticket: no synthetic physical success.
        return self.ticket.query() and monotonic() >= self.deadline

    def duration_ms(self):
        return self.ticket.duration_ms()


class DelayedCopyBackend:
    def __init__(self, backend, direction, delay_s):
        if direction not in ('H2D', 'D2H') or delay_s < 0:
            raise ValueError('invalid DMA diagnostic delay')
        self.backend, self.direction, self.delay_s = backend, direction, delay_s
        self.arena = backend.arena

    def initialize(self):
        self.backend.initialize()

    def launch(self, plan):
        ticket = self.backend.launch(plan)
        return DelayedTicket(ticket, self.delay_s) if plan.direction == self.direction else ticket

    def close(self):
        self.backend.close()


from dataclasses import dataclass
from time import thread_time_ns
from nebulasd.kv.transfer import CopyExecutor,CopyReceipt


@dataclass(frozen=True)
class ProfiledReceipt(CopyReceipt):
    cpu_ns: int = 0


class ProfiledCopyExecutor(CopyExecutor):
    def _run(self, plan, enqueued_ns):
        start=thread_time_ns()
        receipt=super()._run(plan,enqueued_ns)
        cpu=thread_time_ns()-start
        return ProfiledReceipt(**{name:getattr(receipt,name) for name in CopyReceipt.__dataclass_fields__},cpu_ns=cpu)
