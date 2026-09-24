"""Cold cohort reclamation after all autonomous owners have joined.

Append-only payloads remain valid until no worker can read them. Preserve the
existing stop/restart policy; no old management ACK channel participates.
"""

from nebulasd.core.ids import REQUEST_EPOCH
from nebulasd.ipc.native import library
from nebulasd.table.storage import TablePartition
from .admission import AdmissionCapacityError, SessionBudget
from .client_idle import quiescent

class CohortRecycler:
    def __init__(self, engine):
        self.engine = engine
        self.next_epoch = None
        self.completed = 0

    @property
    def busy(self):
        return self.next_epoch is not None

    def progress(self):
        e = self.engine
        if not self.busy:
            if not e.registry.records or not quiescent(e):
                return False
            epoch = REQUEST_EPOCH.next(e.registry.epoch)
            self.next_epoch = epoch
            e.supervisor.stop_workers()
            return True
        r = e.resources
        if not quiescent(e):
            raise RuntimeError('retirement barrier lost quiescence')
        for slot, record in e.registry.records.items():
            if any(library().sd_load(pins.address + slot * 64) for pins in (r.pins, r.draft_pins)):
                raise RuntimeError('retirement with live HostKV pin')
            e.registry.allocator.recycle(slot, record.input.epoch, quiescent=True)
            if any(w.draft_banked for w in r.specs):
                e.registry.draft_allocator.recycle(slot, record.input.epoch, quiescent=True)
        r.recycle_payloads()
        # Exclusive ownership after all workers have joined. No worker may
        # publish request rows now; bank/runtime facts and sequences are retained.
        for partition in r.table._partitions.values():
            empty = TablePartition(layout=partition.layout, block_kind=partition.block_kind,
                                   capacity_rows=partition.capacity_rows)
            partition.segment.buffer[:] = empty._bytes
        e.registry.epoch = self.next_epoch
        e.registry.records.clear()
        e.registry.identities.clear()
        e.registry.budget = SessionBudget(r)
        e.candidates.clear()
        if hasattr(e, "scheduling_progress"):
            e.scheduling_progress.pending = {"D": (), "T": ()}
            e.scheduling_progress.changed()
        request_kinds = set(r.table._partitions)
        e.rows = {k: v for k, v in e.rows.items() if k[0] not in request_kinds}
        e.reader.reset_requests(request_kinds)
        from dataclasses import replace
        from .work_ledger import WorkLedger
        r.specs = tuple(replace(w,generation=w.generation+1) for w in r.specs)
        reset_compute_times = getattr(e.scheduler, 'reset_compute_times', None)
        if reset_compute_times is not None:
            reset_compute_times()
        e.ledger = WorkLedger(r)
        e.scheduling_progress.static_rows.clear()
        e.scheduling_progress.set_workers(r.specs)
        e.reader.reset_pending()
        e.supervisor.start()
        self.next_epoch = None
        self.completed += 1
        return True

    def check_admission(self):
        if self.busy:
            raise AdmissionCapacityError('retirement in progress; poll/drain then retry')
