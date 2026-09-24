"""Draft-owned fixed dual Bank and copy runtime partitions."""
from nebulasd.core.enums import (StateChangeBlockKind as K, WorkerRole, CopyStatus,
                                    BankRole, BankState, validate_enum)
from nebulasd.core.ids import U64, U32
from nebulasd.ipc.draft_protocol import bank_id as check_bank
from .draft_fences import expect, read, publish


class DraftRegistryWriter:
    def __init__(self, registry): self.registry = registry

    def _validate(self, row, generation):
        expect(read(self.registry, K.WORKER_COMMON, row), role=int(WorkerRole.DRAFT), worker_generation=generation)

    def publish_copy_runtime(self, *, worker_row, worker_generation, copy_op_seq, copy_status,
                             copy_start_time_ns=0, copy_bytes=0):
        self._validate(worker_row, worker_generation)
        publish(self.registry, K.WORKER_DRAFT_COPY_RUNTIME, worker_row, dict(
            copy_op_seq=copy_op_seq, copy_status=int(validate_enum(CopyStatus, copy_status)),
            copy_start_time_ns=copy_start_time_ns, copy_bytes=copy_bytes))

    def publish_bank(self, *, worker_row, worker_generation, bank_id, bank_epoch, batch_seq,
                     role, state, capacity_blocks, alloc_ptr_blocks, capacity_rows, alloc_rows):
        self._validate(worker_row, worker_generation)
        check_bank(bank_id)
        U64.validate(bank_epoch)
        U64.validate(batch_seq)
        for count in (capacity_blocks, alloc_ptr_blocks, capacity_rows, alloc_rows): U32.validate(count)
        if alloc_rows > capacity_rows or alloc_ptr_blocks > capacity_blocks:
            raise ValueError('Draft Bank allocation exceeds capacity')
        publish(self.registry, K.WORKER_DRAFT_BANK, worker_row * self.registry.bank_rows_per_worker + bank_id,
                dict(bank_id=bank_id, bank_epoch=bank_epoch, batch_seq=batch_seq,
                     role=int(validate_enum(BankRole, role)), state=int(validate_enum(BankState, state)),
                     capacity_blocks=capacity_blocks, alloc_ptr_blocks=alloc_ptr_blocks,
                     capacity_rows=capacity_rows, alloc_rows=alloc_rows))
