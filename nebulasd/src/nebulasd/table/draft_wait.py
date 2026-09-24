"""Fixed Draft WAIT fields from an already validated immutable command.

Only encoding is specialized. PreparedPublication retains sequence fencing and
normal row seqlock/ring stores. Offsets/types are compiled from the existing ABI.
"""
from struct import Struct

from nebulasd.core.draft_contracts import DraftHostAllocation
from nebulasd.core.enums import H2DStatus
from nebulasd.core.ids import OP_SEQ, U64
from .prepared import PreparedPublication

_ALLOCATION = tuple(DraftHostAllocation.__dataclass_fields__)
_FIELDS = ('request_epoch', 'snapshot_round_id', 'snapshot_version', 'snapshot_handle',
    'logical_kv_len', 'next_round_id', 'observed_prepare_seq', 'destination_worker_id',
    'destination_worker_generation', 'next_owner_epoch', 'destination_bank_id',
    'destination_bank_epoch', 'prepared_batch_seq', *_ALLOCATION,
    'status', 'result_code', 'local_row', 'gpu_ready_version', 'copied_blocks',
    'copy_start_time_ns', 'copy_bytes')


class DraftWaitEncoder:
    def __init__(self, partition):
        self.partition = partition
        fields = tuple(partition._field_by_name[n] for n in _FIELDS)
        codes = { (1, False): 'B', (4, False): 'I', (8, False): 'Q', (4, True): 'i' }
        self.packer = Struct('<' + ''.join('QII' if f.type.name == 'arena_handle'
            else codes[f.type.size, f.type.signed] for f in fields))
        cursor, slices = 0, []
        for field in fields:
            slices.append((field.name, cursor, cursor + field.type.size))
            cursor += field.type.size
        self.slices = tuple(slices)
        self.native = None
        if hasattr(partition, '_pack_fields'):
            offsets, lengths, _, count = partition._pack_fields(tuple(
                (partition._offset_by_name[f.name], bytes(f.type.size)) for f in fields))
            self.native = offsets, lengths, count

    def capture(self, command, sequences):
        rows = []
        for r in command.requests:
            i, h = r.source, r.snapshot_handle
            raw = self.packer.pack(i.request_epoch, i.round_id, i.snapshot_version,
                h.offset, h.length, h.generation, i.logical_kv_len, r.next_round_id,
                r.prepare_seq, command.worker_id, command.worker_generation,
                r.next_owner_epoch, command.standby_bank_id, command.next_bank_epoch,
                command.batch_seq, *(getattr(i.allocation, n) for n in _ALLOCATION),
                int(H2DStatus.WAIT_HOST), 0, 0, 0, 0, 0, 0)
            if self.native is None:
                encoded = tuple((n, raw[a:b]) for n, a, b in self.slices)
            else:
                offsets, lengths, count = self.native
                encoded = offsets, lengths, raw, count
            old = sequences[r.request_slot]
            rows.append((r.request_slot, old, 0 if old == U64.invalid else OP_SEQ.next(old), encoded))
        return PreparedPublication(self.partition, tuple(rows))
