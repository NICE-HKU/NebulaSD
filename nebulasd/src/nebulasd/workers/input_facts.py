"""Captured input facts on the existing control/execution channel, never commands.

Gate fields are checked by Dependencies once. Only fields consumed by the input
compiler cross the channel; routing reuses the accepted WORK and its selectors.
"""
from dataclasses import dataclass
from struct import Struct, error as StructError
from nebulasd.core.handles import ArenaHandle
from .work import Selector, MAX_ROWS

NAMES = ('source', 'predecessor', 'classified')
FIELDS = {
    Selector.TARGET_HOST: ('ready_version', 'logical_kv_len'),
    Selector.DRAFT_HOST: ('ready_version', 'logical_kv_len', 'snapshot_handle',
        'snapshot_round_id', 'snapshot_version', 'source_op_seq', 'source_worker_id',
        'source_worker_generation', 'owner_epoch', 'valid_blocks', 'arena_id',
        'arena_generation', 'layout_id', 'host_slot_generation', 'writer_lease_generation',
        'offset_blocks', 'host_slot', 'capacity_blocks', 'block_size'),
    Selector.PROPOSAL: ('proposal_handle',),
    Selector.DELTA: ('committed_delta_handle', 'accepted_draft_count'),
    Selector.CLASSIFIED: ('lifecycle',),
    Selector.TARGET_DECISION: ('lifecycle', 'last_committed_token', 'logical_kv_len'),
}
_HEADER = Struct('<IQI')
_ROW = Struct('<HBBQ')
_SCALAR = Struct('<Q')
_HANDLE = Struct('<QII')
# One control turn checks at most this many dependencies. Large WORKs stream
# several frames; source readiness is never held for the remaining selectors.
FACT_BUDGET = 64


@dataclass(frozen=True, slots=True)
class InputFact:
    index: int
    name: str
    selector: Selector
    snapshot: dict
    observed_ns: int = 0

    @classmethod
    def capture(cls, work, event):
        seq, index, name = event.key
        if seq != work.work_seq:
            raise ValueError('capture WORK mismatch')
        selector = getattr(work.rows[index], name).selector
        return cls(index, name, selector,
            {n: event.snapshot.get(n) for n in FIELDS[selector]}, event.observed_ns)


def encode_inputs(data):
    events = data['events']
    if not 0 < len(events) <= FACT_BUDGET:
        raise ValueError('input fact frame bound exceeded')
    chunks = [_HEADER.pack(data['worker_id'], data['worker_generation'], len(events))]
    for event in events:
        if not 0 <= event.index < MAX_ROWS:
            raise ValueError('input member out of range')
        chunks.append(_ROW.pack(event.index, NAMES.index(event.name), event.selector, event.observed_ns))
        for name in FIELDS[event.selector]:
            value = event.snapshot[name]
            chunks.append(_HANDLE.pack(value.offset, value.length, value.generation)
                if name.endswith('_handle') else _SCALAR.pack(value))
    return b''.join(chunks)


def decode_inputs(raw):
    try:
        worker, generation, count = _HEADER.unpack_from(raw)
        if not 0 < count <= FACT_BUDGET:
            raise ValueError('input fact frame bound exceeded')
        pos, events = _HEADER.size, []
        for _ in range(count):
            index, name_id, selector, observed = _ROW.unpack_from(raw, pos)
            pos += _ROW.size
            if index >= MAX_ROWS or name_id >= len(NAMES):
                raise ValueError('invalid input fact member/type')
            selector = Selector(selector)
            snapshot = {}
            for name in FIELDS[selector]:
                codec = _HANDLE if name.endswith('_handle') else _SCALAR
                values = codec.unpack_from(raw, pos)
                pos += codec.size
                snapshot[name] = ArenaHandle(*values) if codec is _HANDLE else values[0]
            events.append(InputFact(index, NAMES[name_id], selector, snapshot, observed))
        if pos != len(raw):
            raise ValueError('trailing input fact bytes')
        return dict(worker_id=worker, worker_generation=generation, events=tuple(events))
    except StructError as error:
        raise ValueError('truncated input facts') from error
