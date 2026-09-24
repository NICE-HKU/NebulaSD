"""CPU-only physical row allocator, owned by Runtime. No GPU cleanup."""
from dataclasses import dataclass
from enum import Enum, auto


class Phase(Enum):
    FREE = auto()
    FILLING = auto()
    READY = auto()
    COMPUTING = auto()
    EXPORTING = auto()


@dataclass(frozen=True, slots=True)
class Layout:
    bank_id: int
    bank_epoch: int
    rows: tuple[int, ...]
    offsets: tuple[int, ...]
    capacities: tuple[int, ...]


@dataclass(slots=True)
class Bank:
    bank_id: int
    epoch: int = 0
    phase: Phase = Phase.FREE
    current: object = None
    pending: object = None
    layout: Layout | None = None
    observation_seq: int = 0


class Banks:
    def __init__(self, blocks_per_bank, capacity_rows):
        if min(blocks_per_bank, capacity_rows) <= 0:
            raise ValueError('positive physical capacities required')
        self.capacity_rows = capacity_rows
        self.blocks_per_bank = blocks_per_bank
        self.free_rows = set(range(capacity_rows))
        self.banks = (Bank(0), Bank(1))

    def allocate(self, work):
        bank = self.banks[work.bank_id]
        if bank.layout is not None:
            raise RuntimeError('Bank still physically owned')
        if sum(r.capacity_blocks for r in work.rows) > self.blocks_per_bank:
            raise ValueError('WORK exceeds physical Bank')
        if len(work.rows) > len(self.free_rows):
            return None
        if work.bank_epoch <= bank.epoch:
            raise ValueError('Bank epoch must advance')
        rows = tuple(sorted(self.free_rows)[:len(work.rows)])
        self.free_rows.difference_update(rows)
        layout = Layout(work.bank_id, work.bank_epoch, rows,
            tuple(work.bank_id * self.blocks_per_bank + r.destination_offset for r in work.rows),
            tuple(r.capacity_blocks for r in work.rows))
        bank.layout, bank.epoch, bank.phase = layout, work.bank_epoch, Phase.FILLING
        bank.observation_seq += 1
        return layout

    def release(self, layout):
        bank = self.banks[layout.bank_id]
        if bank.layout is not layout:
            raise RuntimeError('release must name the owned layout')
        self.free_rows.update(layout.rows)
        bank.layout, bank.phase = None, Phase.FREE
        bank.observation_seq += 1
