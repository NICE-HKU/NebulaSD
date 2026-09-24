"""Scheduler-owned actual compute ends, independent of WORK retirement.

Keep only the latest executed round per (slot, stage). Alternating request
execution makes this sufficient for the next stage's predecessor; it is not a
history or a source of execution authorization. Unknown times stay unknown.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComputeEnd:
    epoch: int
    round_id: int
    end_ns: int


class ComputeTimes:
    def __init__(self):
        self._latest = {}

    def observe(self, work, result):
        stage = 'D' if work.operation.name.startswith('DRAFT') else 'T'
        end = result['compute_end_ns']
        if not 0 <= result['compute_start_ns'] <= end:
            raise ValueError('invalid compute interval')
        changed = set()
        # Result indices refer to original WORK rows, including when some
        # members were skipped. Never attribute batch service to skipped rows.
        for output in result['rows']:
            index = output['index'] if isinstance(output, dict) else output.index
            if not 0 <= index < len(work.rows):
                raise ValueError('compute result index outside WORK')
            row = work.rows[index]
            key = row.slot, stage
            current = self._latest.get(key)
            identity = row.epoch, row.round_id
            if current is not None:
                previous = current.epoch, current.round_id
                if identity < previous:
                    continue
                if identity == previous:
                    if end != current.end_ns:
                        raise ValueError('conflicting compute end for request round')
                    continue
            self._latest[key] = ComputeEnd(row.epoch, row.round_id, end)
            changed.add(row.slot)
        return changed

    def observe_row(self, row):
        from nebulasd.scheduler.views import value as v
        from nebulasd.core.enums import StateChangeBlockKind as K
        if row.block_kind not in (K.REQUEST_DRAFT, K.REQUEST_TARGET_COMPUTE) or v(row, 'status') != 2:
            return
        stage = 'D' if row.block_kind == K.REQUEST_DRAFT else 'T'
        start, end = v(row, 'compute_start_ns', 0), v(row, 'compute_end_ns', 0)
        if not 0 <= start <= end:
            raise ValueError('invalid shared compute interval')
        key = row.row, stage
        latest = self._latest.get(key)
        identity = v(row, 'request_epoch'), v(row, 'round_id')
        if latest is None or identity > (latest.epoch, latest.round_id):
            self._latest[key] = ComputeEnd(*identity, end)

    def latest(self, slot, epoch, stage):
        value = self._latest.get((slot, stage))
        return value if value is not None and value.epoch == epoch else None

    def end_ns(self, slot, epoch, stage, round_id):
        value = self.latest(slot, epoch, stage)
        return value.end_ns if value is not None and value.round_id == round_id else None

    def clear(self):
        self._latest.clear()
