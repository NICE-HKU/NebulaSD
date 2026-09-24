"""Actual predecessor clocks survive migration, skipping and identity reuse."""
from types import SimpleNamespace as NS
import pytest
from nebulasd.scheduler.compute_times import ComputeTimes
from nebulasd.workers.work import WorkKind


def work(stage, epoch=1, round_id=1, worker=0):
    return NS(operation=WorkKind.DRAFT_DECODE if stage == 'D' else WorkKind.TARGET_VERIFY,
              worker_id=worker, rows=tuple(NS(slot=i, epoch=epoch, round_id=round_id) for i in range(3)))


def result(end, indices=(0, 2)):
    return dict(compute_start_ns=end-10, compute_end_ns=end,
                rows=[dict(index=i) for i in indices])


@pytest.mark.parametrize('stage', ['D', 'T'])
def test_executed_members_only_and_exact_predecessor_round(stage):
    times = ComputeTimes()
    times.observe(work(stage), result(100))
    assert times.end_ns(0, 1, stage, 1) == 100
    assert times.end_ns(2, 1, stage, 1) == 100
    assert times.end_ns(1, 1, stage, 1) is None  # skipped member
    assert times.end_ns(0, 1, stage, 2) is None  # future round
    times.observe(work(stage, round_id=2, worker=7), result(200))
    times.observe(work(stage), result(100))  # late older result cannot rewind
    assert times.end_ns(0, 1, stage, 2) == 200
    assert times.end_ns(0, 1, stage, 1) is None
    assert not times.observe(work(stage, round_id=2, worker=7), result(200))
    with pytest.raises(ValueError, match='conflicting'):
        times.observe(work(stage, round_id=2, worker=7), result(210))


def test_slot_reuse_and_both_stages_are_independent():
    times = ComputeTimes()
    times.observe(work('D', round_id=9), result(100))
    times.observe(work('T', round_id=9), result(200))
    times.observe(work('D', epoch=2), result(300))
    times.observe(work('D', round_id=10), result(250))
    assert times.end_ns(0, 2, 'D', 1) == 300
    assert times.end_ns(0, 1, 'D', 9) is None
    assert times.end_ns(0, 2, 'T', 9) is None
    assert times.end_ns(0, 1, 'T', 9) == 200
    times.clear()
    assert times.end_ns(0, 2, 'D', 1) is None
