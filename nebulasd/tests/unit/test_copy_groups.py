"""Ordered completion fences preserve whole-plan readiness during grouped H2D."""
from threading import Event
from types import SimpleNamespace as NS

import pytest

from nebulasd.kv.transfer import CopyExecutor, CopyPlan, CopyRegion


@pytest.mark.parametrize('group_size', [1, 2, 4, 8])
def test_groups_wait_for_last_ticket_and_only_complete_after_final_group(group_size):
    count = 9
    gates = [Event() for _ in range(count)]
    queried = [Event() for _ in range(count)]
    launches, durations = [], []
    dependency = object()
    extent = NS(capacity_blocks=count)
    regions = tuple(CopyRegion(extent, i, i, 1) for i in range(count))
    plan = CopyPlan('H2D', regions, (dependency,), tuple(range(count)))

    class Ticket:
        def __init__(self, index):
            self.index = index
        def query(self):
            # Only the final ticket of each group may be polled.
            assert (self.index + 1) % group_size == 0 or self.index == count - 1
            queried[self.index].set()
            return gates[self.index].is_set()
        def duration_ms(self):
            end = min((self.index // group_size + 1) * group_size, count) - 1
            assert gates[end].is_set(), 'timing read before ordered group completion'
            durations.append(self.index)
            return self.index + 0.5

    class Backend:
        arena = NS(descriptor=NS(block_bytes=16))
        def launch(self, chunk):
            index = len(launches)
            if index >= group_size:
                assert gates[index // group_size * group_size - 1].is_set()
            launches.append(chunk)
            return Ticket(index)
        def close(self):
            pass

    executor = CopyExecutor(Backend(), h2d_chunk_bytes=32, h2d_group_size=group_size)
    try:
        future = executor.submit(plan)
        for end in range(group_size - 1, count + group_size - 1, group_size):
            end = min(end, count - 1)
            assert queried[end].wait(2)
            assert len(launches) == end + 1
            assert not future.done()
            with pytest.raises(RuntimeError, match='outstanding'):
                executor.submit(plan)
            gates[end].set()
        receipt = future.result(2)
        assert tuple(c.regions[0] for c in launches) == regions
        assert [c.dependencies for c in launches] == [(dependency,)] + [()] * (count - 1)
        assert [c.round_ids for c in launches] == [(i,) for i in range(count)]
        assert durations == list(range(count))
        assert receipt.duration_ms == sum(i + 0.5 for i in range(count))
        assert receipt.enqueued_ns <= receipt.submitted_ns <= receipt.launch_returned_ns <= receipt.completed_ns
    finally:
        for gate in gates:
            gate.set()
        executor.close()


@pytest.mark.parametrize('value', ['0', '-1', 'bad'])
def test_invalid_group_environment_fails_before_starting_executor(monkeypatch, value):
    monkeypatch.setenv('STARSD_H2D_GROUP_SIZE', value)
    with pytest.raises(ValueError):
        CopyExecutor(NS())


def test_default_environment_and_explicit_group_size(monkeypatch):
    backend = NS(close=lambda: None)
    monkeypatch.delenv('STARSD_H2D_GROUP_SIZE', raising=False)
    default = CopyExecutor(backend)
    default.close()
    assert default._h2d_group_size == 4
    monkeypatch.setenv('STARSD_H2D_GROUP_SIZE', '8')
    inherited = CopyExecutor(backend)
    explicit = CopyExecutor(backend, h2d_group_size=1)
    inherited.close()
    explicit.close()
    assert inherited._h2d_group_size == 8 and explicit._h2d_group_size == 1


def test_event_wait_keeps_future_pending_without_polling():
    entered, release = Event(), Event()
    extent = NS(capacity_blocks=2)
    plan = CopyPlan('H2D', tuple(CopyRegion(extent, i, i, 1) for i in range(2)))
    launches = []
    class Ticket:
        def synchronize(self):
            assert len(launches) == 2
            entered.set()
            assert release.wait(2)
        def query(self):
            raise AssertionError('event mode must not poll')
        def duration_ms(self):
            assert release.is_set()
            return 1.0
    backend = NS(arena=NS(descriptor=NS(block_bytes=16)),
                 launch=lambda p: launches.append(p) or Ticket(), close=lambda: None)
    copy = CopyExecutor(backend, h2d_chunk_bytes=32, h2d_wait_mode='event')
    try:
        f = copy.submit(plan)
        assert entered.wait(2)
        assert not f.done()
        release.set()
        receipt = f.result(2)
        assert receipt.duration_ms == 2
        assert receipt.launch_returned_ns <= receipt.last_pending_ns <= receipt.completed_ns
    finally:
        release.set()
        copy.close()


def test_event_wait_failure_is_propagated_and_forbids_reuse():
    def fail():
        raise RuntimeError('DMA event failed')
    backend = NS(launch=lambda p: NS(synchronize=fail), close=lambda: None)
    copy = CopyExecutor(backend, h2d_wait_mode='event')
    plan = CopyPlan('H2D', (CopyRegion(NS(capacity_blocks=1), 0, 0, 1),))
    try:
        with pytest.raises(RuntimeError, match='DMA event failed'):
            copy.submit(plan).result(2)
        with pytest.raises(RuntimeError, match='DMA event failed'):
            copy.submit(plan)
    finally:
        copy.close()


def test_invalid_wait_mode_rejected(monkeypatch):
    monkeypatch.setenv('STARSD_H2D_WAIT_MODE', 'wrong')
    with pytest.raises(ValueError, match='wait mode'):
        CopyExecutor(NS())


def test_event_mode_keeps_query_only_backends_and_d2h_contract():
    calls = []
    class Ticket:
        def query(self):
            calls.append('query')
            return True
        def duration_ms(self):
            return 1.0
    backend = NS(launch=lambda p: Ticket(), close=lambda: None)
    copy = CopyExecutor(backend, h2d_wait_mode='event')
    extent = NS(capacity_blocks=1)
    try:
        for direction in ('H2D', 'D2H'):
            copy.submit(CopyPlan(direction, (CopyRegion(extent, 0, 0, 1),))).result(2)
        assert calls == ['query', 'query']
    finally:
        copy.close()


def test_event_wait_on_one_executor_does_not_block_another():
    entered, release = Event(), Event()
    class Ticket:
        def synchronize(self):
            entered.set()
            assert release.wait(2)
        def duration_ms(self):
            return 1.0
    slow = CopyExecutor(NS(launch=lambda p: Ticket(), close=lambda: None), h2d_wait_mode='event')
    fast = CopyExecutor(NS(launch=lambda p: NS(synchronize=lambda: None, duration_ms=lambda: 1.0),
                           close=lambda: None), h2d_wait_mode='event')
    plan = CopyPlan('H2D', (CopyRegion(NS(capacity_blocks=1), 0, 0, 1),))
    try:
        pending = slow.submit(plan)
        assert entered.wait(2)
        assert fast.submit(plan).result(2).duration_ms == 1
        assert not pending.done()
        release.set()
        pending.result(2)
    finally:
        release.set()
        slow.close()
        fast.close()


def test_d2h_uses_query_even_with_synchronizable_ticket():
    calls = []
    ticket = NS(query=lambda: calls.append('query') or True,
                synchronize=lambda: calls.append('synchronize'), duration_ms=lambda: 1.0)
    copy = CopyExecutor(NS(launch=lambda p: ticket, close=lambda: None), h2d_wait_mode='event')
    try:
        copy.submit(CopyPlan('D2H', (CopyRegion(NS(capacity_blocks=1), 0, 0, 1),))).result(2)
        assert calls == ['query']
    finally:
        copy.close()


def test_default_wait_mode_is_event_and_explicit_poll_wins(monkeypatch):
    monkeypatch.delenv('STARSD_H2D_WAIT_MODE', raising=False)
    monkeypatch.delenv('STARSD_H2D_CHUNK_BYTES', raising=False)
    backend = NS(close=lambda: None)
    default = CopyExecutor(backend)
    try:
        assert default._h2d_wait_mode == 'event' and default._h2d_chunk_bytes == 0
    finally:
        default.close()
    monkeypatch.setenv('STARSD_H2D_WAIT_MODE', 'event')
    explicit = CopyExecutor(backend, h2d_wait_mode='poll')
    try:
        assert explicit._h2d_wait_mode == 'poll'
    finally:
        explicit.close()
