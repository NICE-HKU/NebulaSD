"""Held physical tickets test independence and fail-stop cleanup, without CUDA."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from nebulasd.kv.bank_transfer import BankCopyExecutor


class Backend:
    def __init__(self):
        self.started, self.gate, self.closed = Event(), Event(), Event()
        self.failure = None

    def launch(self, plan):
        self.started.set()
        return self

    def query(self):
        if not self.gate.is_set():
            return False
        if self.failure:
            raise self.failure
        return True

    def duration_ms(self):
        return 0

    def close(self):
        assert self.gate.is_set()
        self.closed.set()


@pytest.mark.parametrize('first', (0, 1))
def test_other_bank_completes_and_reuses_slot_while_first_is_held(first):
    backends = [Backend(), Backend()]
    copy = BankCopyExecutor(backends)
    try:
        slow = copy.submit_batch('H2D', (first, 3, 5))
        assert backends[first].started.wait(2)
        with pytest.raises(RuntimeError, match='outstanding'):
            copy.submit_batch('D2H', (first, 3, 5))
        fast = copy.submit_batch('D2H', (1-first, 7, 9))
        backends[1-first].gate.set()
        fast.result(2)
        assert not slow.done()
        copy.submit_batch('H2D', (1-first, 8, 10)).result(2)
        assert not slow.done()
    finally:
        for backend in backends:
            backend.gate.set()
        copy.close()
    assert all(b.closed.is_set() for b in backends)


@pytest.mark.parametrize('failed', (0, 1))
def test_failure_forbids_new_work_and_close_waits_other_bank(failed):
    backends = [Backend(), Backend()]
    copy = BankCopyExecutor(backends)
    futures = [copy.submit_batch('copy', (i, 1, i+1)) for i in range(2)]
    for backend in backends:
        assert backend.started.wait(2)
    backends[failed].failure = RuntimeError('DMA failed')
    backends[failed].gate.set()
    with pytest.raises(RuntimeError, match='DMA failed'):
        futures[failed].result(2)
    with pytest.raises(RuntimeError, match='DMA failed'):
        copy.submit_batch('new', (1-failed, 2, 3))
    with ThreadPoolExecutor(1) as closer:
        closed = closer.submit(copy.close)
        assert not closed.done()
        assert not backends[1-failed].closed.is_set()
        backends[1-failed].gate.set()
        with pytest.raises(RuntimeError, match='DMA failed'):
            closed.result(2)
    assert all(b.closed.is_set() for b in backends)
    with pytest.raises(RuntimeError, match='DMA failed'):
        copy.close()
