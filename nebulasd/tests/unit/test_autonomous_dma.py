import pytest
from nebulasd.workers.dma import DMA


class Backend:
    def initialize(self):
        pass
    def close(self):
        pass


@pytest.mark.parametrize('explicit,expected', [(None, 8192), (0, 0), (4096, 4096)])
def test_chunk_default_inherits_environment_and_explicit_value_wins(monkeypatch, explicit, expected):
    monkeypatch.setenv('STARSD_H2D_CHUNK_BYTES', '8192')
    args = {} if explicit is None else {'h2d_chunk_bytes': explicit}
    dma = DMA((Backend(), Backend()), **args)
    try:
        assert [e._h2d_chunk_bytes for e in dma.executors] == [expected, expected]
    finally:
        dma.close()


def test_completion_delay_never_replaces_real_cuda_completion(monkeypatch):
    from nebulasd.workers import diagnostics
    from types import SimpleNamespace
    clock = [10.0]
    done = [False]
    monkeypatch.setattr(diagnostics, 'monotonic', lambda: clock[0])
    ticket = diagnostics.DelayedTicket(SimpleNamespace(query=lambda: done[0], duration_ms=lambda: 1.25), .5)
    assert not ticket.query()
    done[0] = True
    assert not ticket.query()
    clock[0] += 1
    assert ticket.query()
    done[0] = False
    assert not ticket.query()
    assert ticket.duration_ms() == 1.25
