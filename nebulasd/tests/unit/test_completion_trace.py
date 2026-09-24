import json
from types import SimpleNamespace

import pytest

from nebulasd.observability.completion_trace import CompletionTrace
from test_autonomous_engine import engine


def test_trace_preserves_return_exception_and_bounds(tmp_path):
    trace = CompletionTrace(2)
    error = ValueError('original failure')
    def operation(value):
        if value < 0:
            raise error
        return value + 3
    obj = SimpleNamespace(operation=operation)
    trace.wrap(obj, 'operation', 'op', lambda args, kwargs: args)
    assert obj.operation(4) == 7
    assert obj.operation(5) == 8
    with pytest.raises(ValueError) as caught:
        obj.operation(-1)
    assert caught.value is error
    trace.flush(tmp_path)
    saved = json.loads((tmp_path/'completion-trace.json').read_text())
    assert saved['total_events'] == 3 and saved['dropped_events'] == 1
    assert [e[4] for e in saved['events']] == [[5], [-1]]
    assert all(e[2] >= e[1] and e[3] >= 0 for e in saved['events'])


def test_trace_rehooks_ledger_after_cohort_and_records_credit(engine, tmp_path):
    from nebulasd.observability.completion_trace import attach
    from nebulasd.engine.work_ledger import WorkLedger
    from nebulasd.data.generation_config_arena import DraftGenerationConfig
    from test_autonomous_engine import publish_target
    e = engine
    trace = attach(e, tmp_path)
    e.admit('a', (1, 2, 3), DraftGenerationConfig(8, 4))
    e.step()
    assert any(event[0] == 'capacity' for event in trace.events)
    work = e.supervisor.pairs[1].work[0]
    publish_target(e, work)
    e.step()
    assert any(event[0] == 'credit' and tuple(event[4]) == e.ledger.key(work)
               for event in trace.events)
    # Replacement occurs only at the all-owner cohort barrier in production.
    replacement = WorkLedger(e.resources)
    e.ledger = replacement
    e.scheduling_progress.pending = {'D': (), 'T': ()}
    e.scheduling_progress.dirty = {'D': set(), 'T': set()}
    e._retry_dispatch = False
    e.step()
    assert 'capacity' in replacement.__dict__ and 'observe_completions' in replacement.__dict__
    assert replacement.observe_completions() is False
