"""Scheduling consumes owner state without local ENGINE or imported-row mirrors."""
import pytest
from test_autonomous_engine import engine, publish_target
from nebulasd.core.enums import StateChangeBlockKind as K, Lifecycle
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from nebulasd.table.storage import FieldValue


@pytest.mark.parametrize('implementation', ['python', 'cpp', 'check'])
@pytest.mark.parametrize('policy', ['existing', 'service_interval'])
def test_owner_request_state_drives_both_policies(engine, monkeypatch, implementation, policy):
    e = engine
    monkeypatch.setenv('STARSD_SCHEDULER_IMPL', implementation)
    if policy == 'service_interval':
        from nebulasd.scheduler.service_interval import ServiceIntervalScheduler
        e.scheduler = ServiceIntervalScheduler(estimator=e.scheduler._estimator,
            draft_service_gap_ms=0, target_service_gap_ms=0)
        e.scheduler.enable_native_updates()
    e.admit('a', (1, 2, 3), DraftGenerationConfig(8, 4))
    initial = e.registry.records[0].input
    e.step()
    work = e.supervisor.pairs[1].work[0]
    publish_target(e, work)
    e._observe_facts()
    from nebulasd.scheduler.eligibility import active, target_result
    view = e._scheduling_view()
    assert active(view, e.candidates[0]) and target_result(view, e.candidates[0]) is not None
    e.step()
    assert e.supervisor.pairs[0].work
    assert initial.output_count == 0 and e.candidates[0].output_count == 1
    assert e.registry.records[0].output == (4,)
    assert not any(k in (K.REQUEST_ENGINE, K.REQUEST_H2D, K.REQUEST_DRAFT_H2D) for k, _ in e.rows)
    request = e.candidates[0]
    e.registry.records[0].lifecycle = Lifecycle.FINISHED
    e._request_changed(e.registry.records[0])
    assert not active(e._scheduling_view(), request) and not e.candidates


def test_import_rows_remain_shared_but_engine_never_reads_them(engine, monkeypatch):
    e = engine
    e._observe_facts()
    e.admit('a', (1, 2, 3), DraftGenerationConfig(8, 4))
    e.step()
    for kind in (K.REQUEST_H2D, K.REQUEST_DRAFT_H2D):
        part = e.resources.table.partition(kind)
        part._publish(0, 1, (FieldValue('request_epoch', 1), FieldValue('status', 3)))
    # A real Worker table remains readable, even though the owner drops hints.
    assert e.resources.table.partition(K.REQUEST_H2D).read_stable(0).get('status') == 3
    validate = e.ledger.validate_fact
    def check(row):
        assert row.block_kind not in (K.REQUEST_H2D, K.REQUEST_DRAFT_H2D)
        return validate(row)
    monkeypatch.setattr(e.ledger, 'validate_fact', check)
    e._observe_facts()
    assert not any(k in (K.REQUEST_H2D, K.REQUEST_DRAFT_H2D) for k, _ in e.rows)
