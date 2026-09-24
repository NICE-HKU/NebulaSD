"""Client clock semantics and input configuration checks."""
import pytest
from nebulasd.api import RequestHandle,StreamEvent
from nebulasd.config import NebulaSDConfig
from nebulasd.core.enums import Lifecycle
from examples.support.output_metrics import OutputMetrics


def test_chunk_intervals_do_not_invent_per_token_timestamps():
    metrics = OutputMetrics()
    request = RequestHandle(0,1)
    metrics.admit(request,0)
    metrics.observe(StreamEvent(request,(1,2,3),Lifecycle.ACTIVE),1_000_000)
    metrics.observe(StreamEvent(request,(4,),Lifecycle.FINISHED),3_000_000)
    result = metrics.summary(0,4_000_000)
    assert result['ttft']['p50_ms']==1
    assert result['output_chunk_interval']['samples']==1
    assert result['output_chunk_interval']['p50_ms']==2
    assert result['output_tokens_per_second']==1000
    assert result['actual_chunk_sizes']==[1,3]


def test_independent_batch_size_defaults_and_target_bank_rows():
    config = NebulaSDConfig(max_batch_size=5,bank_rows=5)
    assert config.draft_max_batch_size==5
    assert config.target_max_batch_size==5
    assert config.draft_max_batch_tokens==512
    assert config.target_verify_max_batch_tokens==512
    config = NebulaSDConfig(max_batch_size=5,draft_max_batch_size=7,target_max_batch_size=3,bank_rows=3)
    assert config.draft_max_batch_size==7
    assert config.target_max_batch_size==3
    config = NebulaSDConfig(max_batch_tokens=384,draft_max_batch_tokens=256,
                              target_verify_max_batch_tokens=128)
    assert config.max_batch_tokens==384
    assert config.draft_max_batch_tokens==256
    assert config.target_verify_max_batch_tokens==128


@pytest.mark.parametrize('changes',[dict(command_ring_capacity=3),dict(state_change_ring_capacity=0),
    dict(devices=(0,0)),dict(startup_timeout=float('nan')),dict(gpu_memory_fraction=1),
    dict(max_batch_size=9,target_max_batch_size=9,bank_rows=8),dict(max_batch_size=129,bank_rows=258),
    dict(draft_max_batch_size=0),dict(target_max_batch_size=0),dict(draft_max_batch_size=129),
    dict(target_max_batch_size=129,bank_rows=129),dict(draft_max_batch_tokens=0),
    dict(target_verify_max_batch_tokens=0),dict(failure_policy=2)])
def test_invalid_config_rejected_before_startup(changes):
    with pytest.raises(ValueError):
        NebulaSDConfig(**changes)


def test_independent_bank_capacities_reach_worker_specs(monkeypatch):
    from nebulasd.engine import bootstrap
    from nebulasd.config import HostKVLayout
    default = NebulaSDConfig(bank_blocks=256)
    assert (default.draft_bank_blocks, default.target_bank_blocks) == (256, 256)
    config = NebulaSDConfig(draft_bank_blocks=2560, target_bank_blocks=768,
        scheduler_execution="completion",draft_placement="stagewise",target_placement="stagewise",
        cost_model="table",backend_cost_table="test.json")
    from nebulasd.engine import model_factory
    monkeypatch.setattr(model_factory,"model_layout",lambda *a,**k:HostKVLayout())
    def capture(specs, **kwargs):
        assert [s.bank_blocks for s in specs] == [2560, 768]
        raise RuntimeError('captured capacities before allocation')
    monkeypatch.setattr(bootstrap, 'ControlResources', capture)
    with pytest.raises(RuntimeError, match='captured capacities'):
        bootstrap.create(config, worker_factory=None, worker_options=None,
                         kv_layout=None, observer=None)
    for field in ('draft_bank_blocks', 'target_bank_blocks'):
        for value in (0, -1, True):
            with pytest.raises(ValueError):
                NebulaSDConfig(**{field: value})
