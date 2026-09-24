"""Timestamp/identity assertions must fail rather than manufacture telemetry."""
import pytest
from support.draft_latency_report import publication_delay, observed_ready


def test_observer_can_read_before_publication_call_returns():
    assert publication_delay(150,dict(begin_ns=100,end_ns=200))==dict(lower_ms=0,upper_ms=.00005)
    assert publication_delay(300,dict(begin_ns=100,end_ns=200))==dict(lower_ms=.0001,upper_ms=.0002)


def test_observation_before_matching_publication_is_not_clamped_into_success():
    with pytest.raises(AssertionError,match='observation precedes'):
        publication_delay(99,dict(begin_ns=100,end_ns=200))


@pytest.mark.parametrize('bad_field',['snapshot_version','arena_generation','destination_bank_epoch','copied_blocks'])
def test_ready_join_rejects_stale_snapshot_allocation_or_bank(bad_field):
    identity=dict(request_slot=0,request_epoch=2,round_id=3,snapshot_version=4,logical_kv_len=31,valid_blocks=2,allocation=dict(arena_generation=5))
    pub=dict(identity=identity,fact_kind='REQUEST_DRAFT_H2D',status=3,publish_seq=8,worker=1,bank_id=0,bank_epoch=9,batch_seq=10,next_owner_epoch=2,snapshot_handle=dict(offset=0,length=200,generation=5))
    fields=dict(status=3,snapshot_round_id=3,snapshot_version=4,logical_kv_len=31,result_code=0,arena_generation=5,destination_worker_id=1,destination_bank_id=0,destination_bank_epoch=9,prepared_batch_seq=10,next_owner_epoch=2,gpu_ready_version=4,copied_blocks=2,snapshot_handle=pub['snapshot_handle'])
    fields[bad_field]+=1
    obs={('REQUEST_DRAFT_H2D',0,2,8):[dict(observed_ns=100,fields=fields)]}
    with pytest.raises(AssertionError,match='mismatched ready identity'):
        observed_ready(pub,obs)
