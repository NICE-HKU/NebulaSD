"""Partial completion publications preserve complete shared facts and fences."""
from dataclasses import replace
import pytest
from contracts.test_draft_migration_contract import table, host_ready
from support.draft_contract_fixture import seed, prepare
from nebulasd.core.enums import StateChangeBlockKind as K, H2DStatus, D2HStatus
from nebulasd.table.draft_fences import read, publish, prepare_values
from nebulasd.table.draft_writers import DraftDestinationCopyWriter, DraftSourceCopyWriter
from nebulasd.table.storage import TableProtocolError


def values(snapshot):
    return {f.name:f.value for f in snapshot.fields}


def test_h2d_completion_retains_identity_and_new_prepare_replaces_it(table):
    s,h=seed(table);host_ready(table,s,h);w=DraftDestinationCopyWriter(table)
    for seq,epoch,row in [(1,1,3),(2,2,4)]:
        p=prepare(s,h,prepare_seq=seq,epoch=epoch)
        publish(table,K.REQUEST_DISPATCH,0,prepare_values(p,p.requests[0]))
        w.publish_h2d(command=p,request=p.requests[0],status=H2DStatus.WAIT_HOST)
        w.publish_h2d(command=p,request=p.requests[0],status=H2DStatus.IN_H2D,local_row=row,copy_start_time_ns=100)
        before=read(table,K.REQUEST_DRAFT_H2D,0)
        w.publish_h2d(command=p,request=p.requests[0],status=H2DStatus.GPU_READY,local_row=row,copy_bytes=999)
        after=read(table,K.REQUEST_DRAFT_H2D,0)
        expected=values(before)
        expected.update(status=int(H2DStatus.GPU_READY),gpu_ready_version=1,copied_blocks=3,copy_start_time_ns=0,copy_bytes=999)
        assert values(after)==expected
        assert after.publish_seq==before.publish_seq+1
        assert after.get('destination_bank_epoch')==epoch and after.get('observed_prepare_seq')==seq


def test_d2h_completion_retains_snapshot_and_allocation(table):
    s,h=seed(table);w=DraftSourceCopyWriter(table)
    args=dict(identity=s.identity,snapshot_handle=h,bank_id=0,bank_epoch=1,batch_seq=10)
    w.publish_d2h(**args,status=D2HStatus.IN_D2H,copy_start_time_ns=100,copy_bytes=777)
    before=read(table,K.REQUEST_DRAFT_D2H,0)
    w.publish_d2h(**args,status=D2HStatus.HOST_READY,copy_bytes=777)
    after=read(table,K.REQUEST_DRAFT_D2H,0);expected=values(before)
    expected.update(status=int(D2HStatus.HOST_READY),ready_version=1,copy_start_time_ns=0)
    assert values(after)==expected and after.publish_seq==before.publish_seq+1


def test_partial_completion_does_not_bypass_scalar_type_validation(table):
    s,h=seed(table);w=DraftSourceCopyWriter(table)
    args=dict(identity=s.identity,snapshot_handle=h,bank_epoch=1,batch_seq=10)
    w.publish_d2h(**args,bank_id=0,status=D2HStatus.IN_D2H)
    before=read(table,K.REQUEST_DRAFT_D2H,0)
    with pytest.raises(TypeError):
        w.publish_d2h(**args,bank_id=False,status=D2HStatus.HOST_READY)
    assert read(table,K.REQUEST_DRAFT_D2H,0)==before


@pytest.mark.parametrize('kind,field',[(K.REQUEST_ENGINE,'request_epoch'),(K.REQUEST_DRAFT_HOSTKV,'writer_lease_generation'),(K.REQUEST_DISPATCH,'draft_next_owner_epoch'),(K.REQUEST_DRAFT_D2H,'ready_version')])
def test_completion_rereads_changed_external_fences(table,kind,field):
    s,h=seed(table);host_ready(table,s,h);p=prepare(s,h)
    publish(table,K.REQUEST_DISPATCH,0,prepare_values(p,p.requests[0]))
    w=DraftDestinationCopyWriter(table)
    w.publish_h2d(command=p,request=p.requests[0],status=H2DStatus.IN_H2D,local_row=3)
    before=read(table,K.REQUEST_DRAFT_H2D,0)
    publish(table,kind,0,{field:read(table,kind,0).get(field)+1})
    with pytest.raises(TableProtocolError):
        w.publish_h2d(command=p,request=p.requests[0],status=H2DStatus.GPU_READY,local_row=3)
    assert read(table,K.REQUEST_DRAFT_H2D,0)==before
