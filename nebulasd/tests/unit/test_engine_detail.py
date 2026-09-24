from contextlib import ExitStack
from types import SimpleNamespace
from nebulasd.observability.engine_detail import attach_control
import json


def test_control_trace_preserves_coalesced_notifications(tmp_path, monkeypatch):
    monkeypatch.setenv('STARSD_ENGINE_DETAIL_DIR', str(tmp_path))
    rings=[];sent=[]
    bell=SimpleNamespace(ring=lambda:rings.append(1))
    def advance(record):
        if record['result']==0:bell.ring()
        record['result']+=1
        return True,False
    publisher=SimpleNamespace(_advance=advance,doorbells=())
    outgoing=SimpleNamespace(put_nowait=sent.append)
    record=dict(work=SimpleNamespace(work_seq=7,rows=(1,2)),result=0)
    with ExitStack() as stack:
        attach_control({'worker_id':2},publisher,SimpleNamespace(doorbell=bell),stack)
        assert publisher._advance(record)==(True,False)
        publisher._advance(record)
        outgoing.put_nowait(('RESULT',7,{}))
    events=json.loads(next(tmp_path.glob('*.json')).read_text())['events']
    assert len(rings)==1 and len(sent)==1
    assert [(e['name'],e['seq']) for e in events]==[
        ('detail.worker_bell',7),('detail.result_rows_ready',7)]
