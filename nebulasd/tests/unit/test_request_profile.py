from nebulasd.observability.request_profile import summarize_requests


def event(name, a, b=None, epoch=1, **fields):
    return dict(name=name, start_ns=a*1000000, end_ns=(a if b is None else b)*1000000,
                keys=[[0,epoch,0]], **fields)


def test_lifecycle_prefill_overlap_and_shared_batch():
    events = [event('client.submitted',0,request_id=9),
              event('client.output_available',90,terminal=True,lifecycle='FINISHED'),
              event('client.output_observed',100,terminal=True),
              event('backend.prefill_batch_async',10,40,category='backend_wall'),
              event('backend.run_batch',30,60,category='backend_wall'),
              event('gpu.target',10,40,category='gpu'),
              event('gpu.draft',30,60,category='gpu'),
              event('gpu.H2D',50,80,category='gpu')]
    events[-1]['keys'].append([1,1,0])
    result = summarize_requests(events)
    row = result['requests'][0]
    assert row['complete']
    assert row['total_ms']==100
    assert row['output_available_to_observed_ms']==10
    assert row['backend_service_ratio']==.5
    assert row['gpu_compute_ratio']==.5
    assert row['exclusive_gpu_ms']==dict(compute=50,copy_without_compute=20,other=30)
    assert row['interval_union_ms']['gpu_copy']==30


def test_epoch_reuse_loss_missing_observation_and_no_gpu():
    events = [event('client.submitted',0),event('client.output_available',10,terminal=True),
              event('client.submitted',20,epoch=2),event('client.output_available',40,epoch=2,terminal=True),
              event('client.output_observed',50,epoch=2,terminal=True)]
    rows = summarize_requests(events)['requests']
    assert not rows[0]['complete'] and rows[0]['total_ms']==10
    assert rows[0]['output_available_to_observed_ms'] is None
    assert rows[1]['complete'] and rows[1]['total_ms']==30
    assert rows[1]['gpu_compute_ratio'] is None
    assert summarize_requests(events,dropped=1)['complete_requests']==0


def test_client_hooks_cover_read_stream_and_cancel(tmp_path):
    from types import SimpleNamespace as NS
    from collections import deque
    from nebulasd.engine.client import TokenEngine
    from nebulasd.core.enums import Lifecycle
    from nebulasd.observability.client_profiling import attach_client
    from nebulasd.observability.profiling import ProfileRecorder
    from nebulasd.api import RequestHandle
    engine=NS(outputs=NS(),_check_owner=lambda:None)
    client=TokenEngine(NS(),engine)
    original=client.read
    recorder=ProfileRecorder(tmp_path,'engine',100)
    attach_client(client,recorder)
    assert client.read != original
    for i in range(2):
        handle=RequestHandle(i,client._epoch)
        client._handles[i]=handle
        client._records[i]=NS(input=NS(slot=0,epoch=i+1),lifecycle=Lifecycle.ACTIVE)
        client._buffers[i]=deque()
    h=client._handles[0]
    terminal=next(x for x in Lifecycle if x.name not in ('ACTIVE','CANCELLED'))
    engine.outputs.on_tokens('0',(1,2),terminal)
    assert len(client.read(h))==1
    assert client.read(h)==()
    h=client._handles[1]
    def cancel(identity):client._records[int(identity)].lifecycle=Lifecycle.CANCELLED
    engine.cancel=cancel
    client.cancel(h)
    assert len(list(client.stream(h)))==1
    client.cancel(h)
    assert sum(e['name']=='client.output_available' for e in recorder.events)==2
    assert sum(e['name']=='client.output_observed' for e in recorder.events)==2


def test_shared_service_not_divided_and_clipped_to_lifecycle():
    events=[]
    for epoch, start, end in [(1,10,30),(2,20,50)]:
        events.extend([event('client.submitted',start,epoch=epoch),
                       event('client.output_available',end,epoch=epoch,terminal=True),
                       event('client.output_observed',end,epoch=epoch,terminal=True)])
    compute=event('gpu.target',0,40,category='gpu')
    compute['keys']=[[0,1,0],[0,2,0]]
    events.append(compute)
    a,b=summarize_requests(events)['requests']
    assert a['interval_union_ms']['gpu_compute']==20
    assert b['interval_union_ms']['gpu_compute']==20
    assert a['gpu_compute_ratio']==1
    assert b['gpu_compute_ratio']==2/3
