"""Profiling correlation and uncertainty contracts, without GPU dependencies."""
import json
from nebulasd.observability.profiling import ProfileRecorder
from nebulasd.observability.profile_report import summarize


def test_bounded_recorder_reports_loss(tmp_path):
    p = ProfileRecorder(tmp_path,'engine',2)
    for n in range(5):
        p.record('test',n)
    p.flush()
    r = json.loads((tmp_path/'engine.json').read_text())
    assert r['total_events']==5 and r['dropped_events']==3
    assert [e['start_ns'] for e in r['events']]==[3,4]
    assert summarize(tmp_path)['dropped_events']==3


def test_epoch_identity_and_missing_round_endpoint(tmp_path):
    p = ProfileRecorder(tmp_path,'worker-1',100)
    for epoch,round_id,now in ((1,0,10),(1,1,20),(2,1,30)):
        p.record('fact.publish',now,now+1,keys=[[0,epoch,round_id]],
                 kind='REQUEST_TARGET_COMPUTE',row=0,publish_seq=round_id,status=2)
    p.flush()
    r = summarize(tmp_path)
    assert len(r['rounds'])==2
    assert r['rounds'][0]['round_wall_ms']==10/1e6
    assert r['rounds'][1]['missing']==['previous_target_result']
    assert not any(item['complete'] for item in r['rounds'])


def test_sync_and_async_spans_do_not_claim_same_cpu_semantics(tmp_path):
    import asyncio
    class Task:
        def sync(self):
            return 7
        async def async_(self):
            return 8
    p = ProfileRecorder(tmp_path,'engine',10)
    task = Task()
    p.wrap(task,'sync','sync')
    p.wrap(task,'async_','async')
    assert task.sync()==7
    assert asyncio.run(task.async_())==8
    assert p.events[0]['cpu_ns'] is not None
    assert p.events[1]['cpu_ns'] is None


def test_backend_items_preserve_identity_in_executor(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    class Backend:
        def run_batch(self, items):
            return len(items)
    recorder = ProfileRecorder(tmp_path, 'worker-0', 10)
    backend = Backend()
    recorder.wrap(backend, 'run_batch', 'backend.run_batch', items=True, category='backend_wall')
    items = [SimpleNamespace(request_slot=3, request_epoch=7, round_id=2)]
    with ThreadPoolExecutor(1) as executor:
        assert executor.submit(backend.run_batch, items).result() == 1
    assert recorder.events[0]['keys'] == [[3,7,2]]
    assert recorder.keys.get() == ()


def test_light_aggregates_short_calls_but_keeps_slow_and_failed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from nebulasd.observability import profiling
    ticks = iter((0, 10, 20, 200020, 300000, 300010))
    monkeypatch.setattr(profiling, 'perf_counter_ns', lambda: next(ticks))
    recorder = ProfileRecorder(tmp_path, 'engine', 10, mode='light')
    task = SimpleNamespace(refresh=lambda: 7)
    recorder.wrap(task, 'refresh', 'engine.ledger')
    assert task.refresh() == task.refresh() == 7
    def fail():
        raise ValueError('preserved')
    task.fail = fail
    recorder.wrap(task, 'fail', 'engine.classify')
    import pytest
    with pytest.raises(ValueError, match='preserved'):
        task.fail()
    recorder.flush()
    report = json.loads((tmp_path/'engine.json').read_text())
    assert report['short_calls']['engine.ledger'][:2] == [1, 10]
    assert len(report['events']) == 2
    assert report['events'][-1]['failed']
    assert recorder.keys.get() == ()
