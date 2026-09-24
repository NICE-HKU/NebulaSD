"""Bounded observation, local authority and delivery/retirement ordering."""
from types import SimpleNamespace
import pytest
from test_autonomous_engine import engine, publish_target
from nebulasd.core.enums import StateChangeBlockKind as K, Lifecycle
from nebulasd.core.ids import U64
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from nebulasd.ipc.state_change_ring import StateChangeRing, StateChangeEntry
from nebulasd.table.storage import RequestSchedulingTable, StableReadConflict
from nebulasd.table.reader import IncrementalTableReader
from nebulasd.table.writers import EngineTableWriter
from nebulasd.engine.client import TokenEngine


def publish(table, slot, seq=1):
    EngineTableWriter(table).publish_active(slot=slot,publish_seq=seq,request_epoch=1,
        current_round_id=seq,arrival_seq=slot,prompt_token_count=3,max_new_tokens=8,spec_token_limit=4)


def test_bounded_reader_recovers_overflow_without_unbounded_scan():
    ring=StateChangeRing(2);table=RequestSchedulingTable(12,ring=ring)
    for slot in range(12):publish(table,slot)
    reader=IncrementalTableReader(request_table=table,rings=(ring,));reader.max_entries=3
    seen={}
    for _ in range(500):
        batch=reader.poll()
        assert batch.scanned_rows<=3 and len(batch.views)<=3
        seen.update({(r.block_kind,r.row):r for r in batch.views})
        if not reader.has_pending():break
    else:pytest.fail('overflow recovery did not converge')
    assert all(seen[K.REQUEST_ENGINE,i].get('request_epoch')==1 for i in range(12))


def test_ring_rotation_services_quiet_worker_under_hot_producer():
    hot,cold=StateChangeRing(32),StateChangeRing(32)
    table=RequestSchedulingTable(8)
    for slot in range(8):publish(table,slot)
    cold.push(StateChangeEntry(K.REQUEST_ENGINE,7,1))
    reader=IncrementalTableReader(request_table=table,rings=(hot,cold));reader.max_entries=1
    seen=set()
    for _ in range(6):
        hot.push(StateChangeEntry(K.REQUEST_ENGINE,0,1))
        seen.update(r.row for r in reader.poll().views)
    assert 7 in seen


def test_conflicting_stable_read_does_not_starve_other_rows(monkeypatch):
    ring=StateChangeRing(16);table=RequestSchedulingTable(4,ring=ring)
    for slot in range(4):publish(table,slot)
    reader=IncrementalTableReader(request_table=table,rings=(ring,));reader.max_entries=2
    original=reader._read_entry
    def read(entry):
        if entry.row==0:raise StableReadConflict('writer is publishing')
        return original(entry)
    monkeypatch.setattr(reader,'_read_entry',read)
    seen=set()
    for _ in range(5):seen.update(r.row for r in reader.poll().views)
    assert seen=={1,2,3} and reader.has_pending()
    monkeypatch.setattr(reader,'_read_entry',original)
    assert [r.row for r in reader.poll().views]==[0]
    assert not reader.has_pending()


def test_second_overflow_revisits_already_scanned_rows():
    ring=StateChangeRing(1);table=RequestSchedulingTable(4,ring=ring)
    for slot in range(4):publish(table,slot)
    reader=IncrementalTableReader(request_table=table,rings=(ring,));reader.max_entries=2
    reader.poll();reader.poll()
    publish(table,3,2)
    publish(table,0,2)
    for _ in range(500):
        reader.poll()
        if not reader.has_pending():break
    assert reader.cached_view(K.REQUEST_ENGINE,0).publish_seq==2


def test_local_dispatch_never_requires_shared_readback(engine):
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    assert e.rows[K.REQUEST_DISPATCH,0].get('target_run_seq')==1
    assert e.resources.table.partition(K.REQUEST_DISPATCH).read_publish_seq(0)==U64.invalid
    work=e.supervisor.pairs[1].work[0];publish_target(e,work)
    shared_seq=e.resources.table.partition(K.REQUEST_ENGINE).read_publish_seq(0)
    e.step()
    assert e.candidates[0].output_count==1 and e.registry.records[0].output==(4,)
    assert e.resources.table.partition(K.REQUEST_ENGINE).read_publish_seq(0)==shared_seq
    e.reader._rescan=True
    for _ in range(20):e._observe_facts()
    assert e.candidates[0].output_count==1
    assert (K.REQUEST_ENGINE,0) not in e.rows
    assert e._scheduling_view().request_states[0].current_round==0


def test_output_delivery_follows_dispatch_but_precedes_poll_return(engine):
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    publish_target(e,e.supervisor.pairs[1].work[0])
    delivered=[]
    def callback(identity,tokens,lifecycle):
        assert e.supervisor.pairs[0].work
        assert e.registry.records[0].output==tokens
        delivered.append(tokens)
    e.outputs.on_tokens=callback;e.step()
    assert delivered==[(4,)] and not e.outputs.pending


def test_terminal_output_survives_registry_retirement(engine):
    e=engine;client=TokenEngine(SimpleNamespace(max_proposal_depth=4),e)
    h=client.submit((1,2,3),DraftGenerationConfig(8,4,stop_token_ids=(4,)))
    e.step();publish_target(e,e.supervisor.pairs[1].work[0])
    def retire():
        if e.registry.records[0].lifecycle==Lifecycle.FINISHED:
            e.registry.records.clear();return True
        return False
    e.recycler.progress=retire
    client.poll()
    assert client.result(h)==(4,)
    events=client.read(h)
    assert len(events)==1 and events[0].token_ids==(4,)
    assert not e.supervisor.pairs[0].work


def test_completion_scan_handles_physical_then_output(engine):
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    work=e.supervisor.pairs[1].work[0]
    publish_target(e,work)
    e.ledger.observe_completions();e.ledger.refresh(e.rows)
    assert e.ledger.records
    e._observe_facts();e.ledger.observe_completions();e.ledger.refresh(e.rows)
    assert not e.ledger.records and not e.ledger.by_worker[1]
    e.outputs.flush()
    assert e.registry.records[0].output==(4,)


def test_empty_observation_still_dispatches_local_admission(engine):
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4))
    e._observe_facts()
    e.step()
    assert len(e.supervisor.pairs[1].work)==1


def test_cohort_reset_discards_stale_hints_and_refreshes_worker_generation(engine):
    from nebulasd.table.prepared import PreparedRow
    e=engine
    e._observe_facts()
    e.admit('old',(1,2,3),DraftGenerationConfig(1,1))
    from nebulasd.workers.work import WorkKind
    old_work = SimpleNamespace(operation=WorkKind.TARGET_PREFILL,
        rows=(SimpleNamespace(slot=0, epoch=1, round_id=0),))
    e.scheduler.observe_compute_result(old_work,
        dict(rows=[dict(index=0)], compute_start_ns=10, compute_end_ns=20))
    e.registry.records[0].lifecycle=Lifecycle.FINISHED
    e._request_changed(e.registry.records[0])
    e.scheduling_progress.static_rows[(True,0,1)]='old allocation'
    e.supervisor.stop_workers=lambda:None
    def start():
        for w in e.resources.specs:
            partition=e.resources.registry.partition(K.WORKER_COMMON)
            PreparedRow(partition,w.worker_id,dict(worker_id=w.worker_id,
                worker_generation=w.generation,status=1,role=w.role),()).publish(())
    e.supervisor.start=start
    assert e.recycler.progress() and e.recycler.busy
    assert e.recycler.progress() and not e.recycler.busy
    assert not e.reader._pending and not e.reader._scan
    assert not e.scheduling_progress.static_rows
    assert e.scheduler.compute_times.end_ns(0, 1, 'T', 0) is None
    assert all(w.generation==2 for w in e.scheduling_progress.workers.values())
    e.admit('new',(1,2,3),DraftGenerationConfig(8,4));e.step()
    work=e.supervisor.pairs[1].work[-1]
    assert work.worker_generation==2 and work.rows[0].epoch==2
    assert set(e.scheduling_progress.static_rows)=={(False,0,2)}
    publish_target(e, work)
    e._observe_facts()
    assert e.scheduler.compute_times.end_ns(0, 2, 'T', 0) == 20



def test_failed_output_callback_poison_stops_engine(engine):
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    publish_target(e,e.supervisor.pairs[1].work[0])
    def fail(*args):raise RuntimeError('callback failed')
    e.outputs.on_tokens=fail
    with pytest.raises(RuntimeError,match='callback failed'):e.step()
    assert e._poisoned


def test_last_phase_ledger_event_keeps_owner_running(engine,monkeypatch):
    from nebulasd.workers.work import WorkKind
    e=engine;e._observe_facts()
    progress=e.scheduling_progress
    progress.dirty={'D':set(),'T':set()}
    e._retry_dispatch=False
    calls=[]
    def observe():
        calls.append(1)
        if len(calls)==2:
            e.ledger.scheduling_events.append(('compute',SimpleNamespace(
                operation=WorkKind.TARGET_VERIFY,worker_id=1)))
        return False
    monkeypatch.setattr(e,'_observe_facts',observe)
    progress.advance()
    assert e._retry_dispatch and e.ledger.scheduling_events


def test_local_fields_remain_immutable_without_payload(engine):
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    before=e.rows[K.REQUEST_DISPATCH,0]
    assert before.payload is None
    e._publish_local(K.REQUEST_DISPATCH,0,dict(target_run_seq=27))
    after=e.rows[K.REQUEST_DISPATCH,0]
    assert before.get('target_run_seq')==1 and after.get('target_run_seq')==27
    assert after.payload is None


def test_native_payload_snapshot_matches_eager_fields_and_survives_buffer_reuse():
    from nebulasd.table.native_storage import NativeTablePartition
    from nebulasd.engine.local_state import SchedulingSnapshotReader
    from nebulasd.core.handles import ArenaHandle
    from nebulasd.table.storage import FieldValue
    from nebulasd.table.storage import WorkerSchedulingRegistry
    references = (RequestSchedulingTable(2), WorkerSchedulingRegistry(2))
    read = SchedulingSnapshotReader()
    for old in [p for table in references for p in table._partitions.values()]:
        p = NativeTablePartition(layout=old.layout, block_kind=old.block_kind, capacity_rows=2)
        try:
            fields = tuple(FieldValue(name, ArenaHandle.null() if f.type.name == 'arena_handle' else 0)
                           for name,f in p._field_by_name.items() if name != 'publish_seq')
            p._publish(0, 1, fields)
            lazy = read(p, 0)
            assert not lazy._values
            eager = p.read_stable(0)
            assert lazy.payload == eager.payload
            assert tuple(lazy.fields) == eager.fields
            untouched = read(p, 0)
            changed = tuple(FieldValue(name, ArenaHandle(256, 16, 3) if f.type.name == "arena_handle"
                            else -123 if f.type.signed else 123)
                            for name,f in p._field_by_name.items() if name != "publish_seq")
            p._publish(0, 2, changed)
            newer = read(p, 0)
            assert tuple(newer.fields) == p.read_stable(0).fields
            assert tuple(untouched.fields) == eager.fields
            assert newer.publish_seq == 2 and lazy.publish_seq == 1
            assert lazy.payload == eager.payload
            with pytest.raises(KeyError):lazy.get('missing_field')
            with pytest.raises(StableReadConflict):read(p, 1)
        finally:
            p.segment.close()
            p.segment.unlink()


def test_priority_snapshot_bypasses_backlog_and_decodes_once(monkeypatch):
    ring = StateChangeRing(32)
    table = RequestSchedulingTable(8, ring=ring)
    for slot in range(8):publish(table, slot)
    reader = IncrementalTableReader(request_table=table, rings=(ring,))
    reader.max_entries = 1
    reader.priority_rows = ((K.REQUEST_ENGINE, 7),)
    reads = []
    original = reader.read_snapshot
    def read(partition, row):
        reads.append((partition.block_kind, row))
        return original(partition, row)
    reader.read_snapshot = read
    first = reader.poll()
    assert 7 in [row.row for row in first.views]
    for _ in range(10):reader.poll()
    assert reads.count((K.REQUEST_ENGINE, 7)) == 1
    publish(table, 7, 2)
    assert reader.poll().views[0].publish_seq == 2
    for _ in range(10):reader.poll()
    assert reads.count((K.REQUEST_ENGINE, 7)) == 2


def test_result_fact_does_not_confirm_physical_retirement(engine):
    from test_work_publications import publisher, result
    e=engine;e.admit('a',(1,2,3),DraftGenerationConfig(8,4));e.step()
    w=e.supervisor.pairs[1].work[0];p=publisher(e);result(p,w);p.publish_results(w.work_seq)
    e._observe_facts();e.ledger.observe_completions();e.ledger.refresh(e.rows)
    record=e.ledger.records[e.ledger.key(w)]
    assert not record.physical_done and record.completion is None
    assert record.applied == {0}
    assert e.scheduler.compute_times.end_ns(0,1,'T',0) == 20


def test_result_batch_updates_all_candidates_before_scheduling(engine, monkeypatch):
    e = engine
    for name in ('a', 'b'):e.admit(name, (1, 2, 3), DraftGenerationConfig(8, 4))
    e.step()
    work = e.supervisor.pairs[1].work[0]
    assert len(work.rows) == 2
    publish_target(e, work)
    batches = []
    original = e._request_changed
    def publish_many(record):
        batches.append(record.input.slot)
        return original(record)
    monkeypatch.setattr(e, '_request_changed', publish_many)
    schedule = e.scheduler.schedule
    def check(view, **kwargs):
        assert all(view.requests[r.slot].output_count == 1 for r in work.rows)
        return schedule(view, **kwargs)
    monkeypatch.setattr(e.scheduler, 'schedule', check)
    e.step()
    assert batches == [r.slot for r in work.rows]
