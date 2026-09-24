from dataclasses import replace
import pytest
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.table.storage import RequestSchedulingTable
from nebulasd.data.shared_arenas import SharedTokenArena, ArenaRouter
from types import SimpleNamespace
from nebulasd.workers.completion import CompletionArena
from nebulasd.workers.target.publication import TargetPublisher
from nebulasd.workers.work import Outcome
from test_autonomous_work import work


@pytest.fixture
def publisher():
    tokens = SharedTokenArena(65536, generation=2, writer=True)
    completions = CompletionArena(4096)
    table = RequestSchedulingTable(1)
    owner = TargetPublisher(table, completions, block_bytes=1024, outputs=ArenaRouter((tokens,),tokens), configs=SimpleNamespace(read_config=lambda h: SimpleNamespace(all_stop_token_ids=())))
    try:
        yield owner
    finally:
        tokens.close()
        tokens.segment.unlink()
        completions.close(unlink=True)


def test_physical_before_result_does_not_publish_host_ready(publisher):
    w = allocated(publisher, work())
    publisher.reserve(w)
    publisher.consume(('PHYSICAL', 1, dict(outcomes=[1], observed_ns=80, d2h_submitted_ns=60)))
    assert publisher.step() == ()
    assert publisher.completions.read(0) is None
    assert publisher.table.partition(K.REQUEST_D2H).read_publish_seq(0) == (1 << 64)-1
    publisher.consume(('RESULT', 1, dict(compute_start_ns=10,compute_end_ns=20,rows=[dict(index=0, tokens=[42], accepted=0,
        logical=6, version=1, dirty_begin=0, dirty_blocks=1)])))
    assert publisher.step(3) == (1,)
    ready = publisher.table.partition(K.REQUEST_D2H).read_stable(0)
    result = publisher.table.partition(K.REQUEST_TARGET_COMPUTE).read_stable(0)
    assert ready.get('copy_bytes') == 2048
    assert ready.get('copy_start_time_ns') == 60
    assert publisher.outputs.writer.read_tokens(result.get('committed_delta_handle')) == (42,)
    assert publisher.completions.read(0).physical_done_ns == 80


def test_shutdown_completion_does_not_invent_compute_or_host_ready(publisher):
    publisher.reserve(work())
    publisher.consume(('PHYSICAL', 1, dict(outcomes=[3], observed_ns=80, d2h_submitted_ns=0)))
    assert publisher.step() == (1,)
    assert publisher.completions.read(0).members[0].outcome == Outcome.SKIPPED_SHUTDOWN
    for kind in (K.REQUEST_TARGET_COMPUTE, K.REQUEST_D2H):
        assert publisher.table.partition(kind).read_publish_seq(0) == (1 << 64)-1


def test_result_budget_rejected_before_payload_allocation(publisher):
    before = publisher.outputs.writer._head
    with pytest.raises(ValueError, match='budget'):
        publisher.reserve(replace(work(), result_bytes=1))
    assert not publisher.records and publisher.outputs.writer._head == before


def allocated(pub, w):
    return replace(w, rows=tuple(replace(r, prompt_count=6,
        output=pub.outputs.writer.reserve_output(r.max_new_tokens)) for r in w.rows))


def test_registration_does_not_allocate_second_output_area(publisher):
    w = allocated(publisher, work())
    before = publisher.outputs.writer._head
    publisher.reserve(w)
    assert publisher.outputs.writer._head == before


@pytest.mark.parametrize('native', [False, True])
def test_event_budget_fairness_prepared_fields_and_coalesced_wakeup(native):
    from nebulasd.table.native_storage import request_table, close_table_partitions
    table = request_table(11) if native else RequestSchedulingTable(11)
    class Bell:
        calls = 0
        def ring(self):
            self.calls += 1
    bell = Bell()
    for kind in (K.REQUEST_H2D,K.REQUEST_TARGET_COMPUTE,K.REQUEST_D2H):
        table.partition(kind)._doorbell = bell
    tokens = SharedTokenArena(8192,generation=19,writer=True)
    completions = CompletionArena(8192)
    pub = TargetPublisher(table,completions,block_bytes=256, outputs=ArenaRouter((tokens,),tokens), configs=SimpleNamespace(read_config=lambda h: SimpleNamespace(all_stop_token_ids=())))
    a = work()
    a = replace(a,result_bytes=32768,rows=tuple(replace(a.rows[0],slot=i,destination_offset=8*i) for i in range(10)))
    b = replace(work(2,1),completion_offset=1024,rows=(replace(work().rows[0],slot=10),))
    try:
        for w in (a,b):
            w = allocated(pub, w)
            pub.reserve(w)
            pub.consume(('PHYSICAL',w.work_seq,dict(outcomes=[1]*len(w.rows),observed_ns=88,d2h_submitted_ns=77)))
            pub.consume(('RESULT',w.work_seq,dict(compute_start_ns=10,compute_end_ns=20,rows=[dict(index=i,tokens=(100+r.slot,),accepted=0,
                logical=6,version=9,dirty_begin=0,dirty_blocks=1) for i,r in enumerate(w.rows)])))
        assert pub.step(event_budget=2) == ()
        assert pub.last_published_rows == 11 and bell.calls == 1
        part = table.partition(K.REQUEST_TARGET_COMPUTE)
        assert part.read_stable(0).get('target_kv_version') == 9
        assert part.read_stable(10).get('last_committed_token') == 110
        assert part.read_stable(1).get('target_kv_version') == 9
        retired=[]
        while pub.records:
            retired.extend(pub.step(event_budget=2))
            assert pub.last_published_rows in (0, 11)
        assert retired == [1,2]  # Both WORKs get a turn after each complete event.
        assert completions.read(0).physical_done_ns == 88
        for slot in range(11):
            result=part.read_stable(slot)
            assert tokens.read_tokens(result.get('committed_delta_handle')) == (100+slot,)
            host=table.partition(K.REQUEST_D2H).read_stable(slot)
            assert host.get('copy_bytes') == 512 and host.get('copy_start_time_ns') == 77
    finally:
        tokens.close(); tokens.segment.unlink(); completions.close(unlink=True)
        if native:
            close_table_partitions(table._partitions,unlink=True)
