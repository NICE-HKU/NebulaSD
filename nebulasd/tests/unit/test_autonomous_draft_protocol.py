import pytest
from nebulasd.workers.channel import encode,decode,LocalChannel
from nebulasd.workers.work import Work
from test_autonomous_draft_publication import draft_work


def test_draft_work_identity_roundtrip():
    w=draft_work()
    assert Work.from_bytes(w.to_bytes())==w


def test_draft_wire_actual_length_and_truncation():
    data=dict(compute_start_ns=4,compute_end_ns=8,rows=[dict(index=3,proposal_kind=1,
        logical=16,version=9,dirty_begin=0,dirty_blocks=1,committed_count=7,proposal=(40,41))])
    raw=encode('DRAFT_RESULT',data)
    assert decode('DRAFT_RESULT',raw)==data
    for n in range(len(raw)):
        with pytest.raises(ValueError):
            decode('DRAFT_RESULT',raw[:n])
    channel=LocalChannel()
    try:
        channel.put_nowait(('DRAFT_RESULT',17,data))
        assert channel.get_nowait()==('DRAFT_RESULT',17,data)
        assert channel.bytes_sent==16+len(raw)
    finally:
        channel.close(unlink=True)

from contextlib import ExitStack
from dataclasses import replace
from queue import Empty,Full
from time import sleep,monotonic,perf_counter_ns
from types import SimpleNamespace as NS
from nebulasd.workers.draft.process import DraftProcesses
from nebulasd.workers.resources import PayloadDescriptor
from nebulasd.workers.completion import CompletionArena
from nebulasd.data.shared_arenas import SharedProposalArena
from nebulasd.data.draft_snapshot_arena import SharedDraftSnapshotArena
from nebulasd.table.native_storage import request_table,table_descriptors,close_table_partitions
from nebulasd.core.enums import StateChangeBlockKind as K


def draft_protocol_execution(options,commands,results,wake,ready,result_wake):
    with ExitStack() as stack:
        commands,results=LocalChannel(commands),LocalChannel(results)
        stack.callback(commands.close);stack.callback(results.close)
        ready.set()
        pending={}
        stopping=False
        while True:
            try:
                message=commands.get_nowait()
                kind,payload=message[:2]
            except Empty:
                wake.wait(.001);wake.clear()
                continue
            if kind=='WORK':
                w=Work.from_bytes(payload)
                pending[w.work_seq]=w
                if not options.get('hold'):
                    now=perf_counter_ns()
                    results.put_nowait(('DRAFT_RESULT',w.work_seq,dict(compute_start_ns=now,compute_end_ns=now,
                        rows=[dict(index=i,proposal=(101,),proposal_kind=1,logical=r.prompt_count+1,
                            version=1,dirty_begin=0,dirty_blocks=1,committed_count=1) for i,r in enumerate(w.rows)])))
                    results.put_nowait(('PHYSICAL',w.work_seq,dict(outcomes=(1,)*len(w.rows),d2h_submitted_ns=now,
                        observed_ns=now,facts=())))
                    result_wake.set()
            elif kind=='RETIRE_RECORD':
                del pending[payload]
            elif kind in ('DRAIN','SHUTDOWN'):
                stopping=True
                if options.get('hold'):
                    for seq,w in pending.items():
                        results.put_nowait(('PHYSICAL',seq,dict(outcomes=(3,)*len(w.rows),d2h_submitted_ns=0,
                            observed_ns=perf_counter_ns(),facts=())))
                        result_wake.set()
            if stopping and not pending:
                return


@pytest.fixture
def draft_resources():
    with ExitStack() as stack:
        table=request_table(5)
        stack.callback(close_table_partitions,table._partitions,unlink=True)
        proposals=SharedProposalArena(8192,generation=51)
        snapshots=SharedDraftSnapshotArena(8192,generation=61)
        for arena in (proposals,snapshots):
            stack.callback(arena.segment.unlink);stack.callback(arena.close)
        completion=CompletionArena(8192)
        stack.callback(completion.close,unlink=True)
        opts=dict(configs=(),tokens=(),slots=5,table=table_descriptors(table),result_proposals=PayloadDescriptor.of(proposals),
            result_snapshots=PayloadDescriptor.of(snapshots),completions=completion.segment.descriptor,
            blocks_per_bank=16,capacity_rows=4,host=NS(block_bytes=1024,descriptor_generation=13))
        yield opts,table,proposals,completion


def test_formal_draft_pids_budget_backpressure_and_drain(draft_resources):
    opts,table,proposals,completions=draft_resources
    with DraftProcesses(opts|dict(diagnostic_publication_delay_s=.1,publication_event_budget=1),
            execution_entry=draft_protocol_execution) as group:
        assert len(set(group.pids.values()))==3
        works=[]
        for i in range(5):
            w=draft_work()
            r=replace(w.rows[0],slot=i,
                predecessor=replace(w.rows[0].predecessor,slot=i),classified=replace(w.rows[0].classified,slot=i))
            works.append(replace(w,work_seq=i+1,rows=(r,),completion_offset=completions.reserve(1)))
        for w in works[:4]:
            group.submit(w)
        with pytest.raises(Full):
            group.submit(works[4])
        done=[]
        deadline=monotonic()+5
        while len(done)<5 and monotonic()<deadline:
            for w in works:
                if w.work_seq in done:
                    continue
                if completions.read(w.completion_offset) is not None:
                    group.retire(w.work_seq)
                    done.append(w.work_seq)
                    if len(done)==1:
                        group.submit(works[4]);group.stop()
            sleep(.001)
        assert done==[1,2,3,4,5]
        group.join()
        assert set(group.exitcodes.values())=={0}
        for i in range(5):
            fact=table.partition(K.REQUEST_DRAFT).read_stable(i)
            assert proposals.read_proposal(fact.get('proposal_handle')).draft_token_ids==(101,)
        assert not proposals._writer and proposals._head==0


@pytest.mark.parametrize('victim',['control','execution'])
def test_draft_failure_cleans_both_processes(draft_resources,victim):
    opts,*_=draft_resources
    group=DraftProcesses(opts,execution_entry=draft_protocol_execution)
    try:
        group.wait_ready(5)
        getattr(group,victim).kill()
        deadline=monotonic()+5
        while group.monitor.is_alive() and monotonic()<deadline:
            sleep(.01)
        with pytest.raises(RuntimeError):
            group.check()
        group.close(timeout=0)
        assert group.channels==[] and group._draft_resources_closed
        assert len(group.exitcodes)==2 and all(v is not None for v in group.exitcodes.values())
    finally:
        group.close(timeout=0)


def test_draft_shutdown_without_produced_input(draft_resources):
    opts,table,_,completions=draft_resources
    with DraftProcesses(opts|dict(hold=True),execution_entry=draft_protocol_execution) as group:
        group.submit(draft_work())
        group.stop(shutdown=True)
        from test_autonomous_target_control import await_completion
        await_completion(group,completions)
        assert completions.read(0).members[0].outcome==3
        assert table.partition(K.REQUEST_DRAFT).read_publish_seq(0)==(1<<64)-1
        group.join()


def test_shutdown_passes_full_credit_waiting_work(draft_resources):
    opts,_,_,completions=draft_resources
    with DraftProcesses(opts|dict(hold=True),execution_entry=draft_protocol_execution) as group:
        for i in range(4):
            w=draft_work()
            r=w.rows[0]
            r=replace(r,slot=i,predecessor=replace(r.predecessor,slot=i),classified=replace(r.classified,slot=i))
            group.submit(replace(w,work_seq=i+1,rows=(r,),completion_offset=completions.reserve(1)))
        group.stop(shutdown=True)
        from test_autonomous_target_control import await_completion
        deadline=monotonic()+5
        while len(group.pending_work) and monotonic()<deadline:
            for i in tuple(group.pending_work):
                from nebulasd.workers.completion import completion_bytes
                if completions.read((i-1)*completion_bytes(1)) is not None:group.retire(i)
            sleep(.001)
        assert not group.pending_work
        group.join()
        assert set(group.exitcodes.values())=={0}
