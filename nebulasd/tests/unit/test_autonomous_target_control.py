"""Real spawned control/lifetime tests; execution here tests protocol, not CUDA."""
from contextlib import ExitStack
from dataclasses import replace
from queue import Empty
from time import monotonic, perf_counter_ns, sleep
import multiprocessing as mp
import os
import pytest

from nebulasd.workers.channel import LocalChannel
from nebulasd.workers.work import Work
from nebulasd.workers.target.service import TargetProcesses
from nebulasd.workers.resources import PayloadDescriptor
from nebulasd.workers.completion import CompletionArena
from nebulasd.data.shared_arenas import SharedTokenArena, SharedConfigArena
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from nebulasd.table.native_storage import request_table, table_descriptors, close_table_partitions
from nebulasd.core.enums import StateChangeBlockKind as K
from test_autonomous_work import work


def protocol_execution(options, commands, results, wake, ready, result_wake):
    if options.get('stall_execution_start'):
        while True:
            sleep(1)
    if options.get('fail_execution_start'):
        raise RuntimeError('injected execution startup failure')
    with ExitStack() as stack:
        commands,results = LocalChannel(commands),LocalChannel(results)
        stack.callback(commands.close);stack.callback(results.close)
        pending = {}
        stopping = False
        ready.set()
        while True:
            try:
                message=commands.get_nowait()
                kind,data=message[:2]
            except Empty:
                wake.wait(.001);wake.clear()
                continue
            if kind=='WORK':
                w=Work.from_bytes(data)
                pending[w.work_seq]=w
                if not options.get('hold_work'):
                    now=perf_counter_ns()
                    results.put_nowait(('RESULT',w.work_seq,dict(compute_start_ns=now,compute_end_ns=now,
                        rows=[dict(index=i,tokens=(100+r.slot,),accepted=0,logical=6,version=1,
                                   dirty_begin=0,dirty_blocks=1) for i,r in enumerate(w.rows)])))
                    results.put_nowait(('PHYSICAL',w.work_seq,dict(outcomes=(1,)*len(w.rows),
                        d2h_submitted_ns=now,observed_ns=perf_counter_ns(),facts=[])))
                    result_wake.set()
            elif kind=='RETIRE_RECORD':
                del pending[data]
            elif kind in ('DRAIN','SHUTDOWN'):
                stopping=True
                if kind=='SHUTDOWN' and options.get('hold_work'):
                    for seq,w in pending.items():
                        results.put_nowait(('PHYSICAL',seq,dict(outcomes=(3,)*len(w.rows),
                            d2h_submitted_ns=0,observed_ns=perf_counter_ns(),facts=[])))
                        result_wake.set()
            if stopping and not pending:
                return


@pytest.fixture
def resources():
    table=request_table(2)
    tokens=SharedTokenArena(8192,generation=71,writer=True)
    configs=SharedConfigArena(8192,writer=True)
    config_handle=configs.write_config(DraftGenerationConfig(16,4))
    tokens.reserve_output(16)
    completions=CompletionArena(8192)
    from types import SimpleNamespace
    options=dict(configs=(PayloadDescriptor.of(configs),),tokens=(PayloadDescriptor.of(tokens),),slots=2,table=table_descriptors(table),result_tokens=PayloadDescriptor.of(tokens),
                 completions=completions.segment.descriptor,host=SimpleNamespace(block_bytes=1024),test_config_handle=config_handle)
    try:
        yield options,table,tokens,completions
    finally:
        close_table_partitions(table._partitions,unlink=True)
        tokens.close();tokens.segment.unlink();completions.close(unlink=True)
        configs.close();configs.segment.unlink()


def allocated_work(resources):
    from nebulasd.core.handles import ArenaHandle
    options, table, tokens, completions = resources
    w=work()
    return replace(w, rows=(replace(w.rows[0], prompt_count=6,
        output=ArenaHandle(0,64,71), config=options['test_config_handle']),))


def await_completion(group, completions, timeout=5):
    deadline=monotonic()+timeout
    while monotonic()<deadline:
        group.check()
        completed=completions.read(0)
        if completed is not None:
            group.retire(completed.work_seq)
            return completed
        sleep(.001)
    raise TimeoutError('control did not publish WORK completion')


def test_control_owns_publication_and_private_credit_drain(resources):
    options,table,tokens,completions=resources
    with TargetProcesses(options,execution_entry=protocol_execution) as group:
        assert len(set(group.pids.values()))==3
        group.submit(allocated_work(resources))
        await_completion(group,completions)
        row=table.partition(K.REQUEST_TARGET_COMPUTE).read_stable(0)
        assert tokens.read_tokens(row.get('committed_delta_handle'))==(100,)
        assert completions.read(0).members[0].outcome==1
        assert tokens._head==64  # Worker appends only inside the reserved range.
        group.stop();group.join()
        assert [p.exitcode for p in group.processes]==[0,0]


def test_shutdown_retires_unproduced_work_without_fake_result(resources):
    options,table,tokens,completions=resources
    with TargetProcesses(options|{'hold_work':True},execution_entry=protocol_execution) as group:
        group.submit(allocated_work(resources))
        group.stop(shutdown=True)
        await_completion(group,completions)
        assert completions.read(0).members[0].outcome==3
        assert table.partition(K.REQUEST_TARGET_COMPUTE).read_publish_seq(0)==(1<<64)-1
        group.join()


@pytest.mark.parametrize('failing', ['execution_start','control_start','execution_live','control_live'])
def test_sibling_failure_is_joined_and_channels_unlinked(resources,failing):
    options,*_=resources
    if failing=='execution_start':
        options=options|{'fail_execution_start':True}
    elif failing=='control_start':
        options=options|{'tokens':None}
    group=TargetProcesses(options,execution_entry=protocol_execution)
    names=[segment.descriptor.name for channel in group.channels for segment in (channel.ring.segment,channel.payload)]
    try:
        if failing.endswith('start'):
            with pytest.raises(RuntimeError):
                group.wait_ready(5)
        else:
            group.wait_ready(5)
            victim=group.execution if failing=='execution_live' else group.control
            victim.kill()
            deadline=monotonic()+5
            while group.monitor.is_alive() and monotonic()<deadline:
                sleep(.01)
            with pytest.raises(RuntimeError):
                group.check()
        group.close(timeout=0)
        assert len(group.exitcodes)==2 and all(v is not None for v in group.exitcodes.values())
        assert all(p._closed for p in group.processes)
        assert group.channels==[]
        from multiprocessing.shared_memory import SharedMemory
        for name in names:
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=name)
    finally:
        group.close(timeout=0)


def test_startup_timeout_kills_both_children(resources):
    options,*_=resources
    group=TargetProcesses(options|{'stall_execution_start':True},execution_entry=protocol_execution)
    with pytest.raises(TimeoutError):
        group.wait_ready(.5)
    assert group.closed and all(p._closed for p in group.processes)


def test_close_drains_unconsumed_results_before_unmapping(resources):
    options,table,tokens,completions=resources
    group=TargetProcesses(options|{'diagnostic_publication_delay_s':.05},execution_entry=protocol_execution)
    group.wait_ready(5)
    group.submit(allocated_work(resources))
    group.close(timeout=5)  # Engine abandons consumption, global production has stopped.
    assert set(group.exitcodes.values())=={0}
    assert completions.read(0) is not None
    assert all(p._closed for p in group.processes)
