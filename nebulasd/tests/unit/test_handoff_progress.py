"""Handoff progress preserves single submission and physical retirement fences."""
from concurrent.futures import Future
from threading import Thread
from types import SimpleNamespace as NS
import pytest
from nebulasd.workers.banks import Banks
from nebulasd.workers.runtime import Runtime
from nebulasd.workers.isolated import InlineJobs
from test_autonomous_work import work

class Role:
    input_pool = InlineJobs()
    def __init__(self):
        self.metadata = Future()
        self.result = Future()
        self.export = Future()
        self.executions = []
    def compile_import(self, *args):
        return self.input_pool.submit(lambda: NS(copy_plan=lambda _: None))
    def compile_compute(self, spec, *args):
        return self.input_pool.submit(lambda: spec.work_seq)
    def write_metadata(self, *args):return self.metadata
    def execute(self, plan):
        self.executions.append(plan)
        return self.result
    def submit(self, bank, plan):return self.export

@pytest.mark.parametrize('metadata_before', [False, True])
def test_inline_progress_and_concurrent_completion_are_exactly_once(metadata_before):
    role=Role();runtime=Runtime(Banks(16, 1),role,role,profile=True)
    assert runtime.accept(work())
    if metadata_before:role.metadata.set_result(object())
    runtime.step()
    if not metadata_before:
        t=Thread(target=lambda: role.metadata.set_result(object()));t.start();t.join()
    runtime.step()
    assert role.executions==[1]
    assert runtime.records[1].plan==1
    assert not {'IMPORT_INPUT','COMPILE'} & runtime.records[1].jobs.keys()
    for _ in range(20):runtime.step()
    assert role.executions==[1]
    t=Thread(target=lambda: role.result.set_result(NS(executed_rows=(0,),export_plan='D2H')))
    t.start();t.join();runtime.step()
    assert not runtime.records[1].physical_done
    assert not runtime.banks.free_rows
    with pytest.raises(RuntimeError,match='physical completion'):runtime.publication_retired(1)
    role.export.set_result(NS(submitted_ns=1,completed_ns=2));runtime.step()
    assert runtime.records[1].physical_done and runtime.banks.free_rows=={0}
    runtime.publication_retired(1)
    assert not runtime.records


def test_inline_exception_stops_owner_without_releasing_bank():
    role=Role()
    def bad(*args):raise ValueError('invalid dependency result')
    role.compile_compute=bad
    runtime=Runtime(Banks(16,1),role,role)
    runtime.accept(work())
    with pytest.raises(ValueError,match='invalid dependency'):runtime.step()
    assert not runtime.banks.free_rows and not role.executions
    with pytest.raises(RuntimeError,match='failed'):runtime.step()

@pytest.mark.parametrize('terminal,budget', [(False,64), (True,64), (False,3)])
def test_incremental_plan_fills_each_member_once_and_handles_terminal_last_member(terminal,budget):
    from dataclasses import replace
    from nebulasd.workers.target.inputs import Plan,Input
    from nebulasd.workers.input_facts import InputFact
    from nebulasd.workers.work import Selector
    from nebulasd.core.enums import Lifecycle
    from test_autonomous_runtime import successor
    role=Role();role.incremental_inputs=True;role.max_batch_tokens=budget;built=[]
    def compile(spec,captured,layout,live):
        built.append(live)
        return Plan(spec,tuple(Input(i,layout.rows[i],4,1,(),9,(10,),'linear',8,(),None,0,8) for i in live))
    role.compile_compute=compile;role.metadata.set_result(object())
    base=successor(0);row=base.rows[0]
    second=replace(row,slot=1,destination_offset=8,source=replace(row.source,slot=1),
        predecessor=replace(row.predecessor,slot=1),classified=replace(row.classified,slot=1))
    spec=replace(base,rows=(row,second));runtime=Runtime(Banks(32,2),role,role)
    assert runtime.accept(spec)
    def deliver(events):
        runtime.receive_inputs(spec.work_seq,dict(worker_id=spec.worker_id,worker_generation=spec.worker_generation,events=events))
    events=[InputFact(i,'source',Selector.TARGET_HOST,dict(ready_version=1,logical_kv_len=4)) for i in (0,1)]
    events.extend([InputFact(0,'classified',Selector.CLASSIFIED,dict(lifecycle=Lifecycle.ACTIVE)),
                   InputFact(0,'predecessor',Selector.PROPOSAL,dict(proposal_handle=None))])
    if not terminal:events.append(InputFact(1,'classified',Selector.CLASSIFIED,dict(lifecycle=Lifecycle.ACTIVE)))
    deliver(events);runtime.step();runtime.step()
    assert built==[(0,)] and not role.executions
    for _ in range(5):runtime.step()
    assert built==[(0,)]
    deliver([InputFact(1,'classified',Selector.CLASSIFIED,dict(lifecycle=Lifecycle.FINISHED))] if terminal else
            [InputFact(1,'predecessor',Selector.PROPOSAL,dict(proposal_handle=None))])
    if budget==3:
        with pytest.raises(ValueError, match='token capacity'):runtime.step()
        assert not role.executions and not runtime.banks.free_rows
        return
    runtime.step()
    assert built==([(0,)] if terminal else [(0,),(1,)])
    assert len(role.executions)==1
    assert tuple(r.index for r in role.executions[0].rows)==((0,) if terminal else (0,1))
    for _ in range(10):runtime.step()
    assert len(role.executions)==1
