from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import pytest
from threading import Event
from time import perf_counter_ns
from types import SimpleNamespace as NS

from nebulasd.kv.arena import SharedHostKVArena
from nebulasd.kv.transfer import CopyPlan, CopyRegion, CopyReceipt
from nebulasd.workers.direct_import import encode, read, serve, import_regions


@pytest.mark.parametrize("compact", [False, True])
def test_queued_import_starts_after_export_without_execution_receiving_reply(compact):
    ctx = mp.get_context('spawn')
    incoming, sender = ctx.Pipe(duplex=False)
    owner, dma = ctx.Pipe()
    slot = ctx.Array('Q', 6)
    wake, export_started, export_finish, imported = Event(), Event(), Event(), Event()
    arena = SharedHostKVArena.create(total_blocks=8, block_bytes=8,
        dtype='torch.float16', kv_block_shape=(4,))
    extent = arena.make_extent(request_slot=0, request_epoch=1, host_slot=0,
        host_slot_generation=1, writer_lease_generation=1, offset_blocks=0, capacity_blocks=8)
    def execute(plan):
        if plan.direction == 'D2H':
            assert [(r.gpu_begin_block,r.host_begin_block,r.block_count) for r in plan.regions] == [(0,0,1)]
            export_started.set()
            assert export_finish.wait(5)
        else:
            assert export_finish.is_set()
            assert [(r.gpu_begin_block,r.host_begin_block,r.block_count) for r in plan.regions] == [(4,2,2)]
            imported.set()
        now = perf_counter_ns()
        return CopyReceipt(now, now, 0.25)
    with ThreadPoolExecutor(1) as pool:
        job = pool.submit(serve, incoming, dma, slot, wake, execute, extent)
        try:
            # Initial work claims a Bank without an H2D, like prefill.
            sender.send_bytes(encode(1, 10, ()))
            assert wake.wait(5)
            assert read(slot, 1) is not None
            sender.send_bytes(encode(2, 20, ((4,2,2),)))
            owner.send(('EXPORT', ((0,0,1),)) if compact else CopyPlan('D2H', (CopyRegion(extent,0,0,1),)))
            assert export_started.wait(5)
            assert not imported.is_set()
            export_finish.set()
            # Deliberately do NOT recv the D2H receipt or run any Runtime step.
            assert imported.wait(5)
            owner.recv()
            owner.send('RELEASE')
            assert owner.recv().cpu_ns == 0
            owner.send(None)
            job.result(5)
            assert read(slot, 2)[0] == 20
        finally:
            export_finish.set()
            for connection in (incoming,sender,owner,dma):connection.close()
            arena.close();arena.unlink()


def test_import_descriptor_uses_frozen_offsets_and_actual_source_length():
    rows=(NS(source=object(),host_offset=100,destination_offset=8),
          NS(source=None,host_offset=200,destination_offset=20))
    w=NS(bank_id=1,operation=NS(name='TARGET_VERIFY'),rows=rows)
    assert import_regions(w,256,[dict(logical_kv_len=17),{}])==((264,100,2),)
    w.operation.name='DRAFT_DECODE'
    assert import_regions(w,256,[dict(valid_blocks=3),{}])==((264,100,3),)


def test_ledger_allows_only_one_successor_behind_completed_compute():
    from nebulasd.engine.work_ledger import WorkLedger
    from nebulasd.core.enums import WorkerRole, StateChangeBlockKind as K
    worker=NS(worker_id=0,role=WorkerRole.DRAFT,max_batch_size=64,bank_rows=128)
    ledger=WorkLedger(NS(specs=[worker]));ledger.direct_imports=True
    banks={0:dict(bank_epoch=1,state=1),1:dict(bank_epoch=1,state=4)}
    view=NS(row=lambda kind,index:banks[index])
    def record(bank,seq,done):
        return NS(work=NS(worker_id=0,bank_id=bank,rows=[None]*64),physical_done=False,compute_done=done)
    ledger.records={1:record(0,1,True),2:record(1,2,False)}
    ledger.by_worker[0] = ledger.records
    assert ledger.capacity(view,worker)[1]==64
    ledger.records[3]=record(0,3,False)
    assert ledger.capacity(view,worker) is None
    del ledger.records[3]
    ledger.records[1].compute_done=False
    assert ledger.capacity(view,worker) is None


def test_scheduler_source_ready_is_exact_epoch_and_version_for_both_models():
    from nebulasd.scheduler.completion import host_ready
    from nebulasd.core.enums import StateChangeBlockKind as K, D2HStatus
    for stage,host_kind,result_kind,issued,version in [
        ('D',K.REQUEST_DRAFT_D2H,K.REQUEST_DRAFT,'draft_issue_seq','snapshot_version'),
        ('T',K.REQUEST_D2H,K.REQUEST_TARGET_COMPUTE,'target_run_seq','target_kv_version')]:
        rows={K.REQUEST_DISPATCH:{issued:1},host_kind:dict(request_epoch=2,status=D2HStatus.HOST_READY,ready_version=3),result_kind:{version:3}}
        view=NS(row=lambda kind,slot:rows.get(kind));request=NS(slot=0,epoch=2)
        assert host_ready(view,stage,request)
        rows[host_kind]['ready_version']=2
        assert not host_ready(view,stage,request)
        rows[host_kind]['ready_version']=3;rows[host_kind]['request_epoch']=1
        assert not host_ready(view,stage,request)
        rows[K.REQUEST_DISPATCH][issued]=0
        assert host_ready(view,stage,request)


def test_direct_import_does_not_wait_for_metadata_but_compute_does():
    from test_autonomous_runtime import Jobs
    from test_autonomous_work import work
    from nebulasd.workers.runtime import Runtime
    from nebulasd.workers.banks import Banks, Phase
    from nebulasd.workers.work import Outcome
    jobs=Jobs();jobs.direct_imports=True
    jobs.import_receipt=lambda bank,seq:(1,CopyReceipt(2,3,0.0))
    jobs.release_bank=lambda bank:jobs.job('release',bank)
    rt=Runtime(Banks(128,8),jobs,jobs)
    w=work();assert rt.accept(w)
    state=rt.records[w.work_seq]
    # Arrival of H2D may precede even input capture. It must not start compute.
    rt.step()
    assert state.imported and state.import_inputs is None and rt.compute is None
    # A skipped job must still retire its DMA reservation before releasing Bank.
    state.outcomes[:]=[Outcome.SKIPPED_FINISHED]*len(w.rows)
    jobs.finish('input',w.work_seq,NS(rows=()))
    jobs.finish('plan',w.work_seq,None)
    rt.step()
    assert not state.physical_done and 'D2H' in state.jobs
    jobs.finish('release',w.bank_id,CopyReceipt(4,5,0.0))
    rt.step()
    assert state.physical_done and rt.banks.banks[w.bank_id].phase==Phase.FREE


def test_compute_waits_for_metadata_after_early_import_completion():
    from test_autonomous_runtime import Jobs
    from test_autonomous_work import work
    from nebulasd.workers.runtime import Runtime
    from nebulasd.workers.banks import Banks
    jobs = Jobs()
    jobs.direct_imports = True
    jobs.import_receipt = lambda bank, seq: (1, CopyReceipt(2, 3, 0.0))
    rt = Runtime(Banks(128, 8), jobs, jobs)
    w = work()
    assert rt.accept(w)
    rt.step()
    jobs.finish('input', w.work_seq, NS(rows=()))
    jobs.finish('plan', w.work_seq, w.work_seq)
    rt.step()
    assert rt.records[w.work_seq].imported
    assert rt.compute is None
    jobs.finish('metadata', w.bank_id, object())
    rt.step()
    assert rt.compute is rt.records[w.work_seq]
    # Subsequent polls must not reset the computing Bank to READY.
    rt.step()
    assert list(k for k in jobs.jobs if k[0] == 'compute') == [('compute', w.work_seq)]
