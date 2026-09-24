"""Opt-in read-only evaluation of frozen batches, with exact fact provenance.

No candidate search, prediction, mutation or CUDA timing. The predicates mirror
Run gates and retain every field read on a successful path for offline audit.
"""
from time import perf_counter_ns
from nebulasd.core.enums import BankState, ComputeStatus, H2DStatus, D2HStatus, StateChangeBlockKind as K
from nebulasd.scheduler.eligibility import online, target_result, draft_ready, restored
from nebulasd.scheduler.draft_placement import matches, source
from nebulasd.scheduler.views import value as v
from nebulasd.table.draft_fences import prepare_values, h2d_values, allocation_values


def predicates(view, worker, p):
    draft = p.kind.name == 'PREPARE_DRAFT_BANK'
    def inputs(w):
        for item in p.requests:
            r=w.requests.get(item.request_slot)
            if r is None or r.epoch != item.request_epoch:return False
            if draft:
                t=target_result(w,r)
                if t is None or v(t,'round_id') != item.source.round_id:return False
            elif not draft_ready(w,r,item.round_id):return False
        return True
    def kv(w):
        b=w.row(K.WORKER_DRAFT_BANK if draft else K.WORKER_BANK,worker.worker_id*2+p.standby_bank_id)
        if not matches(b,dict(state=BankState.READY,bank_epoch=p.next_bank_epoch,batch_seq=p.batch_seq)):return False
        for item in p.requests:
            r=w.requests.get(item.request_slot)
            if r is None:return False
            if not draft:
                if not restored(w,r,p,item):return False
                continue
            if not matches(w.row(K.REQUEST_DRAFT_H2D,item.request_slot),dict(**h2d_values(p,item),
                status=H2DStatus.GPU_READY,result_code=0,gpu_ready_version=item.source.snapshot_version,
                copied_blocks=item.source.valid_blocks)):return False
            if not matches(w.row(K.REQUEST_DRAFT_D2H,item.request_slot),dict(request_epoch=item.request_epoch,
                status=D2HStatus.HOST_READY,ready_version=item.source.snapshot_version,snapshot_handle=item.snapshot_handle,
                snapshot_version=item.source.snapshot_version,snapshot_round_id=item.source.round_id,
                source_worker_id=item.source.worker_id,source_worker_generation=item.source.worker_generation,
                source_op_seq=item.source.op_seq,owner_epoch=item.source.owner_epoch,logical_kv_len=item.source.logical_kv_len,
                valid_blocks=item.source.valid_blocks,result_code=0,**allocation_values(item.source.allocation))):return False
        return True
    def free(w):
        return online(w,worker) and v(w.row(K.WORKER_DRAFT_RUNTIME if draft else K.WORKER_TARGET_COMPUTE_RUNTIME,
            worker.worker_id),'compute_status') == ComputeStatus.IDLE
    def fences(w):
        if not online(w,worker):return False
        if not draft:return True  # Target restore/input predicates include dispatch and identity fences.
        if p.worker_generation != worker.generation:return False
        return all((r:=w.requests.get(i.request_slot)) is not None and source(w,r)==i.source
            and matches(w.row(K.REQUEST_DISPATCH,i.request_slot),prepare_values(p,i)) for i in p.requests)
    return {'in':inputs,'KV':kv,'free':free,'fences':fences}


class EvidenceView:
    def __init__(self,view):self.view,self.reads=view,{}
    def __getattr__(self,name):return getattr(self.view,name)
    def row(self,kind,index):
        raw=self.view.row(kind,index)
        if raw is None:return None
        reads=self.reads.setdefault((kind.name,index,raw.publish_seq),{})
        class Row:
            def get(self,name):
                value=raw.get(name);reads[name]=value;return value
        return Row()
    def evidence(self):
        return [dict(kind=k,row=r,publish_seq=s,fields=f) for (k,r,s),f in self.reads.items()]


def attach(scheduler,recorder):
    original=scheduler.schedule
    previous={}
    def schedule(view, **kwargs):
        if kwargs.get("phase") not in (None, "ready"):
            return original(view, **kwargs)  # Planning reservations are not issued facts.
        live=set()
        workers={w.worker_id:w for w in view.workers}
        for wid,p in view.prepared.items():
            identity=(wid,p.command_seq);live.add(identity)
            gates=predicates(view,workers[wid],p)
            before=previous.setdefault(identity,{})
            request_state=tuple((i.request_slot, id(view.requests.get(i.request_slot))) for i in p.requests)
            for name,check in gates.items():
                cached=before.get(name+'_cache')
                if cached is not None and cached[0]==request_state and all(
                    (row:=view.row(kind,index)) is not None and row.publish_seq==seq
                    for kind,index,seq in cached[1]):
                    continue  # Exact same successful predicate inputs, including RequestInput objects.
                ready=bool(check(view))
                if ready and before.get(name) is True:
                    # Successful paths have the same fields for this immutable Prepare.
                    before[name+'_cache']=(request_state,tuple((kind,index,view.row(kind,index).publish_seq)
                        for kind,index,_ in cached[1]))
                    continue
                before.pop(name+'_cache',None)
                if before.get(name) == ready:continue
                evidence=[]
                if ready:
                    tracked=EvidenceView(view)
                    assert check(tracked)
                    evidence=tracked.evidence()
                    before[name+'_cache']=(request_state,tuple((K[kind],index,seq)
                        for kind,index,seq in tracked.reads))
                recorder.record('scheduler.gate',perf_counter_ns(),keys=[],worker=wid,prepare_seq=p.command_seq,
                    gate=name,ready=ready,evidence=evidence)
                before[name]=ready
            ledger=wid not in view.inflight
            if before.get('ledger') != ledger:
                recorder.record('scheduler.gate',perf_counter_ns(),keys=[],worker=wid,prepare_seq=p.command_seq,
                    gate='ledger',ready=ledger,evidence=[])
                before['ledger']=ledger
        for identity in tuple(previous):
            if identity not in live:del previous[identity]
        return original(view, **kwargs)
    scheduler.schedule=schedule
