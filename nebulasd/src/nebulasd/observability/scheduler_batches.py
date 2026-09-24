"""Profile-only local snapshots explaining pre-prepare group splits; no IPC."""
from time import perf_counter_ns
from dataclasses import replace
from nebulasd.core.enums import BankRole, WorkerRole, StateChangeBlockKind as K
from nebulasd.scheduler.eligibility import active, online, target_result
from nebulasd.scheduler.grouped_placement import related_batches
from nebulasd.scheduler.views import value as v


def attach_batch_diagnostics(scheduler, recorder):
    """Wrap only a profiled instance; unprofiled scheduling has zero hooks."""
    original = scheduler.schedule

    def schedule(view, **kwargs):
        if kwargs.get("phase") == "ready":
            return original(view, **kwargs)
        start = perf_counter_ns()
        commands = original(view, **kwargs)
        if getattr(scheduler, 'execution_mode', None) == 'completion':
            recorder.record('scheduler.bounded_frontier', start, perf_counter_ns(), keys=[],
                stage=kwargs.get('phase'), **scheduler.last_metrics)
            return commands  # Never replay unbounded diagnostics/prediction.
        draft_workers=[]
        for w in view.workers:
            if w.role != WorkerRole.DRAFT:continue
            entries=[]
            for r in sorted(view.requests.values(),key=lambda r:(r.arrival_seq,r.slot)):
                if not active(view,r):continue
                result=target_result(view,r);dispatch=view.row(K.REQUEST_DISPATCH,r.slot)
                issued=v(dispatch,'draft_issue_seq',0)
                reason=('target_not_consumed_or_fenced' if result is None else
                        'affinity' if not w.draft_banked and issued and v(dispatch,'draft_worker_id')!=w.worker_id else
                        'already_issued' if issued and v(dispatch,'draft_round_id',0)>v(result,'round_id') else
                        'worker_offline' if not online(view,w) else 'eligible')
                entries.append(dict(key=[r.slot,r.epoch,v(result,'round_id',-1)+1],reason=reason,
                    prompt=r.prompt_count,output=r.output_count,depth=r.proposal_depth,blocks=r.capacity_blocks,
                    first=not bool(issued)))
            current=view.inflight.get(w.worker_id)
            draft_workers.append(dict(worker=w.worker_id,busy=current is not None,
                inflight_seq=getattr(current,'command_seq',None),entries=entries,
                max_requests=w.max_batch_size,max_tokens=w.max_batch_tokens))
        recorder.record('scheduler.draft_candidates',start,perf_counter_ns(),keys=[],workers=draft_workers,
            commands=[dict(worker=c.worker_id,kind=c.kind.name,seq=c.command_seq) for c in commands],
            decision='commands' if commands else 'no_dispatch')
        prepares = [c for c in commands if c.kind.name == 'PREPARE_TARGET_BANK']
        if hasattr(scheduler._estimator, 'batch_prediction') and prepares:
            inflight=dict(view.inflight)
            inflight.update({c.worker_id:c for c in commands if c.kind.name in
                             ('DRAFT_BATCH','RUN_DRAFT_BATCH','RUN_TARGET_BATCH','TARGET_PREFILL_BATCH')})
            prediction_view=replace(view,inflight=inflight)
            for command in prepares:
                worker=next(w for w in view.workers if w.worker_id==command.worker_id)
                prediction=scheduler._estimator.batch_prediction(prediction_view,worker,
                    [view.requests[i.request_slot] for i in command.requests],start)
                recorder.record('scheduler.batch_prediction',start,perf_counter_ns(),
                    keys=[[i.request_slot,i.request_epoch,i.round_id] for i in command.requests],
                    worker=command.worker_id,batch_seq=command.batch_seq,**prediction)
        if prepares:
            frozen = {r.request_slot for c in view.prepared.values() if c.kind.name == 'PREPARE_TARGET_BANK' for r in c.requests}
            candidates = sorted((r for r in view.requests.values()
                                 if r.slot not in frozen and target_result(view, r) is not None),
                                key=lambda r: (r.arrival_seq, r.slot))
            groups = [[dict(slot=r.slot, epoch=r.epoch,
                           round=v(view.row(K.REQUEST_TARGET_COMPUTE, r.slot), 'round_id')+1,
                           blocks=r.capacity_blocks, tokens=r.proposal_depth+1) for r in group]
                      for group in related_batches(view, candidates)]
            targets = []
            for w in view.workers:
                if w.role != WorkerRole.TARGET:
                    continue
                bank = scheduler._bank(view, w, BankRole.STANDBY)
                targets.append(dict(worker=w.worker_id, online=online(view,w),
                    idle=scheduler._compute_idle(view,w), inflight=w.worker_id in view.inflight,
                    occupied=w.worker_id in view.prepared or any(c.worker_id==w.worker_id
                        and c.kind.name=='TARGET_PREFILL_BATCH' for c in commands),
                    state=v(bank,'state'), max_requests=w.max_batch_size,
                    tokens=w.verify_max_batch_tokens, blocks=v(bank,'capacity_blocks'),
                    rows=max(0,v(bank,'capacity_rows',0)-scheduler._other_rows(view,w,bank)) if bank else 0))
            recorder.record('scheduler.batch_candidates', start, perf_counter_ns(), keys=[],
                policy=scheduler._target_placement, groups=groups, targets=targets,
                prepares=[dict(worker=c.worker_id, batch_seq=c.batch_seq,
                    keys=[[r.request_slot,r.request_epoch,r.round_id] for r in c.requests]) for c in prepares])
        return commands
    scheduler.schedule = schedule
