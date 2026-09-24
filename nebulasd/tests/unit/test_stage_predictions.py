from types import SimpleNamespace as N
from nebulasd.observability.stage_predictions import attach
from nebulasd.observability.profiling import ProfileRecorder


def test_predictions_are_observed_without_recomputing_or_changing_decisions(tmp_path):
    calls=[]
    result=dict(input_ready_s=1.,worker_ready_s=2.,kv_ready_s=3.,start_s=3.)
    def predict(*args,**kw): calls.append(args); return result
    estimator=N(stage_prediction=predict)
    row=N(slot=7,epoch=2)
    item=N(request_slot=7,request_epoch=2,next_round_id=4)
    command=N(kind=N(name='PREPARE_DRAFT_BANK'),worker_id=0,command_seq=12,requests=[item])
    decisions=[command]
    def schedule(view):
        assert scheduler._estimator.stage_prediction(view,'D',N(worker_id=0),[row],100,initial=False) is result
        return decisions
    scheduler=N(_estimator=estimator,schedule=schedule,_clock=lambda:200)
    recorder=ProfileRecorder(tmp_path,'engine',20,mode='light')
    attach(scheduler,recorder)
    assert scheduler.schedule(None) is decisions
    assert len(calls)==1
    event=recorder.events[-1]
    assert event['prediction']==result and event['keys']==[[7,2,4]]
    # No stale prediction can leak into the next schedule invocation.
    decisions.clear()
    scheduler.schedule(None)
    assert len(recorder.events)==1


def test_frozen_replaced_estimator_and_exception_restore(tmp_path):
    from dataclasses import dataclass, replace
    import pytest
    @dataclass(frozen=True)
    class Estimator:
        value: int = 1
        def stage_prediction(self,*args,**kwargs): return {'start_s':self.value}
    scheduler=N(_estimator=Estimator(),_clock=lambda:1)
    def schedule(view):
        assert scheduler._estimator.stage_prediction(None,'D',N(worker_id=0),[],1)['start_s']==2
        raise RuntimeError('original failure')
    scheduler.schedule=schedule
    attach(scheduler,ProfileRecorder(tmp_path,'engine',20,mode='light'))
    current=replace(scheduler._estimator,value=2)
    scheduler._estimator=current
    with pytest.raises(RuntimeError,match='original failure'): scheduler.schedule(None)
    assert scheduler._estimator is current
