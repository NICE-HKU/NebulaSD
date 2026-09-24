from nebulasd.scheduler.cost_table import CostTable
from examples.support.fixed_context_cost import FixedContextCostTable


def test_fixed_slice_ignores_context_and_preserves_copy_bytes():
    rows=[]
    for stage in ('draft_cached','target_verify','H2D'):
        for b,kv,byte_count,ms in ([(1,512,0,10),(8,512,0,17),(32,96,0,99)] if stage!='H2D' else [(1,0,100,1),(1,0,200,2)]):
            rows.append(dict(stage=stage,batch=b,kv=kv,bytes=byte_count,depth=0,sync=0,samples=20,boundary='test',p50_ms=ms,p95_ms=ms))
    original=CostTable(dict(schema_version=1,compatibility={},rows=rows),expected={})
    fixed=FixedContextCostTable(original,512)
    assert fixed.predict('target_verify',batch=8,kv=656).seconds==.017
    assert fixed.predict('target_verify',batch=4,kv=96).seconds==.013
    assert fixed.predict('H2D',batch=1,byte_count=150).seconds==.0015
    assert fixed.summary()['effective_compute_contexts']==[512]
    assert len(original.rows)==8
    import pytest
    with pytest.raises(ValueError,match='requires measured/interpolated batch'):
        fixed.predict('target_verify',batch=9,kv=512)
