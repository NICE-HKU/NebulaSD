"""Experiment-only compute-table slice; real KV/copy inputs are untouched."""
from collections import Counter
from nebulasd.scheduler.cost_table import CostTable


class FixedContextCostTable:
    def __init__(self, table, context):
        self.context = context
        rows = [dict(r) for r in table.rows if r['stage'] in ('H2D','D2H') or r['kv'] == context]
        self.table = CostTable(dict(schema_version=1,compatibility=dict(table.compatibility),rows=rows),
                               expected=dict(table.compatibility))
        self.reset()

    def native_cost_spec(self):
        self.native_active = True
        return self.table, self.context

    def reset(self):
        self.counts = Counter()
        self.original_contexts = set()
        self.effective_contexts = set()

    def predict(self, stage, *, batch, kv=0, depth=0, sync=0, byte_count=0):
        copy = stage in ('H2D','D2H')
        if not copy:
            self.original_contexts.add(kv)
            kv = self.context
            self.effective_contexts.add(kv)
        result = self.table.predict(stage,batch=batch,kv=kv,depth=depth,sync=sync,byte_count=byte_count)
        if not copy and result.method.split(';')[0] not in ('exact','bilinear'):
            raise ValueError(f'fixed-context experiment requires measured/interpolated batch: {stage} {batch} {result.method}')
        self.counts[stage,result.method] += 1
        return result

    def summary(self):
        return dict(fixed_compute_context=self.context,
                    native_active=getattr(self,"native_active",False),
                    query_audit_scope="Python calls only; native enforces the same fixed-context bounds",
                    effective_compute_contexts=sorted(self.effective_contexts),
                    original_compute_context_range=[min(self.original_contexts),max(self.original_contexts)] if self.original_contexts else [],
                    queries=[dict(stage=s,method=m,count=n) for (s,m),n in sorted(self.counts.items())],
                    copy_uses_actual_bytes=True)
