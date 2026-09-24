"""Numeric bridge for the autonomous scheduler; no Python callbacks from C++.

Immutable request/table snapshots are cached by identity. Only changed rows are
repacked; pointer order follows the Python candidate mapping exactly. Lifetime is
owned by the synchronous Engine scheduler, never by the DMA/worker processes.
"""
import ctypes as C
import weakref
from time import perf_counter_ns
from . import native_schema as schema
from .cost_table import CostTable
from .measured_placement import MeasuredPlacementEstimator
from .eligibility import online
from .views import value as v
from nebulasd.core.enums import StateChangeBlockKind as K, WorkerRole
from nebulasd.ipc.native import library


def structure(name, fields):
    return type(name, (C.Structure,), {'_fields_': [(n, C.c_int64) for n in fields]})
Request = structure('Request', schema.REQUEST)
Worker = structure('Worker', schema.WORKER)
Record = structure('Record', schema.RECORD)
class CostRow(C.Structure):
    _fields_ = [(n,C.c_int64) for n in ('stage','depth','sync','batch','shape')] + [(n,C.c_double) for n in ('p50','p95')]
class Projection(C.Structure):
    _fields_ = [('destination',C.c_void_p),('payload',C.c_char_p),
                ('fields',C.POINTER(C.c_int32)),('count',C.c_int32)]
class Prediction(C.Structure):
    _fields_ = [(n,C.c_double) for n in ('input','free','kv','compute','start','finish','cost')]
class Output(C.Structure):
    _fields_ = [(n,C.c_int64) for n in ('worker','initial','offset','count')] + [('prediction',Prediction)]
Metrics = structure('Metrics', 'scanned frontier limit cells fits partitions blocked'.split())


def estimator_base(estimator):
    while hasattr(estimator, 'original'):
        estimator = estimator.original
    return estimator


def cost_spec(table):
    if type(table) is CostTable:
        return table, -1
    if hasattr(table, 'native_cost_spec'):
        return table.native_cost_spec()
    raise TypeError('native scheduler requires CostTable or an explicit native_cost_spec')


def supported(estimator, view):
    estimator = estimator_base(estimator)
    return (type(estimator) is MeasuredPlacementEstimator and view.work_state is not None
            and not view.prepared and not view.inflight
            and all(type(t) is CostTable or hasattr(t, 'native_cost_spec') for t in
                    (estimator.table, estimator.draft_copy_table or estimator.table)))


class NativeCompletion:
    def __init__(self, estimator):
        self.estimator = estimator_base(estimator)
        self.lib = lib = library()
        lib.sd_scheduler_project.argtypes = [C.POINTER(Projection),C.c_int]
        lib.sd_scheduler_project.restype = None
        lib.sd_scheduler_size.argtypes = [C.c_int]
        lib.sd_scheduler_size.restype = C.c_int
        if any(lib.sd_scheduler_size(i) != C.sizeof(t) for i,t in enumerate((Request,Worker,Record,Output,Metrics))):
            raise RuntimeError('native scheduler ABI mismatch; rebuild the native library')
        lib.sd_scheduler_create.restype = C.c_void_p
        lib.sd_scheduler_create.argtypes = [C.POINTER(CostRow),C.c_int,C.c_int64,C.POINTER(CostRow),C.c_int,C.c_int64,C.c_int64,C.c_int64]
        lib.sd_scheduler_destroy.argtypes = [C.c_void_p]
        lib.sd_scheduler_destroy.restype = None
        lib.sd_scheduler_error.argtypes = []
        lib.sd_scheduler_error.restype = C.c_char_p
        lib.sd_scheduler_run.restype = C.c_int
        lib.sd_scheduler_run.argtypes = [C.c_void_p,C.POINTER(C.POINTER(Request)),C.c_int,C.POINTER(Worker),C.c_int,
            C.POINTER(Record),C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.c_int,C.c_int,C.c_double,
            C.POINTER(Output),C.POINTER(C.c_int64),C.POINTER(Metrics)]
        def costs(table):
            table,context = cost_spec(table)
            rows = (CostRow*len(table.rows))(*(CostRow(schema.STAGES.index(r['stage']),r['depth'],r['sync'],r['batch'],
                r['bytes'] if r['stage'] in ('H2D','D2H') else r['kv'],r['p50_ms'],r['p95_ms']) for r in table.rows))
            return rows,context
        target,context = costs(self.estimator.table)
        draft,dcontext = costs(self.estimator.draft_copy_table or self.estimator.table)
        self.handle = lib.sd_scheduler_create(target,len(target),context,draft,len(draft),dcontext,
            self.estimator.block_bytes,self.estimator.draft_block_bytes)
        if not self.handle:
            raise RuntimeError(lib.sd_scheduler_error().decode())
        self._finalizer = weakref.finalize(self,lib.sd_scheduler_destroy,self.handle)
        self.requests = {}
        self.worker_specs = None
        self.worker_rows = ()
        self.workers = (Worker*0)()
        self.record_works = ()
        self.record_cache = {}
        self.record_rows = ()
        self.records = (Record*0)()
        self.members = (C.c_int64*0)()
        self.outputs = (Output*0)()
        self.slots = (C.c_int64*1)()
        self.metrics = Metrics()
        self.projections = []
        self.field_maps = {}
        self.request_order = None
        self.pointers = None
        self.groups = [(prefix,getattr(K,kind),fields.split()) for prefix,(kind,fields) in schema.ROW_FIELDS.items()]
        from nebulasd.table.storage import REQUEST_BLOCK_KIND_TO_LAYOUT
        for prefix,kind,fields in self.groups:
            layout={f.field.name:f for f in REQUEST_BLOCK_KIND_TO_LAYOUT[kind].field_layouts}
            projection=[]
            for name in fields:
                f=layout[name]
                projection.extend((f.offset-8,getattr(Request,prefix+'_'+name).offset,
                                   -f.size if f.field.type.signed else f.size))
            self.field_maps[prefix]=(C.c_int32*len(projection))(*projection)

    def request(self, r, view, compute_times):
        saved = self.requests.get(r.slot)
        if saved is None:
            numeric=Request();saved=[numeric,C.pointer(numeric),None,{}];self.requests[r.slot]=saved
        numeric,pointer,previous,rows = saved
        if previous is not r:
            for field,value in zip(schema.REQUEST[:10],(r.slot,r.epoch,r.arrival_seq,r.prompt_count,r.output_count,
                    r.max_new_tokens,r.proposal_depth,r.capacity_blocks,r.admitted_ns,r.ready_ns)):
                setattr(numeric,field,value)
            saved[2]=r
        for prefix,kind,fields in self.groups:
            if prefix == 'e' and view.request_states is not None:
                state = view.request_states[r.slot]
                numeric.e_request_epoch = state.input.epoch
                numeric.e_lifecycle = int(state.lifecycle)
                numeric.e_current_round_id = max(state.current_round, 0)
                continue
            row=view.row(kind,r.slot)
            # Production snapshots are immutable; explicit dict test views can mutate.
            if prefix in rows and rows[prefix] is row and (row is None or hasattr(row,'fields')):
                continue
            payload=getattr(row,'payload',None)
            if payload is not None:
                self.projections.append(Projection(C.addressof(numeric),payload,self.field_maps[prefix],len(fields)))
                rows[prefix]=row
                continue
            values={} if row is None else row._values if hasattr(row, '_values') else {f.name:f.value for f in row.fields} if hasattr(row,'fields') else row
            for field in fields:
                value=values.get(field)
                setattr(numeric,prefix+'_'+field,-1 if value is None else int(value))
            rows[prefix]=row
        for stage, prefix in (('D', 'draft'), ('T', 'target')):
            end = compute_times.latest(r.slot, r.epoch, stage)
            setattr(numeric, prefix + '_compute_round', -1 if end is None else end.round_id)
            setattr(numeric, prefix + '_compute_end_ns', -1 if end is None else end.end_ns)
        return pointer

    def sync_workers(self, specs):
        # WorkerSpec is immutable. Rebuild on configuration, generation or order
        # changes; runtime and phase-dependent capacity are refreshed below.
        if specs != self.worker_specs:
            self.workers = (Worker*len(specs))()
            self.worker_rows = tuple(self.workers)
            for w, row in zip(specs, self.worker_rows):
                draft = w.role == WorkerRole.DRAFT
                row.id, row.draft, row.generation = w.worker_id, draft, w.generation
                row.max_batch, row.block_size, row.bank_blocks = w.max_batch_size, w.block_size, w.bank_blocks
                row.initial_tokens = w.max_batch_tokens if draft else w.prefill_max_batch_tokens
                row.prepare_tokens = w.max_batch_tokens if draft else w.verify_max_batch_tokens
            self.worker_specs = tuple(specs)

    def sync_records(self, live):
        # WORK is frozen. Record completion flags can change without any change
        # to the member layout, including between D and T in one Engine step.
        records = tuple(live.values())
        if len(records) != len(self.record_works) or any(
                record.work is not work for record, work in zip(records, self.record_works)):
            self.record_works = tuple(record.work for record in records)
            for key in self.record_cache.keys()-live.keys():
                del self.record_cache[key]
            numeric, members = [], []
            for key, record in live.items():
                work = record.work
                saved = self.record_cache.get(key)
                if saved is None or saved[0] is not work:
                    imports = [r for r in work.rows if r.source is not None]
                    row = Record(work.worker_id, int(work.operation), work.work_seq, 0, 0,
                        sum(r.capacity_blocks for r in imports), len(imports),
                        sum(r.capacity_blocks for r in work.rows), len(work.rows), len(work.rows))
                    saved = (work, row, tuple(r.slot for r in work.rows))
                    self.record_cache[key] = saved
                numeric.append(saved[1])
                members.extend(saved[2])
            self.records = (Record*len(records))(*numeric)
            self.record_rows = tuple(self.records)
            self.members = (C.c_int64*len(members))(*members)
        for record, row in zip(records, self.record_rows):
            row.compute_done, row.physical_done = record.compute_done, record.physical_done

    def schedule(self, owner, view, phase, now, destinations, compute_destinations, prepare_destinations):
        from .completion import initial_capacity,prepare_capacity
        started=perf_counter_ns()
        estimator=estimator_base(owner._estimator)
        # Production delivers changed slots while merging existing facts. No second observer.
        self.projections.clear()
        order=tuple(view.requests)
        dirty=getattr(owner,'_native_dirty',None)
        changed=set(order) if dirty is None else dirty.intersection(view.requests)
        changed.update(view.requests.keys()-self.requests.keys())
        for slot in changed:self.request(view.requests[slot],view,owner.compute_times)
        if self.projections:
            updates=(Projection*len(self.projections))(*self.projections)
            self.lib.sd_scheduler_project(updates,len(updates))
        for slot in self.requests.keys()-view.requests.keys():del self.requests[slot]
        if dirty is not None:dirty.clear()
        if self.request_order != order:
            self.pointers=(C.POINTER(Request)*len(order))(*(self.requests[slot][1] for slot in order))
            self.request_order=order
        pointers=self.pointers
        self.sync_workers(view.workers)
        banks={};eligible=[]
        for w, row in zip(view.workers, self.worker_rows):
            d=w.role==WorkerRole.DRAFT
            selected=(d==(phase=='D') and (destinations is None or w.worker_id in destinations) and online(view,w))
            if selected and d and not w.draft_banked:raise ValueError('completion requires banked Draft workers')
            ci=initial_capacity(view,w) if selected and (compute_destinations is None or w.worker_id in compute_destinations) else None
            cp=prepare_capacity(view,w) if selected and (prepare_destinations is None or w.worker_id in prepare_destinations) else None
            if selected:eligible.append(w.worker_id)
            for initial,capacity in ((True,ci),(False,cp)):
                if capacity:banks[w.worker_id,initial]=capacity[0]
            rt=view.row(K.WORKER_DRAFT_RUNTIME if d else K.WORKER_TARGET_COMPUTE_RUNTIME,w.worker_id)
            copy=view.row(K.WORKER_DRAFT_COPY_RUNTIME if d else K.WORKER_TARGET_COPY_RUNTIME,w.worker_id)
            cap=getattr(owner,'initial_batch_limit',None)
            row.online = online(view,w)
            row.initial_rows = min(ci[1],cap) if ci and cap is not None else ci[1] if ci else 0
            row.prepare_rows = cp[1] if cp else 0
            row.initial_blocks = min(w.bank_blocks,v(ci[0],'capacity_blocks',0)) if ci else 0
            row.prepare_blocks = min(w.bank_blocks,v(cp[0],'capacity_blocks',0)) if cp else 0
            row.runtime_seq = v(rt,'current_batch_seq' if d else 'compute_batch_seq',-1)
            row.runtime_start = v(rt,'compute_start_time_ns',0)
            row.runtime_status = v(rt,'compute_status',-1)
            row.copy_status = v(copy,'copy_status',-1)
        self.sync_records(view.work_state.records)
        workers, records, members = self.workers, self.records, self.members
        if len(self.outputs) < len(workers):
            self.outputs = (Output*len(workers))()
        if len(self.slots) < len(view.requests):
            self.slots = (C.c_int64*max(len(view.requests),2*len(self.slots)))()
        # C++ fully writes Metrics and only outputs[:n]/selected slot ranges are
        # read. Unused capacity may contain old data and is never an input.
        outputs, slots, metrics = self.outputs, self.slots, self.metrics
        synced=perf_counter_ns()
        args=(self.handle,pointers,len(pointers),workers,len(workers),records,len(records),members,
            now,phase=='D',view.work_state.direct_imports,owner.frontier_factor,owner.service_weights[phase],outputs,slots,C.byref(metrics))
        if owner.service_interval:
            run = self.lib.sd_scheduler_run_service_interval_v3
            run.restype = C.c_int
            run.argtypes = [*self.lib.sd_scheduler_run.argtypes, C.c_double, C.c_double, C.c_int]
            n=run(*args,owner.service_gaps_s[phase],owner.service_batch_delay_s,
                  bool(getattr(self.estimator, "ignore_kv_time", False)))
        else:
            n=self.lib.sd_scheduler_run(*args)
        returned=perf_counter_ns()
        if n<0:raise RuntimeError(self.lib.sd_scheduler_error().decode())
        from .native_plan import NativePlan, PlanMember
        from nebulasd.ipc.command_kinds import CommandKind
        decisions=[];owner.native_predictions={}
        self.banks=banks
        for out in outputs[:n]:
            initial=bool(out.initial)
            bank=banks[out.worker,initial]
            kind=(CommandKind.DRAFT_BATCH if initial else CommandKind.PREPARE_DRAFT_BANK) if phase=='D' else (
                CommandKind.TARGET_PREFILL_BATCH if initial else CommandKind.PREPARE_TARGET_BANK)
            members=[]
            for slot in slots[out.offset:out.offset+out.count]:
                r=self.requests[slot][0]
                round_id=(r.t_round_id+1 if initial else r.d_round_id+1) if phase=='D' else (0 if initial else r.t_round_id+1)
                members.append(PlanMember(slot,r.epoch,round_id))
            worker=next(w for w in view.workers if w.worker_id==out.worker)
            command=NativePlan(out.worker,worker.generation,view.sequences[out.worker],kind,v(bank,'bank_id'),tuple(members))
            decisions.append(command)
            p=out.prediction
            owner.native_predictions[worker.worker_id,command.command_seq]=(now,dict(input_ready_s=p.input,worker_ready_s=p.free,
                kv_ready_s=p.kv,compute_s=p.compute,start_s=p.start,finish_s=p.finish,cost_s=p.cost))
        owner.blocked_workers=({wid for wid in eligible if (wid,True) not in banks and (wid,False) not in banks}
            | {w.worker_id for i,w in enumerate(view.workers) if metrics.blocked & (1 << i)})
        owner.blocked_workers.difference_update(c.worker_id for c in decisions)
        owner.last_metrics=dict(implementation='cpp',sync_ns=synced-started,native_ns=returned-synced,
            build_ns=perf_counter_ns()-returned,total_ns=perf_counter_ns()-started,scanned=metrics.scanned,
            frontier=metrics.frontier,frontier_limit=metrics.limit,prediction_cells=metrics.cells,fit_checks=metrics.fits,
            partitions=metrics.partitions,batches=n)
        return tuple(decisions)
