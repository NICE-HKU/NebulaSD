"""First batch decision is the complete authorization; no ready/RUN join."""
from queue import Full
from time import perf_counter_ns
from nebulasd.core.enums import StateChangeBlockKind as K, WorkerRole, Lifecycle, WorkerStatus
from nebulasd.scheduler.views import value as v
from nebulasd.workers.work import Work, RowWork, WorkKind, TableDependency, Selector


class WorkProgress:
    def __init__(self, engine):
        self.engine = engine
        self.pending = {'D':(),'T':()}
        self.static_rows = {}
        self.set_workers(engine.resources.specs)
        self._update_depth = 0
        self._wake_all = set()
        self._wake_workers = {'D': set(), 'T': set()}
        self._changed_slots = set()
        self.dirty = {'D':set(), 'T':set()}
        self.schedule_counts = {'D':0, 'T':0}
        self.trigger_counts = {}
        self.first = 'T'
        if hasattr(engine.scheduler, 'enable_native_updates'):
            engine.scheduler.enable_native_updates()

    def set_workers(self, specs):
        self.workers = {w.worker_id: w for w in specs}
        self.role_workers = {stage: frozenset(w.worker_id for w in specs
            if (w.role == WorkerRole.DRAFT) == (stage == 'D')) for stage in ('D', 'T')}

    def begin_updates(self):
        self._update_depth += 1

    def end_updates(self):
        self._update_depth -= 1
        if self._update_depth:
            return
        if self._changed_slots:
            self.engine.scheduler.requests_changed(self._changed_slots)
            self._changed_slots.clear()
        for stage in ('D', 'T'):
            workers = self.role_workers[stage] if stage in self._wake_all else self._wake_workers[stage]
            if workers:
                self.dirty[stage].update(workers)
            self._wake_workers[stage].clear()
        self._wake_all.clear()

    def _wake(self, stage, reason, worker=None):
        # Counts retain raw trigger semantics, not merged set-update counts.
        self.trigger_counts[reason] = self.trigger_counts.get(reason, 0) + 1
        if self._update_depth:
            if worker is None:
                self._wake_all.add(stage)
            elif stage not in self._wake_all:
                self._wake_workers[stage].add(worker)
        else:
            self.dirty[stage].update(self.role_workers[stage] if worker is None else (worker,))

    def observed(self, row, previous):
        kind = row.block_kind
        if kind in (K.REQUEST_DISPATCH, K.REQUEST_DRAFT, K.REQUEST_TARGET_COMPUTE,
                    K.REQUEST_D2H, K.REQUEST_DRAFT_D2H, K.REQUEST_DRAFT_HOSTKV):
            self._request_dirty(row.row)
        if kind == K.WORKER_COMMON:
            if v(row, 'status') == WorkerStatus.ONLINE and (
                    v(previous, 'status') != WorkerStatus.ONLINE or
                    v(previous, 'worker_generation') != v(row, 'worker_generation')):
                w = self.workers.get(row.row)
                if w is not None:
                    self._wake('D' if w.role == WorkerRole.DRAFT else 'T', 'online', w.worker_id)
        elif kind in (K.WORKER_DRAFT_BANK, K.WORKER_BANK):
            # Publication and completion receipts may arrive in either order.
            # A late EMPTY fact must reopen admission even after record retirement.
            if v(row, 'state') == 0 and any(v(row, f) != v(previous, f)
                    for f in ('state', 'bank_epoch', 'alloc_rows')):
                self._wake('D' if kind == K.WORKER_DRAFT_BANK else 'T',
                    'bank_available', row.row // 2)
        elif kind in (K.REQUEST_DRAFT_D2H, K.REQUEST_D2H) and self.engine.ledger.direct_imports:
            if v(row, 'status') == 2 and (v(previous, 'ready_version') != v(row, 'ready_version')
                    or v(previous, 'request_epoch') != v(row, 'request_epoch')
                    or v(previous, 'status') != v(row, 'status')):
                self._wake('D' if kind == K.REQUEST_DRAFT_D2H else 'T', 'host_ready')
        elif kind in (K.REQUEST_DRAFT, K.REQUEST_TARGET_COMPUTE) and v(row, 'status') == 2:
            self.engine.scheduler.observe_result(row)
            self._wake('D', 'request_result')
            self._wake('T', 'request_result')
        elif kind == K.REQUEST_DISPATCH:
            for field, stage in (('draft_issue_seq','T'), ('target_run_seq','D')):
                if v(row, field, 0) and v(row, field) != v(previous, field):
                    self._wake(stage, 'issued_' + stage)

    def _request_dirty(self, slot):
        if self._update_depth:
            self._changed_slots.add(slot)
        else:
            self.engine.scheduler.requests_changed((slot,))

    def request_changed(self, record, *, admitted=False):
        self._request_dirty(record.input.slot)
        if admitted:
            self._wake('T', 'admission')
        elif record.lifecycle == Lifecycle.ACTIVE and record.input.output_count:
            dispatch = self.engine.rows.get((K.REQUEST_DISPATCH, record.input.slot))
            if not v(dispatch, 'draft_issue_seq', 0):
                self._wake('D', 'initial_classified')

    def _events(self):
        events = self.engine.ledger.scheduling_events
        while events:
            kind, work = events.popleft()
            stage = 'D' if work.operation.name.startswith('DRAFT') else 'T'
            self._wake(stage, kind, work.worker_id)
    def changed(self):
        for stage in ('D', 'T'):
            self._wake(stage, 'management')

    def _recover_idle(self):
        self.changed()

    def build(self, plan):
        e = self.engine
        worker = (self.workers[plan.worker_id] if hasattr(self, 'workers') else
                  next(w for w in e.resources.specs if w.worker_id == plan.worker_id))
        draft = worker.role == WorkerRole.DRAFT
        initial = plan.kind.name in ('DRAFT_BATCH','TARGET_PREFILL_BATCH')
        operation = (WorkKind.DRAFT_INITIAL if initial else WorkKind.DRAFT_DECODE) if draft else (
            WorkKind.TARGET_PREFILL if initial else WorkKind.TARGET_VERIFY)
        items = plan.new_requests if plan.kind.name == 'DRAFT_BATCH' else plan.requests
        bank = plan.bank.bank_id if plan.kind.name == 'DRAFT_BATCH' else plan.bank_id if initial else plan.standby_bank_id
        epoch = e.ledger.bank_epochs[worker.worker_id,bank]+1
        kind = K.WORKER_DRAFT_BANK if draft else K.WORKER_BANK
        epoch = max(epoch,v(e.rows.get((kind,worker.worker_id*2+bank)),'bank_epoch',0)+1)
        rows,offset = [],0
        for item in items:
            record = e.registry.records[item.request_slot]
            r = record.input
            target = e.rows.get((K.REQUEST_TARGET_COMPUTE,r.slot))
            previous = e.rows.get((K.REQUEST_DRAFT,r.slot))
            round_id = (v(target,'round_id')+1 if draft else 0) if initial else (
                item.next_round_id if draft else item.round_id)
            if not hasattr(self, 'static_rows'):
                self.static_rows = {}
            key = (draft, r.slot, r.epoch)
            static = self.static_rows.get(key)
            if static is None:
                host = e.rows[K.REQUEST_DRAFT_HOSTKV if draft else K.REQUEST_HOSTKV,r.slot]
                static = (v(host,'offset_blocks'),v(host,'capacity_blocks'),v(host,'arena_id',0),
                    v(host,'host_slot_generation'),v(host,'host_slot'),v(host,'writer_lease_generation'),
                    v(host,'layout_id') if draft else None)
                self.static_rows[key] = static
            def dependency(kind, ticket, selector):
                return TableDependency(kind,r.slot,r.epoch,ticket,selector)
            source = None if initial else dependency(K.REQUEST_DRAFT_D2H if draft else K.REQUEST_D2H,
                v(previous,'snapshot_version') if draft else v(target,'target_kv_version'),
                Selector.DRAFT_HOST if draft else Selector.TARGET_HOST)
            predecessor = dependency(K.REQUEST_TARGET_COMPUTE,round_id-1,Selector.DELTA) if draft else (
                None if initial else dependency(K.REQUEST_DRAFT,round_id,Selector.PROPOSAL))
            classified = None if operation == WorkKind.TARGET_PREFILL else dependency(
                K.REQUEST_TARGET_COMPUTE, round_id-1, Selector.TARGET_DECISION)
            rows.append(RowWork(r.slot,r.epoch,round_id,round_id if draft else round_id+1,offset,r.capacity_blocks,
                *static[:6],r.prompt,r.config,record.reservation,
                min(r.proposal_depth,r.max_new_tokens-r.output_count) if draft or not initial else 1,
                r.max_new_tokens,r.prompt_count,source,predecessor,classified,
                (0 if initial else v(previous,'owner_epoch')+1) if draft else None,
                static[6]))
            offset += r.capacity_blocks
        completion = e.resources.completions.reserve(len(rows))
        if completion is None:
            raise RuntimeError('admitted completion budget exhausted')
        return Work(worker.worker_id,worker.generation,e.ledger.sequences[worker.worker_id],operation,bank,epoch,
            completion,1024+sum(512+12*(r.token_budget+1) for r in rows),tuple(rows))

    def publish_dispatch(self, work):
        e = self.engine
        draft = work.operation.name.startswith('DRAFT')
        updates = []
        for r in work.rows:
            fields = (dict(draft_issue_seq=r.run_seq,draft_round_id=r.round_id,draft_worker_id=work.worker_id,
                draft_worker_generation=work.worker_generation,draft_owner_epoch=r.owner_epoch,
                draft_prepare_seq=r.run_seq) if draft else dict(target_prepare_seq=r.run_seq,
                target_run_seq=r.run_seq,target_round_id=r.round_id,planned_target_id=work.worker_id))
            fields['request_epoch'] = r.epoch
            # Retain the opposite role in an immutable compact local snapshot.
            updates.append((r.slot, fields))
        e._publish_locals(K.REQUEST_DISPATCH, updates)

    def freeze_import(self, work):
        e = self.engine
        if not getattr(e.supervisor.pairs[work.worker_id], 'direct_imports', False):
            return None
        from nebulasd.workers.direct_import import import_regions
        draft = work.operation.name.startswith('DRAFT')
        kind = K.REQUEST_DRAFT_D2H if draft else K.REQUEST_D2H
        sources = []
        for r in work.rows:
            fact = e.rows.get((kind,r.slot))
            if r.source is not None and (v(fact,'request_epoch'),v(fact,'ready_version'),v(fact,'status')) != (
                    r.epoch,r.source.expected_ticket,2):
                raise ValueError('direct import source does not match frozen WORK')
            sources.append(dict(valid_blocks=v(fact,'valid_blocks'),logical_kv_len=v(fact,'logical_kv_len')))
        worker = self.workers[work.worker_id]
        return import_regions(work,worker.bank_blocks,sources,worker.block_size)

    def dispatch_work(self, work, plan, import_plan):
        """Send exactly the frozen WORK/DMA ranges, then register authorization."""
        e = self.engine
        start = perf_counter_ns()
        pair = e.supervisor.pairs[work.worker_id]
        if import_plan is not None:
            pair.submit(work, import_plan=import_plan)
        else:
            pair.submit(work)
        e.ledger.sent(work,plan)
        self.publish_dispatch(work)
        e.dispatch_latency.add(perf_counter_ns()-start)
        if e.observer is not None:
            e.observer.dispatched(work,perf_counter_ns())

    def advance(self):
        e = self.engine
        e.ledger.direct_imports = any(getattr(p, 'direct_imports', False)
                                     for p in e.supervisor.pairs.values())
        sent = 0
        self._events()
        order = self.first, 'D' if self.first == 'T' else 'T'
        self.first = order[1]
        for phase in order:
            self._events()
            if not self.pending[phase] and self.dirty[phase]:
                view = e._scheduling_view()
                destinations = {wid for wid in self.dirty[phase]
                    if e.ledger.capacity(view, self.workers[wid]) is not None}
                self.dirty[phase].clear()
                if not destinations:
                    continue
                self.schedule_counts[phase] += 1
                start = perf_counter_ns()
                plans = e.scheduler.schedule(view,phase=phase,destinations=destinations)
                e.schedule_latency.add(perf_counter_ns()-start)
                work_plans = ((None, p, None) for p in plans)
            else:
                work_plans = self.pending[phase]
            pending, blocked = [], set()
            for work,plan,import_plan in work_plans:
                if plan.worker_id in blocked:
                    pending.append((work,plan,import_plan))
                    continue
                if work is None:
                    work = self.build(plan)
                    import_plan = self.freeze_import(work)
                try:
                    self.dispatch_work(work, plan, import_plan)
                except Full:
                    pending.append((work,plan,import_plan))
                    blocked.add(work.worker_id)
                    continue
                sent += 1
                self._wake(phase, 'issued_capacity', work.worker_id)
            self.pending[phase] = tuple(pending)
            # Make issued predecessor identities visible to the other stage in
            # this invocation. No ready, H2D or compute gate is added here.
            e._observe_facts()
            e.ledger.refresh(e.rows)
        e._retry_dispatch = (any(self.pending.values()) or any(self.dirty.values())
                             or bool(e.ledger.scheduling_events) or e.reader.has_pending())
        return sent
