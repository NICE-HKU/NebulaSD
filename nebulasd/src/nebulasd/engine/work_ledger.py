"""Frozen WORK identities, request result fences and bounded Bank reservations."""
from dataclasses import dataclass, field
from collections import deque
from nebulasd.core.enums import StateChangeBlockKind as K, WorkerRole
from nebulasd.scheduler.views import value as v
from nebulasd.workers.work import Outcome


@dataclass
class Record:
    work: object
    applied: set = field(default_factory=set)
    completion: object = None
    compute_done: bool = False
    physical_done: bool = False


@dataclass(frozen=True, slots=True)
class Authorization:
    record: Record
    member: object


class WorkLedger:
    def __init__(self, resources):
        self.resources = resources
        self.sequences = {w.worker_id:1 for w in resources.specs}
        self.records, self.request_index = {}, {}
        self.by_request = {}
        self.by_worker = {w.worker_id: {} for w in resources.specs}
        self.bank_epochs = {(w.worker_id,b):0 for w in resources.specs for b in (0,1)}
        self.rows = {}
        self.direct_imports = False
        self.max_outstanding = {}
        self.completed = []
        self.scheduling_events = deque()

    def key(self, work):
        return work.worker_id,work.worker_generation,work.work_seq

    def sent(self, work, plan):
        key = self.key(work)
        if key in self.records:
            raise ValueError('duplicate WORK identity')
        record = Record(work)
        for row in work.rows:
            index = (work.operation.name.startswith('DRAFT'),row.slot,row.epoch,row.round_id,row.run_seq)
            if index in self.request_index:
                raise ValueError('duplicate request authorization')
            context = Authorization(record, row)
            self.request_index[index] = context
            self.by_request.setdefault(index[:3], {})[index[3:]] = context
        self.records[key] = record
        self.by_worker.setdefault(work.worker_id, {})[work.work_seq] = record
        self.sequences[work.worker_id] = work.work_seq+1
        self.bank_epochs[work.worker_id,work.bank_id] = work.bank_epoch
        n = len(self.by_worker[work.worker_id])
        self.max_outstanding[work.worker_id] = max(n,self.max_outstanding.get(work.worker_id,0))

    def observe_completions(self):
        changed = False
        # At most four outstanding WORKs per worker. Recheck even without hints.
        for record in self.records.values():
            if record.completion is not None:
                continue
            work = record.work
            completion = self.resources.completions.read(work.completion_offset)
            if completion is None:
                continue
            if (completion.worker_generation, completion.work_seq, completion.bank_id, completion.bank_epoch) != (
                    work.worker_generation, work.work_seq, work.bank_id, work.bank_epoch):
                raise ValueError('completion identity mismatch')
            if tuple((m.slot,m.epoch,m.round_id) for m in completion.members) != tuple(
                    (r.slot,r.epoch,r.round_id) for r in work.rows):
                raise ValueError('completion members mismatch')
            before = record.compute_done, record.physical_done
            record.completion = completion
            record.compute_done = record.physical_done = True
            self._schedule_changes(record, before)
            changed = True
        return changed

    def validate_fact(self, row):
        if row.block_kind not in (K.REQUEST_DRAFT, K.REQUEST_TARGET_COMPUTE):
            return self._validate_copy(row)
        if v(row, 'status') != 2:
            return True
        draft = row.block_kind == K.REQUEST_DRAFT
        index = (draft,row.row,v(row,'request_epoch'),v(row,'round_id'),
                 v(row,'observed_issue_seq' if draft else 'observed_run_seq'))
        context = self.request_index.get(index)
        if context is None:
            raise ValueError('result has no WORK authorization')
        record, member = context.record, context.member
        w = record.work
        if (v(row,'worker_id' if draft else 'target_id'),
            v(row,'worker_generation' if draft else 'target_generation'),
            v(row,'bank_id'),v(row,'bank_epoch')) != (w.worker_id,w.worker_generation,w.bank_id,w.bank_epoch):
            raise ValueError('result WORK fence mismatch')
        if v(row, 'result_code') != 0:
            raise ValueError('failed WORK result')
        if draft and (v(row,'batch_seq'),v(row,'owner_epoch')) != (w.work_seq,member.owner_epoch):
            raise ValueError('Draft result ownership fence mismatch')
        return context

    def result_applied(self, context, previous):
        """Required owner state is updated; this is not client delivery/ACK."""
        member, record = context.member, context.record
        record.applied.add(member.slot)
        draft = record.work.operation.name.startswith('DRAFT')
        if member.round_id > v(previous, 'round_id', 0 if draft else -1) + 1:
            for (round_id, _), prior in self.by_request[draft, member.slot, member.epoch].items():
                if round_id <= member.round_id:
                    prior.record.applied.add(member.slot)

    def _validate_copy(self, row):
        kinds = {K.REQUEST_H2D:(False, True), K.REQUEST_D2H:(False,False),
                 K.REQUEST_DRAFT_H2D:(True,True), K.REQUEST_DRAFT_D2H:(True,False)}
        if row.block_kind not in kinds:
            return True
        draft, imported = kinds[row.block_kind]
        if v(row, 'status') != (3 if imported else 2):
            return True
        round_field = ('next_round_id' if imported else 'snapshot_round_id') if draft else 'round_id'
        run_field = ('observed_prepare_seq' if imported else 'source_op_seq') if draft else (
                     'observed_prepare_seq' if imported else 'd2h_op_seq')
        epoch, round_id, run = v(row,'request_epoch'),v(row,round_field),v(row,run_field)
        context = self.request_index.get((draft,row.row,epoch,round_id,run))
        result = self.rows.get((K.REQUEST_DRAFT if draft else K.REQUEST_TARGET_COMPUTE,row.row))
        if context is None:
            # Completion may retire before a budgeted reader sees this copy row.
            # The accepted result retains the same ownership fence. Superseded
            # copies cannot make a newer result eligible and are discarded.
            if (v(result,'request_epoch'),v(result,'round_id')) != (epoch,round_id):
                return False
            worker = v(result,'worker_id' if draft else 'target_id')
            generation = v(result,'worker_generation' if draft else 'target_generation')
            bank, bank_epoch = v(result,'bank_id'),v(result,'bank_epoch')
        else:
            work, member = context.record.work, context.member
            worker,generation,bank,bank_epoch = work.worker_id,work.worker_generation,work.bank_id,work.bank_epoch
            if imported and member.source is not None and v(row,'gpu_ready_version') != member.source.expected_ticket:
                raise ValueError('copy dependency ticket mismatch')
        worker_field = ('destination_worker_id' if imported else 'source_worker_id') if draft else 'target_id'
        generation_field = ('destination_worker_generation' if imported else 'source_worker_generation') if draft else 'target_generation'
        bank_field = 'destination_bank_id' if imported else 'source_bank_id'
        epoch_field = 'destination_bank_epoch' if imported else 'source_bank_epoch'
        if (v(row,worker_field),v(row,generation_field),v(row,bank_field),v(row,epoch_field)) != (worker,generation,bank,bank_epoch):
            raise ValueError('copy WORK fence mismatch')
        if not imported and (v(result,'request_epoch'),v(result,'round_id')) == (epoch,round_id):
            if v(row,'ready_version') != v(result,'snapshot_version' if draft else 'target_kv_version'):
                raise ValueError('HostKV/result ticket mismatch')
        if v(row,'result_code') != 0:
            raise ValueError('failed copy fact')
        return True

    def _schedule_changes(self, record, before):
        for index, name in enumerate(('compute', 'physical')):
            if not before[index] and (record.compute_done, record.physical_done)[index]:
                self.scheduling_events.append((name, record.work))

    def refresh(self, rows):
        """Advance local compute/retirement using cached completion only."""
        self.rows = rows
        changed = False
        for key in tuple(self.records):
            record = self.records.get(key)
            if record is None:
                continue
            before = record.compute_done,record.physical_done
            w = record.work
            draft = w.operation.name.startswith('DRAFT')
            b = rows.get((K.WORKER_DRAFT_BANK if draft else K.WORKER_BANK,w.worker_id*2+w.bank_id))
            epoch = v(b,'bank_epoch',-1)
            runtime = rows.get((K.WORKER_DRAFT_RUNTIME if draft else K.WORKER_TARGET_COMPUTE_RUNTIME,w.worker_id))
            if v(runtime,'current_batch_seq' if draft else 'compute_batch_seq') == w.work_seq and v(runtime,'compute_start_time_ns',0) and v(runtime,'compute_status') == 0:
                record.compute_done = True
            # Newer Bank allocation proves old compute has ended. Physical
            # release still requires the matching immutable completion record.
            if epoch > w.bank_epoch or (epoch == w.bank_epoch and v(b,'state') in (0,1)):
                record.compute_done = True
            self._schedule_changes(record, before)
            changed |= before != (record.compute_done,record.physical_done)
            if record.completion is None:
                continue
            if any(m.outcome == Outcome.EXECUTED and m.slot not in record.applied
                   for m in record.completion.members):
                continue
            for r in w.rows:
                self.request_index.pop((draft,r.slot,r.epoch,r.round_id,r.run_seq))
                members = self.by_request[draft,r.slot,r.epoch]
                del members[r.round_id,r.run_seq]
                if not members:
                    del self.by_request[draft,r.slot,r.epoch]
            self.completed.append(record.completion)
            del self.records[key]
            del self.by_worker[w.worker_id][w.work_seq]
            self.scheduling_events.append(('credit', w))
            changed = True
        return changed

    def capacity(self, view, worker):
        records = tuple(self.by_worker.get(worker.worker_id, {}).values())
        if len(records) >= 4:
            return None
        kind = K.WORKER_DRAFT_BANK if worker.role == WorkerRole.DRAFT else K.WORKER_BANK
        reserved = {b:[r for r in records if r.work.bank_id == b and not r.physical_done]
                    for b in (0, 1)}
        # Prefer an empty Bank. Otherwise queue one successor behind an export;
        # it may import directly once DMA completes that export.
        for pending in (False, True) if self.direct_imports else (False,):
            for bank_id in (0, 1):
                bank = view.row(kind,worker.worker_id*2+bank_id)
                owners = reserved[bank_id]
                if pending:
                    if len(owners) != 1 or not owners[0].compute_done:
                        continue
                elif owners or v(bank,'state') != 0:
                    continue
                used = max((len(r.work.rows) for r in reserved[1-bank_id]), default=0)
                rows = min(worker.max_batch_size,worker.bank_rows-used)
                if rows <= 0:
                    continue
                values = {f.name:f.value for f in bank.fields} if hasattr(bank,'fields') else dict(bank)
                values['bank_epoch'] = max(v(bank,'bank_epoch'),self.bank_epochs[worker.worker_id,bank_id])
                return values,rows
        return None

    def worker_free(self, estimator, view, worker, now_ns):
        now = now_ns/1e9
        duration = 0.0
        for record in self.by_worker.get(worker.worker_id, {}).values():
            if record.compute_done:
                continue
            requests = [view.requests[r.slot] for r in record.work.rows if r.slot in view.requests]
            name = record.work.operation.name
            stage = dict(TARGET_PREFILL='target_prefill',TARGET_VERIFY='target_verify',DRAFT_INITIAL='draft_first',DRAFT_DECODE='draft_cached')[name]
            service = estimator._duration(stage,requests,sync=2 if stage == 'draft_cached' else 0)
            draft = name.startswith('DRAFT')
            runtime = view.row(K.WORKER_DRAFT_RUNTIME if draft else K.WORKER_TARGET_COMPUTE_RUNTIME,worker.worker_id)
            if v(runtime,'current_batch_seq' if draft else 'compute_batch_seq') == record.work.work_seq and v(runtime,'compute_start_time_ns',0):
                service = max(0.0,service-(now_ns-v(runtime,'compute_start_time_ns'))/1e9) if v(runtime,'compute_status') == 1 else 0.0
            duration += service
        return now+duration

    def copy_free(self, estimator, view, worker_id, now):
        # Conservative sum across both lanes and accepted pending imports;
        # unknown start clocks never make occupied transfers free immediately.
        seconds = 0.0
        for record in self.by_worker.get(worker_id, {}).values():
            w = record.work
            if record.physical_done:
                continue
            draft = w.operation.name.startswith('DRAFT')
            block_bytes = estimator.draft_block_bytes if draft else estimator.block_bytes
            for direction in ('H2D','D2H'):
                rows = [r for r in w.rows if direction == 'D2H' or r.source is not None]
                seconds += estimator._copy(direction,sum(r.capacity_blocks for r in rows)*2*block_bytes,len(rows),draft=draft)
        return now+seconds
