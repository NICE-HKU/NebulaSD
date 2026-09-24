"""Draft payload publication joins physical completion without owning GPU resources."""
from collections import deque
from time import perf_counter_ns
from struct import Struct
from nebulasd.core.draft_contracts import DraftSnapshot, DraftSnapshotIdentity, _IDENTITY, _HANDLE
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.table.prepared import PreparedRow
from nebulasd.workers.target.publication import TargetPublisher
from nebulasd.workers.completion import WorkCompletion, MemberCompletion
from nebulasd.workers.work import Outcome
from .inputs import allocation


def reserve_payloads(batches):
    """Single writer: precheck every arena before moving any head. No rollback."""
    if len({id(a) for a, lengths in batches}) != len(batches):
        raise ValueError('duplicate output arena')
    for arena, lengths in batches:
        arena._before_write()
        if any(n <= 0 for n in lengths) or arena._head + sum(lengths) > arena.capacity_bytes:
            raise ValueError('Draft output arena capacity exhausted')
    result = []
    for arena, lengths in batches:
        handles = []
        for n in lengths:
            handles.append(ArenaHandle(arena._head, n, arena.generation))
            arena._head += n
        # Reserved bytes are not reachable from result tables until written.
        arena._commit()
        result.append(tuple(handles))
    return tuple(result)


class DraftPublisher(TargetPublisher):
    # Reuse the complete-event fair loop and publication query contract.
    def __init__(self, table, proposals, snapshots, completions, *, host, block_size=16, profile=False):
        self.table, self.proposals, self.snapshots, self.completions = table, proposals, snapshots, completions
        self.host = host
        self.profile = profile
        self.times = [] if profile else None
        self.block_size, self.block_bytes = block_size, host.block_bytes
        self.records, self.order = {}, deque()
        self.last_published_rows = 0
        self.doorbells = tuple({p._doorbell for k in (K.REQUEST_DRAFT_H2D,K.REQUEST_DRAFT,K.REQUEST_DRAFT_D2H)
            for p in (table.partition(k),) if p._doorbell is not None})

    def reserve(self, work):
        for _ in self.reserve_steps(work):
            pass

    def reserve_steps(self, work):
        from nebulasd.workers.channel import LocalChannel
        bound = 1024 + sum(512+4*r.token_budget for r in work.rows)
        if work.result_bytes < bound or bound > LocalChannel.WIDTH or work.work_seq in self.records:
            raise ValueError('invalid Draft output frame reservation')
        from nebulasd.workers.completion import completion_bytes
        self.completions._check(work.completion_offset, completion_bytes(len(work.rows)))
        if self.completions.native.sd_load(self.completions.segment.address+work.completion_offset):
            raise ValueError('Draft completion slot already published')
        prepared = []
        for r in work.rows:
            alloc = allocation(r, self.host.descriptor_generation, self.block_size)
            fields = {n:getattr(alloc,n) for n in alloc.__dataclass_fields__}
            imp = PreparedRow(self.table.partition(K.REQUEST_DRAFT_H2D), r.slot,
                fields | dict(request_epoch=r.epoch, next_round_id=r.round_id, observed_prepare_seq=r.run_seq,
                    destination_worker_id=work.worker_id,destination_worker_generation=work.worker_generation,
                    next_owner_epoch=r.owner_epoch,destination_bank_id=work.bank_id,
                    destination_bank_epoch=work.bank_epoch,prepared_batch_seq=work.work_seq,status=3,result_code=0),
                ('snapshot_version','gpu_ready_version','copied_blocks','copy_start_time_ns','copy_bytes',
                 'snapshot_round_id','logical_kv_len','local_row','snapshot_handle'))
            result = PreparedRow(self.table.partition(K.REQUEST_DRAFT),r.slot,
                dict(request_epoch=r.epoch,round_id=r.round_id,observed_issue_seq=r.run_seq,
                    worker_id=work.worker_id,worker_generation=work.worker_generation,owner_epoch=r.owner_epoch,
                    bank_id=work.bank_id,bank_epoch=work.bank_epoch,batch_seq=work.work_seq,status=2,result_code=0),
                ('snapshot_version','logical_kv_len','valid_blocks','draft_state_handle','proposal_handle',
                 'proposal_token_count','dirty_begin_block','dirty_block_count','compute_start_ns','compute_end_ns'))
            host = PreparedRow(self.table.partition(K.REQUEST_DRAFT_D2H),r.slot,
                fields | dict(request_epoch=r.epoch,snapshot_round_id=r.round_id,source_op_seq=r.run_seq,
                    source_worker_id=work.worker_id,source_worker_generation=work.worker_generation,
                    owner_epoch=r.owner_epoch,source_bank_id=work.bank_id,source_bank_epoch=work.bank_epoch,
                    source_batch_seq=work.work_seq,status=2,result_code=0),
                ('snapshot_version','ready_version','snapshot_handle','logical_kv_len','valid_blocks',
                 'copy_start_time_ns','copy_bytes'))
            prepared.append((imp,result,host))
            yield
        proposal_handles, snapshot_handles = reserve_payloads(((self.proposals,tuple(8+4*r.token_budget for r in work.rows)),
            (self.snapshots,(DraftSnapshot.byte_size,)*len(work.rows))))
        templates = []
        for row, proposal in zip(work.rows, proposal_handles):
            alloc = allocation(row, self.host.descriptor_generation, self.block_size)
            logical = row.prompt_count + 1
            identity = DraftSnapshotIdentity(row.slot, row.epoch, row.round_id, row.run_seq,
                work.worker_id, work.worker_generation, row.owner_epoch, 0, logical,
                (logical+self.block_size-1)//self.block_size, alloc)
            snapshot = DraftSnapshot(identity, row.prompt, row.config,
                ArenaHandle(row.output.offset, 4, row.output.generation),
                ArenaHandle(proposal.offset, 12, proposal.generation), row.prompt_count, 1, 1)
            templates.append(bytearray(snapshot.to_bytes()))
            yield
        self.records[work.work_seq] = dict(work=work,prepared=prepared,proposals=proposal_handles,
            snapshot_templates=templates,
            snapshots=snapshot_handles,messages={},imported=0,result=0,host=0,members=[])
        self.order.append(work.work_seq)

    def consume(self, message):
        kind,seq,data = message
        record = self.records[seq]
        if kind not in ('DRAFT_IMPORTED','DRAFT_RESULT','PHYSICAL') or kind in record['messages']:
            raise ValueError('invalid or duplicate Draft fact')
        if kind == 'PHYSICAL' and len(data['outcomes']) != len(record['work'].rows):
            raise ValueError('Draft completion member count mismatch')
        self._validate_members(record, kind, data)
        record['messages'][kind] = data

    def _publish_result(self, record):
        work = record['work']
        result = record['messages'].get('DRAFT_RESULT')
        rows = () if result is None else result['rows']
        if record['result'] < len(rows):
            out = rows[record['result']]
            i = out['index']
            row = work.rows[i]
            tokens = out['proposal']
            if not 0 < len(tokens) <= row.token_budget or out['committed_count']*4 > row.output.length:
                raise ValueError('Draft result exceeds WORK reservation')
            valid = (out['logical']+self.block_size-1)//self.block_size
            if out['dirty_begin']+out['dirty_blocks'] != valid:
                raise ValueError('Draft dirty range must cover valid tail')
            raw = Struct('<II').pack(out['proposal_kind'],len(tokens))+Struct(f'<{len(tokens)}I').pack(*tokens)
            reserved = record['proposals'][i]
            proposal = ArenaHandle(reserved.offset,len(raw),reserved.generation)
            self.proposals._bytes[proposal.offset:proposal.end_offset] = raw
            # Static identity/allocation/handles were checked at reserve. Only
            # result-dependent counts and ranges remain on the handoff path.
            if (out['committed_count'] <= 0 or out['proposal_kind'] != 1
                    or out['logical'] != row.prompt_count + out['committed_count'] + len(tokens) - 1
                    or valid > row.host_capacity):
                raise ValueError('Draft snapshot result identity/count mismatch')
            raw_snapshot = record['snapshot_templates'][i]
            DraftSnapshot._header.pack_into(raw_snapshot, 0, DraftSnapshot.version,
                row.prompt_count, out['committed_count'], len(tokens))
            Struct('<QII').pack_into(raw_snapshot, DraftSnapshot._header.size + 48,
                out['version'], out['logical'], valid)
            handles = DraftSnapshot._header.size + DraftSnapshotIdentity.byte_size
            _HANDLE.pack_into(raw_snapshot, handles + 32,
                row.output.offset, 4*out['committed_count'], row.output.generation)
            _HANDLE.pack_into(raw_snapshot, handles + 48,
                proposal.offset, proposal.length, proposal.generation)
            handle = record['snapshots'][i]
            self.snapshots._bytes[handle.offset:handle.end_offset] = raw_snapshot
            trace = getattr(self, 'handoff_trace', None)
            begin = perf_counter_ns() if trace is not None and trace.enabled else 0
            record['prepared'][i][1].publish((out['version'],out['logical'],valid,handle,proposal,
                len(tokens),out['dirty_begin'],out['dirty_blocks'],result['compute_start_ns'],result['compute_end_ns']))
            if begin:
                trace.mark('ROW_PUBLISHED', work.work_seq, i, begin)
            record['result'] += 1
            self.last_published_rows += 1
            return True
        return False

    def _advance(self, record):
        messages, work = record['messages'], record['work']
        imported = messages.get('DRAFT_IMPORTED')
        if imported is not None and record['imported'] < len(imported['rows']):
            out = imported['rows'][record['imported']]
            record['prepared'][out['index']][0].publish((out['version'],out['version'],out['blocks'],
                imported['submitted_ns'],2*self.block_bytes*out['blocks'],out['snapshot_round'],out['logical'],
                out['local_row'],out['snapshot_handle']))
            record['imported'] += 1
            self.last_published_rows += 1
            return True,False
        result = messages.get('DRAFT_RESULT')
        rows = () if result is None else result['rows']
        if self._publish_result(record):
            return True,False
        physical = messages.get('PHYSICAL')
        if physical is None or (result is None and Outcome.EXECUTED in physical['outcomes']):
            return False,False
        if record['host'] < len(rows):
            out = rows[record['host']]
            record['prepared'][out['index']][2].publish((out['version'],out['version'],record['snapshots'][out['index']],
                out['logical'],(out['logical']+self.block_size-1)//self.block_size,
                physical['d2h_submitted_ns'],2*self.block_bytes*out['dirty_blocks']))
            if self.profile:
                self.times.append((work.work_seq,'CONTROL_HOST_READY',perf_counter_ns()))
            record['host'] += 1
            self.last_published_rows += 1
            return True,False
        if len(record['members']) < len(work.rows):
            i = len(record['members'])
            r = work.rows[i]
            record['members'].append(MemberCompletion(r.slot,r.epoch,r.round_id,Outcome(physical['outcomes'][i])))
            return True,False
        self._validate_completion(record)
        self.completions.publish(work.completion_offset,WorkCompletion(work.worker_generation,work.work_seq,
            work.bank_epoch,physical['observed_ns'],work.bank_id,tuple(record['members'])))
        return True,True

    def take_times(self):
        if not self.profile:
            return ()
        times,self.times=self.times,[]
        return times
