"""WORK-event CPU publication; physical resource ownership stays in Runtime."""
from collections import deque
from time import perf_counter_ns
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.table.prepared import PreparedRow
from nebulasd.workers.completion import WorkCompletion, MemberCompletion
from nebulasd.workers.work import Outcome


class TargetPublisher:
    def __init__(self, table, completions, *, outputs, configs, block_bytes, block_size=16):
        self.table, self.completions = table, completions
        self.outputs, self.configs = outputs, configs
        self.records, self.order = {}, deque()
        self.block_size, self.block_bytes = block_size, block_bytes
        self.last_published_rows = 0
        self.doorbells = tuple({p._doorbell for k in (K.REQUEST_H2D,K.REQUEST_TARGET_COMPUTE,K.REQUEST_D2H)
                               for p in (table.partition(k),) if p._doorbell is not None})

    def reserve(self, work):
        for _ in self.reserve_steps(work):
            pass

    def reserve_steps(self, work):
        from nebulasd.workers.channel import LocalChannel
        bound = 1024 + sum(512 + 12*(r.token_budget + 1) for r in work.rows)
        if work.result_bytes < bound or bound > LocalChannel.WIDTH - 8:
            raise ValueError('WORK result frame budget is insufficient')
        if work.work_seq in self.records:
            raise ValueError('duplicate WORK result reservation')
        from nebulasd.workers.completion import completion_bytes
        self.completions._check(work.completion_offset,completion_bytes(len(work.rows)))
        if self.completions.native.sd_load(self.completions.segment.address+work.completion_offset):
            raise ValueError('Target completion slot already published')
        prepared = []
        for row in work.rows:
            common = dict(request_epoch=row.epoch,round_id=row.round_id,target_generation=work.worker_generation,
                          target_id=work.worker_id,result_code=0)
            imported = PreparedRow(self.table.partition(K.REQUEST_H2D),row.slot,
                common | dict(observed_prepare_seq=row.run_seq,destination_bank_epoch=work.bank_epoch,
                              destination_bank_id=work.bank_id,status=3),
                ('source_host_version','gpu_ready_version','copied_blocks','copy_start_time_ns','copy_bytes'))
            result = PreparedRow(self.table.partition(K.REQUEST_TARGET_COMPUTE),row.slot,
                common | dict(observed_run_seq=row.run_seq,bank_epoch=work.bank_epoch,bank_id=work.bank_id,status=2),
                ('target_kv_version','accepted_draft_count','committed_delta_count','last_committed_token',
                 'logical_kv_len','dirty_begin_block','dirty_block_count','committed_delta_handle',
                 'output_count','output_finished','output_handle','compute_start_ns','compute_end_ns'))
            host = PreparedRow(self.table.partition(K.REQUEST_D2H),row.slot,
                common | dict(d2h_op_seq=row.run_seq,source_bank_epoch=work.bank_epoch,source_bank_id=work.bank_id,
                              host_slot_generation=row.host_generation,writer_version=row.writer_generation,status=2),
                ('ready_version','committed_blocks','logical_kv_len','copy_start_time_ns','copy_bytes'))
            prepared.append((imported,result,host))
            yield
        self.records[work.work_seq] = dict(work=work,prepared=prepared,
            messages={}, imported=0,result=0,host=0,members=[])
        self.order.append(work.work_seq)

    def consume(self, message):
        """Accept a bounded fact; publication is advanced separately by step()."""
        kind,seq,data = message
        record = self.records[seq]
        if kind not in ('IMPORTED','RESULT','PHYSICAL') or kind in record['messages']:
            raise RuntimeError('invalid or duplicate Target fact')
        if kind == 'PHYSICAL' and len(data['outcomes']) != len(record['work'].rows):
            raise ValueError('completion member count mismatch')
        self._validate_members(record, kind, data)
        record['messages'][kind] = data

    def _validate_members(self, record, kind, data):
        if kind == 'PHYSICAL':
            return
        indices = [r['index'] for r in data['rows']]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(record['work'].rows) for i in indices):
            raise ValueError('publication members do not match WORK')

    def _validate_completion(self, record):
        result = record['messages'].get('RESULT', record['messages'].get('DRAFT_RESULT', {}))
        indices = sorted(out['index'] for out in result.get('rows', ()))
        executed = [i for i,o in enumerate(record['messages']['PHYSICAL']['outcomes']) if o == Outcome.EXECUTED]
        if indices != executed:
            raise ValueError('executed completion members differ from published results')

    def _publish_result(self, record):
        work = record['work']
        result = record['messages'].get('RESULT')
        rows = () if result is None else result['rows']
        if record['result'] < len(rows):
            out = rows[record['result']]
            i = out['index']
            tokens = tuple(out['tokens'])
            row = work.rows[i]
            if not tokens:
                raise ValueError('empty executed Target output')
            count = out['logical'] + 1 - row.prompt_count
            previous_count = count - len(tokens)
            previous = (self.table.partition(K.REQUEST_TARGET_COMPUTE).read_stable(row.slot)
                        if row.round_id else None)
            if row.round_id == 0:
                if previous_count != 0:
                    raise ValueError('initial output count mismatch')
            elif (previous.get('request_epoch') != row.epoch
                    or previous.get('round_id') != row.round_id-1
                    or previous.get('output_count') != previous_count
                    or previous.get('output_finished')):
                raise ValueError('output append predecessor mismatch')
            config = self.configs.read_config(row.config)
            if not 0 < count <= row.max_new_tokens or any(t in config.all_stop_token_ids for t in tokens[:-1]):
                raise ValueError('Target output violates stop/length contract')
            finished = count == row.max_new_tokens or tokens[-1] in config.all_stop_token_ids
            arena = self.outputs.arenas[row.output.generation]
            prefix = arena.append_reserved(row.output, previous_count, tokens)
            from nebulasd.core.handles import ArenaHandle
            handle = ArenaHandle(prefix.offset+4*previous_count, 4*len(tokens), prefix.generation)
            trace = getattr(self, 'handoff_trace', None)
            begin = perf_counter_ns() if trace is not None and trace.enabled else 0
            record['prepared'][i][1].publish((out['version'],out['accepted'],len(tokens),tokens[-1],
                out['logical'],out['dirty_begin'],out['dirty_blocks'],handle,
                count,int(finished),prefix,result['compute_start_ns'],result['compute_end_ns']))
            if begin:
                trace.mark('ROW_PUBLISHED', work.work_seq, i, begin)
            record['result'] += 1
            self.last_published_rows += 1
            return True
        return False

    def _advance(self, record):
        """One row (or one completion commit) maximum; return progressed/retired."""
        messages,work = record['messages'],record['work']
        imported = messages.get('IMPORTED')
        if imported is not None and record['imported'] < len(imported['rows']):
            row = imported['rows'][record['imported']]
            record['prepared'][row['index']][0].publish((row['version'],row['version'],row['blocks'],
                imported['submitted_ns'],2*self.block_bytes*row['blocks']))
            record['imported'] += 1
            self.last_published_rows += 1
            return True,False
        result = messages.get('RESULT')
        rows = () if result is None else result['rows']
        if self._publish_result(record):
            return True,False
        physical = messages.get('PHYSICAL')
        if physical is None or (result is None and Outcome.EXECUTED in physical['outcomes']):
            return False,False
        if record['host'] < len(rows):
            out = rows[record['host']]
            record['prepared'][out['index']][2].publish((out['version'],
                (out['logical']+self.block_size-1)//self.block_size,out['logical'],
                physical['d2h_submitted_ns'],2*self.block_bytes*out['dirty_blocks']))
            record['host'] += 1
            self.last_published_rows += 1
            return True,False
        members = record['members']
        if len(members) < len(work.rows):
            i = len(members)
            row = work.rows[i]
            members.append(MemberCompletion(row.slot,row.epoch,row.round_id,Outcome(physical['outcomes'][i])))
            return True,False
        self._validate_completion(record)
        self.completions.publish(work.completion_offset,WorkCompletion(work.worker_generation,work.work_seq,
            work.bank_epoch,physical['observed_ns'],work.bank_id,tuple(members)))
        return True,True

    def publish_results(self, seq):
        """Drain bounded result rows immediately; bank/host retirement is separate."""
        record = self.records[seq]
        published = False
        while self._publish_result(record):
            published = True
        if published:
            for bell in self.doorbells:
                bell.ring()
        return published

    def _event(self, record):
        messages = record['messages']
        for kind, cursor in (('IMPORTED', 'imported'), ('DRAFT_IMPORTED', 'imported'),
                             ('RESULT', 'result'), ('DRAFT_RESULT', 'result')):
            data = messages.get(kind)
            if data is not None and record[cursor] < len(data['rows']):
                return cursor, len(data['rows'])
        physical = messages.get('PHYSICAL')
        result = messages.get('RESULT', messages.get('DRAFT_RESULT'))
        if physical is None or (result is None and Outcome.EXECUTED in physical['outcomes']):
            return None
        count = len(result['rows']) if result is not None else 0
        return ('host', count) if record['host'] < count else ('members', len(record['work'].rows))

    def step(self, event_budget=1):
        """Round-robin WORK events; never yield halfway through a ready event.

        Member counts come from the actual receipt, including skipped members.
        Retirement remains a separate event so HostReady can be sent first.
        """
        if event_budget <= 0:
            raise ValueError('publication event budget must be positive')
        self.last_published_rows = 0
        retired, idle = [],0
        while event_budget and self.order and idle < len(self.order):
            seq = self.order.popleft()
            record = self.records[seq]
            event = self._event(record)
            progressed, done = False, False
            if event is not None:
                cursor, count = event
                while True:
                    progressed, done = self._advance(record)
                    if done or (cursor != 'members' and record[cursor] == count):
                        break
            if done:
                del self.records[seq]
                retired.append(seq)
            else:
                self.order.append(seq)
            if progressed:
                event_budget -= 1
                idle = 0
            else:
                idle += 1
        # Row hints remain loss-tolerant; shared table/completion hold the facts.
        # Wake each Engine doorbell at most once after these complete events.
        if self.last_published_rows or retired:
            for bell in self.doorbells:
                bell.ring()
        return tuple(retired)

    def take_times(self):
        return ()
