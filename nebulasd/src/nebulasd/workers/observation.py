"""Persistent execution facts and global control-owned scheduling projection."""
from struct import Struct
from nebulasd.core.enums import StateChangeBlockKind as K, WorkerRole, WorkerStatus
from nebulasd.table.prepared import PreparedRow


class Observation:
    """One execution writer; fixed Bank facts plus last model job host clocks."""
    codec = Struct('<' + 'QQIIII'*2 + 'QQQ')

    def __init__(self, descriptor=None):
        from nebulasd.ipc.mapped_segment import MappedSegment
        from nebulasd.ipc.native import library
        self.segment = MappedSegment.create(8+self.codec.size,'worker-observation:1') if descriptor is None else MappedSegment(descriptor)
        if self.segment.descriptor.schema != 'worker-observation:1':
            raise ValueError('worker observation ABI mismatch')
        self.native,self.seq,self.last = library(),0,None

    def write(self, runtime, clock):
        states={'FREE':0,'EXPORTING':1,'FILLING':2,'READY':3,'COMPUTING':4}
        values=[]
        for b in runtime.banks.banks:
            state=b.current
            values.extend((b.epoch,state.spec.work_seq if b.layout is not None else 0,states[b.phase.name],
                sum(b.layout.capacities) if b.layout is not None else 0,len(b.layout.rows) if b.layout is not None else 0,
                1 if b.phase.name=='EXPORTING' else 2 if state is not None and 'H2D' in state.jobs else 0))
        raw=self.codec.pack(*values,*clock)
        if raw == self.last:
            return
        self.native.sd_store(self.segment.address,self.seq+1)
        self.segment.buffer[8:] = raw
        self.seq+=2
        self.native.sd_store(self.segment.address,self.seq)
        self.last=raw

    def read(self):
        seq=self.native.sd_load(self.segment.address)
        if not seq or seq%2 or seq==self.seq:
            return None
        raw=bytes(self.segment.buffer[8:])
        if self.native.sd_load(self.segment.address)!=seq:
            return None
        self.seq=seq
        values=self.codec.unpack(raw)
        return (values[:6],values[6:12]),values[12:]

    def close(self,unlink=False):
        self.segment.close()
        if unlink:
            self.segment.unlink()


class Projection:
    def __init__(self, options, stack):
        from nebulasd.table.native_storage import worker_table, close_table_partitions
        from nebulasd.table.writers import WorkerRegistryWriter
        from nebulasd.ipc.native_ring import NativeStateChangeRing
        event = NativeStateChangeRing(options['event_capacity'], descriptor=options['event'], doorbell=options['engine_bell'])
        stack.callback(event.close)
        registry = worker_table(options['worker_count'], descriptors=options['global_registry'], ring=event)
        stack.callback(close_table_partitions, registry._partitions)
        self.observation = Observation(options['observation'])
        stack.callback(self.observation.close)
        w, draft = options['worker_id'], options['role'] == WorkerRole.DRAFT
        WorkerRegistryWriter(registry).publish_common(worker_row=w,publish_seq=options['worker_generation'],
            worker_id=w,role=options['role'],worker_generation=options['worker_generation'],status=WorkerStatus.ONLINE,
            command_consumer_seq=0,max_batch_size=options['max_batch_size'],max_batch_tokens=options['max_batch_tokens'])
        self.banks = tuple(PreparedRow(registry.partition(K.WORKER_DRAFT_BANK if draft else K.WORKER_BANK),w*2+i,
            dict(bank_id=i,capacity_blocks=options['blocks_per_bank'],capacity_rows=options['capacity_rows']),
            ('bank_epoch','batch_seq','state','alloc_ptr_blocks','alloc_rows','role')) for i in (0,1))
        self.runtime = PreparedRow(registry.partition(K.WORKER_DRAFT_RUNTIME if draft else K.WORKER_TARGET_COMPUTE_RUNTIME),w,{},
            ('current_batch_seq' if draft else 'compute_batch_seq','compute_status','compute_start_time_ns'))
        self.copy = PreparedRow(registry.partition(K.WORKER_DRAFT_COPY_RUNTIME if draft else K.WORKER_TARGET_COPY_RUNTIME),w,{},
            ('copy_op_seq','copy_status','copy_start_time_ns','copy_bytes'))
        self.block_bytes = options['host'].block_bytes

    def step(self):
        captured = self.observation.read()
        if captured is None:
            return False
        values,clock = captured
        for pub,(epoch,work,state,blocks,rows,direction) in zip(self.banks, values):
            pub.publish((epoch,work,state,blocks,rows,1 if state == 4 else 2))
        work,start,end = clock
        self.runtime.publish((work,int(bool(start and not end)),start))
        copying = [x for x in values if x[5]]
        self.copy.publish((max((x[1] for x in copying),default=0),
            copying[0][5] if copying else 0,0,sum(x[3]*2*self.block_bytes for x in copying)))
        return True


def execution_observation(options, stack):
    observation = Observation(options['observation'])
    stack.callback(observation.close)
    return observation
