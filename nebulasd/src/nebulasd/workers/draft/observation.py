"""Coalesced execution facts, never a resource allocator or phase authorization.

A private latest-value seqlock avoids adding phase traffic to the result ring.
Only control publishes the compatible public Bank/copy partitions.
"""
from struct import Struct
from nebulasd.ipc.mapped_segment import MappedSegment
from nebulasd.ipc.native import library
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.table.prepared import PreparedRow

_BANK = Struct('<QQIIII')  # epoch, work, state, allocated blocks/rows, copy direction


class Observation:
    def __init__(self, descriptor=None):
        self.segment = MappedSegment.create(8+2*_BANK.size,'draft-observation:1') if descriptor is None else MappedSegment(descriptor)
        if self.segment.descriptor.schema != 'draft-observation:1':
            raise ValueError('invalid Draft observation schema')
        self.native = library()
        self.last = None
        self.seq = 0

    def write(self, banks):
        state_ids={'FREE':0,'FILLING':2,'READY':3,'COMPUTING':4,'EXPORTING':1}
        values=tuple((b.epoch,b.current.spec.work_seq if b.layout is not None else 0,state_ids[b.phase.name],
            sum(b.layout.capacities) if b.layout is not None else 0,len(b.layout.rows) if b.layout is not None else 0,
            1 if b.phase.name=='EXPORTING' else 2 if b.current is not None and 'H2D' in b.current.jobs else 0)
            for b in banks.banks)
        if values == self.last:
            return
        self.native.sd_store(self.segment.address,self.seq+1)
        for i,v in enumerate(values):
            _BANK.pack_into(self.segment.buffer,8+i*_BANK.size,*v)
        self.seq+=2
        self.native.sd_store(self.segment.address,self.seq)
        self.last=values

    def read(self):
        seq=self.native.sd_load(self.segment.address)
        if not seq or seq%2 or seq==self.seq:
            return None
        raw=bytes(self.segment.buffer[8:])
        if self.native.sd_load(self.segment.address)!=seq:
            return None
        self.seq=seq
        return tuple(_BANK.unpack_from(raw,i*_BANK.size) for i in range(2))

    def close(self, unlink=False):
        self.segment.close()
        if unlink:
            self.segment.unlink()


class Projection:
    def __init__(self, registry, observation, blocks, rows):
        self.observation=observation
        self.banks=tuple(PreparedRow(registry.partition(K.WORKER_DRAFT_BANK),i,
            dict(bank_id=i,capacity_blocks=blocks,capacity_rows=rows),
            ('bank_epoch','batch_seq','state','alloc_ptr_blocks','alloc_rows','role')) for i in range(2))
        self.copy=PreparedRow(registry.partition(K.WORKER_DRAFT_COPY_RUNTIME),0,
            dict(copy_start_time_ns=0,copy_bytes=0),('copy_op_seq','copy_status'))

    def step(self):
        values=self.observation.read()
        if values is None:
            return False
        for pub,(epoch,work,state,blocks,rows,direction) in zip(self.banks,values):
            pub.publish((epoch,work,state,blocks,rows,1 if state==4 else 2))
        # Legacy scalar is an aggregate hint, not a two-lane occupancy ledger.
        active=[(work,direction) for epoch,work,state,blocks,rows,direction in values if direction]
        self.copy.publish(max(active,default=(0,0)))
        return True
