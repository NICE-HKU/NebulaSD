"""Control-owned registered native reads; all memory remains owned by the table.

Registration validates addresses/field layouts once. A scan checks every bounded
registered row in C and returns only changed stable snapshots. Python reference
tables retain the ordinary Dependencies implementation used by protocol tests.
"""
import ctypes as C
from struct import Struct
from nebulasd.core.handles import ArenaHandle

class Snapshot(dict):
    __slots__ = ('publish_seq',)

class NativeDependencies:
    def __init__(self, table, fields, rules):
        self.table, self.fields, self.rules = table, fields, rules
        self.registrations = None
        self.dirty = True
        self.keys = ()
        self.layouts = {}
        for kind in {int(rule[0]) for rule in rules.values()}:
            p = table.partition(kind)
            names = tuple(dict.fromkeys(name for selector, rule in rules.items()
                if int(rule[0]) == kind for name in fields[selector]))
            dec = []
            for name in names:
                typ=p._field_by_name[name].type
                fmt='<QII' if typ.name=='arena_handle' else '<'+{(1,False):'B',(4,False):'I',(4,True):'i',(8,False):'Q'}[typ.size,typ.signed]
                dec.append((name,Struct(fmt),p._offset_by_name[name]-8,typ.name=='arena_handle'))
            self.layouts[kind] = tuple(dec)

    def rebuild(self, pending):
        self.registrations = tuple(pending)
        self.dirty = False
        groups = {}
        for key, watch in pending.items():
            dep=watch.dependency;identity=(dep.kind,dep.slot)
            groups.setdefault(identity,[]).append(watch)
        self.keys=tuple(groups)
        self.watches=tuple(groups.values())
        n=len(groups)
        parts=[self.table.partition(kind) for kind,slot in self.keys]
        for p,(_,slot) in zip(parts,self.keys):p._validate_row(slot)
        self.native=parts[0].native if n else None
        self.stride=max((p._payload_size-8 for p in parts),default=1)
        self.addresses=(C.c_size_t*n)(*(p.segment.address+p._row_base(slot) for p,(_,slot) in zip(parts,self.keys)))
        self.sizes=(C.c_uint*n)(*(p._payload_size for p in parts))
        self.previous=(C.c_uint64*n)(*((1<<64)-1 for _ in parts))
        self.sequences=(C.c_uint64*n)();self.changed=(C.c_uint*n)()
        self.output=C.create_string_buffer(n*self.stride)
        self.decoders = [self.layouts[kind] for kind, slot in self.keys]

    def scan(self, pending):
        if not pending:return {}
        if self.dirty:self.rebuild(pending)
        if not self.keys:return {}
        count=self.native.sd_watch_scan(self.addresses,self.sizes,self.previous,len(self.keys),self.output,self.stride,self.sequences,self.changed)
        snapshots={}
        for j in range(count):
            i=self.changed[j];row=Snapshot();row.publish_seq=self.sequences[i]
            offset=i*self.stride
            for name,codec,position,handle in self.decoders[i]:
                values=codec.unpack_from(self.output,offset+position)
                row[name]=ArenaHandle(*values) if handle else values[0]
            for watch in self.watches[i]:
                if watch.last_seq!=row.publish_seq:snapshots[watch.key]=row
        return snapshots
