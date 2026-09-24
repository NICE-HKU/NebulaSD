"""Single-owner native observation; Engine.rows owns all accepted payloads.

Native retains only versions, pending hint indices and bounded scan cursors.
No Python pending/seen/view mirror is maintained on this path.
Each poll returns at most one latest stable payload per (kind, row); capture
reuses its output slot when the same row changes twice during that poll.
"""
import ctypes as C
import weakref
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.engine.local_state import PayloadDecoder, PayloadRow
from nebulasd.ipc.native import library
from .reader import TableUpdateBatch, ENGINE_IGNORED_KINDS

class Partition(C.Structure):
    _fields_ = [('base', C.c_size_t)] + [(n,C.c_uint) for n in ('stride','size','count','kind')]
class Ring(C.Structure):
    _fields_ = [('base', C.c_size_t),('capacity',C.c_uint64)]
class Output(C.Structure):
    _fields_ = [('kind',C.c_uint),('row',C.c_uint),('seq',C.c_uint64),('size',C.c_uint),('reserved',C.c_uint)]

class NativeTableReader:
    def __init__(self, *, request_table, worker_registry, rings, priority_rows, max_entries=128):
        if max_entries<=0:raise ValueError('observation budget must be positive')
        self.max_entries=max_entries
        self.partitions=tuple(p for table in (request_table,worker_registry) for p in table._partitions.values()
                              if p.block_kind not in ENGINE_IGNORED_KINDS)
        self.rings=rings  # Keep mappings alive until native registration is destroyed.
        self.codecs={int(p.block_kind):PayloadDecoder(p).decode for p in self.partitions}
        self.kinds={int(p.block_kind):p.block_kind for p in self.partitions}
        lib=self.lib=library()
        lib.sd_reader_create.argtypes=[C.POINTER(Partition),C.c_uint,C.POINTER(Ring),C.c_uint,C.POINTER(C.c_uint),C.c_uint]
        lib.sd_reader_create.restype=C.c_void_p
        lib.sd_reader_destroy.argtypes=[C.c_void_p];lib.sd_reader_destroy.restype=None
        lib.sd_reader_error.argtypes=[];lib.sd_reader_error.restype=C.c_char_p
        lib.sd_reader_poll.argtypes=[C.c_void_p,C.c_uint,C.POINTER(Output),C.c_void_p,C.c_uint,C.POINTER(C.c_uint),C.POINTER(C.c_uint)]
        lib.sd_reader_poll.restype=C.c_int
        lib.sd_reader_pending.argtypes=[C.c_void_p];lib.sd_reader_pending.restype=C.c_int
        lib.sd_reader_state.argtypes=[C.c_void_p];lib.sd_reader_state.restype=C.c_uint
        lib.sd_reader_rescan.argtypes=[C.c_void_p];lib.sd_reader_rescan.restype=None
        lib.sd_reader_reset.argtypes=[C.c_void_p,C.POINTER(C.c_uint),C.c_uint];lib.sd_reader_reset.restype=None
        ps=(Partition*len(self.partitions))(*(Partition(p.segment.address,p._row_stride,p._payload_size,p.capacity_rows,int(p.block_kind)) for p in self.partitions))
        rs=(Ring*len(rings))(*(Ring(r.address,r.capacity) for r in rings))
        keys=(C.c_uint*(2*len(priority_rows)))(*(x for pair in priority_rows for x in pair))
        self.handle=lib.sd_reader_create(ps,len(ps),rs,len(rs),keys,len(priority_rows))
        if not self.handle:raise RuntimeError(lib.sd_reader_error().decode())
        self._finalizer=weakref.finalize(self,lib.sd_reader_destroy,self.handle)
        self.stride=max(p._payload_size-8 for p in self.partitions)
        self.output=(Output*(max_entries+len(priority_rows)))()
        self.payload=C.create_string_buffer(len(self.output)*self.stride)
        self.bytes=memoryview(self.payload).cast('B')
        self.scanned,self.overflow=C.c_uint(),C.c_uint()
        self.on_read=None

    def poll(self):
        n=self.lib.sd_reader_poll(self.handle,self.max_entries,self.output,self.payload,self.stride,
                                  C.byref(self.scanned),C.byref(self.overflow))
        if n<0:raise RuntimeError(self.lib.sd_reader_error().decode())
        if self.on_read is not None:self.on_read(self.output,n)
        rows=tuple(PayloadRow(self.kinds[o.kind],o.row,o.seq,
                    bytes(self.bytes[i*self.stride:i*self.stride+o.size]),self.codecs[o.kind],{})
                   for i,o in enumerate(self.output[:n]))
        return TableUpdateBatch(rows,bool(self.overflow.value or self.scanned.value),self.scanned.value)

    def has_pending(self):
        return bool(self.lib.sd_reader_pending(self.handle))

    def reset_requests(self, kinds):
        kinds=tuple(kinds);values=(C.c_uint*len(kinds))(*kinds)
        self.lib.sd_reader_reset(self.handle,values,len(values))

    def reset_pending(self):
        self.lib.sd_reader_reset(self.handle,None,0)

    # Read-only native diagnostics; no mirrored Python retry state.
    @property
    def _pending(self):return bool(self.lib.sd_reader_state(self.handle)&1)
    @property
    def _scan(self):return bool(self.lib.sd_reader_state(self.handle)&2)
    @property
    def _rescan(self):return bool(self.lib.sd_reader_state(self.handle)&4)
    @_rescan.setter
    def _rescan(self, value):
        if not value:raise ValueError('only the retirement barrier may clear recovery')
        self.lib.sd_reader_rescan(self.handle)
