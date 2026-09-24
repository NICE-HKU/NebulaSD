"""Worker-local SPSC: 16-byte commit header plus reserved variable-length payload.

Only actual payload bytes are copied. Cold controls have no payload. Immutable
binary records carry host results; CUDA objects never cross this channel.
"""
from dataclasses import dataclass
from contextlib import ExitStack
from queue import Empty, Full
from struct import Struct
from nebulasd.ipc.native_ring import NativeRing
from nebulasd.ipc.mapped_segment import MappedSegment

_HEADER = Struct('<IIQ')
_U32 = Struct('<I')
_U64 = Struct('<Q')
_RESULT = Struct('<QQI')
_ROW = Struct('<IIIQIII')
_DRAFT_IMPORT = Struct('<IQIQIIQII')
_DRAFT_ROW = Struct('<IIIQIIII')
_IMPORT = Struct('<IQI')
_PHYSICAL = Struct('<QQII')
_FACT = Struct('<IQ')
_HOST_ROW = Struct('<IQI')
_KINDS = {'WORK':1, 'RETIRE_RECORD':2, 'DRAIN':3, 'SHUTDOWN':4, 'RESULT':5, 'PHYSICAL':6, 'IMPORTED':7, 'DRAFT_RESULT':8, 'DRAFT_IMPORTED':9, 'INPUTS':10, 'COMPUTE':11, 'COMPUTED':12}
_NAMES = {v:k for k,v in _KINDS.items()}
_FACTS = ('WORK_ACCEPTED','LAYOUT_ALLOCATED','DEPENDENCY_OBSERVED','IMPORT_INPUT_DONE',
    'COMPILE_DONE','METADATA_DONE','H2D_SUBMITTED','H2D_DONE','COMPUTE_SELECTION_OBSERVED',
    'COMPUTE_SUBMITTED','COMPUTE_DONE','D2H_SUBMITTED','D2H_DONE','BANK_FREE','WORK_PHYSICALLY_DONE','RESULT_READY',
    'CONTROL_RESULT_RECEIVED','CONTROL_RESULT_PUBLISHED','CONTROL_WORK_PUBLISHED',
    'CONTROL_PUBLICATION_BUSY_BEGIN','CONTROL_PUBLICATION_BUSY_END','CONTROL_HOST_READY',
    'H2D_LAUNCH','H2D_OBSERVED','H2D_CPU_NS','D2H_LAUNCH','D2H_OBSERVED','D2H_CPU_NS',
    'SOURCE_CONTROL_CAPTURED','SOURCE_RECEIVED','PREDECESSOR_CONTROL_CAPTURED','PREDECESSOR_RECEIVED',
    'CLASSIFIED_CONTROL_CAPTURED','CLASSIFIED_RECEIVED')
_FACT_IDS = {name:i for i,name in enumerate(_FACTS)}


@dataclass(frozen=True)
class ChannelDescriptor:
    ring: object
    payload: object


def value(row, key):
    return row[key] if isinstance(row, dict) else getattr(row, key)


def encode(kind, data):
    if kind in ('COMPUTE', 'COMPUTED'):
        import pickle
        return pickle.dumps(data, protocol=5)
    if kind == 'INPUTS':
        from .input_facts import encode_inputs
        return encode_inputs(data)
    if kind == 'WORK':
        return data
    if kind == 'IMPORTED':
        rows = data['rows']
        return Struct('<QI').pack(data['submitted_ns'], len(rows)) + b''.join(
            _IMPORT.pack(value(r,'index'), value(r,'version'), value(r,'blocks')) for r in rows)
    if kind == 'RESULT':
        chunks = [_RESULT.pack(data['compute_start_ns'], data['compute_end_ns'], len(data['rows']))]
        for row in data['rows']:
            tokens = value(row,'tokens')
            chunks.append(_ROW.pack(value(row,'index'), value(row,'accepted'), value(row,'logical'),
                value(row,'version'), value(row,'dirty_begin'), value(row,'dirty_blocks'), len(tokens)))
            chunks.append(Struct(f'<{len(tokens)}I').pack(*tokens))
        return b''.join(chunks)
    if kind == 'DRAFT_IMPORTED':
        chunks = [Struct('<QI').pack(data['submitted_ns'],len(data['rows']))]
        for row in data['rows']:
            h = value(row,'snapshot_handle')
            chunks.append(_DRAFT_IMPORT.pack(*(value(row,k) for k in
                ('index','version','blocks','snapshot_round','logical','local_row')),h.offset,h.length,h.generation))
        return b''.join(chunks)
    if kind == 'DRAFT_RESULT':
        chunks = [_RESULT.pack(data['compute_start_ns'], data['compute_end_ns'], len(data['rows']))]
        for row in data['rows']:
            tokens = value(row, 'proposal')
            chunks.append(_DRAFT_ROW.pack(value(row,'index'), value(row,'proposal_kind'), value(row,'logical'),
                value(row,'version'), value(row,'dirty_begin'), value(row,'dirty_blocks'),
                value(row,'committed_count'), len(tokens)))
            chunks.append(Struct(f'<{len(tokens)}I').pack(*tokens))
        return b''.join(chunks)
    if kind == 'PHYSICAL':
        outcomes, facts = data['outcomes'], data.get('facts', ())
        return (_PHYSICAL.pack(data['d2h_submitted_ns'], data['observed_ns'], len(outcomes), len(facts))
            + bytes(outcomes) + b''.join(_FACT.pack(_FACT_IDS[k], t) for k,t in facts))
    return b''


def decode(kind, raw):
    if kind in ('COMPUTE', 'COMPUTED'):
        import pickle
        return pickle.loads(raw)
    if kind == 'INPUTS':
        from .input_facts import decode_inputs
        return decode_inputs(raw)
    if kind == 'WORK':
        return raw
    if kind == 'IMPORTED':
        submitted, count = Struct('<QI').unpack_from(raw)
        if len(raw) != 12 + count * _IMPORT.size:
            raise ValueError('invalid IMPORTED payload size')
        rows = []
        for i in range(count):
            index, version, blocks = _IMPORT.unpack_from(raw, 12 + i*_IMPORT.size)
            rows.append(dict(index=index, version=version, blocks=blocks))
        return dict(submitted_ns=submitted, rows=rows)
    if kind == 'RESULT':
        start, end, count = _RESULT.unpack_from(raw)
        pos, rows = _RESULT.size, []
        for _ in range(count):
            index, accepted, logical, version, dirty, blocks, n = _ROW.unpack_from(raw, pos)
            pos += _ROW.size
            if n > (len(raw)-pos)//4:
                raise ValueError('truncated result tokens')
            tokens = Struct(f'<{n}I').unpack_from(raw, pos)
            pos += 4*n
            rows.append(dict(index=index, accepted=accepted, logical=logical, version=version,
                             dirty_begin=dirty, dirty_blocks=blocks, tokens=tokens))
        if pos != len(raw):
            raise ValueError('trailing result bytes')
        return dict(compute_start_ns=start, compute_end_ns=end, rows=rows)
    if kind == 'DRAFT_IMPORTED':
        from nebulasd.core.handles import ArenaHandle
        if len(raw) < 12:
            raise ValueError('truncated Draft import')
        submitted,count = Struct('<QI').unpack_from(raw)
        if count > 256 or len(raw) != 12+count*_DRAFT_IMPORT.size:
            raise ValueError('invalid Draft import length')
        rows=[]
        for i in range(count):
            values=_DRAFT_IMPORT.unpack_from(raw,12+i*_DRAFT_IMPORT.size)
            rows.append(dict(zip(('index','version','blocks','snapshot_round','logical','local_row'),values[:6]))|
                {'snapshot_handle':ArenaHandle(*values[6:])})
        return dict(submitted_ns=submitted,rows=rows)
    if kind == 'DRAFT_RESULT':
        if len(raw) < _RESULT.size:
            raise ValueError('truncated Draft result header')
        start, end, count = _RESULT.unpack_from(raw)
        if count > 256:
            raise ValueError('Draft result member bound exceeded')
        pos, rows = _RESULT.size, []
        for _ in range(count):
            if len(raw)-pos < _DRAFT_ROW.size:
                raise ValueError('truncated Draft result row')
            index, proposal_kind, logical, version, dirty, blocks, committed, n = _DRAFT_ROW.unpack_from(raw,pos)
            pos += _DRAFT_ROW.size
            if index >= 256 or proposal_kind != 1 or not n or n > (len(raw)-pos)//4:
                raise ValueError('invalid Draft proposal')
            tokens = Struct(f'<{n}I').unpack_from(raw,pos)
            pos += n*4
            rows.append(dict(index=index,proposal_kind=proposal_kind,logical=logical,version=version,
                dirty_begin=dirty,dirty_blocks=blocks,committed_count=committed,proposal=tokens))
        if pos != len(raw) or len({r['index'] for r in rows}) != len(rows):
            raise ValueError('invalid Draft result byte count or duplicate member')
        return dict(compute_start_ns=start,compute_end_ns=end,rows=rows)
    if kind == 'PHYSICAL':
        submit, done, count, nfacts = _PHYSICAL.unpack_from(raw)
        pos = _PHYSICAL.size
        if len(raw) != pos + count + nfacts*_FACT.size:
            raise ValueError('invalid PHYSICAL payload size')
        outcomes = tuple(raw[pos:pos+count])
        pos += count
        facts = []
        for i in range(nfacts):
            key, time = _FACT.unpack_from(raw, pos+i*_FACT.size)
            facts.append((_FACTS[key], time))
        return dict(d2h_submitted_ns=submit, observed_ns=done, outcomes=outcomes, facts=facts)
    if raw:
        raise ValueError('control must not have payload')
    return None


class LocalChannel:
    CAPACITY = 16
    WIDTH = 262144  # Reserved capacity per slot, NOT transmitted frame size.

    def __init__(self, descriptor=None):
        self.ring = NativeRing(self.CAPACITY, _HEADER.size, None if descriptor is None else descriptor.ring)
        self.payload = (MappedSegment.create(self.CAPACITY*self.WIDTH, 'worker-payload:2')
                        if descriptor is None else MappedSegment(descriptor.payload))
        if self.payload.descriptor.schema != 'worker-payload:2':
            raise ValueError('worker channel ABI mismatch')
        self.bytes_sent = self.bytes_received = 0

    @property
    def descriptor(self):
        return ChannelDescriptor(self.ring.segment.descriptor, self.payload.descriptor)

    def put_nowait(self, message):
        if not self.ring.can_push():
            raise Full
        kind = message[0]
        seq = message[1] if kind in ('RETIRE_RECORD','RESULT','DRAFT_RESULT','PHYSICAL','IMPORTED','DRAFT_IMPORTED','INPUTS') else 0
        data = message[1] if kind in ('WORK', 'COMPUTE', 'COMPUTED') else message[2] if len(message) == 3 else None
        raw = encode(kind, data)
        if len(raw) > self.WIDTH:
            raise ValueError('message exceeds reserved worker payload')
        offset = self.ring.tail() % self.CAPACITY * self.WIDTH
        self.payload.buffer[offset:offset+len(raw)] = raw
        if not self.ring.push(_HEADER.pack(_KINDS[kind], len(raw), seq)):
            raise Full
        self.bytes_sent += _HEADER.size + len(raw)

    def peek_kind(self):
        header = self.ring.peek()
        return None if header is None else _NAMES[_HEADER.unpack(header)[0]]

    def get_nowait(self):
        if self.ring.head() == self.ring.tail():
            raise Empty
        header = self.ring.peek()
        if header is None:
            raise Empty
        kind, size, seq = _HEADER.unpack(header)
        if size > self.WIDTH:
            raise ValueError('invalid worker payload size')
        offset = self.ring.head() % self.CAPACITY * self.WIDTH
        raw = bytes(self.payload.buffer[offset:offset+size])
        self.ring.ack()
        self.bytes_received += _HEADER.size + size
        name = _NAMES[kind]
        data = decode(name, raw)
        if name in ('WORK', 'COMPUTE', 'COMPUTED'):
            return name, data
        if name == 'RETIRE_RECORD':
            return name, seq
        if name in ('DRAIN','SHUTDOWN'):
            return name, None
        return name, seq, data

    def close(self, *, unlink=False):
        with ExitStack() as stack:
            if unlink:
                stack.callback(self.payload.unlink)
                stack.callback(self.ring.segment.unlink)
            stack.callback(self.payload.close)
            stack.callback(self.ring.close)
