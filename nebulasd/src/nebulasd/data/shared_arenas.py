"""Single-writer shared payload arenas; handles remain immutable until an all-owner retirement barrier."""

from nebulasd.core.handles import ArenaHandle
from nebulasd.ipc.mapped_segment import MappedSegment
from nebulasd.ipc.native import library
from .token_arena import TokenArena
from .generation_config_arena import GenerationConfigArena
from .proposal_arena import ProposalArena


class _SharedPayload:
    def _attach(self, capacity, generation, descriptor, writer):
        self._native = library()
        schema = f"{type(self).__name__}:{capacity}:{generation}"
        self.segment = (MappedSegment.create(capacity + 64, schema) if descriptor is None
                        else MappedSegment(descriptor))
        if self.segment.descriptor.schema != schema:
            self.segment.close()
            raise ValueError("shared payload schema mismatch")
        self._bytes = self.segment.buffer[64:]
        self._writer = writer

    def _before_write(self):
        if not self._writer:
            raise RuntimeError("read-only payload attachment")

    def _commit(self):
        self._native.sd_store(self.segment.address, self._head)

    def _validate_handle(self, handle):
        super()._validate_handle(handle)
        if handle.offset + handle.length > self._native.sd_load(self.segment.address):
            raise ValueError("payload handle exceeds published extent")

    def recycle_quiescent(self, stride):
        """All attachments advance namespaces; only the owner resets publication.

        Called exclusively by the acknowledged cohort barrier, never as an
        allocator fast path. Stride preserves each writer's namespace.
        """
        from nebulasd.core.ids import ARENA_GENERATION
        generation = self._generation + stride
        ARENA_GENERATION.validate(generation)
        self._generation = generation
        self._head = 0
        if hasattr(self, '_allocations'):
            self._allocations.clear()
            self._live_handles.clear()
            self._tail = 0
        if self._writer:
            self._commit()

    def reset_quiescent(self):
        raise RuntimeError('use the all-owner retirement barrier')

    def close(self):
        self._bytes.release()
        self.segment.close()


class SharedTokenArena(_SharedPayload, TokenArena):
    def __init__(self, capacity_bytes, *, generation=1, descriptor=None, writer=False):
        super().__init__(capacity_bytes, generation=generation)
        self._attach(capacity_bytes, generation, descriptor, writer)

    def write_tokens(self, tokens):
        return self.write_token_batches((tokens,))[0]

    def write_token_batches(self, batches):
        self._before_write()
        result = super().write_token_batches(batches)
        self._commit()
        return result

    def reserve_output(self, max_tokens):
        self._before_write()
        return self.write_tokens((0,) * max_tokens)

    def append_output(self, reservation, count, tokens):
        """Append once into a request's bounded output region; do not copy its prefix."""
        self._before_write()
        self._validate_handle(reservation)
        if count < 0 or (count + len(tokens)) * 4 > reservation.length:
            raise ValueError("output reservation exhausted")
        raw = b"".join(t.to_bytes(4, "little") for t in tokens)
        start = reservation.offset + count * 4
        self._bytes[start:start + len(raw)] = raw
        return ArenaHandle(reservation.offset, (count + len(tokens)) * 4, self.generation)

    def append_reserved(self, reservation, count, tokens):
        """WORK grants the sole writer this range, never the arena allocation head.

        Engine reserves disjoint request ranges. Successive Target owners append
        after the preceding published prefix; no submitted prefix is rewritten.
        """
        self._validate_handle(reservation)
        if count < 0 or (count + len(tokens))*4 > reservation.length:
            raise ValueError('output reservation exhausted')
        start = reservation.offset + 4*count
        raw = b''.join(t.to_bytes(4, 'little') for t in tokens)
        self._bytes[start:start+len(raw)] = raw
        return ArenaHandle(reservation.offset, (count+len(tokens))*4, reservation.generation)


class SharedConfigArena(_SharedPayload, GenerationConfigArena):
    def __init__(self, capacity_bytes, *, generation=1, descriptor=None, writer=False):
        super().__init__(capacity_bytes, generation=generation)
        self._attach(capacity_bytes, generation, descriptor, writer)

    def write_config(self, config):
        self._before_write()
        result = super().write_config(config)
        self._commit()
        return result


class SharedProposalArena(_SharedPayload, ProposalArena):
    def __init__(self, capacity_bytes, *, generation=1, descriptor=None, writer=False):
        super().__init__(capacity_bytes, generation=generation)
        self._attach(capacity_bytes, generation, descriptor, writer)

    def write_proposals(self, payloads):
        self._before_write()
        result = super().write_proposals(payloads)
        self._commit()
        return result

    def _validate_live_allocation(self, handle):
        # read_proposal already validated bounds/generation/published extent.
        # Shared handles stay live through the all-owner cohort barrier.
        pass


class ArenaRouter:
    """Generation namespaces identify each single-writer arena without copying data."""
    def __init__(self, arenas, writer=None):
        self.arenas = {arena.generation: arena for arena in arenas}
        self.writer = writer

    def read_tokens(self, handle):
        return self.arenas[handle.generation].read_tokens(handle)

    def read_config(self, handle):
        return self.arenas[handle.generation].read_config(handle)

    def read_proposal(self, handle):
        return self.arenas[handle.generation].read_proposal(handle)

    def write_token_batches(self, batches):
        return self.writer.write_token_batches(batches)

    def write_proposals(self, payloads):
        return self.writer.write_proposals(payloads)
