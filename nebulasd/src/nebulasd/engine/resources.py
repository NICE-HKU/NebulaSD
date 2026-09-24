"""Create/attach all native control resources before spawning workers."""

from dataclasses import dataclass

from nebulasd.core.enums import WorkerRole
from nebulasd.data.draft_snapshot_arena import SharedDraftSnapshotArena
from nebulasd.data.shared_arenas import SharedTokenArena, SharedConfigArena, SharedProposalArena, ArenaRouter
from nebulasd.ipc.mapped_segment import MappedSegment
from nebulasd.ipc.native_ring import NativeStateChangeRing
from nebulasd.kv.arena import SharedHostKVArena
from nebulasd.table.native_storage import request_table, worker_table, table_descriptors, close_table_partitions


@dataclass(frozen=True)
class ResourceDescriptor:
    specs: tuple
    slots: int
    ring_capacity: int
    payload_capacity: int
    request_tables: dict
    worker_tables: dict
    events: tuple
    tokens: tuple
    proposals: dict
    configs: object
    host: object
    pins: object
    draft_host: object
    draft_pins: object
    snapshots: dict
    draft_layout_id: int
    state_ring_capacity: int = 0


class ControlResources:
    def __init__(self, specs, *, slots=64, ring_capacity=64, payload_capacity=4 << 20,
                 host_blocks=4096, block_bytes=16, dtype="uint8", kv_block_shape=(),
                 descriptor=None, owner=None, doorbell=None, state_ring_capacity=None,
                 draft_layout=None, draft_layout_id=1,
                 draft_host_numa_policy="default", draft_host_numa_nodes=(),
                 target_host_numa_policy="default", target_host_numa_nodes=()):
        state_ring_capacity = state_ring_capacity or ring_capacity
        self.specs, self.slots = tuple(specs), slots
        self._created = descriptor is None
        self._closed = False
        self._arenas, self._segments, self._rings = [], [], []
        self.table = self.registry = self.host = self.draft_host = self.completions = None
        self.draft_layout_id = draft_layout_id if descriptor is None else descriptor.draft_layout_id
        if not specs or {w.role for w in specs} != {WorkerRole.DRAFT, WorkerRole.TARGET}:
            raise ValueError("at least one Draft and Target are required")
        if len({w.block_size for w in specs}) != 1:
            raise ValueError("all workers must share the HostKV block size")
        count = max(w.worker_id for w in specs) + 1
        if len({w.worker_id for w in specs}) != len(specs):
            raise ValueError("duplicate worker id")
        def desc(field):
            return None if descriptor is None else getattr(descriptor, field)
        try:
            if descriptor is None:
                from nebulasd.workers.completion import CompletionArena
                self.completions = CompletionArena(payload_capacity)
            for i in range(count + 1):
                event = NativeStateChangeRing(state_ring_capacity,
                descriptor=None if descriptor is None else descriptor.events[i],
                doorbell=doorbell if (owner is None and i == count) or owner == i else None)
                self._rings.append(event)
            self.events = tuple(self._rings)
            ring = self.events[count if owner is None else owner]
            self.table = request_table(slots, descriptors=desc("request_tables"), ring=ring)
            self.registry = worker_table(count, descriptors=desc("worker_tables"), ring=ring)
            arena = SharedTokenArena(payload_capacity, generation=1,
                descriptor=None if descriptor is None else descriptor.tokens[0], writer=owner is None)
            self.tokens = [arena]
            self._arenas.append(arena)
            self.token_router = ArenaRouter(self.tokens, arena if owner is None else None)
            self.proposals, self.snapshots = {}, {}
            for w in specs:
                if w.role == WorkerRole.DRAFT:
                    arena = SharedProposalArena(payload_capacity, generation=w.worker_id + 1,
                        descriptor=None if descriptor is None else descriptor.proposals[w.worker_id], writer=owner == w.worker_id)
                    self.proposals[w.worker_id] = arena
                    self._arenas.append(arena)
                    snapshot = SharedDraftSnapshotArena(payload_capacity, generation=w.worker_id + 1,
                        descriptor=None if descriptor is None else descriptor.snapshots[w.worker_id], writer=owner == w.worker_id)
                    self.snapshots[w.worker_id] = snapshot
                    self._arenas.append(snapshot)
            self.snapshot_arenas = {a.generation: a for a in self.snapshots.values()}
            self.proposal_router = ArenaRouter(self.proposals.values(), self.proposals.get(owner))
            self.configs = SharedConfigArena(payload_capacity, descriptor=desc("configs"), writer=owner is None)
            self._arenas.append(self.configs)
            self.host = (SharedHostKVArena.create(total_blocks=host_blocks, block_bytes=block_bytes,
                                                    dtype=dtype, kv_block_shape=kv_block_shape,
                                                    numa_policy=target_host_numa_policy, numa_nodes=target_host_numa_nodes)
                         if descriptor is None else SharedHostKVArena.attach(descriptor.host))
            self.pins = (MappedSegment.create(slots * 64, f"host-pins:{slots}") if descriptor is None
                         else MappedSegment(descriptor.pins))
            self._segments.append(self.pins)
            from nebulasd.config import HostKVLayout
            layout = draft_layout or HostKVLayout(block_bytes, dtype, kv_block_shape)
            self.draft_host = (SharedHostKVArena.create(total_blocks=host_blocks, block_bytes=layout.block_bytes,
                dtype=layout.dtype, kv_block_shape=layout.block_shape,
                numa_policy=draft_host_numa_policy, numa_nodes=draft_host_numa_nodes) if descriptor is None
                else SharedHostKVArena.attach(descriptor.draft_host))
            self.draft_pins = (MappedSegment.create(slots * 64, f'draft-host-pins:{slots}') if descriptor is None
                else MappedSegment(descriptor.draft_pins))
            self._segments.append(self.draft_pins)
            self.descriptor = ResourceDescriptor(self.specs, slots, ring_capacity, payload_capacity,
                table_descriptors(self.table), table_descriptors(self.registry),
                tuple(r.segment.descriptor for r in self.events),
                tuple(a.segment.descriptor for a in self.tokens),
                {k: a.segment.descriptor for k, a in self.proposals.items()},
                self.configs.segment.descriptor, self.host.descriptor, self.pins.descriptor,
                self.draft_host.descriptor, self.draft_pins.descriptor,
                {k: a.segment.descriptor for k, a in self.snapshots.items()}, self.draft_layout_id,
                state_ring_capacity)
        except BaseException:
            self.close()
            raise

    @classmethod
    def attach(cls, descriptor, owner, doorbell):
        return cls(descriptor.specs, slots=descriptor.slots, ring_capacity=descriptor.ring_capacity,
                   payload_capacity=descriptor.payload_capacity, descriptor=descriptor, owner=owner, doorbell=doorbell,
                   state_ring_capacity=descriptor.state_ring_capacity or descriptor.ring_capacity)

    def recycle_payloads(self):
        """Keep physical mappings and registrations, fence old payload handles."""
        if self.completions is not None:
            self.completions.segment.buffer[:] = bytes(self.completions.segment.descriptor.size)
            self.completions.next_offset = 0
        stride = len(self.tokens) + 1
        for arena in (*self.tokens, *self.proposals.values(), *self.snapshots.values(), self.configs):
            arena.recycle_quiescent(stride)
        self.token_router.arenas = {a.generation: a for a in self.tokens}
        self.proposal_router.arenas = {a.generation: a for a in self.proposals.values()}
        self.snapshot_arenas.clear()
        self.snapshot_arenas.update({a.generation: a for a in self.snapshots.values()})

    def close(self):
        """Caller must join every child before closing creator-owned mappings."""
        if self._closed:
            return
        self._closed = True
        if self.completions is not None:
            self.completions.close(unlink=self._created)
        for arena in reversed(self._arenas):
            arena.close()
            if self._created:
                arena.segment.unlink()
        for table in (self.table, self.registry):
            if table is not None:
                close_table_partitions(table._partitions, unlink=self._created)
        for ring in reversed(self._rings):
            ring.close()
            if self._created:
                ring.segment.unlink()
        for segment in self._segments:
            segment.close()
            if self._created:
                segment.unlink()
        for host in (self.host, self.draft_host):
            if host is not None:
                host.close()
                if self._created:
                    host.unlink()
