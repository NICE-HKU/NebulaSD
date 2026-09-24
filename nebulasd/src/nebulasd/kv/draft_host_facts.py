"""Draft HostKV versions describe actual, possibly unverified proposal KV."""
from dataclasses import replace
from nebulasd.core.enums import D2HStatus, StateChangeBlockKind as K
from nebulasd.core.ids import KV_VERSION, U64
from nebulasd.table.draft_fences import (
    allocation_values, expect, read, validate_allocation,
)


class DraftHostKVFacts:
    def __init__(self, allocator, table, *, arena_id, layout_id, block_size):
        self.allocator, self.table = allocator, table
        self.arena_id, self.layout_id, self.block_size = arena_id, layout_id, block_size
        self._shared_allocation = (getattr(allocator, "table", None) is table
            and getattr(allocator, "allocation_kind", None) == K.REQUEST_DRAFT_HOSTKV)

    def allocation(self, identity):
        validate_allocation(self.table, identity)
        extent = self.extent_for(identity)
        if not self._shared_allocation and self.allocator.require(identity.request_slot, identity.request_epoch) != extent:
            raise ValueError('Draft HostKV extent mismatch')
        return extent

    def extent_for(self, identity):
        """Capture a validated snapshot allocation protected by worker ownership."""
        a = identity.allocation
        if (a.arena_id, a.arena_generation, a.layout_id, a.block_size) != (
                self.arena_id, self.allocator.arena.descriptor.descriptor_generation,
                self.layout_id, self.block_size):
            raise ValueError("Draft HostKV arena/layout mismatch")
        return self.allocator.arena.make_extent(request_slot=identity.request_slot,
            request_epoch=identity.request_epoch, host_slot=a.host_slot,
            host_slot_generation=a.host_slot_generation, writer_lease_generation=a.writer_lease_generation,
            offset_blocks=a.offset_blocks, capacity_blocks=a.capacity_blocks)

    def source(self, identity, handle):
        return self.source_ready(identity, handle, self.allocation(identity))

    def source_ready(self, identity, handle, extent):
        partition = self.table.partition(K.REQUEST_DRAFT_D2H)
        if partition.read_publish_seq(identity.request_slot) == U64.invalid:
            return None
        row = partition.read_stable(identity.request_slot)
        if row.get("request_epoch") != identity.request_epoch:
            return None
        version = row.get("snapshot_version")
        if KV_VERSION.is_newer(version, identity.snapshot_version):
            raise ValueError("Draft HostKV snapshot was overwritten")
        if version != identity.snapshot_version or row.get("status") != D2HStatus.HOST_READY:
            return None
        expect(row, snapshot_round_id=identity.round_id, ready_version=identity.snapshot_version,
            snapshot_handle=handle, source_worker_id=identity.worker_id,
            source_worker_generation=identity.worker_generation, owner_epoch=identity.owner_epoch,
            source_op_seq=identity.op_seq, valid_blocks=identity.valid_blocks,
            logical_kv_len=identity.logical_kv_len, result_code=0, **allocation_values(identity.allocation))
        return extent

    def write_extent(self, identity, persistence, *, allocation=None):
        """Validate the persisted prefix before replacing its ready fact.

        A partial export is legal only when the shared baseline matches the
        version imported by this session. No Target committed-prefix semantics
        enter this check. The caller acquires the write pin before publishing
        IN_D2H, and retains it through HOST_READY publication.
        """
        extent = allocation if allocation is not None else self.allocation(identity)
        if persistence.snapshot_version != identity.snapshot_version:
            raise ValueError("Draft persistence/snapshot version mismatch")
        dirty = persistence.export_range(logical_kv_len=identity.logical_kv_len, block_size=self.block_size)
        partition = self.table.partition(K.REQUEST_DRAFT_D2H)
        previous = None
        if partition.read_publish_seq(identity.request_slot) != U64.invalid:
            row = read(self.table, K.REQUEST_DRAFT_D2H, identity.request_slot)
            if row.get("request_epoch") == identity.request_epoch:
                previous = row
        if persistence.persisted_version is not None:
            if previous is None:
                raise ValueError("Draft incremental export has no HostKV baseline")
            expect(previous, status=int(D2HStatus.HOST_READY), result_code=0,
                   ready_version=persistence.persisted_version,
                   snapshot_version=persistence.persisted_version,
                   **allocation_values(identity.allocation))
            if not KV_VERSION.is_newer(identity.snapshot_version, persistence.persisted_version):
                raise ValueError("Draft export version must advance")
            if dirty.begin_block > previous.get("valid_blocks"):
                raise ValueError("Draft dirty range leaves an unpersisted prefix gap")
        elif previous is not None:
            # A fresh session cannot silently replace an existing allocation's
            # version history, even if it happens to request a full export.
            raise ValueError("Draft full rebuild requires a fresh HostKV allocation")
        # HostKVExtent calls this field committed_blocks. For the shared DMA
        # primitive it means persisted blocks; Draft acceptance is unrelated.
        return replace(extent, committed_blocks=0 if previous is None else previous.get("valid_blocks"),
                       kv_version=0 if persistence.persisted_version is None else persistence.persisted_version), dirty
