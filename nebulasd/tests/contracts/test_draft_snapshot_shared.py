"""Real shared-memory metadata and native fact visibility; no GPU work."""
from dataclasses import replace
import multiprocessing as mp
import os

import pytest

from nebulasd.core.draft_contracts import DraftSnapshot, DraftSnapshotIdentity
from nebulasd.core.enums import ProposalKind, D2HStatus, StateChangeBlockKind as K
from nebulasd.core.handles import ArenaHandle
from nebulasd.data.draft_snapshot_arena import SharedDraftSnapshotArena, validate_snapshot_payloads
from nebulasd.data.shared_arenas import SharedTokenArena, SharedConfigArena, SharedProposalArena
from nebulasd.data.generation_config_arena import DraftGenerationConfig
from nebulasd.data.proposal_arena import ProposalPayload
from nebulasd.ipc.native_ring import NativeStateChangeRing
from nebulasd.table.native_storage import request_table, table_descriptors, close_table_partitions
from nebulasd.table.draft_writers import DraftSourceCopyWriter
from nebulasd.table.reader import IncrementalTableReader
from support.draft_contract_fixture import seed, compute

pytestmark = pytest.mark.skipif(not os.environ.get('STARSD_NEXT_NATIVE_LIBRARY'), reason='explicit native library build required')


def child_read_and_publish(descriptors, arena_descriptors, ring_descriptor, raw_identity, handle_fields):
    ring = NativeStateChangeRing(8, descriptor=ring_descriptor)
    table = request_table(4, descriptors=descriptors, ring=ring)
    classes = (SharedTokenArena, SharedConfigArena, SharedProposalArena, SharedDraftSnapshotArena)
    arenas = []
    try:
        for cls, descriptor, generation in zip(classes, arena_descriptors, (11, 12, 13, 14), strict=True):
            arenas.append(cls(4096, descriptor=descriptor, generation=generation))
        tokens, configs, proposals, metadata = arenas
        identity = DraftSnapshotIdentity.from_bytes(raw_identity)
        handle = ArenaHandle(*handle_fields)
        s = metadata.read_snapshot(handle, expected=identity, expected_layout_id=987)
        def no_history_copy(*args): raise AssertionError('metadata validation copied token history')
        tokens.read_tokens = no_history_copy
        validate_snapshot_payloads(s, token_arenas={11:tokens}, config_arenas={12:configs}, proposal_arenas={13:proposals})
        assert s.identity.logical_kv_len == s.prompt_count + s.committed_output_count + s.proposal_count - 1
        try:
            metadata.read_snapshot(replace(handle, generation=99), expected=identity, expected_layout_id=987)
        except ValueError:
            pass
        else:
            raise AssertionError('stale shared generation accepted')
        DraftSourceCopyWriter(table).publish_d2h(identity=identity, snapshot_handle=handle,
            bank_id=0, bank_epoch=1, batch_seq=10, status=D2HStatus.HOST_READY)
    finally:
        for arena in arenas: arena.close()
        close_table_partitions(table._partitions)
        ring.close()


def test_spawn_resolves_all_handles_without_history_copy_and_notifies_native_reader():
    table = request_table(4)
    ring = NativeStateChangeRing(8)
    arenas = []
    child = None
    try:
        classes = (SharedTokenArena, SharedConfigArena, SharedProposalArena, SharedDraftSnapshotArena)
        for cls, generation in zip(classes, (11, 12, 13, 14), strict=True):
            arenas.append(cls(4096, generation=generation, writer=True))
        tokens, configs, proposals, metadata = arenas
        s, _ = seed(table)
        output_reservation = tokens.reserve_output(64)
        s = replace(s, prompt_handle=tokens.write_tokens(tuple(range(30))),
            committed_output_handle=tokens.append_output(output_reservation, 0, (9,)),
            generation_config_handle=configs.write_config(DraftGenerationConfig(64, 4)),
            proposal_handle=proposals.write_proposal(ProposalPayload(ProposalKind.LINEAR, (1, 2, 3))))
        handle = metadata.write_snapshot(s)
        compute(table, s, handle, 0, 1, 10)
        # Target can finish while this snapshot is in transit. Append to the
        # Engine's reservation; the old snapshot still names exactly old C.
        newer_output = tokens.append_output(output_reservation, 1, (10, 11))
        assert tokens.read_tokens(s.committed_output_handle) == (9,)
        assert tokens.read_tokens(newer_output) == (9, 10, 11)
        assert metadata.read_snapshot(handle, expected=s.identity, expected_layout_id=987) == s
        # Start with an existing sequence; the child must continue this row.
        DraftSourceCopyWriter(table).publish_d2h(identity=s.identity, snapshot_handle=handle,
            bank_id=0, bank_epoch=1, batch_seq=10, status=D2HStatus.IN_D2H)
        child = mp.get_context('spawn').Process(target=child_read_and_publish, args=(
            table_descriptors(table), tuple(a.segment.descriptor for a in arenas), ring.segment.descriptor,
            s.identity.to_bytes(), (handle.offset, handle.length, handle.generation)))
        child.start()
        child.join(15)
        assert child.exitcode == 0
        reader = IncrementalTableReader(request_table=table, rings=(ring,))
        update = reader.poll()
        assert len(update.views) == 1
        fact = update.views[0]
        assert fact.block_kind == K.REQUEST_DRAFT_D2H and fact.publish_seq == 1
        assert fact.get('ready_version') == 1 and fact.get('snapshot_handle') == handle
        assert fact.get('status') == D2HStatus.HOST_READY
        with pytest.raises(ValueError):
            metadata.read_snapshot(handle, expected=replace(s.identity, snapshot_version=2), expected_layout_id=987)
        with pytest.raises(ValueError, match="layout"):
            metadata.read_snapshot(handle, expected=s.identity, expected_layout_id=123)
        # Every attachment advances generation only at the existing retirement barrier.
        metadata.recycle_quiescent(100)
        with pytest.raises(ValueError): metadata.read_snapshot(handle, expected=s.identity, expected_layout_id=987)
    finally:
        if child is not None and child.is_alive():
            child.kill()
            child.join()
        for arena in arenas:
            arena.close()
            arena.segment.unlink()
        close_table_partitions(table._partitions, unlink=True)
        ring.close()
        ring.segment.unlink()
