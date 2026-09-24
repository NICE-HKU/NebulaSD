"""Draft migration CPU wire/observation contracts, not KV/DMA validation."""
from dataclasses import replace
import os
import pytest

from nebulasd.core.draft_contracts import DraftSnapshot, DraftSnapshotIdentity
from nebulasd.core.enums import StateChangeBlockKind as K, D2HStatus, H2DStatus
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.ids import RequestFence
from nebulasd.ipc.protocol import (encode_command_payload, decode_command_payload, CommandHeader,
    WorkerCommandCache)
from nebulasd.ipc.draft_protocol import validate_prepared_run
from nebulasd.ipc.command_arena import CommandArena
from nebulasd.ipc.command_ring import CommandRing
from nebulasd.ipc.dispatch_fenced_consumer import DispatchFencedConsumer
from nebulasd.ipc.state_change_ring import StateChangeRing
from nebulasd.scheduler.commands import DispatchPlane, WorkerCommandEndpoint, DispatchPlanePoisoned
from nebulasd.table.storage import RequestSchedulingTable, TableProtocolError
from nebulasd.table.draft_fences import read, publish
from nebulasd.table.draft_writers import (DraftHostKVAllocatorWriter, DraftSourceCopyWriter,
    DraftDestinationCopyWriter, DraftMigrationComputeWriter)
from nebulasd.table.reader import IncrementalTableReader
from support.draft_contract_fixture import snapshot, seed, prepare, run, compute, target_ready


@pytest.fixture(params=['python', 'native'])
def table(request):
    if request.param == 'python':
        yield RequestSchedulingTable(4)
    else:
        if not os.environ.get('STARSD_NEXT_NATIVE_LIBRARY'):
            pytest.skip('explicit native library build required')
        from nebulasd.table.native_storage import request_table, close_table_partitions
        t = request_table(4)
        try:
            yield t
        finally:
            close_table_partitions(t._partitions, unlink=True)


def plane(table):
    endpoints = tuple(WorkerCommandEndpoint(i, 9, CommandRing(8), CommandArena(16384)) for i in (1, 2))
    return DispatchPlane(endpoints=endpoints, request_table=table), endpoints


def host_ready(table, s, h, bank=0, epoch=1, batch=10):
    DraftSourceCopyWriter(table).publish_d2h(identity=s.identity, snapshot_handle=h, bank_id=bank,
        bank_epoch=epoch, batch_seq=batch, status=D2HStatus.HOST_READY)


def gpu_ready(table, p):
    writer = DraftDestinationCopyWriter(table)
    for row in p.requests:
        writer.publish_h2d(command=p, request=row, status=H2DStatus.GPU_READY, local_row=3)


def test_fixed_snapshot_and_commands_round_trip():
    s = snapshot()
    assert DraftSnapshot.from_bytes(s.to_bytes()) == s
    assert len(s.to_bytes()) == DraftSnapshot.byte_size
    p = prepare(s, ArenaHandle(0, DraftSnapshot.byte_size, 10))
    for c in (p, run(p)):
        raw = encode_command_payload(c)
        assert decode_command_payload(c.kind, raw, worker_id=c.worker_id,
            worker_generation=9, command_seq=c.command_seq) == c
        for bad in (raw[:-1], raw + b'\0', raw[:4] + b'\0' * 4 + raw[8:]):
            with pytest.raises(ValueError):
                decode_command_payload(c.kind, bad, worker_id=c.worker_id, worker_generation=9, command_seq=c.command_seq)


@pytest.mark.parametrize('field,value', [('snapshot_version', 2), ('owner_epoch', 2),
    ('worker_id', 2), ('worker_generation', 10), ('request_epoch', 6), ('round_id', 2), ('op_seq', 11)])
def test_snapshot_identity_rejects_stale_fences(field, value):
    s = snapshot()
    with pytest.raises(ValueError, match='identity'):
        s.identity.validate_expected(replace(s.identity, **{field:value}))


@pytest.mark.parametrize('field', ['arena_id', 'arena_generation', 'layout_id', 'host_slot_generation',
    'writer_lease_generation', 'offset_blocks', 'host_slot', 'capacity_blocks', 'block_size'])
def test_all_allocation_fields_are_fenced(table, field):
    s, h = seed(table)
    allocation = replace(s.identity.allocation, **{field:getattr(s.identity.allocation, field) + 1})
    DraftHostKVAllocatorWriter(table).publish_allocation(request=RequestFence(0, 5), allocation=allocation)
    with pytest.raises(TableProtocolError):
        host_ready(table, s, h)
    d, _ = plane(table)
    with pytest.raises(TableProtocolError):
        d.dispatch(prepare(s, h))


@pytest.mark.parametrize('begin,count', [(0, 1), (1, 1), (0, 0), (3, 1)])
def test_compute_rejects_incomplete_or_out_of_bounds_dirty_tail_before_publish(table, begin, count):
    s, h = seed(table)
    before = read(table, K.REQUEST_DRAFT, 0)
    with pytest.raises(ValueError, match='actual KV tail'):
        DraftMigrationComputeWriter(table).publish_ready(snapshot=s, snapshot_handle=h,
            bank_id=0, bank_epoch=1, batch_seq=10, dirty_begin_block=begin, dirty_block_count=count)
    after = read(table, K.REQUEST_DRAFT, 0)
    assert after == before


def test_compute_allows_metadata_only_dirty_range_at_valid_end(table):
    s, h = seed(table)
    DraftMigrationComputeWriter(table).publish_ready(snapshot=s, snapshot_handle=h,
        bank_id=0, bank_epoch=1, batch_seq=10,
        dirty_begin_block=s.identity.valid_blocks, dirty_block_count=0)
    fact = read(table, K.REQUEST_DRAFT, 0)
    assert fact.get('dirty_begin_block') == s.identity.valid_blocks
    assert fact.get('dirty_block_count') == 0


def test_prepare_before_host_ready_and_run_requires_target_and_h2d(table):
    s, h = seed(table)
    d, endpoints = plane(table)
    p = prepare(s, h)
    d.dispatch(p)
    assert read(table, K.REQUEST_DISPATCH, 0).get('draft_worker_id') == 1
    DraftDestinationCopyWriter(table).publish_h2d(command=p, request=p.requests[0], status=H2DStatus.WAIT_HOST)
    # No ready facts have been published; readers cannot manufacture them.
    with pytest.raises((TableProtocolError, RuntimeError)):
        d.dispatch(run(p))
    host_ready(table, s, h)
    gpu_ready(table, p)
    with pytest.raises((TableProtocolError, RuntimeError)):
        d.dispatch(run(p))
    target_ready(table, s)
    d.dispatch(run(p))
    dispatch = read(table, K.REQUEST_DISPATCH, 0)
    assert (dispatch.get('draft_worker_id'), dispatch.get('draft_owner_epoch')) == (2, 1)
    with pytest.raises(TableProtocolError):
        gpu_ready(table, p)  # prepare is no longer authorized after owner commit


def test_a_b_a_and_cross_worker_publish_sequences(table):
    s, h = seed(table)
    original = s
    d, _ = plane(table)
    previous = read(table, K.REQUEST_DRAFT, 0).publish_seq
    bank, epoch, batch = 0, 1, 10
    for attempt, dest in enumerate((2, 1), 1):
        p = prepare(s, h, worker=dest, prepare_seq=attempt)
        d.dispatch(p)
        host_ready(table, s, h, bank, epoch, batch)
        gpu_ready(table, p)
        target_ready(table, s)
        r = run(p)
        d.dispatch(r)
        DraftMigrationComputeWriter(table).publish_running(command=r, request=r.requests[0])
        s = replace(s, identity=replace(s.identity, worker_id=dest, owner_epoch=attempt,
            round_id=attempt + 1, op_seq=10 + attempt, snapshot_version=attempt + 1))
        h = ArenaHandle(attempt * DraftSnapshot.byte_size, DraftSnapshot.byte_size, 10 + attempt)
        bank, epoch, batch = p.standby_bank_id, p.next_bank_epoch, p.batch_seq
        compute(table, s, h, bank, epoch, batch)
        current = read(table, K.REQUEST_DRAFT, 0).publish_seq
        assert current == previous + 2
        previous = current
    assert s.identity.worker_id == original.identity.worker_id
    assert s.identity.owner_epoch == 2
    with pytest.raises(TableProtocolError):
        compute(table, original, ArenaHandle(0, DraftSnapshot.byte_size, 10), 0, 1, 10)


@pytest.mark.parametrize('change', [dict(owner_epoch=0), dict(snapshot_version=2), dict(request_epoch=6),
    dict(round_id=3), dict(run_seq=10)])
def test_run_wrong_request_fences_rejected_before_ring(table, change):
    s, h = seed(table)
    p = prepare(s, h)
    d, endpoints = plane(table)
    d.dispatch(p)
    r = run(p)
    with pytest.raises((ValueError, TableProtocolError)):
        d.dispatch(replace(r, requests=(replace(r.requests[0], **change),)))
    assert read(table, K.REQUEST_DISPATCH, 0).get('draft_owner_epoch') == 0


def test_worker_cache_exact_batch_fences_and_duplicate_consumption():
    s = snapshot()
    p = prepare(s, ArenaHandle(0, DraftSnapshot.byte_size, 10))
    r = run(p)
    cache = WorkerCommandCache(worker_id=2, worker_generation=9)
    cache.apply(p)
    for bad in (replace(r, expected_batch_seq=999), replace(r, active_bank_epoch=2),
                replace(r, requests=(replace(r.requests[0], snapshot_version=2),))):
        with pytest.raises(ValueError): cache.apply(bad)
    cache.apply(r)
    with pytest.raises(ValueError): cache.apply(replace(r, command_seq=2))
    with pytest.raises(ValueError): validate_prepared_run(p, replace(r, worker_generation=10))


def test_batch_membership_order_capacity_and_pending_overwrite():
    s = snapshot()
    p = prepare(s, ArenaHandle(0, DraftSnapshot.byte_size, 10))
    second = replace(p.requests[0], source=replace(s.identity, request_slot=1), destination_offset_blocks=3)
    p = replace(p, requests=(p.requests[0], second))
    r = run(p)
    for rows in (r.requests[:1], r.requests[::-1]):
        with pytest.raises(ValueError): validate_prepared_run(p, replace(r, requests=rows))
    with pytest.raises(ValueError): replace(p, requests=(p.requests[0], p.requests[0]))
    with pytest.raises(ValueError): replace(p.requests[0], destination_capacity_blocks=2)
    cache = WorkerCommandCache(worker_id=2, worker_generation=9)
    cache.apply(p)
    with pytest.raises(ValueError): cache.apply(replace(p, command_seq=1))


def test_dispatch_fenced_consumer_waits_for_post_ring_facts(table):
    s, h = seed(table)
    p = prepare(s, h)
    d, endpoints = plane(table)
    e = endpoints[1]
    raw = encode_command_payload(p)
    handle = e.arena.allocate(command_seq=0, payload=raw)
    e.ring.publish(CommandHeader(0, 9, p.kind, handle.offset, handle.length))
    consumer = DispatchFencedConsumer(e.ring, table, 2)
    assert consumer.consume(expected_worker_generation=9, arena=e.arena) is None
    d._draft_dispatch.preflight(p)
    d._draft_dispatch.sent(p)
    assert consumer.consume(expected_worker_generation=9, arena=e.arena).decode(worker_id=2) == p


def test_post_ring_failure_is_fail_stop(table, monkeypatch):
    s, h = seed(table)
    d, _ = plane(table)
    def fail(command): raise RuntimeError('injected post-ring failure')
    monkeypatch.setattr(d._draft_dispatch, 'sent', fail)
    with pytest.raises(DispatchPlanePoisoned): d.dispatch(prepare(s, h))
    with pytest.raises(DispatchPlanePoisoned): d.dispatch(prepare(s, h))


def test_new_partitions_recovered_by_ring_overflow():
    ring = StateChangeRing(1)
    t = RequestSchedulingTable(1, ring=ring)
    s, h = seed(t)
    host_ready(t, s, h)
    reader = IncrementalTableReader(request_table=t, rings=(ring,))
    batch = reader.poll()
    assert batch.overflow_recovered
    assert reader.cached_view(K.REQUEST_DRAFT_D2H, 0).get('ready_version') == 1
    assert reader.cached_view(K.REQUEST_DRAFT_HOSTKV, 0).get('layout_id') == 987


def test_initial_bank_contract_roundtrip_and_dispatch_fences(table):
    from nebulasd.ipc.protocol import DraftBatchCommand, NewRequestData, DraftInitialBank
    from nebulasd.table.writers import EngineTableWriter
    EngineTableWriter(table).publish_active(slot=0, publish_seq=0, request_epoch=5,
        current_round_id=1, arrival_seq=1, prompt_token_count=30, max_new_tokens=64, spec_token_limit=4)
    s = snapshot()
    row = NewRequestData(0, 5, 1, 10, 3, s.prompt_handle, s.committed_output_handle,
                         s.generation_config_handle, ArenaHandle.null())
    bank = DraftInitialBank(0, 1, 10, 16, (3,))
    c = DraftBatchCommand(1, 9, 0, (row,), (), bank=bank)
    raw = encode_command_payload(c)
    assert decode_command_payload(c.kind, raw, worker_id=1, worker_generation=9, command_seq=0) == c
    with pytest.raises(ValueError): replace(c, bank=replace(bank, capacity_blocks=(2,)))
    with pytest.raises(ValueError): replace(c, bank=replace(bank, capacity_blocks=(3, 3)))
    d, _ = plane(table)
    d.dispatch(c)
    dispatch = read(table, K.REQUEST_DISPATCH, 0)
    assert dispatch.get('draft_run_bank_epoch') == 1
    assert dispatch.get('draft_run_batch_seq') == 10
    assert dispatch.get('draft_owner_epoch') == dispatch.get('draft_source_snapshot_version') == 0
    DraftHostKVAllocatorWriter(table).publish_allocation(request=RequestFence(0, 5), allocation=s.identity.allocation)
    compute(table, s, ArenaHandle(0, DraftSnapshot.byte_size, 10), 0, 1, 10)
    with pytest.raises(TableProtocolError): d.dispatch(replace(c, command_seq=1))


def test_slot_reuse_and_stale_metadata_handle_rejected(table):
    from nebulasd.table.writers import EngineTableWriter
    s, h = seed(table)
    d, _ = plane(table)
    with pytest.raises(TableProtocolError): d.dispatch(prepare(s, replace(h, generation=h.generation + 1)))
    EngineTableWriter(table).publish_active(slot=0, publish_seq=1, request_epoch=6,
        current_round_id=1, arrival_seq=2, prompt_token_count=30, max_new_tokens=64, spec_token_limit=4)
    with pytest.raises(TableProtocolError): host_ready(table, s, h)
    with pytest.raises(TableProtocolError): d.dispatch(prepare(s, h))


def test_allocation_changed_after_gpu_ready_cannot_authorize_run(table):
    s, h = seed(table)
    d, _ = plane(table)
    p = prepare(s, h)
    d.dispatch(p)
    host_ready(table, s, h)
    gpu_ready(table, p)
    target_ready(table, s)
    DraftHostKVAllocatorWriter(table).publish_allocation(request=RequestFence(0, 5),
        allocation=replace(s.identity.allocation, host_slot_generation=3))
    with pytest.raises(TableProtocolError): d.dispatch(run(p))


def test_run_must_use_matching_target_delta(table):
    s, h = seed(table)
    d, _ = plane(table)
    p = prepare(s, h)
    d.dispatch(p)
    host_ready(table, s, h)
    gpu_ready(table, p)
    target_ready(table, s)
    r = run(p)
    with pytest.raises(TableProtocolError):
        d.dispatch(replace(r, requests=(replace(r.requests[0], token_delta_handle=ArenaHandle(999, 8, 1)),)))


def test_draft_registry_has_independent_two_banks_and_copy_runtime():
    from nebulasd.core.enums import WorkerRole, WorkerStatus, BankRole, BankState, CopyStatus
    from nebulasd.table.storage import WorkerSchedulingRegistry
    from nebulasd.table.writers import WorkerRegistryWriter, DraftRegistryWriter
    ring = StateChangeRing(16)
    registry = WorkerSchedulingRegistry(2, ring=ring)
    WorkerRegistryWriter(registry).publish_common(worker_row=1, publish_seq=0,
        worker_id=1, worker_generation=9, role=WorkerRole.DRAFT, status=WorkerStatus.ONLINE,
        command_consumer_seq=0, max_batch_size=8, max_batch_tokens=128)
    writer = DraftRegistryWriter(registry)
    for bank in (0, 1):
        writer.publish_bank(worker_row=1, worker_generation=9, bank_id=bank, bank_epoch=1,
            batch_seq=1, role=BankRole.ACTIVE if bank == 0 else BankRole.STANDBY,
            state=BankState.EMPTY, capacity_blocks=64, alloc_ptr_blocks=0, capacity_rows=8, alloc_rows=0)
    writer.publish_copy_runtime(worker_row=1, worker_generation=9, copy_op_seq=0, copy_status=CopyStatus.H2D, copy_bytes=1024)
    with pytest.raises(TableProtocolError):
        writer.publish_copy_runtime(worker_row=1, worker_generation=8, copy_op_seq=1, copy_status=CopyStatus.IDLE)
    reader = IncrementalTableReader(worker_registry=registry, rings=(ring,))
    assert len(reader.poll().views) == 4
    assert reader.cached_view(K.WORKER_DRAFT_BANK, 3).get('bank_id') == 1
    assert reader.cached_view(K.WORKER_DRAFT_COPY_RUNTIME, 1).get('copy_bytes') == 1024
    from nebulasd.core.ids import U64
    assert registry.partition(K.WORKER_BANK).read_publish_seq(3) == U64.invalid
