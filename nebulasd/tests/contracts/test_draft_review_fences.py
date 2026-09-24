"""Regression for initial dispatch races and terminal copy facts, Python/native."""
from dataclasses import replace
import pytest
from nebulasd.core.enums import StateChangeBlockKind as K, D2HStatus, H2DStatus
from nebulasd.core.ids import U64
from nebulasd.core.handles import ArenaHandle
from nebulasd.ipc.protocol import DraftBatchCommand, NewRequestData, DraftInitialBank, CommandHeader, encode_command_payload
from nebulasd.ipc.dispatch_fenced_consumer import DispatchFencedConsumer
from nebulasd.scheduler.commands import DispatchPlanePoisoned
from nebulasd.table.storage import TableProtocolError
from nebulasd.table.writers import EngineTableWriter, DispatcherTableWriter
from nebulasd.table.draft_fences import read, publish, prepare_values
from nebulasd.table.draft_writers import DraftSourceCopyWriter, DraftDestinationCopyWriter
from support.draft_contract_fixture import snapshot, seed, prepare, run, target_ready, compute
from contracts.test_draft_migration_contract import table, plane, host_ready, gpu_ready


def initial(table, slots=(0,)):
    s = snapshot()
    for slot in slots:
        EngineTableWriter(table).publish_active(slot=slot, publish_seq=0, request_epoch=5,
            current_round_id=1, arrival_seq=slot, prompt_token_count=30, max_new_tokens=64, spec_token_limit=4)
    rows = tuple(NewRequestData(slot, 5, 1, 10, 3, s.prompt_handle, s.committed_output_handle,
                               s.generation_config_handle, ArenaHandle.null()) for slot in slots)
    # Deliberately distinct ring, operation and batch sequences.
    return DraftBatchCommand(1, 9, 0, rows, (), bank=DraftInitialBank(0, 1, 70, 16, (3,) * len(slots)))


@pytest.mark.parametrize('destination', [1, 2])
def test_initial_duplicate_before_compute_is_rejected_atomically(table, destination):
    command = initial(table, (0, 1))
    dispatcher, endpoints = plane(table)
    first = replace(command, new_requests=command.new_requests[:1], bank=replace(command.bank, capacity_blocks=(3,)))
    dispatcher.dispatch(first)
    before = read(table, K.REQUEST_DISPATCH, 0)
    # Valid member first, duplicate member last: no partial publication allowed.
    duplicate = replace(command, worker_id=destination, command_seq=1, new_requests=command.new_requests[::-1])
    with pytest.raises(TableProtocolError, match='already dispatched'):
        dispatcher.dispatch(duplicate)
    assert read(table, K.REQUEST_DISPATCH, 0) == before
    assert table.partition(K.REQUEST_DISPATCH).read_publish_seq(1) == U64.invalid
    assert endpoints[1].ring.is_empty()
    consumer = DispatchFencedConsumer(endpoints[0].ring, table, 1)
    assert consumer.consume(expected_worker_generation=9, arena=endpoints[0].arena).decode(worker_id=1) == first
    assert endpoints[0].ring.is_empty()


@pytest.mark.parametrize('change', [{'request_epoch': 6}, {'request_epoch': 4}])
def test_initial_wrong_request_preflight_preserves_entire_batch(table, change):
    command = initial(table, (0, 1))
    dispatcher, endpoints = plane(table)
    bad = replace(command, new_requests=(command.new_requests[0], replace(command.new_requests[1], **change)))
    with pytest.raises(TableProtocolError):
        dispatcher.dispatch(bad)
    assert endpoints[0].ring.is_empty()
    assert all(table.partition(K.REQUEST_DISPATCH).read_publish_seq(i) == U64.invalid for i in (0, 1))


def test_initial_waits_for_post_ring_facts_and_old_slot_epoch(table):
    command = initial(table)
    dispatcher, endpoints = plane(table)
    endpoint = endpoints[0]
    raw = encode_command_payload(command)
    handle = endpoint.arena.allocate(command_seq=command.command_seq, payload=raw)
    endpoint.ring.publish(CommandHeader(command.command_seq, 9, command.kind, handle.offset, handle.length))
    consumer = DispatchFencedConsumer(endpoint.ring, table, 1)
    assert consumer.consume(expected_worker_generation=9, arena=endpoint.arena) is None
    publish(table, K.REQUEST_DISPATCH, 0, dict(request_epoch=4, draft_issue_seq=999))
    assert consumer.consume(expected_worker_generation=9, arena=endpoint.arena) is None
    # Target-only fact for current lifetime, before the initial Draft fact.
    publish(table, K.REQUEST_DISPATCH, 0, dict(request_epoch=5, draft_issue_seq=0, target_run_seq=7))
    assert consumer.consume(expected_worker_generation=9, arena=endpoint.arena) is None
    dispatcher._preflight_dispatch_facts(command)
    dispatcher._publish_dispatch_facts(command)
    assert consumer.consume(expected_worker_generation=9, arena=endpoint.arena).decode(worker_id=1) == command


def test_initial_draft_round_is_not_the_engine_last_target_round(table):
    command = initial(table)
    # Engine tracks the accepted Target round; Draft computes the next round.
    publish(table, K.REQUEST_ENGINE, 0, dict(current_round_id=0))
    dispatcher, endpoints = plane(table)
    dispatcher.dispatch(command)
    consumer = DispatchFencedConsumer(endpoints[0].ring, table, 1)
    assert consumer.consume(expected_worker_generation=9, arena=endpoints[0].arena).decode(worker_id=1) == command


@pytest.mark.parametrize('field,value', [('request_epoch', 6), ('draft_round_id', 2),
    ('draft_issue_seq', 11), ('draft_issue_seq', 9), ('draft_worker_id', 2), ('draft_worker_generation', 10),
    ('draft_owner_epoch', 1), ('draft_run_bank_id', 1), ('draft_run_bank_epoch', 2),
    ('draft_run_batch_seq', 71), ('draft_source_snapshot_version', 1)])
def test_initial_consumer_rejects_replaced_identity(table, field, value):
    command = initial(table)
    dispatcher, endpoints = plane(table)
    dispatcher.dispatch(command)
    publish(table, K.REQUEST_DISPATCH, 0, {field: value})
    consumer = DispatchFencedConsumer(endpoints[0].ring, table, 1)
    with pytest.raises(TableProtocolError):
        consumer.consume(expected_worker_generation=9, arena=endpoints[0].arena)
    assert consumer.pending is not None  # fail-stop, never silently execute/drop it


def test_initial_post_ring_publication_failure_poisoned(table, monkeypatch):
    command = initial(table)
    dispatcher, endpoints = plane(table)
    def fail(command):
        raise RuntimeError('post-ring injected failure')
    monkeypatch.setattr(dispatcher, '_publish_dispatch_facts', fail)
    with pytest.raises(DispatchPlanePoisoned):
        dispatcher.dispatch(command)
    assert not endpoints[0].ring.is_empty()
    with pytest.raises(DispatchPlanePoisoned):
        dispatcher.dispatch(replace(command, command_seq=1))


def test_initial_generation_and_zero_bank_epoch_rejected_before_ring(table):
    command = initial(table)
    dispatcher, endpoints = plane(table)
    for bad in (replace(command, worker_generation=10), replace(command, bank=replace(command.bank, bank_epoch=0))):
        with pytest.raises((ValueError, TableProtocolError)):
            dispatcher.dispatch(bad)
        assert endpoints[0].ring.is_empty()
        assert table.partition(K.REQUEST_DISPATCH).read_publish_seq(0) == U64.invalid


def test_slot_reuse_target_first_does_not_relabel_old_draft_dispatch(table):
    command = initial(table)
    dispatcher, endpoints = plane(table)
    dispatcher.dispatch(command)
    EngineTableWriter(table).publish_active(slot=0, publish_seq=1, request_epoch=6,
        current_round_id=1, arrival_seq=2, prompt_token_count=30, max_new_tokens=64, spec_token_limit=4)
    writer = DispatcherTableWriter(table)
    writer.publish_run_command_sent(slot=0, publish_seq=writer.next_publish_seq(0), request_epoch=6,
        target_run_seq=1, target_round_id=0, planned_target_generation=9, planned_bank_epoch=1,
        planned_target_id=3, planned_bank_id=0)
    fact = read(table, K.REQUEST_DISPATCH, 0)
    assert fact.get('draft_issue_seq') == fact.get('draft_run_bank_epoch') == 0
    fresh = replace(command, command_seq=1, new_requests=(replace(command.new_requests[0], request_epoch=6),))
    dispatcher.dispatch(fresh)
    assert read(table, K.REQUEST_DISPATCH, 0).get('draft_issue_seq') == 10
    assert read(table, K.REQUEST_DISPATCH, 0).get('target_run_seq') == 1
    # Reuse requires a real retirement barrier (Step4); stale old delivery must
    # still fail rather than run under the new slot identity.
    with pytest.raises(TableProtocolError):
        DispatchFencedConsumer(endpoints[0].ring, table, 1).consume(expected_worker_generation=9, arena=endpoints[0].arena)


@pytest.mark.parametrize('status', [D2HStatus.IN_D2H, D2HStatus.HOST_READY])
def test_terminal_d2h_rejects_replay_without_invalidating_run(table, status):
    s, h = seed(table)
    dispatcher, _ = plane(table)
    p = prepare(s, h)
    dispatcher.dispatch(p)
    host_ready(table, s, h)
    gpu_ready(table, p)
    target_ready(table, s)
    before = read(table, K.REQUEST_DRAFT_D2H, 0)
    with pytest.raises(TableProtocolError, match='duplicate or regressing'):
        DraftSourceCopyWriter(table).publish_d2h(identity=s.identity, snapshot_handle=h,
            bank_id=0, bank_epoch=1, batch_seq=10, status=status, copy_bytes=999)
    assert read(table, K.REQUEST_DRAFT_D2H, 0) == before
    dispatcher.dispatch(run(p))


@pytest.mark.parametrize('status', [H2DStatus.WAIT_HOST, H2DStatus.IN_H2D, H2DStatus.GPU_READY])
def test_terminal_h2d_rejects_replay_and_payload_change(table, status):
    s, h = seed(table)
    dispatcher, _ = plane(table)
    p = prepare(s, h)
    dispatcher.dispatch(p)
    host_ready(table, s, h)
    gpu_ready(table, p)
    before = read(table, K.REQUEST_DRAFT_H2D, 0)
    with pytest.raises(TableProtocolError):
        DraftDestinationCopyWriter(table).publish_h2d(command=p, request=p.requests[0], status=status, local_row=99)
    assert read(table, K.REQUEST_DRAFT_H2D, 0) == before


def test_copy_new_versions_and_reprepare_wrap_are_not_blocked_by_completion(table):
    s, h = seed(table)
    # Jump to a wrap boundary via explicit fixture facts, not a scheduler policy.
    s = replace(s, identity=replace(s.identity, snapshot_version=U64.max_valid))
    publish(table, K.REQUEST_DISPATCH, 0, dict(draft_source_snapshot_version=U64.max_valid - 1))
    compute(table, s, h, 0, 1, 10)
    host_ready(table, s, h)
    for version in (0, 1):
        s = replace(s, identity=replace(s.identity, snapshot_version=version))
        publish(table, K.REQUEST_DISPATCH, 0, dict(draft_source_snapshot_version=U64.max_valid if version == 0 else 0))
        compute(table, s, h, 0, 1, 10)
        host_ready(table, s, h)
        assert read(table, K.REQUEST_DRAFT_D2H, 0).get('ready_version') == version
    writer = DraftDestinationCopyWriter(table)
    for seq in (U64.max_valid, 0, 1):
        p = prepare(s, h, prepare_seq=1)
        p = replace(p, requests=(replace(p.requests[0], prepare_seq=seq),))
        publish(table, K.REQUEST_DISPATCH, 0, prepare_values(p, p.requests[0]))
        for state in (H2DStatus.WAIT_HOST, H2DStatus.IN_H2D, H2DStatus.GPU_READY):
            writer.publish_h2d(command=p, request=p.requests[0], status=state, local_row=2)
        assert read(table, K.REQUEST_DRAFT_H2D, 0).get('observed_prepare_seq') == seq


def test_new_request_epoch_can_restart_copy_versions(table):
    from nebulasd.core.ids import RequestFence
    from nebulasd.table.draft_writers import DraftHostKVAllocatorWriter
    s, h = seed(table)
    host_ready(table, s, h)
    p = prepare(s, h)
    publish(table, K.REQUEST_DISPATCH, 0, prepare_values(p, p.requests[0]))
    gpu_ready(table, p)
    EngineTableWriter(table).publish_active(slot=0, publish_seq=1, request_epoch=6,
        current_round_id=1, arrival_seq=2, prompt_token_count=30, max_new_tokens=64, spec_token_limit=4)
    s = replace(s, identity=replace(s.identity, request_epoch=6))
    DraftHostKVAllocatorWriter(table).publish_allocation(request=RequestFence(0, 6), allocation=s.identity.allocation)
    publish(table, K.REQUEST_DISPATCH, 0, dict(request_epoch=6, draft_source_snapshot_version=0))
    compute(table, s, h, 0, 1, 10)
    host_ready(table, s, h)
    p = prepare(s, h)
    publish(table, K.REQUEST_DISPATCH, 0, prepare_values(p, p.requests[0]))
    writer = DraftDestinationCopyWriter(table)
    writer.publish_h2d(command=p, request=p.requests[0], status=H2DStatus.WAIT_HOST)
    gpu_ready(table, p)
    assert read(table, K.REQUEST_DRAFT_D2H, 0).get('request_epoch') == 6
    assert read(table, K.REQUEST_DRAFT_H2D, 0).get('request_epoch') == 6


def test_inflight_h2d_cannot_change_local_row(table):
    s, h = seed(table)
    host_ready(table, s, h)
    p = prepare(s, h)
    publish(table, K.REQUEST_DISPATCH, 0, prepare_values(p, p.requests[0]))
    writer = DraftDestinationCopyWriter(table)
    writer.publish_h2d(command=p, request=p.requests[0], status=H2DStatus.IN_H2D, local_row=3)
    before = read(table, K.REQUEST_DRAFT_H2D, 0)
    with pytest.raises(TableProtocolError, match='row changed'):
        writer.publish_h2d(command=p, request=p.requests[0], status=H2DStatus.GPU_READY, local_row=4)
    assert read(table, K.REQUEST_DRAFT_H2D, 0) == before
    writer.publish_h2d(command=p, request=p.requests[0], status=H2DStatus.GPU_READY, local_row=3)


def test_duplicate_and_overflowed_notifications_preserve_terminal_facts(table):
    from nebulasd.ipc.state_change_ring import StateChangeRing, StateChangeEntry
    from nebulasd.table.reader import IncrementalTableReader
    from nebulasd.table.native_storage import NativeTablePartition
    if isinstance(table.partition(K.REQUEST_DRAFT), NativeTablePartition):
        from nebulasd.ipc.native_ring import NativeStateChangeRing
        ring = NativeStateChangeRing(2)
    else:
        ring = StateChangeRing(2)
    try:
        s, h = seed(table)
        host_ready(table, s, h)
        p = prepare(s, h)
        publish(table, K.REQUEST_DISPATCH, 0, prepare_values(p, p.requests[0]))
        gpu_ready(table, p)
        reader = IncrementalTableReader(request_table=table, rings=(ring,))
        before = {kind: read(table, kind, 0) for kind in (K.REQUEST_DRAFT_D2H, K.REQUEST_DRAFT_H2D)}
        for fact in before.values():
            hint = StateChangeEntry(fact.block_kind, 0, fact.publish_seq)
            ring.push(hint)
            ring.push(hint)
            reader.poll()
            ring.push(hint)
            assert not reader.poll().views  # duplicate hint never republishes fact
        for _ in range(3):
            ring.push(StateChangeEntry(K.REQUEST_DRAFT_D2H, 0, before[K.REQUEST_DRAFT_D2H].publish_seq))
        assert reader.poll().overflow_recovered
        for kind, fact in before.items():
            assert reader.cached_view(kind, 0) == read(table, kind, 0) == fact
    finally:
        if hasattr(ring, 'close'):
            ring.close()
            ring.segment.unlink()
