from dataclasses import replace
import pytest
from nebulasd.core.handles import ArenaHandle
from nebulasd.core.enums import StateChangeBlockKind as K, Lifecycle
from nebulasd.workers.work import Work, WorkKind, RowWork, TableDependency, Selector, Outcome
from nebulasd.workers.completion import CompletionArena, WorkCompletion, MemberCompletion, completion_bytes
from nebulasd.workers.dependencies import Dependencies
from nebulasd.table.storage import RequestSchedulingTable
from nebulasd.table.writers import EngineTableWriter
from nebulasd.ipc.protocol import encode_command_payload, decode_command_payload


def work(seq=1, bank=0, epoch=1):
    handle = ArenaHandle(16, 4, 1)
    row = RowWork(0, 1, 0, seq, 0, 8, 0, 8, 0, 1, 0, 1,
                  handle, handle, handle, 4, 16, 4)
    return Work(0, 1, seq, WorkKind.TARGET_PREFILL, bank, epoch, 0, 4096, (row,))


def test_work_wire_future_input_round_trip():
    w = work()
    row = replace(w.rows[0], source=TableDependency(K.REQUEST_D2H, 0, 1, 2, Selector.TARGET_HOST),
        predecessor=TableDependency(K.REQUEST_DRAFT, 0, 1, 2, Selector.PROPOSAL),
        classified=TableDependency(K.REQUEST_ENGINE, 0, 1, 2, Selector.CLASSIFIED))
    w = replace(w, operation=WorkKind.TARGET_VERIFY, rows=(row,))
    raw = encode_command_payload(w)
    assert decode_command_payload(w.kind, raw, worker_id=0, worker_generation=1, command_seq=1) == w
    with pytest.raises(ValueError):
        Work.from_bytes(raw[:-1])
    with pytest.raises(ValueError):
        Work.from_bytes(raw + b'\0')
    with pytest.raises(ValueError):
        decode_command_payload(w.kind, raw, worker_id=1, worker_generation=1, command_seq=1)


def test_classification_prefill_zero_not_ready_and_finished_unblocks_missing_input():
    table = RequestSchedulingTable(1)
    writer = EngineTableWriter(table)
    def publish(seq, ticket, lifecycle):
        writer.publish_active(slot=0, publish_seq=seq, request_epoch=1, current_round_id=0,
            arrival_seq=0, prompt_token_count=4, max_new_tokens=1, spec_token_limit=4,
            classified_result_ticket=ticket, lifecycle=lifecycle)
    publish(0, 0, Lifecycle.ACTIVE)
    watches = Dependencies(table)
    watches.add(('work', 0, 'classified'), TableDependency(K.REQUEST_ENGINE, 0, 1, 1, Selector.CLASSIFIED))
    assert watches.poll() == []
    publish(1, 1, Lifecycle.FINISHED)
    assert watches.poll()[0].snapshot.get('lifecycle') == Lifecycle.FINISHED
    assert not watches.pending
    # A newly attached worker discovers the already-published level too.
    watches.add(('next', 0, 'classified'), TableDependency(K.REQUEST_ENGINE, 0, 1, 7, Selector.CLASSIFIED))
    assert len(watches.poll()) == 1


def test_completion_reservation_and_native_publication():
    arena = CompletionArena(2 * completion_bytes(1))
    peer = CompletionArena(descriptor=arena.segment.descriptor)
    try:
        first, second = arena.reserve(1), arena.reserve(1)
        assert arena.reserve(1) is None
        assert peer.read(first) is None
        record = WorkCompletion(1, 2, 1, 123, 0,
            (MemberCompletion(0, 1, 0, Outcome.SKIPPED_FINISHED),))
        # Independent reserved records may complete out of dispatch order.
        peer.publish(second, record)
        assert arena.read(second) == record
        assert arena.read(first) is None
        with pytest.raises(RuntimeError, match='overwrite'):
            peer.publish(second, record)
    finally:
        peer.close()
        arena.close(unlink=True)


def test_local_channel_complete_frame_and_bounded_credit():
    from queue import Empty, Full
    from nebulasd.workers.channel import LocalChannel
    channel = LocalChannel()
    peer = LocalChannel(channel.descriptor)
    try:
        for i in range(channel.CAPACITY):
            channel.put_nowait(('WORK', work(seq=i+1).to_bytes()))
        with pytest.raises(Full):
            channel.put_nowait(('DRAIN', None))
        for i in range(channel.CAPACITY):
            kind, raw = peer.get_nowait()
            assert kind == 'WORK' and Work.from_bytes(raw).work_seq == i+1
        with pytest.raises(Empty):
            peer.get_nowait()
        channel.put_nowait(('SHUTDOWN', None))
        assert peer.get_nowait() == ('SHUTDOWN', None)
    finally:
        peer.close()
        channel.close(unlink=True)


def test_watchers_decode_only_selected_fields_including_cold_classification():
    from nebulasd.workers.dependencies import _FIELDS
    table = RequestSchedulingTable(1)
    for selector, names in _FIELDS.items():
        from nebulasd.workers.dependencies import _RULES
        kind = _RULES[selector][0]
        # Publish a row before exercising the seqlock reader and projection.
        if table.partition(kind).read_publish_seq(0) == (1<<64)-1:
            table.partition(kind)._publish(0,0,())
        snapshot = table.partition(kind).read_stable(0,field_names=names)
        assert tuple(f.name for f in snapshot.fields) == names
    writer = EngineTableWriter(table)
    writer.publish_active(slot=0,publish_seq=1,request_epoch=1,current_round_id=0,
        classified_result_ticket=1,arrival_seq=0,prompt_token_count=4,max_new_tokens=4,
        spec_token_limit=4,lifecycle=Lifecycle.ACTIVE)
    watches = Dependencies(table)
    watches.add((1,0,'classified'),TableDependency(K.REQUEST_ENGINE,0,1,1,Selector.CLASSIFIED))
    captured = watches.poll()[0].snapshot
    assert captured.get('classified_result_ticket') == 1
    assert tuple(f.name for f in captured.fields) == _FIELDS[Selector.CLASSIFIED]


def test_old_work_version_cannot_attach_to_shared_output_protocol():
    raw=bytearray(work().to_bytes())
    raw[4:6]=(3).to_bytes(2,'little')
    with pytest.raises(ValueError,match='WORK'):
        Work.from_bytes(raw)
