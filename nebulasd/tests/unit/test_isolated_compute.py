"""Prepared-job transport and owner completion boundaries, without CUDA."""
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty
from threading import Event
from types import SimpleNamespace as NS

import pytest

from nebulasd.workers.channel import LocalChannel
from nebulasd.workers.isolated import ControlRuntime, InlineJobs
from nebulasd.workers.target.inputs import Plan
from nebulasd.workers.target.backend import Result, publish_clock
from nebulasd.workers.work import WorkKind


def test_prepared_job_drops_protocol_state_and_returns_owned_result():
    wire = LocalChannel()
    peer = LocalChannel(wire.descriptor)
    owner = ControlRuntime.__new__(ControlRuntime)
    owner.commands, owner.execution_wake, owner.pending = wire, Event(), None
    try:
        # Protocol-only state cannot even be pickled; execution must not receive it.
        spec = NS(work_seq=7, operation=WorkKind.TARGET_VERIFY, protocol=lambda: None)
        future = owner.execute(Plan(spec, ()))
        assert owner.execution_wake.is_set() and not future.done()
        kind, job = peer.get_nowait()
        assert kind == 'COMPUTE'
        assert vars(job.spec) == dict(work_seq=7, operation=WorkKind.TARGET_VERIFY)
        with pytest.raises(RuntimeError, match='concurrent'):
            owner.execute(Plan(spec, ()))
        result = Result((), (), None, 10, 20)
        wire.put_nowait(('COMPUTED', result))
        assert peer.get_nowait() == ('COMPUTED', result)
    finally:
        peer.close()
        wire.close(unlink=True)


def test_owner_completes_model_future_before_advancing_runtime():
    owner = ControlRuntime.__new__(ControlRuntime)
    owner.results = LocalChannel()
    owner.pending = Future()
    finished = owner.pending
    calls = []
    owner.trace = NS(mark=lambda *args: None)
    owner.runtime = NS(compute=NS(spec=NS(work_seq=1)), step=lambda: calls.append(finished.result()) or True)
    owner.observation, owner.stopping = None, False
    try:
        result = Result((), (), None, 10, 20)
        owner.results.put_nowait(('COMPUTED', result))
        assert owner.step()
        assert calls == [result] and owner.pending is None
    finally:
        owner.results.close(unlink=True)


def test_inline_preparation_fails_in_owner_without_future():
    def bad():
        raise ValueError('invalid input')
    with pytest.raises(ValueError, match='invalid input'):
        InlineJobs().submit(bad)
    assert InlineJobs().submit(lambda: 42) == 42


def test_shared_compute_clock_is_published_before_wakeup():
    import multiprocessing as mp
    slot = mp.get_context('spawn').Array('Q', 3)
    seen = []
    backend = NS(clock_slot=slot, clock_wakeup=NS(set=lambda: seen.append(tuple(slot[:]))))
    publish_clock(backend, 12, 100, 0)
    publish_clock(backend, 12, 100, 200)
    assert seen == [(12, 100, 0), (12, 100, 200)]
    assert backend.compute_clock == seen[-1]


def test_publication_adapter_preserves_slotted_draft_fields_and_handle():
    from nebulasd.workers.draft.backend import Output
    from nebulasd.workers.draft.inputs import Imported
    from nebulasd.core.handles import ArenaHandle
    row = Output(0, (1, 2), 1, 20, 3, 1, 1, 4)
    handle = ArenaHandle(0, 64, 1)
    imported = Imported(0, 3, 2, 1, 20, 0, handle)
    state = NS(imported=True, import_inputs=NS(rows=(imported,)), h2d_receipt=NS(submitted_ns=11),
               result=NS(rows=(row,), compute_start_ns=12, compute_end_ns=13), physical_done=False)
    owner = ControlRuntime.__new__(ControlRuntime)
    owner.runtime, owner.sent = NS(records={7: state}), {}
    owner.result_kind, owner.import_kind = 'DRAFT_RESULT', 'DRAFT_IMPORTED'
    assert owner.get_nowait()[2]['rows'][0]['snapshot_handle'] == handle
    assert owner.get_nowait()[2]['rows'][0]['proposal'] == (1, 2)
    with pytest.raises(Empty):
        owner.get_nowait()


@pytest.mark.parametrize('draft', [False, True])
def test_control_constructs_cpu_compiler_and_accepts_decoded_work(draft, monkeypatch):
    from contextlib import ExitStack
    from nebulasd.workers import isolated
    from nebulasd.workers.target.inputs import TargetInputs
    from nebulasd.workers.draft.inputs import DraftInputs
    from nebulasd.workers.work import Work
    from test_autonomous_work import work
    from test_autonomous_draft_publication import draft_work

    dma = NS(close=lambda: None, write_metadata=lambda *a: None)
    monkeypatch.setattr(isolated, 'ControlDMA', lambda options: dma)
    monkeypatch.setattr(isolated, 'attach_inputs', lambda options, stack:
                        dict(host=None, tokens=None, configs=None, proposals=None))
    options = dict(isolated_role='draft' if draft else 'target', blocks_per_bank=8, capacity_rows=2,
                   snapshots=(), layout_id=1, host_arena_id=1)
    w = draft_work() if draft else work()
    def decode_again(*args):
        raise AssertionError('same-process WORK was decoded twice')
    monkeypatch.setattr(Work, 'from_bytes', decode_again)
    with ExitStack() as stack:
        owner = ControlRuntime(options, stack, NS(), NS(), Event(), Event())
        assert type(owner.role) is (DraftInputs if draft else TargetInputs)
        assert not hasattr(owner.role, 'model') and not hasattr(owner.role, 'execute')
        assert owner.runtime.execute == owner.execute
        assert owner.runtime.write_metadata == dma.write_metadata
        owner.accept(w)
        assert owner.runtime.records[w.work_seq].spec is w
        with pytest.raises(ValueError, match='duplicate WORK'):
            owner.accept(w)
        with pytest.raises(ValueError, match='generation'):
            owner.receive_inputs(w.work_seq, dict(worker_id=0, worker_generation=999, events=()))
        owner.receive_inputs(w.work_seq, dict(worker_id=w.worker_id,
                            worker_generation=w.worker_generation, events=()))
