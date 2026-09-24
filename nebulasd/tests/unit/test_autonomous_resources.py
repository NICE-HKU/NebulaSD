import pytest
from nebulasd.data.shared_arenas import SharedTokenArena
from nebulasd.workers.resources import PayloadDescriptor, attach_router
from test_autonomous_work import work
from nebulasd.workers.work import Work
from dataclasses import replace


def test_multiple_arena_generations_and_post_barrier_attachment():
    writers = [SharedTokenArena(128, generation=g, writer=True) for g in (17, 23)]
    readers = []
    try:
        handles = [a.write_tokens((g,)) for a, g in zip(writers, (17, 23))]
        router, readers = attach_router(tuple(PayloadDescriptor.of(a) for a in writers), SharedTokenArena)
        assert [router.read_tokens(h) for h in handles] == [(17,), (23,)]
        for a in readers:
            a.close()
        readers = []
        for a in writers:
            a.recycle_quiescent(32)
        new_handles = [a.write_tokens((99,)) for a in writers]
        router, readers = attach_router(tuple(PayloadDescriptor.of(a) for a in writers), SharedTokenArena)
        assert [router.read_tokens(h) for h in new_handles] == [(99,), (99,)]
        with pytest.raises(KeyError):
            router.read_tokens(handles[0])
        with pytest.raises(ValueError, match='ambiguous'):
            attach_router((PayloadDescriptor.of(writers[0]),)*2, SharedTokenArena)
    finally:
        for a in readers:
            a.close()
        for a in writers:
            a.close()
            a.segment.unlink()


def test_work_preserves_allocation_identity_independent_of_request_slot():
    w = work()
    w = replace(w, rows=(replace(w.rows[0], host_slot=83,
                  host_generation=2**40+1, writer_generation=2**41+3),))
    assert Work.from_bytes(w.to_bytes()) == w
