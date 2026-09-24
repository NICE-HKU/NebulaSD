from __future__ import annotations

import pytest

from nebulasd.core.enums import ProposalKind
from nebulasd.core.handles import ArenaHandle
from nebulasd.data.generation_config_arena import DraftGenerationConfig, GenerationConfigArena
from nebulasd.data.proposal_arena import ProposalArena, ProposalPayload
from nebulasd.data.token_arena import TokenArena


def test_token_arena_round_trip_and_bounds() -> None:
    arena = TokenArena(16, generation=3)
    handle = arena.write_tokens((1, 2, 3))

    assert handle == ArenaHandle(0, 12, 3)
    assert arena.read_tokens(handle) == (1, 2, 3)
    with pytest.raises(ValueError, match="stale"):
        arena.read_tokens(ArenaHandle(handle.offset, handle.length, 2))
    with pytest.raises(ValueError, match="capacity"):
        arena.write_tokens((4, 5))
    arena.reset_quiescent()
    with pytest.raises(ValueError, match="stale"):
        arena.read_tokens(handle)
    assert arena.write_tokens((4,)) == ArenaHandle(0, 4, 4)


def test_generation_config_arena_uses_fixed_schema() -> None:
    arena = GenerationConfigArena(64)
    config = DraftGenerationConfig(
        max_new_tokens=8,
        proposal_depth=4,
        stop_token_ids=(11, 12),
        eos_token_id=99,
    )
    handle = arena.write_config(config)

    assert arena.read_config(handle) == config
    assert handle.length == 24
    arena.reset_quiescent()
    with pytest.raises(ValueError, match="stale"):
        arena.read_config(handle)


def test_proposal_arena_round_trip_and_generation_fence() -> None:
    arena = ProposalArena(64, generation=7)
    payload = ProposalPayload(ProposalKind.LINEAR, (101, 102, 103))
    handle = arena.write_proposal(payload)

    assert arena.read_proposal(handle) == payload
    with pytest.raises(ValueError, match="stale"):
        arena.read_proposal(ArenaHandle(handle.offset, handle.length, 6))


def test_proposal_arena_batch_write_and_fifo_release_requires_reset_for_reuse() -> None:
    arena = ProposalArena(32)
    first, second = arena.write_proposals(
        (
            ProposalPayload(ProposalKind.LINEAR, (1,)),
            ProposalPayload(ProposalKind.LINEAR, (2,)),
        )
        )
    with pytest.raises(ValueError, match="FIFO"):
        arena.release(second)

    arena.release(first)
    with pytest.raises(ValueError, match="reset_quiescent"):
        arena.write_proposal(ProposalPayload(ProposalKind.LINEAR, (3,)))
    with pytest.raises(ValueError, match="live allocation"):
        arena.read_proposal(first)
    arena.release(second)
    arena.reset_quiescent()
    with pytest.raises(ValueError, match="stale"):
        arena.read_proposal(second)
    third = arena.write_proposal(ProposalPayload(ProposalKind.LINEAR, (3,)))
    assert third == ArenaHandle(0, 12, 2)
    assert arena.read_proposal(third).draft_token_ids == (3,)


def test_proposal_arena_rejects_stale_aba_handle_after_reset_reuse() -> None:
    arena = ProposalArena(12, generation=1)
    first = arena.write_proposal(ProposalPayload(ProposalKind.LINEAR, (1,)))
    arena.release(first)
    arena.reset_quiescent()
    second = arena.write_proposal(ProposalPayload(ProposalKind.LINEAR, (2,)))

    assert first.offset == second.offset
    assert first.length == second.length
    assert first.generation != second.generation
    with pytest.raises(ValueError, match="stale"):
        arena.read_proposal(first)
    with pytest.raises(ValueError, match="stale"):
        arena.release(first)
    assert arena.read_proposal(second).draft_token_ids == (2,)


def test_proposal_arena_release_many_preflights_before_commit() -> None:
    arena = ProposalArena(64)
    first, second, third = arena.write_proposals(
        (
            ProposalPayload(ProposalKind.LINEAR, (1,)),
            ProposalPayload(ProposalKind.LINEAR, (2,)),
            ProposalPayload(ProposalKind.LINEAR, (3,)),
        )
    )

    with pytest.raises(ValueError, match="FIFO"):
        arena.release_many((first, third))

    assert arena.read_proposal(first).draft_token_ids == (1,)
    assert arena.read_proposal(second).draft_token_ids == (2,)
    assert arena.read_proposal(third).draft_token_ids == (3,)
    arena.release_many((first, second, third))
    with pytest.raises(ValueError, match="live allocation"):
        arena.read_proposal(first)
