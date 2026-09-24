import importlib.util

import pytest
from pathlib import Path


def load_speculative_module():
    module_path = Path(__file__).resolve().parents[1] / "swiftllm" / "speculative.py"
    spec = importlib.util.spec_from_file_location("swiftllm_speculative_for_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


speculative = load_speculative_module()


class DummyRequest:
    def __init__(self, prompt_len, output_token_ids):
        self.request_id = 3
        self.prompt_len = prompt_len
        self.output_token_ids = output_token_ids


def test_dflash_linear_verification_rows():
    req = DummyRequest(prompt_len=100, output_token_ids=[10, 11, 12])
    proposal = speculative.DraftProposal(
        request_id=req.request_id,
        kind="linear",
        draft_token_ids=[21, 22, 23],
    )

    plan = speculative.build_linear_verification_plan(req, proposal)

    assert plan.input_token_ids == [12, 21, 22, 23]
    assert plan.seq_lens == [103, 104, 105, 106]
    assert plan.output_row_count == 4



def test_dflash_block_verification_rows_match_native_layout():
    req = DummyRequest(prompt_len=100, output_token_ids=[10, 11, 12])
    proposal = speculative.DraftProposal(
        request_id=req.request_id,
        kind="dflash_block",
        draft_token_ids=[21, 22, 23],
    )

    plan = speculative.build_verification_plan(req, proposal)

    assert plan.proposal_kind == "dflash_block"
    assert plan.input_token_ids == [12, 21, 22, 23]
    assert plan.seq_lens == [103, 104, 105, 106]
    assert plan.output_row_count == 4


def test_dflash_block_acceptance_matches_native_formula():
    req = DummyRequest(prompt_len=100, output_token_ids=[10, 11, 12])
    proposal = speculative.DraftProposal(
        request_id=req.request_id,
        kind="dflash_block",
        draft_token_ids=[21, 22, 23],
    )
    plan = speculative.build_verification_plan(req, proposal)

    accepted, num_accepted_draft_tokens = speculative.compute_acceptance_for_plan(
        plan,
        posterior_token_ids=[21, 99, 0, 0],
    )

    assert accepted == [21, 99]
    assert num_accepted_draft_tokens == 1

def test_greedy_acceptance_includes_bonus_after_first_reject():
    accepted, num_accepted_draft_tokens = speculative.compute_greedy_acceptance(
        draft_token_ids=[21, 22, 23],
        posterior_token_ids=[21, 99, 0, 0],
    )

    assert accepted == [21, 99]
    assert num_accepted_draft_tokens == 1


def test_greedy_acceptance_truncates_draft_count_with_budget_cases():
    cases = [
        ([21, 22, 23], [21, 22, 23, 99], 1, [21], 1),
        ([21, 22, 23], [21, 99, 0, 0], 1, [21], 1),
        ([21, 22, 23], [99, 0, 0, 0], 1, [99], 0),
        ([21, 22, 23], [21, 22, 23, 99], 0, [], 0),
    ]
    for draft, posterior, remaining, expected_tokens, expected_count in cases:
        accepted, num_accepted_draft_tokens = speculative.compute_greedy_acceptance(
            draft_token_ids=draft,
            posterior_token_ids=posterior,
            remaining_output_len=remaining,
        )
        assert accepted == expected_tokens
        assert num_accepted_draft_tokens == expected_count


def test_greedy_acceptance_truncates_at_first_stop_token():
    accepted, num_accepted_draft_tokens = speculative.compute_greedy_acceptance(
        draft_token_ids=[21, 22, 23],
        posterior_token_ids=[21, 22, 23, 99],
        stop_token_ids={22},
    )

    assert accepted == [21, 22]
    assert num_accepted_draft_tokens == 2


def test_greedy_acceptance_rejects_negative_remaining_output_len():
    with pytest.raises(ValueError, match="remaining_output_len must be non-negative"):
        speculative.compute_greedy_acceptance(
            draft_token_ids=[21, 22, 23],
            posterior_token_ids=[21, 22, 23, 99],
            remaining_output_len=-1,
        )


def test_dflash_block_acceptance_truncates_draft_count_through_delegate():
    req = DummyRequest(prompt_len=100, output_token_ids=[10])
    proposal = speculative.DraftProposal(
        request_id=req.request_id,
        kind="dflash_block",
        draft_token_ids=[21, 22],
    )
    plan = speculative.build_verification_plan(req, proposal)

    accepted, num_accepted_draft_tokens = speculative.compute_acceptance_for_plan(
        plan,
        posterior_token_ids=[21, 22, 99],
        remaining_output_len=1,
    )

    assert accepted == [21]
    assert num_accepted_draft_tokens == 1


def test_unique_allocation_aggregation():
    assert speculative.aggregate_max_lens_by_seq_id(
        [3, 3, 3, 7, 7],
        [101, 102, 103, 55, 56],
    ) == {3: 103, 7: 56}


def test_eagle_tree_proposal_is_reserved_not_implemented():
    req = DummyRequest(prompt_len=100, output_token_ids=[10])
    proposal = speculative.DraftProposal(
        request_id=req.request_id,
        kind="tree",
        draft_token_ids=[21, 22],
        retrieve_indices=object(),
        tree_mask=object(),
        tree_position_ids=object(),
    )

    try:
        speculative.build_verification_plan(req, proposal)
    except NotImplementedError as exc:
        assert "EAGLE/tree verification is not implemented yet" in str(exc)
    else:
        raise AssertionError("tree proposal should raise NotImplementedError")


def test_greedy_acceptance_covers_zero_one_and_all_patterns():
    cases = [
        ([21, 22, 23], [99, 0, 0, 0], [99], 0),
        ([21, 22, 23], [21, 99, 0, 0], [21, 99], 1),
        ([21, 22, 23], [21, 22, 23, 99], [21, 22, 23, 99], 3),
    ]
    for draft_token_ids, posterior_token_ids, expected_accepted, expected_count in cases:
        accepted, num_accepted_draft_tokens = speculative.compute_greedy_acceptance(
            draft_token_ids=draft_token_ids,
            posterior_token_ids=posterior_token_ids,
        )
        assert accepted == expected_accepted
        assert num_accepted_draft_tokens == expected_count
