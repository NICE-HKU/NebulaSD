import dataclasses
from typing import Any


@dataclasses.dataclass
class DraftProposal:
    """
    Draft tokens supplied by an external draft side, such as StarSD.

    ``kind="linear"`` is a plain speculative proposal where the target verifies
    ``[anchor] + draft_token_ids``. ``kind="dflash_block"`` uses the same row
    layout, but names the DFlash block protocol explicitly: the first row is the
    unstored anchor token and the following rows are the draft block suffix that
    native StarSD writes into ``block_output_ids[:, 1:]``. ``kind="tree"``
    reserves the EAGLE control-plane shape; SwiftLLM does not implement tree
    attention verification yet.
    """

    request_id: int
    kind: str
    draft_token_ids: list[int]

    retrieve_indices: Any | None = None
    tree_mask: Any | None = None
    tree_position_ids: Any | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class VerifyPlanItem:
    request: Any
    input_token_ids: list[int]
    seq_lens: list[int]
    draft_token_ids: list[int]
    proposal_kind: str = "linear"
    output_row_start: int = -1
    output_row_count: int = 0


class EmbeddedDraftProvider:
    """
    Optional embedded draft interface reserved for local experiments.

    The primary integration path is external: StarSD provides draft tokens to
    SwiftLLMTargetWorker.submit_verify(...).
    """

    async def init_request(self, request, **kwargs):
        raise NotImplementedError

    async def propose(self, requests):
        raise NotImplementedError

    async def update_request(self, request, accepted_token_ids, **kwargs):
        raise NotImplementedError

    async def finish_request(self, request):
        raise NotImplementedError


class NoOpEmbeddedDraftProvider(EmbeddedDraftProvider):
    async def init_request(self, request, **kwargs):
        return None

    async def propose(self, requests):
        return {}

    async def update_request(self, request, accepted_token_ids, **kwargs):
        return None

    async def finish_request(self, request):
        return None


def build_linear_verification_plan(request, proposal: DraftProposal) -> VerifyPlanItem:
    if not request.output_token_ids:
        raise ValueError("linear speculative verification requires an unstored anchor token")

    verify_tokens = [request.output_token_ids[-1]] + proposal.draft_token_ids
    base_seq_len = request.prompt_len + len(request.output_token_ids)
    seq_lens = [base_seq_len + i for i in range(len(verify_tokens))]
    return VerifyPlanItem(
        request=request,
        input_token_ids=verify_tokens,
        seq_lens=seq_lens,
        draft_token_ids=list(proposal.draft_token_ids),
        proposal_kind="linear",
        output_row_count=len(verify_tokens),
    )


def build_dflash_block_verification_plan(request, proposal: DraftProposal) -> VerifyPlanItem:
    if not request.output_token_ids:
        raise ValueError("DFlash block verification requires an unstored anchor token")

    # Native StarSD-DFlash verifies a block whose first token is the current
    # anchor and whose suffix is draft_token_ids. SwiftLLM rows carry the same
    # positions as individual decode rows so they can still be continuous-batched.
    verify_tokens = [request.output_token_ids[-1]] + proposal.draft_token_ids
    base_seq_len = request.prompt_len + len(request.output_token_ids)
    seq_lens = [base_seq_len + i for i in range(len(verify_tokens))]
    return VerifyPlanItem(
        request=request,
        input_token_ids=verify_tokens,
        seq_lens=seq_lens,
        draft_token_ids=list(proposal.draft_token_ids),
        proposal_kind="dflash_block",
        output_row_count=len(verify_tokens),
    )


def build_verification_plan(request, proposal: DraftProposal) -> VerifyPlanItem:
    if proposal.kind == "linear":
        return build_linear_verification_plan(request, proposal)
    if proposal.kind == "dflash_block":
        return build_dflash_block_verification_plan(request, proposal)
    if proposal.kind == "tree":
        raise NotImplementedError("EAGLE/tree verification is not implemented yet")
    raise ValueError(f"Unsupported draft proposal kind: {proposal.kind}")


def compute_greedy_acceptance(
    draft_token_ids: list[int],
    posterior_token_ids: list[int],
    remaining_output_len: int | None = None,
    stop_token_ids=(),
) -> tuple[list[int], int]:
    """
    Return (tokens_to_append, num_accepted_draft_tokens) for greedy verification.
    """

    if remaining_output_len is not None and remaining_output_len < 0:
        raise ValueError("remaining_output_len must be non-negative")
    if len(posterior_token_ids) < len(draft_token_ids) + 1:
        raise ValueError("posterior_token_ids must contain K+1 tokens for K draft tokens")

    num_accepted = 0
    while (
        num_accepted < len(draft_token_ids)
        and posterior_token_ids[num_accepted] == draft_token_ids[num_accepted]
    ):
        num_accepted += 1

    accepted = draft_token_ids[:num_accepted] + [posterior_token_ids[num_accepted]]
    if remaining_output_len is not None:
        accepted = accepted[:remaining_output_len]
    stops = frozenset(int(token) for token in stop_token_ids)
    if any(token < 0 for token in stops):
        raise ValueError("stop_token_ids must be non-negative")
    for index, token_id in enumerate(accepted):
        if token_id in stops:
            accepted = accepted[: index + 1]
            break
    committed_count = len(accepted)
    num_accepted = min(num_accepted, committed_count)
    return accepted, num_accepted


def compute_dflash_block_acceptance(
    draft_token_ids: list[int],
    posterior_token_ids: list[int],
    remaining_output_len: int | None = None,
    stop_token_ids=(),
) -> tuple[list[int], int]:
    """
    Return native DFlash block acceptance for a verified block.

    This mirrors StarSD's target-side formula:
    ``acceptance_length = cumprod(block_output_ids[:, 1:] == posterior[:, :-1]).sum()``.
    The returned token list includes the bonus posterior token at the first
    rejection, matching the target output update in native DFlash.
    """

    return compute_greedy_acceptance(
        draft_token_ids=draft_token_ids,
        posterior_token_ids=posterior_token_ids,
        remaining_output_len=remaining_output_len,
        stop_token_ids=stop_token_ids,
    )


def compute_acceptance_for_plan(
    plan_item: VerifyPlanItem,
    posterior_token_ids: list[int],
    remaining_output_len: int | None = None,
    stop_token_ids=(),
) -> tuple[list[int], int]:
    if plan_item.proposal_kind == "dflash_block":
        return compute_dflash_block_acceptance(
            plan_item.draft_token_ids,
            posterior_token_ids,
            remaining_output_len=remaining_output_len,
            stop_token_ids=stop_token_ids,
        )
    return compute_greedy_acceptance(
        plan_item.draft_token_ids,
        posterior_token_ids,
        remaining_output_len=remaining_output_len,
        stop_token_ids=stop_token_ids,
    )


def aggregate_max_lens_by_seq_id(
    seq_ids_list: list[int],
    seq_lens_list: list[int],
) -> dict[int, int]:
    if len(seq_ids_list) != len(seq_lens_list):
        raise ValueError("seq_ids_list and seq_lens_list must have the same length")

    max_lens_by_seq_id: dict[int, int] = {}
    for seq_id, seq_len in zip(seq_ids_list, seq_lens_list):
        max_lens_by_seq_id[seq_id] = max(max_lens_by_seq_id.get(seq_id, 0), seq_len)
    return max_lens_by_seq_id
