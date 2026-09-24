"""Backend-local helpers shared by SwiftLLM engine and direct target runners."""

from __future__ import annotations

import dataclasses

from swiftllm.server.structs import Request
from swiftllm.speculative import VerifyPlanItem, build_verification_plan


class RequestIdManager:
    """Maintain worker-local request rows without depending on Scheduler."""

    def __init__(self, max_id: int):
        self.max_id = int(max_id)
        self.available_ids = list(range(self.max_id))
        self.available_ids.reverse()

    def get_id(self) -> int:
        if not self.available_ids:
            raise RuntimeError("No more available request ids. Please increase max_seqs_in_block_table")
        return self.available_ids.pop()

    def free_id(self, req_id: int) -> None:
        req_id = int(req_id)
        if req_id < 0 or req_id >= self.max_id:
            raise ValueError("request id is out of range")
        if req_id in self.available_ids:
            raise RuntimeError("request id is already free")
        self.available_ids.append(req_id)

    def free_ids(self, req_ids: list[int]) -> None:
        for req_id in req_ids:
            self.free_id(req_id)


@dataclasses.dataclass
class ForwardRow:
    request: Request
    input_ids: list[int]
    seq_id: int
    seq_len: int | None = None
    kind: str = "prefill"
    output_row_start: int = -1


@dataclasses.dataclass
class BatchPlan:
    prefill_rows: list[ForwardRow]
    decode_rows: list[ForwardRow]
    normal_decode_rows: list[ForwardRow]
    verify_plan_items: list[VerifyPlanItem]

    @property
    def rows(self) -> list[ForwardRow]:
        return self.prefill_rows + self.decode_rows

    def input_ids_list(self) -> list[list[int]]:
        return [row.input_ids for row in self.rows]

    def seq_ids_list(self) -> list[int]:
        return [row.seq_id for row in self.rows]

    def decoding_seq_lens_list(self) -> list[int]:
        return [row.seq_len for row in self.decode_rows if row.seq_len is not None]


def build_batch_plan(cur_batch: list[Request]) -> BatchPlan:
    prefill_rows = [
        ForwardRow(req, req.prompt_token_ids, req.request_id, kind="prefill")
        for req in cur_batch
        if req.is_prefill_stage()
    ]
    decode_rows: list[ForwardRow] = []
    normal_decode_rows: list[ForwardRow] = []
    verify_plan_items: list[VerifyPlanItem] = []

    for req in cur_batch:
        if req.is_prefill_stage():
            continue
        if req.spec_proposal is not None:
            plan_item = build_verification_plan(req, req.spec_proposal)
            plan_item.output_row_start = len(prefill_rows) + len(decode_rows)
            plan_item.output_row_count = len(plan_item.input_token_ids)
            verify_plan_items.append(plan_item)
            for token_id, seq_len in zip(plan_item.input_token_ids, plan_item.seq_lens):
                decode_rows.append(
                    ForwardRow(
                        req,
                        [token_id],
                        req.request_id,
                        seq_len=seq_len,
                        kind="spec_verify",
                    )
                )
        else:
            row = ForwardRow(
                req,
                [req.next_input_token()],
                req.request_id,
                seq_len=req.logical_decode_seq_len(),
                kind="normal_decode",
            )
            row.output_row_start = len(prefill_rows) + len(decode_rows)
            decode_rows.append(row)
            normal_decode_rows.append(row)

    return BatchPlan(prefill_rows, decode_rows, normal_decode_rows, verify_plan_items)
