"""Compact execution preserves canonical verify rows, acceptance and crop."""
import asyncio
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import swiftllm
import swiftllm.server.starsd_target_facade as canonical_facade
for module in (swiftllm, canonical_facade):
    print(f'{module.__name__}.__file__={Path(module.__file__).resolve()}', flush=True)
    assert Path(module.__file__).resolve().is_relative_to(ROOT)

from swiftllm.server.starsd_compact_target import CompactVerifyBatchPlan, CompactVerifyRequest
from swiftllm.server.starsd_process_local_facade import SwiftLLMProcessLocalTargetFacade
from swiftllm.server.target_worker import SwiftLLMTargetWorker
from swiftllm.speculative import DraftProposal, build_verification_plan, compute_acceptance_for_plan
from swiftllm.worker.model import ModelForwardOutput


class Manager:
    block_size = 16
    active_bank_id = 0
    double_bank_enabled = True

    def __init__(self):
        self.bank = SimpleNamespace(epoch=3, batch_id='9', ready_event=None,
            request_ranges={r: SimpleNamespace(logical_kv_len=512, kv_version=5, num_blocks=33)
                            for r in (4, 7)})

    def get_bank_descriptor(self, bank):
        assert bank == 0
        return self.bank

    def validate_bank_location(self, location):
        assert location in self.bank.request_ranges.values()

    def set_bank_location_kv_version(self, row, *, bank_id, bank_epoch, kv_version):
        assert bank_id == 0 and bank_epoch == 3
        location = self.bank.request_ranges[row]
        location.kv_version = kv_version
        return location


class Model:
    k_cache = None

    def __init__(self, manager):
        self.gpu_block_manager = manager
        self.inputs = None
        self.crops = None

    def forward(self, tokens, rows, lengths, **kwargs):
        self.inputs = tokens, rows, lengths, kwargs
        return ModelForwardOutput([11, 12, 90, 91, 21, 22, 23, 24])

    def crop_seqs_resources(self, rows, lengths):
        self.crops = rows, lengths
        for row, length in zip(rows, lengths):
            self.gpu_block_manager.bank.request_ranges[row].logical_kv_len = length
            self.gpu_block_manager.bank.request_ranges[row].kv_version += 1


class Worker:
    initialized = True
    _batch_lock = None
    _select_row_hidden_states = SwiftLLMTargetWorker._select_row_hidden_states
    _hidden_payload_for_rows = SwiftLLMTargetWorker._hidden_payload_for_rows

    def __init__(self):
        self.model = Model(Manager())
        self.model_config = SimpleNamespace(vocab_size=1000)
        self.engine_config = SimpleNamespace(max_batch_size=8, max_tokens_in_batch=64,
                                             speculative_max_draft_tokens=4)
        self.handoffs = 0

    async def _run_on_model_async(self, fn):
        self.handoffs += 1
        return fn()

    def _target_hidden_layer_ids(self):
        return None

    def _allocated_block_count_for_row(self, row):
        logical = self.model.gpu_block_manager.bank.request_ranges[row].logical_kv_len
        return (logical + 15) // 16


def plan(kind='linear', remaining=100, stops=()):
    return CompactVerifyBatchPlan(0, 3, 9, (
        CompactVerifyRequest('a', 'tag-a', 4, 512, 5, 10, (11, 12, 13), remaining, stops, kind),
        CompactVerifyRequest('b', 'tag-b', 7, 512, 5, 20, (21, 22, 23), remaining, stops, kind),
    ))


@pytest.mark.parametrize('kind', ('linear', 'dflash_block'))
@pytest.mark.parametrize('remaining,stops', ((100, ()), (2, ()), (100, (12, 22))))
def test_same_rows_acceptance_and_crop_without_history(kind, remaining, stops):
    worker = Worker()
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    compact = plan(kind, remaining, stops)
    result = asyncio.run(facade.verify_batch_compact(compact))
    expected_tokens, expected_rows, expected_lens = [], [], []
    for index, (item, actual) in enumerate(zip(compact.requests, result.value)):
        history = [0] * 32 + [item.anchor_token_id]
        request = SimpleNamespace(prompt_len=480, output_token_ids=history)
        expected_plan = build_verification_plan(request,
            DraftProposal(item.request_row, kind, list(item.draft_token_ids)))
        posterior = [11, 12, 90, 91] if index == 0 else [21, 22, 23, 24]
        tokens, count = compute_acceptance_for_plan(expected_plan, posterior,
            remaining_output_len=remaining, stop_token_ids=stops)
        expected_tokens.extend([token] for token in expected_plan.input_token_ids)
        expected_rows.extend([item.request_row] * 4)
        expected_lens.extend(expected_plan.seq_lens)
        assert actual.accepted_token_ids == tuple(tokens)
        assert actual.num_accepted_draft_tokens == count
        assert actual.logical_kv_len == 512 + len(tokens)
        assert actual.finished == (len(tokens) == remaining or tokens[-1] in stops)
        assert actual.kv_version == 7
        assert actual.target_hidden is None
    assert worker.model.inputs[:3] == (expected_tokens, expected_rows, expected_lens)
    assert worker.model.crops == ([4, 7], [r.logical_kv_len for r in result.value])
    assert worker.handoffs == 1


@pytest.mark.parametrize('change', (
    dict(active_bank_id=1), dict(active_bank_epoch=4), dict(batch_seq=10),
    dict(requests=()), dict(requests=(plan().requests[0],) * 2),
    dict(requests=(replace(plan().requests[0], kv_version=4),)),
    dict(requests=(replace(plan().requests[0], logical_kv_len=511),)),
    dict(requests=(replace(plan().requests[0], anchor_token_id=1000),)),
    dict(requests=(replace(plan().requests[0], remaining_output_len=0),)),
))
def test_stale_or_invalid_plan_fails_before_model_mutation(change):
    worker = Worker()
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(facade.verify_batch_compact(replace(plan(), **change)))
    assert worker.model.inputs is None and worker.handoffs == 0


def test_capacity_is_checked_on_actual_gpu_bank():
    worker = Worker()
    worker.model.gpu_block_manager.bank.request_ranges[4].num_blocks = 32
    facade = SwiftLLMProcessLocalTargetFacade(worker=worker)
    with pytest.raises(RuntimeError, match='reserved Bank capacity'):
        asyncio.run(facade.verify_batch_compact(plan()))
    assert worker.model.inputs is None
