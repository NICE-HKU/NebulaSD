from __future__ import annotations

from dataclasses import dataclass

import pytest

from swiftllm.engine_config import EngineConfig
from swiftllm.server import draft_session


@dataclass
class _Scalar:
    value: int = 0

    def item(self):
        return self.value


class _BlockManager:
    def __init__(self, rows: int):
        self.num_seq_allocated_blocks = [_Scalar() for _ in range(rows)]


@dataclass
class _ModelConfig:
    vocab_size: int = 1000
    max_position_embeddings: int = 4096


class _FakeModel:
    fail_forward = False
    instances = []

    def __init__(self, config):
        self.config = config
        self.gpu_block_manager = None
        self.model_config = _ModelConfig()
        self.load_calls = 0
        self.forward_calls = []
        self.crop_calls = []
        self.free_calls = []
        self.tokens: dict[int, list[int]] = {}
        self.__class__.instances.append(self)

    def load_weights(self):
        self.load_calls += 1

    def profile_num_blocks(self):
        return 64

    def init_kvcache_and_swap(self, _num_blocks):
        self.gpu_block_manager = _BlockManager(self.config.max_seqs_in_block_table)

    def forward(self, input_ids, rows, decode_lens):
        if self.fail_forward:
            raise RuntimeError("forward poisoned")
        self.forward_calls.append((tuple(map(tuple, input_ids)), tuple(rows), tuple(decode_lens)))
        num_prefill = len(input_ids) - len(decode_lens)
        out = []
        for index, (tokens, row) in enumerate(zip(input_ids, rows, strict=True)):
            if index < num_prefill:
                self.tokens[row] = list(tokens)
            else:
                assert len(self.tokens[row]) + 1 == decode_lens[index - num_prefill]
                self.tokens[row].extend(tokens)
            self.gpu_block_manager.num_seq_allocated_blocks[row].value = (len(self.tokens[row]) + 15) // 16
            out.append(_next(self.tokens[row]))
        return out

    def crop_seqs_resources(self, rows, lengths):
        self.crop_calls.append((tuple(rows), tuple(lengths)))
        for row, length in zip(rows, lengths, strict=True):
            del self.tokens[row][length:]
            self.gpu_block_manager.num_seq_allocated_blocks[row].value = (length + 15) // 16

    def free_seqs_resources(self, rows):
        self.free_calls.append(tuple(rows))
        for row in rows:
            self.tokens.pop(row, None)
            self.gpu_block_manager.num_seq_allocated_blocks[row].value = 0


def _config(**changes):
    values = dict(
        model_path="/local/model",
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.5,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=64,
        max_batch_size=4,
        max_tokens_in_batch=64,
        speculative_method="none",
        enable_double_bank=False,
    )
    values.update(changes)
    return EngineConfig(**values)


def _next(tokens):
    return (sum(tokens) * 13 + len(tokens)) % 997


def test_exact_prefill_decode_crop_release_and_replay(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", _FakeModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(_config())
    adapter.initialize()
    keys = (draft_session.DraftSessionKey("a", 1), draft_session.DraftSessionKey("b", 2))

    prefill = adapter.prefill_batch(
        (
            draft_session.DraftPrefillItem(keys[0], (1, 2)),
            draft_session.DraftPrefillItem(keys[1], (3, 4, 5)),
        )
    )
    decoded = adapter.decode_batch(
        (
            draft_session.DraftDecodeItem(keys[0], prefill[0].token_id, 2),
            draft_session.DraftDecodeItem(keys[1], prefill[1].token_id, 3),
        )
    )
    adapter.crop_batch(
        (
            draft_session.DraftCropItem(keys[0], decoded[0].logical_kv_len, 2),
            draft_session.DraftCropItem(keys[1], decoded[1].logical_kv_len, 3),
        )
    )

    assert tuple(item.key for item in prefill) == keys
    assert tuple(item.logical_kv_len for item in decoded) == (3, 4)
    assert adapter.snapshot().active_session_count == 2
    assert adapter.snapshot().allocated_gpu_block_count == 2

    adapter.release_batch(keys)
    adapter.release_batch(keys)
    assert adapter.snapshot().active_session_count == 0
    assert adapter.snapshot().release_tombstone_count == 2
    assert adapter.snapshot().available_row_count == 4


def test_whole_batch_preflight_has_zero_forward_or_session_side_effect(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", _FakeModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(_config())
    adapter.initialize()
    model = _FakeModel.instances[-1]
    key = draft_session.DraftSessionKey("a", 1)

    with pytest.raises(draft_session.DraftSessionError, match="unique"):
        adapter.prefill_batch(
            (
                draft_session.DraftPrefillItem(key, (1,)),
                draft_session.DraftPrefillItem(key, (2,)),
            )
        )

    assert model.forward_calls == []
    assert adapter.snapshot().active_session_count == 0
    assert adapter.snapshot().available_row_count == 4


def test_prefill_row_acquisition_failure_rolls_back_without_session_or_forward(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", _FakeModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(_config(max_seqs_in_block_table=2))
    adapter.initialize()
    model = _FakeModel.instances[-1]
    manager = adapter.request_id_manager
    assert manager is not None
    baseline = tuple(manager.available_ids)
    original_get_id = manager.get_id
    calls = 0

    def fail_second():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected row fault")
        return original_get_id()

    monkeypatch.setattr(manager, "get_id", fail_second)

    with pytest.raises(draft_session.DraftSessionError, match="row acquisition failed"):
        adapter.prefill_batch(
            (
                draft_session.DraftPrefillItem(draft_session.DraftSessionKey("a", 1), (1,)),
                draft_session.DraftPrefillItem(draft_session.DraftSessionKey("b", 1), (2,)),
            )
        )

    assert model.forward_calls == []
    assert adapter.snapshot().active_session_count == 0
    assert sorted(manager.available_ids) == sorted(baseline)


def test_adapter_output_fence_rejects_malformed_forward_as_fatal(monkeypatch):
    class BadTokenModel(_FakeModel):
        def forward(self, input_ids, rows, decode_lens):
            super().forward(input_ids, rows, decode_lens)
            return [1000 for _row in rows]

    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", BadTokenModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(_config())
    adapter.initialize()

    with pytest.raises(draft_session.FatalDraftSessionError, match="vocab capacity"):
        adapter.prefill_batch((draft_session.DraftPrefillItem(draft_session.DraftSessionKey("a", 1), (1, 2)),))


def test_session_capacity_exact_boundary_and_plus_one_prefill(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", _FakeModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(
        _config(block_size=4, max_blocks_per_seq=2, max_tokens_in_batch=32)
    )
    adapter.initialize()
    model = _FakeModel.instances[-1]
    ok_key = draft_session.DraftSessionKey("ok", 1)
    bad_key = draft_session.DraftSessionKey("bad", 1)

    adapter.prefill_batch((draft_session.DraftPrefillItem(ok_key, tuple(range(8))),))
    calls = len(model.forward_calls)

    with pytest.raises(draft_session.DraftSessionError, match="session capacity"):
        adapter.prefill_batch((draft_session.DraftPrefillItem(bad_key, tuple(range(9))),))

    assert len(model.forward_calls) == calls
    assert bad_key not in adapter._sessions


def test_session_capacity_uses_model_position_limit_and_batch_second_zero_side_effect(monkeypatch):
    class ShortContextModel(_FakeModel):
        def __init__(self, config):
            super().__init__(config)
            self.model_config.max_position_embeddings = 5

    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", ShortContextModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(
        _config(block_size=8, max_blocks_per_seq=8, max_tokens_in_batch=64)
    )
    adapter.initialize()
    model = _FakeModel.instances[-1]
    good_key = draft_session.DraftSessionKey("good", 1)
    bad_key = draft_session.DraftSessionKey("bad", 1)
    baseline_rows = tuple(adapter.request_id_manager.available_ids)

    with pytest.raises(draft_session.DraftSessionError, match="session capacity"):
        adapter.prefill_batch(
            (
                draft_session.DraftPrefillItem(good_key, (1, 2, 3)),
                draft_session.DraftPrefillItem(bad_key, (1, 2, 3, 4, 5, 6)),
            )
        )

    assert model.forward_calls == []
    assert adapter.snapshot().active_session_count == 0
    assert adapter.request_id_manager is not None
    assert adapter.request_id_manager.available_ids == list(baseline_rows)


def test_decode_capacity_plus_one_rejects_before_forward(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", _FakeModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(
        _config(block_size=4, max_blocks_per_seq=1, max_tokens_in_batch=32)
    )
    adapter.initialize()
    model = _FakeModel.instances[-1]
    key = draft_session.DraftSessionKey("a", 1)
    seed = adapter.prefill_batch((draft_session.DraftPrefillItem(key, (1, 2, 3, 4)),))[0]
    calls = len(model.forward_calls)

    with pytest.raises(draft_session.DraftSessionError, match="session capacity"):
        adapter.decode_batch((draft_session.DraftDecodeItem(key, seed.token_id, 4),))

    assert len(model.forward_calls) == calls
    assert adapter.snapshot().active_session_count == 1


def test_decode_fence_rejects_before_forward_and_unknown_forward_is_fatal(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", _FakeModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(_config())
    adapter.initialize()
    model = _FakeModel.instances[-1]
    key = draft_session.DraftSessionKey("a", 1)
    seed = adapter.prefill_batch((draft_session.DraftPrefillItem(key, (1, 2)),))[0]
    calls = len(model.forward_calls)

    with pytest.raises(draft_session.DraftSessionError, match="logical KV"):
        adapter.decode_batch((draft_session.DraftDecodeItem(key, seed.token_id, 1),))
    assert len(model.forward_calls) == calls

    model.fail_forward = True
    with pytest.raises(draft_session.FatalDraftSessionError, match="after forward started"):
        adapter.decode_batch((draft_session.DraftDecodeItem(key, seed.token_id, 2),))
    assert adapter.snapshot().active_session_count == 1


def test_shutdown_releases_all_sessions_once(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", _FakeModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(_config())
    adapter.initialize()
    model = _FakeModel.instances[-1]
    key = draft_session.DraftSessionKey("a", 1)
    adapter.prefill_batch((draft_session.DraftPrefillItem(key, (1, 2)),))

    adapter.shutdown()
    adapter.shutdown()

    assert model.free_calls == [(0,)]
    assert adapter.snapshot().active_session_count == 0


def test_release_row_bookkeeping_failure_is_fatal_without_tombstone(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", _FakeModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(_config())
    adapter.initialize()
    key = draft_session.DraftSessionKey("a", 1)
    adapter.prefill_batch((draft_session.DraftPrefillItem(key, (1, 2)),))
    manager = adapter.request_id_manager
    assert manager is not None

    def fail_free(_row):
        raise RuntimeError("injected row free fault")

    monkeypatch.setattr(manager, "free_id", fail_free)

    with pytest.raises(draft_session.FatalDraftSessionError, match="row bookkeeping"):
        adapter.release_batch((key,))

    snapshot = adapter.snapshot()
    assert snapshot.active_session_count == 1
    assert snapshot.release_tombstone_count == 0


def test_release_batch_second_row_bookkeeping_failure_is_fatal_without_tombstone(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(draft_session, "LlamaModel", _FakeModel)
    adapter = draft_session.SwiftLLMDraftSessionAdapter(_config())
    adapter.initialize()
    keys = (draft_session.DraftSessionKey("a", 1), draft_session.DraftSessionKey("b", 1))
    adapter.prefill_batch(
        (
            draft_session.DraftPrefillItem(keys[0], (1, 2)),
            draft_session.DraftPrefillItem(keys[1], (3, 4)),
        )
    )
    manager = adapter.request_id_manager
    assert manager is not None
    original_free = manager.free_id
    calls = 0

    def fail_second(row):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second row free fault")
        original_free(row)

    monkeypatch.setattr(manager, "free_id", fail_second)

    with pytest.raises(draft_session.FatalDraftSessionError, match="row bookkeeping"):
        adapter.release_batch(keys)

    snapshot = adapter.snapshot()
    assert snapshot.active_session_count == 2
    assert snapshot.release_tombstone_count == 0
    with pytest.raises(draft_session.FatalDraftSessionError, match="fail-stop state"):
        adapter.release_batch(keys)


def test_double_bank_configuration_is_rejected():
    with pytest.raises(ValueError, match="ordinary bitmap"):
        draft_session.SwiftLLMDraftSessionAdapter(_config(enable_double_bank=True))
