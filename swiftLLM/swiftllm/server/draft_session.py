"""Typed process-local ordinary SwiftLLM sessions for draft generation."""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Sequence

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.server.backend_local import RequestIdManager
from swiftllm.worker.model import LlamaModel


class DraftSessionError(ValueError):
    """A request was rejected before any model/session mutation."""


class FatalDraftSessionError(RuntimeError):
    """A model or allocator failure made process-local state untrustworthy."""


@dataclasses.dataclass(frozen=True)
class DraftSessionKey:
    request_id: str
    request_epoch: int

    def __post_init__(self) -> None:
        if not str(self.request_id):
            raise ValueError("request_id must be non-empty")
        if int(self.request_epoch) < 0:
            raise ValueError("request_epoch must be non-negative")
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "request_epoch", int(self.request_epoch))


@dataclasses.dataclass(frozen=True)
class DraftPrefillItem:
    key: DraftSessionKey
    input_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        tokens = tuple(int(token) for token in self.input_token_ids)
        if not tokens or any(token < 0 for token in tokens):
            raise ValueError("prefill input_token_ids must be non-empty and non-negative")
        object.__setattr__(self, "input_token_ids", tokens)


@dataclasses.dataclass(frozen=True)
class DraftDecodeItem:
    key: DraftSessionKey
    input_token_id: int
    expected_logical_kv_len: int

    def __post_init__(self) -> None:
        if int(self.input_token_id) < 0:
            raise ValueError("decode input_token_id must be non-negative")
        if int(self.expected_logical_kv_len) < 0:
            raise ValueError("expected_logical_kv_len must be non-negative")
        object.__setattr__(self, "input_token_id", int(self.input_token_id))
        object.__setattr__(self, "expected_logical_kv_len", int(self.expected_logical_kv_len))


@dataclasses.dataclass(frozen=True)
class DraftCropItem:
    key: DraftSessionKey
    expected_logical_kv_len: int
    target_logical_kv_len: int

    def __post_init__(self) -> None:
        current = int(self.expected_logical_kv_len)
        target = int(self.target_logical_kv_len)
        if current < 0 or target < 0:
            raise ValueError("crop logical lengths must be non-negative")
        if target > current:
            raise ValueError("crop target cannot exceed current logical KV length")
        object.__setattr__(self, "expected_logical_kv_len", current)
        object.__setattr__(self, "target_logical_kv_len", target)


@dataclasses.dataclass(frozen=True)
class DraftForwardResult:
    key: DraftSessionKey
    token_id: int
    logical_kv_len: int

    def __post_init__(self) -> None:
        token = int(self.token_id)
        length = int(self.logical_kv_len)
        if token < 0:
            raise ValueError("forward token_id must be non-negative")
        if length < 0:
            raise ValueError("forward logical_kv_len must be non-negative")
        object.__setattr__(self, "token_id", token)
        object.__setattr__(self, "logical_kv_len", length)


@dataclasses.dataclass(frozen=True)
class DraftSessionResourceSnapshot:
    initialized: bool
    active_session_count: int
    release_tombstone_count: int
    available_row_count: int
    row_capacity: int
    allocated_gpu_block_count: int
    prefill_batch_sizes: tuple[int, ...]
    decode_batch_sizes: tuple[int, ...]
    crop_batch_sizes: tuple[int, ...]


@dataclasses.dataclass
class _DraftSession:
    key: DraftSessionKey
    row: int
    logical_kv_len: int


class SwiftLLMDraftSessionAdapter:
    """Own ordinary bitmap rows/KV without a scheduler or target bank facade."""

    def __init__(self, engine_config: EngineConfig, *, replay_capacity: int = 4096) -> None:
        if bool(getattr(engine_config, "enable_double_bank", False)) and not getattr(self, "_bank_sessions", False):
            raise ValueError("draft sessions require the ordinary bitmap block manager")
        if int(replay_capacity) <= 0:
            raise ValueError("replay_capacity must be positive")
        self.engine_config = engine_config
        self.model: LlamaModel | None = None
        self.request_id_manager: RequestIdManager | None = None
        self._sessions: dict[DraftSessionKey, _DraftSession] = {}
        self._released: OrderedDict[DraftSessionKey, None] = OrderedDict()
        self._replay_capacity = int(replay_capacity)
        self._initialized = False
        self._shutdown = False
        self._poisoned = False
        self._prefill_batch_sizes: list[int] = []
        self._decode_batch_sizes: list[int] = []
        self._crop_batch_sizes: list[int] = []

    def initialize(self) -> None:
        if self._shutdown:
            raise DraftSessionError("draft session adapter is shut down")
        if self._initialized:
            return
        try:
            model = LlamaModel(self.engine_config)
            model.load_weights()
            num_blocks = model.profile_num_blocks()
            model.init_kvcache_and_swap(num_blocks)
        except Exception as exc:
            raise FatalDraftSessionError(f"SwiftLLM draft model initialization failed: {exc}") from exc
        self.model = model
        self.request_id_manager = RequestIdManager(self.engine_config.max_seqs_in_block_table)
        self._initialized = True

    def prefill_batch(self, items: Sequence[DraftPrefillItem]) -> tuple[DraftForwardResult, ...]:
        items = tuple(items)
        self.preflight_batch(prefill_batches=(items,))
        manager = self._require_row_manager()
        rows: list[int] = []
        try:
            for _item in items:
                rows.append(manager.get_id())
        except Exception as exc:
            for row in rows:
                manager.free_id(row)
            raise DraftSessionError("draft session row acquisition failed") from exc
        for item, row in zip(items, rows, strict=True):
            self._sessions[item.key] = _DraftSession(item.key, row, len(item.input_token_ids))
        try:
            tokens = self._forward(
                [list(item.input_token_ids) for item in items],
                rows,
                [],
            )
            self._validate_tokens(tokens, expected_count=len(items))
        except Exception as exc:
            self._poisoned = True
            raise FatalDraftSessionError(f"SwiftLLM draft prefill failed after session allocation: {exc}") from exc
        self._prefill_batch_sizes.append(len(items))
        return tuple(
            DraftForwardResult(item.key, int(token), len(item.input_token_ids))
            for item, token in zip(items, tokens, strict=True)
        )

    def decode_batch(self, items: Sequence[DraftDecodeItem]) -> tuple[DraftForwardResult, ...]:
        items = tuple(items)
        self.preflight_batch(decode_batches=(items,))
        sessions = self._preflight_live(items, expected_attr="expected_logical_kv_len")
        try:
            tokens = self._forward(
                [[item.input_token_id] for item in items],
                [session.row for session in sessions],
                [session.logical_kv_len + 1 for session in sessions],
            )
            self._validate_tokens(tokens, expected_count=len(items))
        except Exception as exc:
            self._poisoned = True
            raise FatalDraftSessionError(f"SwiftLLM draft decode failed after forward started: {exc}") from exc
        for session in sessions:
            session.logical_kv_len += 1
        self._decode_batch_sizes.append(len(items))
        return tuple(
            DraftForwardResult(item.key, int(token), session.logical_kv_len)
            for item, token, session in zip(items, tokens, sessions, strict=True)
        )

    def crop_batch(self, items: Sequence[DraftCropItem]) -> None:
        items = tuple(items)
        self.preflight_batch(crop_batches=(items,))
        sessions = self._preflight_live(items, expected_attr="expected_logical_kv_len")
        try:
            self._require_model().crop_seqs_resources(
                [session.row for session in sessions],
                [item.target_logical_kv_len for item in items],
            )
        except Exception as exc:
            self._poisoned = True
            raise FatalDraftSessionError(f"SwiftLLM draft crop failed after mutation started: {exc}") from exc
        for item, session in zip(items, sessions, strict=True):
            session.logical_kv_len = item.target_logical_kv_len
        self._crop_batch_sizes.append(len(items))

    def release_batch(self, keys: Sequence[DraftSessionKey]) -> None:
        keys = tuple(keys)
        if not keys or len(set(keys)) != len(keys):
            raise DraftSessionError("release requires unique session keys")
        live = tuple(key in self._sessions for key in keys)
        replay = tuple(key in self._released for key in keys)
        if all(replay):
            return
        if not all(live) or any(replay):
            raise DraftSessionError("release contains unknown or mixed live/replayed session")
        sessions = tuple(self._sessions[key] for key in keys)
        try:
            self._require_model().free_seqs_resources([session.row for session in sessions])
        except Exception as exc:
            self._poisoned = True
            raise FatalDraftSessionError(f"SwiftLLM draft release failed after mutation started: {exc}") from exc
        manager = self._require_row_manager()
        try:
            for session in sessions:
                manager.free_id(session.row)
        except Exception as exc:
            self._poisoned = True
            raise FatalDraftSessionError(f"SwiftLLM draft release row bookkeeping failed after GPU release: {exc}") from exc
        for key in keys:
            self._sessions.pop(key)
            self._released[key] = None
            self._released.move_to_end(key)
        while len(self._released) > self._replay_capacity:
            self._released.popitem(last=False)

    def snapshot(self) -> DraftSessionResourceSnapshot:
        allocated = 0
        if self.model is not None and self.model.gpu_block_manager is not None:
            allocated = sum(
                int(self.model.gpu_block_manager.num_seq_allocated_blocks[session.row].item())
                for session in self._sessions.values()
            )
        available = 0 if self.request_id_manager is None else len(self.request_id_manager.available_ids)
        return DraftSessionResourceSnapshot(
            self._initialized,
            len(self._sessions),
            len(self._released),
            available,
            int(self.engine_config.max_seqs_in_block_table),
            allocated,
            tuple(self._prefill_batch_sizes),
            tuple(self._decode_batch_sizes),
            tuple(self._crop_batch_sizes),
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        errors: list[str] = []
        if self._sessions:
            try:
                self.release_batch(tuple(self._sessions))
            except Exception as exc:
                errors.append(str(exc))
        self._shutdown = True
        self._initialized = False
        self.model = None
        self.request_id_manager = None
        self._poisoned = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if errors:
            raise RuntimeError("; ".join(errors))

    def preflight_batch(
        self,
        *,
        prefill_batches: Sequence[Sequence[DraftPrefillItem]] = (),
        decode_batches: Sequence[Sequence[DraftDecodeItem]] = (),
        crop_batches: Sequence[Sequence[DraftCropItem]] = (),
    ) -> None:
        """Validate a whole draft-session transaction before any KV mutation."""

        self._check_ready()
        manager = self._require_row_manager()
        prefill_batches = tuple(tuple(batch) for batch in prefill_batches)
        decode_batches = tuple(tuple(batch) for batch in decode_batches)
        crop_batches = tuple(tuple(batch) for batch in crop_batches)
        if not prefill_batches and not decode_batches and not crop_batches:
            raise DraftSessionError("draft session preflight requires at least one batch")

        logical_by_key = {key: session.logical_kv_len for key, session in self._sessions.items()}
        new_keys: set[DraftSessionKey] = set()
        capacity = self._session_logical_capacity()

        for batch in crop_batches:
            self._preflight_simulated_live(batch, logical_by_key, expected_attr="expected_logical_kv_len")
            for item in batch:
                if item.target_logical_kv_len > logical_by_key[item.key]:
                    raise DraftSessionError("crop target cannot exceed current logical KV length")
                if item.target_logical_kv_len > capacity:
                    raise DraftSessionError("draft session logical length exceeds session capacity")
                logical_by_key[item.key] = item.target_logical_kv_len

        for batch in decode_batches:
            self._preflight_simulated_live(batch, logical_by_key, expected_attr="expected_logical_kv_len")
            self._check_forward_capacity(tuple((item.input_token_id,) for item in batch))
            for item in batch:
                next_len = logical_by_key[item.key] + 1
                if next_len > capacity:
                    raise DraftSessionError("draft session logical length exceeds session capacity")
                logical_by_key[item.key] = next_len

        new_row_count = sum(len(batch) for batch in prefill_batches)
        if new_row_count > len(manager.available_ids):
            raise DraftSessionError("draft session row capacity exhausted")
        for batch in prefill_batches:
            self._preflight_simulated_new(batch, logical_by_key, new_keys)
            self._check_forward_capacity(tuple(item.input_token_ids for item in batch))
            for item in batch:
                if len(item.input_token_ids) > capacity:
                    raise DraftSessionError("draft session logical length exceeds session capacity")
                new_keys.add(item.key)
                logical_by_key[item.key] = len(item.input_token_ids)

    def _preflight_live(self, items: tuple, *, expected_attr: str) -> tuple[_DraftSession, ...]:
        self._check_ready()
        keys = tuple(item.key for item in items)
        if not keys or len(set(keys)) != len(keys):
            raise DraftSessionError("batch requires unique live session keys")
        sessions = []
        for item in items:
            session = self._sessions.get(item.key)
            if session is None:
                raise DraftSessionError("unknown live draft session")
            if session.logical_kv_len != int(getattr(item, expected_attr)):
                raise DraftSessionError("draft session logical KV length fence mismatch")
            sessions.append(session)
        if len(items) > int(self.engine_config.max_batch_size):
            raise DraftSessionError("draft batch exceeds max_batch_size")
        return tuple(sessions)

    def _preflight_simulated_new(
        self,
        items: tuple[DraftPrefillItem, ...],
        logical_by_key: dict[DraftSessionKey, int],
        new_keys: set[DraftSessionKey],
    ) -> None:
        keys = tuple(item.key for item in items)
        if not keys or len(set(keys)) != len(keys):
            raise DraftSessionError("prefill requires unique session keys")
        if any(key in logical_by_key or key in self._released or key in new_keys for key in keys):
            raise DraftSessionError("prefill session key already exists")
        if len(items) > int(self.engine_config.max_batch_size):
            raise DraftSessionError("draft prefill batch exceeds max_batch_size")

    def _preflight_simulated_live(
        self,
        items: tuple,
        logical_by_key: dict[DraftSessionKey, int],
        *,
        expected_attr: str,
    ) -> None:
        keys = tuple(item.key for item in items)
        if not keys or len(set(keys)) != len(keys):
            raise DraftSessionError("batch requires unique live session keys")
        if len(items) > int(self.engine_config.max_batch_size):
            raise DraftSessionError("draft batch exceeds max_batch_size")
        for item in items:
            if item.key in self._released or item.key not in logical_by_key:
                raise DraftSessionError("unknown live draft session")
            if logical_by_key[item.key] != int(getattr(item, expected_attr)):
                raise DraftSessionError("draft session logical KV length fence mismatch")

    def _forward(self, input_ids: list[list[int]], rows: list[int], decode_lens: list[int]) -> list[int]:
        self._check_forward_capacity(tuple(tuple(item) for item in input_ids))
        return list(self._require_model().forward(input_ids, rows, decode_lens))

    def _check_forward_capacity(self, input_ids: tuple[tuple[int, ...], ...]) -> None:
        if len(input_ids) > int(self.engine_config.max_batch_size):
            raise DraftSessionError("draft forward exceeds max_batch_size")
        if sum(len(item) for item in input_ids) > int(self.engine_config.max_tokens_in_batch):
            raise DraftSessionError("draft forward exceeds max_tokens_in_batch")

    def _validate_tokens(self, tokens: list[int], *, expected_count: int) -> None:
        if len(tokens) != int(expected_count):
            raise FatalDraftSessionError("SwiftLLM draft forward result count mismatch")
        vocab_size = self._model_vocab_size()
        for token in tokens:
            token_id = int(token)
            if token_id < 0 or token_id >= vocab_size:
                raise FatalDraftSessionError("SwiftLLM draft forward token outside model vocab capacity")

    def _model_vocab_size(self) -> int:
        model_config = getattr(self._require_model(), "model_config", None)
        vocab_size = getattr(model_config, "vocab_size", None)
        if vocab_size is None:
            raise FatalDraftSessionError("SwiftLLM draft model vocab capacity is unavailable")
        vocab = int(vocab_size)
        if vocab <= 0:
            raise FatalDraftSessionError("SwiftLLM draft model vocab capacity must be positive")
        return vocab

    def _session_logical_capacity(self) -> int:
        block_capacity = int(self.engine_config.max_blocks_per_seq) * int(self.engine_config.block_size)
        model_config = getattr(self._require_model(), "model_config", None)
        max_positions = getattr(model_config, "max_position_embeddings", None)
        if max_positions is None:
            max_positions = block_capacity
        position_capacity = int(max_positions)
        capacity = min(block_capacity, position_capacity)
        if capacity <= 0:
            raise FatalDraftSessionError("SwiftLLM draft session capacity must be positive")
        return capacity

    def _check_ready(self) -> None:
        if self._shutdown:
            raise DraftSessionError("draft session adapter is shut down")
        if self._poisoned:
            raise FatalDraftSessionError("draft session adapter is in fail-stop state")
        if not self._initialized:
            raise DraftSessionError("draft session adapter is not initialized")

    def _require_model(self) -> LlamaModel:
        self._check_ready()
        if self.model is None:
            raise FatalDraftSessionError("initialized adapter is missing its model")
        return self.model

    def _require_row_manager(self) -> RequestIdManager:
        self._check_ready()
        if self.request_id_manager is None:
            raise FatalDraftSessionError("initialized adapter is missing its row manager")
        return self.request_id_manager


__all__ = (
    "DraftCropItem",
    "DraftDecodeItem",
    "DraftForwardResult",
    "DraftPrefillItem",
    "DraftSessionError",
    "DraftSessionKey",
    "DraftSessionResourceSnapshot",
    "FatalDraftSessionError",
    "SwiftLLMDraftSessionAdapter",
)
