"""Reserved Draft sessions over the canonical double-bank block manager.

The metadata owner reserves disjoint rows on its metadata stream. Compute only
uses the activated batch. DMA owns explicit leases until its event has retired;
neither a ready fact nor a request cancellation is permission to free a lease.
No KV tensor is cloned here and restoration never runs prefill.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import RLock

import torch

from .draft_session import (
    SwiftLLMDraftSessionAdapter, DraftSessionKey, DraftSessionError,
    FatalDraftSessionError, DraftForwardResult, _DraftSession,
)


@dataclass(frozen=True)
class DraftBankItem:
    key: DraftSessionKey
    capacity_blocks: int
    logical_kv_len: int = 0
    snapshot_version: int = 0

    def __post_init__(self):
        if not isinstance(self.key, DraftSessionKey):
            raise TypeError("expected DraftSessionKey")
        if self.capacity_blocks <= 0 or self.logical_kv_len < 0 or self.snapshot_version < 0:
            raise ValueError("invalid Draft reservation capacity/length/version")


@dataclass(frozen=True)
class DraftBankBatch:
    bank_id: int
    bank_epoch: int
    batch_seq: int
    items: tuple[DraftBankItem, ...]
    locations: tuple
    restored: bool


@dataclass(frozen=True)
class DraftBankLease:
    bank_id: int
    bank_epoch: int
    batch_seq: int
    sequence: int


class SwiftLLMDraftBankSessionAdapter(SwiftLLMDraftSessionAdapter):
    """Two fixed banks, one compute batch, and exact local replica retirement.

Ordinary sessions remain available to the older standalone API. Bank mode is
explicit and disallows all implicit row allocation and key-only release.
"""
    _bank_sessions = True

    def __init__(self, engine_config, **kwargs):
        if not engine_config.enable_double_bank:
            raise ValueError("bank sessions require enable_double_bank")
        super().__init__(engine_config, **kwargs)
        self._batches = {}
        self._ready_banks = set()
        self._consumed_banks = set()
        self._compute_bank = None
        self._leases = set()
        self._lease_seq = 0
        self._metadata_lock = RLock()
        self._metadata_stream = None
        self._compute_stream = None
        self._metadata_ready = {}
        self._compute_done = {}

    def initialize(self):
        super().initialize()
        cache = self._require_model().k_cache
        if cache.is_cuda and self._metadata_stream is None:
            self._metadata_stream = torch.cuda.Stream(device=cache.device)
            self._compute_stream = torch.cuda.Stream(device=cache.device)
            self._metadata_stream.wait_stream(torch.cuda.current_stream(cache.device))
            self._compute_stream.wait_stream(torch.cuda.current_stream(cache.device))

    @contextmanager
    def metadata_context(self):
        # This lock never covers a model forward or an asynchronous DMA wait.
        with self._metadata_lock:
            cache = self._require_model().k_cache
            with torch.inference_mode(), \
                    (torch.cuda.device(cache.device) if cache.is_cuda else nullcontext()), \
                    (nullcontext() if self._metadata_stream is None else torch.cuda.stream(self._metadata_stream)):
                yield

    def metadata_event(self):
        if self._metadata_stream is None:
            return None
        event = torch.cuda.Event()
        event.record(self._metadata_stream)
        return event

    def describe_banks(self):
        manager = self._require_model().gpu_block_manager
        return tuple(manager.get_bank_descriptor(i) for i in (0, 1))

    @property
    def active_bank_id(self):
        return self._require_model().gpu_block_manager.active_bank_id

    @property
    def computing(self):
        return self._compute_bank is not None

    def caches(self):
        model = self._require_model()
        return model.k_cache, model.v_cache

    @property
    def logical_capacity(self):
        return self._session_logical_capacity()

    def batch_state(self, bank_id):
        batch = self._batches.get(bank_id)
        if batch is None:
            return "EMPTY", None
        if self._compute_bank == bank_id:
            return "COMPUTING", batch
        if bank_id in self._consumed_banks:
            return "DRAINING", batch
        return ("READY" if bank_id in self._ready_banks else "PREPARING"), batch

    def can_reserve_keys(self, request_ids, count, bank_id):
        with self._metadata_lock:
            self._check_ready()
            occupied = {i.key.request_id for b in self._batches.values() for i in b.items}
            return (bank_id not in self._batches and not occupied.intersection(request_ids)
                    and count <= len(self._require_row_manager().available_ids))

    def validate_batch(self, batch):
        self._require_batch(batch)

    def logical_lengths(self, batch):
        self._require_batch(batch)
        return tuple(self._sessions[item.key].logical_kv_len for item in batch.items)

    def logical_length(self, batch, key):
        self._require_batch(batch)
        if key not in {i.key for i in batch.items} or key not in self._sessions:
            raise DraftSessionError("session is not live in this Draft Bank")
        return self._sessions[key].logical_kv_len

    def reserve_batch(self, items, *, bank_id, bank_epoch, batch_seq, restored):
        """Reserve growth capacity and restore row metadata, but not GPU_READY.

        The returned locations describe the destination, never source rows.
        Call complete_import only after the H2D completion event has retired.
        """
        self._check_ready()
        items = tuple(items)
        with self.metadata_context():
            manager = self._require_model().gpu_block_manager
            bank = manager.get_bank_descriptor(bank_id)
            if bank_id in self._batches or bank.request_ranges:
                raise DraftSessionError("Draft Bank still owns a batch")
            if bank_epoch != bank.epoch + 1 or batch_seq < 0:
                raise DraftSessionError("Draft Bank epoch/batch fence mismatch")
            if self._compute_bank is not None and bank_id == manager.active_bank_id:
                raise DraftSessionError("cannot prepare the computing Bank")
            if not items or len(items) > self.engine_config.max_batch_size:
                raise DraftSessionError("invalid Draft reservation batch size")
            keys = tuple(item.key for item in items)
            occupied = {i.key for b in self._batches.values() for i in b.items}
            if len(set(keys)) != len(keys) or occupied.intersection(keys) or set(keys).intersection(self._released):
                raise DraftSessionError("Draft reservation key is already owned or released")
            rows = self._require_row_manager()
            if len(items) > len(rows.available_ids) or sum(i.capacity_blocks for i in items) > bank.num_blocks:
                raise DraftSessionError("Draft Bank row/block capacity exhausted")
            for item in items:
                if (item.capacity_blocks > self.engine_config.max_blocks_per_seq
                        or item.logical_kv_len > min(self._session_logical_capacity(), item.capacity_blocks * manager.block_size)
                        or (restored and item.logical_kv_len == 0)
                        or (not restored and (item.logical_kv_len or item.snapshot_version))):
                    raise DraftSessionError("invalid Draft reservation logical length/capacity")
            acquired = []
            try:
                for _ in items:
                    acquired.append(rows.get_id())
            except Exception as exc:
                for row in acquired:
                    rows.free_id(row)
                raise DraftSessionError("Draft row acquisition failed") from exc
            try:
                locations = manager.reserve_in_bank_batch_atomic(bank_id, [
                    (row, item.capacity_blocks, item.logical_kv_len, item.snapshot_version, str(batch_seq))
                    for row, item in zip(acquired, items, strict=True)
                ], reset_bank=True)
                # Reservation fills capacity; expose only imported valid blocks.
                for row, item, location in zip(acquired, items, locations, strict=True):
                    manager._set_valid_blocks_for_location(row, location,
                        (item.logical_kv_len + manager.block_size - 1) // manager.block_size)
            except Exception as exc:
                # Even rollback can enqueue CUDA work. Keep rows/model alive
                # until explicit stream retirement in shutdown.
                self._poisoned = True
                raise FatalDraftSessionError("Draft reservation metadata failed") from exc
            batch = DraftBankBatch(bank_id, bank_epoch, batch_seq, items, tuple(locations), restored)
            self._batches[bank_id] = batch
            self._metadata_ready[bank_id] = self.metadata_event()
            if not restored:
                self._ready_banks.add(bank_id)
            return batch

    def complete_import(self, batch):
        with self.metadata_context():
            self._require_batch(batch)
            if not batch.restored or batch.bank_id in self._ready_banks:
                raise DraftSessionError("invalid or duplicate Draft import completion")
            if any(lease.bank_id == batch.bank_id for lease in self._leases):
                raise DraftSessionError("Draft import DMA has not retired")
            self._ready_banks.add(batch.bank_id)

    def begin_compute(self, batch):
        with self.metadata_context():
            self._require_batch(batch)
            if (self._compute_bank is not None or batch.bank_id not in self._ready_banks
                    or batch.bank_id in self._consumed_banks):
                raise DraftSessionError("Draft batch is not ready or was already consumed")
            if any(lease.bank_id == batch.bank_id for lease in self._leases):
                raise DraftSessionError("Draft Bank has a live DMA lease")
            manager = self._require_model().gpu_block_manager
            if manager.active_bank_id != batch.bank_id:
                manager.swap_active_standby()
            if batch.restored:
                for item, location in zip(batch.items, batch.locations, strict=True):
                    self._sessions[item.key] = _DraftSession(item.key, location.request_id, item.logical_kv_len)
            self._compute_bank = batch.bank_id
            self._consumed_banks.add(batch.bank_id)

    @contextmanager
    def compute_context(self, batch):
        """Enter on the compute thread, after owner-loop begin_compute.

        The completion event is a process-local CopyPlan dependency. It is not
        a cross-worker handle, and must be supplied to D2H by the caller.
        """
        self._require_batch(batch)
        if self._compute_bank != batch.bank_id:
            raise DraftSessionError("Draft compute context requires the active batch")
        cache = self._require_model().k_cache
        with (torch.cuda.device(cache.device) if cache.is_cuda else nullcontext()), \
                (torch.cuda.stream(self._compute_stream) if cache.is_cuda else nullcontext()):
            ready = self._metadata_ready[batch.bank_id]
            if ready is not None:
                torch.cuda.current_stream(cache.device).wait_event(ready)
            try:
                yield
            except BaseException:
                self._poisoned = True
                raise
            finally:
                if cache.is_cuda:
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(cache.device))
                    self._compute_done[batch.bank_id] = event

    def end_compute(self, batch):
        with self.metadata_context():
            self._require_batch(batch)
            if self._compute_bank != batch.bank_id:
                raise DraftSessionError("Draft compute completion fence mismatch")
            self._compute_bank = None
            return self._compute_done.get(batch.bank_id)

    def acquire_copy_lease(self, batch):
        with self.metadata_context():
            self._require_batch(batch)
            if self._compute_bank == batch.bank_id or any(l.bank_id == batch.bank_id for l in self._leases):
                raise DraftSessionError("Draft Bank is computing or already leased")
            self._lease_seq += 1
            lease = DraftBankLease(batch.bank_id, batch.bank_epoch, batch.batch_seq, self._lease_seq)
            self._leases.add(lease)
            return lease

    def release_copy_lease(self, lease):
        with self.metadata_context():
            if lease not in self._leases:
                raise DraftSessionError("unknown or retired Draft copy lease")
            self._leases.remove(lease)

    def retire_batch(self, batch):
        """Local replica release. Deliberately does not create a tombstone."""
        with self.metadata_context():
            self._require_batch(batch)
            if self._compute_bank == batch.bank_id or any(l.bank_id == batch.bank_id for l in self._leases):
                raise DraftSessionError("cannot retire a referenced Draft Bank")
            event = self._compute_done.get(batch.bank_id)
            if event is not None and not event.query():
                raise DraftSessionError("Draft compute stream has not retired")
            manager = self._require_model().gpu_block_manager
            releases = [(batch.bank_id, batch.bank_epoch, loc.request_id,
                         loc.request_start_block, loc.num_blocks, str(batch.batch_seq)) for loc in batch.locations]
            manager.release_bank_ranges_exact_batch(releases, apply=False)
            try:
                manager.release_bank_ranges_exact_batch(releases)
                for item, location in zip(batch.items, batch.locations, strict=True):
                    self._sessions.pop(item.key, None)
                    self._require_row_manager().free_id(location.request_id)
            except Exception as exc:
                self._poisoned = True
                raise FatalDraftSessionError("Draft exact retirement failed") from exc
            del self._batches[batch.bank_id]
            self._ready_banks.discard(batch.bank_id)
            self._consumed_banks.discard(batch.bank_id)
            self._metadata_ready.pop(batch.bank_id, None)
            self._compute_done.pop(batch.bank_id, None)

    def preflight_batch(self, *, prefill_batches=(), decode_batches=(), crop_batches=()):
        with self._metadata_lock:
            batch = self._active_batch()
            capacities = {item.key: item.capacity_blocks * self.engine_config.block_size for item in batch.items}
            for groups in (prefill_batches, decode_batches, crop_batches):
                for group in groups:
                    for item in group:
                        if item.key not in capacities:
                            raise DraftSessionError("session is not a member of the computing Draft Bank")
                        length = (len(item.input_token_ids) if hasattr(item, "input_token_ids") else
                                  item.target_logical_kv_len if hasattr(item, "target_logical_kv_len") else
                                  item.expected_logical_kv_len + 1)
                        if length > capacities[item.key]:
                            raise DraftSessionError("Draft growth exceeds reserved capacity")
            # New rows are already reserved, so base preflight's row-capacity
            # test must not count them a second time.
            if prefill_batches:
                if batch.restored:
                    raise DraftSessionError("restored Draft session cannot prefill")
                logical = {key: value.logical_kv_len for key, value in self._sessions.items()}
                new_keys = set()
                for group in prefill_batches:
                    self._preflight_simulated_new(tuple(group), logical, new_keys)
                    self._check_forward_capacity(tuple(i.input_token_ids for i in group))
                    for item in group:
                        if len(item.input_token_ids) > self._session_logical_capacity():
                            raise DraftSessionError("Draft prefill exceeds context capacity")
                        logical[item.key] = len(item.input_token_ids)
                        new_keys.add(item.key)
            if decode_batches or crop_batches:
                super().preflight_batch(decode_batches=decode_batches, crop_batches=crop_batches)

    def prefill_batch(self, items):
        items = tuple(items)
        self.preflight_batch(prefill_batches=(items,))
        batch = self._active_batch()
        locations = {item.key: loc for item, loc in zip(batch.items, batch.locations, strict=True)}
        rows = [locations[item.key].request_id for item in items]
        try:
            tokens = self._forward([list(i.input_token_ids) for i in items], rows, [])
            self._validate_tokens(tokens, expected_count=len(items))
            for item, row in zip(items, rows, strict=True):
                self._sessions[item.key] = _DraftSession(item.key, row, len(item.input_token_ids))
        except Exception as exc:
            self._poisoned = True
            raise FatalDraftSessionError("reserved Draft prefill failed") from exc
        self._prefill_batch_sizes.append(len(items))
        return tuple(DraftForwardResult(i.key, int(t), len(i.input_token_ids)) for i, t in zip(items, tokens, strict=True))

    def release_batch(self, keys):
        raise DraftSessionError("bank sessions require exact retire_batch after references retire")

    def shutdown(self):
        if self._shutdown:
            return
        if self._compute_bank is not None or self._leases:
            raise DraftSessionError("retire Draft compute and DMA before shutdown")
        if self._metadata_stream is not None:
            self._metadata_stream.synchronize()
        if self._compute_stream is not None:
            self._compute_stream.synchronize()
        for event in self._compute_done.values():
            event.synchronize()
        if self._poisoned:
            raise FatalDraftSessionError("poisoned Draft sessions retain model resources until process exit")
        for batch in tuple(self._batches.values()):
            self.retire_batch(batch)
        if self._metadata_stream is not None:
            self._metadata_stream.synchronize()
        super().shutdown()

    def _require_batch(self, batch):
        self._check_ready()
        if self._batches.get(batch.bank_id) != batch:
            raise DraftSessionError("stale Draft Bank/batch identity")

    def _active_batch(self):
        self._check_ready()
        if self._compute_bank is None:
            raise DraftSessionError("Draft compute requires an activated Bank")
        return self._batches[self._compute_bank]
