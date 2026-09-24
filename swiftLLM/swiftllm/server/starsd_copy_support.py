"""Narrow process-local copy integration for a direct Target facade.

The caller is the single metadata owner. DMA threads receive tensors/pointers,
not this object. Bank rows must be disjoint from an in-flight compute batch;
there is no global ModelRunner lock or per-copy executor round trip.
"""

from __future__ import annotations

from contextlib import contextmanager


async def record_compute_ready(worker):
    """Record on the model executor's stream, never the listener's stream."""
    cache = getattr(getattr(worker, "model", None), "k_cache", None)
    if not getattr(cache, "is_cuda", False):
        return None

    def record():
        import torch
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=cache.device))
        return event

    return await worker._run_on_model_async(record)


class SwiftLLMCopySupport:
    def __init__(self, facade) -> None:
        self.facade = facade
        self._reserved: dict[tuple[str, str], int] = {}
        self._metadata_stream = None

    def caches(self):
        model = self.facade.worker.model
        return model.k_cache, model.v_cache

    @contextmanager
    def metadata_context(self):
        import torch

        cache, _ = self.caches()
        torch.cuda.set_device(cache.device)
        if self._metadata_stream is None:
            self._metadata_stream = torch.cuda.Stream(device=cache.device)
        with torch.inference_mode(), torch.cuda.stream(self._metadata_stream):
            yield

    def metadata_event(self):
        import torch

        if self._metadata_stream is None:
            return None
        event = torch.cuda.Event()
        event.record(self._metadata_stream)
        return event

    def reserve_rows(self, keys: tuple[tuple[str, str], ...]) -> tuple[int, ...]:
        manager = self.facade.worker.request_id_manager
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate import session key")
        if any(key in self._reserved or key in self.facade.worker.sessions for key in keys):
            raise RuntimeError("import would overwrite a live Target session")
        if len(manager.available_ids) < len(keys):
            raise RuntimeError("Target row capacity exhausted")
        rows = tuple(manager.get_id() for _ in keys)
        self._reserved.update(zip(keys, rows))
        return rows

    def release_drained(self, ranges: tuple[tuple[int, int, int, int, int, str], ...],
                        keys: tuple[tuple[str, str], ...]) -> None:
        """Release exact ranges and their rows after DMA completion.

This also retires ordinary planned-prefill sessions. Their cached bank fence
must not be used later to free a range that has already moved to another bank.
"""
        if len(ranges) != len(keys) or len(keys) != len(set(keys)):
            raise ValueError("drained ranges/session keys must match one-to-one")
        worker = self.facade.worker
        manager = self.facade._require_block_manager()
        for item, key in zip(ranges, keys):
            row = item[2]
            session = worker.sessions.get(key)
            actual = int(session.request.request_id) if session is not None else self._reserved.get(key)
            if actual != row:
                raise RuntimeError("drained session row mismatch")
            if row in worker.request_id_manager.available_ids:
                raise RuntimeError("drained row was already released")
        manager.release_bank_ranges_exact_batch(list(ranges), apply=False)
        manager.release_bank_ranges_exact_batch(list(ranges))
        # GPU metadata clears must complete before these rows can be reallocated.
        if self._metadata_stream is not None:
            self._metadata_stream.synchronize()
        for item, key in zip(ranges, keys):
            worker.sessions.pop(key, None)
            self._reserved.pop(key, None)
            worker.request_id_manager.free_id(item[2])

    def discard_empty_prepared(self, bank_id: int, bank_epoch: int, batch_seq: int) -> None:
        self.facade._require_block_manager().discard_empty_prepared_bank(bank_id, bank_epoch, str(batch_seq))

    def reset_drained_prefill(self, bank_id: int, bank_epoch: int) -> None:
        """Reuse a fully drained active Bank for a new planned prefill.

        The caller owns the compute lane and has retired its drain lease.
        Exact releases must already have removed every canonical range.
        """
        manager = self.facade._require_block_manager()
        bank = manager.get_bank_descriptor(bank_id)
        if int(bank.epoch) != bank_epoch or bank.request_ranges:
            raise RuntimeError("prefill reset requires an empty, epoch-matched Bank")
        manager.reset_bank(bank_id)

    def close(self) -> None:
        """Cold-path whole-worker retirement after compute and DMA have joined."""
        manager = self.facade._require_block_manager()
        with self.metadata_context():
            ranges = []
            for bank in self.facade.describe_banks():
                descriptor = manager.get_bank_descriptor(bank.bank_id)
                ranges.extend((bank.bank_id, bank.bank_epoch, int(row), int(location.request_start_block),
                               int(location.num_blocks), descriptor.batch_id)
                              for row, location in descriptor.request_ranges.items())
            manager.release_bank_ranges_exact_batch(ranges)
            self._metadata_stream.synchronize()
        worker = self.facade.worker
        worker.sessions.clear()
        ids = worker.request_id_manager
        ids.available_ids = list(reversed(range(ids.max_id)))
        self._reserved.clear()
        self._metadata_stream = None
