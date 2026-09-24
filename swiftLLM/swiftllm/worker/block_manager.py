from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Optional

import torch

from .kernels.block_mgmt import (
    set_block_table_and_num_seq_alloc_blocks,
    unset_block_table_and_num_seq_alloc_blocks,
    gather_allocated_blocks_and_unset,
)


class KVBankRole(str, Enum):
    ACTIVE = "ACTIVE"
    STANDBY = "STANDBY"
    FREE = "FREE"
    PREPARED = "PREPARED"


@dataclass(frozen=True)
class GPUBankLocation:
    worker_id: str
    device_id: str
    model_kind: str
    bank_id: int
    bank_epoch: int
    bank_base_block: int
    request_start_block: int
    num_blocks: int
    logical_kv_len: int
    kv_version: int = 0
    request_id: Optional[int] = None

    @property
    def first_physical_block(self) -> int:
        return int(self.bank_base_block) + int(self.request_start_block)

    @property
    def end_physical_block(self) -> int:
        return self.first_physical_block + int(self.num_blocks)


@dataclass
class KVBankDescriptor:
    bank_id: int
    base_block: int
    num_blocks: int
    alloc_ptr: int = 0
    epoch: int = 0
    role: KVBankRole = KVBankRole.FREE
    batch_id: Optional[str] = None
    request_ranges: Dict[int, GPUBankLocation] = field(default_factory=dict)
    ready_event: Any = None

    @property
    def remaining_blocks(self) -> int:
        return int(self.num_blocks) - int(self.alloc_ptr)

    def snapshot(self) -> "KVBankDescriptor":
        return KVBankDescriptor(
            bank_id=int(self.bank_id),
            base_block=int(self.base_block),
            num_blocks=int(self.num_blocks),
            alloc_ptr=int(self.alloc_ptr),
            epoch=int(self.epoch),
            role=KVBankRole(self.role),
            batch_id=self.batch_id,
            request_ranges=dict(self.request_ranges),
            ready_event=self.ready_event,
        )


class BlockManager:
    """
    BlockManager - Manage the block table and free blocks on CPU / GPU.

    The default path is the original bitmap allocator. When double-bank mode is
    explicitly enabled, GPU KV blocks are allocated from active/standby banks
    using a bump pointer and contiguous physical block ranges. The attention
    kernel still consumes the same block_table; only the fill strategy changes.
    """

    def __init__(
        self,
        device_name: str,
        num_blocks: int,
        max_seqs_in_block_table: int,
        max_blocks_per_seq: int,
        block_size: int,
        *,
        enable_double_bank: bool = False,
        worker_id: str = "",
        device_id: str = "cuda:0",
        model_kind: str = "target",
    ):
        self.device_name = device_name
        self.num_free_blocks = num_blocks
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.double_bank_enabled = bool(enable_double_bank)
        self.worker_id = str(worker_id or device_name)
        self.device_id = str(device_id)
        self.model_kind = str(model_kind)
        self.nonzero_allocate_calls = 0

        # seq_id |-> number of blocks allocated for this sequence
        self.num_seq_allocated_blocks = torch.zeros(
            (max_seqs_in_block_table,),
            dtype=torch.int32,
            device="cuda"
        )
        # (seq_id, block_index) |-> block_id
        self.block_table = torch.empty(
            (max_seqs_in_block_table, max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        # block_id |-> whether this block is free or not. Kept for legacy path
        # and diagnostics; double-bank allocation does not scan it.
        self.is_block_free = torch.ones(
            (num_blocks,),
            dtype=torch.bool,
            device="cuda"
        )
        self._banks: Dict[int, KVBankDescriptor] = {}
        self._active_bank_id = 0
        self._standby_bank_id = 1
        if self.double_bank_enabled:
            self._init_double_banks()

    def _init_double_banks(self) -> None:
        bank_size = int(self.num_blocks) // 2
        if bank_size <= 0:
            raise RuntimeError("SwiftLLM double bank requires at least two KV blocks")
        usable_blocks = bank_size * 2
        if usable_blocks != int(self.num_blocks):
            # Remainder blocks are intentionally left unused in bank mode so that
            # both banks have equal capacity and bank_id -> base range is stable.
            self.num_free_blocks = usable_blocks
        self._banks = {
            0: KVBankDescriptor(
                bank_id=0,
                base_block=0,
                num_blocks=bank_size,
                role=KVBankRole.ACTIVE,
            ),
            1: KVBankDescriptor(
                bank_id=1,
                base_block=bank_size,
                num_blocks=bank_size,
                role=KVBankRole.STANDBY,
            ),
        }
        self._active_bank_id = 0
        self._standby_bank_id = 1

    @property
    def active_bank_id(self) -> int:
        return int(self._active_bank_id)

    @property
    def standby_bank_id(self) -> int:
        return int(self._standby_bank_id)

    def _allocate_blocks(self, num_blocks: int) -> torch.Tensor:
        """
        Allocate the requested number of blocks, update relevant status, and
        return the block IDs. This is the legacy bitmap allocator.
        """
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(f"No enough free blocks available on {self.device_name} ({self.num_blocks} in total, {self.num_free_blocks} free, {num_blocks} requested)")
        self.nonzero_allocate_calls += 1
        selected_blocks = torch.nonzero(self.is_block_free)[:num_blocks].view(-1)
        self.num_free_blocks -= num_blocks
        self.is_block_free[selected_blocks] = False
        return selected_blocks

    def _free_blocks(self, block_ids: torch.Tensor):
        """
        Free the specified blocks, and update relevant status.
        """
        self.num_free_blocks += len(block_ids)
        self.is_block_free[block_ids] = True

    @torch.inference_mode()
    def reserve_in_bank(
        self,
        bank_id: int,
        request_id: int,
        num_blocks: int,
        *,
        logical_kv_len: Optional[int] = None,
        kv_version: int = 0,
        batch_id: Optional[str] = None,
    ) -> GPUBankLocation:
        if not self.double_bank_enabled:
            raise RuntimeError("reserve_in_bank requires STARSD_ENABLE_SWIFTLLM_DOUBLE_BANK=1")
        bank = self._require_bank(bank_id)
        request_id = int(request_id)
        num_blocks = int(num_blocks)
        logical_kv_len = int(logical_kv_len if logical_kv_len is not None else num_blocks * self.block_size)
        kv_version = int(kv_version)
        if request_id < 0 or request_id >= int(self.num_seq_allocated_blocks.shape[0]):
            raise IndexError("request_id is out of block table row range")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if num_blocks > int(self.block_table.shape[1]):
            raise ValueError("num_blocks exceeds block table row capacity")
        if logical_kv_len < 0:
            raise ValueError("logical_kv_len must be non-negative")
        if kv_version < 0:
            raise ValueError("kv_version must be non-negative")
        if request_id in bank.request_ranges:
            raise RuntimeError(f"request_id {request_id} already has a range in bank {bank_id}")
        if num_blocks > bank.remaining_blocks:
            raise RuntimeError(
                f"Bank {bank_id} capacity exceeded: remaining={bank.remaining_blocks}, requested={num_blocks}"
            )
        start = int(bank.alloc_ptr)
        bank.alloc_ptr += num_blocks
        if batch_id is not None:
            bank.batch_id = str(batch_id)
        location = GPUBankLocation(
            worker_id=self.worker_id,
            device_id=self.device_id,
            model_kind=self.model_kind,
            bank_id=int(bank.bank_id),
            bank_epoch=int(bank.epoch),
            bank_base_block=int(bank.base_block),
            request_start_block=start,
            num_blocks=num_blocks,
            logical_kv_len=logical_kv_len,
            kv_version=kv_version,
            request_id=request_id,
        )
        bank.request_ranges[request_id] = location
        self._fill_block_table_for_location(request_id, location)
        return location

    @torch.inference_mode()
    def reserve_in_bank_batch_atomic(
        self,
        bank_id: int,
        requests: list[tuple[int, int, Optional[int], int, Optional[str]]],
        *,
        reset_bank: bool = False,
    ) -> list[GPUBankLocation]:
        """Reserve a contiguous batch in one bank or restore all touched state.

        Each request tuple is `(request_id, num_blocks, logical_kv_len,
        kv_version, batch_id)`. This method intentionally stays narrow for
        StarSD integration: it owns the block-table and bank descriptor rollback,
        while row ownership remains with SwiftLLM's RequestIdManager.
        """
        if not self.double_bank_enabled:
            raise RuntimeError("reserve_in_bank_batch_atomic requires double-bank mode")
        bank = self._require_bank(bank_id)
        snapshot = bank.snapshot()
        normalized: list[tuple[int, int, Optional[int], int, Optional[str]]] = []
        seen: set[int] = set()
        total_blocks = 0
        for request_id, num_blocks, logical_kv_len, kv_version, batch_id in requests:
            request_id = int(request_id)
            num_blocks = int(num_blocks)
            logical_kv_len = None if logical_kv_len is None else int(logical_kv_len)
            kv_version = int(kv_version)
            if request_id < 0 or request_id >= int(self.num_seq_allocated_blocks.shape[0]):
                raise IndexError("request_id is out of block table row range")
            if num_blocks < 0:
                raise ValueError("num_blocks must be non-negative")
            if num_blocks > int(self.block_table.shape[1]):
                raise ValueError("num_blocks exceeds block table row capacity")
            if logical_kv_len is not None and logical_kv_len < 0:
                raise ValueError("logical_kv_len must be non-negative")
            if kv_version < 0:
                raise ValueError("kv_version must be non-negative")
            if request_id in seen or request_id in bank.request_ranges:
                raise RuntimeError(f"request_id {request_id} already has a range in bank {bank_id}")
            seen.add(request_id)
            total_blocks += num_blocks
            normalized.append((request_id, num_blocks, logical_kv_len, kv_version, None if batch_id is None else str(batch_id)))
        remaining_blocks = int(bank.num_blocks) if reset_bank else bank.remaining_blocks
        if total_blocks > remaining_blocks:
            raise RuntimeError(
                f"Bank {bank_id} capacity exceeded: remaining={remaining_blocks}, requested={total_blocks}"
            )
        rows = [item[0] for item in normalized]
        old_counts = self.num_seq_allocated_blocks[rows].clone() if rows else None
        old_table = self.block_table[rows, :].clone() if rows else None
        out: list[GPUBankLocation] = []
        try:
            if reset_bank:
                self.reset_bank(int(bank_id))
            for request_id, num_blocks, logical_kv_len, kv_version, batch_id in normalized:
                out.append(
                    self.reserve_in_bank(
                        int(bank_id),
                        request_id,
                        num_blocks,
                        logical_kv_len=logical_kv_len,
                        kv_version=kv_version,
                        batch_id=batch_id,
                    )
                )
        except Exception:
            bank.alloc_ptr = int(snapshot.alloc_ptr)
            bank.epoch = int(snapshot.epoch)
            bank.role = snapshot.role
            bank.batch_id = snapshot.batch_id
            bank.request_ranges = dict(snapshot.request_ranges)
            bank.ready_event = snapshot.ready_event
            if rows:
                row_tensor = torch.tensor(rows, dtype=torch.long, device=self.num_seq_allocated_blocks.device)
                self.num_seq_allocated_blocks[row_tensor] = old_counts
                self.block_table[row_tensor, :] = old_table
            raise
        return out

    def reset_bank(self, bank_id: int) -> KVBankDescriptor:
        if not self.double_bank_enabled:
            raise RuntimeError("reset_bank requires double-bank mode")
        bank = self._require_bank(bank_id)
        bank.alloc_ptr = 0
        bank.epoch += 1
        bank.batch_id = None
        bank.request_ranges = {}
        bank.ready_event = None
        if bank.role == KVBankRole.PREPARED:
            bank.role = KVBankRole.STANDBY
        return bank.snapshot()

    def discard_empty_prepared_bank(self, bank_id: int, bank_epoch: int, batch_id: str) -> None:
        """After exact DMA/row retirement, permit a new prepare without epoch drift."""
        bank = self._require_bank(bank_id)
        if (bank_id != self.standby_bank_id or bank.role != KVBankRole.PREPARED
                or bank.epoch != bank_epoch or str(bank.batch_id) != str(batch_id)
                or bank.request_ranges):
            raise RuntimeError("discard requires exact empty prepared standby Bank")
        bank.role = KVBankRole.STANDBY
        bank.alloc_ptr = 0
        bank.batch_id = None
        bank.ready_event = None

    def swap_active_standby(self) -> tuple[KVBankDescriptor, KVBankDescriptor]:
        if not self.double_bank_enabled:
            raise RuntimeError("swap_active_standby requires double-bank mode")
        old_active = self._require_bank(self._active_bank_id)
        old_standby = self._require_bank(self._standby_bank_id)
        old_active.role = KVBankRole.STANDBY
        old_standby.role = KVBankRole.ACTIVE
        self._active_bank_id, self._standby_bank_id = self._standby_bank_id, self._active_bank_id
        return self.get_bank_descriptor(self._active_bank_id), self.get_bank_descriptor(self._standby_bank_id)

    def get_bank_descriptor(self, bank_id: int) -> KVBankDescriptor:
        return self._require_bank(bank_id).snapshot()

    def validate_bank_location(self, location: GPUBankLocation) -> None:
        if not self.double_bank_enabled:
            raise RuntimeError("validate_bank_location requires double-bank mode")
        bank = self._require_bank(int(location.bank_id))
        if int(location.bank_epoch) != int(bank.epoch):
            raise RuntimeError(
                f"stale bank epoch for bank {location.bank_id}: location={location.bank_epoch}, current={bank.epoch}"
            )
        if int(location.bank_base_block) != int(bank.base_block):
            raise RuntimeError("bank base block mismatch")
        if int(location.request_start_block) < 0 or int(location.num_blocks) < 0:
            raise RuntimeError("invalid bank location range")
        if int(location.request_start_block) + int(location.num_blocks) > int(bank.num_blocks):
            raise RuntimeError("bank location exceeds bank range")

    def mark_bank_prepared(self, bank_id: int, *, batch_id: Optional[str] = None, ready_event: Any = None) -> KVBankDescriptor:
        bank = self._require_bank(bank_id)
        bank.role = KVBankRole.PREPARED
        if batch_id is not None:
            bank.batch_id = str(batch_id)
        bank.ready_event = ready_event
        return bank.snapshot()

    def allocate_blocks_for_seqs(self, seq_ids: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
        """
        Allocate blocks for sequences, making sure that seq #i has at least
        ceil(target_lengths[i] / block_size) blocks allocated.

        Return new blocks allocated for the sequences. (useful for swapping)
        """
        if self.double_bank_enabled:
            return self._allocate_blocks_for_seqs_from_active_bank(
                seq_ids.device,
                self._normalize_sequence_length_batch(seq_ids, target_lens, operation="bank allocation"),
            )

        target_num_blocks = (target_lens + (self.block_size-1)) // self.block_size
        assert (self.num_seq_allocated_blocks[seq_ids] <= target_num_blocks).all(),             f"""(On {self.device_name}) Logic error: Some sequences have more blocks already allocated than needed.
                seq_ids: {seq_ids}, target_lens: {target_lens}, target_num_blocks: {target_num_blocks},
                self.num_seq_allocated_blocks[seq_ids]: {self.num_seq_allocated_blocks[seq_ids]}"""
        block_needed = target_num_blocks - self.num_seq_allocated_blocks[seq_ids]
        new_blocks = self._allocate_blocks(torch.sum(block_needed).item())

        set_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, new_blocks, seq_ids, block_needed)

        return new_blocks

    def _allocate_blocks_for_seqs_from_active_bank(
        self,
        result_device: torch.device,
        items: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        bank = self._require_bank(self._active_bank_id)

        new_reservations: list[tuple[int, int, int]] = []
        existing_updates: list[tuple[int, int, int, GPUBankLocation]] = []
        total_new_capacity = 0
        for seq_id, target_blocks, target_len in items:
            current_blocks = int(self.num_seq_allocated_blocks[seq_id].item())
            if target_blocks <= current_blocks:
                continue
            if seq_id not in bank.request_ranges:
                new_reservations.append((seq_id, target_blocks, target_len))
                total_new_capacity += target_blocks
            else:
                location = bank.request_ranges[seq_id]
                self.validate_bank_location(location)
                if target_blocks > int(location.num_blocks):
                    raise RuntimeError(
                        f"Bank reservation capacity exceeded for seq {seq_id}: "
                        f"capacity={location.num_blocks}, requested={target_blocks}"
                    )
                existing_updates.append((seq_id, target_blocks, target_len, location))
        if total_new_capacity > bank.remaining_blocks:
            raise RuntimeError(
                f"Bank {bank.bank_id} capacity exceeded: remaining={bank.remaining_blocks}, requested={total_new_capacity}"
            )

        rows = [seq_id for seq_id, _, _ in items]
        old_bank = bank.snapshot()
        old_counts = self.num_seq_allocated_blocks[rows].clone() if rows else None
        old_table = self.block_table[rows, :].clone() if rows else None
        new_block_tensors = []
        try:
            for seq_id, target_blocks, target_len in new_reservations:
                location = self.reserve_in_bank(
                    bank.bank_id,
                    seq_id,
                    target_blocks,
                    logical_kv_len=target_len,
                )
                if target_blocks:
                    new_block_tensors.append(torch.arange(location.first_physical_block, location.first_physical_block + target_blocks, dtype=torch.int32, device=result_device))
            for seq_id, target_blocks, target_len, old_location in existing_updates:
                current_blocks = int(self.num_seq_allocated_blocks[seq_id].item())
                location = replace(
                    old_location,
                    logical_kv_len=target_len,
                )
                bank.request_ranges[seq_id] = location
                self._set_valid_blocks_for_location(seq_id, location, target_blocks)
                if target_blocks > current_blocks:
                    new_start = int(location.first_physical_block) + current_blocks
                    new_block_tensors.append(torch.arange(new_start, new_start + (target_blocks - current_blocks), dtype=torch.int32, device=result_device))
        except Exception:
            bank.alloc_ptr = int(old_bank.alloc_ptr)
            bank.epoch = int(old_bank.epoch)
            bank.role = old_bank.role
            bank.batch_id = old_bank.batch_id
            bank.request_ranges = dict(old_bank.request_ranges)
            bank.ready_event = old_bank.ready_event
            if rows:
                row_tensor = torch.tensor(rows, dtype=torch.long, device=self.num_seq_allocated_blocks.device)
                self.num_seq_allocated_blocks[row_tensor] = old_counts
                self.block_table[row_tensor, :] = old_table
            raise
        if not new_block_tensors:
            return torch.empty((0,), dtype=torch.int32, device=result_device)
        return torch.cat(new_block_tensors)

    def crop_blocks_for_seqs(self, seq_ids: torch.Tensor, target_lens: torch.Tensor):
        """
        Crop sequences to the blocks needed for target_lens.
        """
        if self.double_bank_enabled:
            plans = []
            for seq_id, target_blocks, target_len in self._normalize_sequence_length_batch(seq_ids, target_lens, operation="crop"):
                cur_num_blocks = int(self.num_seq_allocated_blocks[seq_id].item())
                if target_blocks > cur_num_blocks:
                    raise RuntimeError(
                        f"Cannot crop seq {seq_id} on {self.device_name} from "
                        f"{cur_num_blocks} blocks up to {target_blocks} blocks"
                    )
                bank = self._bank_for_request(seq_id)
                location = None if bank is None else bank.request_ranges[seq_id]
                if location is not None:
                    self.validate_bank_location(location)
                    if target_blocks > int(location.num_blocks):
                        raise RuntimeError("crop target exceeds reservation capacity")
                plans.append((seq_id, target_blocks, target_len, bank, location))
            for seq_id, target_blocks, logical_len, bank, location in plans:
                self.num_seq_allocated_blocks[seq_id] = target_blocks
                if bank is not None and location is not None:
                    bank.request_ranges[seq_id] = replace(
                        location,
                        logical_kv_len=int(logical_len),
                        kv_version=int(location.kv_version) + 1,
                    )
            return
        target_num_blocks = (target_lens + (self.block_size-1)) // self.block_size
        for idx, (seq_id_tensor, target_num_blocks_tensor) in enumerate(zip(seq_ids, target_num_blocks)):
            seq_id = int(seq_id_tensor.item())
            target_num_blocks_item = int(target_num_blocks_tensor.item())
            cur_num_blocks = int(self.num_seq_allocated_blocks[seq_id].item())
            if target_num_blocks_item > cur_num_blocks:
                raise RuntimeError(
                    f"Cannot crop seq {seq_id} on {self.device_name} from "
                    f"{cur_num_blocks} blocks up to {target_num_blocks_item} blocks"
                )
            if target_num_blocks_item == cur_num_blocks:
                continue
            freed_blocks = self.block_table[seq_id, target_num_blocks_item:cur_num_blocks]
            self._free_blocks(freed_blocks)
            self.num_seq_allocated_blocks[seq_id] = target_num_blocks_item

    def set_bank_location_kv_version(self, request_id: int, *, bank_id: int, bank_epoch: int, kv_version: int) -> GPUBankLocation:
        if not self.double_bank_enabled:
            raise RuntimeError("set_bank_location_kv_version requires double-bank mode")
        bank = self._require_bank(int(bank_id))
        request_id = int(request_id)
        kv_version = int(kv_version)
        if kv_version < 0:
            raise ValueError("kv_version must be non-negative")
        if int(bank.epoch) != int(bank_epoch):
            raise RuntimeError("bank epoch mismatch while setting kv_version")
        location = bank.request_ranges.get(request_id)
        if location is None:
            raise RuntimeError("request has no bank location")
        updated = replace(location, kv_version=kv_version)
        bank.request_ranges[request_id] = updated
        return updated

    def _normalize_sequence_length_batch(
        self,
        seq_ids: torch.Tensor,
        target_lens: torch.Tensor,
        *,
        operation: str,
    ) -> list[tuple[int, int, int]]:
        if seq_ids.ndim != 1 or target_lens.ndim != 1:
            raise ValueError(f"{operation} seq_ids and target_lens must be one-dimensional tensors")
        if seq_ids.numel() != target_lens.numel():
            raise ValueError(f"{operation} seq_ids and target_lens must have the same length")
        if seq_ids.dtype == torch.bool or target_lens.dtype == torch.bool or seq_ids.is_floating_point() or target_lens.is_floating_point():
            raise TypeError(f"{operation} seq_ids and target_lens must be integer tensors")
        seq_ids_cpu = [int(item) for item in seq_ids.detach().cpu().tolist()]
        target_lens_cpu = [int(item) for item in target_lens.detach().cpu().tolist()]
        if len(set(seq_ids_cpu)) != len(seq_ids_cpu):
            raise RuntimeError(f"duplicate sequence id in {operation} batch")
        max_row = int(self.num_seq_allocated_blocks.shape[0])
        items: list[tuple[int, int, int]] = []
        for seq_id, target_len in zip(seq_ids_cpu, target_lens_cpu, strict=True):
            if seq_id < 0 or seq_id >= max_row:
                raise IndexError("sequence id is out of block table range")
            if target_len < 0:
                raise ValueError("target logical length must be non-negative")
            target_blocks = (target_len + (self.block_size - 1)) // self.block_size
            items.append((seq_id, target_blocks, target_len))
        return items

    def free_blocks_for_seqs(self, seq_ids: torch.Tensor):
        """
        Free blocks for sequences.
        """
        if self.double_bank_enabled:
            for seq_id_tensor in seq_ids:
                seq_id = int(seq_id_tensor.item())
                self.num_seq_allocated_blocks[seq_id] = 0
                bank = self._bank_for_request(seq_id)
                if bank is not None:
                    bank.request_ranges.pop(seq_id, None)
            return
        self.num_free_blocks += torch.sum(self.num_seq_allocated_blocks[seq_ids]).item()
        unset_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, seq_ids, self.is_block_free)

    @torch.inference_mode()
    def release_bank_ranges_exact_batch(self, releases: list[tuple[int, int, int, int, int, Optional[str]]], *, apply: bool = True) -> None:
        """Release exact StarSD bank ranges without guessing by row.

        Each item is `(bank_id, bank_epoch, row, start_block, capacity_blocks,
        batch_id)`. The whole batch is prevalidated before any descriptor or
        block-table mutation.
        """
        if not self.double_bank_enabled:
            raise RuntimeError("release_bank_ranges_exact_batch requires double-bank mode")
        normalized: list[tuple[int, int, int, int, int, Optional[str]]] = []
        seen: set[tuple[int, int]] = set()
        max_row = int(self.num_seq_allocated_blocks.shape[0])
        for bank_id, bank_epoch, row, start_block, capacity_blocks, batch_id in releases:
            bank_id = int(bank_id)
            bank_epoch = int(bank_epoch)
            row = int(row)
            start_block = int(start_block)
            capacity_blocks = int(capacity_blocks)
            batch_id = None if batch_id is None else str(batch_id)
            if row < 0 or row >= max_row:
                raise IndexError("release row is out of block table range")
            if capacity_blocks < 0:
                raise ValueError("release capacity must be non-negative")
            if (bank_id, row) in seen:
                raise RuntimeError("duplicate exact bank release row")
            seen.add((bank_id, row))
            bank = self._require_bank(bank_id)
            location = bank.request_ranges.get(row)
            if location is None:
                raise RuntimeError("exact bank release range is not live")
            if int(bank.epoch) != bank_epoch or int(location.bank_epoch) != bank_epoch:
                raise RuntimeError("exact bank release epoch mismatch")
            if int(location.request_start_block) != start_block:
                raise RuntimeError("exact bank release start mismatch")
            if int(location.num_blocks) != capacity_blocks:
                raise RuntimeError("exact bank release capacity mismatch")
            if batch_id is not None and str(bank.batch_id) != batch_id:
                raise RuntimeError("exact bank release batch_id mismatch")
            normalized.append((bank_id, bank_epoch, row, start_block, capacity_blocks, batch_id))
        if not apply:
            return
        for bank_id, _bank_epoch, row, _start_block, _capacity_blocks, _batch_id in normalized:
            bank = self._require_bank(bank_id)
            location = bank.request_ranges.pop(row)
            current_blocks = int(self.num_seq_allocated_blocks[row].item())
            if (
                current_blocks <= int(location.num_blocks)
                and current_blocks > 0
                and int(self.block_table[row, 0].item()) == int(location.first_physical_block)
            ):
                self.num_seq_allocated_blocks[row] = 0

    def gather_allocated_blocks_and_free(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Gather the block IDs allocated for the specified sequences and mark them as free.
        """
        if self.double_bank_enabled:
            tensors = []
            for seq_id_tensor in seq_ids:
                seq_id = int(seq_id_tensor.item())
                tensors.append(self.get_allocated_block_ids(seq_id).to(device=seq_ids.device))
            self.free_blocks_for_seqs(seq_ids)
            if not tensors:
                return torch.empty((0,), dtype=torch.int32, device=seq_ids.device)
            return torch.cat(tensors)
        gathered_block_ids = gather_allocated_blocks_and_unset(self.num_seq_allocated_blocks, self.block_table, seq_ids, self.is_block_free)
        self.num_free_blocks += len(gathered_block_ids)
        return gathered_block_ids

    def get_allocated_block_ids(self, seq_id: int) -> torch.Tensor:
        """
        Return a cloned tensor of block IDs currently allocated for one sequence.
        """
        if seq_id < 0 or seq_id >= self.num_seq_allocated_blocks.shape[0]:
            raise IndexError(f"seq_id {seq_id} is out of block table range")
        num_blocks = int(self.num_seq_allocated_blocks[seq_id].item())
        return self.block_table[seq_id, :num_blocks].clone()

    def get_num_allocated_blocks(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the number of blocks allocated for the specified sequences.
        """
        return self.num_seq_allocated_blocks[seq_ids]

    def _fill_block_table_for_location(self, request_id: int, location: GPUBankLocation) -> None:
        self.validate_bank_location(location)
        self._set_valid_blocks_for_location(request_id, location, int(location.num_blocks))

    def _set_valid_blocks_for_location(self, request_id: int, location: GPUBankLocation, valid_blocks: int) -> None:
        self.validate_bank_location(location)
        nblocks = int(valid_blocks)
        if nblocks == 0:
            self.num_seq_allocated_blocks[int(request_id)] = 0
            return
        if nblocks > int(location.num_blocks):
            raise RuntimeError("valid blocks cannot exceed reservation capacity")
        if nblocks > self.block_table.shape[1]:
            raise RuntimeError(f"request needs {nblocks} blocks, block table row only has {self.block_table.shape[1]}")
        physical_start = int(location.first_physical_block)
        self.block_table[int(request_id), :nblocks] = torch.arange(
            physical_start,
            physical_start + nblocks,
            dtype=torch.int32,
            device=self.block_table.device,
        )
        self.num_seq_allocated_blocks[int(request_id)] = nblocks

    def _require_bank(self, bank_id: int) -> KVBankDescriptor:
        bank_id = int(bank_id)
        if bank_id not in self._banks:
            raise KeyError(f"unknown bank_id {bank_id}")
        return self._banks[bank_id]

    def _bank_for_request(self, request_id: int) -> Optional[KVBankDescriptor]:
        request_id = int(request_id)
        for bank in self._banks.values():
            if request_id in bank.request_ranges:
                return bank
        return None
