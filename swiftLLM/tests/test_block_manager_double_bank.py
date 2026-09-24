from __future__ import annotations

import unittest
from types import MethodType, SimpleNamespace

import torch

from swiftllm.server.scheduler import RequestIdManager
from swiftllm.server.starsd_target_facade import (
    StarsDReleaseRequest,
    StarsDReserveRequest,
    SwiftLLMStarsDTargetFacade,
    _batch_id_from_reservation_id,
)
from swiftllm.server.target_worker import SwiftLLMTargetWorker, _SessionBankFence, _TargetSession
from swiftllm.worker.block_manager import BlockManager, KVBankRole


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for SwiftLLM BlockManager tests")
class BlockManagerDoubleBankTests(unittest.TestCase):
    def make_manager(self, *, num_blocks: int = 32, enable_double_bank: bool = True) -> BlockManager:
        return BlockManager(
            "GPU",
            num_blocks,
            max_seqs_in_block_table=16,
            max_blocks_per_seq=64,
            block_size=16,
            enable_double_bank=enable_double_bank,
            worker_id="target_w0",
            device_id="cuda:0",
            model_kind="target",
        )

    def test_bank_ranges_are_equal_non_overlapping_with_remainder_unused(self) -> None:
        mgr = self.make_manager(num_blocks=17)
        bank0 = mgr.get_bank_descriptor(0)
        bank1 = mgr.get_bank_descriptor(1)

        self.assertEqual(bank0.base_block, 0)
        self.assertEqual(bank0.num_blocks, 8)
        self.assertEqual(bank1.base_block, 8)
        self.assertEqual(bank1.num_blocks, 8)
        self.assertEqual(bank0.base_block + bank0.num_blocks, bank1.base_block)
        self.assertEqual(bank0.role, KVBankRole.ACTIVE)
        self.assertEqual(bank1.role, KVBankRole.STANDBY)

    def test_bump_allocation_is_contiguous_and_capacity_checked(self) -> None:
        mgr = self.make_manager(num_blocks=16)
        loc_a = mgr.reserve_in_bank(0, request_id=3, num_blocks=3, logical_kv_len=48)
        loc_b = mgr.reserve_in_bank(0, request_id=4, num_blocks=5, logical_kv_len=80)

        self.assertEqual(loc_a.first_physical_block, 0)
        self.assertEqual(loc_a.end_physical_block, 3)
        self.assertEqual(loc_b.first_physical_block, 3)
        self.assertEqual(loc_b.end_physical_block, 8)
        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            mgr.reserve_in_bank(0, request_id=5, num_blocks=1)
        with self.assertRaisesRegex(RuntimeError, "already has a range"):
            mgr.reserve_in_bank(0, request_id=3, num_blocks=1)

    def test_block_table_fill_uses_contiguous_physical_ids(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        loc = mgr.reserve_in_bank(0, request_id=2, num_blocks=6, logical_kv_len=96)
        torch.cuda.synchronize()

        block_ids = mgr.get_allocated_block_ids(2).detach().cpu().tolist()
        self.assertEqual(block_ids, list(range(loc.first_physical_block, loc.end_physical_block)))
        self.assertEqual(int(mgr.num_seq_allocated_blocks[2].item()), 6)

    def test_batch_atomic_reserve_rolls_back_descriptor_and_tensors(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        before = mgr.get_bank_descriptor(1)
        rows = [1, 2]
        mgr.block_table[rows, :] = -7
        old_table = mgr.block_table[rows, :].clone()
        old_counts = mgr.num_seq_allocated_blocks[rows].clone()
        original = mgr.reserve_in_bank
        calls = {"count": 0}

        def flaky_reserve(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("injected reserve failure")
            return original(*args, **kwargs)

        mgr.reserve_in_bank = flaky_reserve
        with self.assertRaisesRegex(RuntimeError, "injected"):
            mgr.reserve_in_bank_batch_atomic(
                1,
                [(1, 2, 0, 0, "batch"), (2, 2, 0, 0, "batch")],
                reset_bank=True,
            )
        after = mgr.get_bank_descriptor(1)
        self.assertEqual(after, before)
        self.assertEqual(after.request_ranges, {})
        self.assertTrue(torch.equal(mgr.block_table[rows, :], old_table))
        self.assertTrue(torch.equal(mgr.num_seq_allocated_blocks[rows], old_counts))

    def test_reset_is_constant_shape_and_epoch_increment(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        loc = mgr.reserve_in_bank(0, request_id=2, num_blocks=6)
        before = mgr.get_bank_descriptor(0)
        reset = mgr.reset_bank(0)

        self.assertEqual(before.epoch, 0)
        self.assertEqual(reset.epoch, 1)
        self.assertEqual(reset.alloc_ptr, 0)
        self.assertEqual(reset.request_ranges, {})
        with self.assertRaisesRegex(RuntimeError, "stale bank epoch"):
            mgr.validate_bank_location(loc)

    def test_swap_active_standby_ping_pong_roles(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        active, standby = mgr.swap_active_standby()
        self.assertEqual(mgr.active_bank_id, 1)
        self.assertEqual(mgr.standby_bank_id, 0)
        self.assertEqual(active.role, KVBankRole.ACTIVE)
        self.assertEqual(standby.role, KVBankRole.STANDBY)

    def test_starsd_facade_uses_request_id_manager_rows_without_conflict(self) -> None:
        facade = self.make_facade(rows=4)
        ordinary_row = facade.worker.request_id_manager.get_id()
        active, standby = facade.bank_snapshot()
        reservations = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 2, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        self.assertNotEqual(reservations[0].row, ordinary_row)
        facade.worker.request_id_manager.free_id(ordinary_row)
        self.assertEqual(len(set(facade.worker.request_id_manager.available_ids)), len(facade.worker.request_id_manager.available_ids))

    def test_starsd_facade_row_reuse_and_repeated_swap_replay(self) -> None:
        facade = self.make_facade(rows=4)
        seen_rows = []
        for idx in range(8):
            active, standby = facade.bank_snapshot()
            batch_id = f"batch-{idx}"
            reservations = facade.reserve_standby_batch(
                (StarsDReserveRequest(f"req-{idx}", 1, idx, 2, batch_id),),
                expected_bank_id=standby.bank_id,
                expected_bank_epoch=standby.bank_epoch,
            )
            seen_rows.append(reservations[0].row)
            first_swap = facade.mark_prepared_and_swap(
                bank_id=reservations[0].bank_id,
                bank_epoch=reservations[0].bank_epoch,
                batch_id=batch_id,
            )
            second_swap = facade.mark_prepared_and_swap(
                bank_id=reservations[0].bank_id,
                bank_epoch=reservations[0].bank_epoch,
                batch_id=batch_id,
            )
            self.assertEqual(first_swap, second_swap)
            asyncio_result = __import__("asyncio").run(
                facade.release((StarsDReleaseRequest(f"req-{idx}", 1, idx, reservations[0].reservation_id),))
            )
            self.assertEqual(asyncio_result["status"], "ok")
            stats = facade.resource_stats()
            self.assertEqual(stats.reservation_count, 0)
            self.assertEqual(stats.allocated_range_count, 0)
            self.assertEqual(stats.free_row_count, 4)
            self.assertEqual(len(set(facade.worker.request_id_manager.available_ids)), 4)
        self.assertEqual(set(seen_rows), {0})

    def test_starsd_facade_get_id_failure_returns_all_acquired_rows(self) -> None:
        facade = self.make_facade(rows=4)
        manager = facade.worker.model.gpu_block_manager
        row_manager = facade.worker.request_id_manager
        before_ids = list(row_manager.available_ids)
        before_active, before_standby = facade.bank_snapshot()
        before_table = manager.block_table.clone()
        before_counts = manager.num_seq_allocated_blocks.clone()
        original_get_id = row_manager.get_id
        calls = {"count": 0}

        def flaky_get_id():
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("injected get_id failure")
            return original_get_id()

        row_manager.get_id = flaky_get_id
        with self.assertRaisesRegex(RuntimeError, "injected get_id failure"):
            facade.reserve_standby_batch(
                tuple(StarsDReserveRequest(f"req-{idx}", 1, idx, 1, "batch") for idx in range(4)),
                expected_bank_id=before_standby.bank_id,
                expected_bank_epoch=before_standby.bank_epoch,
            )
        self.assertEqual(row_manager.available_ids, before_ids)
        self.assertEqual(len(set(row_manager.available_ids)), len(row_manager.available_ids))
        self.assertEqual(facade.bank_snapshot(), (before_active, before_standby))
        self.assertEqual(facade._resource_rows, {})
        self.assertEqual(facade._reserve_batches, {})
        self.assertTrue(torch.equal(manager.block_table, before_table))
        self.assertTrue(torch.equal(manager.num_seq_allocated_blocks, before_counts))

    def test_starsd_facade_release_backend_failure_is_zero_side_effect_then_replay(self) -> None:
        facade = self.make_facade(rows=4, fail_release_once=True)
        manager = facade.worker.model.gpu_block_manager
        active, standby = facade.bank_snapshot()
        reservations = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 2, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        reservation = reservations[0]
        facade.mark_prepared_and_swap(bank_id=reservation.bank_id, bank_epoch=reservation.bank_epoch, batch_id="batch")
        facade.worker.sessions[(reservation.reservation_id, reservation.request_id)] = SimpleNamespace(
            bank_adapter=True,
            request=SimpleNamespace(request_id=reservation.row),
        )
        before_ids = list(facade.worker.request_id_manager.available_ids)
        before_ranges = dict(manager.get_bank_descriptor(reservation.bank_id).request_ranges)
        before_rows = dict(facade._resource_rows)
        before_batches = dict(facade._reserve_batches)
        before_released = dict(facade._released)
        request = StarsDReleaseRequest("req", 1, 0, reservation.reservation_id)

        with self.assertRaisesRegex(RuntimeError, "injected release failure"):
            __import__("asyncio").run(facade.release((request,)))
        self.assertEqual(facade.worker.request_id_manager.available_ids, before_ids)
        self.assertEqual(manager.get_bank_descriptor(reservation.bank_id).request_ranges, before_ranges)
        self.assertEqual(facade._resource_rows, before_rows)
        self.assertEqual(facade._reserve_batches, before_batches)
        self.assertEqual(facade._released, before_released)

        ack = __import__("asyncio").run(facade.release((request,)))
        self.assertEqual(ack["status"], "ok")
        self.assertEqual(facade.resource_stats().reservation_count, 0)
        self.assertEqual(facade.resource_stats().allocated_range_count, 0)
        self.assertEqual(len(facade.worker.sessions), 0)
        self.assertEqual(len(set(facade.worker.request_id_manager.available_ids)), len(facade.worker.request_id_manager.available_ids))
        self.assertEqual(len(facade.worker.request_id_manager.available_ids), 4)
        release_calls = facade.worker.release_calls
        replay = __import__("asyncio").run(facade.release((request,)))
        self.assertEqual(replay, ack)
        self.assertEqual(facade.worker.release_calls, release_calls)
        self.assertEqual(len(set(facade.worker.request_id_manager.available_ids)), 4)

    def test_starsd_facade_batch_release_uses_exact_session_keys_for_same_request_id(self) -> None:
        facade = self.make_facade(rows=4)
        active, standby = facade.bank_snapshot()
        reservations = facade.reserve_standby_batch(
            (
                StarsDReserveRequest("req", 1, 0, 2, "batch"),
                StarsDReserveRequest("req", 1, 1, 2, "batch"),
            ),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        facade.mark_prepared_and_swap(bank_id=reservations[0].bank_id, bank_epoch=reservations[0].bank_epoch, batch_id="batch")
        for reservation in reservations:
            facade.worker.sessions[(reservation.reservation_id, reservation.request_id)] = SimpleNamespace(
                bank_adapter=True,
                request=SimpleNamespace(request_id=reservation.row),
            )
        ack = __import__("asyncio").run(
            facade.release(
                tuple(
                    StarsDReleaseRequest(reservation.request_id, reservation.request_epoch, reservation.round_id, reservation.reservation_id)
                    for reservation in reservations
                )
            )
        )
        self.assertEqual(ack["backend"]["released_count"], 2)
        self.assertEqual(facade.worker.sessions, {})
        stats = facade.resource_stats()
        self.assertEqual(stats.reservation_count, 0)
        self.assertEqual(stats.allocated_range_count, 0)
        self.assertEqual(stats.free_row_count, 4)
        self.assertEqual(len(set(facade.worker.request_id_manager.available_ids)), 4)
        release_calls = facade.worker.release_calls
        replay = __import__("asyncio").run(
            facade.release(
                tuple(
                    StarsDReleaseRequest(reservation.request_id, reservation.request_epoch, reservation.round_id, reservation.reservation_id)
                    for reservation in reservations
                )
            )
        )
        self.assertEqual(facade.worker.release_calls, release_calls)
        self.assertEqual(replay["released_rows"], ack["released_rows"])
        self.assertEqual(facade.worker.sessions, {})
        self.assertEqual(facade.resource_stats().reservation_count, 0)
        self.assertEqual(len(set(facade.worker.request_id_manager.available_ids)), 4)

    def test_real_worker_release_method_unions_exact_and_legacy_filters(self) -> None:
        facade = self.make_facade(rows=4)
        worker = facade.worker
        rows = [worker.request_id_manager.get_id() for _ in range(3)]
        worker.sessions[("tag-a", "req")] = SimpleNamespace(bank_adapter=True, request=SimpleNamespace(request_id=rows[0]))
        worker.sessions[("tag-b", "req")] = SimpleNamespace(bank_adapter=True, request=SimpleNamespace(request_id=rows[1]))
        worker.sessions[("tag-c", "other")] = SimpleNamespace(bank_adapter=True, request=SimpleNamespace(request_id=rows[2]))

        released = __import__("asyncio").run(
            worker.release_bank_adapter_sessions(
                request_ids=["req"],
                client_tags={"req": "tag-b"},
                exact_session_keys=(("tag-a", "req"),),
            )
        )
        self.assertEqual(released["released_count"], 2)
        self.assertNotIn(("tag-a", "req"), worker.sessions)
        self.assertNotIn(("tag-b", "req"), worker.sessions)
        self.assertIn(("tag-c", "other"), worker.sessions)

        no_op = __import__("asyncio").run(worker.release_bank_adapter_sessions(request_ids=[]))
        self.assertEqual(no_op["released_count"], 0)
        self.assertIn(("tag-c", "other"), worker.sessions)

        final = __import__("asyncio").run(worker.release_bank_adapter_sessions())
        self.assertEqual(final["released_count"], 1)
        self.assertEqual(worker.sessions, {})
        self.assertEqual(len(worker.request_id_manager.available_ids), 4)
        self.assertEqual(len(set(worker.request_id_manager.available_ids)), 4)

    def test_exact_prefill_session_release_uses_epoch_round_fence(self) -> None:
        facade = self.make_facade(rows=4)
        worker = facade.worker
        manager = worker.model.gpu_block_manager
        standby = manager.get_bank_descriptor(manager.standby_bank_id)
        rows = [worker.request_id_manager.get_id() for _ in range(2)]
        for round_id, row in enumerate(rows):
            location = manager.reserve_in_bank(
                standby.bank_id,
                request_id=row,
                num_blocks=2,
                logical_kv_len=17 + round_id,
                kv_version=0,
                batch_id=None,
            )
            worker.sessions[(f"starsd-prefill:req:epoch1:round{round_id}", "req")] = _TargetSession(
                SimpleNamespace(request_id=row),
                f"starsd-prefill:req:epoch1:round{round_id}",
                "req",
                bank_adapter=False,
                bank_fence=_SessionBankFence(
                    bank_id=location.bank_id,
                    bank_epoch=location.bank_epoch,
                    row=row,
                    start_block=location.request_start_block,
                    capacity_blocks=location.num_blocks,
                    batch_id=None,
                ),
            )
        self.assertEqual(facade.resource_stats().session_count, 2)

        ack = __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, None),)))
        self.assertEqual(ack["prefill_backend"]["released_count"], 1)
        self.assertEqual(facade.resource_stats().session_count, 1)
        self.assertNotIn(("starsd-prefill:req:epoch1:round0", "req"), worker.sessions)
        self.assertIn(("starsd-prefill:req:epoch1:round1", "req"), worker.sessions)
        self.assertIn(rows[0], worker.request_id_manager.available_ids)
        self.assertNotIn(rows[1], worker.request_id_manager.available_ids)
        self.assertNotIn(rows[0], manager.get_bank_descriptor(standby.bank_id).request_ranges)
        self.assertIn(rows[1], manager.get_bank_descriptor(standby.bank_id).request_ranges)

        replay = __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, None),)))
        self.assertEqual(replay["prefill_backend"]["released_count"], 0)
        self.assertTrue(replay["prefill_backend"].get("replayed"))
        self.assertEqual(len(worker.request_id_manager.available_ids), len(set(worker.request_id_manager.available_ids)))

        __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 1, None),)))
        self.assertEqual(worker.sessions, {})
        self.assertEqual(facade.resource_stats().session_count, 0)
        self.assertEqual(len(worker.request_id_manager.available_ids), 4)
        self.assertEqual(len(set(worker.request_id_manager.available_ids)), 4)

    def test_exact_prefill_release_rejects_never_seen_key(self) -> None:
        facade = self.make_facade(rows=4)
        with self.assertRaisesRegex(RuntimeError, "unknown exact prefill session"):
            __import__("asyncio").run(facade.release((StarsDReleaseRequest("missing", 1, 0, None),)))
        self.assertEqual(facade.resource_stats().session_count, 0)
        self.assertEqual(facade.resource_stats().allocated_range_count, 0)

    def test_exact_prefill_release_wrong_round_does_not_release_live_session(self) -> None:
        facade = self.make_facade(rows=4)
        worker = facade.worker
        manager = worker.model.gpu_block_manager
        standby = manager.get_bank_descriptor(manager.standby_bank_id)
        row = worker.request_id_manager.get_id()
        location = manager.reserve_in_bank(
            standby.bank_id,
            request_id=row,
            num_blocks=2,
            logical_kv_len=17,
            kv_version=0,
            batch_id=None,
        )
        worker.sessions[("starsd-prefill:req:epoch1:round1", "req")] = _TargetSession(
            SimpleNamespace(request_id=row),
            "starsd-prefill:req:epoch1:round1",
            "req",
            bank_adapter=False,
            bank_fence=_SessionBankFence(
                bank_id=location.bank_id,
                bank_epoch=location.bank_epoch,
                row=row,
                start_block=location.request_start_block,
                capacity_blocks=location.num_blocks,
                batch_id=None,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "unknown exact prefill session"):
            __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, None),)))
        self.assertIn(("starsd-prefill:req:epoch1:round1", "req"), worker.sessions)
        self.assertIn(row, manager.get_bank_descriptor(standby.bank_id).request_ranges)
        self.assertNotIn(row, worker.request_id_manager.available_ids)

    def test_exact_prefill_batch_second_forged_has_zero_side_effect(self) -> None:
        facade = self.make_facade(rows=4)
        worker = facade.worker
        manager = worker.model.gpu_block_manager
        standby = manager.get_bank_descriptor(manager.standby_bank_id)
        row = worker.request_id_manager.get_id()
        location = manager.reserve_in_bank(
            standby.bank_id,
            request_id=row,
            num_blocks=2,
            logical_kv_len=17,
            kv_version=0,
            batch_id=None,
        )
        worker.sessions[("starsd-prefill:req-a:epoch1:round0", "req-a")] = _TargetSession(
            SimpleNamespace(request_id=row),
            "starsd-prefill:req-a:epoch1:round0",
            "req-a",
            bank_adapter=False,
            bank_fence=_SessionBankFence(
                bank_id=location.bank_id,
                bank_epoch=location.bank_epoch,
                row=row,
                start_block=location.request_start_block,
                capacity_blocks=location.num_blocks,
                batch_id=None,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "unknown exact prefill session"):
            __import__("asyncio").run(
                facade.release((
                    StarsDReleaseRequest("req-a", 1, 0, None),
                    StarsDReleaseRequest("req-b", 1, 0, None),
                ))
            )
        self.assertIn(("starsd-prefill:req-a:epoch1:round0", "req-a"), worker.sessions)
        self.assertIn(row, manager.get_bank_descriptor(standby.bank_id).request_ranges)
        self.assertNotIn(row, worker.request_id_manager.available_ids)

    def test_exact_prefill_cpu_free_failure_keeps_range_for_retry(self) -> None:
        facade = self.make_facade(rows=4, fail_cpu_free_once=True)
        worker = facade.worker
        manager = worker.model.gpu_block_manager
        standby = manager.get_bank_descriptor(manager.standby_bank_id)
        row = worker.request_id_manager.get_id()
        location = manager.reserve_in_bank(
            standby.bank_id,
            request_id=row,
            num_blocks=2,
            logical_kv_len=17,
            kv_version=0,
            batch_id=None,
        )
        worker.sessions[("starsd-prefill:req:epoch1:round0", "req")] = _TargetSession(
            SimpleNamespace(request_id=row),
            "starsd-prefill:req:epoch1:round0",
            "req",
            bank_adapter=False,
            bank_fence=_SessionBankFence(
                bank_id=location.bank_id,
                bank_epoch=location.bank_epoch,
                row=row,
                start_block=location.request_start_block,
                capacity_blocks=location.num_blocks,
                batch_id=None,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "injected cpu free failure"):
            __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, None),)))
        self.assertIn(("starsd-prefill:req:epoch1:round0", "req"), worker.sessions)
        self.assertIn(row, manager.get_bank_descriptor(standby.bank_id).request_ranges)
        self.assertNotIn(row, worker.request_id_manager.available_ids)

        ack = __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, None),)))
        self.assertEqual(ack["prefill_backend"]["released_count"], 1)
        self.assertNotIn(("starsd-prefill:req:epoch1:round0", "req"), worker.sessions)
        self.assertNotIn(row, manager.get_bank_descriptor(standby.bank_id).request_ranges)
        self.assertIn(row, worker.request_id_manager.available_ids)
        replay = __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, None),)))
        self.assertEqual(replay["prefill_backend"]["released_count"], 0)

    def test_resident_ranges_project_valid_blocks_from_block_manager_counts(self) -> None:
        facade = self.make_facade(rows=4)
        _active, standby = facade.bank_snapshot()
        reservation = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 4, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )[0]
        facade.mark_prepared_and_swap(bank_id=reservation.bank_id, bank_epoch=reservation.bank_epoch, batch_id="batch")
        manager = facade.worker.model.gpu_block_manager
        location = manager.get_bank_descriptor(reservation.bank_id).request_ranges[reservation.row]

        manager._set_valid_blocks_for_location(reservation.row, location, 2)
        first = facade.resident_ranges()
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].capacity_blocks, 4)
        self.assertEqual(first[0].valid_blocks, 2)

        manager._set_valid_blocks_for_location(reservation.row, location, 3)
        second = facade.resident_ranges()
        self.assertEqual(second[0].capacity_blocks, 4)
        self.assertEqual(second[0].valid_blocks, 3)

    def test_resident_ranges_follow_crop_and_reject_forged_valid_blocks(self) -> None:
        facade = self.make_facade(rows=4)
        _active, standby = facade.bank_snapshot()
        reservation = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 4, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )[0]
        facade.mark_prepared_and_swap(bank_id=reservation.bank_id, bank_epoch=reservation.bank_epoch, batch_id="batch")
        manager = facade.worker.model.gpu_block_manager
        manager.crop_blocks_for_seqs(
            torch.tensor([reservation.row], dtype=torch.int32, device="cuda"),
            torch.tensor([33], dtype=torch.int32, device="cuda"),
        )
        projected = facade.resident_ranges()
        self.assertEqual(projected[0].capacity_blocks, 4)
        self.assertEqual(projected[0].valid_blocks, int(manager.num_seq_allocated_blocks[reservation.row].item()))
        self.assertEqual(projected[0].valid_blocks, 3)

        manager.num_seq_allocated_blocks[reservation.row] = 5
        with self.assertRaisesRegex(RuntimeError, "valid blocks"):
            facade.resident_ranges()

    def test_resident_reuse_rebinds_exact_active_range_to_next_round(self) -> None:
        facade = self.make_facade(rows=4)
        _active, standby = facade.bank_snapshot()
        reservation = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 4, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )[0]
        facade.mark_prepared_and_swap(bank_id=reservation.bank_id, bank_epoch=reservation.bank_epoch, batch_id="batch")
        manager = facade.worker.model.gpu_block_manager
        location = manager.get_bank_descriptor(reservation.bank_id).request_ranges[reservation.row]
        manager._set_valid_blocks_for_location(reservation.row, location, 2)
        facade.worker.sessions[(reservation.reservation_id, "req")] = _TargetSession(
            SimpleNamespace(request_id=reservation.row),
            reservation.reservation_id,
            "req",
            bank_adapter=True,
        )
        item = {
            "request_id": "req",
            "request_epoch": 1,
            "previous_round_id": 0,
            "round_id": 1,
            "reservation_id": reservation.reservation_id,
            "bank_id": reservation.bank_id,
            "bank_epoch": reservation.bank_epoch,
            "row": reservation.row,
            "start_block": reservation.start_block,
            "capacity_blocks": reservation.capacity_blocks,
            "valid_blocks": 2,
            "kv_version": int(location.kv_version),
        }

        reused = facade.reuse_resident_batch((item,))

        self.assertEqual(len(reused), 1)
        self.assertEqual(reused[0].round_id, 1)
        self.assertEqual(reused[0].row, reservation.row)
        self.assertEqual(facade.resident_ranges()[0].round_id, 1)
        self.assertNotIn(("req", 1, 0, reservation.reservation_id), facade._resource_rows)
        self.assertIn(("req", 1, 1, reservation.reservation_id), facade._resource_rows)
        replay = facade.reuse_resident_batch((item,))
        self.assertEqual(replay, reused)

        before = (dict(facade._resource_rows), dict(facade._reserve_batches))
        bad = dict(item)
        bad["bank_id"] += 1
        with self.assertRaisesRegex(RuntimeError, "fence|active|location"):
            facade.reuse_resident_batch((bad,))
        self.assertEqual((dict(facade._resource_rows), dict(facade._reserve_batches)), before)
        with self.assertRaisesRegex(RuntimeError, "unknown exact reservation"):
            __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, reservation.reservation_id),)))
        __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 1, reservation.reservation_id),)))
        self.assertEqual(facade.resident_ranges(), ())


    def test_resident_reuse_batch_order_replay_and_fences(self) -> None:
        facade = self.make_facade(rows=6)
        _active, standby = facade.bank_snapshot()
        reservations = facade.reserve_standby_batch(
            (
                StarsDReserveRequest("req-a", 1, 0, 4, "batch"),
                StarsDReserveRequest("req-b", 1, 0, 4, "batch"),
            ),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        facade.mark_prepared_and_swap(bank_id=standby.bank_id, bank_epoch=standby.bank_epoch + 1, batch_id="batch")
        manager = facade.worker.model.gpu_block_manager
        active = manager.get_bank_descriptor(standby.bank_id)
        items = []
        for index, reservation in enumerate(reservations):
            location = active.request_ranges[reservation.row]
            manager._set_valid_blocks_for_location(reservation.row, location, 2 + index)
            facade.worker.sessions[(reservation.reservation_id, reservation.request_id)] = _TargetSession(
                SimpleNamespace(request_id=reservation.row),
                reservation.reservation_id,
                reservation.request_id,
                bank_adapter=True,
            )
            items.append(
                {
                    "request_id": reservation.request_id,
                    "request_epoch": reservation.request_epoch,
                    "previous_round_id": 0,
                    "round_id": 1,
                    "reservation_id": reservation.reservation_id,
                    "bank_id": reservation.bank_id,
                    "bank_epoch": reservation.bank_epoch,
                    "row": reservation.row,
                    "start_block": reservation.start_block,
                    "capacity_blocks": reservation.capacity_blocks,
                    "valid_blocks": 2 + index,
                    "kv_version": int(location.kv_version),
                }
            )

        before = (dict(facade._resource_rows), dict(facade._reserve_batches))
        skipped = dict(items[0])
        skipped["round_id"] = 2
        with self.assertRaisesRegex(RuntimeError, "next consecutive round"):
            facade.reuse_resident_batch((skipped,))
        forged_valid = dict(items[0])
        forged_valid["valid_blocks"] = 1
        with self.assertRaisesRegex(RuntimeError, "valid_blocks fence"):
            facade.reuse_resident_batch((forged_valid,))
        self.assertEqual((dict(facade._resource_rows), dict(facade._reserve_batches)), before)

        first_only = facade.reuse_resident_batch((items[0],))
        self.assertEqual(first_only[0].request_id, "req-a")
        before_mixed = (dict(facade._resource_rows), dict(facade._reserve_batches))
        with self.assertRaisesRegex(RuntimeError, "mixed replay/new"):
            facade.reuse_resident_batch(tuple(items))
        self.assertEqual((dict(facade._resource_rows), dict(facade._reserve_batches)), before_mixed)

        facade = self.make_facade(rows=6)
        _active, standby = facade.bank_snapshot()
        reservations = facade.reserve_standby_batch(
            (
                StarsDReserveRequest("req-a", 1, 0, 4, "batch2"),
                StarsDReserveRequest("req-b", 1, 0, 4, "batch2"),
            ),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        facade.mark_prepared_and_swap(bank_id=standby.bank_id, bank_epoch=standby.bank_epoch + 1, batch_id="batch2")
        manager = facade.worker.model.gpu_block_manager
        active = manager.get_bank_descriptor(standby.bank_id)
        items = []
        for index, reservation in enumerate(reservations):
            location = active.request_ranges[reservation.row]
            manager._set_valid_blocks_for_location(reservation.row, location, 2 + index)
            facade.worker.sessions[(reservation.reservation_id, reservation.request_id)] = _TargetSession(
                SimpleNamespace(request_id=reservation.row),
                reservation.reservation_id,
                reservation.request_id,
                bank_adapter=True,
            )
            items.append(
                {
                    "request_id": reservation.request_id,
                    "request_epoch": reservation.request_epoch,
                    "previous_round_id": 0,
                    "round_id": 1,
                    "reservation_id": reservation.reservation_id,
                    "bank_id": reservation.bank_id,
                    "bank_epoch": reservation.bank_epoch,
                    "row": reservation.row,
                    "start_block": reservation.start_block,
                    "capacity_blocks": reservation.capacity_blocks,
                    "valid_blocks": 2 + index,
                    "kv_version": int(location.kv_version),
                }
            )
        reused = facade.reuse_resident_batch(tuple(items))
        self.assertEqual(tuple(item.request_id for item in reused), ("req-a", "req-b"))
        replayed = facade.reuse_resident_batch(tuple(items))
        self.assertEqual(replayed, reused)


    def test_resident_ranges_empty_after_release(self) -> None:
        facade = self.make_facade(rows=4)
        _active, standby = facade.bank_snapshot()
        reservation = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 4, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )[0]
        facade.mark_prepared_and_swap(bank_id=reservation.bank_id, bank_epoch=reservation.bank_epoch, batch_id="batch")
        self.assertEqual(len(facade.resident_ranges()), 1)
        __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, reservation.reservation_id),)))
        self.assertEqual(facade.resident_ranges(), ())

    def test_facade_reservation_id_batch_id_uses_rightmost_bank_suffix(self) -> None:
        batch_id = "target-0:gen3:bank1:epoch0:req-1:1:0"
        reservation_id = f"{batch_id}:bank1:epoch1:row0"
        self.assertEqual(_batch_id_from_reservation_id(reservation_id), batch_id)

    def test_starsd_h2d_preflight_rejects_second_item_before_gpu_mutation(self) -> None:
        facade = self.make_facade(rows=4)
        facade.worker.model.k_cache = torch.zeros((32, 4), dtype=torch.uint8)
        facade.worker.model.v_cache = torch.zeros((32, 4), dtype=torch.uint8)
        _active, standby = facade.bank_snapshot()
        reservations = facade.reserve_standby_batch(
            (
                StarsDReserveRequest("req-a", 1, 0, 2, "batch"),
                StarsDReserveRequest("req-b", 1, 0, 2, "batch"),
            ),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        items = [self._h2d_item(reservation, copied_blocks=2) for reservation in reservations]
        items[1]["bank_epoch"] += 1
        before_k = facade.worker.model.k_cache.clone()
        before_v = facade.worker.model.v_cache.clone()
        with self.assertRaisesRegex(RuntimeError, "epoch"):
            facade.prepare_bank_from_host(tuple(items), (memoryview(b"a" * 8), memoryview(b"b" * 8)), (memoryview(b"c" * 8), memoryview(b"d" * 8)))
        self.assertTrue(torch.equal(facade.worker.model.k_cache, before_k))
        self.assertTrue(torch.equal(facade.worker.model.v_cache, before_v))

    def test_starsd_d2h_preflight_rejects_second_item_before_host_mutation(self) -> None:
        facade = self.make_facade(rows=4)
        facade.worker.model.k_cache = torch.arange(128, dtype=torch.uint8, device="cuda").reshape(32, 4)
        facade.worker.model.v_cache = torch.arange(128, 256, dtype=torch.uint8, device="cuda").reshape(32, 4)
        _active, standby = facade.bank_snapshot()
        reservations = facade.reserve_standby_batch(
            (
                StarsDReserveRequest("req-a", 1, 0, 2, "batch"),
                StarsDReserveRequest("req-b", 1, 0, 2, "batch"),
            ),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        facade.mark_prepared_and_swap(bank_id=reservations[0].bank_id, bank_epoch=reservations[0].bank_epoch, batch_id="batch")
        items = [self._d2h_item(reservation, dirty_begin=0, dirty_count=1) for reservation in reservations]
        items[1]["start_block"] += 1
        k_payloads = [bytearray(4), bytearray(4)]
        v_payloads = [bytearray(4), bytearray(4)]
        with self.assertRaisesRegex(RuntimeError, "fence"):
            facade.export_dirty_to_host(tuple(items), tuple(memoryview(item) for item in k_payloads), tuple(memoryview(item) for item in v_payloads))
        self.assertEqual(k_payloads, [bytearray(4), bytearray(4)])
        self.assertEqual(v_payloads, [bytearray(4), bytearray(4)])

    def test_starsd_d2h_rejects_stale_kv_version_before_host_mutation(self) -> None:
        facade = self.make_facade(rows=4)
        facade.worker.model.k_cache = torch.arange(128, dtype=torch.uint8, device="cuda").reshape(32, 4)
        facade.worker.model.v_cache = torch.arange(128, 256, dtype=torch.uint8, device="cuda").reshape(32, 4)
        _active, standby = facade.bank_snapshot()
        reservation = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 2, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )[0]
        facade.mark_prepared_and_swap(bank_id=reservation.bank_id, bank_epoch=reservation.bank_epoch, batch_id="batch")
        descriptor = facade.worker.model.gpu_block_manager.get_bank_descriptor(reservation.bank_id)
        location = descriptor.request_ranges[reservation.row]
        item = self._d2h_item(reservation, dirty_begin=0, dirty_count=1)
        item["kv_version"] = int(location.kv_version) + 1
        k_payload = bytearray(4)
        v_payload = bytearray(4)
        with self.assertRaisesRegex(RuntimeError, "kv_version"):
            facade.export_dirty_to_host((item,), (memoryview(k_payload),), (memoryview(v_payload),))
        self.assertEqual(k_payload, bytearray(4))
        self.assertEqual(v_payload, bytearray(4))

    def test_starsd_bank_protection_rejects_reset_swap_and_wrong_release(self) -> None:
        facade = self.make_facade(rows=4)
        active, standby = facade.bank_snapshot()
        facade.protect_bank_epoch("protect-standby", standby.bank_id, standby.bank_epoch)
        facade.protect_bank_epoch("protect-standby", standby.bank_id, standby.bank_epoch)
        with self.assertRaisesRegex(RuntimeError, "another bank"):
            facade.protect_bank_epoch("protect-standby", standby.bank_id, standby.bank_epoch + 1)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            facade.reserve_standby_batch(
                (StarsDReserveRequest("blocked", 1, 0, 1, "blocked"),),
                expected_bank_id=standby.bank_id,
                expected_bank_epoch=standby.bank_epoch,
            )
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            facade.release_bank_epoch("wrong", standby.bank_id, standby.bank_epoch)
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            facade.release_bank_epoch("protect-standby", standby.bank_id, standby.bank_epoch + 1)
        self.assertIn("protect-standby", facade._bank_protections[(standby.bank_id, standby.bank_epoch)])
        facade.release_bank_epoch("protect-standby", standby.bank_id, standby.bank_epoch)

        reservations = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 1, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        facade.protect_bank_epoch("protect-active", active.bank_id, active.bank_epoch)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            facade.mark_prepared_and_swap(bank_id=reservations[0].bank_id, bank_epoch=reservations[0].bank_epoch, batch_id="batch")
        facade.release_bank_epoch("protect-active", active.bank_id, active.bank_epoch)
        facade.mark_prepared_and_swap(bank_id=reservations[0].bank_id, bank_epoch=reservations[0].bank_epoch, batch_id="batch")
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            facade.release_bank_epoch("protect-active", active.bank_id, active.bank_epoch)

    def test_starsd_batch_protection_prevalidates_before_commit(self) -> None:
        facade = self.make_facade(rows=4)
        active, _standby = facade.bank_snapshot()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            facade.protect_bank_epochs((("ok", active.bank_id, active.bank_epoch), ("bad", active.bank_id, active.bank_epoch + 1)))
        self.assertEqual(facade._bank_protections, {})
        self.assertEqual(facade._protection_index, {})

    def test_starsd_prepare_rejects_protected_old_active_before_copy_or_swap(self) -> None:
        facade = self.make_facade(rows=4)
        facade.worker.model.k_cache = torch.zeros((32, 4), dtype=torch.uint8)
        facade.worker.model.v_cache = torch.zeros((32, 4), dtype=torch.uint8)
        old_active, standby = facade.bank_snapshot()
        reservations = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 2, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        item = self._h2d_item(reservations[0], copied_blocks=2)
        before_k = facade.worker.model.k_cache.clone()
        before_v = facade.worker.model.v_cache.clone()
        before_active, before_standby = facade.bank_snapshot()
        before_batches = dict(facade._reserve_batches)
        facade.protect_bank_epoch("dirty-old-active", old_active.bank_id, old_active.bank_epoch)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            facade.prepare_bank_from_host((item,), (memoryview(b"a" * 8),), (memoryview(b"b" * 8),))
        self.assertTrue(torch.equal(facade.worker.model.k_cache, before_k))
        self.assertTrue(torch.equal(facade.worker.model.v_cache, before_v))
        self.assertEqual(facade.bank_snapshot(), (before_active, before_standby))
        self.assertEqual(facade._reserve_batches, before_batches)
        facade.release_bank_epoch("dirty-old-active", old_active.bank_id, old_active.bank_epoch)

    def test_exact_release_after_prepare_allows_next_standby_reserve(self) -> None:
        facade = self.make_facade(rows=4)
        _active, standby = facade.bank_snapshot()
        reservation = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 2, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )[0]
        facade.mark_prepared_and_swap(
            bank_id=reservation.bank_id,
            bank_epoch=reservation.bank_epoch,
            batch_id="batch",
        )
        __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, reservation.reservation_id),)))
        _active2, standby2 = facade.bank_snapshot()
        next_reservation = facade.reserve_standby_batch(
            (StarsDReserveRequest("req-next", 1, 0, 2, "batch-next"),),
            expected_bank_id=standby2.bank_id,
            expected_bank_epoch=standby2.bank_epoch,
        )[0]
        self.assertEqual(next_reservation.bank_id, standby2.bank_id)
        self.assertFalse(facade._resource_rows.get(("req", 1, 0, reservation.reservation_id)))
        self.assertNotIn("batch", facade._reserve_batches)

    def test_exact_release_only_removes_specified_bank_same_row(self) -> None:
        facade = self.make_facade(rows=4)
        manager = facade.worker.model.gpu_block_manager
        active_id = manager.active_bank_id
        standby_id = manager.standby_bank_id
        active = manager.reserve_in_bank(active_id, 1, 2, batch_id="active-batch")
        standby = manager.reserve_in_bank(standby_id, 1, 3, batch_id="standby-batch")
        manager.release_bank_ranges_exact_batch([
            (active.bank_id, active.bank_epoch, 1, active.request_start_block, active.num_blocks, "active-batch")
        ])
        self.assertNotIn(1, manager.get_bank_descriptor(active_id).request_ranges)
        self.assertIn(1, manager.get_bank_descriptor(standby_id).request_ranges)
        self.assertEqual(manager.get_bank_descriptor(standby_id).request_ranges[1], standby)

    def test_exact_release_batch_forged_second_item_zero_side_effect(self) -> None:
        facade = self.make_facade(rows=4)
        _active, standby = facade.bank_snapshot()
        reservations = facade.reserve_standby_batch(
            (
                StarsDReserveRequest("req-a", 1, 0, 2, "batch"),
                StarsDReserveRequest("req-b", 1, 0, 2, "batch"),
            ),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )
        manager = facade.worker.model.gpu_block_manager
        before = self._manager_snapshot(manager)
        bad = [
            (
                reservations[0].bank_id,
                reservations[0].bank_epoch,
                reservations[0].row,
                reservations[0].start_block,
                reservations[0].capacity_blocks,
                "batch",
            ),
            (
                reservations[1].bank_id,
                reservations[1].bank_epoch,
                reservations[1].row,
                reservations[1].start_block + 1,
                reservations[1].capacity_blocks,
                "batch",
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "start"):
            manager.release_bank_ranges_exact_batch(bad)
        self.assertEqual(self._manager_snapshot(manager), before)

    def test_facade_release_backend_failure_preserves_raw_and_metadata_then_replays(self) -> None:
        facade = self.make_facade(rows=4, fail_release_once=True)
        _active, standby = facade.bank_snapshot()
        reservation = facade.reserve_standby_batch(
            (StarsDReserveRequest("req", 1, 0, 2, "batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.bank_epoch,
        )[0]
        facade.mark_prepared_and_swap(
            bank_id=reservation.bank_id,
            bank_epoch=reservation.bank_epoch,
            batch_id="batch",
        )
        session_key = (reservation.reservation_id, reservation.request_id)
        session = SimpleNamespace(bank_adapter=True, request=SimpleNamespace(request_id=reservation.row))
        facade.worker.sessions[session_key] = session
        manager = facade.worker.model.gpu_block_manager
        before = self._manager_snapshot(manager)
        before_rows = dict(facade._resource_rows)
        before_batches = dict(facade._reserve_batches)
        with self.assertRaisesRegex(RuntimeError, "injected release failure"):
            __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, reservation.reservation_id),)))
        self.assertEqual(self._manager_snapshot(manager), before)
        self.assertEqual(facade._resource_rows, before_rows)
        self.assertEqual(facade._reserve_batches, before_batches)
        self.assertIn(session_key, facade.worker.sessions)

        first = __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, reservation.reservation_id),)))
        second = __import__("asyncio").run(facade.release((StarsDReleaseRequest("req", 1, 0, reservation.reservation_id),)))
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertNotIn(("req", 1, 0, reservation.reservation_id), facade._resource_rows)
        self.assertNotIn("batch", facade._reserve_batches)
        self.assertNotIn(session_key, facade.worker.sessions)
        self.assertEqual(facade._available_rows().count(reservation.row), 1)

    def test_ordinary_prefill_exact_release_clears_standby_range_before_next_reserve(self) -> None:
        facade = self.make_facade(rows=4)
        manager = facade.worker.model.gpu_block_manager
        standby = manager.get_bank_descriptor(manager.standby_bank_id)
        location = manager.reserve_in_bank(
            standby.bank_id,
            request_id=0,
            num_blocks=2,
            logical_kv_len=17,
            kv_version=0,
            batch_id=None,
        )
        request = SimpleNamespace(request_id=0)
        facade.worker.sessions[("starsd-prefill:req", "req")] = _TargetSession(
            request,
            "starsd-prefill:req",
            "req",
            bank_adapter=False,
            bank_fence=_SessionBankFence(
                bank_id=location.bank_id,
                bank_epoch=location.bank_epoch,
                row=0,
                start_block=location.request_start_block,
                capacity_blocks=location.num_blocks,
                batch_id=None,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "residual_ranges"):
            facade.reserve_standby_batch(
                (StarsDReserveRequest("blocked", 1, 0, 1, "blocked-batch"),),
                expected_bank_id=standby.bank_id,
                expected_bank_epoch=standby.epoch,
            )

        result = __import__("asyncio").run(
            facade.worker.release_exact_sessions((("starsd-prefill:req", "req"),))
        )
        self.assertEqual(result["released_count"], 1)
        self.assertNotIn(0, manager.get_bank_descriptor(standby.bank_id).request_ranges)
        self.assertNotIn(("starsd-prefill:req", "req"), facade.worker.sessions)
        next_reservation = facade.reserve_standby_batch(
            (StarsDReserveRequest("next", 1, 0, 1, "next-batch"),),
            expected_bank_id=standby.bank_id,
            expected_bank_epoch=standby.epoch,
        )[0]
        self.assertEqual(next_reservation.bank_id, standby.bank_id)


    def make_facade(
        self,
        *,
        rows: int = 4,
        fail_release_once: bool = False,
        fail_cpu_free_once: bool = False,
    ) -> SwiftLLMStarsDTargetFacade:
        worker = SimpleNamespace()
        worker.initialized = True
        worker.request_id_manager = RequestIdManager(rows)
        worker.sessions = {}
        worker.release_calls = 0
        worker.fail_release_once = bool(fail_release_once)
        cpu_manager = SimpleNamespace(fail_free_once=bool(fail_cpu_free_once), free_calls=0, freed_rows=())

        def free_blocks_for_seqs(seq_ids):
            cpu_manager.free_calls += 1
            if cpu_manager.fail_free_once:
                cpu_manager.fail_free_once = False
                raise RuntimeError("injected cpu free failure")
            cpu_manager.freed_rows = tuple(int(item) for item in seq_ids.detach().cpu().tolist())

        cpu_manager.free_blocks_for_seqs = free_blocks_for_seqs
        worker.model = SimpleNamespace(
            gpu_block_manager=BlockManager(
                "GPU",
                32,
                max_seqs_in_block_table=rows,
                max_blocks_per_seq=16,
                block_size=16,
                enable_double_bank=True,
                worker_id="target_w0",
                device_id="cuda:0",
                model_kind="target",
            ),
            cpu_block_manager=cpu_manager,
        )
        async def run_on_model_async(func, *args, **kwargs):
            return func(*args, **kwargs)

        worker._run_on_model_async = run_on_model_async
        worker._release_bank_adapter_session = MethodType(SwiftLLMTargetWorker._release_bank_adapter_session, worker)
        worker._release_exact_prefill_ranges = MethodType(SwiftLLMTargetWorker._release_exact_prefill_ranges, worker)
        worker.release_exact_sessions = MethodType(SwiftLLMTargetWorker.release_exact_sessions, worker)
        real_release = MethodType(SwiftLLMTargetWorker.release_bank_adapter_sessions, worker)

        async def release_bank_adapter_sessions(request_ids=None, client_tags=None, exact_session_keys=None):
            worker.release_calls += 1
            if worker.fail_release_once:
                worker.fail_release_once = False
                raise RuntimeError("injected release failure")
            return await real_release(request_ids=request_ids, client_tags=client_tags, exact_session_keys=exact_session_keys)

        worker.release_bank_adapter_sessions = release_bank_adapter_sessions
        return SwiftLLMStarsDTargetFacade(SimpleNamespace(max_seqs_in_block_table=rows), worker=worker)

    @staticmethod
    def _h2d_item(reservation, *, copied_blocks: int):
        return {
            "plan_id": f"h2d-{reservation.reservation_id}",
            "request_id": reservation.request_id,
            "request_epoch": reservation.request_epoch,
            "round_id": reservation.round_id,
            "reservation_id": reservation.reservation_id,
            "bank_id": reservation.bank_id,
            "bank_epoch": reservation.bank_epoch,
            "row": reservation.row,
            "start_block": reservation.start_block,
            "capacity_blocks": reservation.capacity_blocks,
            "copy_block_count": copied_blocks,
        }

    @staticmethod
    def _d2h_item(reservation, *, dirty_begin: int, dirty_count: int):
        return {
            "plan_id": f"d2h-{reservation.reservation_id}",
            "request_id": reservation.request_id,
            "request_epoch": reservation.request_epoch,
            "round_id": reservation.round_id,
            "bank_id": reservation.bank_id,
            "bank_epoch": reservation.bank_epoch,
            "row": reservation.row,
            "start_block": reservation.start_block,
            "capacity_blocks": reservation.capacity_blocks,
            "dirty_begin_block": dirty_begin,
            "dirty_block_count": dirty_count,
            "post_crop_logical_kv_len": dirty_begin + dirty_count,
        }

    def test_allocate_blocks_for_seqs_bank_path_avoids_nonzero(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        seq_ids = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        target_lens = torch.tensor([32, 64], dtype=torch.int32, device="cuda")
        new_blocks = mgr.allocate_blocks_for_seqs(seq_ids, target_lens)
        torch.cuda.synchronize()

        self.assertEqual(mgr.nonzero_allocate_calls, 0)
        self.assertEqual(new_blocks.detach().cpu().tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(mgr.get_allocated_block_ids(1).detach().cpu().tolist(), [0, 1])
        self.assertEqual(mgr.get_allocated_block_ids(2).detach().cpu().tolist(), [2, 3, 4, 5])

    def test_flag_off_legacy_allocator_still_uses_bitmap_nonzero(self) -> None:
        mgr = self.make_manager(num_blocks=32, enable_double_bank=False)
        seq_ids = torch.tensor([1], dtype=torch.int32, device="cuda")
        target_lens = torch.tensor([32], dtype=torch.int32, device="cuda")
        new_blocks = mgr.allocate_blocks_for_seqs(seq_ids, target_lens)
        torch.cuda.synchronize()

        self.assertFalse(mgr.double_bank_enabled)
        self.assertEqual(mgr.nonzero_allocate_calls, 1)
        self.assertEqual(new_blocks.numel(), 2)
        self.assertEqual(int(mgr.num_seq_allocated_blocks[1].item()), 2)
        mgr.free_blocks_for_seqs(seq_ids)
        self.assertEqual(mgr.num_free_blocks, 32)

    def test_prereserved_capacity_avoids_non_tail_extension(self) -> None:
        mgr = self.make_manager(num_blocks=64)
        loc_a = mgr.reserve_in_bank(0, request_id=1, num_blocks=8, logical_kv_len=32)
        loc_b = mgr.reserve_in_bank(0, request_id=2, num_blocks=8, logical_kv_len=32)
        mgr._set_valid_blocks_for_location(1, loc_a, 2)
        mgr._set_valid_blocks_for_location(2, loc_b, 2)
        seq_ids = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        target_lens = torch.tensor([48, 64], dtype=torch.int32, device="cuda")

        new_blocks = mgr.allocate_blocks_for_seqs(seq_ids, target_lens)
        torch.cuda.synchronize()

        self.assertEqual(new_blocks.detach().cpu().tolist(), [2, 10, 11])
        self.assertEqual(mgr.nonzero_allocate_calls, 0)
        self.assertEqual(mgr.get_allocated_block_ids(1).detach().cpu().tolist(), [0, 1, 2])
        self.assertEqual(mgr.get_allocated_block_ids(2).detach().cpu().tolist(), [8, 9, 10, 11])
        self.assertEqual(mgr.get_bank_descriptor(0).request_ranges[1].num_blocks, 8)
        self.assertEqual(mgr.get_bank_descriptor(0).request_ranges[2].num_blocks, 8)

    def test_non_tail_extension_guard_remains_when_capacity_insufficient(self) -> None:
        mgr = self.make_manager(num_blocks=64)
        loc1 = mgr.reserve_in_bank(0, request_id=1, num_blocks=2, logical_kv_len=32)
        mgr.reserve_in_bank(0, request_id=2, num_blocks=2, logical_kv_len=32)
        mgr._set_valid_blocks_for_location(1, loc1, 1)
        seq_ids = torch.tensor([1], dtype=torch.int32, device="cuda")
        target_lens = torch.tensor([80], dtype=torch.int32, device="cuda")

        with self.assertRaisesRegex(RuntimeError, "reservation capacity exceeded"):
            mgr.allocate_blocks_for_seqs(seq_ids, target_lens)
        self.assertEqual(mgr.nonzero_allocate_calls, 0)

    def test_reserved_capacity_valid_growth_is_atomic_and_does_not_move_alloc_ptr(self) -> None:
        mgr = self.make_manager(num_blocks=64)
        loc = mgr.reserve_in_bank(0, request_id=7, num_blocks=16, logical_kv_len=16, kv_version=4)
        mgr._set_valid_blocks_for_location(7, loc, 1)
        bank = mgr.get_bank_descriptor(0)
        alloc_ptr = bank.alloc_ptr
        for target_blocks in (2, 3, 4):
            target_len = target_blocks * mgr.block_size
            new_blocks = mgr.allocate_blocks_for_seqs(
                torch.tensor([7], dtype=torch.int32, device="cuda"),
                torch.tensor([target_len], dtype=torch.int32, device="cuda"),
            )
            torch.cuda.synchronize()
            current = mgr.get_bank_descriptor(0).request_ranges[7]
            self.assertEqual(current.num_blocks, 16)
            self.assertEqual(mgr.get_bank_descriptor(0).alloc_ptr, alloc_ptr)
            self.assertEqual(int(mgr.num_seq_allocated_blocks[7].item()), target_blocks)
            self.assertEqual(new_blocks.detach().cpu().tolist(), [target_blocks - 1])

    def test_reserved_capacity_exceed_batch_is_zero_side_effect(self) -> None:
        mgr = self.make_manager(num_blocks=64)
        loc = mgr.reserve_in_bank(0, request_id=8, num_blocks=4, logical_kv_len=16, kv_version=3)
        mgr._set_valid_blocks_for_location(8, loc, 1)
        before = self._manager_snapshot(mgr)
        with self.assertRaisesRegex(RuntimeError, "reservation capacity exceeded"):
            mgr.allocate_blocks_for_seqs(
                torch.tensor([8], dtype=torch.int32, device="cuda"),
                torch.tensor([80], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)

    def test_existing_reservation_batch_second_failure_is_zero_side_effect(self) -> None:
        mgr = self.make_manager(num_blocks=64)
        loc1 = mgr.reserve_in_bank(0, request_id=9, num_blocks=4, logical_kv_len=16, kv_version=3)
        loc2 = mgr.reserve_in_bank(0, request_id=10, num_blocks=2, logical_kv_len=16, kv_version=5)
        mgr._set_valid_blocks_for_location(9, loc1, 1)
        mgr._set_valid_blocks_for_location(10, loc2, 1)
        before = self._manager_snapshot(mgr)
        with self.assertRaisesRegex(RuntimeError, "reservation capacity exceeded"):
            mgr.allocate_blocks_for_seqs(
                torch.tensor([9, 10], dtype=torch.int32, device="cuda"),
                torch.tensor([32, 64], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)

    def test_allocate_and_crop_reject_mismatched_batch_lengths_without_side_effects(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        before = self._manager_snapshot(mgr)
        with self.assertRaisesRegex(ValueError, "same length"):
            mgr.allocate_blocks_for_seqs(
                torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
                torch.tensor([16], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)
        with self.assertRaisesRegex(ValueError, "same length"):
            mgr.allocate_blocks_for_seqs(
                torch.tensor([1], dtype=torch.int32, device="cuda"),
                torch.tensor([16, 32], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)
        with self.assertRaisesRegex(ValueError, "same length"):
            mgr.crop_blocks_for_seqs(
                torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
                torch.tensor([16], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)
        with self.assertRaisesRegex(ValueError, "same length"):
            mgr.crop_blocks_for_seqs(
                torch.tensor([1], dtype=torch.int32, device="cuda"),
                torch.tensor([16, 32], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)

    def test_allocate_and_crop_reject_bad_tensor_shape_and_dtype_without_side_effects(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        before = self._manager_snapshot(mgr)
        cases = (
            (
                torch.tensor([[1]], dtype=torch.int32, device="cuda"),
                torch.tensor([16], dtype=torch.int32, device="cuda"),
                ValueError,
                "one-dimensional",
            ),
            (
                torch.tensor([1], dtype=torch.float32, device="cuda"),
                torch.tensor([16], dtype=torch.int32, device="cuda"),
                TypeError,
                "integer",
            ),
            (
                torch.tensor([1], dtype=torch.int32, device="cuda"),
                torch.tensor([True], dtype=torch.bool, device="cuda"),
                TypeError,
                "integer",
            ),
        )
        for seq_ids, target_lens, error_type, pattern in cases:
            with self.assertRaisesRegex(error_type, pattern):
                mgr.allocate_blocks_for_seqs(seq_ids, target_lens)
            self.assertEqual(self._manager_snapshot(mgr), before)
            with self.assertRaisesRegex(error_type, pattern):
                mgr.crop_blocks_for_seqs(seq_ids, target_lens)
            self.assertEqual(self._manager_snapshot(mgr), before)

    def test_negative_target_len_rejected_and_empty_batch_is_noop(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        loc = mgr.reserve_in_bank(0, request_id=6, num_blocks=4, logical_kv_len=64, kv_version=5)
        mgr._set_valid_blocks_for_location(6, loc, 4)
        before = self._manager_snapshot(mgr)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            mgr.allocate_blocks_for_seqs(
                torch.tensor([6], dtype=torch.int32, device="cuda"),
                torch.tensor([-1], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            mgr.crop_blocks_for_seqs(
                torch.tensor([6], dtype=torch.int32, device="cuda"),
                torch.tensor([-1], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)

        new_blocks = mgr.allocate_blocks_for_seqs(
            torch.empty((0,), dtype=torch.int32, device="cuda"),
            torch.empty((0,), dtype=torch.int32, device="cuda"),
        )
        self.assertEqual(new_blocks.numel(), 0)
        mgr.crop_blocks_for_seqs(
            torch.empty((0,), dtype=torch.int32, device="cuda"),
            torch.empty((0,), dtype=torch.int32, device="cuda"),
        )
        self.assertEqual(self._manager_snapshot(mgr), before)

    def test_invalid_reserve_and_version_inputs_are_zero_side_effect(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        before = self._manager_snapshot(mgr)
        with self.assertRaisesRegex(IndexError, "row range"):
            mgr.reserve_in_bank(0, request_id=99, num_blocks=1)
        self.assertEqual(self._manager_snapshot(mgr), before)
        with self.assertRaisesRegex(ValueError, "logical_kv_len"):
            mgr.reserve_in_bank(0, request_id=1, num_blocks=1, logical_kv_len=-1)
        self.assertEqual(self._manager_snapshot(mgr), before)
        with self.assertRaisesRegex(ValueError, "kv_version"):
            mgr.reserve_in_bank(0, request_id=1, num_blocks=1, kv_version=-1)
        self.assertEqual(self._manager_snapshot(mgr), before)
        with self.assertRaisesRegex(ValueError, "row capacity"):
            mgr.reserve_in_bank(0, request_id=1, num_blocks=65)
        self.assertEqual(self._manager_snapshot(mgr), before)

        loc = mgr.reserve_in_bank(0, request_id=1, num_blocks=1, logical_kv_len=16, kv_version=0)
        after_reserve = self._manager_snapshot(mgr)
        with self.assertRaisesRegex(ValueError, "kv_version"):
            mgr.set_bank_location_kv_version(1, bank_id=0, bank_epoch=loc.bank_epoch, kv_version=-1)
        self.assertEqual(self._manager_snapshot(mgr), after_reserve)

    def test_crop_updates_logical_allocated_blocks_without_bitmap_free(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        seq_ids = torch.tensor([1], dtype=torch.int32, device="cuda")
        target_lens = torch.tensor([80], dtype=torch.int32, device="cuda")
        mgr.allocate_blocks_for_seqs(seq_ids, target_lens)
        mgr.crop_blocks_for_seqs(seq_ids, torch.tensor([33], dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()

        self.assertEqual(int(mgr.num_seq_allocated_blocks[1].item()), 3)
        bank = mgr.get_bank_descriptor(0)
        self.assertEqual(bank.request_ranges[1].num_blocks, 5)
        self.assertEqual(bank.request_ranges[1].logical_kv_len, 33)
        self.assertEqual(mgr.get_allocated_block_ids(1).detach().cpu().tolist(), [0, 1, 2])

    def test_crop_same_block_updates_logical_len_and_kv_version_once(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        location = mgr.reserve_in_bank(0, request_id=3, num_blocks=2, logical_kv_len=16, kv_version=5)
        self.assertEqual(location.kv_version, 5)
        mgr.crop_blocks_for_seqs(
            torch.tensor([3], dtype=torch.int32, device="cuda"),
            torch.tensor([17], dtype=torch.int32, device="cuda"),
        )
        torch.cuda.synchronize()

        bank = mgr.get_bank_descriptor(0)
        updated = bank.request_ranges[3]
        self.assertEqual(updated.num_blocks, 2)
        self.assertEqual(updated.logical_kv_len, 17)
        self.assertEqual(updated.kv_version, 6)
        self.assertEqual(int(mgr.num_seq_allocated_blocks[3].item()), 2)

    def test_failed_crop_does_not_advance_kv_version(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        loc = mgr.reserve_in_bank(0, request_id=4, num_blocks=2, logical_kv_len=16, kv_version=7)
        mgr._set_valid_blocks_for_location(4, loc, 1)
        before = self._manager_snapshot(mgr)
        with self.assertRaisesRegex(RuntimeError, "Cannot crop"):
            mgr.crop_blocks_for_seqs(
                torch.tensor([4], dtype=torch.int32, device="cuda"),
                torch.tensor([49], dtype=torch.int32, device="cuda"),
            )
        torch.cuda.synchronize()
        self.assertEqual(self._manager_snapshot(mgr), before)

    def test_crop_batch_duplicate_and_second_failure_are_zero_side_effect(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        loc1 = mgr.reserve_in_bank(0, request_id=11, num_blocks=4, logical_kv_len=64, kv_version=2)
        loc2 = mgr.reserve_in_bank(0, request_id=12, num_blocks=2, logical_kv_len=32, kv_version=8)
        mgr._set_valid_blocks_for_location(11, loc1, 4)
        mgr._set_valid_blocks_for_location(12, loc2, 1)
        before = self._manager_snapshot(mgr)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            mgr.crop_blocks_for_seqs(
                torch.tensor([11, 11], dtype=torch.int32, device="cuda"),
                torch.tensor([32, 16], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)
        with self.assertRaisesRegex(RuntimeError, "Cannot crop"):
            mgr.crop_blocks_for_seqs(
                torch.tensor([11, 12], dtype=torch.int32, device="cuda"),
                torch.tensor([32, 64], dtype=torch.int32, device="cuda"),
            )
        self.assertEqual(self._manager_snapshot(mgr), before)

    def test_successful_crop_increments_kv_version_once_per_request(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        loc = mgr.reserve_in_bank(0, request_id=13, num_blocks=4, logical_kv_len=64, kv_version=10)
        mgr._set_valid_blocks_for_location(13, loc, 4)
        for expected_version, target_len, target_blocks in ((11, 48, 3), (12, 33, 3), (13, 16, 1)):
            mgr.crop_blocks_for_seqs(
                torch.tensor([13], dtype=torch.int32, device="cuda"),
                torch.tensor([target_len], dtype=torch.int32, device="cuda"),
            )
            current = mgr.get_bank_descriptor(0).request_ranges[13]
            self.assertEqual(current.num_blocks, 4)
            self.assertEqual(current.kv_version, expected_version)
            self.assertEqual(current.logical_kv_len, target_len)
            self.assertEqual(int(mgr.num_seq_allocated_blocks[13].item()), target_blocks)

    def test_set_bank_location_kv_version_uses_authoritative_bank_epoch(self) -> None:
        mgr = self.make_manager(num_blocks=32)
        mgr.reserve_in_bank(1, request_id=5, num_blocks=2, logical_kv_len=32, kv_version=0)
        updated = mgr.set_bank_location_kv_version(5, bank_id=1, bank_epoch=0, kv_version=9)
        self.assertEqual(updated.kv_version, 9)
        self.assertEqual(mgr.get_bank_descriptor(1).request_ranges[5].kv_version, 9)
        with self.assertRaisesRegex(RuntimeError, "epoch"):
            mgr.set_bank_location_kv_version(5, bank_id=1, bank_epoch=1, kv_version=10)

    def test_target_hidden_rows_preserve_accepted_order_and_zero_accept(self) -> None:
        worker = object.__new__(SwiftLLMTargetWorker)
        rows = [
            torch.tensor([1.0, 2.0], device="cuda"),
            torch.tensor([3.0, 4.0], device="cuda"),
            torch.tensor([5.0, 6.0], device="cuda"),
        ]

        self.assertIsNone(SwiftLLMTargetWorker._hidden_for_rows(worker, rows, 1, 0))
        hidden = SwiftLLMTargetWorker._hidden_for_rows(worker, rows, 1, 2)
        self.assertEqual(tuple(hidden.shape), (1, 2, 2))
        self.assertEqual(hidden.detach().cpu().tolist(), [[[3.0, 4.0], [5.0, 6.0]]])

        batched_rows = [
            torch.tensor([[7.0, 8.0]], device="cuda"),
            torch.tensor([[9.0, 10.0], [11.0, 12.0]], device="cuda"),
        ]
        batched = SwiftLLMTargetWorker._hidden_for_rows(worker, batched_rows, 0, 2)
        self.assertEqual(tuple(batched.shape), (1, 3, 2))
        self.assertEqual(batched.detach().cpu().tolist(), [[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])

    def test_target_hidden_payload_waits_producer_event_and_rejects_missing_event(self) -> None:
        worker = object.__new__(SwiftLLMTargetWorker)
        rows = [
            torch.tensor([1.0, 2.0], device="cuda"),
            torch.tensor([3.0, 4.0], device="cuda"),
        ]
        with self.assertRaisesRegex(RuntimeError, "producer ready event"):
            SwiftLLMTargetWorker._hidden_payload_for_rows(worker, rows, 0, 2, None)

        producer_event = torch.cuda.Event()
        producer_event.record(torch.cuda.current_stream())
        hidden, ready_event = SwiftLLMTargetWorker._hidden_payload_for_rows(worker, rows, 0, 2, producer_event)
        self.assertIsNotNone(ready_event)
        torch.cuda.current_stream().wait_event(ready_event)
        torch.cuda.synchronize()
        self.assertEqual(tuple(hidden.shape), (1, 2, 2))
        self.assertEqual(hidden.detach().cpu().tolist(), [[[1.0, 2.0], [3.0, 4.0]]])

    @staticmethod
    def _manager_snapshot(mgr: BlockManager):
        banks = tuple(
            (
                bank_id,
                descriptor.alloc_ptr,
                descriptor.epoch,
                descriptor.role,
                descriptor.batch_id,
                tuple(sorted(descriptor.request_ranges.items())),
            )
            for bank_id, descriptor in sorted(mgr._banks.items())
        )
        return (
            banks,
            mgr.num_seq_allocated_blocks.detach().cpu().tolist(),
            mgr.block_table.detach().cpu().tolist(),
        )


if __name__ == "__main__":
    unittest.main()
