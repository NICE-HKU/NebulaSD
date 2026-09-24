import unittest
from types import SimpleNamespace

import torch

from swiftllm.server.structs import RawRequest, Request
from swiftllm.server.target_worker import SwiftLLMTargetWorker, TargetTask, _TargetSession


class _FakeRequestIdManager:
    def __init__(self, ids):
        self.ids = list(ids)
        self.freed = []

    def get_id(self):
        if not self.ids:
            raise RuntimeError("no fake ids left")
        return self.ids.pop(0)

    def free_id(self, req_id):
        self.freed.append(req_id)


class _FakeBlockManager:
    def __init__(self, *, num_blocks=8, max_seqs=8, max_blocks_per_seq=8, block_size=4):
        self.num_blocks = num_blocks
        self.num_free_blocks = num_blocks
        self.block_size = block_size
        self.num_seq_allocated_blocks = torch.zeros((max_seqs,), dtype=torch.int32)
        self.block_table = torch.empty((max_seqs, max_blocks_per_seq), dtype=torch.int32)
        self.is_block_free = torch.ones((num_blocks,), dtype=torch.bool)

    def allocate_blocks_for_seqs(self, seq_ids, target_lens):
        new_blocks = []
        for seq_id_tensor, target_len_tensor in zip(seq_ids.cpu(), target_lens.cpu()):
            seq_id = int(seq_id_tensor.item())
            target_len = int(target_len_tensor.item())
            target_num_blocks = (target_len + self.block_size - 1) // self.block_size
            cur_num_blocks = int(self.num_seq_allocated_blocks[seq_id].item())
            if target_num_blocks < cur_num_blocks:
                raise RuntimeError("cannot shrink in allocate")
            needed = target_num_blocks - cur_num_blocks
            free = torch.nonzero(self.is_block_free).view(-1)[:needed]
            if int(free.numel()) != needed:
                raise RuntimeError("not enough fake blocks")
            if needed:
                self.is_block_free[free] = False
                self.block_table[seq_id, cur_num_blocks:target_num_blocks] = free.to(torch.int32)
                new_blocks.extend(int(block_id) for block_id in free.tolist())
                self.num_free_blocks -= needed
            self.num_seq_allocated_blocks[seq_id] = target_num_blocks
        return torch.tensor(new_blocks, dtype=torch.int32)

    def free_blocks_for_seqs(self, seq_ids):
        for seq_id_tensor in seq_ids.cpu():
            seq_id = int(seq_id_tensor.item())
            cur_num_blocks = int(self.num_seq_allocated_blocks[seq_id].item())
            block_ids = self.block_table[seq_id, :cur_num_blocks]
            if cur_num_blocks:
                self.is_block_free[block_ids] = True
                self.num_free_blocks += cur_num_blocks
            self.num_seq_allocated_blocks[seq_id] = 0

    def get_allocated_block_ids(self, seq_id):
        num_blocks = int(self.num_seq_allocated_blocks[seq_id].item())
        return self.block_table[seq_id, :num_blocks].clone()


class _FakeModel:
    def __init__(self, *, block_size=4):
        self.gpu_block_manager = _FakeBlockManager(block_size=block_size)
        self.k_cache = torch.zeros((8, 2, 1, block_size, 3), dtype=torch.float32)
        self.v_cache = torch.zeros_like(self.k_cache)

    def free_seqs_resources(self, seq_ids_list):
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32)
        self.gpu_block_manager.free_blocks_for_seqs(seq_ids)


def _make_worker(model, next_ids):
    worker = object.__new__(SwiftLLMTargetWorker)
    worker.engine_config = SimpleNamespace(block_size=4, speculative_max_draft_tokens=8)
    worker.model = model
    worker.event_loop = None
    worker.request_id_manager = _FakeRequestIdManager(next_ids)
    worker.pending_target_tasks = None
    worker.sessions = {}
    worker._worker_task = None
    worker.initialized = True
    return worker


class SwiftLLMTargetWorkerMigrationTests(unittest.TestCase):
    def test_request_finishes_at_stop_before_budget(self):
        request = Request(RawRequest("", 8, stop_token_ids={55}))
        request.output_token_ids = [10, 55]

        self.assertTrue(request.is_finished())

    def test_export_import_restores_request_state_and_copies_kv_blocks(self):
        model_a = _FakeModel()
        worker_a = _make_worker(model_a, next_ids=[])

        req_a = Request(RawRequest("", 12, stop_token_ids={151643, 151645}))
        req_a.prompt_len = 5
        req_a.output_token_ids = [101, 102, 103]
        req_a.request_id = 1
        req_a.spec_enabled = True
        req_a.spec_stats = {
            "num_draft_tokens": 4,
            "num_accepted_tokens": 2,
            "num_spec_steps": 1,
        }
        logical_kv_len = req_a.logical_kv_len_after_current_state()
        model_a.gpu_block_manager.allocate_blocks_for_seqs(
            torch.tensor([req_a.request_id], dtype=torch.int32),
            torch.tensor([logical_kv_len], dtype=torch.int32),
        )
        src_block_ids = model_a.gpu_block_manager.get_allocated_block_ids(req_a.request_id)
        src_k = torch.arange(
            src_block_ids.numel() * model_a.k_cache[0].numel(),
            dtype=model_a.k_cache.dtype,
        ).reshape((src_block_ids.numel(),) + tuple(model_a.k_cache.shape[1:]))
        src_v = src_k + 1000
        model_a.k_cache[src_block_ids] = src_k
        model_a.v_cache[src_block_ids] = src_v
        worker_a.sessions[("client-a", "req-a")] = _TargetSession(req_a, "client-a", "req-a")
        free_blocks_before = model_a.gpu_block_manager.num_free_blocks

        state = worker_a.export_session_for_migration("client-a", "req-a")

        self.assertEqual(state["old_internal_request_id"], 1)
        self.assertEqual(state["logical_kv_len"], 7)
        self.assertEqual(state["num_blocks"], 2)
        self.assertEqual(model_a.gpu_block_manager.num_free_blocks, free_blocks_before)
        self.assertTrue(torch.equal(state["k_cache_blocks"], src_k))
        self.assertTrue(torch.equal(state["v_cache_blocks"], src_v))
        self.assertGreater(state["kv_bytes"], 0)

        model_b = _FakeModel()
        worker_b = _make_worker(model_b, next_ids=[4])
        summary = worker_b.import_session_from_migration(state, client_tag="client-b", request_id="req-b")

        self.assertEqual(summary["old_internal_request_id"], 1)
        self.assertEqual(summary["new_internal_request_id"], 4)
        self.assertIn(("client-b", "req-b"), worker_b.sessions)

        req_b = worker_b.sessions[("client-b", "req-b")].request
        self.assertNotEqual(req_b.request_id, req_a.request_id)
        self.assertEqual(req_b.prompt_len, req_a.prompt_len)
        self.assertEqual(req_b.output_len, req_a.output_len)
        self.assertEqual(req_b.stop_token_ids, req_a.stop_token_ids)
        self.assertEqual(req_b.output_token_ids, req_a.output_token_ids)
        self.assertEqual(req_b.spec_stats, req_a.spec_stats)
        self.assertTrue(req_b.spec_enabled)

        dst_block_ids = model_b.gpu_block_manager.get_allocated_block_ids(req_b.request_id)
        self.assertEqual(int(dst_block_ids.numel()), state["num_blocks"])
        self.assertTrue(torch.equal(model_b.k_cache[dst_block_ids], state["k_cache_blocks"]))
        self.assertTrue(torch.equal(model_b.v_cache[dst_block_ids], state["v_cache_blocks"]))

        task = TargetTask(
            task_id="verify-after-import",
            client_tag="client-b",
            request_id="req-b",
            phase="verify",
            payload={"draft_token_ids": [201, 202], "proposal_kind": "dflash_block"},
        )
        prepared = worker_b._prepare_model_task(task)
        self.assertIs(prepared, req_b)
        self.assertEqual(req_b.spec_proposal.kind, "dflash_block")
        self.assertEqual(req_b.spec_proposal.draft_token_ids, [201, 202])


if __name__ == "__main__":
    unittest.main()
