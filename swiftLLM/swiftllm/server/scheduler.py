from collections import deque

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.server.backend_local import RequestIdManager
from swiftllm.utils import cdiv
from swiftllm.server.structs import Request


class Scheduler:
    """
    A strict FCFS scheduler for the LLM engine, which supports paged attention
    as well as swapping in/out
    """

    def __init__(self, model_config: LlamaModelConfig, engine_config: EngineConfig, num_gpu_blocks: int):
        self.model_config = model_config
        self.engine_config = engine_config
        self.num_gpu_blocks = num_gpu_blocks

        # Request in the following three deques are sorted by their arrival time
        self.waiting_q = deque()
        self.running_q: list[Request] = []
        self.swapped_q = deque()

        # Number of GPU blocks occupied by decoding requests
        # This number should always equal to sum(self._get_block_needed(req) for req in self.running_q)
        self.num_decoding_gpu_blocks = 0
        self.num_free_cpu_blocks = engine_config.num_cpu_blocks

        self.request_id_manager = RequestIdManager(engine_config.max_seqs_in_block_table)
    
    def _get_block_needed(self, request: Request) -> int:
        """
        Get the number of blocks needed for a request.
        """
        if request.spec_proposal is not None and request.spec_proposal.kind == "linear":
            max_len_this_iter = (
                request.prompt_len
                + len(request.output_token_ids)
                + len(request.spec_proposal.draft_token_ids)
            )
        else:
            max_len_this_iter = request.prompt_len + request.get_cur_output_len()
        return cdiv(max_len_this_iter, self.engine_config.block_size)

    def _get_request_iter_width(self, request: Request) -> int:
        if request.is_prefill_stage():
            return request.prompt_len
        if request.spec_proposal is not None and request.spec_proposal.kind == "linear":
            return 1 + len(request.spec_proposal.draft_token_ids)
        return 1

    def _get_request_row_count(self, request: Request) -> int:
        if request.is_prefill_stage():
            return 1
        if request.spec_proposal is not None and request.spec_proposal.kind == "linear":
            return 1 + len(request.spec_proposal.draft_token_ids)
        return 1

    def _get_requests_iter_width(self, requests: list[Request]) -> int:
        return sum(self._get_request_iter_width(req) for req in requests)

    def _get_requests_row_count(self, requests: list[Request]) -> int:
        return sum(self._get_request_row_count(req) for req in requests)
    
    def on_requests_arrival(self, requests: list[Request]):
        """
        Called when a batch of new requests arrives and finishes tokenization
        """
        self.waiting_q.extend(requests)
    
    def get_next_batch(self) -> tuple[list[Request], list[Request], list[Request]]:
        """
        Called when the engine wants a new batch to be forwarded
        Returns (new_batch, newly_swapped_in, newly_swapped_out)
        """
        if not self.swapped_q:
            # Try to launch a new prefill batch
            cur_batch = []
            cur_batch_block_needed = 0
            cur_num_tokens_sum = 0
            cur_row_count = 0
            while self.waiting_q:
                cur_seq: Request = self.waiting_q[0]
                cur_seq_block_needed = self._get_block_needed(cur_seq)
                cur_seq_row_count = self._get_request_row_count(cur_seq)
                cur_seq_width = self._get_request_iter_width(cur_seq)
                if  cur_row_count + cur_seq_row_count <= self.engine_config.max_batch_size and \
                    self._get_requests_row_count(self.running_q) + cur_row_count + cur_seq_row_count <= self.engine_config.max_batch_size and \
                    cur_batch_block_needed + cur_seq_block_needed + self.num_decoding_gpu_blocks <= self.num_gpu_blocks and \
                    cur_num_tokens_sum + cur_seq_width <= self.engine_config.max_tokens_in_batch:
                    cur_batch.append(cur_seq)
                    cur_batch_block_needed += cur_seq_block_needed
                    cur_num_tokens_sum += cur_seq_width
                    cur_row_count += cur_seq_row_count
                    self.waiting_q.popleft()
                else:
                    # Strict FCFS
                    break
            if cur_batch:
                # Going to launch a prefill batch
                # If you want decoding requests to be piggybacked, you can do it here
                for req in cur_batch:
                    req.request_id = self.request_id_manager.get_id()
                self.running_q.extend(cur_batch)
                self.num_decoding_gpu_blocks += cur_batch_block_needed
                return cur_batch, [], []
        
        # Try to launch a decoding batch
        # TODO Optimize this `sum` if possible
        self.num_decoding_gpu_blocks = sum(self._get_block_needed(req) for req in self.running_q)
        newly_swapped_out = []
        while self._get_requests_row_count(self.running_q) > self.engine_config.max_batch_size or \
              self._get_requests_iter_width(self.running_q) > self.engine_config.max_tokens_in_batch or \
              self.num_decoding_gpu_blocks > self.num_gpu_blocks:
            # Preempt the last running seq
            victim = self.running_q.pop()
            self.num_decoding_gpu_blocks -= self._get_block_needed(victim)
            newly_swapped_out.append(victim)

        newly_swapped_in = []
        if newly_swapped_out:
            self.swapped_q.extendleft(newly_swapped_out)
        else:
            # No swap-out triggered, try to swap in some requests if possible
            while self.swapped_q:
                cur_seq = self.swapped_q[0]
                num_cur_seq_blocks = self._get_block_needed(cur_seq)
                if self._get_requests_row_count(self.running_q) + self._get_request_row_count(cur_seq) <= self.engine_config.max_batch_size and \
                   self._get_requests_iter_width(self.running_q) + self._get_request_iter_width(cur_seq) <= self.engine_config.max_tokens_in_batch and \
                   self.num_decoding_gpu_blocks + num_cur_seq_blocks <= self.num_gpu_blocks:
                    self.running_q.append(cur_seq)
                    self.num_decoding_gpu_blocks += num_cur_seq_blocks
                    self.swapped_q.popleft()
                    newly_swapped_in.append(cur_seq)
                else:
                    break
        
        return self.running_q, newly_swapped_in, list(reversed(newly_swapped_out))
    
    def on_batch_finish(self, batch: list[Request]):
        """
        Called when a batch finishes
        """
        self.request_id_manager.free_ids([
            req.request_id
            for req in batch
            if req.is_finished()
        ])
        self.running_q = [
            req
            for req in self.running_q
            if not req.is_finished()
        ]
