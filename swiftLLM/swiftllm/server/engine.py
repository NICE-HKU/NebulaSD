import asyncio
import functools
from typing import AsyncGenerator

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.model import LlamaModel
from swiftllm.utils import GB
from swiftllm.speculative import compute_acceptance_for_plan

from .backend_local import BatchPlan, ForwardRow, build_batch_plan
from .tokenization_engine import TokenizationEngine
from .structs import Request, RawRequest, StepOutput
from .scheduler import Scheduler


class Engine:
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)
        self.initialized = False

        # The following fields will be created on `init_model()`
        self.model = None
        self.event_loop = None
        self.scheduler = None
        self.tokenization_engine = None

        self.untokenized_raw_requests: list[tuple[Request, str]] = []

    async def _run_on_model_async(self, func, *args, **kwargs):
        """
        Run a function on the model asynchronously, and return the result
        """
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial)

    async def initialize(self):
        self.event_loop = asyncio.get_event_loop()

        print("[Engine] Initializing model...")
        self.model = LlamaModel(self.engine_config)

        print("[Engine] Loading weights...")
        self.model.load_weights()

        print("[Engine] Profiling kv blocks...")
        num_gpu_blocks = self.model.profile_num_blocks()
        num_cpu_blocks = self.engine_config.num_cpu_blocks
        block_size_bytes = self.engine_config.block_size*self.model_config.get_kvslot_size()
        print(f"[Engine] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")
        print(f"[Engine] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks*block_size_bytes/GB:.2f} GB)")

        print("[Engine] Allocating kv cache and swap...")
        self.model.init_kvcache_and_swap(num_gpu_blocks)

        print("[Engine] Initializing scheduler...")
        self.scheduler = Scheduler(self.model, self.engine_config, num_gpu_blocks)

        print("[Engine] Initializing tokenization engine...")
        self.tokenization_engine = TokenizationEngine.remote(self.engine_config)

        print("[Engine] Model initialized")
        self.initialized = True
    
    async def add_request_and_stream(self, raw_request: RawRequest) -> AsyncGenerator[StepOutput, None]:
        """
        Add a raw request to the engine and stream the output of the request (streaming mode)
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        while True:
            step_output = await request.output_q.get()
            yield step_output
            request.output_q.task_done()
            if step_output.request.is_finished():
                break
    
    async def add_request_and_wait(self, raw_request: RawRequest) -> tuple[Request, list[int]]:
        """
        Add a raw request to the engine and wait for the completion (non-streaming mode)

        Return the output token ids
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        await request.finished_event.wait()
        return (request, request.output_token_ids)

    async def _tokenize_raw_request_event_loop(self):
        """
        Event loop for tokenizing raw requests
        """
        while True:
            if not self.untokenized_raw_requests:
                # No new raw requests, sleep for a bit
                await asyncio.sleep(0.002)
                continue

            # Tokenize the raw request in batch
            cur_untokenized_raw_requests = self.untokenized_raw_requests
            self.untokenized_raw_requests = []

            prompts = [prompt for _, prompt in cur_untokenized_raw_requests]
            prompt_token_ids = await self.tokenization_engine.batched_tokenize.remote(prompts)

            new_requests = []
            for (request, _), prompt_token_id in zip(cur_untokenized_raw_requests, prompt_token_ids):
                request.prompt_token_ids = prompt_token_id
                request.prompt_len = len(prompt_token_id)
                new_requests.append(request)

            self.scheduler.on_requests_arrival(new_requests)
            await asyncio.sleep(0.001)  # yield the event loop
    
    async def _main_event_loop(self):
        """
        Event loop for forwarding the model
        """
        while True:
            # Get the next batch from the scheduler
            cur_batch, cur_swap_in, cur_swap_out = self.scheduler.get_next_batch()
            if not cur_batch and not cur_swap_in and not cur_swap_out:
                # No new batch, sleep for a bit
                await asyncio.sleep(0.005)
                continue

            # Perform swap in/out
            if cur_swap_out:
                await self._run_on_model_async(
                    self.model.swap_out_seqs,
                    [req.request_id for req in cur_swap_out]
                )
            if cur_swap_in:
                await self._run_on_model_async(
                    self.model.swap_in_seqs,
                    [req.request_id for req in cur_swap_in]
                )
            
            # Forward the model
            batch_plan = build_batch_plan(cur_batch)
            output_tokens = await self._run_on_model_async(
                self.model.forward,
                batch_plan.input_ids_list(),
                batch_plan.seq_ids_list(),
                batch_plan.decoding_seq_lens_list()
            )

            # Deal with output tokens
            finished_req_ids = []
            spec_crop_req_ids = []
            spec_crop_target_lens = []

            def append_output_token(req: Request, token_id: int):
                req.output_token_ids.append(token_id)
                req.output_q.put_nowait(StepOutput(token_id, req))
                if req.is_finished() and req.request_id not in finished_req_ids:
                    finished_req_ids.append(req.request_id)
                    req.finished_event.set()

            for row_idx, row in enumerate(batch_plan.prefill_rows):
                append_output_token(row.request, output_tokens[row_idx])

            for row in batch_plan.normal_decode_rows:
                append_output_token(row.request, output_tokens[row.output_row_start])

            for plan_item in batch_plan.verify_plan_items:
                req = plan_item.request
                posterior = output_tokens[
                    plan_item.output_row_start:plan_item.output_row_start + plan_item.output_row_count
                ]
                remaining = req.output_len - len(req.output_token_ids)
                accepted_token_ids, num_accepted_draft_tokens = compute_acceptance_for_plan(
                    plan_item,
                    posterior,
                    remaining_output_len=remaining,
                )
                for token_id in accepted_token_ids:
                    append_output_token(req, token_id)

                req.spec_stats["num_draft_tokens"] += len(plan_item.draft_token_ids)
                req.spec_stats["num_accepted_tokens"] += num_accepted_draft_tokens
                req.spec_stats["num_spec_steps"] += 1
                req.spec_proposal = None

                spec_crop_req_ids.append(req.request_id)
                spec_crop_target_lens.append(req.logical_kv_len_after_current_state())

            if spec_crop_req_ids:
                await self._run_on_model_async(
                    self.model.crop_seqs_resources,
                    spec_crop_req_ids,
                    spec_crop_target_lens
                )
            await self._run_on_model_async(
                self.model.free_seqs_resources,
                finished_req_ids
            )
            
            # Inform the scheduler
            self.scheduler.on_batch_finish(cur_batch)
    
    async def start_all_event_loops(self):
        """
        Start all event loops
        """
        assert self.initialized, "Engine not initialized. Please call `initialize()` before starting the event loop."
        await asyncio.gather(
            self._tokenize_raw_request_event_loop(),
            self._main_event_loop()
        )
