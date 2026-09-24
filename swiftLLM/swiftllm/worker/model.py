import dataclasses
import itertools
import math
import os
from typing import Any

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights
from swiftllm.worker.block_manager import BlockManager
from swiftllm.utils import GB
from swiftllm.speculative import aggregate_max_lens_by_seq_id
import swiftllm_c

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer
from .infer_state import LlamaInferState

_ENABLE_DIVERGENCE_TRACE = bool(os.environ.get("STARSD_TARGET_DIVERGENCE_TRACE_DIR"))


@dataclasses.dataclass
class ModelForwardOutput:
    token_ids: list[int]
    hidden_states: torch.Tensor | None = None
    hidden_ready_event: torch.cuda.Event | None = None
    forward_timing: dict[str, Any] | None = None


class LlamaModel:
    """
    LlamaModel - A Llama model that can be used for inference.

    This class also acts as a "worker" that resides on a particular GPU, waiting
    for the control plane (the scheduler) to send commands.

    To initialize, please:
    - call __init__()
    - call load_weights()
    - call profile_num_blocks() on one worker
    - call init_kvcache_and_swap()
    """

    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig
    ):
        """
        Initialize the LlamaModel.
        """
        self.engine_config = engine_config

        # Load model config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)

        # Weight and RoPE cache
        self.weight = None
        self._cos_cached = self._sin_cached = None

        # Layers
        self.pre_layer = None
        self.transformer_layers = None
        self.post_layer = None

        # KV Cache
        self.num_blocks = None
        self.k_cache = self.v_cache = None
        self.k_swap = self.v_swap = None

        # Block manager
        self.cpu_block_manager = self.gpu_block_manager = None
        self.record_forward_timing = False
        self.forward_timing_records: list[dict[str, Any]] = []
        self._last_forward_timing: dict[str, Any] | None = None

    @torch.inference_mode()
    def load_weights(self):
        """
        Load weights and initialize layers
        """
        # Load weights
        self.weight = load_weights(
            self.model_config,
            torch.float16,
            self.engine_config.model_path,
            self.engine_config.use_dummy
        )

        # Initialize rotary embeddings
        self._init_to_get_rotary()

        # Initialize layers
        decoding_piggyback_stream = torch.cuda.Stream()
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.engine_config,
                self.weight.layers[layer_id],
                decoding_piggyback_stream,
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)

    @torch.inference_mode()
    def profile_num_blocks(self) -> int:
        """
        Profiler the number of GPU blocks

        We run a forged prefill batch with the maximum number of tokens and
        sequences, record the peak memory usage, and infer the number of blocks
        that can be allocated.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Synthesis a prefill batch
        num_tokens = self.engine_config.max_tokens_in_batch
        batch_size = self.engine_config.max_batch_size
        input_lens = [num_tokens // batch_size] * batch_size
        input_lens[-1] += num_tokens % batch_size
        input_ids = [
            [0 for _ in range(input_len)]
            for input_len in input_lens
        ]
        seq_ids = list(range(batch_size))
        self.k_cache = self.v_cache = None # pylint: disable=attribute-defined-outside-init
        _ = self.forward(input_ids, seq_ids, [], ignore_kvcache=True)
        torch.cuda.synchronize()

        # peak_memory = torch.cuda.max_memory_allocated()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        peak_memory = total_memory - free_memory
        useable_memory = total_memory*self.engine_config.gpu_mem_utilization
        print(f"[Model.profile] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB")
        if useable_memory < peak_memory:
            raise RuntimeError(f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB ({total_memory/GB:.2f} GB * {self.engine_config.gpu_mem_utilization})")
        block_size_bytes = self.engine_config.block_size * self.model_config.get_kvslot_size()
        num_gpu_blocks = math.floor((useable_memory - peak_memory) / block_size_bytes)

        torch.cuda.empty_cache()
        return num_gpu_blocks

    @torch.inference_mode()
    def init_kvcache_and_swap(self, num_blocks: int, *, external_layout: bool = False):
        self.num_blocks = num_blocks

        # Initialize KV cache
        kvcache_shape = (
            self.num_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")

        # Initialize KV swap space
        kvswap_shape = (
            self.engine_config.num_cpu_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")

        # Autonomous callers supply an explicit block table to every forward.
        # They alone own row/range allocation; do not create a second allocator.
        if external_layout:
            return

        # Initialize block manager
        self.gpu_block_manager = BlockManager(
            "GPU",
            self.num_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size,
            enable_double_bank=bool(getattr(self.engine_config, "enable_double_bank", False)),
            worker_id="GPU",
            device_id=str(self.k_cache.device),
            model_kind="target",
        )
        self.cpu_block_manager = BlockManager(
            "CPU",
            self.engine_config.num_cpu_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )

    def _init_to_get_rotary(self):
        rope_scaling = self.model_config.rope_scaling
        base = self.model_config.rope_theta
        max_position_embeddings = self.model_config.max_position_embeddings

        # Handle the case where rope_scaling is a dictionary (Llama 3.2)
        if isinstance(rope_scaling, dict):
            scaling_factor = rope_scaling.get('factor', 4.0)
            low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
            high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
            rope_type = rope_scaling.get('rope_type', 'llama3')
            original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings', max_position_embeddings)

            # Calculate maximum sequence length based on scaling factor
            max_seq_len = int(original_max_position_embeddings * scaling_factor)

            # Generate position indices
            dim = self.model_config.head_dim
            t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32)

            # Create frequency array with dimensions split between low and high frequency parts
            dim_half = dim // 2
            split_point = int(dim_half * low_freq_factor / (low_freq_factor + high_freq_factor))

            # Apply different scaling factors to different parts of the frequency spectrum
            inv_freq_low = 1.0 / (base ** (torch.arange(0, split_point * 2, 2, device="cuda", dtype=torch.float32) / dim))
            inv_freq_high = 1.0 / (base ** (torch.arange(split_point * 2, dim, 2, device="cuda", dtype=torch.float32) / dim))

            # Apply scaling factors
            low_positions = t / low_freq_factor
            high_positions = t / high_freq_factor

            # Calculate frequencies for both parts
            freqs_low = torch.outer(low_positions, inv_freq_low)
            freqs_high = torch.outer(high_positions, inv_freq_high)

            # Combine frequencies
            freqs = torch.cat([freqs_low, freqs_high], dim=-1)
        else:
            # Original implementation for scalar rope_scaling
            rope_scaling_factor = rope_scaling
            max_seq_len = max_position_embeddings * rope_scaling_factor

            inv_freq = 1.0 / (base ** (torch.arange(0, self.model_config.head_dim, 2, device="cuda", dtype=torch.float32) / self.model_config.head_dim))
            t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32) / rope_scaling_factor
            freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16)
        self._sin_cached = torch.sin(freqs).to(torch.float16)

    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,    # [total_token_num]
        infer_state: LlamaInferState,
        return_hidden: bool = False,
        hidden_layer_ids: list[int] | None = None,
        kv_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Run a forward pass of the LlamaModel.
        """
        input_embds = self.pre_layer.forward(input_ids)
        residual_buf = torch.zeros_like(input_embds)
        selected_hidden_states = {}
        hidden_layer_id_set = set(hidden_layer_ids or [])
        if hidden_layer_ids:
            invalid_ids = [
                layer_id for layer_id in hidden_layer_ids
                if layer_id < 0 or layer_id >= self.model_config.num_layers
            ]
            if invalid_ids:
                raise ValueError(
                    f"hidden_layer_ids out of range for {self.model_config.num_layers} layers: {invalid_ids}"
                )

        for layer in self.transformer_layers:
            input_embds = layer.forward(
                input_embds,
                residual_buf,
                self.k_cache,
                self.v_cache,
                (kv_block_table if kv_block_table is not None else self.gpu_block_manager.block_table) if not infer_state.ignore_kvcache else None,
                infer_state,
            )
            if return_hidden and layer.layer_id in hidden_layer_id_set:
                selected_hidden_states[layer.layer_id] = input_embds + residual_buf

        input_embds += residual_buf
        if return_hidden and hidden_layer_ids:
            if len(selected_hidden_states) != len(hidden_layer_ids):
                raise RuntimeError(
                    f"collected {len(selected_hidden_states)} hidden layers, expected {len(hidden_layer_ids)}"
                )
            hidden_states = torch.cat([selected_hidden_states[layer_id] for layer_id in hidden_layer_ids], dim=-1)
        else:
            hidden_states = input_embds if return_hidden else None
        output_tokens = self.post_layer.forward(input_embds, infer_state)
        return output_tokens, hidden_states

    @torch.inference_mode()
    def forward(
        self,
        input_ids_list: list[list[int]], # [batch_size, *]
        seq_ids_list: list[int],     # [batch_size]
        decoding_seq_lens_list: list[int], # [num_decoding_seqs]
        ignore_kvcache: bool = False,   # Skip actions related to kv cache, useful when profiling the number of kv blocks
        return_hidden: bool = False,
        hidden_layer_ids: list[int] | None = None,
        return_dict: bool = False,
        kv_block_table: torch.Tensor | None = None,
    ) -> list[int] | ModelForwardOutput:
        """
        Run a forward pass of the LlamaModel.

        This function is a wrapper of the `_forward` function. It prepares the infer_state
        and calls the `_forward` function.

        This function is intended to be called by the server.
        """

        num_prefill_seqs = len(input_ids_list) - len(decoding_seq_lens_list)
        flattened_input_ids = list(itertools.chain(*input_ids_list))
        seq_lengths_list = [len(seq) for seq in input_ids_list[:num_prefill_seqs]] + decoding_seq_lens_list

        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int32, device="cuda")

        batch_size = len(input_ids_list)
        num_tokens = len(flattened_input_ids)

        prefill_seq_lens_list = seq_lengths_list[:num_prefill_seqs]
        prefill_seq_lens = torch.tensor(prefill_seq_lens_list, dtype=torch.int32, device="cuda")
        prefill_start_locs = torch.cumsum(prefill_seq_lens, dim=0, dtype=torch.int32) - prefill_seq_lens
        max_prefill_len = max(prefill_seq_lens_list) if prefill_seq_lens_list else 0

        decoding_seq_lens = torch.tensor(decoding_seq_lens_list, dtype=torch.int32, device="cuda")
        max_decoding_len = max(decoding_seq_lens_list) if decoding_seq_lens_list else 0

        position_indices = torch.cat((
            torch.concat([
                torch.arange(
                    0,
                    prefill_seq_len,
                    device="cuda",
                    dtype=torch.int32
                )
                for prefill_seq_len in prefill_seq_lens_list
            ]) if prefill_seq_lens_list else torch.empty(0, device="cuda", dtype=torch.int32),
            decoding_seq_lens - 1
        ), dim=0)

        if not ignore_kvcache and kv_block_table is None:
            alloc_lens_by_seq_id = aggregate_max_lens_by_seq_id(seq_ids_list, seq_lengths_list)
            alloc_seq_ids = torch.tensor(list(alloc_lens_by_seq_id.keys()), dtype=torch.int32, device="cuda")
            alloc_seq_lens = torch.tensor(list(alloc_lens_by_seq_id.values()), dtype=torch.int32, device="cuda")
            self.gpu_block_manager.allocate_blocks_for_seqs(
                alloc_seq_ids,
                alloc_seq_lens
            )

        # Select the seq_block_size
        #
        # Here we use a simple heuristic:
        #
        # In paged attention phase 1, the grid shape is (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # and among these blocks, num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) blocks are useful.
        # Thus we set seq_block_size to be the largest integer that satisfies
        #      num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) >= 1024
        # to fully utilize the GPU. Here 1024 is a magic number (since most high-end
        # GPUs have ~128 SMs, so ~512 SMSPs. Since the decoding-stage attention
        # is mostly a memory-bound operation, I think 1024 is a reasonable number.)
        #
        # In practice, we use `decoding_seq_lens_sum/seq_block_size` to approximate
        # sum(cdiv(decoding_seq_lens, seq_block_size))

        seq_block_size = 2048
        decoding_seq_lens_sum = sum(decoding_seq_lens_list)
        while self.model_config.num_kv_heads*(decoding_seq_lens_sum/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            max_decoding_len / (seq_block_size//2) <= 128:
            seq_block_size //= 2
        if _ENABLE_DIVERGENCE_TRACE:
            self._last_forward_metadata = {
                "batch_size": int(batch_size),
                "num_tokens": int(num_tokens),
                "num_prefill_seqs": int(num_prefill_seqs),
                "num_prefill_tokens": int(num_tokens - (batch_size - num_prefill_seqs)),
                "num_decoding_seqs": int(batch_size - num_prefill_seqs),
                "seq_ids": [int(item) for item in seq_ids_list],
                "seq_lengths": [int(item) for item in seq_lengths_list],
                "decoding_seq_lens": [int(item) for item in decoding_seq_lens_list],
                "max_prefill_len": int(max_prefill_len),
                "max_decoding_len": int(max_decoding_len),
                "seq_block_size": int(seq_block_size),
                "num_seq_blocks": int((max_decoding_len + seq_block_size - 1) // seq_block_size),
                "kv_cache_dtype": str(getattr(getattr(self, "k_cache", None), "dtype", None)),
            }

        infer_state = LlamaInferState(
            batch_size = batch_size,
            num_tokens = num_tokens,

            seq_ids = seq_ids,
            softmax_scale = self.model_config.head_dim ** -0.5,

            num_prefill_seqs = num_prefill_seqs,
            num_prefill_tokens = num_tokens - (batch_size - num_prefill_seqs),
            prefill_seq_start_locs = prefill_start_locs,
            prefill_seq_start_locs_with_end = torch.cat([
                prefill_start_locs,
                torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
            ]),
            prefill_seq_lens = prefill_seq_lens,
            max_prefill_len = max_prefill_len,

            num_decoding_seqs = batch_size - num_prefill_seqs,
            decoding_seq_lens = decoding_seq_lens,
            max_decoding_len = max_decoding_len,

            seq_block_size = seq_block_size,
            num_seq_blocks = (max_decoding_len + seq_block_size-1) // seq_block_size,

            position_cos = self._cos_cached[position_indices],
            position_sin = self._sin_cached[position_indices],

            ignore_kvcache = ignore_kvcache
        )

        forward_input_ids = torch.tensor(flattened_input_ids, dtype=torch.int32, device="cuda")
        forward_start_event = forward_end_event = None
        if self.record_forward_timing:
            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
            forward_start_event.record()

        output_tokens, hidden_states = self._forward(
            forward_input_ids,
            infer_state,
            return_hidden=return_hidden,
            hidden_layer_ids=hidden_layer_ids,
            kv_block_table=kv_block_table,
        )
        hidden_ready_event = None
        if hidden_states is not None:
            hidden_ready_event = torch.cuda.Event()
            hidden_ready_event.record(torch.cuda.current_stream(hidden_states.device))
        forward_timing = None
        if forward_start_event is not None and forward_end_event is not None:
            forward_end_event.record()
            forward_end_event.synchronize()
            forward_timing = {
                "forward_s": float(forward_start_event.elapsed_time(forward_end_event) / 1000.0),
                "batch_size": int(batch_size),
                "num_tokens": int(num_tokens),
                "num_prefill_seqs": int(num_prefill_seqs),
                "num_decoding_seqs": int(batch_size - num_prefill_seqs),
                "return_hidden": bool(return_hidden),
                "hidden_layer_ids": list(hidden_layer_ids or []),
            }
            self._last_forward_timing = forward_timing
            self.forward_timing_records.append(forward_timing)
        token_ids = output_tokens.tolist()
        if return_dict:
            return ModelForwardOutput(
                token_ids=token_ids,
                hidden_states=hidden_states,
                hidden_ready_event=hidden_ready_event,
                forward_timing=forward_timing,
            )
        return token_ids

    def _swap(
        self,
        seq_ids_list: list[int],
        is_swap_in: bool
    ):
        src_block_manager = self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        dst_block_manager = self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = src_block_manager.get_num_allocated_blocks(seq_ids) * self.engine_config.block_size
        src_block_ids = src_block_manager.gather_allocated_blocks_and_free(seq_ids)
        dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lengths)
        swiftllm_c.swap_blocks(
            src_block_ids.tolist(),
            dst_block_ids.tolist(),
            is_swap_in,

            self.k_cache, self.v_cache,
            self.k_swap, self.v_swap
        )

    @torch.inference_mode()
    def swap_in_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        self._swap(seq_ids_list, True)

    @torch.inference_mode()
    def swap_out_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        self._swap(seq_ids_list, False)

    @torch.inference_mode()
    def crop_seqs_resources(self, seq_ids_list: list[int], target_lens_list: list[int]):
        """
        Crop GPU KV blocks for sequences to their logical target lengths.
        """
        if not seq_ids_list:
            return
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        target_lens = torch.tensor(target_lens_list, dtype=torch.int32, device="cuda")
        self.gpu_block_manager.crop_blocks_for_seqs(seq_ids, target_lens)

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids_list: list[int]):
        """
        Free the resources of the specified sequences.
        """
        if not seq_ids_list:
            return
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        self.gpu_block_manager.free_blocks_for_seqs(seq_ids)
        self.cpu_block_manager.free_blocks_for_seqs(seq_ids)
