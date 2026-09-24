import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F

def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=None, causal=True):
    outs = []
    q_locs = cu_seqlens_q.detach().cpu().tolist()
    k_locs = cu_seqlens_k.detach().cpu().tolist()

    for i in range(len(q_locs) - 1):
        qs, qe = q_locs[i], q_locs[i + 1]
        ks, ke = k_locs[i], k_locs[i + 1]

        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)

        if qi.shape[1] != ki.shape[1]:
            repeat = qi.shape[1] // ki.shape[1]
            ki = ki.repeat_interleave(repeat, dim=1)
            vi = vi.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(
            qi, ki, vi,
            is_causal=causal,
            scale=softmax_scale,
        )
        outs.append(out.squeeze(0).transpose(0, 1))

    return torch.cat(outs, dim=0)

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.infer_state import LlamaInferState

from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace, rmsnorm_inplace
from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace

_ENABLE_FORWARD_PHASE_NVTX = os.environ.get("STARSD_ENABLE_FORWARD_PHASE_NVTX", "0").lower() in {"1", "true", "yes", "on"}


@contextmanager
def _forward_phase_range(name: str):
    if _ENABLE_FORWARD_PHASE_NVTX:
        try:
            torch.cuda.nvtx.range_push(name)
        except Exception:
            yield
        else:
            try:
                yield
            finally:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass
    else:
        yield


class LlamaTransformerLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        decoding_piggyback_stream: torch.cuda.Stream,
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.decoding_piggyback_stream = decoding_piggyback_stream
        self.layer_id = layer_id
    
    def forward(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor, # [num_tokens, hidden_size]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        infer_state: LlamaInferState,
    ) -> torch.Tensor:
        # (fused) Add last layer's residual, and perform RMSNorm
        # Before: input_embds is the output of the last FFN block, and residual_buf
        #         is the residual to be added to input_embds
        # After: input_embds will be RMSNorm(input_embds + residual_buf), and
        #        residual_buf will be input_embds + residual_buf (which will be
        #        used as the residual after the attention block)
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            self.weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        # Calculate QKV
        with _forward_phase_range(f"layer_{self.layer_id}_qkv"):
            q = linear(input_embds, self.weight.q_proj)		# [num_total_tokens, hidden_size]
            k = linear(input_embds, self.weight.k_proj)		# [num_total_tokens, num_kv_heads*head_dim]
            v = linear(input_embds, self.weight.v_proj)		# [num_total_tokens, num_kv_heads*head_dim]
            q = q.view(-1, self.model_config.num_q_heads,  self.model_config.head_dim)	# [num_total_tokens, num_q_heads, head_dim]
            k = k.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]
            v = v.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]

        if getattr(self.model_config, "has_qk_norm", False):
            rmsnorm_inplace(
                q.reshape(-1, self.model_config.head_dim),
                self.weight.q_norm,
                self.model_config.rms_norm_eps,
            )
            rmsnorm_inplace(
                k.reshape(-1, self.model_config.head_dim),
                self.weight.k_norm,
                self.model_config.rms_norm_eps,
            )

        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            infer_state
        )

        if not infer_state.ignore_kvcache:
            with _forward_phase_range(f"layer_{self.layer_id}_store_kvcache"):
                store_kvcache(
                    k, v,
                    k_cache, v_cache,
                    block_table,
                    self.model_config,
                    self.engine_config,
                    infer_state,
                    self.layer_id
                )
        store_kvcache_event = torch.cuda.Event()
        store_kvcache_event.record()

        # Attention
        attention_width = self.model_config.num_q_heads * self.model_config.head_dim
        o = (input_embds if attention_width == self.model_config.hidden_size else
             input_embds.new_empty((input_embds.shape[0], attention_width)))
        if infer_state.num_prefill_seqs > 0:
            # Here the performance of vLLM's flash attention is better than us,
            # so use vllm_flash_attn
            o[:infer_state.num_prefill_tokens, :] = flash_attn_varlen_func(
                q[:infer_state.num_prefill_tokens, :, :],
                k[:infer_state.num_prefill_tokens, :, :],
                v[:infer_state.num_prefill_tokens, :, :],
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.max_prefill_len,
                infer_state.max_prefill_len,
                softmax_scale=infer_state.softmax_scale,
                causal=True
            ).reshape(-1, attention_width)
            # prefill_attention(
            #     q, k, v, o[:infer_state.num_prefill_tokens, :],
            #     self.model_config, self.engine_config, infer_state
            # )
        if infer_state.num_decoding_seqs > 0:
            assert not infer_state.ignore_kvcache
            with torch.cuda.stream(self.decoding_piggyback_stream):
                torch.cuda.current_stream().wait_event(store_kvcache_event)
                with _forward_phase_range(f"layer_{self.layer_id}_paged_attention"):
                    paged_attention(
                        q[infer_state.num_prefill_tokens:, :, :],
                        k_cache, v_cache, block_table,
                        self.model_config, self.engine_config, infer_state,
                        self.layer_id,
                        o[infer_state.num_prefill_tokens:, :],
                    )
                event = torch.cuda.Event()
                event.record()
            # The caller may be the banked Draft compute stream. Join
            # attention there before O projection reads its output.
            torch.cuda.current_stream().wait_event(event)
        
        # Output GEMM
        with _forward_phase_range(f"layer_{self.layer_id}_o_proj"):
            o = linear(o, self.weight.o_proj)	# [num_total_tokens, hidden_size]

        # residual & FFN norm
        fused_add_rmsnorm_inplace(o, residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)
        q = None
        k = None
        v = None

        # FFN
        with _forward_phase_range(f"layer_{self.layer_id}_ffn"):
            up_gate_proj = linear(o, self.weight.up_gate_proj)
            silu_and_mul_inplace(up_gate_proj)
            ffn_out = linear(up_gate_proj[:, :self.model_config.ffn_inter_dim], self.weight.down_proj)

        return ffn_out
    