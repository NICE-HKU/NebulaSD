import os
from contextlib import contextmanager

import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.worker.kernels.linear import linear

_ENABLE_FORWARD_PHASE_NVTX = os.environ.get("STARSD_ENABLE_FORWARD_PHASE_NVTX", "0").lower() in {"1", "true", "yes", "on"}
_DIVERGENCE_TOPK = int(os.environ.get("STARSD_TARGET_DIVERGENCE_TOPK", "0") or "0")
_DIVERGENCE_PROBE_TOKENS = tuple(
    int(item)
    for item in os.environ.get("STARSD_TARGET_DIVERGENCE_PROBE_TOKENS", "2155,4911").split(",")
    if item.strip()
)


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


class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        # Slice to get the last token embedding for each request
        last_token_indices = torch.cat(
            (
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        )
        last_input = torch.empty((infer_state.batch_size, self.model_config.hidden_size), device=input_embds.device, dtype=input_embds.dtype)
        last_input[:, :] = input_embds[last_token_indices, :]
        with _forward_phase_range("post_lm_head"):
            # Apply RMS-norm
            rmsnorm_inplace(
                last_input,
                self.weights.final_norm,
                self.model_config.rms_norm_eps
            )
            logits = linear(last_input, self.weights.lm_head)    # [batch_size, vocab_size]
            output_tokens = torch.argmax(logits, dim=1)
            if _DIVERGENCE_TOPK > 0:
                self._last_logits_probe = _capture_logits_probe(
                    logits,
                    output_tokens,
                    topk=_DIVERGENCE_TOPK,
                    probe_tokens=_DIVERGENCE_PROBE_TOKENS,
                )
        return output_tokens


def _capture_logits_probe(logits: torch.Tensor, output_tokens: torch.Tensor, *, topk: int, probe_tokens: tuple[int, ...]) -> dict:
    scores = logits.detach().float()
    k = min(max(int(topk), 1), int(scores.shape[1]))
    values, indices = torch.topk(scores, k=k, dim=1)
    valid_probe_tokens = [token for token in probe_tokens if 0 <= token < int(scores.shape[1])]
    probe_scores = scores[:, valid_probe_tokens].detach().cpu().tolist() if valid_probe_tokens else []
    return {
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "output_tokens": [int(token) for token in output_tokens.detach().cpu().tolist()],
        "topk_ids": [[int(token) for token in row] for row in indices.detach().cpu().tolist()],
        "topk_scores": [[float(value) for value in row] for row in values.detach().cpu().tolist()],
        "probe_token_ids": valid_probe_tokens,
        "probe_scores": [[float(value) for value in row] for row in probe_scores],
    }
