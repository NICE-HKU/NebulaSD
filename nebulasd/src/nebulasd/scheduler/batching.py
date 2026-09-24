"""Opportunistic batching with an injectable admission-time window policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchWindow:
    delay_ns: int = 0

    def __post_init__(self):
        if self.delay_ns < 0:
            raise ValueError("negative batching window")

    def allow(self, *, now_ns, oldest_ready_ns, full):
        return full or now_ns - oldest_ready_ns >= self.delay_ns


def fit(candidates, *, max_rows, max_tokens, max_blocks, token_cost):
    """Stable FIFO tie-breaking; an oversized item cannot hide smaller work."""
    chosen, tokens, blocks = [], 0, 0
    for request in candidates:
        cost = token_cost(request)
        if len(chosen) >= max_rows:
            break
        if tokens + cost <= max_tokens and blocks + request.capacity_blocks <= max_blocks:
            chosen.append(request)
            tokens += cost
            blocks += request.capacity_blocks
    return chosen
