import asyncio
import os
import time

import torch
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.server.target_worker import SwiftLLMTargetWorker

MODEL_PATH = os.environ.get("TARGET_MODEL_PATH", "")


async def main():
    if not MODEL_PATH:
        raise ValueError("Set TARGET_MODEL_PATH to a local model directory")
    from pathlib import Path
    import swiftllm
    import swiftllm.server.starsd_target_facade as facade
    root = Path(__file__).resolve().parents[1]
    for name, module in (("swiftllm", swiftllm), ("swiftllm.server.starsd_target_facade", facade)):
        path = Path(module.__file__).resolve()
        print(f"{name}.__file__={path}", flush=True)
        assert path.is_relative_to(root)
    cfg = EngineConfig(
        model_path=MODEL_PATH,
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.82,
        num_cpu_blocks=128,
        max_seqs_in_block_table=64,
        max_blocks_per_seq=4096,
        max_batch_size=64,
        max_tokens_in_batch=2048,
        speculative_method="dflash",
        speculative_max_draft_tokens=4,
    )

    worker = SwiftLLMTargetWorker(cfg)
    t0 = time.perf_counter()
    await worker.initialize()
    print(f"INIT_SECONDS {time.perf_counter() - t0:.2f}", flush=True)

    original_forward = worker.model.forward
    forward_calls = []

    def logging_forward(input_ids_list, seq_ids_list, decoding_seq_lens_list, *args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original_forward(input_ids_list, seq_ids_list, decoding_seq_lens_list, *args, **kwargs)
        torch.cuda.synchronize()
        call = {
            "input_ids_list": [list(x) for x in input_ids_list],
            "seq_ids_list": list(seq_ids_list),
            "decoding_seq_lens_list": list(decoding_seq_lens_list),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }
        forward_calls.append(call)
        print("FORWARD", len(forward_calls) - 1, call, flush=True)
        return out

    worker.model.forward = logging_forward
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    prompts = [
        "Short prompt:",
        "A medium length prompt for testing:",
        "This is a longer prompt used to check different prefill lengths in the worker:",
        "Another prompt with enough words to create a different sequence length before decoding begins:",
    ]
    ids = tok(prompts, add_special_tokens=True)["input_ids"]
    print("PROMPT_LENS", [len(x) for x in ids], flush=True)

    clients = [f"client-{i}" for i in range(4)]
    reqs = [f"req-{i}" for i in range(4)]

    prefill = await asyncio.gather(*[
        worker.submit_prefill(clients[i], reqs[i], ids[i], 32)
        for i in range(4)
    ])
    print("AFTER_PREFILL", [r.payload for r in prefill], flush=True)

    await worker.submit_decode(clients[1], reqs[1])
    await worker.submit_verify(clients[2], reqs[2], [101, 102])
    await worker.submit_decode(clients[3], reqs[3])
    await worker.submit_decode(clients[3], reqs[3])

    before = len(forward_calls)
    draft_by_req = [
        [201],
        [211, 212, 213],
        [221, 222],
        [231, 232, 233, 234],
    ]

    mixed = await asyncio.gather(*[
        worker.submit_verify(clients[i], reqs[i], draft_by_req[i])
        for i in range(4)
    ])

    mixed_calls = forward_calls[before:]
    print("MIXED_VERIFY_RESULTS", [r.payload for r in mixed], flush=True)
    print("MIXED_VERIFY_FORWARD_CALLS", mixed_calls, flush=True)

    call = mixed_calls[0]
    counts = {sid: call["seq_ids_list"].count(sid) for sid in set(call["seq_ids_list"])}

    first_lens = []
    seen = set()
    for sid, seq_len in zip(call["seq_ids_list"], call["decoding_seq_lens_list"]):
        if sid not in seen:
            seen.add(sid)
            first_lens.append(seq_len)

    print(
        "DIFFERENT_PROGRESS_CONTINUOUS_VERIFY: PASS",
        {"row_counts_by_seq_id": counts, "first_lens": first_lens},
        flush=True,
    )

    await asyncio.gather(*[
        worker.submit_end(clients[i], reqs[i])
        for i in range(4)
    ])

    worker._worker_task.cancel()
    try:
        await worker._worker_task
    except asyncio.CancelledError:
        pass

asyncio.run(main())