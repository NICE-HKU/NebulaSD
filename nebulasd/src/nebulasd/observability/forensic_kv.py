"""Explicit, disruptive KV evidence capture. Never installed in performance runs."""

import os
from dataclasses import asdict
from pathlib import Path


def install(backend, recorder):
    """Capture valid-prefix blocks before selected verification calls.

    STARSD_FORENSIC_KV_DIR enables synchronous CPU snapshots on the existing
    model executor. This intentionally perturbs timing and is diagnostic only.
    It does not change KV values, Bank metadata, proposals or accepted outputs.
    """
    directory = os.environ.get("STARSD_FORENSIC_KV_DIR")
    if not directory:
        return None
    slots = {int(x) for x in os.environ.get("STARSD_FORENSIC_SLOTS", "2").split(",")}
    rounds = {int(x) for x in os.environ.get("STARSD_FORENSIC_ROUNDS", "8,9,10").split(",")}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    worker = backend.bank_facade.worker

    async def capture(items):
        selected = [
            i
            for i in items
            if i.request_slot in slots and i.request_epoch >= 2 and i.round_id in rounds
        ]
        if not selected:
            return []

        def snapshot():
            import torch

            paths = []
            for item in selected:
                loc = item.bank_location
                bank = worker.model.gpu_block_manager.get_bank_descriptor(loc.bank_id)
                assert int(bank.epoch) == loc.bank_epoch
                length = item.prompt_token_count + len(item.committed_output_token_ids) - 1
                blocks = (length + backend._block_size - 1) // backend._block_size
                start = int(bank.base_block) + loc.offset_blocks
                data = dict(
                    input=asdict(item),
                    logical_kv_len=length,
                    block_size=backend._block_size,
                    physical_start_block=start,
                    k=worker.model.k_cache[start : start + blocks].detach().cpu(),
                    v=worker.model.v_cache[start : start + blocks].detach().cpu(),
                )
                torch.cuda.synchronize()
                path = (
                    directory
                    / f"{recorder.owner}_s{item.request_slot}_e{item.request_epoch}_r{item.round_id}.pt"
                )
                torch.save(data, path)
                paths.append(str(path))
            return paths

        return await worker._run_on_model_async(snapshot)

    return capture
