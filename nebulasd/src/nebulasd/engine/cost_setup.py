"""Cold cost-table compatibility validation; never runs inside scheduling."""

import hashlib
from pathlib import Path


def cost_identity(draft_model, target_model, devices, *, block_size=16, memory_fraction=0.90):
    import torch

    root = Path(__file__).resolve().parents[4]

    def model(path):
        path = Path(path).resolve()
        return dict(
            path=str(path),
            config_sha256=hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        )

    digest = hashlib.sha256()
    for path in sorted((root / "swiftLLM/swiftllm").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    gpus = [torch.cuda.get_device_name(d) for d in devices]
    if len(set(gpus)) != 1:
        raise ValueError("cost table requires homogeneous calibrated GPUs")
    return dict(
        draft_model=model(draft_model),
        target_model=model(target_model),
        gpu=gpus[0],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        dtype="float16",
        backend="canonical-swiftllm",
        swiftllm_sha256=digest.hexdigest(),
        block_size=block_size,
        gpu_memory_fraction=memory_fraction,
    )


def make_estimator(config, layout, draft_layout=None):
    from nebulasd.scheduler.placement import PlacementEstimator

    if config.cost_model == "legacy":
        return PlacementEstimator(block_bytes=layout.block_bytes)
    if layout.dtype != "torch.float16":
        raise ValueError("measured backend requires calibrated FP16 KV layout")
    from nebulasd.scheduler.cost_table import CostTable
    from nebulasd.scheduler.measured_placement import MeasuredPlacementEstimator

    identity = cost_identity(
        config.draft_model_path,
        config.target_model_path,
        config.devices,
        block_size=config.block_size,
        memory_fraction=config.gpu_memory_fraction,
    )
    if getattr(config,'backend_cost_source_sha256',''):
        # Explicitly pinned historical calibration: preserve every measured cost
        # and all hardware/model compatibility checks. No claim of new calibration.
        identity = identity | dict(swiftllm_sha256=config.backend_cost_source_sha256)
    table = CostTable.load(config.backend_cost_table, expected=identity)
    # Fail at startup for missing families, before admitting/fencing requests.
    for stage, sync in (
        ("draft_first", 0),
        ("draft_cached", 1),
        ("draft_cached", 2),
        ("target_verify", 0),
    ):
        table.predict(stage, batch=1, kv=1, depth=config.max_proposal_depth, sync=sync)
    for stage in ("target_prefill", "H2D", "D2H"):
        table.predict(stage, batch=1, kv=1, byte_count=1)
    if draft_layout is not None and draft_layout.dtype != "torch.float16":
        raise ValueError("measured Draft requires calibrated FP16 KV layout")
    return MeasuredPlacementEstimator(table, layout.block_bytes,
        draft_block_bytes=None if draft_layout is None else draft_layout.block_bytes)
