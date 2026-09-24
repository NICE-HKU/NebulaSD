"""Canonical model construction; Engine and Scheduler never import CUDA backends."""
from math import prod
from nebulasd.config import HostKVLayout
from nebulasd.core.enums import WorkerRole
from ..canonical import import_canonical


def model_layout(config, *, draft=False):
    if not config.draft_model_path or not config.target_model_path:
        raise ValueError('draft_model_path and target_model_path are required')
    import_canonical()
    from swiftllm.model_config import LlamaModelConfig
    model = LlamaModelConfig.load_from_model_path(config.draft_model_path if draft else config.target_model_path)
    shape = (model.num_layers,model.num_kv_heads,config.block_size,model.head_dim)
    return HostKVLayout(prod(shape)*2,'torch.float16',shape)
