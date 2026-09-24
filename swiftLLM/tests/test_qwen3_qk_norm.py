from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight


def test_qwen3_config_enables_qk_norm_weight_registration():
    cfg = LlamaModelConfig({
        "model_type": "qwen3",
        "num_hidden_layers": 1,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "intermediate_size": 12288,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
    })
    weight = LlamaTransformerLayerWeight(0, cfg, __import__("torch").float16)
    registered = {item.attr_name: item for item in weight.registered_weights}

    assert cfg.has_qk_norm is True
    assert registered["q_norm"].key == "model.layers.0.self_attn.q_norm.weight"
    assert registered["k_norm"].key == "model.layers.0.self_attn.k_norm.weight"
    assert registered["q_norm"].shape == (128,)
    assert registered["k_norm"].shape == (128,)
