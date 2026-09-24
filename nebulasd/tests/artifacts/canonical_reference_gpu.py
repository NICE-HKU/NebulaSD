"""Independent full-prefix reference; always asserts canonical SwiftLLM imports."""
def reference_many(model_path, device, prompts, count, result):
    from nebulasd.canonical import import_canonical
    import_canonical()
    import torch
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel
    torch.cuda.set_device(device)
    model = LlamaModel(EngineConfig(model_path, False, 16, .9, 0, 4, 32, 4, 4096))
    model.load_weights()
    model.init_kvcache_and_swap(32)
    outputs = [[] for _ in prompts]
    for _ in range(count):
        predictions = model.forward([list(p)+out for p,out in zip(prompts,outputs)], list(range(len(prompts))), [])
        for output, prediction in zip(outputs,predictions):
            output.append(int(prediction))
    result.put(outputs)


