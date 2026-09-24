# Tests

Run from the repository root after installing `nebulasd[test]`:

```bash
python nebulasd/tools/build_native.py --output build/libstarsd.so
STARSD_NEXT_NATIVE_LIBRARY="$PWD/build/libstarsd.so" python -m pytest nebulasd/tests
```

The suite covers architecture boundaries, state/ABI contracts, publication/concurrency, scheduler behavior, Worker execution semantics and Engine integration. Native-dependent tests need a freshly built library. Tests import the production tree, never historical Worker implementations.

For explicit GPU acceptance, first run the public launcher to obtain a validated cost table and native library, then use those paths:

```bash
STARSD_NEXT_NATIVE_LIBRARY=/path/to/run/native/libstarsd.so \
STARSD_SCHEDULER_IMPL=check STARSD_DMA_MODE=process \
python nebulasd/tests/artifacts/autonomous_engine_gpu.py \
  --draft-model /models/Qwen3-0.6B --target-model /models/Qwen3-8B \
  --cost-table /path/to/run/cost-table.json --output /path/to/new-acceptance \
  --workers 2 --requests 8 --tokens 16 --cohorts 2 --slow-ms 5 --eos
```

This uses real models and an independent full-prefix Target reference, delayed publication and two cohorts. All GPU entries print and assert canonical `swiftllm` and Target facade paths. Large-run performance validation is in [examples](../examples/README.md).

Historical archived tests and obsolete one-off experiment scripts are excluded from this public branch; active test coverage is retained.
