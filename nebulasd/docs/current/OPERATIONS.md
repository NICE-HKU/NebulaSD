# Operations

Follow the [root installation](../../../README.md) and [2D2T launcher](../../examples/README.md). Build the CPU library with `python nebulasd/tools/build_native.py --output build/libstarsd.so`, then set `STARSD_NEXT_NATIVE_LIBRARY` to its absolute path. SwiftLLM's CUDA extension is a separate build.

## Application configuration

```python
import os
from nebulasd import create_engine, NebulaSDConfig, GenerationConfig

config = NebulaSDConfig(
    scheduler_execution="completion",
    draft_placement="stagewise", target_placement="stagewise",
    cost_model="table",
    draft_model_path=os.environ["DRAFT_MODEL_PATH"],
    target_model_path=os.environ["TARGET_MODEL_PATH"],
    backend_cost_table=os.environ["BACKEND_COST_TABLE"],
    gpu_memory_fraction=0.9,
)
```

This is a configuration template, not the full 2D2T capacity preset. Size devices, Bank rows/blocks, HostKV, request slots and payload capacity for your workload. The actual benchmark configuration is recorded in its `configuration.json`. The application's cost table must already match resolved model locations and backend identity; the launcher prepares its own validated copy.

Call `create_engine(config)` from a guarded `if __name__ == "__main__":` entry point because Workers use multiprocessing. Submit token tuples with `GenerationConfig`, advance through `poll` or `stream`, consume output, and always close the client in `finally`. There is no permanent scheduling thread advancing requests automatically. Active cancellation is unsupported.

## Benchmark interpretation

The public launcher uses two Draft and two Target workers, D64/T16, depth 4, prompt lengths 96/112/128/144, two resident 32×64 warmups, 512×128 measured requests, HostKV 24576 blocks per role and payload capacity 128 MiB. It pins compute-cost lookup context to 512; actual model contexts and copy bytes still vary. Optional cohort reclamation is deferred during resident warmups only.

`report.json` contains negative-repeat warmups and repeat 0 for the timed run. Throughput includes formal submission and the complete output tail, excluding model startup, warmup, final physical retirement and shutdown. `steady.valid=false` means there is no shared active window; do not interpret its zero CPU accumulator as no CPU usage. Output chunk intervals can include multiple tokens and are not per-token latency.

Check process exit status and `exitcodes.json` as well as report status: the benchmark writes its passed report before closing Workers. The shell launcher checks all three. Reports retain config/source/native identity, output chunks and WORK records. Exact output length alone does not establish token equivalence to an independent reference.

## Diagnostics

Use separate output directories for profiling and unprofiled throughput. Host compute intervals include submission/synchronization; CUDA forward event intervals may include stream gaps. Actual memcpy/kernel activity and overlap need an appropriate GPU trace. `summarize_profile.py` merges recorded observations but does not turn missing measurements into zero work.

Use `STARSD_SCHEDULER_IMPL=check` for decision validation. The thread DMA route changes execution wiring as well as copy service placement; interpret process/thread comparisons accordingly.
