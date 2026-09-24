# Workers

## Process boundaries

| Process | Ownership |
| --- | --- |
| control | WORK validation, dependencies, CPU input compilation, Runtime/Bank allocation, compute submission, result publication and retirement |
| execution | Canonical model, GPU KV/block tables, complete compute Plan execution and compute clocks |
| DMA | CUDA IPC mappings, HostKV registration, ordered H2D/D2H and metadata operations |

`TargetProcesses` also hosts Draft lifecycle wiring. Execution owns the CUDA storage until DMA exits. Three processes do not imply three threads: DMA and metadata service threads remain necessary.

Control constructs `TargetInputs` or `DraftInputs` normally. Runtime receives explicit `execute` and `write_metadata` callbacks. Execution-only backends use `job_threads=False`; they do not create input compilers or local job pools. The explicit thread DMA diagnostic route uses the same compilers and model algorithms with different process wiring.

Across process boundaries WORK is encoded and validated. Within control, immutable Work objects and INPUTS are passed directly; ordered WORK/retirement/stop handling retains its FIFO. The private COMPUTE/COMPUTED channel still uses bounded pickle payloads; the system is not universally fixed-ABI or zero-copy.

## Authorization and inputs

The four WORK operations are Target prefill/verify and Draft initial/decode. A WORK is complete authorization. Runtime waits for its precise dependencies locally without requesting a second RUN from Engine.

Draft uses prompt/config plus Target decisions, and its own previous snapshot/HostKV for continuation. Reconciliation determines retained prefix, suffix and proposal budget. Target verify uses Draft proposals, its previous Target decision and its own HostKV. Production Target anchoring does not read the Engine's cumulative output arena.

Dependencies validate epochs, rounds, handles and tickets. Natural-finish decisions are processed before waiting indefinitely for inputs a finished member no longer needs. Such members become `SKIPPED_FINISHED`; this does not implement external cancellation.

## Local progression

```text
WORK -> dependencies -> layout/row allocation
                   -> CPU Plan preparation
                   -> import and metadata readiness
Plan + H2D + metadata + compute slot -> execute
-> result available -> D2H -> physical release and publication completion
```

Plans can be prepared incrementally and cached per ready member. `Banks` owns actual row/layout allocation. Two banks share the model row pool, allowing preparation/transfers beside compute, but not two concurrent forwards on the same execution resource.

Runtime owns physical progression; Publisher owns shared-fact encoding. Result payload/rows can publish before D2H; HostKV-ready publishes after export. H2D rows remain available to Worker-side consumers but Engine does not subscribe to per-request H2D. Shared completion contains the exact WORK and member outcomes. All-skipped WORKs have completion without result/HostKV member publications.

Failures are fail-fast, not transparent online Worker recovery. Profiling distinguishes control-side preparation/submission from execution-side compute. Host intervals include CPU work and synchronization; they are not CUDA kernel-active durations.

Sources: [Runtime](../../src/nebulasd/workers/runtime.py), [isolated wiring](../../src/nebulasd/workers/isolated.py), [Target inputs](../../src/nebulasd/workers/target/inputs.py), [Draft inputs](../../src/nebulasd/workers/draft/inputs.py), [Publisher](../../src/nebulasd/workers/target/publication.py).
