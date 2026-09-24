# Architecture

NebulaSD separates a single Engine owner from independently scheduled Draft and Target worker pools. The application submits token IDs; Draft proposes tokens; Target verifies them and publishes cumulative accepted output. Same-role workers can migrate request KV through their shared HostKV arena.

The default runtime uses completion scheduling, stagewise placement and a measured cost table. Each logical Worker comprises control, execution and DMA processes. Control owns local execution progression and publication; execution owns model/GPU storage; DMA owns ordered transfers. The Engine reads shared facts and issues complete WORK authorizations rather than individual execution steps.

| Topic | Documentation | Main source |
| --- | --- | --- |
| Requests, state, output | [Engine](ENGINE.md) | `engine/core.py`, `engine/work_progress.py` |
| Local execution | [Workers](WORKER.md) | `workers/isolated.py`, `workers/runtime.py` |
| Resource ownership | [KV](KV.md) | `workers/banks.py`, `workers/dma_process.py` |
| Planning | [Scheduler](SCHEDULER.md) | `scheduler/completion.py`, `scheduler/native_completion.py` |
| Shared facts and commands | [Protocol](PROTOCOL.md) | `workers/work.py`, `engine/work_ledger.py` |
| Build and configuration | [Operations](OPERATIONS.md) | `tools/`, `examples/` |
| Support boundaries | [Limitations](KNOWN_ISSUES.md) | Public API and validation scope |

Source paths above are relative to `src/nebulasd`, except tools/examples. Use only the sibling canonical `swiftLLM` backend. See the [root quickstart](../../../README.md) for installation.
