# Reference cost table

`rtx4090_qwen3.json` preserves the measured rows of the development experiment dated 2026-09-15 (`fixed-cost-table.json`). No latency values were changed for publication. Personal model locations were removed; the launcher binds local paths only after comparing all remaining identity fields.

The table is tied to RTX 4090, Qwen3-0.6B/Qwen3-8B config hashes, FP16, block size 16, GPU memory fraction 0.9, PyTorch 2.11.0+cu128 / CUDA 12.8 and the recorded vendored SwiftLLM source digest. It is historical calibration, not a fresh measurement of every later control-plane refactor. Runtime compatibility checks are preserved.

A schema-version-1 JSON contains `compatibility` and `rows`. Each row identifies `stage`, `batch`, `kv`, `depth`, `sync`, `bytes`, `samples`, `p50_ms`, `p95_ms`, and a timing `boundary`. At least 20 samples and positive durations are required. Required families include `draft_first` (depth 4, sync 0), `draft_cached` (depth 4, sync 1 and 2), `target_prefill`, `target_verify` (depth 4), `H2D` and `D2H`. The public fixed-context benchmark needs compute rows at context 512 covering its batch ranges.

See [CostTable](../../src/nebulasd/scheduler/cost_table.py) and [identity construction](../../src/nebulasd/engine/cost_setup.py). To use other hardware/backend/model versions, supply measured data through `BACKEND_COST_TABLE`; do not edit identity hashes to bypass checks. An automated portable calibrator is not included.
