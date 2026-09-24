# Examples

`run_2d2t.sh` is the public entry point. It builds a native library in a fresh output directory, validates a cost table, invokes `benchmark_2d2t.py`, and fails if execution or Worker shutdown fails. Installation is separate: see the [root README](../../README.md).

Required environment: `DRAFT_MODEL_PATH`, `TARGET_MODEL_PATH`. Optional: `PYTHON`, `OUTPUT_DIR`, `BACKEND_COST_TABLE`, `CUDA_VISIBLE_DEVICES`. `--help` prints usage. Additional CLI arguments control requests, output length, batches, timeout, warmups and profiling. Defaults match the 512×128, D64/T16 2D2T experiment.

`benchmark_2d2t.py` uses synthetic prompt token IDs (lengths 96/112/128/144), a fixed compute-cost context of 512, and proposal depth 4. It deliberately postpones optional cohort recycling through resident warmups, keeping model processes loaded. This is a benchmark policy, not a production API default. Actual KV/copy dimensions are not fixed to 512.

The benchmark requires a source checkout with its sibling `swiftLLM`. `support/` provides timing and fixed-context cost utilities; it does not participate in the production hot path. `prepare_cost_table.py` validates calibration compatibility and writes a path-bound copy without modifying measured timings. `cost_tables/` documents the included reference.

A passed large benchmark checks completion count/length, retirement and clean shutdown. Use `tests/artifacts/autonomous_engine_gpu.py` for a small independent-reference correctness check. Do not interpret throughput from delayed, profiled or correctness runs as the default benchmark throughput.
