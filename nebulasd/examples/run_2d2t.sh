#!/usr/bin/env bash
# Build, validate and run the complete 2D2T benchmark from any working directory.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: DRAFT_MODEL_PATH=/models/Qwen3-0.6B TARGET_MODEL_PATH=/models/Qwen3-8B \
       bash nebulasd/examples/run_2d2t.sh [benchmark options]
Environment:
  PYTHON             Python executable (default: python3)
  OUTPUT_DIR         New run directory (default: outputs/2d2t-<unique suffix>)
  BACKEND_COST_TABLE Measured cost JSON (default: bundled RTX4090/Qwen3 reference)
  CUDA_VISIBLE_DEVICES  Select the four GPUs, e.g. 0,1,2,3
Defaults: 512 requests, 128 generated tokens, D64/T16, two resident warmups.
Options are forwarded to benchmark_2d2t.py, e.g. --requests 8 --tokens 16.
--output, --source, --draft-model, --target-model and --cost-table are managed here.
The Python/CUDA environment and local model weights must already be installed.
EOF
    exit 0
fi
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH to a local model directory}"
: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH to a local model directory}"
for arg in "$@"; do
    case "$arg" in
        --output|--output=*|--source|--source=*|--draft-model|--draft-model=*|--target-model|--target-model=*|--cost-table|--cost-table=*|--devices|--devices=*)
            echo "Use environment variables for paths/GPU selection: $arg" >&2; exit 2 ;;
    esac
done
# Resolve caller-relative paths before running anything from the repository.
DRAFT_MODEL_PATH="$(cd -- "$DRAFT_MODEL_PATH" && pwd)"
TARGET_MODEL_PATH="$(cd -- "$TARGET_MODEL_PATH" && pwd)"
BACKEND_COST_TABLE="${BACKEND_COST_TABLE:-$ROOT/nebulasd/examples/cost_tables/rtx4090_qwen3.json}"
BACKEND_COST_TABLE="$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "$BACKEND_COST_TABLE")"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
    mkdir -- "$OUTPUT_DIR" # Never overwrite an existing experiment.
else
    mkdir -p -- "$ROOT/outputs"
    OUTPUT_DIR="$(mktemp -d "$ROOT/outputs/2d2t-XXXXXXXX")"
fi
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
exec > >(tee "$OUTPUT_DIR/run.log") 2>&1
printf 'Output: %s\n' "$OUTPUT_DIR"
export STARSD_DMA_MODE=process STARSD_SCHEDULER_IMPL=cpp
export STARSD_NEXT_NATIVE_LIBRARY="$OUTPUT_DIR/native/libstarsd.so"
"$PYTHON" "$ROOT/nebulasd/tools/build_native.py" --output "$STARSD_NEXT_NATIVE_LIBRARY"
"$PYTHON" "$ROOT/nebulasd/examples/prepare_cost_table.py" \
    --input "$BACKEND_COST_TABLE" --output "$OUTPUT_DIR/cost-table.json" \
    --draft-model "$DRAFT_MODEL_PATH" --target-model "$TARGET_MODEL_PATH"
"$PYTHON" "$ROOT/nebulasd/examples/benchmark_2d2t.py" \
    --output "$OUTPUT_DIR/run" --draft-model "$DRAFT_MODEL_PATH" \
    --target-model "$TARGET_MODEL_PATH" --cost-table "$OUTPUT_DIR/cost-table.json" "$@"
"$PYTHON" - "$OUTPUT_DIR/run" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
r = json.loads((p / 'report.json').read_text())
codes = json.loads((p / 'exitcodes.json').read_text())
assert r['status'] == 'passed', r['status']
assert codes and all(c == 0 for worker in codes.values() for c in worker.values()), codes
x = r['runs'][-1]
print(f"PASS: {len(x['outputs'])} requests, {sum(map(len,x['outputs']))} tokens, "
      f"{x['output_tokens_per_second']:.1f} token/s, "
      f"request P99 {x['request_latency']['p99_ms']/1000:.3f}s")
print(f"Steady window valid: {x['steady']['valid']}; all Worker exits successful.")
PY
