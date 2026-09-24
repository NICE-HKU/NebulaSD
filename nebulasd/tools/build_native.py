"""Build the CPU control-plane library explicitly, with no CUDA/toolchain download."""

import argparse
from pathlib import Path
import subprocess
import sys
import runpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "csrc/control_plane/native.cpp"
    root = Path(__file__).resolve().parents[1]
    schema = runpy.run_path(str(root / "src/nebulasd/scheduler/native_schema.py"))
    (output.parent / "scheduler_schema.h").write_text(schema["header"]())
    scheduler = root / "csrc/scheduler/scheduler.cpp"
    mode = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    subprocess.run([args.compiler, "-std=c++17", "-O3", "-fPIC", mode, str(source), str(scheduler), "-I", str(output.parent), "-ffp-contract=off", "-o", str(output)], check=True)
    print(output)
