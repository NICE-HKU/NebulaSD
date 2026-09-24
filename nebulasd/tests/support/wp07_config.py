"""One argument contract for CLI, pytest and CPU validation."""

import argparse
import math


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="WP07 real model/copy acceptance")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-devices", default="0")
    parser.add_argument("--draft-model")
    parser.add_argument("--draft-device", type=int, default=1)
    parser.add_argument("--draft-workers", type=int, default=1)
    parser.add_argument("--draft-memory-fraction", type=float, default=0.60)
    parser.add_argument("--draft-slots", default="0,1")
    parser.add_argument("--draft-start", choices=("after-prepare", "after-result"), default="after-prepare",
                        help="keep the baseline schedule or start Draft independently of Bank preparation")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=48)
    parser.add_argument("--request-capacity-blocks", type=int, default=8)
    parser.add_argument("--target-memory-fraction", type=float, default=0.65)
    parser.add_argument("--copy-poll-us", type=float, default=100)
    parser.add_argument("--owner-idle-us", type=float, default=100)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-overlap", action="store_true")
    args = parser.parse_args(argv)
    args.draft_slots = tuple(int(v) for v in args.draft_slots.split(",") if v)
    devices = tuple(int(v) for v in args.target_devices.split(","))
    all_devices = devices + (() if args.draft_model is None else (args.draft_device,))
    if min(all_devices) < 0 or len(all_devices) != len(set(all_devices)):
        raise ValueError("Target devices and Draft device must be distinct non-negative ids")
    if not 2 <= args.rounds <= 128 or not 0 <= args.warmup_rounds < args.rounds:
        raise ValueError("use 2..128 rounds and fewer warmup rounds")
    if not 1 <= args.prompt_tokens <= 512 or not 1 <= args.request_capacity_blocks <= 64:
        raise ValueError("use 1..512 prompt tokens and 1..64 capacity blocks per request")
    required = (args.prompt_tokens + args.rounds * 3 + 15) // 16
    if required > args.request_capacity_blocks:
        raise ValueError(f"request capacity too small; need {required} blocks")
    if not 1 <= args.draft_workers <= 2 or any(s not in (0, 1) for s in args.draft_slots):
        raise ValueError("use 1..2 Draft workers and slots 0 and/or 1")
    for value in (args.copy_poll_us, args.owner_idle_us, args.timeout):
        if not math.isfinite(value) or value < 0:
            raise ValueError("timing settings must be finite and non-negative")
    if args.timeout == 0:
        raise ValueError("timeout must be positive")
    for value in (args.target_memory_fraction, args.draft_memory_fraction):
        if not 0 < value <= 1:
            raise ValueError("memory fraction must be in (0, 1]")
    if args.require_overlap and (not args.draft_model or not args.draft_slots):
        raise ValueError("four-way overlap requires a Draft model and at least one Draft slot")
    return args


def args_from_env(env, output):
    """Pytest uses the CLI parser, so newly added defaults cannot drift."""
    argv = ["--target-model", env["STARSD_NEXT_TARGET_MODEL_PATH"], "--output", str(output),
            "--rounds", "4", "--warmup-rounds", "1", "--prompt-tokens", "31"]
    names = {
        "TARGET_DEVICES": "target-devices", "DRAFT_MODEL_PATH": "draft-model",
        "DRAFT_DEVICE": "draft-device", "DRAFT_WORKERS": "draft-workers",
        "DRAFT_MEMORY_FRACTION": "draft-memory-fraction", "DRAFT_SLOTS": "draft-slots",
        "WP07_DRAFT_START": "draft-start",
        "WP07_COPY_POLL_US": "copy-poll-us", "WP07_OWNER_IDLE_US": "owner-idle-us",
        "WP07_ROUNDS": "rounds", "WP07_REQUEST_CAPACITY_BLOCKS": "request-capacity-blocks",
    }
    for name, option in names.items():
        if "STARSD_NEXT_" + name in env:
            argv.extend(("--" + option, env["STARSD_NEXT_" + name]))
    if env.get("STARSD_NEXT_REQUIRE_OVERLAP") == "1":
        argv.append("--require-overlap")
    return parse_args(argv)
