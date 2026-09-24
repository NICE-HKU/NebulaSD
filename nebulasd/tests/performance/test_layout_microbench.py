"""CPU-only ABI layout/read microbenchmarks."""

from __future__ import annotations

import statistics
import time

from nebulasd.table.layout import ENGINE_BLOCK, TARGET_COMPUTE_BLOCK


def _read_u64(buffer: bytearray, offset: int) -> int:
    return int.from_bytes(buffer[offset : offset + 8], "little")


def _write_u64(buffer: bytearray, offset: int, value: int) -> None:
    buffer[offset : offset + 8] = value.to_bytes(8, "little")


def _percentiles(samples: list[int]) -> dict[str, float]:
    ordered = sorted(samples)
    last = len(ordered) - 1
    return {
        "p50_ns": float(statistics.median(ordered)),
        "p95_ns": float(ordered[min(last, round(last * 0.95))]),
        "p99_ns": float(ordered[min(last, round(last * 0.99))]),
    }


def test_cpu_only_stable_read_microbench_outputs_percentiles() -> None:
    capacity = 4096
    batch_rows = 512
    warmup_batches = 20
    measured_batches = 200
    row_stride = TARGET_COMPUTE_BLOCK.row_stride
    table = bytearray(capacity * row_stride)
    publish_offset = TARGET_COMPUTE_BLOCK.field_offset("publish_seq")
    epoch_offset = TARGET_COMPUTE_BLOCK.field_offset("request_epoch")
    round_offset = TARGET_COMPUTE_BLOCK.field_offset("round_id")
    status_offset = TARGET_COMPUTE_BLOCK.field_offset("status")

    for row in range(capacity):
        base = row * row_stride
        _write_u64(table, base + epoch_offset, row + 1)
        _write_u64(table, base + round_offset, row % 17)
        table[base + status_offset : base + status_offset + 4] = (2).to_bytes(4, "little")
        _write_u64(table, base + publish_offset, row * 2)

    samples: list[int] = []
    checksum = 0
    for batch in range(warmup_batches + measured_batches):
        first_row = (batch * batch_rows) % capacity
        start = time.perf_counter_ns()
        for index in range(batch_rows):
            row = (first_row + index) % capacity
            base = row * row_stride
            before = _read_u64(table, base + publish_offset)
            epoch = _read_u64(table, base + epoch_offset)
            round_id = _read_u64(table, base + round_offset)
            after = _read_u64(table, base + publish_offset)
            if before == after:
                checksum ^= epoch ^ round_id ^ before
        elapsed = time.perf_counter_ns() - start
        if batch >= warmup_batches:
            samples.append(elapsed // batch_rows)

    percentiles = _percentiles(samples)
    print(
        "stable_read_microbench "
        f"p50_ns={percentiles['p50_ns']:.1f} "
        f"p95_ns={percentiles['p95_ns']:.1f} "
        f"p99_ns={percentiles['p99_ns']:.1f}"
    )
    assert checksum >= 0
    assert 0 < percentiles["p50_ns"] <= percentiles["p95_ns"] <= percentiles["p99_ns"]


def test_engine_block_stride_scan_microbench_outputs_percentiles() -> None:
    capacity = 4096
    batch_rows = 512
    warmup_batches = 20
    measured_batches = 200
    row_stride = ENGINE_BLOCK.row_stride
    table = bytearray(capacity * row_stride)
    epoch_offset = ENGINE_BLOCK.field_offset("request_epoch")
    lifecycle_offset = ENGINE_BLOCK.field_offset("lifecycle")

    for row in range(capacity):
        base = row * row_stride
        _write_u64(table, base + epoch_offset, row)
        table[base + lifecycle_offset : base + lifecycle_offset + 4] = (1).to_bytes(4, "little")

    samples: list[int] = []
    active = 0
    for batch in range(warmup_batches + measured_batches):
        first_row = (batch * batch_rows) % capacity
        start = time.perf_counter_ns()
        for index in range(batch_rows):
            row = (first_row + index) % capacity
            base = row * row_stride
            active += int.from_bytes(table[base + lifecycle_offset : base + lifecycle_offset + 4], "little")
            active ^= _read_u64(table, base + epoch_offset)
        elapsed = time.perf_counter_ns() - start
        if batch >= warmup_batches:
            samples.append(elapsed // batch_rows)

    percentiles = _percentiles(samples)
    print(
        "engine_stride_scan_microbench "
        f"p50_ns={percentiles['p50_ns']:.1f} "
        f"p95_ns={percentiles['p95_ns']:.1f} "
        f"p99_ns={percentiles['p99_ns']:.1f}"
    )
    assert active >= 0
    assert 0 < percentiles["p50_ns"] <= percentiles["p95_ns"] <= percentiles["p99_ns"]
