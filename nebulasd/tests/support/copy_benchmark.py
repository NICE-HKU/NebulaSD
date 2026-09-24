"""Identical buffers/ranges, alternating A/B order, byte checks outside timing."""

from nebulasd.kv.arena import SharedHostKVArena, HostKVWriteLease
from nebulasd.kv.registration_pool import HostRegistrationPool
from nebulasd.kv.cuda_transfer import CudaCopyBackend
from nebulasd.kv.transfer import CopyExecutor, CopyPlan, CopyRegion
from nebulasd.observability.copy_timing import percentiles
from support.legacy_copy_backend import LegacyCopyBackend


def run_case(shape, blocks, batch, args, *, on_row=None):
    import torch
    capacity = blocks + 2  # A guard block on each side, per request and plane.
    k = torch.empty((batch * capacity, *shape), dtype=torch.float16, device=args.device)
    v = torch.empty_like(k)
    arena = SharedHostKVArena.create(total_blocks=batch * capacity,
        block_bytes=k[0].numel() * k.element_size(), dtype=str(k.dtype), kv_block_shape=tuple(shape))
    pool = HostRegistrationPool()
    backends = dict(direct=CudaCopyBackend(arena=arena, k_cache=k, v_cache=v, registration_pool=pool),
                    old_staged=LegacyCopyBackend(arena, k, v, pool))
    backends = {name: backend for name,backend in backends.items() if name in getattr(args,'paths',('direct','old_staged'))}
    executors = {name: CopyExecutor(backend, poll_interval_s=args.copy_poll_us / 1e6)
                 for name, backend in backends.items()}
    extents = [arena.make_extent(request_slot=i, request_epoch=1, host_slot=i,
        host_slot_generation=1, writer_lease_generation=1, offset_blocks=i * capacity,
        capacity_blocks=capacity).next_version(capacity) for i in range(batch)]
    rows = []
    try:
        for direction in ("D2H", "H2D"):
            plan = CopyPlan(direction, tuple(CopyRegion(e, i * capacity + 1, 1, blocks)
                                             for i, e in enumerate(extents)))
            for iteration in range(args.warmup + args.samples):
                order = ("direct", "old_staged") if iteration % 2 == 0 else ("old_staged", "direct")
                for name in order:
                    if name not in backends:
                        continue
                    prepare_buffers(arena, extents, k, v, blocks, direction, iteration)
                    torch.cuda.current_stream(args.device).synchronize()  # Fixture setup only.
                    receipt = executors[name].submit(plan).result(timeout=args.timeout)
                    check_buffers(arena, extents, k, v, blocks, direction, iteration)
                    rows.append(dict(path=name, direction=direction, iteration=iteration,
                        measured=iteration >= args.warmup, blocks=blocks, batch=batch,
                        copy_bytes=2 * blocks * batch * arena.descriptor.block_bytes,
                        cuda_interval_ms=receipt.duration_ms,
                        launch_call_ms=(receipt.launch_returned_ns - receipt.submitted_ns) / 1e6,
                        submit_to_last_launch_return_ms=(receipt.launch_returned_ns - receipt.submitted_ns) / 1e6,
                        enqueued_ns=receipt.enqueued_ns, observed_ns=receipt.completed_ns,
                        executor_queue_ms=(receipt.submitted_ns - receipt.enqueued_ns) / 1e6,
                        enqueue_to_observed_ms=(receipt.completed_ns - receipt.enqueued_ns) / 1e6,
                        completion_detection_window_ms=(receipt.completed_ns - receipt.last_pending_ns) / 1e6))
                    if on_row is not None:
                        on_row(rows[-1])
    finally:
        errors = []
        for executor in executors.values():
            try:
                executor.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"benchmark retirement failed; arena retained: {errors}") from errors[0]
        arena.close()
        arena.unlink()
    return rows


def pattern(plane, slot, iteration):
    return 1 + (plane * 79 + slot * 7 + iteration) % 180


def prepare_buffers(arena, extents, k, v, blocks, direction, iteration):
    import torch
    for slot, extent in enumerate(extents):
        size = arena.descriptor.block_bytes
        begin = slot * extent.capacity_blocks
        lease = HostKVWriteLease.for_extent(extent, dirty_begin_block=0, dirty_block_count=extent.capacity_blocks)
        view = arena.writer_view(extent, lease=lease)
        try:
            for plane, (cache, host) in enumerate(((k, view.k), (v, view.v))):
                cache[begin:begin + extent.capacity_blocks].view(torch.uint8).fill_(204)
                host[:] = bytes([221]) * len(host)
                value = pattern(plane, slot, iteration)
                if direction == "D2H":
                    cache[begin + 1:begin + blocks + 1].view(torch.uint8).fill_(value)
                else:
                    host[size:size * (blocks + 1)] = bytes([value]) * (size * blocks)
        finally:
            view.release()


def check_buffers(arena, extents, k, v, blocks, direction, iteration):
    import torch
    size = arena.descriptor.block_bytes
    for slot, extent in enumerate(extents):
        begin = slot * extent.capacity_blocks
        hosts = arena.read_kv(extent, begin_block=0, block_count=extent.capacity_blocks)
        for plane, (cache, host) in enumerate(zip((k, v), hosts)):
            gpu = cache[begin:begin + extent.capacity_blocks].view(torch.uint8).cpu().numpy().tobytes()
            value = bytes([pattern(plane, slot, iteration)]) * size * blocks
            assert gpu[size:-size] == host[size:-size] == value, "copy payload mismatch"
            assert host[:size] == host[-size:] == bytes([221]) * size, "HostKV guard overwritten"
            assert gpu[:size] == gpu[-size:] == bytes([204]) * size, "GPU guard overwritten"


def summarize(rows):
    groups = {}
    for row in rows:
        if not row["measured"]:
            continue
        key = (row["blocks"], row["batch"], row["direction"], row["path"])
        groups.setdefault(key, []).append(row)
    report = []
    for (blocks, batch, direction, path), samples in groups.items():
        metrics = {name: percentiles([r[name] for r in samples])
                   for name in samples[0] if name.endswith("_ms")}
        report.append(dict(blocks=blocks, batch=batch, direction=direction, path=path,
            samples=len(samples), copy_bytes=samples[0]["copy_bytes"], metrics=metrics,
            effective_payload_gbps_p50=samples[0]["copy_bytes"] / metrics["cuda_interval_ms"]["p50_ms"] / 1e6))
    return report


def compare_paths(summary):
    """Ratios > 1 mean the direct path took longer; no hidden pass threshold."""
    groups = {}
    for row in summary:
        groups.setdefault((row["blocks"], row["batch"], row["direction"]), {})[row["path"]] = row
    comparisons = []
    for (blocks, batch, direction), paths in groups.items():
        ratios = {}
        for name, metric in paths["direct"]["metrics"].items():
            old = paths["old_staged"]["metrics"][name]
            ratios[name] = {quantile.removesuffix("_ms"): metric[quantile] / old[quantile] if old[quantile] else None
                            for quantile in ("p50_ms", "p95_ms", "p99_ms")}
        comparisons.append(dict(blocks=blocks, batch=batch, direction=direction,
                                direct_over_old_duration_ratio=ratios))
    return comparisons
