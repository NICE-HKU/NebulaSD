"""Opt-in protected KV evidence. Diagnostic runs only; never performance runs."""
from dataclasses import asdict
from hashlib import sha256
import json
from math import prod
from pathlib import Path


def valid_spans(length, block_size, block_bytes, shape=()):
    """Offsets of valid tokens in canonical [block, layer, head, token, dim] KV."""
    if min(length, block_size, block_bytes) <= 0:
        raise ValueError('invalid KV dimensions')
    if shape and (len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape)
                  or shape[-2] != block_size
                  or block_bytes % prod(shape)):
        raise ValueError('unsupported canonical KV block shape')
    groups = prod(shape[:-2]) if shape else 1
    if block_bytes % (groups * block_size):
        raise ValueError('KV token byte stride is not integral')
    stride = block_bytes // (groups * block_size)
    for block in range((length + block_size - 1) // block_size):
        valid = min(block_size, length - block * block_size)
        yield block, tuple((group * block_size * stride, valid * stride) for group in range(groups))


def prefix_digest(read_block, *, length, block_size, block_bytes, shape=()):
    digest, size = sha256(), 0
    for block, spans in valid_spans(length, block_size, block_bytes, shape):
        raw = read_block(block)
        if len(raw) != block_bytes:
            raise ValueError('short KV block evidence')
        for offset, count in spans:
            digest.update(raw[offset:offset + count])
            size += count
    return dict(sha256=digest.hexdigest(), bytes=size)


def compare_prefix(left, right, *, length, block_size, block_bytes, shape=()):
    hashes, size, equal = (sha256(), sha256()), 0, True
    for block, spans in valid_spans(length, block_size, block_bytes, shape):
        a, b = left(block), right(block)
        if len(a) != block_bytes or len(b) != block_bytes:
            raise ValueError('short KV block evidence')
        for offset, count in spans:
            x, y = a[offset:offset+count], b[offset:offset+count]
            equal = equal and x == y  # Direct byte comparison, not only hashes.
            hashes[0].update(x)
            hashes[1].update(y)
            size += count
    return tuple(dict(sha256=h.hexdigest(), bytes=size) for h in hashes), equal


def install(worker, directory, *, backend, max_records=100000):
    """All reads occur after DMA but before lease/pin release or ready facts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'kv-worker-{worker.worker_id}.jsonl'
    lane, arena = worker.copy_lane, worker.copy_lane.facts.allocator.arena
    descriptor = arena.descriptor
    count = 0
    if lane.before_copy_release is not None:
        raise ValueError('KV evidence hook already installed')

    def capture(flight):
        nonlocal count
        if flight.lease is None or not flight.pins:
            raise AssertionError('KV evidence requires live GPU lease and HostKV pins')
        identities = ([e.snapshot.identity for e in flight.exports] if flight.plan.direction == 'D2H'
                      else [r.source for r in flight.command.requests])
        for identity, location, region in zip(identities, flight.batch.locations, flight.plan.regions, strict=True):
            if count >= max_records:
                raise RuntimeError('KV evidence capacity exhausted; cannot claim complete coverage')
            physical = lane.banks.physical_begin(location)
            def gpu_block(plane, index):
                memory = lane.executor.executors[flight.key[0]].backend
                if backend == 'cpu':
                    start = (physical + index) * descriptor.block_bytes
                    return bytes(getattr(memory, plane)[start:start + descriptor.block_bytes])
                import torch
                cache = getattr(memory, plane + '_cache')
                if not cache.is_cuda:
                    raise AssertionError('GPU evidence requires actual CUDA tensors')
                with torch.cuda.device(cache.device), torch.inference_mode():
                    # Bounded D2H diagnostic read; no extra GPU clone or session serialization.
                    cpu = cache[physical + index].detach().cpu().contiguous()
                    return bytes(cpu.view(torch.uint8).reshape(-1).tolist())
            options = dict(length=identity.logical_kv_len, block_size=identity.allocation.block_size,
                           block_bytes=descriptor.block_bytes, shape=descriptor.kv_block_shape)
            gpu, host, equal = {}, {}, True
            for plane, plane_index in (('k', 0), ('v', 1)):
                (gpu[plane], host[plane]), same = compare_prefix(lambda i: gpu_block(plane, i),
                    lambda i: arena.read_kv(region.extent, begin_block=i, block_count=1)[plane_index], **options)
                equal = equal and same
            row = dict(schema=1, backend=backend, worker=worker.worker_id, direction=flight.plan.direction,
                identity=asdict(identity), bank_id=flight.batch.bank_id, bank_epoch=flight.batch.bank_epoch,
                batch_seq=flight.batch.batch_seq, cancelled=lane._discard, gpu=gpu, host=host,
                copied_blocks=region.block_count, dirty_begin_block=region.host_begin_block,
                copy_bytes=region.block_count * descriptor.block_bytes * 2, equal=equal)
            with path.open('a') as out:
                out.write(json.dumps(row) + '\n')
            count += 1
            if not equal:
                raise AssertionError('Draft valid KV differs between GPU/CPU backend and HostKV')
    lane.before_copy_release = capture
