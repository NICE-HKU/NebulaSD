"""Engine-to-DMA imports; one completion slot and ordered KV reuse per Bank.

The scheduler owns source readiness. DMA retains each imported Bank until its
export completes. Control owns metadata rows and joins readiness before compute."""
from multiprocessing.connection import wait
from time import perf_counter_ns

from nebulasd.kv.transfer import CopyPlan, CopyRegion, CopyReceipt
from .diagnostics import ProfiledReceipt


def import_regions(work, blocks_per_bank, sources, block_size=16):
    """Only absolute host/GPU block offsets and lengths cross the fast path."""
    regions = []
    for row, source in zip(work.rows, sources):
        if row.source is None:
            continue
        blocks = source['valid_blocks'] if work.operation.name.startswith('DRAFT') else (
            source['logical_kv_len'] + block_size - 1) // block_size
        regions.append((work.bank_id * blocks_per_bank + row.destination_offset,
                        row.host_offset, blocks))
    return tuple(regions)


def encode(seq, queued, regions):
    from struct import pack
    return pack('<QQ', seq, queued) + b''.join(pack('<QQQ', *r) for r in regions)


def decode(raw):
    from struct import unpack_from, iter_unpack
    return (*unpack_from('<QQ', raw), tuple(iter_unpack('<QQQ', raw[16:])))


def publish(slot, seq, queued, receipt, wake):
    with slot.get_lock():
        slot[:] = (seq, queued, receipt.submitted_ns, receipt.completed_ns,
                   round(receipt.duration_ms * 1e6), getattr(receipt, 'cpu_ns', 0))
    wake.set()


def read(slot, seq):
    with slot.get_lock():
        values = tuple(slot[:])
    if values[0] != seq:
        return None
    _, queued, started, ended, duration, cpu = values
    return queued, ProfiledReceipt(started, ended, duration / 1e6, cpu_ns=cpu)


def serve(imports, exports, slot, wake, execute, extent):
    """No execution round trip between an export and the next queued import."""
    occupied = False
    while True:
        if occupied:
            plan = exports.recv()
            if plan is None:
                return
            if plan == 'RELEASE':
                now = perf_counter_ns()
                receipt = ProfiledReceipt(now, now, 0.0)
            else:
                if isinstance(plan, tuple) and plan[0] == 'EXPORT':
                    plan = CopyPlan('D2H', tuple(CopyRegion(extent, gpu, host, count)
                                              for gpu, host, count in plan[1]))
                receipt = execute(plan)
            occupied = False
            exports.send(receipt)
        else:
            available = wait((imports, exports))
            if imports not in available:
                if exports.recv() is None:
                    return
                raise RuntimeError('export without an owned Bank')
            seq, queued, regions = decode(imports.recv_bytes())
            if not regions:
                now = perf_counter_ns()
                receipt = CopyReceipt(now, now, 0.0)
            else:
                plan = CopyPlan('H2D', tuple(CopyRegion(extent, gpu, host, count)
                                           for gpu, host, count in regions))
                receipt = execute(plan)
            occupied = True
            publish(slot, seq, queued, receipt, wake)
