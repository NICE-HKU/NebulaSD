"""Execution-owned KV process: CUDA IPC at startup, host-only plans thereafter.

The execution process retains both cache allocations until this child exits.
Each bank has one ordered request/reply connection and at most one outstanding
whole-plan Future. Only the child submits memcpy or waits for CUDA completion.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from threading import Thread

from nebulasd.kv.transfer import CopyPlan, CopyRegion, HostCompletedFence


def transport_plan(plan):
    if any(not isinstance(dep, HostCompletedFence) for dep in plan.dependencies):
        raise ValueError('DMA process requires host-completed dependencies')
    return replace(plan, dependencies=())


def bounded_regions(plan, block_bytes, budget):
    """Split oversized regions too; preserve every block, extent and round ID."""
    if plan.direction != 'H2D' or not budget:
        return plan
    blocks = budget // (2 * block_bytes)
    if blocks < 1:
        raise ValueError('H2D window must fit one K/V block')
    regions, rounds = [], []
    for i, region in enumerate(plan.regions):
        for offset in range(0, max(1, region.block_count), blocks):
            count = min(blocks, region.block_count - offset)
            regions.append(CopyRegion(region.extent, region.gpu_begin_block + offset,
                                      region.host_begin_block + offset, count))
            if plan.round_ids:
                rounds.append(plan.round_ids[i])
    return replace(plan, regions=tuple(regions), round_ids=tuple(rounds))


def run_dma(host_descriptor, connections, ready, options):
    from contextlib import ExitStack
    from nebulasd.kv.arena import SharedHostKVArena
    from nebulasd.kv.cuda_transfer import CudaCopyBackend
    from .dma import DMA
    # Receive after spawn: never retain imported tensors in Process._args,
    # whose lifetime extends into Python module teardown.
    storage = ready.recv()
    k_cache, v_cache = storage[:2]
    block_table = storage[2] if len(storage) == 3 else None
    storage = None
    with ExitStack() as stack:
        host = SharedHostKVArena.attach(host_descriptor)
        stack.callback(host.close)
        backend = CudaCopyBackend(arena=host, k_cache=k_cache, v_cache=v_cache)
        backends = [backend, backend.fork()]
        if options.get('diagnostic_copy_delay') is not None:
            from .diagnostics import DelayedCopyBackend
            bank, direction, seconds = options['diagnostic_copy_delay']
            backends[bank] = DelayedCopyBackend(backends[bank], direction, seconds)
        for b in backends:
            stack.callback(b.close)
        budget = int(os.environ.get('STARSD_H2D_CHUNK_BYTES') or (32 << 20))
        group = int(os.environ.get('STARSD_H2D_GROUP_SIZE') or '1')
        dma = DMA(backends, h2d_chunk_bytes=budget, h2d_group_size=group,
                  profile=options.get('profile', False))
        stack.callback(dma.close)

        def serve(bank):
            connection = connections[bank]
            try:
                if options.get('direct_imports'):
                    from .direct_import import serve as serve_imports
                    def execute(plan):
                        plan = bounded_regions(plan, host.descriptor.block_bytes, budget)
                        return dma.submit(bank, plan).result()
                    extent = host.make_extent(request_slot=0, request_epoch=1, host_slot=0,
                        host_slot_generation=1, writer_lease_generation=1, offset_blocks=0,
                        capacity_blocks=host.descriptor.total_blocks)
                    serve_imports(options['import_connections'][bank], connection,
                        options['import_slots'][bank], options['import_wake'], execute, extent)
                    return
                while True:
                    plan = connection.recv()
                    if plan is None:
                        return
                    if plan.dependencies:
                        raise ValueError('CUDA dependency crossed DMA transport')
                    plan = bounded_regions(plan, host.descriptor.block_bytes, budget)
                    receipt = dma.submit(bank, plan).result()
                    connection.send(receipt)
            finally:
                connection.close()

        def metadata(bank):
            # A separate channel never puts Metadata ahead of queued H2D.
            import torch
            connection = options['metadata_child_connections'][bank]
            torch.cuda.set_device(block_table.device)
            stream = torch.cuda.Stream(device=block_table.device)
            capacity = options['blocks_per_bank']
            indices = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
            values = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
            gpu_indices = torch.empty_like(indices, device=block_table.device)
            gpu_values = torch.empty_like(values, device=block_table.device)
            import numpy as np
            ix, val = indices.numpy(), values.numpy()
            try:
                while True:
                    layout = connection.recv()
                    if layout is None:
                        return
                    n = 0
                    for row, begin, count in zip(layout.rows, layout.offsets, layout.capacities):
                        ix[n:n+count] = np.arange(row * block_table.shape[1], row * block_table.shape[1] + count)
                        val[n:n+count] = np.arange(begin, begin + count)
                        n += count
                    with torch.cuda.stream(stream):
                        gpu_indices[:n].copy_(indices[:n], non_blocking=True)
                        gpu_values[:n].copy_(values[:n], non_blocking=True)
                        block_table.view(-1).index_copy_(0, gpu_indices[:n], gpu_values[:n])
                        done = torch.cuda.Event()
                        done.record(stream)
                    done.synchronize()
                    connection.send(HostCompletedFence())
            finally:
                connection.close()

        # Independent bank loops: one sleeping H2D event never prevents the
        # other bank from receiving/submitting D2H. No computation runs here.
        with ThreadPoolExecutor(4 if block_table is not None else 2, thread_name_prefix='dma-bank') as pool:
            jobs = [pool.submit(serve, bank) for bank in range(2)]
            if block_table is not None:
                jobs.extend(pool.submit(metadata, bank) for bank in range(2))
            ready.send(('READY', os.getpid()))
            ready.close()
            # A failed bank must crash the whole worker, not strand its peer.
            def fail_stop(future):
                if future.exception() is not None:
                    import traceback
                    traceback.print_exception(future.exception())
                    os._exit(1)
            for job in jobs:
                job.add_done_callback(fail_stop)
            for job in jobs:
                job.result()
    # Release imported storage before the execution owner may exit.
    backend = backends = dma = k_cache = v_cache = block_table = None


class ProcessDMA:
    def __init__(self, *, arena, k_cache, v_cache, options=None, wake=None, block_table=None):
        import torch.multiprocessing  # Registers CUDA tensor IPC reducers.
        from .target.service import _child
        options = {} if options is None else options
        context = mp.get_context('spawn')
        pairs = (list(zip(options['dma_connections'], options['dma_child_connections']))
                 if options.get('isolated_compute') else [context.Pipe() for _ in range(2)])
        self.metadata_connections = options.get('metadata_connections', ())
        self.connections = [p[0] for p in pairs]
        child_connections = [p[1] for p in pairs]
        ready, child_ready = context.Pipe()
        self._caches = (k_cache, v_cache)
        self.import_slots = options.get("import_slots")
        self.direct_imports = bool(options.get("direct_imports"))
        self._closed = False
        self._failure = None
        self._futures = [None, None]
        self._pool = ThreadPoolExecutor(2, thread_name_prefix='dma-reply')
        self.process = context.Process(name='kv-migration', target=_child,
            args=(run_dma, (arena.descriptor, child_connections,
                           child_ready, options), os.getpid()))
        try:
            self.process.start()
            for connection in child_connections:
                connection.close()
            child_ready.close()
            ready.send((k_cache, v_cache) if block_table is None else (k_cache, v_cache, block_table))
            if not wait((ready, self.process.sentinel), timeout=120) or not ready.poll():
                raise RuntimeError('KV process failed to initialize')
            kind, self.pid = ready.recv()
            if kind != 'READY':
                raise RuntimeError('invalid KV process startup reply')
            if options.get('dma_pid') is not None:
                options['dma_pid'].value = self.pid
            self._monitor = Thread(target=self._watch, args=(wake,), daemon=True,
                                   name='dma-supervisor')
            self._monitor.start()
        except BaseException:
            self._closed = True
            if self.process.pid is not None:
                self.process.terminate()
                self.process.join()
            for connection in self.connections + child_connections:
                connection.close()
            child_ready.close()
            self._pool.shutdown(wait=True)
            raise
        finally:
            ready.close()

    def _watch(self, wake):
        wait((self.process.sentinel,))
        if not self._closed:
            self._failure = RuntimeError('KV migration process exited unexpectedly')
            if wake is not None:
                wake.set()

    def check(self):
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError('KV migration process is closed')

    def import_receipt(self, bank_id, seq):
        from .direct_import import read
        return read(self.import_slots[bank_id], seq)

    def release_bank(self, bank_id):
        return self._exchange(bank_id, 'RELEASE')

    def submit(self, bank_id, plan):
        return self._exchange(bank_id, transport_plan(plan))

    def _exchange(self, bank_id, plan):
        self.check()
        if bank_id not in (0, 1):
            raise ValueError('invalid DMA bank')
        previous = self._futures[bank_id]
        if previous is not None:
            if not previous.done():
                raise RuntimeError('copy executor has an outstanding batch')
            previous.result()
        # Pickle only small host descriptors; tensor export happened at startup.
        # No owner-thread pipe send or CUDA API, even under continuous compute.
        def exchange():
            connection = self.connections[bank_id]
            connection.send(plan)
            return connection.recv()
        future = self._pool.submit(exchange)
        self._futures[bank_id] = future
        return future

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._pool.shutdown(wait=True)
            for connection in getattr(self, 'metadata_connections', ()):
                connection.send(None)
                connection.close()
            for connection in self.connections:
                connection.send(None)
            self.process.join(30)
            if self.process.is_alive() or self.process.exitcode != 0:
                raise RuntimeError('KV process did not close cleanly')
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            self._monitor.join()
            for connection in self.connections:
                connection.close()
            self.process.close()
            self._caches = None
