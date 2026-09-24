"""Two independent chunk executors and CUDA streams supplied at bootstrap."""
from nebulasd.kv.transfer import CopyExecutor


class DMA:
    def __init__(self, backends, *, poll_interval_s=0.0001, h2d_chunk_bytes=None, h2d_group_size=None, profile=False):
        if len(backends) != 2 or backends[0] is backends[1]:
            raise ValueError('two independent DMA backends required')
        for backend in backends:
            backend.initialize()
        executor = CopyExecutor
        if profile:
            from .diagnostics import ProfiledCopyExecutor
            executor = ProfiledCopyExecutor
        self.executors = tuple(executor(b, poll_interval_s=poll_interval_s,
            h2d_chunk_bytes=h2d_chunk_bytes, h2d_group_size=h2d_group_size) for b in backends)

    def submit(self, bank_id, plan):
        return self.executors[bank_id].submit(plan)

    def close(self):
        # Cold shutdown only, after drain. Never called from an owner tick.
        for executor in self.executors:
            executor.close()
