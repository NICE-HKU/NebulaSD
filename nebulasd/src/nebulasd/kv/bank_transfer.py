"""Two independent single-outstanding DMA owners; no direction queues."""
from .transfer import CopyExecutor


def bank_key(batch):
    if isinstance(batch, tuple):
        key = batch
    elif hasattr(batch, 'standby_bank_id'):
        key = batch.standby_bank_id, batch.next_bank_epoch, batch.batch_seq
    else:
        key = batch.bank_id, batch.bank_epoch, batch.batch_seq
    if len(key) != 3 or key[0] not in (0, 1):
        raise ValueError('copy requires Bank 0 or 1')
    return key


class BankCopyExecutor:
    def __init__(self, backends, *, poll_interval_s=0.0001):
        if len(backends) != 2 or backends[0] is backends[1]:
            raise ValueError('two independently owned copy backends required')
        self.executors = tuple(CopyExecutor(b, poll_interval_s=poll_interval_s) for b in backends)
        self._closed = False
        self._close_error = None

    def check_health(self):
        if self._closed:
            raise RuntimeError('Bank copy executors closed')
        for executor in self.executors:
            future = executor._future
            if future is not None and future.done():
                future.result()  # A failure in either Bank forbids new work.

    def submit_batch(self, plan, batch):
        self.check_health()
        return self.executors[bank_key(batch)[0]].submit(plan)

    def close(self):
        if self._close_error is not None:
            raise self._close_error
        if self._closed:
            return
        self._closed = True
        error = None
        # Join BOTH streams even when one failed. Callers retain pins/mappings
        # if any cleanup fails; the shared pool alone owns final unregister.
        for executor in self.executors:
            try:
                executor.close()
                if executor._future is not None:
                    executor._future.result()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            self._close_error = error
            raise error
