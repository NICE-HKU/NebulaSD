"""Engine-owned lifetime of two sibling Target processes and their local channels."""
import ctypes
from contextlib import ExitStack
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from queue import Empty, Full
import signal
from threading import Event, Thread, Lock
from time import monotonic, sleep

from nebulasd.workers.channel import LocalChannel
from .process import run_target
from .control import run_control


def _child(entry, args, parent_pid):
    # Linux PDEATHSIG closes the orphan window even if Engine is killed rather
    # than unwinding its context manager. Recheck after installing the signal.
    libc = ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0) != 0:
        raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if os.getppid() != parent_pid:
        raise RuntimeError('Target parent exited during spawn')
    entry(*args)


class TargetProcesses:
    """Own only process/transport lifetime; no Publisher or GPU state in Engine.

    execution_entry is an explicit process-protocol test seam, never a runtime
    fallback. The default always loads the real Target execution process.
    """
    def __init__(self, options, *, context=None, execution_entry=run_target, control_entry=run_control, role_name="target"):
        self.context = context or mp.get_context('spawn')
        self.channels = []
        self.pending_work = set()
        self.processes = []
        self.failure = None
        self.stopping = Event()
        self.monitor_done = Event()
        self.closed = False
        self.cleanup_lock = Lock()
        self.exitcodes = {}
        self.monitor = None
        self.import_channels = []
        self.dma_channels = []
        self.metadata_channels = []
        self.direct_imports = ("global_registry" in options and
            execution_entry.__name__ in ("run_target", "run_draft") and
            (os.environ.get("STARSD_DMA_MODE") or "process") == "process")
        try:
            for _ in range(3):
                self.channels.append(LocalChannel())
            self.commands,execution_commands,execution_results = self.channels
            self.wake,self.execution_wake = self.context.Event(),self.context.Event()
            self.control_ready,self.execution_ready = self.context.Event(),self.context.Event()
            self.execution_stopped = self.context.Event()
            self.dma_pid = self.context.Value('q', 0, lock=False)
            options = options | dict(dma_pid=self.dma_pid)
            if self.direct_imports:
                self.import_channels = [self.context.Pipe(duplex=False) for _ in range(2)]
                self.import_slots = [self.context.Array('Q', 6) for _ in range(2)]
                options = options | dict(direct_imports=True,
                    import_connections=[pair[0] for pair in self.import_channels],
                    import_slots=self.import_slots,
                    import_wake=self.execution_wake)
            if (os.environ.get('STARSD_DMA_MODE') or 'process') == 'process' and execution_entry.__name__ in ('run_target', 'run_draft'):
                self.dma_channels = [self.context.Pipe() for _ in range(2)]
                self.metadata_channels = [self.context.Pipe() for _ in range(2)]
                self.compute_clock = self.context.Array('Q', 3)
                options = options | dict(isolated_compute=True, isolated_role=role_name,
                    dma_connections=[p[0] for p in self.dma_channels],
                    dma_child_connections=[p[1] for p in self.dma_channels],
                    metadata_connections=[p[0] for p in self.metadata_channels],
                    metadata_child_connections=[p[1] for p in self.metadata_channels],
                    compute_clock=self.compute_clock)
                if self.direct_imports:
                    options['import_wake'] = self.wake
            pid = os.getpid()
            self.execution = self.context.Process(name=f'{role_name}-execution',target=_child,args=(execution_entry,
                (options,execution_commands.descriptor,execution_results.descriptor,self.execution_wake,
                 self.execution_ready,self.wake),pid))
            self.control = self.context.Process(name=f'{role_name}-control',target=_child,args=(control_entry,
                (options,self.commands.descriptor,execution_commands.descriptor,
                 execution_results.descriptor,self.wake,self.execution_wake,self.control_ready,
                 self.execution_stopped),pid))
            for process in (self.execution,self.control):
                process.start()
                self.processes.append(process)
            self._pids = dict(engine=pid,control=self.control.pid,execution=self.execution.pid)
            self.monitor = Thread(target=self._monitor,name='target-supervisor',daemon=True)
            self.monitor.start()
        except BaseException:
            self._terminate()
            try:
                self._unlink()
            finally:
                for process in self.processes:
                    process.close()
            raise

    @property
    def pids(self):
        pids = dict(self._pids)
        if self.dma_pid.value:
            pids['dma'] = self.dma_pid.value
        return pids

    def _monitor(self):
        pending = {p.sentinel:p for p in self.processes}
        while pending and not self.monitor_done.is_set():
            for sentinel in wait(tuple(pending),timeout=.05):
                process = pending.pop(sentinel)
                process.join()
                self.exitcodes[process.name] = process.exitcode
                if process.exitcode != 0 or not self.stopping.is_set():
                    self.failure = f'{process.name} exited {process.exitcode}'
                    self._terminate()
                    return
                if process is self.execution:
                    self.execution_stopped.set()
                    self.wake.set()

    def check(self):
        if self.closed:
            raise RuntimeError('Target process group is closed')
        if self.failure is not None:
            raise RuntimeError(self.failure)

    def wait_ready(self, timeout=240):
        deadline = monotonic()+timeout
        try:
            while not (self.control_ready.is_set() and self.execution_ready.is_set()):
                self.check()
                if monotonic() >= deadline:
                    raise TimeoutError('Target process startup timed out')
                self.control_ready.wait(.02) if not self.control_ready.is_set() else self.execution_ready.wait(.02)
            self.check()
        except BaseException:
            self.close(timeout=0)
            raise

    def submit_import(self, work, regions):
        from time import perf_counter_ns
        from nebulasd.workers.direct_import import encode
        self.import_channels[work.bank_id][1].send_bytes(encode(work.work_seq, perf_counter_ns(), regions or ()))

    def submit(self, work, *, import_plan=None):
        self.check()
        if self.stopping.is_set():
            raise RuntimeError('Target is draining')
        from nebulasd.workers.work import MAX_OUTSTANDING
        if work.work_seq in self.pending_work:
            raise ValueError('duplicate outstanding WORK')
        if len(self.pending_work) >= MAX_OUTSTANDING:
            raise Full
        if not self.commands.ring.can_push():
            raise Full
        if self.direct_imports:
            self.submit_import(work, import_plan)
        self.commands.put_nowait(('WORK',work.to_bytes()))
        self.pending_work.add(work.work_seq)
        self.wake.set()

    def retire(self, seq):
        """Engine calls only after reading the authoritative shared completion."""
        self.pending_work.discard(seq)

    def stop(self, *, shutdown=False):
        self.check()
        already_stopping = self.stopping.is_set()
        self.stopping.set()
        try:
            self.commands.put_nowait(('SHUTDOWN' if shutdown else 'DRAIN',None))
        except Full:
            if not already_stopping:
                self.stopping.clear()
            raise
        self.wake.set()

    def join(self, timeout=30):
        # Exactly one thread reaps child statuses. Concurrent Process.join/poll
        # calls can race waitpid and report a transient None exit code.
        self.monitor.join(timeout)
        self.check()
        if self.monitor.is_alive():
            raise TimeoutError('Target process drain timed out')


    def _terminate(self):
        with self.cleanup_lock:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join(2)
                if process.is_alive():
                    process.kill()
                    process.join()
                self.exitcodes[process.name] = process.exitcode

    def _unlink(self):
        for pair in (*self.import_channels, *self.dma_channels, *self.metadata_channels):
            for connection in pair:
                connection.close()
        self.import_channels.clear()
        self.dma_channels.clear()
        self.metadata_channels.clear()
        channels,self.channels = self.channels,[]
        with ExitStack() as stack:
            for channel in channels:
                stack.callback(channel.close,unlink=True)

    def close(self, timeout=30):
        if self.closed:
            return
        try:
            if not self.failure and self.monitor is not None and self.monitor.is_alive() and timeout:
                deadline = monotonic()+timeout
                while True:
                    try:
                        self.stop(shutdown=True)
                        break
                    except Full:
                        if monotonic() >= deadline:
                            break
                        sleep(.001)
                    except RuntimeError:
                        break
                # Control retires from shared completion without Engine consumption.
                while monotonic()<deadline and self.monitor.is_alive() and not self.failure:
                    sleep(.001)
        finally:
            self.monitor_done.set()
            if self.monitor is not None:
                self.monitor.join(3)
            self._terminate()
            try:
                self._unlink()  # Only after both processes have been joined.
            finally:
                for process in self.processes:
                    process.close()
                self.closed = True

    def __enter__(self):
        self.wait_ready()
        return self

    def __exit__(self, *error):
        self.close()
