"""Linux eventfd (macOS datagram fallback) carries wake hints, never payloads."""

import select
import socket
import errno
import os
from multiprocessing.reduction import DupFd


class ProcessDoorbell:
    def __init__(self, receiver, sender):
        self.receiver, self.sender = receiver, sender
        self.receiver.setblocking(False)
        self.sender.setblocking(False)

    @classmethod
    def create(cls):
        if hasattr(os, "eventfd"):
            return EventFDDoorbell(os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC))
        return cls(*socket.socketpair(type=socket.SOCK_DGRAM))

    def ring(self):
        try:
            self.sender.send(b"\1")
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.ENOBUFS):
                raise
            # Hints may coalesce/drop under OS pressure; bounded idle rechecks
            # still observe the authoritative rings/Tables without data loss.

    def drain(self):
        while True:
            try:
                if not self.receiver.recv(4096):
                    return
            except BlockingIOError:
                return

    def wait(self, timeout):
        select.select([self.receiver], [], [], timeout)
        self.drain()

    def close(self):
        self.receiver.close()
        self.sender.close()


class EventFDDoorbell:
    """Spawn transfers an FD duplicate, preserving one shared kernel counter."""
    def __init__(self, fd):
        self.fd = fd
        self.receiver = self  # Same selectable interface as the macOS receiver.

    def fileno(self):
        return self.fd

    def __getstate__(self):
        return DupFd(self.fd)

    def __setstate__(self, duplicate):
        self.__init__(duplicate.detach())

    def ring(self):
        try:
            os.eventfd_write(self.fd, 1)
        except BlockingIOError:
            pass  # Counter already contains an unread wakeup.

    def drain(self):
        try:
            os.eventfd_read(self.fd)
        except BlockingIOError:
            pass

    def wait(self, timeout):
        select.select([self], [], [], timeout)
        self.drain()

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
