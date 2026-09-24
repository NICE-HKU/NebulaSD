"""Opt-in Linux placement of a new HostKV mapping, before CUDA registration.

No process-wide policy or CPU affinity changes. Default allocation remains lazy.
"""
import ctypes
import ctypes.util
import os
import re
import sys
from pathlib import Path


def validate_policy(policy, nodes):
    if policy not in ('default', 'bind', 'interleave'):
        raise ValueError('HostKV NUMA policy must be default, bind or interleave')
    if not isinstance(nodes, tuple) or any(type(n) is not int or n < 0 for n in nodes):
        raise ValueError('HostKV NUMA nodes must be a tuple of nonnegative integers')
    if len(set(nodes)) != len(nodes):
        raise ValueError('HostKV NUMA nodes must be unique')
    if (policy == 'default') != (not nodes):
        raise ValueError('default requires no NUMA nodes; bind/interleave require nodes')


def _ranges(text):
    result = set()
    for part in text.strip().split(','):
        ends = [int(x) for x in part.split('-')]
        result.update(range(ends[0], ends[-1] + 1))
    return result


def available_nodes():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('explicit HostKV NUMA placement requires Linux')
    status = Path('/proc/self/status').read_text()
    allowed = _ranges(re.search(r'^Mems_allowed_list:\s*(.+)$', status, re.M).group(1))
    return tuple(n for n in sorted(allowed)
                 if (p := Path(f'/sys/devices/system/node/node{n}/meminfo')).exists()
                 and int(re.search(r'MemTotal:\s+(\d+)', p.read_text()).group(1)) > 0)


def residency(address):
    """Observe this mapping's resident pages, without moving or faulting them."""
    for line in Path('/proc/self/numa_maps').read_text().splitlines():
        if int(line.split()[0], 16) == address:
            return {int(n): int(count) for n, count in re.findall(r'\bN(\d+)=(\d+)', line)}
    raise RuntimeError('HostKV mapping not found in numa_maps')


def _mbind(address, size, policy, nodes):
    name = ctypes.util.find_library('numa')
    if not name:
        raise RuntimeError('explicit HostKV NUMA placement requires libnuma')
    lib = ctypes.CDLL(name, use_errno=True)
    word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
    maxnode = max(nodes) + 1
    mask = (ctypes.c_ulong * ((maxnode + word_bits - 1) // word_bits))()
    for node in nodes:
        mask[node // word_bits] |= 1 << (node % word_bits)
    bind = lib.mbind
    bind.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                     ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_uint]
    bind.restype = ctypes.c_long
    # libnuma uses bitmap capacity + 1 for the syscall maxnode bound.
    if bind(address, size, 2 if policy == 'bind' else 3, mask, len(mask) * word_bits + 1, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, 'HostKV mbind: ' + os.strerror(err))


def place_new_mapping(shm, size, policy, nodes):
    validate_policy(policy, nodes)
    if policy == 'default':
        return None  # No Linux/libnuma dependency, prefault or policy mutation.
    allowed = available_nodes()
    if not set(nodes).issubset(allowed):
        raise ValueError(f'HostKV nodes {nodes} are not available memory nodes {allowed}')
    address = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
    _mbind(address, size, policy, nodes)
    # Reserve tmpfs backing before touching it: ENOSPC must raise, not SIGBUS.
    # The shared-memory inode range now has the requested NUMA policy.
    os.posix_fallocate(shm._fd, 0, size)
    ctypes.memset(address, 0, size)
    pages = residency(address)
    expected = (size + os.sysconf('SC_PAGE_SIZE') - 1) // os.sysconf('SC_PAGE_SIZE')
    if sum(pages.values()) != expected or not set(pages).issubset(nodes):
        raise RuntimeError(f'HostKV NUMA placement verification failed: {pages}, expected {expected} pages on {nodes}')
    if policy == 'interleave' and max(pages.get(n, 0) for n in nodes) - min(pages.get(n, 0) for n in nodes) > 1:
        raise RuntimeError(f'HostKV interleave is not balanced: {pages}')
    return dict(policy=policy, nodes=list(nodes), resident_pages=pages, total_bytes=size,
                page_size=os.sysconf('SC_PAGE_SIZE'), verified=True)
