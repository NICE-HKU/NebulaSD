"""Native CommandRing smoke tests."""

from __future__ import annotations

import pathlib
import subprocess


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_native_command_ring_push_pop_full_wrap_and_stale_generation(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "ring_harness.cpp"
    binary = tmp_path / "ring_harness"
    source.write_text(
        r'''
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "nebulasd/csrc/control_plane/ring.cpp"

int main() {
  using namespace nebulasd::control_plane;

  CommandHeader entries[2] = {};
  std::atomic<std::uint64_t> head{0};
  std::atomic<std::uint64_t> tail{0};
  CommandRingView ring(entries, 2, &head, &tail);

  CommandHeader out{};
  CommandHeader h0{0, 9, 1, 0, 1, 0};
  CommandHeader h1{1, 9, 1, 1, 1, 0};
  CommandHeader h2{2, 9, 1, 2, 1, 0};

  assert(ring.TryPop(9, &out) == CommandRingResult::kEmpty);
  assert(ring.TryPush(h0) == CommandRingResult::kOk);
  assert(ring.TryPush(h1) == CommandRingResult::kOk);
  assert(ring.TryPush(h2) == CommandRingResult::kFull);
  assert(ring.TryPop(10, &out) == CommandRingResult::kStaleWorkerGeneration);
  assert(ring.TryPop(9, &out) == CommandRingResult::kOk);
  assert(out.command_seq == 0);
  assert(ring.TryPush(h2) == CommandRingResult::kOk);
  assert(ring.TryPop(9, &out) == CommandRingResult::kOk);
  assert(out.command_seq == 1);
  assert(ring.TryPop(9, &out) == CommandRingResult::kOk);
  assert(out.command_seq == 2);
  assert(ring.TryPop(9, &out) == CommandRingResult::kEmpty);

  struct SharedRing {
    CommandHeader entries[2];
    std::atomic<std::uint64_t> head;
    std::atomic<std::uint64_t> tail;
  };
  void* mem = mmap(nullptr, sizeof(SharedRing), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  assert(mem != MAP_FAILED);
  auto* shared = static_cast<SharedRing*>(mem);
  new (&shared->head) std::atomic<std::uint64_t>(0);
  new (&shared->tail) std::atomic<std::uint64_t>(0);
  shared->entries[0] = {};
  shared->entries[1] = {};

  pid_t pid = fork();
  assert(pid >= 0);
  if (pid == 0) {
    CommandRingView child_ring(shared->entries, 2, &shared->head, &shared->tail);
    CommandHeader child_header{10, 12, 1, 64, 16, 0};
    assert(child_ring.TryPush(child_header) == CommandRingResult::kOk);
    _exit(0);
  }
  int status = 0;
  assert(waitpid(pid, &status, 0) == pid);
  assert(WIFEXITED(status));
  assert(WEXITSTATUS(status) == 0);
  CommandRingView parent_ring(shared->entries, 2, &shared->head, &shared->tail);
  assert(parent_ring.TryPop(12, &out) == CommandRingResult::kOk);
  assert(out.command_seq == 10);
  assert(munmap(mem, sizeof(SharedRing)) == 0);
  return 0;
}
''',
        encoding="utf-8",
    )
    subprocess.run(
        ["c++", "-std=c++17", "-I.", str(source), "-o", str(binary)],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run([str(binary)], cwd=REPO_ROOT, check=True)
