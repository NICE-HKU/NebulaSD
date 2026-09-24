#include <atomic>
#include <cstddef>
#include <cstdint>

namespace nebulasd::control_plane {

#pragma pack(push, 1)
struct CommandHeader {
  std::uint64_t command_seq;
  std::uint64_t worker_generation;
  std::uint32_t command_kind;
  std::uint64_t payload_offset;
  std::uint32_t payload_length;
  std::uint32_t flags;
};
#pragma pack(pop)

static_assert(sizeof(CommandHeader) == 36, "CommandHeader must match Python ABI");
static_assert(offsetof(CommandHeader, command_seq) == 0, "command_seq offset must match Python ABI");
static_assert(offsetof(CommandHeader, worker_generation) == 8, "worker_generation offset must match Python ABI");
static_assert(offsetof(CommandHeader, command_kind) == 16, "command_kind offset must match Python ABI");
static_assert(offsetof(CommandHeader, payload_offset) == 20, "payload_offset offset must match Python ABI");
static_assert(offsetof(CommandHeader, payload_length) == 28, "payload_length offset must match Python ABI");
static_assert(offsetof(CommandHeader, flags) == 32, "flags offset must match Python ABI");

enum class CommandRingResult : std::uint32_t {
  kOk = 0,
  kEmpty = 1,
  kFull = 2,
  kStaleWorkerGeneration = 3,
};

class CommandRingView {
 public:
  CommandRingView(CommandHeader* entries, std::uint64_t capacity, std::atomic<std::uint64_t>* head,
                  std::atomic<std::uint64_t>* tail)
      : entries_(entries), capacity_(capacity), head_(head), tail_(tail) {}

  CommandRingResult TryPush(const CommandHeader& header) {
    const std::uint64_t tail = tail_->load(std::memory_order_relaxed);
    const std::uint64_t head = head_->load(std::memory_order_acquire);
    if (tail - head >= capacity_) {
      return CommandRingResult::kFull;
    }

    entries_[tail % capacity_] = header;
    tail_->store(tail + 1, std::memory_order_release);
    return CommandRingResult::kOk;
  }

  CommandRingResult TryPop(std::uint64_t expected_worker_generation, CommandHeader* out) {
    const std::uint64_t head = head_->load(std::memory_order_relaxed);
    const std::uint64_t tail = tail_->load(std::memory_order_acquire);
    if (head == tail) {
      return CommandRingResult::kEmpty;
    }

    const CommandHeader header = entries_[head % capacity_];
    if (header.worker_generation != expected_worker_generation) {
      return CommandRingResult::kStaleWorkerGeneration;
    }

    *out = header;
    head_->store(head + 1, std::memory_order_release);
    return CommandRingResult::kOk;
  }

 private:
  CommandHeader* entries_;
  std::uint64_t capacity_;
  std::atomic<std::uint64_t>* head_;
  std::atomic<std::uint64_t>* tail_;
};

}  // namespace nebulasd::control_plane
