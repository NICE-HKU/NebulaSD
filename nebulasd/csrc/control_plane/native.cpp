// Small C ABI for mapped Table rows and SPSC rings. No Python/GPU dependency.
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>

static_assert(__atomic_always_lock_free(8, nullptr), "shared u64 atomics must be lock-free");
namespace {
using U = std::uint64_t;
constexpr U invalid = std::numeric_limits<U>::max();
U load(const void* p) { return __atomic_load_n(static_cast<const U*>(p), __ATOMIC_SEQ_CST); }
void store(void* p, U v) { __atomic_store_n(static_cast<U*>(p), v, __ATOMIC_SEQ_CST); }
auto bytes(void* p) { return static_cast<unsigned char*>(p); }
}
extern "C" {
unsigned sd_native_abi() { return 1; }
U sd_load(const void* p) { return load(p); }
void sd_store(void* p, U v) { store(p, v); }
U sd_exchange(void* p, U v) { return __atomic_exchange_n(static_cast<U*>(p), v, __ATOMIC_SEQ_CST); }
int sd_cas(void* p, U expected, U desired) {
  return __atomic_compare_exchange_n(static_cast<U*>(p), &expected, desired, false,
                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// SC byte accesses also make concurrent payload reads legal under C++'s
// memory model. Readers cannot observe a new payload under an old sequence.
void sd_table_publish(void* row, U seq, const unsigned* offsets,
                      const unsigned* lengths, const unsigned char* data, unsigned n) {
  store(row, invalid);
  for (unsigned i = 0, pos = 0; i < n; ++i)
    for (unsigned j = 0; j < lengths[i]; ++j)
      __atomic_store_n(bytes(row) + offsets[i] + j, data[pos++], __ATOMIC_SEQ_CST);
  store(row, seq);
}
int sd_table_read(void* row, unsigned size, unsigned char* out, U* seq) {
  U a = load(row);
  if (a == invalid) return 0;
  for (unsigned i = 8; i < size; ++i)
    out[i - 8] = __atomic_load_n(bytes(row) + i, __ATOMIC_SEQ_CST);
  U b = load(row);
  if (a != b || b == invalid) return 0;
  *seq = b;
  return 1;
}

// Bounded registered rows, one stable copy only when its publication changes.
// The caller owns all arrays and immutable registrations. No waits/callbacks.
unsigned sd_watch_scan(const std::uintptr_t* rows, const unsigned* sizes,
                       U* previous, unsigned count, unsigned char* output,
                       unsigned stride, U* sequences, unsigned* changed) {
  unsigned found = 0;
  for (unsigned i = 0; i < count; ++i) {
    void* row = reinterpret_cast<void*>(rows[i]);
    U seq = load(row);
    if (seq == invalid || seq == previous[i]) continue;
    if (!sd_table_read(row, sizes[i], output + i * stride, &seq)) continue;
    previous[i] = sequences[i] = seq;
    changed[found++] = i;
  }
  return found;
}

// head, tail and flags occupy separate cache lines; entries start at 192.
int sd_ring_push(void* ring, U capacity, U width, const void* record) {
  U tail = load(bytes(ring) + 64), head = load(ring);
  if (tail - head >= capacity) return 0;
  std::memcpy(bytes(ring) + 192 + (tail % capacity) * width, record, width);
  store(bytes(ring) + 64, tail + 1);
  return 1;
}
int sd_ring_peek(void* ring, U capacity, U width, void* record) {
  U head = load(ring), tail = load(bytes(ring) + 64);
  if (head == tail) return 0;
  std::memcpy(record, bytes(ring) + 192 + (head % capacity) * width, width);
  return 1;
}
void sd_ring_ack(void* ring) { store(ring, load(ring) + 1); }
// Single consumer, bounded snapshot of the available hint prefix. Publish head
// only after copying, so the producer cannot overwrite any unread entry.
U sd_ring_drain(void* ring, U capacity, U width, U limit, void* output) {
  U head = load(ring), tail = load(bytes(ring) + 64);
  U count = tail - head;
  if (count > limit) count = limit;
  if (count > capacity) count = capacity;
  U first = capacity - head % capacity;
  if (first > count) first = count;
  std::memcpy(output, bytes(ring) + 192 + (head % capacity) * width, first * width);
  if (first < count)
    std::memcpy(static_cast<unsigned char*>(output) + first * width,
                bytes(ring) + 192, (count - first) * width);
  if (count) store(ring, head + count);
  return count;
}
// A prepared table row and its loss-tolerant Engine hint share one bounded
// native call. The table release commit always precedes the ring publication.
int sd_table_publish_notify(void* row, U seq, const unsigned* offsets,
                            const unsigned* lengths, const unsigned char* data,
                            unsigned n, void* ring, U capacity,
                            unsigned kind, unsigned slot, int notify) {
  sd_table_publish(row, seq, offsets, lengths, data, n);
  struct Entry { unsigned kind, slot; U seq; } entry{kind, slot, seq};
  static_assert(sizeof(Entry) == 16);
  if (!sd_ring_push(ring, capacity, sizeof(Entry), &entry))
    store(bytes(ring) + 128, 1);
  return notify && sd_exchange(bytes(ring) + 136, 1) == 0;
}

}

#include "reader.cpp"
