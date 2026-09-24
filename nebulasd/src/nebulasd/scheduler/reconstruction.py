"""A bounded partition menu over one admitted set, never predecessor affinity."""

READY_SLACK_S = 0.002


def partitions(requests, workers, fits, ready, *, slack_s=READY_SLACK_S):
    """At most thirteen partitions: three orders x four sizes, plus seed.

    Every member is individually legal on some destination. Capacity is checked
    cumulatively while packing and again by assignment. Previous batch IDs play
    no part. Ready and shape boundaries are alternatives, not hard affinities.
    """
    orders = (
        list(requests),
        sorted(requests, key=lambda r: (ready[r.slot], r.arrival_seq, r.slot)),
        sorted(requests, key=lambda r: ((r.prompt_count + r.output_count).bit_length(),
                                      r.proposal_depth, ready[r.slot], r.arrival_seq, r.slot)),
    )
    largest = max(w.max_batch_size for w in workers)
    balanced = max(1, (len(requests) + len(workers) - 1) // len(workers))
    seen = set()
    for order_index, order in enumerate(orders):
        for cap in dict.fromkeys((largest, max(1, largest // 2), balanced, 1)):
            batches, batch = [], []
            low, high = float("inf"), float("-inf")
            for r in order:
                compatible = (order_index == 0 or not batch or
                              max(high, ready[r.slot]) - min(low, ready[r.slot]) <= slack_s)
                if batch and (len(batch) == cap or not compatible or
                              not any(fits(w, batch + [r]) for w in workers)):
                    batches.append(tuple(batch))
                    batch = []
                    low, high = float("inf"), float("-inf")
                    if len(batches) >= len(workers):
                        break  # This cap/order cannot fit the admitted set.
                batch.append(r)
                low, high = min(low, ready[r.slot]), max(high, ready[r.slot])
            else:
                if batch:
                    batches.append(tuple(batch))
            if sum(len(b) for b in batches) != len(requests):
                continue
            key = tuple(tuple(r.slot for r in b) for b in batches)
            if len(batches) <= len(workers) and key not in seen:
                seen.add(key)
                yield tuple(batches)
