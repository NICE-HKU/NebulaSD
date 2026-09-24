"""Bounded recent latency samples; hot loops never grow an unbounded trace list."""

from collections import deque


class LatencySamples:
    def __init__(self, capacity=8192):
        self.samples = deque(maxlen=capacity)
        self.count = 0

    def add(self, nanoseconds):
        self.count += 1
        self.samples.append(nanoseconds)

    def summary(self):
        values = sorted(self.samples)
        result = dict(total_samples=self.count, retained_samples=len(values))
        for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99)):
            result[name + "_ms"] = values[min(int((len(values) - 1) * fraction), len(values) - 1)] / 1e6 if values else None
        return result
