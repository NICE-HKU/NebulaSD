"""Versioned, immutable backend timings. No CUDA/imports or implicit legacy fallback."""

from dataclasses import dataclass
from itertools import product
from math import isfinite
import json
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class Prediction:
    seconds: float
    p95_seconds: float
    method: str


class CostTable:
    """Linear interpolation only with every bounding corner present.

    KV queries use the longest member, never the mean. Sparse holes use the
    least expensive measured dominating shape. Out-of-range shapes explicitly
    scale the largest available shape by worst work ratio (an unvalidated
    conservative heuristic, not calibrated extrapolation). Missing stage/depth/
    sync families fail. Short proposals round UP to the next calibrated depth.
    Compatibility errors always fail; they never activate a different backend.
    """

    def __init__(self, data, *, expected):
        if data.get("schema_version") != 1:
            raise ValueError("unsupported backend cost schema")
        if data.get("compatibility") != expected:
            raise ValueError("backend cost table compatibility mismatch")
        self.compatibility = MappingProxyType(dict(expected))
        self.rows = tuple(MappingProxyType(dict(r)) for r in data["rows"])
        self._families = {}
        for r in self.rows:
            names = ("batch", "kv", "depth", "sync", "bytes", "samples")
            if any(type(r.get(n)) is not int or r[n] < 0 for n in names):
                raise ValueError("invalid calibration dimensions")
            if r["batch"] < 1 or r["samples"] < 20 or not r.get("boundary"):
                raise ValueError("require nonempty batch, boundary and >=20 samples")
            if (
                any(not isfinite(r.get(n, float("nan"))) or r[n] <= 0 for n in ("p50_ms", "p95_ms"))
                or r["p95_ms"] < r["p50_ms"]
            ):
                raise ValueError("invalid calibration duration")
            key = (r["stage"], r["depth"], r["sync"])
            family = self._families.setdefault(key, {})
            point = (r["batch"], r["bytes"] if r["stage"] in ("H2D", "D2H") else r["kv"])
            if point in family:
                raise ValueError("duplicate calibration shape")
            family[point] = r
        if not self.rows:
            raise ValueError("empty calibration")

    @classmethod
    def load(cls, path, *, expected):
        return cls(json.loads(Path(path).read_text()), expected=expected)

    def predict(self, stage, *, batch, kv=0, depth=0, sync=0, byte_count=0):
        if (
            any(type(x) is not int or x < 0 for x in (batch, kv, depth, sync, byte_count))
            or batch < 1
        ):
            raise ValueError("invalid cost query")
        depths = sorted(d for s, d, y in self._families if s == stage and y == sync and d >= depth)
        if not depths:
            raise ValueError(f"uncalibrated cost family: {stage}, depth={depth}, sync={sync}")
        family = self._families[stage, depths[0], sync]
        point = (batch, byte_count if stage in ("H2D", "D2H") else kv)
        suffix = ";depth_upper_bound" if depths[0] != depth else ""

        def result(a, b, method):
            return Prediction(a / 1000, b / 1000, method + suffix)

        if point in family:
            r = family[point]
            return result(r["p50_ms"], r["p95_ms"], "exact")
        axes = []
        for i, x in enumerate(point):
            values = sorted({p[i] for p in family})
            low = max((v for v in values if v <= x), default=None)
            high = min((v for v in values if v >= x), default=None)
            axes.append((low, high))
        if all(None not in axis for axis in axes):
            corners = set(product(*axes))
            if all(c in family for c in corners):
                a = b = 0.0
                for c in corners:
                    weight = 1.0
                    for i, (lo, hi) in enumerate(axes):
                        if lo != hi:
                            weight *= ((hi - point[i]) if c[i] == lo else (point[i] - lo)) / (
                                hi - lo
                            )
                    a += weight * family[c]["p50_ms"]
                    b += weight * family[c]["p95_ms"]
                return result(a, b, "bilinear")
        upper = [r for p, r in family.items() if all(a >= b for a, b in zip(p, point))]
        if upper:
            r = min(upper, key=lambda r: r["p95_ms"])
            return result(r["p50_ms"], r["p95_ms"], "sparse_upper_shape")
        p, r = max(family.items(), key=lambda pr: pr[1]["p95_ms"])
        scale = max(1.0, *(a / max(b, 1) for a, b in zip(point, p)))
        return result(r["p50_ms"] * scale, r["p95_ms"] * scale, "out_of_range_work_scaling")
