"""Offline component duration, overlap and exposed-copy summaries."""

from math import ceil


def percentiles(values):
    ordered = sorted(values)
    return {"samples": len(ordered), **{
        f"p{q}_ms": ordered[max(0, ceil(len(ordered) * q / 100) - 1)] if ordered else None
        for q in (50, 95, 99)
    }}


def covered_ns(start, end, intervals):
    """Measure the union, so multiple overlapping kernels are not double-counted."""
    clips = sorted((max(start, a), min(end, b)) for a, b in intervals if a < end and b > start)
    total, previous_end = 0, start
    for a, b in clips:
        total += max(0, b - max(a, previous_end))
        previous_end = max(previous_end, b)
    return total


def summarize_intervals(rows, clock_uncertainty_ns=None):
    clock_uncertainty_ns = clock_uncertainty_ns or {}
    components = {kind: percentiles([(r["end_ns"] - r["start_ns"]) / 1e6
                                    for r in rows if r["kind"] == kind])
                  for kind in ("D2H", "H2D", "target_verify", "draft")}
    overlaps = {}
    for copy in ("D2H", "H2D"):
        for compute in ("target_verify", "draft"):
            ratios, exposed, covered, adjusted = [], [], [], []
            for row in rows:
                if row["kind"] != copy:
                    continue
                peers = [r for r in rows if r["kind"] == compute and (
                    r["device"] == row["device"] if compute == "target_verify"
                    else bool(set(r["slots"]) & set(row["slots"])))]
                duration = row["end_ns"] - row["start_ns"]
                overlap = covered_ns(row["start_ns"], row["end_ns"],
                                     [(r["start_ns"], r["end_ns"]) for r in peers])
                ratios.append(overlap / duration if duration else 0.0)
                covered.append(overlap)
                conservative_peers = []
                for peer in peers:
                    error = (0 if peer["device"] == row["device"] else
                             clock_uncertainty_ns.get(peer["device"], 0)
                             + clock_uncertainty_ns.get(row["device"], 0))
                    if peer["end_ns"] - peer["start_ns"] > 2 * error:
                        conservative_peers.append((peer["start_ns"] + error, peer["end_ns"] - error))
                adjusted.append(covered_ns(row["start_ns"], row["end_ns"], conservative_peers))
                exposed.append((duration - overlap) / 1e6)
            overlaps[f"{copy}_with_{compute}"] = {
                "samples": len(ratios), "overlapped_ns": sum(covered),
                "calibration_adjusted_overlap_ns": sum(adjusted),
                "mean_overlap_ratio": sum(ratios) / len(ratios) if ratios else None,
                "exposed_copy": percentiles(exposed),
            }
    return {"components": components, "overlap": overlaps}
