from support.copy_benchmark import summarize, compare_paths


def test_ab_uses_only_measured_matching_cases_and_reports_duration_ratios():
    def row(path, latency, measured=True):
        return dict(path=path, blocks=1, batch=4, direction="D2H", measured=measured,
                    copy_bytes=1024, cuda_interval_ms=latency, launch_call_ms=latency / 2)
    result = summarize([row("direct", 999, False), row("direct", 1), row("old_staged", 2)])
    assert all(r["samples"] == 1 for r in result)
    comparison = compare_paths(result)[0]
    assert comparison["direct_over_old_duration_ratio"]["cuda_interval_ms"]["p50"] == 0.5
    assert result[0]["effective_payload_gbps_p50"] == 1024 / 1e6
