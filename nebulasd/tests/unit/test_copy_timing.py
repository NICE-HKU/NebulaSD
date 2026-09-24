from nebulasd.observability.copy_timing import covered_ns, summarize_intervals


def test_overlap_unions_intervals_and_separates_request_and_device():
    assert covered_ns(0, 100, [(10, 50), (40, 80), (90, 120)]) == 80
    rows = [
        dict(kind="D2H", start_ns=0, end_ns=100, slots=[1], device=0),
        dict(kind="target_verify", start_ns=10, end_ns=50, slots=[2], device=0),
        dict(kind="target_verify", start_ns=0, end_ns=100, slots=[1], device=1),
        dict(kind="draft", start_ns=30, end_ns=90, slots=[1], device=2),
        dict(kind="draft", start_ns=0, end_ns=100, slots=[9], device=2),
    ]
    report = summarize_intervals(rows)
    assert report["overlap"]["D2H_with_target_verify"]["mean_overlap_ratio"] == 0.4
    assert report["overlap"]["D2H_with_draft"]["mean_overlap_ratio"] == 0.6
    assert report["components"]["H2D"]["samples"] == 0


def test_cross_gpu_overlap_gate_excludes_clock_uncertainty():
    rows = [dict(kind="D2H", start_ns=0, end_ns=100, slots=[0], device=0),
            dict(kind="draft", start_ns=80, end_ns=200, slots=[0], device=1)]
    overlap = summarize_intervals(rows, {0: 30, 1: 30})["overlap"]["D2H_with_draft"]
    assert overlap["overlapped_ns"] == 20
    assert overlap["calibration_adjusted_overlap_ns"] == 0
