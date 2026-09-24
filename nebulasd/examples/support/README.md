# Benchmark support

`output_metrics.py` records client-visible first-token, chunk and request timing. A speculative chunk can contain multiple tokens, so chunk intervals are not per-token latency.

`fixed_context_cost.py` constrains compute-cost lookup to the benchmark context while preserving actual transfer sizes. These helpers are shared with unit tests and are not imported by production code.
