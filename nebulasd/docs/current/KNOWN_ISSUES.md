# Limitations and validation scope

- Production startup requires explicit completion/stagewise/table settings and a compatible measured cost table; bare default configuration is not a ready-to-run service.
- Active-request cancellation is not implemented. Natural EOS/length completion is supported.
- Normal cohort reclamation stops/restarts model processes. Resident benchmark warmups deliberately defer this behavior.
- The tested public preset is single-node 4×RTX4090 with Qwen3-0.6B/Qwen3-8B FP16. Other model/hardware/software combinations need compatible calibration and independent validation.
- The public benchmark uses synthetic prompts and checks completion/length. Small GPU acceptance checks compare against an independent Target reference. Different batching can cause floating-point output differences.
- Worker failures are fail-fast; transparent live recovery and arbitrary process-kill scenarios are not comprehensively validated.
- Short backpressure/delayed-publication tests do not prove all ring saturation or CUDA timing combinations safe. Long-running service stress remains separate from a completed 512×128 benchmark.
- Full run throughput is not steady throughput when the common active window is absent. Single-run differences are not statistically established improvements.
- The bundled calibration is historical and narrowly compatible; no automatic portable recalibration tool is provided.
- NebulaSD-owned code uses Apache-2.0. Before publication, finish the upstream SwiftLLM provenance and per-file modification-notice review described in [release preparation](../PUBLIC_RELEASE.md).

Current WORK/shared-fact completion logic has replaced historical reliable-result receipts. Defects tied to deleted result-commit switches are not current issues.
