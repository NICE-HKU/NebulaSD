# NebulaSD integration

This repository vendors and modifies SwiftLLM for the NebulaSD Draft/Target execution backend. Production orchestration lives in `nebulasd`, not the upstream SwiftLLM server Engine.

Local changes include speculative token planning/verification, Qwen3 attention support, external Bank/block layouts, Draft state restoration, Target execution interfaces and asynchronous/process-local helpers. Some facade/session interfaces remain for tests and compatibility; the default StarSD Worker backend directly owns the canonical model with external layouts.

The upstream README is retained for attribution and background. Its server commands and performance claims are not the NebulaSD quickstart. Use the repository root README for dependencies, native builds and the supported 2D2T run. The Apache-2.0 license and existing source notices remain unchanged.
