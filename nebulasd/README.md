# NebulaSD

The NebulaSD control plane. It admits tokenized requests, schedules independent Draft/Target pools, exchanges facts through shared memory, and delivers committed tokens. Model computation lives in the sibling `swiftLLM` tree.

Start with the [root setup and quickstart](../README.md). Run the [2D2T example](examples/README.md), or read the [current design](docs/current/README.md).

- `src/`: production Python implementation.
- `csrc/`: CPU native shared-memory and scheduler code.
- `tools/`: native build and profile summarization.
- `examples/`: public benchmark, checked calibration and metrics helpers.
- `tests/`: current CPU/protocol tests and explicit GPU acceptance.
- `docs/`: current architecture and limitations, without historical development plans.

The public API is `nebulasd.create_engine`. Configure completion scheduling, stagewise placement and a compatible measured cost table explicitly. Applications tokenize inputs and drive the owner thread. Active cancellation is unsupported; see [operational details](docs/current/OPERATIONS.md).

## License

NebulaSD-owned code is licensed under the [Apache License 2.0](LICENSE). See the [third-party notices](../THIRD_PARTY_NOTICES.md) for the vendored SwiftLLM backend and other dependencies.
