# Documentation

Start with the [root README](../../README.md) for the system model, installation and one-command 2D2T benchmark. All public documentation is in English.

- [Architecture](current/README.md): responsibilities and reading order.
- [Engine](current/ENGINE.md): admission, scheduling, output and recycling.
- [Workers](current/WORKER.md): control, execution and DMA boundaries.
- [KV](current/KV.md): HostKV, Banks and copy dependencies.
- [Scheduler](current/SCHEDULER.md): placement and decision rules.
- [Protocol](current/PROTOCOL.md): shared facts, WORK and completion.
- [Operations](current/OPERATIONS.md): build, configuration and measurements.
- [Limitations](current/KNOWN_ISSUES.md): supported behavior and validation scope.

Historical proposals, development prompts and machine-specific experiment archives are not part of the public documentation. Update the matching current page when changing a protocol or ownership boundary.

[Public release validation and remaining decision](PUBLIC_RELEASE.md).
