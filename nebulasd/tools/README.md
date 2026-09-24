# Tools

`build_native.py --output PATH` generates the scheduler schema header and builds the CPU control-plane/scheduler shared library with a C++17 compiler. Set `STARSD_NEXT_NATIVE_LIBRARY` to that library. It does not install CUDA or build SwiftLLM's extension.

`summarize_profile.py` summarizes recorded Worker profiling data. Profiling is optional and changes timing overhead; keep profiling runs separate from throughput comparisons.

`check_public_tree.py` checks local documentation links, per-directory READMEs, English-only text and hardcoded user-home paths. It checks maintained source files, not Git history or ignored runtime outputs.
