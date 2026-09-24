# Public release preparation

## Completed validation

- Active test suite: 932 passed after removing historical archives and unreferenced experiment helpers.
- CPU native library and SwiftLLM extension built from a separate export containing only public source files.
- Editable installation of both projects succeeded in a separate virtual environment. Large Python/CUDA dependencies were inherited from the existing tested environment; this was not a from-scratch dependency download on a new operating system.
- From outside the exported repository, `examples/run_2d2t.sh` completed its default two resident warmups and 512 requests with 128 output tokens each. All 65536 tokens were produced, physical WORK retired, launcher exit was zero and all recorded Worker exits were zero.
- The exported-source test suite also passed all 932 tests.
- One full timed run measured 2427.9 token/s and request P99 26.283s, without a valid steady window. This is an execution/reproducibility check, not evidence of a speedup.
- Launcher help, missing-model rejection and existing-output protection were checked. Cost relocation tests verify unchanged measurements and rejection of incompatible model/backend/hardware/software identities.
- Published text uses English, local Markdown links resolve and maintained StarSD directories have READMEs. Production Python/C++ execution code was not changed by this packaging work.

The default preset needs four compatible GPUs, local model weights and sufficient shared/pinned host memory. The included calibration is historical and narrowly scoped; see the root README before running.

## Privacy and publication

The public source snapshot excludes ignored experiments, runtime logs, model weights, compiled libraries, editor configuration and Git metadata. Current public text was checked for personal home paths, machine identity, network addresses and credential-like strings. This bounded text review is not a formal security certification. Existing third-party author/license attribution is intentionally retained.

Generated benchmark configuration, provenance and logs record actual model/source/output paths. They stay in ignored output directories. Review/redact them before sharing; do not add a complete local run directory to the public repository.

Deleting files in a branch does not remove earlier versions, author emails or commit messages from Git history. A source-only archive excludes that history. If creating a new public repository from the archive, choose the public commit identity explicitly. Do not assume an existing development branch has sanitized history.

## License and remaining publication checks

The maintainer selected Apache-2.0 for NebulaSD-owned code. The root `LICENSE` and the package-local `nebulasd/LICENSE` contain the standard license text; Python package metadata declares it. SwiftLLM's existing license and attribution remain unchanged. The bracketed copyright fields in the standard license appendix are the original application example, not an asserted project copyright holder.

The intended repository is [NICE-HKU/NebulaSD](https://github.com/NICE-HKU/NebulaSD). Its current contents, default branch, visibility, and license have not been verified from this local environment, and no code has been pushed.

Before the initial publication:

1. Confirm the actual copyright holders and authority to license the contributions. The GitHub organization name alone does not establish copyright ownership.
2. Establish the vendored SwiftLLM upstream revision, compare modified files, retain applicable original notices, and add prominent modification notices to changed upstream files. The current repository-wide integration notice does not complete this per-file review.
3. If backend source comments change, account for the source hash checked by the measured cost table. Review and validate any provenance update; do not bypass compatibility checks or present an unmeasured backend change as calibrated.
4. Read the destination repository before importing this source snapshot. Preserve existing remote files/history, review the staged file list, and push without force only when ready to publish.

The local license setup is complete; these provenance and publication checks remain separate from the recorded execution validation below.

## Package naming validation

The project directory, Python package and distribution now use `nebulasd`; the public configuration class is `NebulaSDConfig`. Import paths, multiprocessing targets, build/schema paths, examples, tests, documentation and ignore rules were updated together. Native exported symbols and `STARSD_*` environment variables retain their existing names. Vendored SwiftLLM source and its calibration identity are unchanged.

After the rename, all 932 tests passed both in the working tree and in a separate source export. A `nebulasd` wheel built successfully, and editable installation/import of the new public API passed. The renamed one-command launcher completed 512 requests with 128 output tokens each, physical retirement and clean Worker exits from outside the export directory. Python/CUDA dependencies were inherited from the tested environment; this was not a new operating-system installation. No compatibility alias package was added.
