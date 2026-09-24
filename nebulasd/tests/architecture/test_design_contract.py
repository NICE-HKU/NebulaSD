"""Architecture tests for work package 00 design boundaries."""

from __future__ import annotations

import ast
import importlib
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "nebulasd"
PACKAGE_ROOT = PROJECT_ROOT / "src" / "nebulasd"
ACTIVE_SOURCE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".py",
    ".sh",
    ".toml",
}
FORBIDDEN_RUNTIME_IMPORT_ROOTS = {"cupy", "multiprocessing", "pycuda", "ray", "swiftllm", "torch"}
SNAP_REVISION_PATH_RE = re.compile(r"/snap/[^/]+/[0-9]+/")


@dataclass(frozen=True, slots=True)
class ImportIssue:
    importer: str
    imported: str
    reason: str
    path: pathlib.Path
    line: int


def _python_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _active_source_files(root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "docs" in path.relative_to(root).parts:
            continue
        if path.name == "CMakeLists.txt" or path.suffix in ACTIVE_SOURCE_SUFFIXES:
            files.append(path)
    return files


def _module_name(path: pathlib.Path, package_root: pathlib.Path) -> str:
    relative = path.relative_to(package_root.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        return ".".join([*parts[:-1], "__init__"])
    return ".".join(parts)


def _resolve_relative_import(importer: str, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""

    importer_parts = importer.split(".")
    package_parts = importer_parts[:-1]
    if level > len(package_parts):
        return module or ""

    base_parts = package_parts[: len(package_parts) - level + 1]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def _iter_imports(path: pathlib.Path, importer: str) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = _resolve_relative_import(importer, node.module, node.level)
            if imported:
                imports.append((imported, node.lineno))
                for alias in node.names:
                    if alias.name != "*":
                        imports.append((f"{imported}.{alias.name}", node.lineno))
    return imports


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _find_import_issues(package_root: pathlib.Path) -> list[ImportIssue]:
    issues: list[ImportIssue] = []
    for path in _python_files(package_root):
        importer = _module_name(path, package_root)
        for imported, line in _iter_imports(path, importer):
            if imported == "starsd" or imported.startswith("starsd."):
                issues.append(
                    ImportIssue(importer, imported, "nebulasd must not import legacy starsd", path, line)
                )

            if _matches(importer, "nebulasd.scheduler"):
                if _matches(imported, "nebulasd.workers"):
                    issues.append(
                        ImportIssue(importer, imported, "scheduler must not import worker backends", path, line)
                    )
                if _matches(imported, "nebulasd.engine"):
                    issues.append(
                        ImportIssue(importer, imported, "scheduler must not import engine registry", path, line)
                    )
                if imported.split(".")[0] in FORBIDDEN_RUNTIME_IMPORT_ROOTS or imported.startswith("cuda"):
                    issues.append(ImportIssue(importer, imported, "scheduler must not import CUDA/runtime", path, line))

            if _matches(importer, "nebulasd.workers"):
                if _matches(imported, "nebulasd.engine"):
                    issues.append(ImportIssue(importer, imported, "workers must not import engine", path, line))
                if _matches(imported, "nebulasd.scheduler"):
                    issues.append(ImportIssue(importer, imported, "workers must not import scheduler policy", path, line))

            if _matches(importer, "nebulasd.core"):
                if any(
                    _matches(imported, forbidden)
                    for forbidden in (
                        "nebulasd.data",
                        "nebulasd.engine",
                        "nebulasd.ipc",
                        "nebulasd.kv",
                        "nebulasd.observability",
                        "nebulasd.scheduler",
                        "nebulasd.table",
                        "nebulasd.workers",
                    )
                ):
                    issues.append(ImportIssue(importer, imported, "core must stay runtime/backend independent", path, line))
                if imported.split(".")[0] in FORBIDDEN_RUNTIME_IMPORT_ROOTS or imported.startswith("cuda"):
                    issues.append(ImportIssue(importer, imported, "core must not import runtime/CUDA/backend", path, line))
    return issues


def _format_issues(issues: list[ImportIssue]) -> str:
    return "\n".join(
        f"{issue.path.relative_to(REPO_ROOT)}:{issue.line}: {issue.importer} -> "
        f"{issue.imported}: {issue.reason}"
        for issue in issues
    )


def test_source_import_graph_respects_design_contract() -> None:
    issues = _find_import_issues(PACKAGE_ROOT)
    assert not issues, _format_issues(issues)


def test_import_checker_catches_documented_negative_edges(tmp_path: pathlib.Path) -> None:
    package_root = tmp_path / "src" / "nebulasd"
    (package_root / "core").mkdir(parents=True)
    (package_root / "engine").mkdir(parents=True)
    (package_root / "scheduler").mkdir(parents=True)
    (package_root / "workers").mkdir(parents=True)
    (package_root / "workers" / "draft").mkdir(parents=True)
    package_root.joinpath("__init__.py").write_text("from starsd import legacy\n", encoding="utf-8")
    package_root.joinpath("core", "bad.py").write_text(
        "\n".join(
            [
                "import multiprocessing",
                "import torch",
                "from nebulasd import data",
            ]
        ),
        encoding="utf-8",
    )
    package_root.joinpath("scheduler", "bad_workers.py").write_text("from nebulasd import workers\n", encoding="utf-8")
    package_root.joinpath("scheduler", "bad_engine.py").write_text("from nebulasd import engine\n", encoding="utf-8")
    package_root.joinpath("scheduler", "bad_runtime.py").write_text(
        "\n".join(
            [
                "import swiftllm",
                "from cuda import cuda",
                "from torch import Tensor",
            ]
        ),
        encoding="utf-8",
    )
    package_root.joinpath("workers", "draft", "bad.py").write_text(
        "\n".join(
            [
                "from nebulasd import engine",
                "from nebulasd import scheduler",
            ]
        ),
        encoding="utf-8",
    )

    issues = _find_import_issues(package_root)
    reasons = {issue.reason for issue in issues}
    assert "scheduler must not import worker backends" in reasons
    assert "scheduler must not import engine registry" in reasons
    assert "scheduler must not import CUDA/runtime" in reasons
    assert "workers must not import engine" in reasons
    assert "workers must not import scheduler policy" in reasons
    assert "core must stay runtime/backend independent" in reasons
    assert "core must not import runtime/CUDA/backend" in reasons
    assert "nebulasd must not import legacy starsd" in reasons


def test_top_level_import_is_side_effect_free() -> None:
    code = (
        "import sys\n"
        "import nebulasd\n"
        "assert 'ray' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'cupy' not in sys.modules\n"
        "assert 'pycuda' not in sys.modules\n"
        "assert 'swiftllm' not in sys.modules\n"
        "assert 'multiprocessing' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=True,
    )


def test_package_imports_from_src_layout() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        module = importlib.import_module("nebulasd")
    finally:
        sys.path.pop(0)

    module_path = pathlib.Path(module.__file__).resolve()
    expected_root = PACKAGE_ROOT.resolve()
    assert module_path.is_relative_to(expected_root)


def test_active_sources_do_not_reference_historical_or_revision_paths() -> None:
    files = _active_source_files(PROJECT_ROOT)
    old_tree = "old_" + "MforN"
    swiftllm_leaf = "swift" + "LLM"
    forbidden_fragments = (
        f"{old_tree}/NebulaSD/{swiftllm_leaf}",
        old_tree,
        f"NebulaSD/{swiftllm_leaf}",
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in text, f"{path.relative_to(REPO_ROOT)} contains forbidden path fragment {fragment!r}"
        assert SNAP_REVISION_PATH_RE.search(text) is None, (
            f"{path.relative_to(REPO_ROOT)} contains a snap revision absolute path"
        )


def test_no_symlinks_inside_nebulasd_sources() -> None:
    symlinks = [path for path in PROJECT_ROOT.rglob("*") if path.is_symlink()]
    assert not symlinks, "nebulasd must not contain symlink/path shim entries"


def test_canonical_swiftllm_does_not_import_nebulasd() -> None:
    swiftllm_root = REPO_ROOT / "swiftLLM"
    assert swiftllm_root.is_dir(), "canonical ./swiftLLM worktree must exist"

    offenders: list[str] = []
    for path in _python_files(swiftllm_root):
        text = path.read_text(encoding="utf-8")
        if "import nebulasd" in text or "from nebulasd" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, "SwiftLLM must not depend on nebulasd: " + ", ".join(offenders)
