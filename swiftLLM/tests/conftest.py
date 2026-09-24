from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def assert_canonical_swiftllm_imports():
    root = Path(__file__).resolve().parents[1]
    try:
        import swiftllm
        import swiftllm.server.starsd_target_facade as starsd_target_facade

        package_path = Path(swiftllm.__file__).resolve()
        facade_path = Path(starsd_target_facade.__file__).resolve()
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        package_path = (root / "swiftllm" / "__init__.py").resolve()
        facade_path = (root / "swiftllm" / "server" / "starsd_target_facade.py").resolve()
    assert package_path.is_file()
    assert facade_path.is_file()
    print(f"swiftllm.__file__={package_path}", flush=True)
    print(f"swiftllm.server.starsd_target_facade.__file__={facade_path}", flush=True)
    assert package_path.is_relative_to(root)
    assert facade_path.is_relative_to(root)
