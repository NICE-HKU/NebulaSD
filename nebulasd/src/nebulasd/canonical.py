"""Explicit repository-only import boundary for production model construction."""
from pathlib import Path
import sys


def import_canonical():
    root = Path(__file__).resolve().parents[3]/'swiftLLM'
    if not root.is_dir():
        raise RuntimeError('canonical ./swiftLLM worktree is required')
    sys.path.insert(0,str(root))
    import swiftllm
    import swiftllm.server.starsd_target_facade as facade
    paths = {'swiftllm.__file__':str(Path(swiftllm.__file__).resolve()),
             'swiftllm.server.starsd_target_facade.__file__':str(Path(facade.__file__).resolve())}
    for name,path in paths.items():
        print(f'{name}={path}',flush=True)
        assert Path(path).is_relative_to(root), f'non-canonical import: {name}'
    return paths
