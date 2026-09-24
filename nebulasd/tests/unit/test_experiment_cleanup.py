import asyncio
from types import SimpleNamespace

import pytest

from support.experiment_cleanup import close_all, close_experiment


def test_failed_owner_does_not_hide_error_or_skip_other_owners():
    visited = []
    class Owner:
        def __init__(self, fail):
            self.fail = fail
        async def close(self):
            visited.append(self.fail)
            if self.fail:
                raise RuntimeError("release failure")
    with pytest.raises(RuntimeError, match="release failure"):
        asyncio.run(close_all([Owner(True), Owner(False)]))
    assert sorted(visited) == [False, True]


def test_draft_cleanup_error_still_joins_target_and_retains_arena():
    visited = []
    async def draft_close():
        raise RuntimeError("draft failed")
    async def target_close():
        visited.append("target")
    world = SimpleNamespace(close=lambda: visited.append("arena"))
    with pytest.raises(RuntimeError, match="draft failed"):
        asyncio.run(close_experiment({}, SimpleNamespace(close=draft_close),
            [SimpleNamespace(runtime=SimpleNamespace(close=target_close))], [], world))
    assert visited == ["target"]


def test_target_shutdown_progresses_all_owners_concurrently():
    async def run():
        other_started = asyncio.Event()
        async def first_close():
            await asyncio.wait_for(other_started.wait(), 1)
        async def second_close():
            other_started.set()
        targets = [SimpleNamespace(runtime=SimpleNamespace(close=close)) for close in (first_close, second_close)]
        await close_experiment({}, None, targets, [], SimpleNamespace(close=lambda: None))
    asyncio.run(run())
