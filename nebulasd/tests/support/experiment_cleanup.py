"""Join every owner and propagate teardown failures without losing the cause."""

import asyncio


async def close_all(owners):
    results = await asyncio.gather(*(owner.close() for owner in owners), return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise RuntimeError(f"owner shutdown failed: {errors}") from errors[0]


async def close_experiment(pending, draft, targets, setup_backends, world):
    tasks = [task for task in pending.values() if task is not None]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    # A Draft failure must not prevent joining Target DMA.
    owners = ([draft] if draft is not None else []) + [t.runtime for t in targets]
    try:
        # Local Targets may hold each other's HostKV pins. Both owner loops
        # must keep advancing while they drain; sequential close can deadlock.
        await close_all(owners)
    except BaseException as exc:
        errors.append(exc)
    for backend in setup_backends[len(targets):]:
        try:
            await backend.shutdown_async()
        except BaseException as exc:
            errors.append(exc)
    # Retain HostKV on a failed retirement: a stream might still reference it.
    if world is not None and not errors:
        world.close()
    if errors:
        raise RuntimeError(f"WP07 shutdown failed: {errors}") from errors[0]
