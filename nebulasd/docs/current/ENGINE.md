# Engine

## API and ownership

`api.create_engine` calls `engine/bootstrap.py`, constructs shared resources and the scheduler, and waits for Workers to become online. The production path requires completion scheduling, stagewise placement on both sides and a table cost model.

`TokenEngine` belongs to the calling owner thread. Health monitoring does not advance scheduling on behalf of the application. Applications tokenize their inputs.

| API | Behavior |
| --- | --- |
| `submit(prompt, config)` | Validate/admit a request and return its client handle |
| `poll()` | Advance one global Engine step |
| `read(handle)` | Drain buffered client events without advancing execution |
| `stream(handle)` | Advance the global Engine while yielding this request's events |
| `result(handle)` | Return completed output token IDs |
| `cancel(handle)` | Active autonomous requests raise `NotImplementedError` |
| `release(handle)` | Drop a terminal client reference, not physical KV ownership |
| `drain()` | Wait for requests, physical retirement and cohort reclamation |
| `close()` | Stop Workers and release resources |

A stream timeout or early iterator exit does not cancel the request. Client session identity is distinct from internal request slot/epoch identity.

`RequestRegistry` validates prompts, budgets and conservative cohort capacity, then reserves payload and both role-specific HostKV allocations. Slots advance within a cohort rather than being immediately reused at request completion. Owner request records hold lifecycle, output progress and round state; local Dispatch rows hold issued authorization. The scheduler borrows the owner mapping during its synchronous call. There is no second local ENGINE progress mirror.

## Step and scheduling

```text
health check -> drain doorbell -> bounded shared-fact read/application
-> one completion scan -> ledger refresh -> reclamation check
-> eligible Draft/Target scheduling and WORK dispatch -> flush client output
```

The normal observation budget is 128, with priority checks for Worker runtime/Bank rows. Stage boundaries can perform additional observation/refresh; shared completion is scanned once per step. Changed-row hints are advisory and overflow recovery scans the authoritative table.

Fact validation uses pre-update cached state to validate authorization and copy fences before applying the accepted updates. Result rows are accepted individually; a batch is not an atomic result transaction. Dirty destinations are combined before planning. Draft/Target phase priority alternates.

A WORK binds Worker generation/sequence, Bank epoch, members, input dependencies, layout and completion reservation. Successful submission is recorded in the ledger before local issued state advances. If admission returns `Full`, the original WORK and completion reservation remain pending and are retried before replanning that phase. The per-Worker outstanding window is four.

## Results and retirement

Target publishes cumulative output and decides EOS/length completion. `OutputManager` verifies identity/round and reads the unconsumed output suffix, then updates request state before subsequent scheduling. Client delivery is flushed at the end of the step. Skipping an intermediate shared-row version does not skip cumulative output tokens.

Compute completion, HostKV readiness, physical completion, result application and client delivery are different conditions. The ledger validates a matching shared completion and retires authorization only after every executed member's required result has been applied. All-skipped WORKs still complete without fabricated result rows.

`CohortRecycler` waits for terminal requests, no outstanding ledger records and no pending dispatch, stops Workers, reuses HostKV/payload capacity, advances identities and restarts Workers. Terminal client outputs can remain available. The benchmark deliberately delays optional cohort reclamation during resident warmups; this is not the normal production recycling policy.

Sources: [core](../../src/nebulasd/engine/core.py), [work progress](../../src/nebulasd/engine/work_progress.py), [ledger](../../src/nebulasd/engine/work_ledger.py), [client](../../src/nebulasd/engine/client.py).
