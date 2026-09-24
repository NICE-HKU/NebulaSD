# Scheduler

Production autonomous execution requires `scheduler_execution="completion"`, both placements `"stagewise"`, and `cost_model="table"`. Policy defaults to `existing`; `service_interval` is explicit. `STARSD_SCHEDULER_IMPL` selects `cpp` (default), `python`, or same-input differential `check`.

Engine combines dirty destinations, checks ledger capacity and calls `schedule(view, phase, destinations)`. Triggers include admission, Worker/Bank availability, issued work and accepted compute/result/physical/credit changes. Pending WORK is retried before replanning that phase. Candidate eligibility still checks lifecycle, rounds, issued identities, HostKV versions and capacities; timing predictions never replace actual Worker dependency gates.

## Existing policy

The planner filters candidates, forms a bounded frontier, predicts input/copy/Worker readiness, consults measured costs and builds batches. It is not simply FIFO or fixed-size batching. Cost families cover Draft first/cached, Target prefill/verify and H2D/D2H. Startup validates model, backend, hardware and layout identity. Fixed context 512 is a benchmark choice, not a general scheduler rule.

## Service-interval policy

This optional policy requires finite nonnegative `draft_service_gap_ms` and `target_service_gap_ms`, plus process DMA. Initial Target prefill retains existing planning. Predecessor compute-end clocks are retained by request epoch/stage/round independently of ledger retirement; unknown clocks are not invented.

For each destination, requests with known completed predecessors are prioritized by age, with deterministic tie-breaking. A capacity-aware first-fit forms top-B. The first member is retained as anchor. Further candidates are appended only if predicted start meets every trial member's fixed bound `max(predecessor_end + gap, singleton_start) + service_batch_delay`. Unknown predecessors use readiness prediction without being labeled actual waiting time. There is no backtracking or refill from outside top-B.

`service_batch_delay_ms` defaults to 30; it is scheduling slack, not a mandatory sleep timer or hardware SLO. `scheduler_ignore_kv_time` changes predictions only and does not disable actual KV dependencies. Eligibility scans and prediction costs remain; the planner is not generally O(B).

## Native boundary

Python synchronizes accepted numeric input buffers; C++ scans candidates and returns plans; Python constructs WORK. C++ does not read shared tables directly, publish results or callback into Python estimators. The owner synchronously waits for the native call. Custom unsupported estimator/view combinations use Python; `check` compares supported paths on identical input and is not a performance mode.

The schema is generated when building the native library. ABI/symbol mismatch fails rather than silently pretending an old library implements new logic. Scheduler kernel timing is only part of full scheduling cost.

Sources: [completion](../../src/nebulasd/scheduler/completion.py), [service interval](../../src/nebulasd/scheduler/service_interval.py), [native bridge](../../src/nebulasd/scheduler/native_completion.py), [native implementation](../../csrc/scheduler/scheduler.cpp).
