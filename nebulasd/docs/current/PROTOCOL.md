# Communication protocol

Shared facts and completion records are authoritative. Rings/eventfd are change and wakeup hints that may be coalesced; missed hints are recovered through table scanning. Shared rows use explicit writer ownership and stable publication/read rules, not arbitrary concurrent writes.

| Boundary | Content |
| --- | --- |
| Engine -> control | Immutable WORK and management commands |
| Engine -> DMA | Import ranges derived from the same WORK; no separate execution authorization |
| Worker -> shared tables/arenas | Results, cumulative output, proposals, snapshots, HostKV-ready and runtime/Bank facts |
| Worker -> completion arena | Physical completion and exact member outcomes |
| control <-> execution | Bounded private COMPUTE Plan / COMPUTED Result payloads |
| control <-> DMA | Metadata, export and transfer completion |

Current shared-table ABI is 6; autonomous WORK is version 4. `ipc/protocol.py` also contains a separate batch-command version 5 used by lower-level contracts; it is not the production WORK encoding. WORK can encode 256 members, public configuration limits batches to 128, and the unretired WORK window is four per Worker. Layout declarations and generated schemas remain authoritative for binary offsets.

## Identity

Correlate WORK by `(worker_id, worker_generation, work_seq)`, also validating Bank id/epoch and completion reservation. Members carry slot/request epoch, round/run sequence, host generation/lease and dependency tickets. A slot, sequence or Bank id without its generation is not a durable key.

Arena handles use offset/length/generation; token positions are byte-based while HostKV ranges are block-based. Logical token length, valid blocks, reserved capacity and dirty range are distinct.

## Fact acceptance

Engine reads bounded latest facts, validates authorization and fences against pre-update state, and applies accepted rows individually. It ignores shared ENGINE/DISPATCH mirrors and per-request H2D rows in its normal observation path. The scheduler sees facts accepted at that invocation, not an artificial batch-result transaction.

Cumulative Target output supports reading the unseen suffix even when intermediate versions are skipped. Worker inputs still match exact WORK dependencies; arbitrary latest data is not automatically ready. There is no second reliable RESULT/HOST_READY receipt protocol on the Engine side.

## Completion

Compute completion, D2H readiness, physical completion, Engine result application and client delivery have separate owners/meanings. The ledger checks completion identity and full membership, then retires authorization once every executed member's result has been applied. Transport credit release alone does not justify losing authorization records. All-skipped WORKs still publish completion.

Changing protocol fields requires checking writers, codecs, dependency readers, Engine validation, native projections and differential tests. Cover backpressure, skipped versions, reordered observations, all-skipped batches and multiple cohorts. Do not collapse unrelated ABI versions into one counter.

Sources: [WORK](../../src/nebulasd/workers/work.py), [table layout](../../src/nebulasd/table/layout_types.py), [ledger](../../src/nebulasd/engine/work_ledger.py).
