# KV ownership and transfers

Draft and Target have separate HostKV arenas matching their model layouts. Same-role Workers share that arena; Draft KV is never used as Target KV. Execution owns GPU storage; Runtime/Banks owns physical layouts; DMA owns copy execution/order; Engine's ledger records authorization rather than freeing actual GPU layouts.

Admission reserves `ceil((prompt_tokens + max_new_tokens) / block_size)` blocks on each side. Allocation offset/capacity remains fixed for the request while ready versions, logical length and dirty ranges change. `block_bytes` is the size of one K or V block; copying both uses `2 * block_bytes * sum(region.block_count)` bytes. Reserved capacity is not actual transfer size.

## Banks

Each Worker has two banks with role-specific capacity, sharing `bank_rows` model row slots. Two full batches need enough combined row slots. Allocation requires advancing Bank epochs and available block/row capacity; release must identify the currently owned layout. Compute completion may permit early successor authorization, but does not permit releasing rows still referenced by export or queued tasks.

## Default direct-import path

1. Engine derives import ranges from WORK and accepted same-role HostKV facts.
2. Submission checks transport/window capacity, sends import descriptors, then publishes WORK.
3. DMA serializes a Bank's import/export/release sequence before accepting its next import.
4. Control independently allocates layouts, prepares Plans and submits metadata writes.
5. Compute waits for actual inputs, H2D, metadata and execution capacity.
6. Control submits D2H after receiving results; DMA completion permits subsequent transfer progression and physical release.

H2D submission can precede control's WORK acceptance. D2H completion can precede Engine's observation of HostKV-ready. Empty transfers are not physical KV memcpy. Numeric addresses do not replace request version/ticket validation.

Process mode defaults: `STARSD_H2D_CHUNK_BYTES=33554432`, `STARSD_H2D_GROUP_SIZE=1`, `STARSD_H2D_WAIT_MODE=event`. Chunk sizes align to K+V blocks. Zero chunk bytes explicitly requests whole-copy submission. Changing thread/process DMA also changes process boundaries and direct-import behavior, so it is not merely a thread-count comparison.

## Identity, placement and recycling

Host slots, leases, arena generations and payload handles must match their owning WORK. Draft snapshots carry restoration/reconciliation state. Dirty export, valid logical prefix and reserved capacity are separate quantities.

Role-specific NUMA policies support `default`, `bind` and `interleave`. Explicit placement validates allowed memory nodes, reserves/touches pages and checks placement before CUDA registration. Failure does not silently fall back; CPU affinity is not changed. `hostkv-numa.json` records initialization, not ongoing migration.

Cohort barriers protect shared capacity/payload reuse and advance generations after quiescence. Normal recycling restarts Workers; retain this distinction when interpreting resident benchmark warmups.

Sources: [arena](../../src/nebulasd/kv/arena.py), [Banks](../../src/nebulasd/workers/banks.py), [DMA](../../src/nebulasd/workers/dma_process.py), [direct import](../../src/nebulasd/workers/direct_import.py), [NUMA](../../src/nebulasd/kv/numa.py).
