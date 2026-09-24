"""History-free bank verification for a process-isolated StarSD executor.

This is a local, typed execution entry point. Hidden tensors and CUDA events
never leave this process. Acceptance uses the existing canonical algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactVerifyRequest:
    request_id: str
    client_tag: str
    request_row: int
    logical_kv_len: int
    kv_version: int
    anchor_token_id: int
    draft_token_ids: tuple[int, ...]
    remaining_output_len: int
    stop_token_ids: tuple[int, ...] = ()
    proposal_kind: str = 'linear'


@dataclass(frozen=True)
class CompactVerifyBatchPlan:
    active_bank_id: int
    active_bank_epoch: int
    batch_seq: int
    requests: tuple[CompactVerifyRequest, ...]


@dataclass(frozen=True)
class CompactVerifyResult:
    request_id: str
    client_tag: str
    request_row: int
    accepted_token_ids: tuple[int, ...]
    num_accepted_draft_tokens: int
    logical_kv_len: int
    kv_version: int
    gpu_valid_blocks: int
    finished: bool
    target_hidden: object | None
    target_hidden_ready_event: object | None


async def verify_compact(facade, plan):
    """Validate local resources and run forward/acceptance/hidden/crop once.

The execution process still owns the Bank lease and must certify physical
completion before publishing any remote completion. The returned CUDA event
is a local dependency, not a cross-process completion descriptor.
"""
    from swiftllm.speculative import VerifyPlanItem, compute_acceptance_for_plan
    from swiftllm.worker.model import ModelForwardOutput
    from .starsd_process_local_facade import LocalCudaResult

    worker = facade.worker
    if not worker.initialized:
        raise RuntimeError('compact Target is not initialized')
    manager = facade._require_block_manager()
    if plan.active_bank_id != int(manager.active_bank_id):
        raise RuntimeError('compact verify active bank mismatch')
    bank = manager.get_bank_descriptor(plan.active_bank_id)
    if int(bank.epoch) != plan.active_bank_epoch or str(bank.batch_id) != str(plan.batch_seq):
        raise RuntimeError('compact verify stale Bank epoch/batch')
    if not plan.requests or len(plan.requests) > worker.engine_config.max_batch_size:
        raise ValueError('compact verify batch capacity exceeded')
    rows = [r.request_row for r in plan.requests]
    keys = [(r.client_tag, r.request_id) for r in plan.requests]
    if len(set(rows)) != len(rows) or len(set(keys)) != len(keys):
        raise ValueError('compact verify duplicate request/row')
    plans, input_ids, seq_ids, seq_lens = [], [], [], []
    for item in plan.requests:
        if item.proposal_kind not in ('linear', 'dflash_block'):
            raise ValueError('unsupported compact Target proposal')
        if item.remaining_output_len <= 0 or item.logical_kv_len <= 0:
            raise ValueError('compact verify requires live KV and output budget')
        location = bank.request_ranges.get(item.request_row)
        if location is None:
            raise RuntimeError('compact verify row absent from Bank')
        manager.validate_bank_location(location)
        if (int(location.logical_kv_len), int(location.kv_version)) != (item.logical_kv_len, item.kv_version):
            raise RuntimeError('compact verify stale local KV length/version')
        tokens = [item.anchor_token_id, *item.draft_token_ids]
        vocab = worker.model_config.vocab_size
        if any(token < 0 or token >= vocab for token in (*tokens, *item.stop_token_ids)):
            raise ValueError('compact verify token outside vocabulary')
        depth = worker.engine_config.speculative_max_draft_tokens
        if depth and len(item.draft_token_ids) > depth:
            raise ValueError('compact verify proposal capacity exceeded')
        if item.logical_kv_len + len(tokens) > int(location.num_blocks) * manager.block_size:
            raise RuntimeError('compact verify exceeds reserved Bank capacity')
        lengths = list(range(item.logical_kv_len + 1, item.logical_kv_len + 1 + len(tokens)))
        plans.append(VerifyPlanItem(None, tokens, lengths, list(item.draft_token_ids),
                                   item.proposal_kind, len(input_ids), len(tokens)))
        input_ids.extend([token] for token in tokens)
        seq_ids.extend([item.request_row] * len(tokens))
        seq_lens.extend(lengths)
    if len(input_ids) > worker.engine_config.max_tokens_in_batch:
        raise ValueError('compact verify token capacity exceeded')

    def run():
        # One model-executor handoff for the entire local operation. The only
        # Python work here is bounded by this batch/proposal, not past history.
        import torch
        cache = worker.model.k_cache
        cuda = bool(getattr(cache, 'is_cuda', False))
        if cuda and bank.ready_event is not None:
            torch.cuda.current_stream(device=cache.device).wait_event(bank.ready_event)
        output = worker.model.forward(input_ids, seq_ids, seq_lens,
            return_hidden=True, hidden_layer_ids=worker._target_hidden_layer_ids(), return_dict=True)
        if not isinstance(output, ModelForwardOutput):
            raise RuntimeError('compact Target requires structured forward output')
        if len(output.token_ids) != len(input_ids):
            raise RuntimeError('compact Target forward row count mismatch')
        row_hidden = worker._select_row_hidden_states(output.hidden_states, input_ids, 0)
        results = []
        for item, verify in zip(plan.requests, plans, strict=True):
            start, count = verify.output_row_start, verify.output_row_count
            accepted, accepted_count = compute_acceptance_for_plan(verify,
                output.token_ids[start:start + count], remaining_output_len=item.remaining_output_len,
                stop_token_ids=item.stop_token_ids)
            hidden, hidden_event = worker._hidden_payload_for_rows(row_hidden, start, len(accepted),
                                                                   output.hidden_ready_event)
            logical = item.logical_kv_len + len(accepted)
            if not accepted:
                raise RuntimeError('compact Target produced no accepted token')
            results.append((item, accepted, accepted_count, logical, hidden, hidden_event))
        worker.model.crop_seqs_resources(rows, [r[3] for r in results])
        normalized = []
        for item, accepted, accepted_count, logical, hidden, hidden_event in results:
            blocks = worker._allocated_block_count_for_row(item.request_row)
            if blocks != (logical + manager.block_size - 1) // manager.block_size:
                raise RuntimeError('compact Target crop block count mismatch')
            # Canonical crop itself advances the Bank location version. Match
            # _attach_bank_verify_metadata: publish the next version of the
            # *post-crop* location, never overwrite it from the input version.
            cropped = manager.get_bank_descriptor(plan.active_bank_id).request_ranges[item.request_row]
            location = manager.set_bank_location_kv_version(item.request_row, bank_id=plan.active_bank_id,
                bank_epoch=plan.active_bank_epoch, kv_version=(int(cropped.kv_version) + 1) % (1 << 64))
            normalized.append(CompactVerifyResult(item.request_id, item.client_tag, item.request_row,
                tuple(accepted), accepted_count, logical, int(location.kv_version), blocks,
                len(accepted) == item.remaining_output_len or accepted[-1] in item.stop_token_ids,
                hidden, hidden_event))
        event = None
        if cuda:
            stream = torch.cuda.current_stream(device=cache.device)
            for result in normalized:
                if result.target_hidden_ready_event is not None:
                    stream.wait_event(result.target_hidden_ready_event)
            event = torch.cuda.Event()
            event.record(stream)
            # Keep the original hidden storage alive until its derived copies
            # retire; allocator lifetime is also tied to the consumer stream.
            if output.hidden_states is not None:
                output.hidden_states.record_stream(stream)
        return LocalCudaResult(True, event, tuple(normalized))

    if worker._batch_lock is None:
        return await worker._run_on_model_async(run)
    async with worker._batch_lock:
        return await worker._run_on_model_async(run)
