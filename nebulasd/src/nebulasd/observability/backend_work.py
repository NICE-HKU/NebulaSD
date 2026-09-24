"""Opt-in call shapes, sync/proposal service intervals and acceptance evidence."""

from time import perf_counter_ns


def attach_backend_work(raw, recorder):
    backend = raw.worker.backend if getattr(raw, 'draft_banked', False) else raw.worker._backend
    if raw.target:
        if not hasattr(backend, "verify_batch_async"):
            return  # CPU fake backends retain their existing generic span hooks.
        from .forensic_kv import install

        capture = install(backend, recorder)
        evidence = {}
        facade = backend.bank_facade
        direct = facade.verify_batch_direct

        async def direct_verify(plan):
            result = await direct(plan)
            evidence["posterior"] = [
                list(r.payload.get("posterior_token_ids", ())) for r in result.value
            ]
            post = facade.worker.model.post_layer
            evidence["logits_probe"] = getattr(post, "_last_logits_probe", None)
            return result

        facade.verify_batch_direct = direct_verify
        original = backend.verify_batch_async

        async def verify(items):
            snapshots = await capture(items) if capture is not None else []
            start = perf_counter_ns()
            out = await original(items)
            recorder.record(
                "target.acceptance",
                start,
                perf_counter_ns(),
                keys=[],
                kv_snapshots=snapshots,
                posterior=evidence.get("posterior"),
                logits_probe=evidence.get("logits_probe"),
                items=[
                    dict(
                        key=[i.request_slot, i.request_epoch, i.round_id],
                        prefix=list(i.committed_output_token_ids),
                        prompt_count=i.prompt_token_count,
                        proposal=list(i.draft_token_ids),
                        accepted=o.accepted_draft_count,
                        committed=list(o.committed_delta),
                        logical_kv_len=o.logical_kv_len,
                        kv_version=o.target_kv_version,
                        bank=[o.bank_location.bank_id, o.bank_location.bank_epoch],
                    )
                    for i, o in zip(items, out, strict=True)
                ],
            )
            return out

        backend.verify_batch_async = verify
        return
    if not hasattr(backend, "_preflight_inputs"):
        return
    original = backend.run_batch
    adapter = backend._require_adapter()
    context = {}
    for name in ("prefill_batch", "decode_batch"):
        method = getattr(adapter, name)

        def wrap(method, name):
            def call(items):
                kind = (
                    "prefill"
                    if name == "prefill_batch"
                    else ("sync" if context.get("sync_left", 0) > 0 else "proposal")
                )
                if kind == "sync":
                    context["sync_left"] -= 1
                begin = perf_counter_ns()
                out = method(items)
                end = perf_counter_ns()
                recorder.record(
                    "draft." + kind,
                    begin,
                    end,
                    keys=context.get("keys", []),
                    request_count=len(items),
                )
                return out

            return call

        setattr(adapter, name, wrap(method, name))

    def run(items):
        # Read-only planning is duplicated only in diagnostic mode, outside the
        # measured backend interval; actual mutation still occurs exactly once.
        prepared = backend._preflight_inputs(tuple(items))
        sync = max((len(p.sync_suffix) for p in prepared), default=0)
        keys = [[i.request_slot, i.request_epoch, i.round_id] for i in items]
        context.update(sync_left=sync, keys=keys)
        start = perf_counter_ns()
        try:
            out = original(items)
        finally:
            context.clear()
        recorder.record(
            "draft.call",
            start,
            perf_counter_ns(),
            keys=keys,
            sync_forwards=sync,
            first_count=sum(p.state is None for p in prepared),
            generation_forwards=max(len(o.draft_token_ids) for o in out) - 1,
            kv_lengths=[len(p.authoritative_prefix) for p in prepared],
            depths=[p.proposal_limit for p in prepared],
            proposals=[list(o.draft_token_ids) for o in out],
            actual_proposal_tokens=sum(len(o.draft_token_ids) for o in out),
        )
        return out

    backend.run_batch = run
