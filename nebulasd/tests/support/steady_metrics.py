"""A common post-first-Draft, pre-first-completion window; no engine mutation."""
from collections import Counter


def steady_summary(first_draft_ready, identities, metrics, commands):
    missing = sorted(set(identities)-set(first_draft_ready))
    if missing:
        return dict(valid=False, reason='missing_first_draft_ready', missing=missing)
    start = max(first_draft_ready[k] for k in identities)
    end = min(metrics.completed.values())
    if end <= start:
        return dict(valid=False, reason='no_common_steady_window', start_ns=start, end_ns=end)
    chunks = [r for r in metrics.chunks if start < r['observed_ns'] <= end]
    tokens = sum(len(r['tokens']) for r in chunks)
    by_worker = {}
    for now,c in commands:
        if not start < now <= end or c.kind.name not in ('RUN_DRAFT_BATCH','RUN_TARGET_BATCH'):
            continue
        row = by_worker.setdefault(c.worker_id, Counter())
        row[len(c.requests)] += 1
    return dict(valid=True, start_ns=start, end_ns=end, duration_s=(end-start)/1e9,
                concurrent_requests=len(identities), output_tokens=tokens,
                output_tokens_per_second=tokens/((end-start)/1e9),
                requests_with_output=len({r['request_id'] for r in chunks if r['tokens']}),
                worker_batch_histograms={k:dict(v) for k,v in by_worker.items()},
                boundary='latest observed first Draft ready to first terminal output observation; excludes initial prefill/first Draft and tail, chunk timestamps')
