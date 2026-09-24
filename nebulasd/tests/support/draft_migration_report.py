"""Fail-closed Step5 evidence checks, independent of torch/CUDA."""
import json
from collections import Counter
from pathlib import Path


def verify_kv(rows, *, backend, require_cross=True):
    if not rows:
        raise ValueError('missing KV evidence')
    sources, imports = {}, []
    for row in rows:
        if row['schema'] != 1 or row['backend'] != backend:
            raise ValueError('KV evidence schema/backend mismatch')
        if set(row['gpu']) != {'k', 'v'} or set(row['host']) != {'k', 'v'}:
            raise ValueError('missing K/V plane evidence')
        if not row['equal'] or row['gpu'] != row['host']:
            raise ValueError('KV mismatch')
        if any(p['bytes'] <= 0 or len(p['sha256']) != 64
               or any(c not in '0123456789abcdef' for c in p['sha256']) for p in row['gpu'].values()):
            raise ValueError('empty/invalid KV digest')
        key = json.dumps(row['identity'], sort_keys=True)
        if row['direction'] == 'D2H':
            if row['worker'] != row['identity']['worker_id'] or key in sources:
                raise ValueError('duplicate or wrong-owner source evidence')
            sources[key] = row
        elif row['direction'] == 'H2D':
            imports.append((key, row))
        else:
            raise ValueError('unknown KV direction')
    cross = 0
    for key, row in imports:
        source = sources.get(key)
        if source is None or source['gpu'] != row['gpu']:
            raise ValueError('missing or inconsistent source for imported snapshot')
        cross += row['worker'] != source['worker'] and not row['cancelled']
    if not any(not r['cancelled'] for _,r in imports) or (require_cross and not cross):
        raise ValueError('no completed cross-worker KV evidence' if require_cross else 'no import evidence')
    return dict(status='passed', backend=backend, sources=len(sources), imports=len(imports), cross_imports=cross,
                copy_bytes=sum(r['copy_bytes'] for r in rows))


def overlap(rows):
    """Only real CUDA, same worker/device clock, other Bank and other requests."""
    pairs = []
    for row in rows:
        if (row.get('backend') != 'gpu' or row.get('clock') != 'cuda_same_worker_origin'
                or row['end_ns'] <= row['start_ns'] or not row['keys']):
            raise ValueError('invalid CUDA timing evidence')
    for copy in (r for r in rows if r['kind'] == 'H2D' and r.get('copy_bytes', 0) > 0):
        for compute in (r for r in rows if r['kind'] == 'compute'):
            if ((copy['worker'], copy['device']) != (compute['worker'], compute['device'])
                    or copy['bank_id'] == compute['bank_id']
                    or {(k[0], k[1]) for k in copy['keys']} & {(k[0], k[1]) for k in compute['keys']}):
                continue
            begin, end = max(copy['start_ns'], compute['start_ns']), min(copy['end_ns'], compute['end_ns'])
            if end > begin:
                pairs.append(dict(worker=copy['worker'], device=copy['device'], h2d_bank=copy['bank_id'],
                    compute_bank=compute['bank_id'], h2d_keys=copy['keys'], compute_keys=compute['keys'],
                    overlap_ns=end-begin, start_ns=begin, end_ns=end))
    unique = {}
    for worker, device in {(p['worker'], p['device']) for p in pairs}:
        end, total = -1, 0
        for p in sorted((p for p in pairs if (p['worker'], p['device']) == (worker, device)), key=lambda p:p['start_ns']):
            total += max(0, p['end_ns'] - max(end, p['start_ns']))
            end = max(end, p['end_ns'])
        unique[f'{worker}:{device}'] = total
    return dict(status='passed' if pairs else 'not_observed', pairs=pairs,
                unique_overlap_ns_by_worker_device=unique,
                note='Interval union within each worker/device; clocks are independent across workers.')


def dispatch_summary(commands):
    owners, batches = {}, Counter()
    for event in commands:
        kind, c = event['kind'], event['command']
        requests = c.get('requests', c.get('new_requests', []))
        batches[f'{kind}:{len(requests)}'] += 1
        if kind in ('DRAFT_BATCH', 'RUN_DRAFT_BATCH'):
            for r in requests:
                owners.setdefault((r['request_slot'], r['request_epoch']), []).append(c['worker_id'])
    aba = [list(key) for key, values in owners.items()
           if any(a == c and a != b for a,b,c in zip(values, values[1:], values[2:]))]
    return dict(batch_distribution=dict(batches), a_b_a_requests=aba,
                draft_owners=[dict(key=list(k), workers=v) for k,v in owners.items()])


def load_kv(directory):
    return [json.loads(line) for path in sorted(Path(directory).glob('kv-worker-*.jsonl'))
            for line in path.read_text().splitlines()]
