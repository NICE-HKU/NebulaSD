"""CPU validation of Step5 content masking and fail-closed evidence reducers."""
from copy import deepcopy
import pytest
from support.draft_migration_evidence import prefix_digest
from support.draft_migration_report import verify_kv, overlap


def test_valid_prefix_masks_each_head_tail_in_last_block():
    # [layer=1, heads=2, tokens=4, dim=2], uint8. Padding is interleaved by head.
    blocks = [bytearray(range(16)), bytearray(range(16, 32))]
    opts = dict(length=5, block_size=4, block_bytes=16, shape=(1, 2, 4, 2))
    before = prefix_digest(lambda i: blocks[i], **opts)
    assert before['bytes'] == 20
    blocks[1][2:8] = bytes([255])*6
    blocks[1][10:16] = bytes([255])*6
    assert prefix_digest(lambda i: blocks[i], **opts) == before
    blocks[1][8] ^= 1  # Valid token in second head.
    assert prefix_digest(lambda i: blocks[i], **opts) != before


def test_bad_layout_and_short_reads_rejected():
    with pytest.raises(ValueError, match='shape'):
        prefix_digest(lambda i: b'', length=1, block_size=4, block_bytes=16, shape=(16,))
    with pytest.raises(ValueError, match='short'):
        prefix_digest(lambda i: b'', length=1, block_size=4, block_bytes=16)


def evidence():
    digest = dict(k=dict(sha256='a'*64, bytes=20), v=dict(sha256='b'*64, bytes=20))
    row = dict(schema=1, backend='cpu', identity=dict(worker_id=0, request_slot=0, request_epoch=1),
               worker=0, direction='D2H', gpu=digest, host=deepcopy(digest), equal=True,
               cancelled=False, copy_bytes=64)
    return [row, dict(deepcopy(row), direction='H2D', worker=1)]


@pytest.mark.parametrize('fault', ['missing_source','duplicate_source','corruption','stale_identity','backend','cancelled'])
def test_kv_report_never_accepts_incomplete_or_mismatched_evidence(fault):
    rows = evidence()
    assert verify_kv(rows, backend='cpu')['cross_imports'] == 1
    if fault == 'missing_source': rows.pop(0)
    elif fault == 'duplicate_source': rows.append(deepcopy(rows[0]))
    elif fault == 'corruption': rows[1]['gpu']['k']['sha256'] = 'c'*64
    elif fault == 'stale_identity': rows[1]['identity']['request_epoch'] = 2
    elif fault == 'backend': rows[1]['backend'] = 'gpu'
    elif fault == 'cancelled': rows[1]['cancelled'] = True
    with pytest.raises(ValueError):
        verify_kv(rows, backend='cpu')


def intervals():
    common = dict(backend='gpu', worker=1, device=0, clock='cuda_same_worker_origin',
                  bank_epoch=3, batch_seq=4)
    return [dict(common, kind='compute', bank_id=0, keys=[[0,1,2]], start_ns=100, end_ns=300),
            dict(common, kind='H2D', bank_id=1, keys=[[1,1,1]], start_ns=200, end_ns=400, copy_bytes=64)]


@pytest.mark.parametrize('field,value', [('worker',2),('device',2),('bank_id',0),('keys',[[0,1,99]]),('copy_bytes',0),('start_ns',300)])
def test_only_other_request_other_bank_same_worker_overlaps(field, value):
    rows = intervals()
    assert overlap(rows)['pairs'][0]['overlap_ns'] == 100
    rows[1][field] = value
    assert overlap(rows)['status'] == 'not_observed'


@pytest.mark.parametrize('field,value', [('backend','cpu'),('clock','host'),('end_ns',100),('keys',[])])
def test_cpu_and_invalid_event_intervals_cannot_claim_gpu_overlap(field, value):
    rows = intervals()
    rows[1][field] = value
    with pytest.raises(ValueError, match='CUDA'):
        overlap(rows)
    assert overlap([])['status'] == 'not_observed'
