from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from nebulasd.workers.target.inputs import TargetInputs
from nebulasd.workers.work import TableDependency, Selector
from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.kv.arena import SharedHostKVArena
from test_autonomous_work import work


def test_import_members_are_frozen_from_copy_plan_not_captured_or_current_live():
    host = SharedHostKVArena.create(total_blocks=16, block_bytes=4)
    backend = TargetInputs(host=host, tokens=None, configs=None, proposals=None,
                           input_pool=ThreadPoolExecutor(1))
    w = work()
    a = replace(w.rows[0], source=TableDependency(K.REQUEST_D2H, 0, 1, 1, Selector.TARGET_HOST))
    b = replace(a, slot=1, host_offset=8, destination_offset=8,
                source=TableDependency(K.REQUEST_D2H, 1, 1, 1, Selector.TARGET_HOST))
    w = replace(w, rows=(a, b))
    captured = ({'source':{'logical_kv_len':16, 'ready_version':1}},)*2
    layout = SimpleNamespace(offsets=(0,8))
    try:
        # Both sources arrived, row 1 finished before the immutable plan was made.
        plan = backend.compile_import(w, captured, layout, (0,)).result()
        assert tuple(r.index for r in plan.rows) == (0,)
        assert len(plan.regions) == 1
        # If row 1 instead finishes after submission, it really was imported.
        in_flight = backend.compile_import(w, captured, layout, (0,1)).result()
        assert tuple(r.index for r in in_flight.rows) == (0,1)
        assert len(in_flight.regions) == 2
    finally:
        backend.input_pool.shutdown()
        host.close()
        host.unlink()
