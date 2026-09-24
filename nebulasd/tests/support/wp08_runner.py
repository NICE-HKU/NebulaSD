"""Cold artifact startup and per-process CUDA timeline aggregation."""
import json
from time import monotonic
from nebulasd.core.enums import WorkerStatus, StateChangeBlockKind as K
from nebulasd.scheduler.views import value
from nebulasd.observability.copy_timing import summarize_intervals
from support.gpu_timeline import measured


def wait_online(engine, timeout):
    deadline = monotonic()+timeout
    while True:
        engine.step()
        if all(value(engine.rows.get((K.WORKER_COMMON,w.worker_id)),'status')==WorkerStatus.ONLINE
               for w in engine.resources.specs):
            return
        if monotonic()>deadline:
            raise TimeoutError('Workers did not become online')
        engine.supervisor.bell.wait(.001)


def merge_gpu_reports(output, warmup_rounds, require_overlap, *, require_draft_progress=False,
                      dispatch_events=(), observations=()):
    reports = [json.loads(path.read_text()) for path in sorted(output.glob('worker-*/report.json'))]
    intervals = [row for report in reports for row in report['intervals']]
    uncertainty = {int(device):width for report in reports
                   for device,width in report['clock_alignment_uncertainty_ns'].items()}
    selected = [row for row in intervals if measured(row['round_ids'],warmup_rounds)]
    summary = summarize_intervals(selected,uncertainty)
    passed = all(m['calibration_adjusted_overlap_ns']>0 for m in summary['overlap'].values())
    from support.wp08_draft_progress import summarize_draft_progress
    progress = summarize_draft_progress(reports,dispatch_events,observations)
    status = 'failed_overlap' if require_overlap and not passed else 'passed'
    if require_draft_progress and progress['status'] != 'demonstrated':
        status = 'inconclusive_draft_progress' if status == 'passed' else status
    traces = [event for path in sorted(output.glob('worker-*/trace.json'))
              for event in json.loads(path.read_text())['traceEvents']]
    (output/'trace.json').write_text(json.dumps(dict(traceEvents=traces)))
    return dict(**summary,intervals=intervals,clock_alignment_uncertainty_ns=uncertainty,
        four_way_overlap_passed=passed,status=status,draft_progress=progress,
        copy_receipts_by_worker={str(r['worker_id']):r['copy_receipts']['summary'] for r in reports})
