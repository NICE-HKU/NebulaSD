"""Merge owner profile files into per-request/epoch/round summaries."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nebulasd.observability.profile_report import summarize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    result = summarize(args.directory)
    path = Path(args.directory)/'summary.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(output=str(path), rounds=len(result['rounds']),
        dropped_events=result['dropped_events'], gpu_dropped_events=result['gpu_dropped_events'],
        round_latency=result['round_latency']), indent=2))
