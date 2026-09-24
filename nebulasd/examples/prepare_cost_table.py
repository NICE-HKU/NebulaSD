"""Bind model locations after checking every other calibration identity field."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nebulasd/src'))


def bind_table(data, expected):
    from nebulasd.scheduler.cost_table import CostTable
    data = json.loads(json.dumps(data))
    for role in ('draft_model', 'target_model'):
        data['compatibility'][role]['path'] = expected[role]['path']
    actual = data['compatibility']
    mismatches = [key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError('Calibration incompatible: ' + ', '.join(sorted(mismatches)) +
                         '. Supply a cost table measured for this backend, models and hardware.')
    CostTable(data, expected=expected)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draft-model', type=Path, required=True)
    parser.add_argument('--target-model', type=Path, required=True)
    parser.add_argument('--devices', type=int, nargs=4, default=[0, 1, 2, 3])
    args = parser.parse_args()
    from nebulasd.canonical import import_canonical
    import_canonical()
    from nebulasd.engine.cost_setup import cost_identity
    import torch
    if len(set(args.devices)) != 4 or any(d < 0 or d >= torch.cuda.device_count() for d in args.devices):
        parser.error('2D2T requires four distinct visible CUDA devices')
    for model in (args.draft_model, args.target_model):
        if not (model / 'config.json').is_file():
            parser.error(f'Missing model config: {model / "config.json"}')
    expected = cost_identity(str(args.draft_model), str(args.target_model), args.devices)
    data = bind_table(json.loads(args.input.read_text()), expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(data, handle, indent=2)
    print(f'Validated cost table: {args.output}', flush=True)


if __name__ == '__main__':
    main()
