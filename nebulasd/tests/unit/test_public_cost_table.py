"""Path relocation must not bypass calibration compatibility or change timings."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from examples.prepare_cost_table import bind_table


@pytest.fixture
def table():
    return json.loads((Path(__file__).resolve().parents[2] /
        'examples/cost_tables/rtx4090_qwen3.json').read_text())


def test_relocation_preserves_measurements_and_input(table):
    original = deepcopy(table)
    expected = deepcopy(table['compatibility'])
    expected['draft_model']['path'] = '/models/draft'
    expected['target_model']['path'] = '/models/target'
    bound = bind_table(table, expected)
    assert table == original
    assert bound['rows'] == original['rows']
    assert bound['compatibility'] == expected


@pytest.mark.parametrize('field', ['gpu', 'torch', 'cuda', 'swiftllm_sha256', 'block_size', 'gpu_memory_fraction'])
def test_relocation_rejects_incompatible_identity(table, field):
    expected = deepcopy(table['compatibility'])
    expected[field] = 'incompatible'
    with pytest.raises(ValueError, match='Calibration incompatible'):
        bind_table(table, expected)


def test_relocation_rejects_model_config_change(table):
    expected = deepcopy(table['compatibility'])
    expected['draft_model']['config_sha256'] = 'different model'
    with pytest.raises(ValueError, match='draft_model'):
        bind_table(table, expected)
