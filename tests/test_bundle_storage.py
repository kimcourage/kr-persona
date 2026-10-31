"""Reject old storage formats before publishing a newly assembled bundle."""
import pickle

import numpy as np
import pytest

from krpersona import paths
from krpersona.generate.resources import BUNDLE_FILES, check_bundle
from krpersona.report.bundle import build, check_storage


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    for name in BUNDLE_FILES:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'unchanged input')
    content = {
        'models/health_models.pkl': {'sem': {'G': np.zeros((2, 2))}},
        'models/labor_income_models.pkl': {'income': {'donors': {0: (np.arange(200) ** 2, np.ones(200))}}},
        'models/family_models.pkl': {'donors': {(0, 0): [(np.array([25., 30.]), 100.) for _ in range(5)]}},
        'census/aggregates.pkl': {'gap': {'value': np.array([25, 30]), 'count': np.array([10, 20])}},
    }
    for name, value in content.items():
        (root / name).write_bytes(pickle.dumps(value))
    monkeypatch.setenv('KRP_ARTIFACTS', str(root))
    monkeypatch.setenv('KRP_BUNDLE', str(tmp_path / 'bundle'))
    return root


def test_bundle_copies_estimated_inputs_without_reduction(source):
    build()
    check_bundle(paths.bundle())
    for name in BUNDLE_FILES:
        assert (paths.bundle() / name).read_bytes() == (source / name).read_bytes()


@pytest.mark.parametrize('name,old', [
    ('models/health_models.pkl', {'sem': {'X': [[1]], 'idx': [1]}}),
    ('models/labor_income_models.pkl', {'income': {'donors': {0: ([10, 20], [2, 3])}}}),
    ('models/family_models.pkl', {'donors': {(0, 0): [(np.array([25., 30.]), 100.)]}}),
    ('census/aggregates.pkl', {'gap': np.array([25, 30])}),
    ('census/aggregates.pkl', {'gap': {'value': [25, 30], 'count': [10, 2]}}),
])
def test_old_storage_does_not_replace_bundle(source, name, old):
    build()
    before = {p.relative_to(paths.bundle()): p.read_bytes()
              for p in paths.bundle().rglob('*') if p.is_file()}
    (source / name).write_bytes(pickle.dumps(old))
    with pytest.raises(ValueError):
        build()
    assert before == {p.relative_to(paths.bundle()): p.read_bytes()
                      for p in paths.bundle().rglob('*') if p.is_file()}
