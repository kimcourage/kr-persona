"""Generation-only inputs, local overrides and missing validation resources."""
import hashlib
import json

import pytest

from krpersona import paths
from krpersona.generate.resources import BUNDLE_FILES, check_bundle, check_inputs


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    files = []
    for name in sorted(BUNDLE_FILES):
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = name.encode()
        path.write_bytes(data)
        files.append(dict(file=name, bytes=len(data), md5=hashlib.md5(data).hexdigest()))
    (bundle / 'MANIFEST.json').write_text(json.dumps(dict(files=files)))
    monkeypatch.setattr(paths, '_C', {})
    monkeypatch.delenv('KRP_DATA', raising=False)
    monkeypatch.delenv('KR_NO_RULE_MASKS', raising=False)
    monkeypatch.setenv('KRP_BUNDLE', str(bundle))
    monkeypatch.setenv('KRP_ARTIFACTS', str(tmp_path / 'artifacts'))
    monkeypatch.setenv('KRP_MATRAIX', str(tmp_path / 'matraix'))
    for path in (paths.graph_path(), paths.dims_path()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    return bundle


def test_generation_inputs_do_not_need_raw_data(inputs):
    check_inputs()
    with pytest.raises(RuntimeError, match='data_dir'):
        paths.data()


def test_local_model_overrides_bundle_per_file(inputs):
    name = 'models/dev_models.pkl'
    local = paths.art(*name.split('/'), mkdir=True)
    local.write_bytes(b'local model')
    (inputs / name).unlink()
    assert paths.inp(*name.split('/')) == local
    check_inputs()


def test_changed_bundle_file_is_rejected(inputs):
    (inputs / 'models/dev_models.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='변경되었거나 손상'):
        check_inputs()


def test_unexpected_manifest_entry_is_rejected(inputs):
    manifest = inputs / 'MANIFEST.json'
    data = json.loads(manifest.read_text())
    data['files'][0]['file'] = '../private.pkl'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='파일 구성'):
        check_bundle(inputs)


def test_missing_file_has_actionable_error(inputs):
    (inputs / 'census/syn_s1_P.parquet').unlink()
    with pytest.raises(FileNotFoundError, match='bundle 폴더'):
        check_inputs()


def test_validation_requires_separate_inputs(inputs):
    with pytest.raises(FileNotFoundError, match='80% 모형'):
        check_inputs(holdout=True)


def test_mask_compilation_can_omit_previous_rule_masks(inputs, monkeypatch):
    (inputs / 'masks/rule_masks.pkl').unlink()
    monkeypatch.setenv('KR_NO_RULE_MASKS', '1')
    check_inputs()


def test_missing_matraix_has_actionable_error(inputs):
    paths.graph_path().unlink()
    with pytest.raises(FileNotFoundError, match='MatrAIx 경로'):
        check_inputs()
