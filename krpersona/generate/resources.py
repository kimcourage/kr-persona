"""Validate generation inputs before loading models or producing outputs."""
import hashlib
import json
import os


BUNDLE_FILES = frozenset({
    'models/pers_params.json', 'models/value_params.json',
    'models/health_models.pkl', 'models/labor_income_models.pkl',
    'models/kgss_models.pkl', 'models/family_models.pkl', 'models/dev_models.pkl',
    'masks/mask_class.pkl', 'masks/rule_masks.pkl',
    'census/syn_s1_H.parquet', 'census/syn_s1_P.parquet', 'census/aggregates.pkl',
})


def check_bundle(directory, selected=None):
    """Check the incoming bundle's explicit file list, sizes and MD5 checksums.

    This detects missing or changed files; it does not assess disclosure risk
    or determine whether the source providers permit publication.
    """
    manifest = json.loads((directory / 'MANIFEST.json').read_text(encoding='utf-8'))
    entries = manifest['files']
    names = [entry['file'] for entry in entries]
    if len(names) != len(BUNDLE_FILES) or set(names) != BUNDLE_FILES:
        raise ValueError('bundle/MANIFEST.json의 파일 구성이 올바르지 않습니다.')
    selected = BUNDLE_FILES if selected is None else set(selected)
    for entry in entries:
        name = entry['file']
        if name not in selected:
            continue
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'생성 입력 파일이 없거나 일반 파일이 아닙니다: {name}')
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.md5(data).hexdigest() != entry['md5']:
            raise ValueError(f'생성 입력 파일이 변경되었거나 손상되었습니다: {name}')


def check_inputs(holdout=False):
    from krpersona import paths

    required = set(BUNDLE_FILES)
    if os.environ.get('KR_NO_RULE_MASKS') == '1':
        required.discard('masks/rule_masks.pkl')
    if holdout:
        required = {name.replace('syn_s1_', 'syn_s5_f0.8_') for name in required}
        required = {name.rsplit('.', 1)[0] + '_80.' + name.rsplit('.', 1)[1]
                    if name.startswith('models/') else name for name in required}
        required |= {'census/H.parquet', 'census/P.parquet'}
    missing = sorted(name for name in required if not paths.inp(*name.split('/')).is_file())
    if missing:
        hint = ('검증 모드는 원자료와 별도로 추정한 80% 모형이 필요합니다.' if holdout else
                'bundle 폴더를 포함하여 저장소 전체를 준비해 주세요.')
        raise FileNotFoundError('생성 입력 파일이 없습니다: ' + ', '.join(missing) + '\n' + hint)
    bundled = {name for name in required
               if paths.inp(*name.split('/')).resolve() == paths.bundle(name).resolve()}
    if bundled:
        check_bundle(paths.bundle(), bundled)
    if not paths.graph_path().is_file() or not paths.dims_path().is_file():
        raise FileNotFoundError('MatrAIx 경로를 확인해 주세요. 설치 방법은 README에 있습니다.')


if __name__ == '__main__':
    check_inputs()
    print('생성 입력 파일 및 MatrAIx 경로 확인 완료')
