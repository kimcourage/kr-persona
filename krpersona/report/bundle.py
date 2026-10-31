"""Copy the current generation inputs into bundle/ without further transformation.

Estimation stores family group means, income quantiles and census frequency
tables directly. Health models omit respondent design matrices. This command
checks those storage conventions before replacing any bundled files.
These checks do not establish source-provider permission or disclosure safety.
Usage: python -m krpersona.report.bundle
"""
from collections import Counter
import datetime
import hashlib
import json
import pickle
import shutil

import numpy as np

from krpersona import paths, __version__
from krpersona.generate.resources import BUNDLE_FILES


def check_storage(root):
    def read(name):
        with (root / name).open('rb') as handle:
            return pickle.load(handle)

    sem = read('models/health_models.pkl')['sem']
    if {'X', 'idx'} & sem.keys():
        raise ValueError('건강 모형에 응답자별 설계행렬 또는 행 번호가 포함되어 있습니다.')
    for values, weights in read('models/labor_income_models.pkl')['income']['donors'].values():
        if len(values) != 200 or len(weights) != 200 or not np.all(np.asarray(weights) == 1):
            raise ValueError('소득 모형에는 구간별 200개 분위값과 동일 가중치가 필요합니다.')
    groups = Counter((tuple(gaps), float(weight))
                     for pool in read('models/family_models.pkl')['donors'].values()
                     for gaps, weight in pool)
    if not groups or min(groups.values()) < 5:
        raise ValueError('가족 모형의 묶음 평균 구조를 확인해 주세요. 최소 묶음 크기는 5입니다.')
    gaps = read('census/aggregates.pkl')['gap']
    if not isinstance(gaps, dict) or not {'value', 'count'} <= gaps.keys():
        raise ValueError('자녀 연령 차이는 개별 기록 배열이 아닌 빈도표여야 합니다.')
    if len(gaps['value']) != len(gaps['count']) or not len(gaps['count']) or min(gaps['count']) < 10:
        raise ValueError('자녀 연령 차이 빈도표의 범주별 빈도는 10 이상이어야 합니다.')


def build():
    source, destination = paths.art(), paths.bundle()
    for name in sorted(BUNDLE_FILES):
        if not (source / name).is_file():
            raise FileNotFoundError(f'생성 입력 파일이 없습니다: {name}')
    check_storage(source)
    entries = []
    for name in sorted(BUNDLE_FILES):
        src, dst = source / name, destination / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        data = dst.read_bytes()
        entries.append(dict(file=name, bytes=len(data), md5=hashlib.md5(data).hexdigest()))
    manifest = dict(version=__version__, built=datetime.date.today().isoformat(),
                    matraix_commit=paths.MATRAIX_COMMIT, files=entries)
    (destination / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(len(entries), 'files,', round(sum(item['bytes'] for item in entries) / 1e6, 1), 'MB in', destination)


if __name__ == '__main__':
    build()
