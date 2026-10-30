"""Classify MatrAIx masks whose target is a Korean attribute, on 60,000 personas of the stage-0 generator:
violated by Korean-made values -> dropped; never violated -> kept as a rejection check.
Output: artifacts/masks/mask_class.pkl. Usage: python -m krpersona.matraix.classify_masks"""
import json, numpy as np, pandas as pd
from krpersona import paths
from krpersona.generate.base import Gen0
from krpersona.matraix.own_masks import korean_masks, mask_hits, rules
G = Gen0(paths.art('census', 'syn_s1_H.parquet'), paths.art('census', 'syn_s1_P.parquet'), seed=11, switch_on=True)
out = G.make(60000)
KM = korean_masks(G.K); rows = []
src = {a: r[0] for a, r in G.rows.items()}
for m in KM:
    h = mask_hits(out, m); n = int(h.sum()) if h is not None else -1
    rows.append((m['mask_id'], m['target'], src.get(m['target']), json.dumps(m['condition'], ensure_ascii=False), ', '.join(m['bad_values']), m.get('constraint_semantics'), n))
T = pd.DataFrame(rows, columns=['mask_id', '대상', '대상 방식', '조건', '금지 값', 'MatrAIx 분류', '6만 명 중 위반'])
T['처리'] = np.where(T['6만 명 중 위반'] > 0, '폐기(한국 값과 충돌)', np.where(T['6만 명 중 위반'] < 0, '해당 없음(조건 속성 없음)', '모순 검사로 유지'))
T.to_pickle(paths.art('masks', 'mask_class.pkl', mkdir=True)); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(T['처리'].value_counts().to_dict())
print(T[T['6만 명 중 위반'] > 0][['mask_id', '대상', '대상 방식', '조건', '금지 값', '6만 명 중 위반']].to_string(index=False))
kept = [m for m, r in zip(KM, rows) if r[-1] == 0]
V = rules(G, kept); print(V.sum().to_string()); print('any', V.any(axis=1).mean())
