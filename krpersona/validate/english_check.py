"""English ability (step 5): generated vs KGSS 2018 held-out respondents, by birth cohort (generation is cohort-matched: birth year = 2020 - age).
usage: KR_HOLDOUT=1 python english_check.py <personas.parquet> <internal.parquet> <out.pkl>"""
import sys, pickle, numpy as np, pandas as pd
from krpersona.blocks import core, kgss
assert core.HOLDOUT, 'run with KR_HOLDOUT=1'
LV = ['Fluent (C1-C2)', 'Intermediate (B1-B2)', 'Basic (A1-A2)', 'None']
COH = [(0, 1954, '1954년 이전'), (1955, 1964, '1955–1964'), (1965, 1974, '1965–1974'), (1975, 1984, '1975–1984'), (1985, 2001, '1985–2001')]

if __name__ == '__main__':
    df, C, B, X_, mx = kgss.load(keep_all=True)
    y = C['english_proficiency'].to_numpy(); ok = (df.YEAR == 2018).to_numpy() & df.ho.to_numpy() & ~np.isnan(y)
    by_r = (2018 - df.AGE.to_numpy())[ok]; yr = y[ok].astype(int); w = df.FINALWT.to_numpy(float)[ok]
    P = pd.read_parquet(sys.argv[1], columns=['english_proficiency', 'highest_education']); I = pd.read_parquet(sys.argv[2], columns=['age'])
    by_s = 2020 - I.age.to_numpy(); ys = P.english_proficiency.map({v: i for i, v in enumerate(LV)}).to_numpy()
    rows = []
    for lo, hi, lab in COH + [(0, 2001, '전체')]:
        mr = (by_r >= lo) & (by_r <= hi); ms = (by_s >= lo) & (by_s <= hi)
        pr = np.bincount(yr[mr], weights=w[mr], minlength=4) / w[mr].sum(); ps = np.bincount(ys[ms], minlength=4) / ms.sum()
        rows.append((lab, int(mr.sum()), *pr, *ps, 0.5 * np.abs(pr - ps).sum()))
    cols = ['출생 연도', 'KGSS 2018 분리분 응답 수'] + [f'분리분 {v}' for v in LV] + [f'생성 {v}' for v in LV] + ['총변동거리']
    T = pd.DataFrame(rows, columns=cols)
    pickle.dump(T, open(sys.argv[3], 'wb')); pd.set_option('display.width', 250); print(T.round(3).to_string(index=False))
