"""Limitation 10 check: correlation of value scores with KGSS attitudes, KGSS 2021 held-out respondents vs generated personas
(80% models, with and without the values -> attitudes links). Ordinal attributes only (religion, religiosity, closest country are nominal).
usage: KR_HOLDOUT=1 python value_link_check.py <internal_with_links.parquet> <internal_without_links.parquet> <out.pkl>"""
import sys, json, pickle, numpy as np, pandas as pd
from krpersona import paths
from krpersona.blocks import core, kgss, values as VM
assert core.HOLDOUT, 'run with KR_HOLDOUT=1'
NOMINAL = {'demo_religion_affiliation', 'religiosity', 'kr_country_felt_closest'}

def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))

if __name__ == '__main__':
    K = pickle.load(open(paths.art('models', 'kgss_models_80.pkl'), 'rb')); VP = json.load(open(paths.art('models', 'value_params_80.json')))
    df, C, B, X_, mx = kgss.load(keep_all=True)
    it = [f'VALUES{i}' for i in range(1, 23)]
    ok = ((df.YEAR == 2021) & df.ho & df[it].apply(lambda c: c.between(1, 6)).all(axis=1)).to_numpy(); d = df[ok]
    V = np.stack([np.mean([7 - d[f'VALUES{i}'].to_numpy(float) for i, v in VM.ITEM_VAL.items() if v == val], axis=0) for val in VM.VALS], 1)
    vbar, sd = np.array(VP['vbar']), np.array(VP['sd'])
    z = (V - V.mean(1, keepdims=True) - vbar[None, :] + vbar.mean()) / sd[None, :]      # same scale as the generated value scores v_*
    w = d.FINALWT.to_numpy(float)
    A = [a for a in K['vlinks'] if a not in NOMINAL]
    cols = [f'k_{a}' for a in A] + [f'v_{j}' for j in range(10)]
    I1, I0 = pd.read_parquet(sys.argv[1], columns=cols), pd.read_parquet(sys.argv[2], columns=cols)
    rows = []
    for a in A:
        y = kgss.recode(C[a].fillna(-99).to_numpy(), K['models'][a]['cat']['codes'])[ok]; m = y >= 0
        for j in range(10):
            rows.append((a, VP['ko'][j], int(m.sum()), wcorr(z[m, j], y[m].astype(float), w[m]),
                         float(np.corrcoef(I1[f'v_{j}'], I1[f'k_{a}'])[0, 1]), float(np.corrcoef(I0[f'v_{j}'], I0[f'k_{a}'])[0, 1])))
    R = pd.DataFrame(rows, columns=['속성', '가치', '분리분 응답 수', '분리분 상관', '연결 후 생성 상관', '연결 전 생성 상관'])
    pickle.dump(R, open(sys.argv[3], 'wb'))
    n = int(R['분리분 응답 수'].median())
    print('attributes', len(A), 'pairs', len(R), 'held-out n (median)', n, 'noise |r| ~', round(np.sqrt(2 / np.pi / n), 3))
    for c in ['분리분 상관', '연결 후 생성 상관', '연결 전 생성 상관']: print(c, 'mean |r|', round(R[c].abs().mean(), 3))
    print('pattern corr with held-out: with links', round(np.corrcoef(R['분리분 상관'], R['연결 후 생성 상관'])[0, 1], 3),
          'without', round(np.corrcoef(R['분리분 상관'], R['연결 전 생성 상관'])[0, 1], 3))
