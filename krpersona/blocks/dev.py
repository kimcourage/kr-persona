"""Developer status (step 5, round 2) from the EAPS B형 2025 상반기 file (KSCO 8차 3-digit occupation, employed, 19+).
classes among 1-digit occupation 2 (전문가 및 관련 종사자) workers:
  1 'Professional developer'    = 222 컴퓨터 시스템 및 소프트웨어 전문가, 223 네트워크 및 정보 보안 전문가, 224 데이터 전문가
  2 'Non-developer technologist' = 221 컴퓨터 하드웨어 및 통신공학 전문가, 225 정보 시스템 및 웹 운영자
  0 other professionals
model: multinomial logistic regression (L2, weights = 시도전국가중값) on sex, 7 age groups, 5 education levels, 12 industry groups.
Everyone outside 1-digit occupation 2 is 'Not a developer' (no data on hobby / student / former developers)."""
import sys, pickle, numpy as np, pandas as pd
from krpersona import paths
from krpersona.blocks import core
from krpersona.blocks.labor_income import KSIC, IND12
DEV, TECH = {222, 223, 224}, {221, 225}
LEVELS = dict(sex=2, ag7=7, edu5=5, ind12=12)

def load():
    e = pd.read_csv(paths.raw('eaps2025h1', 'type_b'), encoding='cp949', dtype=str)
    e['ho'] = core.holdout_mask('eaps_b', len(e))
    f = lambda c: pd.to_numeric(e[c], errors='coerce')
    e = e[(f('만연령') >= 19) & (f('경제활동인구상태코드') == 1)].copy(); f = lambda c: pd.to_numeric(e[c], errors='coerce')
    occ = f('3자리_8차직업소분류코드'); div = f('3자리_11차산업소분류코드') // 10
    sec = pd.Series([next((s for a, b, s in KSIC if a <= x <= b), None) if x == x else None for x in div], index=e.index)
    d = pd.DataFrame(index=e.index)
    d['sex'] = (f('성별코드') == 2).astype(int); d['ag7'] = core.ag7(f('만연령'))
    edu = f('교육정도_학력구분코드') + 1; sch = f('교육정도_수학구분코드').map({1: 1, 2: 2, 3: 5, 4: 4, 5: 3})
    d['edu5'] = core.edu5_census(edu.fillna(1), sch.fillna(1)); d['ind12'] = sec.map(IND12)
    d['occ1'] = occ // 100; d['y'] = np.select([occ.isin(DEV), occ.isin(TECH)], [1, 2], 0); d['w'] = f('시도전국가중값'); d['ho'] = e.ho.to_numpy()
    return d[d.ind12.notna() & d.occ1.notna()]

def design(X):
    """one-hot (first level dropped) of sex, ag7, edu5, ind12 -> (n, p)"""
    cols = [np.ones(len(X))]
    for j, (k, L) in enumerate(LEVELS.items()):
        for l in range(1, L): cols.append((X[:, j] == l).astype(float))
    return np.stack(cols, 1)

def fit():
    from sklearn.linear_model import LogisticRegression
    d = load(); d = d[~d.ho] if core.HOLDOUT else d; p = d[d.occ1 == 2]
    X = design(p[list(LEVELS)].to_numpy(int)); w = p.w.to_numpy(float); w = w / w.mean()
    m = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(X, p.y.to_numpy(), sample_weight=w)
    share = {c: float(np.average(p.y == c, weights=w)) for c in (0, 1, 2)}
    all_emp = {c: float((p.w * (p.y == c)).sum() / d.w.sum()) for c in (1, 2)}
    out = dict(coef=m.coef_, classes=m.classes_.tolist(), n=len(p), share_in_prof=share, share_of_employed=all_emp)
    pickle.dump(out, open(paths.art('models', f'dev_models{core.SUF}.pkl', mkdir=True), 'wb'))
    print('n professionals', len(p), 'class shares among professionals', {k: round(v, 4) for k, v in share.items()}, 'of all employed', {k: round(v, 4) for k, v in all_emp.items()})
    return out

def draw(M, X, rng):
    """X: (n, 4) sex, ag7, edu5, ind12 for 1-digit occupation 2 workers -> class 0/1/2"""
    L = design(X) @ M['coef'].T; L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
    c = P.cumsum(1); u = rng.random(len(X))[:, None]; return np.array(M['classes'])[np.minimum((c < u).sum(1), P.shape[1] - 1)]

if __name__ == '__main__':
    fit()
