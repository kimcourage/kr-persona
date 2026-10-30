# Memorization test: synthesizer trained on half A; compare how often synthetic records hit A-uniques vs B-uniques (B never seen)
# usage: python -m krpersona.census.holdout <tag of a synthesis trained on half A, e.g. syn_s7_f0.5> <seed of the half split>
import sys, numpy as np, pandas as pd, pickle
from krpersona import paths
DIR = paths.art('census'); TAG, SEED = sys.argv[1], int(sys.argv[2])
H = pd.read_parquet(f'{DIR}/H.parquet'); P = pd.read_parquet(f'{DIR}/P.parquet')
A = set(H.hid.sample(frac=0.5, random_state=SEED)); P['half'] = np.where(P.hid.isin(A), 'A', 'B')
H['half'] = np.where(H.hid.isin(A), 'A', 'B')
SH = pd.read_parquet(f'{DIR}/{TAG}_H.parquet'); SP = pd.read_parquet(f'{DIR}/{TAG}_P.parquet').merge(SH[['hid', 'sido', 'dong', 'k', 'tenure_h']], on='hid')
P = P.merge(H[['hid', 'k', 'tenure_h']], on='hid')
def pub(D):
    D = D.copy()
    D['agebr'] = pd.cut(D.age, [-1, 18, 24, 34, 44, 54, 64, 74, 84, 200], labels=False)
    D['urb'] = np.select([(D.dong == 3) & D.sido.isin([11, 21, 22, 23, 24, 25, 26]), D.dong == 3, D.dong == 1], [0, 1, 2], 3)
    D['grad'] = (D.sch == 1).astype(int); D['kc'] = pd.cut(D.k, [0, 1, 2, 4, 99], labels=False); D['limany'] = (D.lim.fillna(0) > 0).astype(int)
    D['persona'] = (D.nat == 1) & (D.bplace != 4) & (D.age >= 19); return D
P, SP = pub(P), pub(SP)
def hsig(D):
    g = D.sort_values(['hid', 'rel', 'age']).assign(t=lambda x: x.rel.astype(int).astype(str) + ':' + x.sex.astype(int).astype(str) + ':' + x.age.astype(int).astype(str))
    return g.groupby('hid').t.agg(','.join)
QS = {'개인: 내부 핵심 13개(정확한 나이, 시도 포함)': ['sido', 'dong', 'sex', 'age', 'edu', 'sch', 'mar', 'status', 'occ', 'ind', 'k', 'rel', 'nat'],
      '개인: 내부 20개': ['sido', 'dong', 'sex', 'age', 'edu', 'sch', 'mar', 'status', 'occ', 'ind', 'k', 'rel', 'nat', 'bplace', 'r5', 'jten', 'place', 'mode', 'lim', 'births'],
      '페르소나 공개 수준 12개(연령대, 도시화, 시도 없음)': ['agebr', 'sex', 'urb', 'edu', 'grad', 'mar', 'econ', 'status', 'occ', 'ind', 'kc', 'tenure_h'],
      '페르소나 공개 수준 12개(정확한 나이)': ['age', 'sex', 'urb', 'edu', 'grad', 'mar', 'econ', 'status', 'occ', 'ind', 'kc', 'tenure_h'],
      '페르소나 공개 수준 16개': ['agebr', 'sex', 'urb', 'edu', 'grad', 'mar', 'econ', 'status', 'occ', 'ind', 'kc', 'tenure_h', 'rel', 'limany', 'mode', 'jten']}
def rate(R, S, cols):
    X = pd.concat([R[cols], S[cols]], ignore_index=True)
    for c in cols:
        if not (X[c].dtype == object or pd.api.types.is_string_dtype(X[c])): X[c] = X[c].fillna(-1)
    g = X.groupby(cols, dropna=False).ngroup().to_numpy(); gr, gs = g[:len(R)], g[len(R):]
    cnt = np.bincount(gr, minlength=g.max() + 1); uq = cnt == 1
    return uq.sum() / len(R) * 100, uq[gs].mean() * 100
rows = []
for lab, cols in QS.items():
    pers = '페르소나' in lab
    RA = P[(P.half == 'A') & (P.persona if pers else True)]; RB = P[(P.half == 'B') & (P.persona if pers else True)]; S = SP[SP.persona] if pers else SP
    ua, sa = rate(RA, S, cols); ub, sb = rate(RB, S, cols)
    rows.append((lab, ua, ub, sa, sb, sa - sb))
for nm, D, SD in [('가구: 시도, 동읍면, 가구원 전원의 (관계, 성, 나이)', None, None)]:
    HA = H[H.half == 'A'].set_index('hid')[['sido', 'dong']].join(hsig(P[P.half == 'A']).rename('mem'))
    HB = H[H.half == 'B'].set_index('hid')[['sido', 'dong']].join(hsig(P[P.half == 'B']).rename('mem'))
    HS = SH.set_index('hid')[['sido', 'dong']].join(hsig(SP).rename('mem'))
    ua, sa = rate(HA, HS, ['sido', 'dong', 'mem']); ub, sb = rate(HB, HS, ['sido', 'dong', 'mem']); rows.append((nm, ua, ub, sa, sb, sa - sb))
    kk = H.set_index('hid').k; ks = SH.set_index('hid').k
    ua, sa = rate(HA[kk.reindex(HA.index) > 1], HS[ks.reindex(HS.index) > 1], ['sido', 'dong', 'mem']); ub, sb = rate(HB[kk.reindex(HB.index) > 1], HS[ks.reindex(HS.index) > 1], ['sido', 'dong', 'mem'])
    rows.append(('가구(2인 이상): 위와 같음', ua, ub, sa, sb, sa - sb))
R = pd.DataFrame(rows, columns=['키 변수', '학습 절반 A: 유일 비율(%)', '보지 않은 절반 B: 유일 비율(%)', '합성이 A의 유일 조합과 일치(%)', '합성이 B의 유일 조합과 일치(%)', '차이(%p)'])
pickle.dump(R, open(f'{DIR}/{TAG}_holdout.pkl', 'wb')); pd.set_option('display.width', 250); print(R.round(2).to_string(index=False))
