# Utility and disclosure checks: synthetic vs real census 2% sample
# usage: python -m krpersona.census.check <tag>
import sys, itertools, pickle, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from krpersona import paths
DIR = paths.art('census'); TAG = sys.argv[1]
H = pd.read_parquet(f'{DIR}/H.parquet'); P = pd.read_parquet(f'{DIR}/P.parquet')
SH = pd.read_parquet(f'{DIR}/{TAG}_H.parquet'); SP = pd.read_parquet(f'{DIR}/{TAG}_P.parquet')
CNT = [f'n{r:02d}' for r in range(2, 15)]
SP = SP.merge(SH[['hid', 'sido', 'dong', 'k', 'gtype', 'tenure_h', 'car', 'dwell', 'pet']], on='hid')
P = P.merge(H[['hid', 'k', 'gtype', 'tenure_h', 'car', 'dwell', 'pet']], on='hid')
SP['hw'] = 1.0; SH['hw'] = 1.0
def prep(D):
    D = D.copy(); D['age5'] = (D.age // 5).clip(upper=17); D['age10'] = pd.cut(D.age, [-1, 18, 29, 39, 49, 59, 69, 200], labels=False)
    D['kc'] = D.k.clip(upper=5); D['limany'] = np.where(D.lim.isna(), np.nan, (D.lim > 0).astype(float))
    D['persona'] = ((D.nat == 1) & (D.bplace != 4) & (D.age >= 19)).astype(int); return D
P, SP = prep(P), prep(SP)
def dist(D, cols, w='hw'):
    g = D.groupby(cols, dropna=False)[w].sum(); return g / g.sum()
def tvd(a, b):
    i = a.index.union(b.index); return 0.5 * np.abs(a.reindex(i, fill_value=0) - b.reindex(i, fill_value=0)).sum() * 100
LAB = {'agebr': '연령대(MatrAIx)', 'urb': '도시화', 'grad': '졸업 여부', 'kc4': '가구원 수(4구간)', 'mem': '가구원 목록', 'sido': '시도', 'dong': '동읍면', 'gtype': '세대유형', 'hkind': '가구구분', 'k': '가구원 수', 'dwell': '거처 종류', 'tenure_h': '점유 형태',
       'car': '자동차 보유', 'pet': '반려동물 조합', 'sex': '성', 'age5': '나이 5세', 'nat': '국적 구분', 'bplace': '출생지', 'bsido': '출생 시도',
       'r5': '5년 전 거주지', 'r1': '1년 전 거주지', 'edu': '교육 정도', 'sch': '수학 구분', 'mar': '혼인 상태', 'fmage': '초혼 연령', 'econ': '경제활동',
       'status': '종사상 지위', 'occ': '직업', 'ind': '산업', 'jten': '근속', 'place': '근로 장소', 'comm': '통근통학', 'mode': '통근 수단',
       'lim': '활동 제약 조합', 'limany': '활동 제약 있음', 'births': '출산아 수', 'deaths': '사망 자녀 수', 'prejob': '결혼 전 취업', 'brk': '경력 단절',
       'soc_pol': '정치분야단체', 'soc_soc': '사회분야단체', 'rel': '가구주 관계', 'age10': '연령대', 'kc': '가구원 수(5+)'}
out = {}
# 1. univariate
rows = []
for c in ['sido', 'dong', 'gtype', 'hkind', 'k', 'dwell', 'tenure_h', 'car', 'pet']:
    rows.append(('가구', LAB[c], len(H[c].dropna().unique()), tvd(dist(H, [c]), dist(SH, [c]))))
for c in ['sex', 'age5', 'rel', 'nat', 'bplace', 'bsido', 'r5', 'r1', 'edu', 'sch', 'mar', 'fmage', 'econ', 'status', 'occ', 'ind', 'jten', 'place',
          'comm', 'mode', 'limany', 'lim', 'births', 'deaths', 'prejob', 'brk', 'soc_pol', 'soc_soc']:
    rows.append(('개인', LAB[c], len(P[c].dropna().unique()), tvd(dist(P, [c]), dist(SP, [c]))))
out['uni'] = pd.DataFrame(rows, columns=['단위', '변수', '범주 수', '총변동거리(%p)'])
# 2. bivariate among key person variables
K = ['sex', 'age10', 'sido', 'dong', 'rel', 'kc', 'edu', 'mar', 'econ', 'status', 'occ', 'ind', 'comm', 'limany', 'tenure_h', 'car']
rows = [(LAB[a], LAB[b], tvd(dist(P, [a, b]), dist(SP, [a, b]))) for a, b in itertools.combinations(K, 2)]
out['bi'] = pd.DataFrame(rows, columns=['변수 1', '변수 2', '총변동거리(%p)']).sort_values('총변동거리(%p)', ascending=False)
# 3. persona population (domestic-born Korean adults): key 3-way tables, weighted
PR, SR = P[P.persona == 1], SP[SP.persona == 1]
rows = []
for cols in [['sex', 'age5', 'edu'], ['age5', 'mar', 'econ'], ['sex', 'occ', 'ind'], ['sido', 'dong', 'age10'], ['age10', 'rel', 'kc'], ['sex', 'age10', 'status', 'occ'],
             ['age10', 'tenure_h', 'car'], ['sex', 'age10', 'mar', 'births']]:
    rows.append((' × '.join(LAB[c] for c in cols), tvd(dist(PR, cols), dist(SR, cols))))
rows.append(('페르소나 대상자 수(합성/실제)', len(SR) / len(PR) * 100))
out['tri'] = pd.DataFrame(rows, columns=['교차표', '총변동거리(%p)'])
# 4. household structure: age gaps
rows = []
hd = P[P.rel == 1].set_index('hid').age; shd = SP[SP.rel == 1].set_index('hid').age
for r, lab in [(2, '배우자 − 가구주'), (3, '자녀 − 가구주'), (5, '가구주 부모 − 가구주'), (7, '손자녀 − 가구주'), (10, '형제자매 − 가구주')]:
    a = P[P.rel == r]; b = SP[SP.rel == r]
    da = a.age - a.hid.map(hd); db = b.age - b.hid.map(shd)
    for q in [0.01, 0.1, 0.5, 0.9, 0.99]:
        rows.append((lab, f'{int(q*100)}%', np.quantile(da, q), np.quantile(db, q)))
out['gap'] = pd.DataFrame(rows, columns=['나이 차', '분위', '실제', '합성'])
# 5. logic checks (same rules on both)
def logic(D, Hh):
    hd = D[D.rel == 1].set_index('hid'); sp = D[D.rel == 2]
    ch = D[D.rel == 3]; chd = ch.age - ch.hid.map(hd.age)
    return {
        '배우자인데 혼인 상태가 배우자 있음이 아님': int((sp.mar != 2).sum()),
        '배우자가 15세 미만': int((sp.age < 15).sum()),
        '배우자와 가구주가 같은 성': int((sp.sex.to_numpy() == sp.hid.map(hd.sex).to_numpy()).sum()),
        '자녀가 가구주보다 12살 미만 어림이 아님(나이 차 > −12)': int((chd > -12).sum()),
        '15세 미만인데 취업': int(((D.age < 15) & D.econ.isin([1, 2, 3])).sum()),
        '19세 미만인데 4년제 졸업 이상': int(((D.age < 19) & (D.edu >= 6) & (D.sch == 1)).sum()),
        '초혼 연령 구간 하한이 현재 나이보다 큼': int((D.fmage.map({1: 0, 2: 15, 3: 20, 4: 25, 5: 30, 6: 35, 7: 40}) > D.age).sum()),
        '근속 20년 이상인데 35세 미만': int(((D.jten == 8) & (D.age < 35)).sum()),
        '사망 자녀 수 > 출산아 수': int((D.deaths > D.births).sum()),
        '1인 가구인데 가구원 수 ≠ 1': int(((Hh.gtype == 91) & (Hh.k != 1)).sum()),
        '가구주 나이 < 15': int((hd.age < 15).sum()),
    }
lr, ls = logic(P, H), logic(SP, SH)
out['logic'] = pd.DataFrame([(k, lr[k], lr[k] / len(P) * 1e5, ls[k], ls[k] / len(SP) * 1e5) for k in lr], columns=['규칙', '실제 건수', '실제(10만 명당)', '합성 건수', '합성(10만 명당)'])
# 6. propensity: can a classifier tell real from synthetic?
F = ['sido', 'dong', 'k', 'gtype', 'rel', 'sex', 'age', 'nat', 'bplace', 'r5', 'r1', 'edu', 'sch', 'mar', 'fmage', 'econ', 'status', 'occ', 'ind', 'jten', 'place',
     'comm', 'mode', 'lim', 'births', 'deaths', 'prejob', 'brk', 'soc_pol', 'soc_soc', 'tenure_h', 'car', 'dwell', 'pet']
rng = np.random.default_rng(0); n = min(100000, len(SP))
ra = P.sample(n, weights='hw', random_state=1, replace=True); sa = SP.sample(n, random_state=2)
X = pd.concat([ra[F], sa[F]]).to_numpy(float); y = np.r_[np.zeros(n), np.ones(n)]
pr = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), X, y, cv=3, method='predict_proba')[:, 1]
auc = roc_auc_score(y, pr); pmse = np.mean((pr - 0.5) ** 2)
# null reference: two independent weighted draws from the real data
rb = P.sample(n, weights='hw', random_state=3, replace=True)
X0 = pd.concat([ra[F], rb[F]]).to_numpy(float)
pr0 = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), X0, y, cv=3, method='predict_proba')[:, 1]
out['prop'] = pd.DataFrame([('실제 대 합성', roc_auc_score(y, pr), pmse), ('실제 대 실제(기준선)', roc_auc_score(y, pr0), np.mean((pr0 - 0.5) ** 2))],
                           columns=['비교', 'AUC', 'pMSE'])
# 7. disclosure: replicated uniques
rows = []
def ru(R, S, cols, lab):
    A = pd.concat([R[cols], S[cols]], ignore_index=True)
    for c in cols:
        if A[c].dtype != object and not pd.api.types.is_string_dtype(A[c]): A[c] = A[c].fillna(-1)
    g = A.groupby(cols, dropna=False).ngroup().to_numpy(); gr, gs = g[:len(R)], g[len(R):]
    cnt = np.bincount(gr, minlength=g.max() + 1); uq = cnt == 1; m = uq[gs]
    rows.append((lab, ', '.join(LAB.get(c, c) for c in cols), int(uq.sum()), uq.sum() / len(R) * 100, int(m.sum()), m.mean() * 100))
Q1 = ['sido', 'dong', 'sex', 'age', 'edu', 'sch', 'mar', 'status', 'occ', 'ind', 'k', 'rel', 'nat']
ru(P, SP, Q1, '개인: 인구학적 핵심 13개')
ru(P, SP, Q1 + ['bplace', 'r5', 'jten', 'place', 'mode', 'lim', 'births'], '개인: 20개')
def hsig(D):
    g = D.sort_values(['hid', 'rel', 'age']).assign(t=lambda x: x.rel.astype(int).astype(str) + ':' + x.sex.astype(int).astype(str) + ':' + x.age.astype(int).astype(str))
    return g.groupby('hid').t.agg(','.join)
HR = H.set_index('hid')[['sido', 'dong']].join(hsig(P).rename('mem')).reset_index(); HSY = SH.set_index('hid')[['sido', 'dong']].join(hsig(SP).rename('mem')).reset_index()
ru(HR, HSY, ['sido', 'dong', 'mem'], '가구: 시도, 동읍면, 가구원 전원의 (관계, 성, 나이)')
k2 = H.set_index('hid').k > 1; ks2 = SH.set_index('hid').k > 1
ru(HR[HR.hid.map(k2)], HSY[HSY.hid.map(ks2)], ['sido', 'dong', 'mem'], '가구(2인 이상): 위와 같음')
def pubk(D):
    D = D.copy(); D['agebr'] = pd.cut(D.age, [-1, 18, 24, 34, 44, 54, 64, 74, 84, 200], labels=False)
    D['urb'] = np.select([(D.dong == 3) & D.sido.isin([11, 21, 22, 23, 24, 25, 26]), D.dong == 3, D.dong == 1], [0, 1, 2], 3)
    D['grad'] = (D.sch == 1).astype(int); D['kc4'] = pd.cut(D.k, [0, 1, 2, 4, 99], labels=False); return D
PRp, SRp = pubk(PR), pubk(SR)
QP = ['agebr', 'sex', 'urb', 'edu', 'grad', 'mar', 'econ', 'status', 'occ', 'ind', 'kc4', 'tenure_h']
ru(PRp, SRp, QP, '페르소나 공개 수준 12개(연령대)')
ru(PRp, SRp, ['age'] + QP[1:], '페르소나 공개 수준 12개(정확한 나이)')
out['disc'] = pd.DataFrame(rows, columns=['단위', '키 변수', '실제에서 유일한 조합 수', '실제 중 비율(%)', '그 조합과 같은 합성 기록 수', '합성 중 비율(%)'])
pickle.dump(out, open(f'{DIR}/{TAG}_check.pkl', 'wb'))
pd.set_option('display.width', 250)
for k, v in out.items():
    print('==', k); print(v.round(3).head(40).to_string(index=False))
