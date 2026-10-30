"""Generated personas vs the source surveys: marginal distributions per block (total variation distance, %p)."""
import sys, json, pickle, numpy as np, pandas as pd
from krpersona import paths
EST = paths.art('models')

def dist(codes, w, K):
    codes = np.asarray(codes); w = np.asarray(w, float); k = codes >= 0
    return np.bincount(codes[k].astype(int), weights=w[k], minlength=K)[:K] / w[k].sum()
def tvd(p, q): return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum() * 100
def row(block, a, src, labels, p, q, note=''):
    return dict(블록=block, 속성=a, 자료=src, 범주=' / '.join(map(str, labels)), 조사=' / '.join(f'{x*100:.1f}' for x in p), 생성=' / '.join(f'{x*100:.1f}' for x in q),
                **{'총변동거리(%p)': tvd(p, q)}, 비고=note)

def compare(int_path, per_path):
    I = pd.read_parquet(int_path); P = pd.read_parquet(per_path); R = []; one = np.ones(len(I))
    # ---------------- health (KNHANES 2022)
    from krpersona.blocks import health as HM
    d = HM.load(); H = pickle.load(open(f'{EST}/health_models.pkl', 'rb'))
    for a in HM.ORDER:
        node, sets, labels = HM.SETS[a]; K = len(sets)
        R.append(row('건강', node, '국민건강영양조사 2022', labels, dist(d[a].fillna(-1), d.w, K), dist(I['h_' + a], one, K)))
    it = [f'p{i}' for i in range(1, 10)]; ok = d[it].notna().all(1); phq = d.loc[ok, it].sum(1)
    it2 = [f'g{i}' for i in range(1, 8)]; ok2 = d[it2].notna().all(1); gad = d.loc[ok2, it2].sum(1)
    cut = lambda x, c: np.digitize(x, c)
    R.append(row('건강', 'kr_depression_level', '국민건강영양조사 2022 PHQ-9', ['0–4', '5–9', '10–14', '15–19', '20+'], dist(cut(phq, [5, 10, 15, 20]), d.w[ok], 5), dist(cut(I.phq, [5, 10, 15, 20]), one, 5)))
    R.append(row('건강', 'kr_anxiety_level', '국민건강영양조사 2022 GAD-7', ['0–4', '5–9', '10–14', '15+'], dist(cut(gad, [5, 10, 15]), d.w[ok2], 4), dist(cut(I.gad, [5, 10, 15]), one, 4)))
    ok3 = ok & ok2 & d.general.notna(); ph, ga, gn = d.loc[ok3, it].sum(1), d.loc[ok3, it2].sum(1), d.loc[ok3, 'general']
    mh = lambda p_, g_, gen: np.select([(p_ >= 20) | (g_ >= 15), (p_ >= 10) | (g_ >= 10), (p_ >= 5) | (g_ >= 5), np.isin(gen, [0, 1])], [3, 2, 1, 0], 1)
    R.append(row('건강', 'health_mental_health', '국민건강영양조사 2022(규칙)', ['Thriving', 'Stable', 'Struggling', 'In crisis'], dist(mh(ph, ga, gn), d.w[ok3], 4), dist(mh(I.phq, I.gad, I.h_general), one, 4)))
    # ---------------- labor (EAPS 2025 H1)
    from krpersona.blocks import labor_income as LI
    e = LI.eaps(); emp = e[e.status.between(1, 6)]
    gw = P.demo_employment_status.astype(str).isin(['Full-time', 'Part-time', 'Self-employed', 'Gig / freelance', 'Unpaid family worker'])
    cs = P.company_size.astype(str)
    R.append(row('노동', 'company_size(규칙)', '지역별고용조사 2025 상반기 A형 취업자', ['Solo / freelance', 'Public sector', '그 밖'],
                 [np.average(emp.status == 5, weights=emp.w), np.average(emp.indO & (emp.status != 5), weights=emp.w), 0], [np.mean(cs[gw] == 'Solo / freelance'), np.mean(cs[gw] == 'Public sector'), 0],
                 '그 밖 = 1 − 앞의 둘(표시 생략)'))
    s = e[(e.status.isin([1, 2, 3, 4, 6])) & ~e.indO & e['size'].between(0, 5)]
    gs = cs[cs.isin(LI.SIZE_OUT)].map({v: i for i, v in enumerate(LI.SIZE_OUT)})
    R.append(row('노동', 'company_size(규모)', '지역별고용조사 2025 상반기 A형(규칙 대상 제외)', LI.SIZE_LAB, dist(s['size'], s.w, 6), dist(gs, np.ones(len(gs)), 6)))
    ww = e[e.status.isin([1, 2, 3]) & e.wage.notna()]
    R.append(row('노동', '월 임금 구간(내부 값)', '지역별고용조사 2025 상반기 A형 임금근로자', ['<100'] + [f'{a}–' for a in LI.WAGE_CUTS], dist(ww.wband, ww.w, 9), dist(I.wage_band[I.wage_band >= 0], np.ones((I.wage_band >= 0).sum()), 9), '만원/월'))
    # ---------------- income (HFCS 2025), adult-weighted
    h = LI.hfcs(); m = pd.read_csv(paths.raw('hfcs2025', 'member'), encoding='cp949', dtype=str)
    m = m[pd.to_numeric(m['가구원정보_동거코드'], errors='coerce').isin([1, 2]) & (pd.to_numeric(m['가구원정보_만연령'], errors='coerce') >= 19)]
    hh = pd.read_csv(paths.raw('hfcs2025', 'household'), encoding='cp949', dtype=str, usecols=['MD제공용_가구고유번호'])
    nad = hh['MD제공용_가구고유번호'].map(m.groupby('MD제공용_가구고유번호').size()).fillna(0).to_numpy(); aw = h.w.to_numpy() * nad
    usd = np.digitize(h.inc, LI.USD); lab = ['<$25k', '$25k-50k', '$50k-100k', '$100k-200k', '$200k+']
    R.append(row('소득', 'demo_household_income', '가계금융복지조사 2025(성인 가중)', lab, dist(usd, aw, 5), dist(P.demo_household_income.astype(str).map({v: i for i, v in enumerate(lab)}), one, 5),
                 '조사는 2024년 소득, 생성은 2020년 센서스 가구 구성에 조사 모형을 적용'))
    ses = ['Low income', 'Lower-middle', 'Middle', 'Upper-middle', 'High income']
    R.append(row('소득', 'socioeconomic_band(규칙)', '정의상 성인 5분위', ses, [0.2] * 5, dist(P.socioeconomic_band.astype(str).map({v: i for i, v in enumerate(ses)}), one, 5)))
    own = h.own.to_numpy(); hs = P.demo_housing_status.astype(str); o_ = hs.isin(['Own outright', 'Mortgage'])
    R.append(row('소득', 'demo_housing_status(자가 중)', '가계금융복지조사 2025 자가 가구(성인 가중)', ['Own outright', 'Mortgage'],
                 dist(h.mort[own], aw[own], 2), dist((hs[o_] == 'Mortgage').astype(int), np.ones(o_.sum()), 2)))
    # ---------------- KGSS
    from krpersona.blocks import kgss as KG
    K = pickle.load(open(f'{EST}/kgss_models.pkl', 'rb')); df, C, B, X_, mx = KG.load()
    for a in K['order']:
        M = K['models'][a]; it = M['cat']; y = KG.recode(C[a].fillna(-99).to_numpy(), it['codes']); Kc = len(it['codes'])
        p = sum(wt * dist(y[(df.YEAR == yr).to_numpy()], df.FINALWT[df.YEAR == yr], Kc) for yr, wt in M['gen_w'].items())
        note = {'출생 코호트 맞춤': f"조사 {max(M['gen_w'])}년 응답. 생성은 출생 코호트를 맞춘 2020년 성인", '연령×시기 상호작용': '연령×시기 상호작용'}.get(M['age_mode'], '')
        yrs = '·'.join(map(str, M['gen_w']))
        R.append(row('태도/가치관/사회인식', a, f'KGSS {yrs}', it['labels'], p, dist(I['k_' + a], one, Kc), note))
    # ---------------- family (KLoSA 2024, cohort matched): living children of 61+ by sex
    from krpersona.blocks import family as FM
    d2 = FM.load(); g = I[I.age >= 61]
    for sx, lab_ in [(1, '남'), (0, '여')]:
        k = d2[d2.male == sx]; gg = g[g.sex == (1 if sx == 1 else 2)]
        R.append(row('가족', f'61세 이상 자녀 수({lab_})', '고령화연구패널 10차(출생 코호트 맞춤)', ['0', '1', '2', '3', '4', '5+'],
                     dist(np.minimum(k.ntot, 5), k.w10wgt_c, 6), dist(np.minimum(gg.n_kids, 5), np.ones(len(gg)), 6), '생성은 동거 + 비동거 자녀로 센서스 출산 기록을 따름(61세 이상 결혼 경험 여성의 무자녀 5.1%). 고령화연구패널은 생존 자녀이고 무자녀가 약 2%로 적음'))
    T = pd.DataFrame(R)
    # values and personality checks
    vp = json.load(open(f'{EST}/value_params.json')); Z = I[[f'v_{j}' for j in range(10)]].to_numpy(); Rg = np.corrcoef(Z.T); Ro = np.array(vp['r_within_obs'])
    ko = vp['ko']; VC = pd.DataFrame([(f'{ko[j]}–{ko[(j + 5) % 10]}', Ro[j, (j + 5) % 10], Rg[j, (j + 5) % 10]) for j in range(5)] +
                                     [(f'{ko[j]}–{ko[(j + 1) % 10]}', Ro[j, (j + 1) % 10], Rg[j, (j + 1) % 10]) for j in range(10)], columns=['가치 쌍', 'KGSS 2021 집단 내 상관', '생성 전체 상관'])
    E = I[[f'e_{d}' for d in 'OCEAN']]
    PC = pd.DataFrame([(f'{"OCEAN"[i]}–{"OCEAN"[j]}', float(np.corrcoef(E.iloc[:, i], E.iloc[:, j])[0, 1])) for i in range(5) for j in range(i + 1, 5)], columns=['요인 쌍', '생성 상관'])
    return dict(T=T, VC=VC, PC=PC)

if __name__ == '__main__':
    out = compare(sys.argv[1], sys.argv[2]); pd.to_pickle(out, sys.argv[3])
    T = out['T']; pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
    print(T[['블록', '속성', '총변동거리(%p)']].round(2).to_string(index=False))
    print(T.groupby('블록')['총변동거리(%p)'].describe().round(2))
