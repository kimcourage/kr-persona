"""Labor block (EAPS 2025 H1): company size, own monthly wage band. Income block (HFCS 2025): household income, mortgage; SES and income quartile cut points."""
import sys, pickle, numpy as np, pandas as pd
from krpersona import paths
from krpersona.blocks import core
KSIC = [(1, 3, 'A'), (5, 8, 'B'), (10, 34, 'C'), (35, 35, 'D'), (36, 39, 'E'), (41, 42, 'F'), (45, 47, 'G'), (49, 52, 'H'), (55, 56, 'I'), (58, 63, 'J'),
        (64, 66, 'K'), (68, 68, 'L'), (70, 73, 'M'), (74, 76, 'N'), (84, 84, 'O'), (85, 85, 'P'), (86, 87, 'Q'), (90, 91, 'R'), (94, 96, 'S'), (97, 98, 'T'), (99, 99, 'U')]
LETTERS = 'ABCDEFGHIJKLMNOPQRSTU'
IND12 = {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7, 'K': 8, 'L': 9, 'M': 9, 'N': 9, 'O': 10, 'P': 10, 'Q': 10, 'R': 11, 'S': 11, 'T': 11, 'U': 10}
IND12_LAB = ['농림어업', '광업·제조업', '전기·가스·수도', '건설', '도소매', '운수·창고', '숙박·음식', '정보통신', '금융·보험', '부동산·전문·사업지원', '공공·교육·보건', '예술·개인서비스·기타']
SIZE_LAB = ['1–4명', '5–9명', '10–29명', '30–99명', '100–299명', '300명 이상']
SIZE_OUT = ['1-4 employees', '5-9 employees', '10-29 employees', '30-99 employees', '100-299 employees', '300+ employees']
SIZE_SETS = [['Startup (<50)'], ['Startup (<50)'], ['Startup (<50)'], ['Startup (<50)', 'SMB (50-500)'], ['SMB (50-500)'], ['SMB (50-500)', 'Mid (500-5k)', 'Enterprise (5k+)']]
WAGE_CUTS = [100, 150, 200, 250, 300, 400, 500, 700]                   # 만원/월
INC_CUTS = list(range(1000, 20001, 1000))                              # 만원/년, 21 bins
USD = [3250, 6500, 13000, 26000]                                       # 1,300원/$ band edges in 만원
EMP4 = ['임금근로', '고용원 있는 자영', '무급가족']                         # employed wage/employer/unpaid (own-account handled by rule)
BGEMP = {'emp_l': (EMP4, 'demo_employment_status', [['Full-time', 'Part-time'], ['Self-employed'], ['Self-employed']])}

def ind12_from_ksic(code3):
    d = pd.to_numeric(code3.str[:2], errors='coerce'); out = pd.Series(np.nan, index=code3.index)
    for lo, hi, L in KSIC: out[(d >= lo) & (d <= hi)] = IND12[L]
    return out

SIDO17 = [11, 21, 22, 23, 24, 25, 26, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39]      # census/EAPS 시도 codes (2020 coding)
def eaps():
    """A형(시군구·대분류) 2025 상반기: same respondents and weights as B형, with residence 시군구 and section-level industry / 1-digit occupation"""
    e = pd.read_csv(paths.raw('eaps2025h1', 'type_a'), encoding='cp949', dtype=str)
    e['ho'] = core.holdout_mask('eaps', len(e))
    f = lambda c: pd.to_numeric(e[c], errors='coerce'); age = f('만연령')
    e = e[(age >= 19) & (f('경제활동인구상태코드') == 1)].copy(); f = lambda c: pd.to_numeric(e[c], errors='coerce')
    d = pd.DataFrame(index=e.index)
    d['sex'] = (f('성별코드') == 2).astype(int); d['ag7'] = core.ag7(f('만연령'))
    edu = f('교육정도_학력구분코드') + 1; sch = f('교육정도_수학구분코드').map({1: 1, 2: 2, 3: 5, 4: 4, 5: 3})   # census coding: 1 졸업 2 재학 3 수료 4 휴학 5 중퇴
    d['edu5'] = core.edu5_census(edu.fillna(1), sch.fillna(1)); d['mar4'] = core.mar4(f('혼인상태코드'))
    st = f('종사상지위코드'); d['status'] = st
    d['emp_l'] = np.select([st.isin([1, 2, 3]), st == 4, st == 6], [0, 1, 2], -1)
    d['occ10'] = pd.to_numeric(e['1자리_8차직업대분류코드'], errors='coerce').fillna(0).clip(0, 9).astype(int)
    sec = e['1자리_11차산업대분류코드']; d['ind12'] = sec.map(IND12); d['indO'] = sec == 'O'
    sido = pd.to_numeric(e['4자리_행정구역시군구코드'].str[:2], errors='coerce'); d['s17'] = sido.map({c: i for i, c in enumerate(SIDO17)})
    d['size'] = f('직장종사자규모코드') - 1; d['wage'] = f('최근3개월_월평균급여')
    d['wband'] = np.digitize(d.wage, WAGE_CUTS); d['w'] = f('시군구가중값'); d['ho'] = e.ho
    return d

def fit_labor(mx, quick=False):
    d = eaps(); d = d[~d.ho] if core.HOLDOUT else d; par = ['sex', 'ag7', 'edu5', 'mar4', 'emp_l', 'occ10', 'ind12', 's17']
    P = [(k, core.BG[k][1], core.BG[k][2]) for k in ['sex', 'ag7', 'edu5', 'mar4']] + [('emp_l', 'demo_employment_status', BGEMP['emp_l'][2]), ('occ10', 'role_function', core.BG['occ10'][2]), ('ind12', None, None), ('s17', None, None)]
    L = [2, 7, 5, 4, 3, 10, 12, 17]; pairs = [(0, 1), (1, 2)]
    s = d[(d.status != 5) & ~d.indO & (d.emp_l >= 0) & d['size'].between(0, 5) & d.ind12.notna() & d.s17.notna()]
    m_size = core.fit_attr(mx, 'company_size', 'company_size', SIZE_SETS, SIZE_LAB, P, L, s[par].to_numpy(int), s['size'].to_numpy(int), s.w.to_numpy(float), pairs, quick=quick)
    m_size['out'] = SIZE_OUT
    print('company_size n', m_size['n'], m_size['info'], round(m_size['ll'], 4), round(m_size['ll_marginal'], 4), flush=True)
    s2 = d[d.status.isin([1, 2, 3]) & d.wage.notna() & d['size'].between(0, 5) & d.ind12.notna() & d.s17.notna()]
    par2 = par + ['size']; P2 = P + [('size', None, None)]; L2 = L + [6]
    m_w = core.fit_attr(mx, 'wage_band', None, [None] * (len(WAGE_CUTS) + 1), [f'<{WAGE_CUTS[0]}'] + [f'{a}–{b}' for a, b in zip(WAGE_CUTS[:-1], WAGE_CUTS[1:])] + [f'{WAGE_CUTS[-1]}+'],
                        P2, L2, s2[par2].to_numpy(int), s2.wband.to_numpy(int), s2.w.to_numpy(float), pairs, quick=quick)
    m_w['cats'] = [None] * (len(WAGE_CUTS) + 1); m_w['lower'] = [0] + WAGE_CUTS
    print('wage_band n', m_w['n'], m_w['info'], round(m_w['ll'], 4), round(m_w['ll_marginal'], 4), flush=True)
    return dict(size=m_size, wage=m_w, ind12_lab=IND12_LAB, ksic=KSIC, ind12=IND12, sido17=SIDO17)

def hfcs():
    h = pd.read_csv(paths.raw('hfcs2025', 'household'), encoding='cp949', dtype=str)
    m = pd.read_csv(paths.raw('hfcs2025', 'member'), encoding='cp949', dtype=str)
    f = lambda c: pd.to_numeric(h[c], errors='coerce')
    mm = m[pd.to_numeric(m['가구원정보_동거코드'], errors='coerce').isin([1, 2])]     # 1 1인 가구, 2 같이 살고 있음 (3·4 따로 삶)
    nemp = (pd.to_numeric(mm['가구원정보_종사상지위코드'], errors='coerce').between(1, 6)).groupby(mm['MD제공용_가구고유번호']).sum()
    d = pd.DataFrame(index=h.index)
    d['sex'] = (f('가구주_성별코드') == 2).astype(int); d['ag7'] = core.ag7(f('가구주_만연령'))
    edu = f('가구주_교육정도_학력코드'); sch = f('가구주_교육정도_수학구분코드').map({1: 1, 2: 2, 3: 4, 4: 5, 5: 3})
    d['edu5'] = core.edu5_census(edu.fillna(1), sch.fillna(1)); d['mar4'] = core.mar4(f('가구주_혼인상태코드'))
    st = f('가구주_종사상지위코드'); d['emp5'] = np.select([st.isin([1, 2, 6]), st == 3, st == 4, st == 5], [0, 1, 2, 3], 4)
    occ = h['가구주_직업대분류코드(8차)'].map({**{str(i): i for i in range(1, 10)}, 'A': 9}); d['occ10'] = pd.to_numeric(occ, errors='coerce').fillna(0).astype(int)
    d['hh5'] = np.minimum(f('가구원수'), 5) - 1; d['nemp4'] = np.minimum(h['MD제공용_가구고유번호'].map(nemp).fillna(0), 3).astype(int)
    d['metro'] = (h['수도권여부'] == 'G1').astype(int)
    d['inc'] = f('경상소득(보완)'); d['ibin'] = np.digitize(d.inc, INC_CUTS)
    d['own'] = f('입주형태코드') == 1; d['mort'] = (f('부채_금융부채_담보대출_대출용도_거주주택구입금액') > 0).astype(int)
    d['w'] = f('가중값'); d['hhsize'] = f('가구원수'); d['ho'] = core.holdout_mask('hfcs', len(h))
    ad = mm[pd.to_numeric(mm['가구원정보_만연령'], errors='coerce') >= 19].groupby('MD제공용_가구고유번호').size()
    d['nad'] = h['MD제공용_가구고유번호'].map(ad).fillna(0).to_numpy()      # co-resident adults (19+)
    return d

def ses_cuts_adult(d):
    """SES quintiles of equivalized income among adults (19+): household weight x co-resident adults"""
    eq = (d.inc / np.sqrt(d.hhsize)).to_numpy(); pw = d.w.to_numpy() * d.nad.to_numpy(); o_ = np.argsort(eq); cw = np.cumsum(pw[o_]) / pw.sum()
    return [float(eq[o_][np.searchsorted(cw, p)]) for p in (.2, .4, .6, .8)]

def fit_income(mx, quick=False):
    d = hfcs(); d = d[~d.ho] if core.HOLDOUT else d; par = ['sex', 'ag7', 'edu5', 'mar4', 'emp5', 'occ10', 'hh5', 'nemp4', 'metro']
    P = [(k, core.BG[k][1], core.BG[k][2]) for k in par]; L = [len(core.BG[k][0]) for k in par]
    pairs = [(1, 2), (0, 3), (1, 4), (6, 7)]
    m_inc = core.fit_attr(mx, 'household_income', None, [None] * (len(INC_CUTS) + 1), [f'{a}–{a + 1000}' for a in [0] + INC_CUTS[:-1]] + ['20000+'], P, L, d[par].to_numpy(int), d.ibin.to_numpy(int), d.w.to_numpy(float), pairs, quick=quick)
    m_inc['cats'] = [None] * (len(INC_CUTS) + 1)
    # within-bin donors (actual amounts) for drawing a value inside the chosen bin
    don = {b: (g.inc.to_numpy(), g.w.to_numpy()) for b, g in d.groupby('ibin')}
    print('household_income n', m_inc['n'], m_inc['info'], round(m_inc['ll'], 4), round(m_inc['ll_marginal'], 4), flush=True)
    o = d[d.own]; o = o.assign(ib7=np.digitize(o.inc, [2000, 3500, 5000, 7000, 10000, 15000]))
    par2 = ['ag7', 'emp5', 'hh5', 'metro', 'ib7']; P2 = [(k, core.BG[k][1], core.BG[k][2]) for k in par2[:4]] + [('ib7', None, None)]; L2 = [7, 5, 5, 2, 7]
    m_mort = core.fit_attr(mx, 'mortgage', 'demo_housing_status', [['Own outright'], ['Mortgage']], ['대출 없음', '거주주택 구입 대출 있음'], P2, L2,
                           o[par2].to_numpy(int), o.mort.to_numpy(int), o.w.to_numpy(float), [(0, 4)], quick=quick)
    print('mortgage n', m_mort['n'], m_mort['info'], round(m_mort['ll'], 4), round(m_mort['ll_marginal'], 4), flush=True)
    # equivalized income cut points (person-weighted): SES quintiles, KNHANES-style quartiles
    eq = d.inc / np.sqrt(d.hhsize); pw = d.w * d.hhsize; o_ = np.argsort(eq.to_numpy()); cw = np.cumsum(pw.to_numpy()[o_]) / pw.sum()
    q = lambda p: float(eq.to_numpy()[o_][np.searchsorted(cw, p)])
    return dict(income=m_inc, donors=don, mortgage=m_mort, ses_cuts=ses_cuts_adult(d), ses_cuts_all_persons=[q(p) for p in (.2, .4, .6, .8)], inc4_cuts=[q(p) for p in (.25, .5, .75)],
                inc_cuts=INC_CUTS, usd_edges=USD, mort_cuts=[2000, 3500, 5000, 7000, 10000, 15000])

if __name__ == '__main__':
    mx = core.MX(); q = '--quick' in sys.argv
    if '--labor-only' in sys.argv:
        K = pickle.load(open(paths.art('models', f'labor_income_models{core.SUF}.pkl'), 'rb')); K['labor'] = fit_labor(mx, q); pickle.dump(K, open(paths.art('models', f'labor_income_models{core.SUF}.pkl', mkdir=True), 'wb')); sys.exit()
    if '--income-only' in sys.argv:
        K = pickle.load(open(paths.art('models', f'labor_income_models{core.SUF}.pkl'), 'rb')); K['income'] = fit_income(mx, q)
        print('SES cuts (adults)', np.round(K['income']['ses_cuts']), 'all persons', np.round(K['income']['ses_cuts_all_persons']), 'inc4', np.round(K['income']['inc4_cuts']))
        pickle.dump(K, open(paths.art('models', f'labor_income_models{core.SUF}.pkl', mkdir=True), 'wb')); sys.exit()
    lab = fit_labor(mx, q); inc = fit_income(mx, q)
    print('SES cuts (만원, equivalized)', np.round(inc['ses_cuts']), 'inc4', np.round(inc['inc4_cuts']))
    pickle.dump(dict(labor=lab, income=inc), open(paths.art('models', f'labor_income_models{core.SUF}.pkl', mkdir=True), 'wb'))
