"""Held-out real data (20% of every source) and persona outputs on common comparison classes.
Each source -> dict(name, real=DataFrame of class codes, w=weights, attrs=[...], to_class={attr: f(persona string Series) -> codes})
A class code of -1 = missing in the real data; persona labels that fit no class get code K ('other')."""
import sys, json, pickle, numpy as np, pandas as pd
from krpersona import paths
from krpersona.blocks import core
EST = paths.art('models')
AGEB = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+']

def lab_map(groups, other=True):
    """groups: list of label lists -> f(Series) -> class codes (unmatched -> len(groups))"""
    m = {v: i for i, g in enumerate(groups) for v in g}
    return lambda s: pd.Series(s).astype(str).map(m).fillna(len(groups)).astype(int).to_numpy(), len(groups)

def age_br(a): return np.digitize(np.asarray(a), [25, 35, 45, 55, 65, 75, 85])     # 19-24 coded as the 18-24 class
F_AGE = lab_map([[b] for b in AGEB]); F_SEX = lab_map([['Man'], ['Woman']])
F_MAR = lab_map([['Single', 'In a relationship', 'Engaged'], ['Married'], ['Widowed'], ['Divorced', 'Separated']])
EDU_FULL = [['No formal', 'Primary'], ['Secondary'], ['Some college'], ["Associate's"], ["Bachelor's"], ["Master's"], ['Doctorate', 'Postdoc']]

# ---------------------------------------------------------------- census
def census():
    P = pd.read_parquet(paths.art('census', 'P.parquet')); H = pd.read_parquet(paths.art('census', 'H.parquet'))
    tr = set(H.hid.sample(frac=0.8, random_state=5)); P = P.merge(H[['hid', 'k', 'hkind', 'tenure_h', 'pet', 'gtype', 'car']], on='hid')
    P = P[(P.nat == 1) & (P.bplace != 4) & (P.age >= 19) & ~P.hid.isin(tr)].reset_index(drop=True)
    e, s, age = P.edu.to_numpy(), P.sch.to_numpy(), P.age.to_numpy(); grad = s == 1
    edu = np.select([e == 1, (e == 2) & grad, e == 2, (e == 3) & grad, e == 3, e == 4, (e == 5) & grad, e == 5, (e == 6) & grad, e == 6, (e == 7) & np.isin(s, [1, 3]), e == 7, (e == 8) & grad, e == 8],
                    [0, 0, 0, 1, 0, 1, 3, 2, 4, 2, 5, 4, 6, 5], 1)
    econ, st = P.econ.to_numpy(), P.status.to_numpy(); work = np.isin(econ, [1, 2, 3]); enr = np.isin(s, [2, 4])
    emp = np.select([work & (st == 1) & np.isin(econ, [1, 3]), work & (st == 1), work, enr], [0, 1, 2, 3], 4)
    metro = P.sido.isin([11, 21, 22, 23, 24, 25, 26]).to_numpy(); dong = P.dong.to_numpy()
    urb = np.select([(dong == 3) & metro, dong == 3, dong == 1], [0, 1, 2], 3)
    k, hk = P.k.to_numpy(), P.hkind.to_numpy(); hh = np.select([np.isin(hk, [4, 5]), k == 1, k == 2, k <= 4], [0, 1, 2, 3], 4)
    ten = np.where(P.rel == 3, 2, np.where(P.tenure_h == 1, 0, 1))
    lim = P.lim.fillna(0).astype(int).to_numpy(); nb = np.array([bin(x).count('1') for x in lim])
    dis = np.select([lim == 0, nb >= 2, np.isin(lim, [1, 2, 32]), lim == 4], [0, 1, 2, 3], 4)
    r1, r5, bp = P.r1.to_numpy(), P.r5.to_numpy(), P.bplace.to_numpy()
    mob = np.select([(r1 == 5) | (r5 == 5), bp == 5, bp == 3, r5 == 6, r5 == 4], [0, 1, 2, 1, 2], 3)
    pet = pd.Series(P.pet.to_numpy()).map({0: 0, 1: 1, 2: 2, 4: 3}).fillna(4).astype(int).to_numpy()
    comm, mode, place = P.comm.to_numpy(), P['mode'].to_numpy(), P.place.to_numpy()
    cm = np.where(np.isin(comm, [1, 2]), pd.Series(mode).map({1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5}).fillna(5).to_numpy(), np.where(work & (place == 2), 6, 7)).astype(int)
    R = pd.DataFrame(dict(age_bracket=age_br(age), gender_identity=(P.sex == 2).astype(int), demo_marital_status=P.mar.astype(int) - 1, highest_education=edu,
                          demo_employment_status=emp, urbanicity=urb, lstyle_household_size=hh, demo_housing_status=ten, demo_disability_status=dis,
                          lifex_geographic_mobility=mob, lstyle_pet_ownership=pet, lstyle_commute_mode=cm))
    T = dict(age_bracket=F_AGE, gender_identity=F_SEX, demo_marital_status=F_MAR, highest_education=lab_map(EDU_FULL),
             demo_employment_status=lab_map([['Full-time'], ['Part-time'], ['Self-employed', 'Gig / freelance', 'Unpaid family worker'], ['Student'], ['Retired', 'Homemaker', 'Unemployed']]),
             urbanicity=lab_map([['Dense urban'], ['Suburban'], ['Small town'], ['Rural']]),
             lstyle_household_size=lab_map([['Communal'], ['Lives alone'], ['2 people'], ['3-4 people'], ['5+ people']]),
             demo_housing_status=lab_map([['Own outright', 'Mortgage'], ['Renting'], ['Living with family']]),
             demo_disability_status=lab_map([['No disability'], ['Multiple'], ['Sensory'], ['Physical'], ['Cognitive']]),
             lifex_geographic_mobility=lab_map([['Moved internationally'], ['Moved nationally'], ['Moved within region'], ['Never left hometown']]),
             lstyle_pet_ownership=lab_map([['None'], ['Dog'], ['Cat'], ['Other'], ['Multiple pets']]),
             lstyle_commute_mode=lab_map([['Walk'], ['Bike'], ['Car'], ['Public transit'], ['Rideshare'], ['Other'], ['Remote'], ['Not applicable']]))
    return dict(name='인구주택총조사 2020(20% 가구)', real=R, w=P.hw.to_numpy(float), to_class=T, extra=dict(k=k, rel=P.rel.to_numpy(), age=age))

# ---------------------------------------------------------------- KNHANES
def knhanes():
    from krpersona.blocks import health as HM
    d = HM.load(); d = d[d.ho].reset_index(drop=True)
    it = [f'p{i}' for i in range(1, 10)]; it2 = [f'g{i}' for i in range(1, 8)]
    phq = d[it].sum(1).where(d[it].notna().all(1)); gad = d[it2].sum(1).where(d[it2].notna().all(1))
    dep = np.digitize(phq.fillna(-1), [5, 10, 15, 20]).astype(int); dep[phq.isna().to_numpy()] = -1
    anx = np.digitize(gad.fillna(-1), [5, 10, 15]).astype(int); anx[gad.isna().to_numpy()] = -1
    gen = d.general.fillna(-1).astype(int).to_numpy(); ok = (dep >= 0) & (anx >= 0) & (gen >= 0)
    p_, g_ = phq.fillna(0).to_numpy(), gad.fillna(0).to_numpy()
    mh = np.select([(p_ >= 20) | (g_ >= 15), (p_ >= 10) | (g_ >= 10), (p_ >= 5) | (g_ >= 5), np.isin(gen, [0, 1])], [3, 2, 1, 0], 1); mh[~ok] = -1
    R = pd.DataFrame(dict(gender_identity=d.sex.to_numpy(), age_bracket=d.ag7.clip(0, 6).to_numpy(), demo_marital_status=d.mar4.to_numpy(),
                          highest_education=d.edu4.map({0: 0, 1: 1, 2: 1, 3: 2}).fillna(-1).astype(int).to_numpy(),
                          health_chronic_condition=d.chronic.to_numpy(), health_general_health=gen, lstyle_smoking=d.smoking.to_numpy(),
                          lstyle_alcohol_use=d.alcohol.to_numpy(), lstyle_exercise_freq=d.exercise.to_numpy(), health_stress_level=d.stress.fillna(-1).astype(int).to_numpy(),
                          kr_depression_level=dep, kr_anxiety_level=anx, health_mental_health=mh))
    T = dict(gender_identity=F_SEX, age_bracket=lab_map([['18-24'], ['25-34'], ['35-44'], ['45-54'], ['55-64'], ['65-74'], ['75-84', '85+']]), demo_marital_status=F_MAR,
             highest_education=lab_map([['No formal', 'Primary'], ['Secondary', 'Some college'], ["Associate's", "Bachelor's", "Master's", 'Doctorate', 'Postdoc']]),
             health_chronic_condition=lab_map([['None'], ['Managed'], ['Multiple']]), health_general_health=lab_map([['Excellent'], ['Good'], ['Fair'], ['Poor']]),
             lstyle_smoking=lab_map([['Never'], ['Former'], ['Occasional'], ['Regular']]), lstyle_alcohol_use=lab_map([['Never'], ['Rarely'], ['Socially'], ['Regularly'], ['Heavily']]),
             lstyle_exercise_freq=lab_map([['Daily'], ['Weekly'], ['Monthly'], ['Rarely', 'Never']]), health_stress_level=lab_map([['Very high'], ['High'], ['Moderate'], ['Low', 'None']]),
             kr_depression_level=lab_map([['Minimal'], ['Mild'], ['Moderate'], ['Moderately severe'], ['Severe']]), kr_anxiety_level=lab_map([['Minimal'], ['Mild'], ['Moderate'], ['Severe']]),
             health_mental_health=lab_map([['Thriving'], ['Stable'], ['Struggling'], ['In crisis']]))
    R.loc[R.highest_education.isna(), 'highest_education'] = -1
    # KNHANES edu4: 1 초졸 이하, 2 중졸, 3 고졸, 4 대졸 이상 -> classes {초졸 이하}, {중졸, 고졸, 대학 미졸업}, {전문대졸 이상}
    return dict(name='국민건강영양조사 2022(20%)', real=R, w=d.w.to_numpy(float), to_class=T, extra=dict(phq=phq.to_numpy(), gen=gen))

# ---------------------------------------------------------------- EAPS
def eaps():
    from krpersona.blocks import labor_income as LI
    e = LI.eaps(); e = e[e.ho].reset_index(drop=True)
    size = np.where(e.status == 5, 0, np.where(e.indO, 1, e['size'].fillna(-3) + 2)).astype(int)
    R = pd.DataFrame(dict(gender_identity=e.sex.to_numpy(), age_bracket=e.ag7.to_numpy(), highest_education=e.edu5.map({0: 0, 1: 0, 2: 1, 3: 2, 4: 3}).to_numpy(), company_size=size))
    T = dict(gender_identity=F_SEX, age_bracket=lab_map([['18-24'], ['25-34'], ['35-44'], ['45-54'], ['55-64'], ['65-74'], ['75-84', '85+']]),
             highest_education=lab_map([['No formal', 'Primary', 'Secondary', 'Some college'], ["Associate's"], ["Bachelor's"], ["Master's", 'Doctorate', 'Postdoc']]),
             company_size=lab_map([['Solo / freelance'], ['Public sector'], ['1-4 employees'], ['5-9 employees'], ['10-29 employees'], ['30-99 employees'], ['100-299 employees'], ['300+ employees']]))
    return dict(name='지역별고용조사 2025 상반기 취업자(20%)', real=R, w=e.w.to_numpy(float), to_class=T,
                scope=lambda P: P.demo_employment_status.astype(str).isin(['Full-time', 'Part-time', 'Self-employed', 'Gig / freelance', 'Unpaid family worker']).to_numpy())

# ---------------------------------------------------------------- HFCS (household attributes, adult weighted)
def hfcs():
    from krpersona.blocks import labor_income as LI
    h = LI.hfcs(); h = h[h.ho].reset_index(drop=True)
    I = pickle.load(open(f'{EST}/labor_income_models_80.pkl', 'rb'))['income']
    usd = np.digitize(h.inc, LI.USD); ses = np.digitize(h.inc / np.sqrt(h.hhsize), I['ses_cuts'])
    R = pd.DataFrame(dict(demo_household_income=usd, socioeconomic_band=ses))
    T = dict(demo_household_income=lab_map([['<$25k'], ['$25k-50k'], ['$50k-100k'], ['$100k-200k'], ['$200k+']]),
             socioeconomic_band=lab_map([['Low income'], ['Lower-middle'], ['Middle'], ['Upper-middle'], ['High income']]))
    return dict(name='가계금융복지조사 2025(20% 가구, 성인 가중)', real=R, w=(h.w * h.nad).to_numpy(float), to_class=T,
                extra=dict(own=h.own.to_numpy(), mort=h.mort.to_numpy(), wad=(h.w * h.nad).to_numpy(float)))

# ---------------------------------------------------------------- KGSS (2016-2025 held-out respondents)
def kgss():
    from krpersona.blocks import kgss as KG
    K = pickle.load(open(f'{EST}/kgss_models_80.pkl', 'rb'))
    df, C, B, X_, mx = KG.load(keep_all=True)
    ho = df.ho.to_numpy() & (df.YEAR >= 2016).to_numpy()
    R = pd.DataFrame(dict(gender_identity=B.sex.to_numpy(), age_bracket=B.ag7.to_numpy(),
                          highest_education=B.edu5.map({0: 0, 1: 0, 2: 1, 3: 2, 4: 3, -1: -1}).to_numpy(), YEAR=df.YEAR.to_numpy()))
    T = dict(gender_identity=F_SEX, age_bracket=lab_map([['18-24'], ['25-34'], ['35-44'], ['45-54'], ['55-64'], ['65-74'], ['75-84', '85+']]),
             highest_education=lab_map([['No formal', 'Primary', 'Secondary', 'Some college'], ["Associate's"], ["Bachelor's"], ["Master's", 'Doctorate', 'Postdoc']]))
    years = {}
    for a in K['order']:
        M = K['models'][a]
        if M['age_mode'] == '출생 코호트 맞춤' or not any(y >= 2016 for y in M['years']): continue
        it = M['cat']; y = KG.recode(C[a].fillna(-99).to_numpy(), it['codes'])
        # comparison classes: merge Korean categories that share an output label (custom or MatrAIx), so MatrAIx personas can be mapped too
        par = list(range(len(it['codes'])))
        def find(i):
            while par[i] != i: i = par[i]
            return i
        labs = {}
        for kk in range(len(it['codes'])):
            ls = ([it['out'][kk]] if it['out'][kk] else []) + (it['cond'][kk] if it['cond'] and it['cond'][kk] else [])
            if a == 'demo_religion_affiliation' and kk == 8: ls = ls + ['Spiritual but unaffiliated']
            for l in ls:
                if l in labs: par[find(kk)] = find(labs[l])
                else: labs[l] = kk
        roots = sorted({find(i) for i in range(len(par))}); cls = {r: j for j, r in enumerate(roots)}
        kmap = np.array([cls[find(i)] for i in range(len(par))])
        R[a] = np.where(y >= 0, kmap[np.maximum(y, 0)], -1)
        groups = [[] for _ in roots]
        for l, kk in labs.items(): groups[kmap[kk]].append(l)
        T[a] = lab_map(groups)
        years[a] = [yy for yy in M['gen_w']]
    return dict(name='KGSS 2016–2025(20%)', real=R[ho].reset_index(drop=True), w=df.FINALWT.to_numpy(float)[ho], to_class=T, years=years)

def matraix_adults(columns):
    """MatrAIx full-DAG reproduction, Korean-culture-Native adults (the step-1 diagnosis sample, artifacts/baseline) as strings"""
    import os
    D = np.load(paths.art('baseline', 'codes.npz')); M = D['kr']; meta = json.load(open(paths.art('baseline', 'meta.json'))); cols = meta['columns']; vals = meta['values']; ci = {c: i for i, c in enumerate(cols)}
    adult = np.isin(M[:, ci['age_bracket']], [vals['age_bracket'].index(b) for b in AGEB]); M = M[adult]
    out = {}
    for c in columns:
        if c in ci:
            v = np.array(vals[c] + [None], dtype=object); x = M[:, ci[c]].astype(int); x[x >= len(vals[c])] = len(vals[c]); out[c] = v[x]
    return pd.DataFrame(out)
