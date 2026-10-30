"""KGSS attitude/values/social-perception block (72 attributes): category coding, estimation (penalized logit with MatrAIx prior mean; years and parents from mapping/kgss_analysis.py), generation."""
import sys, re, json, pickle, time, numpy as np, pandas as pd, pyreadstat, warnings
warnings.filterwarnings('ignore')
from krpersona import paths
from krpersona.blocks import core
from krpersona.mapping.kgss_codes import attr_vars, validity, build
SAV = lambda: str(paths.raw('kgss', 'all'))
L5 = lambda *a: [([i + 1], x) for i, x in enumerate(a)]
EAPO = ['Enthusiast', 'Positive', 'Neutral', 'Skeptical', 'Opposed']
FREQ5 = ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never']
T3 = [([1], '{Enthusiast, Positive}'), ([2], 'Neutral'), ([3], '{Skeptical, Opposed}')]
L7 = lambda a, b, c, d, e: [([1], a), ([2], b), ([3, 4], c), ([5], d), ([6, 7], e)]
MUS = [([1], 'Love'), ([2], 'Like'), ([3], 'Neutral'), ([4, 5], 'Dislike')]
IMP4 = [([1], 'Core value'), ([2], 'Important'), ([3], '{Moderate, Minor}'), ([4], 'Irrelevant')]
TRUST3 = [([0], 'High'), ([1], 'Moderate'), ([2], 'Low')]
# category spec: attribute -> list of (codes in the KGSS coding of kgss_codes.build, output). output: MatrAIx value, '{a, b}' set drawn by MatrAIx,
# or 'Custom label|MatrAIx cond' (custom output with a conditioning category). kr_* outputs are plain labels.
SPEC = {
    'demo_religion_affiliation': [([1, 6], 'Buddhist'), ([5], 'Won Buddhist|Buddhist'), ([2], 'Protestant|Christian'), ([3], 'Catholic|Christian'),
                                  ([7], 'Confucian|Folk / traditional'), ([9], 'Folk / traditional'), ([10, 11, 12], 'Korean new religion|Folk / traditional'),
                                  ([8, 13, 77], 'Other religion|Folk / traditional'), ([4], '{None, Atheist / agnostic}')],
    'religiosity': [([0], 'Secular'), ([1], 'Devout'), ([2], 'Observant'), ([3], 'Nominal|Spiritual')],
    'political_lean': L5('Left', 'Center-left', 'Center', 'Center-right', 'Right'),
    'kr_subjective_class': [([1, 2], 'Lower class'), ([3, 4], 'Lower-middle'), ([5, 6], 'Middle'), ([7, 8], 'Upper-middle'), ([9, 10], 'Upper class')],
    'kr_view_of_north_korea': L5('Someone to help', 'Partner for cooperation', 'Someone to be wary of', 'Enemy'),
    'kr_unification_need': L5('Very necessary', 'Somewhat necessary', 'Not very necessary', 'Not necessary at all'),
    'kr_trust_in_institutions': TRUST3, 'kr_trust_in_news_media': TRUST3, 'kr_trust_in_big_business': [([1], 'High'), ([2], 'Moderate'), ([3], 'Low')],
    'att_organized_religion': T3, 'att_labor_unions': T3,
    'val_patriotism': [([1], 'Core value'), ([2], '{Important, Moderate}'), ([3], 'Minor'), ([4], 'Irrelevant')],
    'kr_country_felt_closest': L5('United States', 'Japan', 'North Korea', 'China', 'Russia'),
    'values_priority': [([3, 7, 10], 'Achievement'), ([5, 6], 'Security'), ([1], 'Novelty'), ([2, 4], 'Community'), ([8], 'Tradition'), ([9], '{Security, Community, Tradition}')],
    'kr_life_satisfaction': L5('Very satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very dissatisfied'),
    'kr_happiness': L5('Very happy', 'Happy', 'Neutral', 'Unhappy', 'Very unhappy'),
    'kr_belief_in_saju': L5('Strongly believes', 'Believes', 'Unsure', 'Doubts', 'Strongly rejects'),
    'kr_attitude_same_sex_relations': L5('Always wrong', 'Mostly wrong', 'Sometimes wrong', 'Not wrong at all'),
    'att_authority': [([1], 'Enthusiast'), ([2], 'Positive'), ([3, 4], 'Neutral'), ([5, 6], 'Skeptical'), ([7], 'Opposed')],
    'kr_family_wealth_for_success': L5('Essential', 'Very important', 'Fairly important', 'Not very important', 'Not important'),
    'kr_living_standard_vs_parents': L5('Much better than parents', 'Somewhat better', 'About the same', 'Somewhat worse', 'Much worse than parents'),
    'health_insurance_status': [([2], 'Comprehensive'), ([1], 'Basic'), ([3], 'Minimal'), ([4], 'Uninsured')],
    'kr_next_generation_outlook': L5('Much better', 'Somewhat better', 'About the same', 'Somewhat worse', 'Much worse'),
    'kr_confidants': [([0], 'None'), ([1], 'One'), ([2, 3], 'A few'), (list(range(4, 1000)), 'Many')],
    'kr_confucian_way_of_life': L5('Yes', 'No'),
    'trait_empathy': 'QUANTILE', 'trait_hope_optimism': 'QUANTILE',
    'att_data_privacy': [([1], 'Enthusiast'), ([2], 'Positive'), ([3], 'Skeptical'), ([4], 'Opposed')],
    'att_globalization': L5('Opposed', 'Skeptical', 'Neutral', 'Positive', 'Enthusiast'),
    'lifex_faith_journey': [([4], 'Lifelong faith'), ([3], 'Converted'), ([2], 'Lapsed / left faith'), ([1], 'Always secular')],
    'topic_politics': L5('Passionate', 'Interested', 'Neutral', 'Indifferent', 'Averse'),
    'att_surveillance': [([1], 'Enthusiast'), ([2], 'Positive'), ([3], 'Skeptical'), ([4], 'Opposed')],
    'att_universal_basic_income': [([5], 'Enthusiast'), ([4], 'Positive'), ([3], 'Neutral'), ([2], 'Skeptical'), ([1], 'Opposed')],
    'att_immigration': L5(*EAPO),
    'demo_political_engagement': [([0], 'Protest participant|Activist'), ([1], 'Has participated|Engaged voter'), ([2], 'Might participate|Occasional voter'), ([3], 'Never participates|Disengaged')],
    'att_nuclear_energy': T3,
    'lstyle_giving': [([1], 'Regular donor'), ([2], 'Occasional'), ([3], 'Rare'), ([4, 5], 'Never')],
    'musg_pop': MUS, 'musg_rock': MUS, 'musg_classical': MUS,
    'habit_naps': L5(*FREQ5), 'lstyle_reading_freq': L5(*FREQ5),
    'hob_karaoke': [([1, 2], 'Avid'), ([3], 'Active'), ([4], 'Occasional'), ([5], '{Curious, Never}')],
    'lstyle_gaming_freq': [([6], 'Daily'), ([5], 'Weekly'), ([3, 4], 'Monthly'), ([1, 2], 'Rarely'), ([0], 'Never')],
    'lstyle_music_listening': [([6], '{All day, Daily}'), ([3, 4, 5], 'Sometimes'), ([0, 1, 2], 'Rarely')],
    'val_family': IMP4, 'val_health': IMP4, 'val_spirituality_faith': IMP4,
    'lstyle_planning_horizon': [([5], 'Day-to-day'), ([4], 'Weekly'), ([3], 'Monthly'), ([2], 'Yearly'), ([1], 'Multi-year')],
    'kr_ancestry_for_korean_identity': L5('Very important', 'Fairly important', 'Not very important', 'Not important at all'),
    'lstyle_volunteering': [([1], '{Daily, Weekly}'), ([2], 'Monthly'), ([3, 4, 5], 'Rarely'), ([6], 'Never')],
    'kr_political_efficacy': [([0], 'Low'), ([1], 'Moderate'), ([2], 'High')],
    'pref_lead_vs_follow': [([6], 'Always leads'), ([4, 5], 'Leans lead'), ([3], 'Situational'), ([1, 2], 'Leans support'), ([0], 'Prefers to follow')],
    'att_ai': L5('Opposed', 'Skeptical', 'Neutral', 'Positive', 'Enthusiast'),
    'att_free_speech': [([1], 'Enthusiast'), ([2], 'Positive'), ([3], 'Skeptical'), ([4], 'Opposed')],
    'val_sustainability': L5('Core value', 'Important', 'Moderate', 'Minor', 'Irrelevant'),
    'health_energy_level': L5('Very high', 'High', 'Moderate', 'Low', 'None'),
    'health_health_literacy': L5('Very high', 'High', 'Moderate', 'Low', 'None'),
    'att_climate_action': L5(*EAPO), 'att_government_regulation': L5(*EAPO),
    'topic_environment': [([5], 'Passionate'), ([4], 'Interested'), ([3], 'Neutral'), ([2], 'Indifferent'), ([1], 'Averse')],
    'health_pain_level': [([1], 'None'), ([2], 'Mild'), ([3], 'Moderate'), ([4, 5], 'Severe')],
    'lstyle_cooking_freq': [([1], 'Daily'), ([2, 3], 'Weekly'), ([4], 'Monthly'), ([5, 6], 'Rarely'), ([7], 'Never')],
    'kr_children_necessity': [([1, 2], 'Not necessary'), ([3, 4, 5], 'Undecided'), ([6, 7], 'Necessary')],
    'demo_birth_order': [([0], 'Only child'), ([1], 'Eldest'), ([2], 'Youngest'), ([3], 'Middle')],
    'risk_tolerance': L7('Risk-seeking', 'Risk-tolerant', 'Balanced', 'Cautious', 'Risk-averse'),
    'mft_authority_subversion': L7('Very high', 'High', 'Average', 'Low', 'Very low'),
    'dospert_financial_risk_tolerance': L7('Very high', 'High', 'Average', 'Low', 'Very low'),
    'topic_anime': [([1], 'Passionate'), ([2], 'Interested'), ([3], 'Neutral'), ([4], '{Indifferent, Averse}')],
    'lifex_travel_breadth': [([0], 'Never left country'), ([1], 'A few countries'), ([2], 'Well-traveled')],
    'att_traditional_gender_roles': L5(*EAPO),
    'topic_spirituality': [([1], '{Passionate, Interested}'), ([2], 'Neutral'), ([3], 'Interested'), ([4], '{Indifferent, Averse}')],
}
# attributes added in step 5 (not in the step-1 tables): English ability from the KGSS 2018 self-rating items
# (read an English newspaper / talk with English speakers / write a letter; 1 = very well ... 5 = not at all), mean of the 3 items:
# <= 1.5 Fluent (C1-C2), <= 3 (average or better) Intermediate (B1-B2), 5 on all items None, otherwise Basic (A1-A2)  [assumed mapping]
ADD_ATTRS = {'english_proficiency': ['ENGPROF1', 'ENGPROF2', 'ENGPROF3']}
ADD_ROWS = [{'속성': 'english_proficiency', '추정에 쓰는 연도': '2018', '생성 시 연도': '2018년', '연령 처리': '출생 코호트 맞춤', '선택 부모(최대 3)': '(배경 변수만)'}]
SPEC['english_proficiency'] = [([0], 'Fluent (C1-C2)'), ([1], 'Intermediate (B1-B2)'), ([2], 'Basic (A1-A2)'), ([3], 'None')]
QUANT = ['Signature', 'Strong', 'Moderate', 'Slight', 'Absent']
BGK = ['sex', 'ag7', 'edu5', 'mar4', 'emp2', 'urban2', 'reg7']      # KGSS background parents
BGC = ['sex', 'ag7', 'edu5']                                         # cohort-matched items (time-varying statuses left out)
XI_CUTS = [-0.84, -0.25, 0.25, 0.84]
OUTB = {'health_energy_level': ['gh'], 'health_pain_level': ['gh'], 'health_health_literacy': ['gh'], 'health_insurance_status': ['gh', 'xd']}
GH_NODE = ('health_general_health', [['Excellent'], ['Good'], ['Fair'], ['Poor']])

def parse_out(o):
    """-> (custom output or None, MatrAIx conditioning set)"""
    if '|' in o: c, s = o.split('|'); return c, [s]
    if o.startswith('{'): return None, [x.strip() for x in o[1:-1].split(',')]
    return o, [o]

def cat_info(a, mx_values):
    spec = SPEC[a]
    if spec == 'QUANTILE': spec = [([i], q) for i, q in enumerate(QUANT)]
    is_mx = a in mx_values
    labels, outs, conds = [], [], []
    for codes, o in spec:
        if is_mx: c, s = parse_out(o); outs.append(c); conds.append(s); labels.append(o)
        else: outs.append(o); conds.append(None); labels.append(o)
    return dict(codes=[c for c, _ in spec], labels=labels, out=outs, cond=conds, mx=is_mx)

def recode(x, codes):
    y = np.full(len(x), -1)
    for k, cs in enumerate(codes): y[np.isin(x, cs)] = k
    return y

# ---------------------------------------------------------------- data
def load(keep_all=False):
    A = attr_vars(); A.update(ADD_ATTRS); allv = sorted({v for vs in A.values() for v in vs})
    ext = ['HEALTHY', 'HEALTH21', 'HEALTH23', 'GRPPARTY', 'SPIRITUA'] + [f'CESD{i}' for i in range(1, 12)]
    bg = ['YEAR', 'AGE', 'SEX', 'EDUC', 'GRADUATE', 'MARITAL', 'EMPLY', 'URBAN', 'REGION', 'FINALWT']
    vit = [f'VALUES{i}' for i in range(1, 23)]
    df, m = pyreadstat.read_sav(SAV(), usecols=sorted(set(bg + allv + ext + vit)))
    df['_row'] = np.arange(len(df))
    df['ho'] = core.holdout_mask('kgss', len(df))
    keep = (df.AGE >= 19) & df.SEX.isin([1, 2])
    if core.HOLDOUT and not keep_all: keep &= ~df.ho
    df = df[keep].reset_index(drop=True); df['YEAR'] = df.YEAR.astype(int)
    V = validity(df, m, sorted(set(allv + ext))); C = build(df, V, A); g = lambda v: df[v].where(V[v])
    E = pd.concat([g(v) for v in ADD_ATTRS['english_proficiency']], axis=1); me = E.mean(axis=1).where(E.notna().all(1))
    C['english_proficiency'] = pd.Series(np.select([me <= 1.5, me <= 3, me >= 5], [0, 1, 3], 2), index=df.index).where(me.notna())
    # background codes on the generator's levels
    e = df.EDUC.to_numpy(); gr = df.GRADUATE.to_numpy(); grad = ~np.isin(gr, [2, 3])     # 중퇴·재학 중 = not completed; missing -> completed
    B = pd.DataFrame(index=df.index)
    B['sex'] = (df.SEX == 2).astype(int); B['ag7'] = core.ag7(df.AGE)
    B['edu5'] = np.select([np.isin(e, [0, 1, 2, 8]) | ((e == 3) & ~grad), (e == 3) | (np.isin(e, [4, 5]) & ~grad), e == 4, (e == 5) | (np.isin(e, [6, 7]) & ~grad), np.isin(e, [6, 7])], [0, 1, 2, 3, 4], -1)
    B.loc[~np.isin(e, range(0, 9)), 'edu5'] = -1
    mar = df.MARITAL.to_numpy(); B['mar4'] = np.select([np.isin(mar, [5, 6]), mar == 1, mar == 2, np.isin(mar, [3, 4])], [0, 1, 2, 3], -1)
    B['emp2'] = np.select([df.EMPLY == 1, df.EMPLY == 2], [0, 1], -1)
    B['urban2'] = np.select([df.URBAN == 1, df.URBAN == 0], [0, 1], -1)
    B['reg7'] = np.where(df.REGION.between(1, 7), df.REGION.fillna(0) - 1, -1).astype(int)
    # out-of-block: general health (KNHANES 4 levels) and depression latent bins (CES-D 11, normal scores within year)
    # self-rated health: KGSS wordings and levels differ from KNHANES (KGSS answers are far more positive), so link by equipercentile within
    # each survey year to the KNHANES 2022 adult distribution (random position inside a KGSS answer category)
    from krpersona.blocks import health as HMm
    kd = HMm.load(); kd = kd[~kd.ho] if core.HOLDOUT else kd; kn = np.bincount(kd.general.dropna().astype(int), weights=kd.w[kd.general.notna()], minlength=4); kn_cum = np.cumsum(kn / kn.sum())[:-1]
    rng0 = np.random.default_rng(11); gh = pd.Series(np.nan, index=df.index)
    for v in ['HEALTHY', 'HEALTH21', 'HEALTH23']:
        x = g(v)
        for y_, d_ in df[x.notna() & gh.isna()].groupby('YEAR'):
            c = x[d_.index].astype(int).to_numpy(); w = d_.FINALWT.to_numpy(float)
            F = np.concatenate([[0], np.cumsum(np.bincount(c, weights=w, minlength=6)[1:6]) / w.sum()])
            u = F[c - 1] + rng0.random(len(c)) * (F[c] - F[c - 1]); gh[d_.index] = np.digitize(u, kn_cum)
    cs = pd.concat([(5 - g(f'CESD{i}')) if i in (4, 6, 9, 11) else g(f'CESD{i}') for i in range(1, 12)], axis=1)
    sc = cs.mean(axis=1).where(cs.notna().sum(1) >= 10)
    from scipy.stats import norm
    rng = np.random.default_rng(7); z = pd.Series(np.nan, index=df.index)
    for y_, d_ in df[sc.notna()].groupby('YEAR'):
        s = sc[d_.index].to_numpy() + rng.random(len(d_)) * 1e-6; w = d_.FINALWT.to_numpy(float); o = np.argsort(s)
        cw = (np.cumsum(w[o]) - w[o] / 2) / w.sum(); zz = np.empty(len(s)); zz[o] = norm.ppf(cw); z[d_.index] = zz
    xd = pd.Series(np.digitize(z, XI_CUTS), index=df.index).where(z.notna())
    party = g('GRPPARTY').map({1: 1, 2: 0, 3: 0, 4: 0})
    # quantile-coded scores cut at the MatrAIx base proportions (Signature = top)
    mx = core.MX()
    L = pd.concat([g(v) if v not in ('LOTR2', 'LOTR4', 'LOTR5') else 6 - g(v) for v in ['LOTR1', 'LOTR2', 'LOTR3', 'LOTR4', 'LOTR5', 'LOTR6']], axis=1)
    raws = {'trait_hope_optimism': L.sum(1).where(L.notna().all(1)), 'trait_empathy': g('OTHERS1') + g('OTHERS2')}
    for a, s in raws.items():
        P0 = np.asarray(mx.prior[a], float); P0 = P0 / P0.sum()          # MatrAIx order Signature..Absent
        q = np.full(len(s), np.nan)
        for y_, d_ in df[s.notna()].groupby('YEAR'):
            v = s[d_.index].to_numpy() + rng.random(len(d_)) * 1e-6; w = d_.FINALWT.to_numpy(float); o = np.argsort(-v)
            cw = (np.cumsum(w[o]) - w[o] / 2) / w.sum(); k = np.digitize(cw, np.cumsum(P0)[:-1]); qq = np.empty(len(v)); qq[o] = k; q[d_.index] = qq
        C[a] = q
    C['topic_spirituality'] = g('SPIRITUA')
    return df, C, B, dict(gh=gh, xd=xd, party=party), mx

# ---------------------------------------------------------------- estimation
def years_of(row):
    yrs = [int(x) for x in str(row['추정에 쓰는 연도']).split(', ')]
    gtxt = row['생성 시 연도']; m_ = re.match(r'(\d{4})·(\d{4}) 보간\((\d+):(\d+)\)', gtxt)
    gw = {int(m_.group(1)): int(m_.group(3)), int(m_.group(2)): int(m_.group(4))} if m_ else {int(gtxt[:4]): 1}
    t = sum(gw.values()); return yrs, {k: v / t for k, v in gw.items()}

def block_table():
    """step-1 KGSS block table plus the attributes added in step 5"""
    T = pickle.load(open(paths.art('mapping', 'kgss_analysis.pkl'), 'rb'))['P']
    return pd.concat([T, pd.DataFrame(ADD_ROWS)], ignore_index=True)

def fit_all(quick=False, only=None):
    t0 = time.time(); df, C, B, X_, mx = load()
    T = block_table()
    info = {a: cat_info(a, mx.values) for a in T['속성']}
    Y = {a: pd.Series(recode(C[a].fillna(-99).to_numpy(), info[a]['codes']), index=df.index) for a in T['속성']}
    w_all = df.FINALWT.to_numpy(float)
    models = {}
    for _, row in T.iterrows():
        a = row['속성']
        if only and a not in only: continue
        yrs, gw = years_of(row); age_mode = row['연령 처리']
        bgk = BGC if age_mode == '출생 코호트 맞춤' else BGK
        par_in = [] if row['선택 부모(최대 3)'] == '(배경 변수만)' else [p.strip() for p in row['선택 부모(최대 3)'].split(',')]
        R = (Y[a] >= 0) & df.YEAR.isin(yrs)
        for k in bgk: R &= B[k] >= 0
        for p in par_in: R &= Y[p] >= 0
        n0 = int(R.sum())
        cols, levels, parents = [], [], []
        for k in bgk:
            lab, node, sets = core.BG[k]; cols.append(B[k].to_numpy()); levels.append(len(lab)); parents.append((k, node, sets))
        yi = {y_: i for i, y_ in enumerate(yrs)}
        if len(yrs) > 1:
            cols.append(df.YEAR.map(yi).fillna(0).astype(int).to_numpy()); levels.append(len(yrs)); parents.append(('year', None, None))
        for p in par_in:
            ip = info[p]; cols.append(Y[p].to_numpy()); levels.append(len(ip['codes']))
            parents.append((p, p if ip['mx'] else None, [s if s is None else (s + ['Spiritual but unaffiliated'] if 'None' in s else s) for s in ip['cond']] if ip['mx'] else None))
        for o in OUTB.get(a, []):
            x = X_[o]; nl = 4 if o == 'gh' else 5; miss = (x.isna() & R).any()
            cols.append(x.fillna(nl).astype(int).to_numpy()); levels.append(nl + (1 if miss else 0))
            if o == 'gh': parents.append(('gh', GH_NODE[0], GH_NODE[1] + ([[]] if miss else [])))
            else: parents.append(('xd', None, None))
            if not miss: R &= x.notna()
        Xa = np.stack(cols, 1)[R.to_numpy()]; ya = Y[a][R].to_numpy(); wa = w_all[R.to_numpy()]; wa = wa / wa.mean()
        pairs = [(0, 1), (1, 2)]
        if age_mode == '연령×시기 상호작용' and len(yrs) > 1: pairs.append((1, len(bgk)))
        it = info[a]; target = a if it['mx'] else None; cats = it['cond'] if it['mx'] else [None] * len(it['codes'])
        if it['mx'] and a == 'demo_religion_affiliation': cats = [s + ['Spiritual but unaffiliated'] if 'None' in s else s for s in cats]
        Mdl = core.fit_attr(mx, a, target, cats, it['labels'], parents, levels, Xa, ya, wa, pairs=pairs, quick=quick)
        Mdl.update(years=yrs, gen_w=gw, age_mode=age_mode, cohort_shift=(2020 - max(gw)) if age_mode == '출생 코호트 맞춤' else 0, bg=bgk,
                   par_in=par_in, outb=OUTB.get(a, []), cat=it, n_window=n0)
        models[a] = Mdl
        print(f"{a:34s} n {Mdl['n']:5d} K {len(it['codes'])} par {par_in + OUTB.get(a, [])} s {Mdl['info'].get('s_main')}/{Mdl['info'].get('s_int')} "
              f"ll {Mdl['ll']:.3f} prior {Mdl['ll_prior']:.3f} marg {Mdl['ll_marginal']:.3f} {round(time.time() - t0)}s", flush=True)
    # political engagement: party membership effect from 2004/2014 (fixed offset added at generation)
    a = 'demo_political_engagement'; R = (Y[a] >= 0) & X_['party'].notna()
    for k in BGC: R &= B[k] >= 0
    Xp = np.stack([B[k].to_numpy() for k in BGC] + [X_['party'].fillna(0).astype(int).to_numpy()], 1)[R.to_numpy()]
    lv = [len(core.BG[k][0]) for k in BGC] + [2]; par = [(k,) + core.BG[k][1:] for k in BGC] + [('party', None, None)]
    it = info[a]; wa = w_all[R.to_numpy()]; wa = wa / wa.mean()
    Mp = core.fit_attr(mx, 'party_offset', a, it['cond'], it['labels'], par, lv, Xp, Y[a][R].to_numpy(), wa, quick=quick)
    D = core.pml.Design(lv); off = Mp['B'][D.off[3] + 1] - Mp['B'][D.off[3]]
    party_share = float(np.average(X_['party'][R], weights=wa))
    yr = sorted(df.YEAR[R].unique().tolist())
    print('party offset', off.round(2), 'n', int(R.sum()), 'years', yr, 'member share', round(party_share, 4))
    extra = dict(party_offset=off.tolist(), party_years=yr, party_n=int(R.sum()), party_share=party_share)
    if only:
        K0 = pickle.load(open(paths.art('models', f'kgss_models{core.SUF}.pkl'), 'rb')); K0['models'].update(models); models = K0['models']
    vl = fit_value_links(df, C, B, X_, Y, models, list(T['속성']))
    pickle.dump(dict(models=models, extra=extra, order=list(T['속성']), vlinks=vl), open(paths.art('models', f'kgss_models{core.SUF}.pkl', mkdir=True), 'wb'))
    return models, extra

# ---------------------------------------------------------------- values -> attitudes
def value_scores(df):
    """KGSS 2021 respondents: relative value scores z (as generated) minus their group mean from the value model (sex, cohort, education)"""
    from krpersona.blocks import values as VM
    VP = json.load(open(paths.art('models', f'value_params{core.SUF}.json')))
    it = [f'VALUES{i}' for i in range(1, 23)]
    ok = (df.YEAR == 2021) & df[it].apply(lambda c: c.between(1, 6)).all(axis=1)
    d = df[ok]
    V = np.stack([np.mean([7 - d[f'VALUES{i}'].to_numpy(float) for i, v in VM.ITEM_VAL.items() if v == val], axis=0) for val in VM.VALS], 1)
    vbar = np.array(VP['vbar']); sd = np.array(VP['sd'])
    z = (V - V.mean(1, keepdims=True) - vbar[None, :] + vbar.mean()) / sd[None, :]
    edu = d.EDUC.to_numpy(); grad = d.GRADUATE.to_numpy() == 1
    edu3 = np.select([(edu >= 6) & (edu <= 7) & grad, np.isin(edu, [4, 5]) & grad, (edu >= 6) & (edu <= 7)], [2, 1, 1], 0)
    th = np.deg2rad(VP['theta_deg']); Uv = np.stack([np.cos(th), np.sin(th)], 1)
    gm = VM.phiV(d.SEX.to_numpy(), 2021 - d.AGE.to_numpy(), edu3) @ np.array(VP['G']).T @ Uv.T / sd[None, :]
    zt = z - gm; w = d.FINALWT.to_numpy(float); zt = zt - np.average(zt, axis=0, weights=w)
    Z = np.full((len(df), 10), np.nan); Z[ok.to_numpy()] = zt
    return Z

def rows_for(M, a, df, B, Y, X_):
    """design matrix columns of a fitted KGSS model for all respondents, and the rows where every column is valid"""
    cols = []; R = (Y[a] >= 0).to_numpy().copy()
    for k in M['bg']: cols.append(B[k].to_numpy()); R &= B[k].to_numpy() >= 0
    if 'year' in M['parents']:
        yi = {y_: i for i, y_ in enumerate(M['years'])}; cols.append(df.YEAR.map(yi).fillna(-1).astype(int).to_numpy()); R &= cols[-1] >= 0
    for p in M['par_in']: cols.append(Y[p].to_numpy()); R &= Y[p].to_numpy() >= 0
    for j, o in enumerate(M['outb']):
        nl = 4 if o == 'gh' else 5; lv = M['levels'][len(cols)]; x = X_[o]
        cols.append(x.fillna(nl).astype(int).to_numpy())
        if lv == nl: R &= x.notna().to_numpy()
    return np.stack(cols, 1), R

def _fit_offset(L0, Z, y, w, K, s, maxiter=300):
    from scipy.optimize import minimize
    n = len(y); Yh = np.zeros((n, K)); Yh[np.arange(n), y] = 1
    def f(b):
        Bv = b.reshape(10, K); Zl = L0 + Z @ Bv; Zl = Zl - Zl.max(1, keepdims=True); E = np.exp(Zl); P = E / E.sum(1, keepdims=True)
        ll = (w * np.log(P[np.arange(n), y] + 1e-300)).sum(); g = Z.T @ (w[:, None] * (P - Yh)) + Bv / s ** 2
        return -ll + 0.5 * (Bv ** 2).sum() / s ** 2, g.ravel()
    return minimize(f, np.zeros(10 * K), jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter)).x.reshape(10, K)

def _ll(L0, Z, Bv, y, w):
    Zl = L0 + Z @ Bv; Zl = Zl - Zl.max(1, keepdims=True); P = np.exp(Zl); P /= P.sum(1, keepdims=True)
    return float((w * np.log(P[np.arange(len(y)), y] + 1e-12)).sum())

def fit_value_links(df, C, B, X_, Y, models, order, grid=(0.025, 0.05, 0.1, 0.2, 0.4), folds=5):
    """for KGSS attributes asked in 2021: logit = (fitted KGSS model) + z_tilde B_v + delta, z_tilde = value scores minus group mean.
    B_v by ridge with 5-fold CV against B_v = 0; delta re-fits the category intercepts so the 2021 KGSS distribution is unchanged."""
    Z = value_scores(df); out = {}
    for a in order:
        M = models[a]
        if 2021 not in M['years'] or M['age_mode'] == '출생 코호트 맞춤': continue
        X, R = rows_for(M, a, df, B, Y, X_); R &= (df.YEAR == 2021).to_numpy() & ~np.isnan(Z).any(1)
        if R.sum() < 300: continue
        Xr = X[R].copy()
        if 'year' in M['parents']: Xr[:, M['parents'].index('year')] = M['years'].index(2021)
        D = core.pml.Design(M['levels'], M['pairs']); L0 = np.log(core.pml.predict(D, M['B'], Xr) + 1e-300)
        y = Y[a].to_numpy()[R]; w = df.FINALWT.to_numpy(float)[R]; w = w / w.mean(); Zr = Z[R]; K = L0.shape[1]
        fold = np.random.default_rng(3).integers(0, folds, len(y))
        cv = {0.0: sum(_ll(L0[fold == f_], Zr[fold == f_], np.zeros((10, K)), y[fold == f_], w[fold == f_]) for f_ in range(folds))}
        for s_ in grid:
            cv[s_] = sum(_ll(L0[fold == f_], Zr[fold == f_], _fit_offset(L0[fold != f_], Zr[fold != f_], y[fold != f_], w[fold != f_], K, s_), y[fold == f_], w[fold == f_]) for f_ in range(folds))
        sb = max(cv, key=cv.get)
        if sb == 0.0: continue
        Bv = _fit_offset(L0, Zr, y, w, K, sb)
        tgt = np.average(np.exp(L0) / np.exp(L0).sum(1, keepdims=True), axis=0, weights=w); dl = np.zeros(K)
        for _ in range(50):
            Zl = L0 + Zr @ Bv + dl; Zl -= Zl.max(1, keepdims=True); P = np.exp(Zl); P /= P.sum(1, keepdims=True)
            dl += np.log(tgt / np.average(P, axis=0, weights=w))
        out[a] = dict(Bv=Bv, delta=dl, s=sb, n=int(R.sum()), cv_gain=(cv[sb] - cv[0.0]) / w.sum(), cv={str(k): v for k, v in cv.items()})
        print(f"value link {a:34s} n {R.sum():4d} s {sb} cv gain/person {out[a]['cv_gain']:.4f}", flush=True)
    return out

# ---------------------------------------------------------------- generation
def census_participation():
    """P(사회분야단체 참여 | sex, ag7, edu5) from the census sample (person weights, age 20+), for the volunteering two-stage rule"""
    R = pd.read_csv(paths.raw('census2020', 'person'), encoding='cp949', dtype=str,
                    usecols=['성별코드', '만연령', '교육정도코드', '수학구분코드', '출생국적_대한민국여부', '현재국적_대한민국여부', '출생지구분코드', '인구가중값',
                             '사회활동_사회분야단체_참여및활동여부'])
    R = R[(R.출생국적_대한민국여부 == '1') & (R.현재국적_대한민국여부 == '1') & (R.출생지구분코드 != '4') & (R.만연령.astype(int) >= 20)]
    y = R.사회활동_사회분야단체_참여및활동여부.astype(float).to_numpy() == 1
    cell = (R.성별코드.astype(int).to_numpy() - 1) * 100 + core.ag7(R.만연령.astype(int).to_numpy()) * 10 + core.edu5_census(R.교육정도코드.astype(int), pd.to_numeric(R.수학구분코드).fillna(1).to_numpy())
    w = R.인구가중값.astype(float).to_numpy()
    t = pd.DataFrame(dict(c=cell, y=y * w, w=w)).groupby('c').sum(); return (t.y / t.w).to_dict(), float((y * w).sum() / w.sum())

def predict_mix(M, X, gen_years_col=None):
    """probabilities at the generation year(s): mix of year-specific predictions with the inverse-distance weights"""
    D = core.pml.Design(M['levels'], M['pairs'])
    if 'year' not in M['parents']: return core.pml.predict(D, M['B'], X)
    j = M['parents'].index('year'); P = 0
    for y_, wgt in M['gen_w'].items():
        Xy = X.copy(); Xy[:, j] = M['years'].index(y_); P = P + wgt * core.pml.predict(D, M['B'], Xy)
    return P

def generate(K, bg, age, oob, rng, part_rate=None):
    """K: kgss_models.pkl dict; bg: dict of background codes (sex, ag7, edu5, mar4, emp2, urban2, reg7) as int arrays;
    age: 2020 age; oob: dict gh (0-3), xd (0-4), party (0/1), soc (0/1 census social-organisation participation).
    returns dict attr -> Korean category index"""
    models = K['models']; n = len(age); Y = {}
    K_LABELS.update({a_: models[a_]['cat']['labels'] for a_ in K['order']})
    for a in K['order']:
        M = models[a]; it = M['cat']; cols = []
        for k in M['bg']:
            if k == 'ag7' and M['cohort_shift']: cols.append(core.ag7(np.maximum(19, age - M['cohort_shift'])))
            else: cols.append(bg[k])
        if 'year' in M['parents']: cols.append(np.zeros(n, int))
        for p in M['par_in']: cols.append(Y[p])
        for o in M['outb']: cols.append(oob[o])
        P = predict_mix(M, np.stack(cols, 1).astype(int))
        vl = None if core.NO_VLINKS else K.get('vlinks', {}).get(a)
        if vl is not None and oob.get('zt') is not None:
            L = np.log(np.maximum(P, 1e-300)) + oob['zt'] @ vl['Bv'] + vl['delta']; L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
        if a == 'demo_political_engagement':
            L = np.log(np.maximum(P, 1e-300)) + np.outer(oob['party'], K['extra']['party_offset']); L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
        al = allowed(a, Y, n, it, oob)
        if al is not None:   # after round 3: a row whose allowed categories all have ~0 probability takes an allowed category uniformly
            Q = P * al; z = Q.sum(1) <= 1e-12   # (before, an all-zero row fell to the first category, which could be a forbidden one)
            if z.any(): Q[z] = np.where(al[z].sum(1, keepdims=True) > 0, al[z], P[z])
            P = Q / np.maximum(Q.sum(1, keepdims=True), 1e-300)
        if a == 'lstyle_volunteering' and part_rate is not None:
            never = it['labels'].index('Never'); pv = 1 - P[:, never]
            cell = bg['sex'] * 100 + bg['ag7'] * 10 + bg['edu5']; pp = np.array([part_rate.get(c, 0.0) for c in cell]); pp[age < 20] = 0
            q = np.clip((pv - pp) / np.maximum(1 - pp, 1e-9), 0, 1)
            vol = np.where(oob['soc'] == 1, True, rng.random(n) < q)
            F = P.copy(); F[:, never] = 0; F /= np.maximum(F.sum(1, keepdims=True), 1e-300)
            P = np.where(vol[:, None], F, np.eye(P.shape[1])[never][None, :].repeat(n, 0))
        c = P.cumsum(1); u = rng.random(n)[:, None] * c[:, -1:]; Y[a] = np.minimum((c < u).sum(1), P.shape[1] - 1)
    return Y

# step 5 round 3: KGSS value -> (condition attribute, its values, forbidden labels); the condition is an earlier KGSS attribute or a Korean output
KPAIRS = {'mft_authority_subversion': [('att_authority', ['Enthusiast'], ['Very low']), ('att_authority', ['Opposed'], ['Very high'])],
          'political_lean': [('big5_liberalism', ['Very high'], ['Right']), ('big5_liberalism', ['Very low'], ['Left'])],
          'kr_happiness': [('kr_depression_level', ['Severe'], ['Very happy'])],
          'trait_hope_optimism': [('cog_optimism', ['Very high'], ['Absent']), ('cog_optimism', ['None'], ['Signature'])],
          'pref_lead_vs_follow': [('trait_leadership', ['Absent'], ['Always leads']), ('trait_leadership', ['Signature'], ['Prefers to follow'])],
          'val_spirituality_faith': [('religiosity', ['Devout'], ['Irrelevant'])],
          'trait_empathy': [('big5_sympathy', ['Very high'], ['Absent']), ('big5_sympathy', ['Very low'], ['Signature'])]}   # after round 3
K_LABELS = {}   # filled by generate(): attribute -> category labels

def allowed(a, Y, n, it, oob=None):
    """logical consistency inside the block and with census values (drawn-in masks)"""
    K = len(it['codes'])
    if a == 'lifex_travel_breadth' and oob is not None and oob.get('abroad') is not None:   # census: lived abroad 1 or 5 years ago
        al = np.ones((n, K)); al[oob['abroad'] == 1, it['labels'].index('Never left country')] = 0; return al
    if oob is not None and oob.get('out') is not None and a in KPAIRS:
        al = np.ones((n, K))
        for cond_attr, cond_vals, bad in KPAIRS[a]:
            src = Y.get(cond_attr) if cond_attr in Y else None
            if src is not None: hit = np.isin(np.array(K_LABELS[cond_attr], dtype=object)[src], cond_vals)
            else: hit = np.isin(np.asarray(oob['out'][cond_attr], dtype=object), cond_vals)
            for b in bad: al[hit, it['labels'].index(b)] = 0
        if a != 'values_priority': return al
    if a == 'values_priority' and oob is not None and oob.get('vp_low') is not None:
        # step 5: the top value priority cannot be a value the person rates Minor / Irrelevant (or Low / Very low) in the value model;
        # a category that is a set of priorities stays allowed while one member is not low (the member is restricted at output time)
        low = oob['vp_low']; al = np.ones((n, K))
        for k, c in enumerate(it['cond']):
            al[:, k] = ~np.all(np.stack([low[m] for m in c], 1), 1)
        return al
    if a in ('religiosity', 'topic_spirituality', 'lifex_faith_journey') and 'demo_religion_affiliation' in Y:
        none = Y['demo_religion_affiliation'] == 8
        al = np.ones((n, K))
        if a == 'religiosity': al[none, 1:] = 0; al[~none, 0] = 0
        if a == 'topic_spirituality': al[np.ix_(none, [0, 1])] = 0; al[np.ix_(~none, [2, 3])] = 0
        if a == 'lifex_faith_journey': al[np.ix_(none, [0, 1])] = 0; al[np.ix_(~none, [2, 3])] = 0
        return al
    return None

def outputs(K, Y):
    """Korean category index -> (output label or None for MatrAIx draw inside the set, conditioning set) per persona"""
    res = {}
    for a, y in Y.items():
        it = K['models'][a]['cat']
        out = np.array(it['out'], dtype=object)[y]
        if a == 'demo_religion_affiliation' and 'topic_spirituality' in Y:
            sp = (y == 8) & (Y['topic_spirituality'] == 2)    # no religion and SPIRITUA=3 -> Spiritual but unaffiliated
            out = out.copy(); out[sp] = 'Spiritual but unaffiliated'
        cond = [it['cond'][k] for k in y] if it['mx'] else None
        if a == 'demo_religion_affiliation' and 'topic_spirituality' in Y: cond = [['Spiritual but unaffiliated'] if s else c for s, c in zip(sp, cond)]
        res[a] = (out, cond)
    return res

if __name__ == '__main__':
    quick = '--quick' in sys.argv
    if '--vlinks-only' in sys.argv:
        p = paths.art('models', f'kgss_models{core.SUF}.pkl'); K = pickle.load(open(p, 'rb')); df, C, B, X_, mx = load()
        info = {a: K['models'][a]['cat'] for a in K['order']}
        Y = {a: pd.Series(recode(C[a].fillna(-99).to_numpy(), info[a]['codes']), index=df.index) for a in K['order']}
        K['vlinks'] = fit_value_links(df, C, B, X_, Y, K['models'], K['order']); pickle.dump(K, open(p, 'wb')); sys.exit()
    only = [a for a in sys.argv[1:] if not a.startswith('--')] or None
    fit_all(quick=quick, only=only)
