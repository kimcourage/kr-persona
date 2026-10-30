"""Stage-0 generator (0단계: census values, rules, temporary personality parameters; no Korean survey models yet): synthetic census base -> Korean attributes (mapping, rules, personality, children) -> MatrAIx for the rest."""
import json, numpy as np, pandas as pd
from krpersona import paths
from krpersona.matraix.sampler import KRSampler
from krpersona.generate.assign import build

NA = 'Not applicable'
REAL_P = lambda: paths.art('census', 'P.parquet')      # prepared real census person extract (census/prep.py)
REAL_RAW = lambda: paths.raw('census2020', 'person')
FMAGE = {1: (12, 14), 2: (15, 19), 3: (20, 24), 4: (25, 29), 5: (30, 34), 6: (35, 39), 7: (40, 60)}
JTEN_LB = {1: 0, 2: 0, 3: 1, 4: 3, 5: 5, 6: 10, 7: 15, 8: 20}
EXP_BINS = [('0-2', 0, 2), ('3-5', 3, 5), ('6-10', 6, 10), ('11-20', 11, 20), ('20+', 20, 99)]
IND = {c: i + 1 for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTU')}
BFI2 = {'E': ['sociability', 'assertiveness', 'energy_level'], 'A': ['compassion', 'respectfulness', 'trust'],
        'C': ['organization', 'productiveness', 'responsibility'], 'N': ['anxiety', 'depression', 'emotional_volatility'],
        'O': ['intellectual_curiosity', 'aesthetic_sensitivity', 'creative_imagination']}
BFI2_DOMAIN = {'E': 'bfi2_domain_extraversion', 'A': 'bfi2_domain_agreeableness', 'C': 'bfi2_domain_conscientiousness',
               'N': 'bfi2_domain_negative_emotionality', 'O': 'bfi2_domain_open_mindedness'}
IPIP = {'N': ['anxiety', 'anger', 'depression', 'self_consciousness', 'immoderation', 'vulnerability'],
        'E': ['friendliness', 'gregariousness', 'assertiveness', 'activity_level', 'excitement_seeking', 'cheerfulness'],
        'O': ['imagination', 'artistic_interest', 'emotionality', 'adventurousness', 'intellect', 'liberalism'],
        'A': ['trust', 'morality', 'altruism', 'cooperation', 'modesty', 'sympathy'],
        'C': ['self_efficacy', 'orderliness', 'dutifulness', 'achievement_striving', 'self_discipline', 'cautiousness']}
DOM = {'O': 'High openness', 'C': 'High conscientiousness', 'E': 'High extraversion', 'A': 'High agreeableness', 'N': 'High neuroticism'}
LEVELS = ['Very high', 'High', 'Average', 'Low', 'Very low']; TAU = np.array([-1.51, -0.51, 0.51, 1.51])
# stage-0 temporary personality parameters (replaced by the KGSS estimates in generate/generator.py)
TMP = dict(B=None, Phi=np.eye(5), lam_bfi2=0.8, lam_ipip=0.8, lam_mbti=0.7, lam_battery=0.7)
FACT = ['O', 'C', 'E', 'A', 'N']
ENGLISH_FIRST = False  # english_proficiency comes before lang_english in the MatrAIx order


def level(z):  # standard score -> 5 MatrAIx levels
    return np.array(LEVELS, dtype=object)[4 - np.searchsorted(TAU, z)]


class Base:
    """synthetic census base: eligible personas, post-stratified to person-weight margins, household-derived features"""
    def __init__(self, sh, sp, rng):
        self.rng = rng
        H = pd.read_parquet(sh); P = pd.read_parquet(sp).merge(H, on='hid')
        P['by'] = 2020 - P.age
        el = (P.nat == 1) & (P.bplace != 4) & (P.age >= 19)
        # own children and grandchildren from the household roster
        ages = lambda rels: P[P.rel.isin(rels)].groupby('hid').age.apply(list)
        c3, c1_10, c2, c7 = ages([3]), ages([1, 10]), ages([2]), ages([7])
        has = lambda r: set(P.hid[P.rel == r])
        h7, h3 = has(7), has(3)
        own = pd.Series([[]] * len(P), dtype=object)
        m12 = el & P.rel.isin([1, 2]); own[m12] = P.hid[m12].map(c3).apply(lambda x: x if isinstance(x, list) else [])
        m5 = el & (P.rel == 5); own[m5] = P.hid[m5].map(c1_10).apply(lambda x: x if isinstance(x, list) else [])
        m6 = el & (P.rel == 6); own[m6] = P.hid[m6].map(c2).apply(lambda x: x if isinstance(x, list) else [])
        m34 = el & P.rel.isin([3, 4]) & P.hid.isin(h7)
        nad = P[P.rel.isin([3, 4]) & (P.age >= 19)].groupby('hid').size(); n3 = P[(P.rel == 3) & (P.age >= 19)].groupby('hid').size()
        res = []
        for h, a_ in zip(P.hid[m34], P.age[m34]):
            gc = [x for x in c7[h] if 15 <= a_ - x <= 45]
            if nad.get(h, 0) > 2: gc = [x for x in gc if self.rng.random() < 1 / max(1, n3.get(h, 1))]
            res.append(gc)
        own[m34] = pd.Series(res, index=P.index[m34], dtype=object)
        P['own'] = own.to_numpy()
        P['grand'] = (P.rel.isin([1, 2]) & P.hid.isin(h7)) | (P.rel.isin([5, 6]) & P.hid.isin(h3))
        # married men in a head-spouse pair use the spouse's fertility record
        f = P[P.rel.isin([1, 2]) & (P.sex == 2)].drop_duplicates('hid').set_index('hid')
        man = el & (P.sex == 1) & (P.mar == 2) & P.rel.isin([1, 2])
        P.loc[man, 'births'] = P.hid[man].map(f.births).to_numpy(); P.loc[man, 'deaths'] = P.hid[man].map(f.deaths).to_numpy()
        keep = ['hid', 'pos', 'rel', 'sex', 'age', 'mar', 'nat', 'bplace', 'r5', 'r1', 'edu', 'sch', 'fmage', 'econ', 'status', 'occ', 'ind', 'jten', 'place',
                'comm', 'mode', 'lim', 'births', 'deaths', 'prejob', 'brk', 'sido', 'dong', 'gtype', 'hkind', 'k', 'tenure_h', 'car', 'pet', 'own', 'grand']
        self.P = P.loc[el, keep].reset_index(drop=True)
        # post-stratification to real person-weighted sex x age5 x sido of the same population
        R = pd.read_csv(REAL_RAW(), encoding='cp949', dtype=str, usecols=['성별코드', '만연령', '행정구역시도코드', '출생국적_대한민국여부', '현재국적_대한민국여부', '출생지구분코드', '인구가중값'])
        R = R[(R.출생국적_대한민국여부 == '1') & (R.현재국적_대한민국여부 == '1') & (R.출생지구분코드 != '4') & (R.만연령.astype(int) >= 19)]
        cell = lambda s, a, d: s.astype(int).astype(str) + '_' + (np.minimum(a.astype(int), 85) // 5).astype(str) + '_' + d.astype(int).astype(str)
        tgt = pd.Series(R.인구가중값.astype(float).to_numpy()).groupby(cell(R.성별코드, R.만연령, R.행정구역시도코드).to_numpy()).sum()
        self.P['cell'] = cell(self.P.sex, self.P.age, self.P.sido).to_numpy()
        cnt = self.P.cell.map(self.P.cell.value_counts())
        self.P['w0'] = self.P.cell.map(tgt).fillna(0) / cnt
        self.P['w'] = self.P.w0
        # mother-child age gap distribution (real census, mothers who are head or spouse with co-resident children)
        rp = pd.read_parquet(REAL_P())
        mo = rp[rp.rel.isin([1, 2]) & (rp.sex == 2)][['hid', 'age']].rename(columns={'age': 'ma'})
        ch = rp[rp.rel == 3][['hid', 'age']].merge(mo, on='hid'); gap = (ch.ma - ch.age); self.gap = gap[(gap >= 15) & (gap <= 50)].to_numpy()

    def draw(self, n):
        p = self.P.w.to_numpy(); p = p / p.sum()
        return self.P.iloc[self.rng.choice(len(self.P), size=n, p=p)].reset_index(drop=True)


class Gen0:
    def __init__(self, sh, sp, seed=0, switch_on=True):
        self.rng = np.random.default_rng(seed)
        self.rows, K0, PH, self.ids, self.emit = build()
        self.K = set(K0); self.PH = PH; self.on = switch_on
        self.base = Base(sh, sp, self.rng)
        self.S = KRSampler(self.K, placeholder=set(PH), seed=seed, extra_masks=self._extra_masks())
        self.V = self.S.values
        self.ph_order = [t for t in self.S.topo if t in set(PH)]
        self.delta = 0.30

    # generation-time masks on (5) MatrAIx attributes that would contradict Korean values
    def _extra_masks(self):
        g = json.load(open(paths.graph_path()))
        V = {n['id']: n['values'] for n in g['nodes']}
        def bad(t, vals): return np.array([0.0 if v in vals else 1.0 for v in V[t]])
        def only(t, vals): return np.array([1.0 if v in vals else 0.0 for v in V[t]])
        ms = []
        for t in V:
            if t.startswith('cult_') and t != 'cult_south_korea': ms.append((t, {'region': ['East Asia']}, bad(t, ['Native'])))
            if t.startswith('lang_') and t != 'lang_korean': ms.append((t, {'region': ['East Asia']}, bad(t, ['Native'])))
        ms.append(('english_proficiency', {'region': ['East Asia']}, bad('english_proficiency', ['Native'])))
        for t in V:   # never left hometown -> has not lived in another country
            if t.startswith('cult_') and t != 'cult_south_korea': ms.append((t, {'lifex_geographic_mobility': ['Never left hometown']}, bad(t, ['Lived there'])))
        EN = {'Native': 'Native', 'Fluent': 'Fluent (C1-C2)', 'Conversational': 'Intermediate (B1-B2)', 'Basic': 'Basic (A1-A2)', 'None': 'None'}
        SRC, TGT = ('lang_english', 'english_proficiency') if ENGLISH_FIRST else ('english_proficiency', 'lang_english')
        for a_, b_ in EN.items():   # the two English attributes describe the same skill: keep them on the same level
            x, y = (a_, b_) if ENGLISH_FIRST else (b_, a_)
            ms.append((TGT, {SRC: [x]}, only(TGT, [y])))
        ms.append(('lifex_relationship_history', {'demo_marital_status': ['Single', 'In a relationship', 'Engaged']}, bad('lifex_relationship_history', ['Married once', 'Remarried', 'Widowed'])))
        ms.append(('lifex_relationship_history', {'demo_marital_status': ['Married', 'Divorced', 'Widowed']}, bad('lifex_relationship_history', ['Limited'])))
        ms.append(('lifex_loss_experience', {'demo_marital_status': ['Widowed']}, only('lifex_loss_experience', ['Lost a partner / child', 'Multiple bereavements'])))
        ms.append(('lifex_education_journey', {'highest_education': ['No formal', 'Primary', 'Secondary', 'Vocational / cert', 'Some college', "Associate's", "Bachelor's"]}, bad('lifex_education_journey', ['Advanced degrees'])))
        ms.append(('lstyle_work_schedule', {'demo_employment_status': ['Full-time', 'Part-time', 'Self-employed', 'Gig / freelance']}, bad('lstyle_work_schedule', ['Unemployed'])))
        ms.append(('lstyle_work_schedule', {'demo_employment_status': ['Student', 'Unemployed', 'Retired', 'Homemaker']}, only('lstyle_work_schedule', ['Unemployed'])))
        ms.append(('lifex_faith_journey', {'demo_religion_affiliation': ['None', 'Atheist / agnostic']}, bad('lifex_faith_journey', ['Lifelong faith', 'Converted', 'Returned to faith'])))
        ms.append(('lifex_faith_journey', {'demo_religion_affiliation': ['Christian', 'Buddhist', 'Folk / traditional', 'Muslim', 'Hindu', 'Jewish', 'Sikh']}, bad('lifex_faith_journey', ['Always secular', 'Lapsed / left faith'])))
        return ms

    # ------------------------------------------------------------ helpers to record values
    def _set(self, t, out, cond=None):
        """out: output strings (custom allowed); cond: MatrAIx labels for conditioning (None -> same as out; sentinel if not a category)"""
        out = np.asarray(out, dtype=object); self.out[t] = out
        self.codes[t] = self.S.code_of(t, out if cond is None else np.asarray(cond, dtype=object)); self.avail.add(t)

    def _alloc(self, t, sets, out_custom=None):
        """sets: list (per row) of allowed MatrAIx labels; draws inside each set with MatrAIx q from already-drawn Korean parents"""
        V = self.V[t]; al = np.zeros((len(sets), len(V)), bool)
        for i, s in enumerate(sets):
            for v in s: al[i, V.index(v)] = True
        empty = ~al.any(1); al[empty, 0] = True
        c = self.S.draw_korean(t, self.codes, self.avail, al); c[empty] = len(V)
        lab = np.array(list(V) + [None], dtype=object)[c]
        out = lab if out_custom is None else np.where(pd.isna(out_custom), lab, out_custom)
        self.out[t] = np.asarray(out, dtype=object); self.codes[t] = c; self.avail.add(t)

    # ------------------------------------------------------------ main
    def make(self, n):
        D = self.base.draw(n); self.D = D; rng = self.rng
        self.out, self.codes, self.avail = {}, {}, set()
        age = D.age.to_numpy(); sex = D.sex.to_numpy()
        # census values mapped to MatrAIx attributes
        self._set('age_bracket', pd.cut(age, [18, 24, 34, 44, 54, 64, 74, 84, 200], labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+']).astype(object))
        self._set('gender_identity', np.where(sex == 1, 'Man', 'Woman'))
        self._set('region', np.full(n, 'East Asia', dtype=object))
        metro = D.sido.isin([11, 21, 22, 23, 24, 25, 26])
        self._set('urbanicity', np.select([(D.dong == 3) & metro, D.dong == 3, D.dong == 1], ['Dense urban', 'Suburban', 'Small town'], 'Rural'))
        e, s = D.edu.to_numpy(), D.sch.to_numpy(); grad = s == 1
        edu = np.select([e == 1, (e == 2) & grad, e == 2, (e == 3) & grad, e == 3, e == 4, (e == 5) & grad, e == 5, (e == 6) & grad, e == 6,
                         (e == 7) & np.isin(s, [1, 3]), e == 7, (e == 8) & grad, e == 8],
                        ['No formal', 'Primary', 'No formal', 'Secondary', 'Primary', 'Secondary', "Associate's", 'Some college', "Bachelor's", 'Some college',
                         "Master's", "Bachelor's", 'DOC', "Master's"], 'Secondary').astype(object)
        self._alloc('highest_education', [['Doctorate', 'Postdoc'] if x == 'DOC' else [x] for x in edu])
        mar = D.mar.to_numpy()
        self._alloc('demo_marital_status', [['Single', 'In a relationship', 'Engaged'] if m == 1 else [{2: 'Married', 3: 'Widowed', 4: 'Divorced'}[m]] for m in mar])
        self._set('demo_citizenship_status', np.full(n, 'Citizen by birth', dtype=object))
        econ, st = D.econ.to_numpy(), D.status.to_numpy(); work = np.isin(econ, [1, 2, 3])
        enrolled = np.isin(s, [2, 4]); ever_f = (sex == 2) & np.isin(mar, [2, 3, 4]); pj, bk = D.prejob.to_numpy(), D.brk.to_numpy()
        emp_sets, emp_out = [], []
        for i in range(n):
            o = None
            if work[i]:
                if st[i] == 1: S_ = ['Full-time'] if econ[i] in (1, 3) else ['Part-time']
                elif st[i] in (2, 3): S_ = ['Self-employed', 'Gig / freelance']
                else: S_, o = ['Self-employed'], 'Unpaid family worker'
            elif enrolled[i]: S_ = ['Student']
            elif age[i] >= 60:
                if not ever_f[i]: S_ = ['Retired']
                elif pj[i] == 2: S_ = ['Homemaker']
                elif bk[i] == 1: S_ = ['Homemaker', 'Retired']
                else: S_ = ['Retired']
            elif ever_f[i] and pj[i] == 2: S_ = ['Homemaker']
            else: S_ = ['Unemployed', 'Homemaker']
            emp_sets.append(S_); emp_out.append(o)
        self._alloc('demo_employment_status', emp_sets, np.array(emp_out, dtype=object))
        ind = D.ind.to_numpy(); occ = D.occ.to_numpy()
        L = {v: k for k, v in IND.items()}
        DMAP = {'A': ['Agriculture'], 'B': ['Manufacturing'], 'C': ['Manufacturing'], 'D': ['Engineering'], 'E': ['Engineering'], 'F': ['Skilled Trades'],
                'G': ['Business & Management'], 'L': ['Business & Management'], 'N': ['Business & Management'], 'H': ['Business & Management'],
                'I': ['Hospitality'], 'T': ['Hospitality'], 'J': ['Software & AI', 'Media & Journalism'], 'K': ['Finance & Economics'],
                'M': ['Law & Policy', 'Engineering', 'Natural Sciences', 'Social Sciences', 'Business & Management'], 'O': ['Public Sector'], 'U': ['Public Sector'],
                'P': ['Education'], 'Q': ['Healthcare & Medicine'], 'R': ['Arts & Humanities'], 'S': ['Skilled Trades', 'Hospitality']}
        dsets = [DMAP[L[int(x)]] if w and x == x else [] for x, w in zip(ind, work)]
        self._alloc('domain', dsets, np.where(work, None, NA).astype(object))
        RCUST = {3: ('Office / clerical', ['Operations', 'Finance', 'HR']), 4: ('Service', ['Operations']), 5: ('Retail sales', ['Sales / GTM']),
                 6: ('Agriculture / fishery', ['Operations']), 7: ('Skilled trades / craft', ['Operations']), 8: ('Machine operation / assembly', ['Operations']),
                 9: ('Elementary labor', ['Operations']), 10: ('Other occupation', ['Operations'])}
        rsets, rout = [], []
        for w_, o_, i_ in zip(work, occ, ind):
            if not w_: rsets.append([]); rout.append(NA); continue
            o_ = int(o_); il = L[int(i_)]
            if o_ == 1: rsets.append(['Executive']); rout.append(None)
            elif o_ == 2:
                rsets.append({'P': ['Teaching'], 'Q': ['Clinical'], 'J': ['Engineering'], 'K': ['Finance'], 'M': ['Research', 'Legal', 'Design', 'Engineering']}.get(il, ['Research', 'Engineering', 'Design'])); rout.append(None)
            else: rsets.append(RCUST[o_][1]); rout.append(RCUST[o_][0])
        self._alloc('role_function', rsets, np.array(rout, dtype=object))
        rel = D.rel.to_numpy(); ten = D.tenure_h.to_numpy(); k = D.k.to_numpy(); hk = D.hkind.to_numpy()
        self._set('lstyle_household_size', np.select([np.isin(hk, [4, 5]), k == 1, k == 2, k <= 4], ['Communal', 'Lives alone', '2 people', '3-4 people'], '5+ people'))
        hs = [['Living with family'] if r == 3 else (['Own outright', 'Mortgage'] if t == 1 else ['Renting']) for r, t in zip(rel, ten)]
        self._alloc('demo_housing_status', hs)
        comm, mode, place = D.comm.to_numpy(), D['mode'].to_numpy(), D.place.to_numpy()
        cm = np.where(np.isin(comm, [1, 2]), pd.Series(mode).map({1: 'Walk', 2: 'Bike', 3: 'Car', 4: 'Public transit', 5: 'Public transit', 6: 'Rideshare', 7: 'Other'}).to_numpy(),
                      np.where(work & (place == 2), 'Remote', NA)).astype(object)
        self._set('lstyle_commute_mode', cm)
        lim = D.lim.fillna(0).astype(int).to_numpy(); nb = np.array([bin(x).count('1') for x in lim])
        self._set('demo_disability_status', np.select([lim == 0, nb >= 2, np.isin(lim, [1, 2, 32]), lim == 4], ['No disability', 'Multiple', 'Sensory', 'Physical'], 'Cognitive'))
        self._alloc('health_vision', [['Low vision', 'Blind'] if x & 1 else ['Normal', 'Corrected'] for x in lim])
        self._alloc('health_hearing', [['Moderate loss', 'Deaf / hard of hearing'] if x & 2 else ['Normal', 'Mild loss'] for x in lim])
        self._alloc('health_mobility', [['Moderate limitation', 'Uses mobility aid'] if x & 4 else ['Full', 'Mild limitation'] for x in lim])
        car = D.car.to_numpy(); blind = self.out['health_vision'] == 'Blind'
        self._alloc('demo_driver_status', [['Non-driver', 'Cannot drive'] if b else ['Daily driver'] if m in (3, 5) else (['Occasional driver', 'Licensed, rarely drives', 'Non-driver'] if c == 1 else ['Licensed, rarely drives', 'Non-driver']) for m, c, b in zip(mode, car, blind)])
        r1, r5, bp = D.r1.to_numpy(), D.r5.to_numpy(), D.bplace.to_numpy()
        self._set('lifex_geographic_mobility', np.select([(r1 == 5) | (r5 == 5), bp == 5, bp == 3, r5 == 6, r5 == 4], ['Moved internationally', 'Moved nationally', 'Moved within region', 'Moved nationally', 'Moved within region'], 'Never left hometown'))
        pet = D.pet.to_numpy()
        self._set('lstyle_pet_ownership', pd.Series(pet).map({0: 'None', 1: 'Dog', 2: 'Cat', 4: 'Other'}).fillna('Multiple pets').to_numpy())

        # relationship length (needs the allocated marital status)
        ms = self.out['demo_marital_status']; fm = D.fmage.to_numpy()
        yrs = np.full(n, np.nan)
        for i in np.flatnonzero(mar == 2):
            lo, hi = FMAGE.get(int(fm[i]), (25, 29)); hi = min(hi, age[i]); lo = min(lo, hi)
            yrs[i] = age[i] - rng.integers(lo, hi + 1) + rng.random()
        rl = [['Not in one'] if m in ('Single', 'Widowed', 'Divorced') else (['Under 1 year', '1-5 years', '5-15 years'] if m in ('In a relationship', 'Engaged') else
              [('Under 1 year' if y < 1 else '1-5 years' if y < 5 else '5-15 years' if y < 15 else '15+ years')]) for m, y in zip(ms, yrs)]
        self._alloc('demo_relationship_length', rl)
        # rule-derived attributes
        by = 2020 - age
        self._set('demo_generation', np.select([by <= 1945, by <= 1964, by <= 1980, by <= 1996], ['Silent', 'Boomer', 'Gen X', 'Millennial'], 'Gen Z'))
        dec = [('1960s-70s', 1960, 1979), ('1980s', 1980, 1989), ('1990s', 1990, 1999), ('2000s', 2000, 2009), ('2010s', 2010, 2019), ('2020s', 2020, 2029)]
        fd = []
        for b in by:
            yrs_ = np.arange(b + 10, min(b + 25, 2020) + 1); wts = np.array([((yrs_ >= a) & (yrs_ <= z)).sum() for _, a, z in dec], float)
            fd.append(dec[rng.choice(6, p=wts / wts.sum())][0] if wts.sum() > 0 else '1960s-70s')
        self._set('lifex_formative_decade', fd)
        for t, v in [('lifex_immigration_generation', 'Native multi-generational'), ('demo_ethnicity_broad', 'East Asian'), ('cultural_background', 'Collectivist (East Asian)'),
                     ('cult_south_korea', 'Native'), ('lang_korean', 'Native'), ('primary_language', 'Korean')]:
            self._set(t, np.full(n, v, dtype=object))
        # children: co-resident own children + non-co-resident ones (births - deaths - co-resident, all ages; stage-0 rule)
        own = D.own.tolist(); bi = D.births.to_numpy(); de = D.deaths.fillna(0).to_numpy()
        kids = []
        for i in range(n):
            c = list(own[i]) if isinstance(own[i], (list, np.ndarray)) else []
            nn = int(max(0, (bi[i] - de[i] if bi[i] == bi[i] else 0) - len(c)))
            for _ in range(nn):
                g_ = self.base.gap[self.base.gap <= age[i]]
                if len(g_): c.append(int(age[i] - rng.choice(g_)))
            if not c and D.grand.iat[i]:
                g_ = self.base.gap[self.base.gap <= age[i] - 15]
                c.append(int(age[i] - rng.choice(g_)) if len(g_) else int(age[i] - 25))
            kids.append(sorted(c))
        self.kids = kids; co = [len(o) if isinstance(o, (list, np.ndarray)) else 0 for o in own]
        grand = D.grand.to_numpy()
        mn = np.array([min(c) if c else np.nan for c in kids]); nk = np.array([len(c) for c in kids])
        self._set('demo_children_count', np.select([nk == 0, mn >= 20, nk == 1, nk == 2], ['None', 'Adult children', '1 child', '2 children'], '3+ children'))
        self._set('demo_parental_status', np.select([grand, nk == 0, mn < 1, mn < 18], ['Grandparent', 'Not a parent', 'New parent', 'Parent of minors'], 'Parent of adults'))
        cores = np.array(co) > 0
        self._set('lifex_parenting_journey', np.select([nk == 0, mn < 13, mn <= 19, cores], ['No children', 'Young children', 'Teenagers', 'Grown children'], 'Empty nester'))
        emp = self.out['demo_employment_status']; working = np.isin(self.codes['demo_employment_status'], [self.V['demo_employment_status'].index(v) for v in ['Full-time', 'Part-time', 'Self-employed', 'Gig / freelance']])
        jt = D.jten.to_numpy()
        ex = []
        for i in range(n):
            lb = JTEN_LB.get(int(jt[i]), 0) if jt[i] == jt[i] else 0; ub = age[i] - 15
            ex.append([b for b, a, z in EXP_BINS if a <= ub and z >= lb] or ['0-2'])
        self._alloc('years_experience', ex)
        exo = self.out['years_experience']
        SEN = {'0-2': ['Student / intern', 'Entry'], '3-5': ['Entry', 'Mid'], '6-10': ['Mid', 'Senior'], '11-20': ['Mid', 'Senior', 'Lead / Principal'], '20+': ['Senior', 'Lead / Principal']}
        sen = []
        for i in range(n):
            if not work[i] and enrolled[i]: sen.append(['Student / intern'])
            elif (age[i] >= 60 and not work[i]) or emp[i] == 'Retired': sen.append(['Retired'])
            elif work[i] and occ[i] == 1: sen.append(['Manager', 'Director', 'VP', 'C-suite'])
            elif work[i] and st[i] == 3: sen.append(['Founder', 'Manager', 'C-suite'])
            elif work[i] and st[i] == 2: sen.append(['Founder', 'Mid', 'Senior'])
            elif not work[i]: sen.append([])
            else: sen.append(SEN[exo[i]])
        self._alloc('seniority', sen, np.array([NA if not x else None for x in sen], dtype=object))
        ls = []
        for i in range(n):
            O = set()
            if enrolled[i]: O.add('Student')
            if working[i] and age[i] < 35: O.add('Early career')
            if nk[i] and mn[i] < 13: O.add('Parent of young kids')
            if 40 <= age[i] <= 64: O.add('Mid-life')
            if (working[i] or emp[i] == 'Unemployed') and 25 <= age[i] <= 59: O.add('Career change')
            if nk[i] and mn[i] >= 20 and not cores[i]: O.add('Empty nester')
            if (age[i] >= 60 and not working[i]) or emp[i] == 'Retired': O.add('Retirement')
            if not O: O = {'Student', 'Early career'} if age[i] < 40 else ({'Mid-life'} if age[i] < 60 else {'Retirement'})
            ls.append(sorted(O))
        self._alloc('life_stage', ls)
        # personality (temporary parameters)
        eta = rng.standard_normal((n, 5)) @ np.linalg.cholesky(TMP['Phi']).T
        zf = {}
        for j, d in enumerate(FACT):
            lam = TMP['lam_bfi2']; zz = []
            for f in BFI2[d]:
                z = lam * eta[:, j] + np.sqrt(1 - lam ** 2) * rng.standard_normal(n); zz.append(z); self._set(f'bfi2_facet_{f}', level(z))
            sd = np.sqrt((3 + (3 * lam) ** 2 - 3 * lam ** 2) / 9); zf[d] = np.mean(zz, axis=0) / sd
            self._set(BFI2_DOMAIN[d], level(zf[d]))
            lam = TMP['lam_ipip']
            for f in IPIP[d]: self._set(f'big5_{f}', level(lam * eta[:, j] + np.sqrt(1 - lam ** 2) * rng.standard_normal(n)))
        lm = TMP['lam_mbti']; let = lambda j: lm * eta[:, j] + np.sqrt(1 - lm ** 2) * rng.standard_normal(n) > 0
        E, N, F_, J = let(2), let(0), let(3), let(1)
        typ = np.char.add(np.char.add(np.where(E, 'E', 'I'), np.where(N, 'N', 'S')), np.char.add(np.where(F_, 'F', 'T'), np.where(J, 'J', 'P')))
        lab = {v.split(' ')[0]: v for v in self.V['mbti_type']}; self._set('mbti_type', np.array([lab[t] for t in typ], dtype=object))
        lb_ = TMP['lam_battery']; zb = lb_ * eta[:, 2] + np.sqrt(1 - lb_ ** 2) * rng.standard_normal(n)
        order = ['Strong introvert', 'Introvert', 'Ambivert', 'Extrovert', 'Strong extrovert']
        pri = np.array([self.S.prior['lstyle_social_battery'][self.V['lstyle_social_battery'].index(v)] for v in order])
        from scipy.stats import norm
        cuts = norm.ppf(np.cumsum(pri)[:-1]); self._set('lstyle_social_battery', np.array(order, dtype=object)[np.searchsorted(cuts, zb)])
        Z = np.stack([zf[d] for d in FACT], 1); srt = np.sort(Z, 1); top = Z.argmax(1)
        dom = np.where((srt[:, -1] >= 0.51) & (srt[:, -1] - srt[:, -2] >= self.delta), np.array([DOM[d] for d in FACT], dtype=object)[top], 'Balanced')
        self._set('dominant_trait', dom); self.Z = Z
        # stage-0 placeholders: (4) survey attributes and rules that need them, from MatrAIx q in MatrAIx order
        for t in self.ph_order:
            c = self.S.draw_korean(t, self.codes, self.avail); self.codes[t] = c; self.out[t] = np.array(self.V[t], dtype=object)[c]; self.avail.add(t)
        missing = [t for t in self.K if t not in self.out]
        assert not missing, missing
        # everything else from MatrAIx
        if self.on:
            self.S.sample_rest(self.codes)
            for t in self.S.rest:
                if t in self.emit: self.out[t] = pd.Categorical.from_codes(self.codes[t], categories=self.V[t])
        return self.out
