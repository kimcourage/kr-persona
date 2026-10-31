"""Persona generator: the base generator (generate/base.py) with the Korean survey models in place of the placeholders.
Order: base -> mapping/rules -> children (+ KLoSA) -> labor -> income -> personality -> values -> health -> KGSS -> MatrAIx rest."""
import json, pickle, numpy as np, pandas as pd
from scipy.stats import norm
from krpersona import paths
from krpersona.census import aggregates
from krpersona.generate.base import Gen0, Base, NA, REAL_P, FMAGE, JTEN_LB, EXP_BINS, IND, BFI2, BFI2_DOMAIN, IPIP, DOM, LEVELS, TAU, FACT, level
from krpersona.matraix.sampler import KRSampler
from krpersona.matraix.graph import graph_values
from krpersona.generate.assign import build, SCHWARTZ
from krpersona.generate.masks import NOT_DEV, NA_DEV, nondev_plan, nondev_masks, rule_pair_masks, conflict_masks, dyn_language_masks
from krpersona.generate.repair import RepairRules
from krpersona.blocks import core, pers, values as VM, kgss as KG, family as FM
# BFI-2 facet loadings: Korean BFI-2 facet-level ESEM, Korean employee sample T1 (n = 856), primary loadings (Choi et al., 2025, Assessment, Table 3)
LAM_BFI2 = {'sociability': .86, 'assertiveness': .55, 'energy_level': .72, 'compassion': .72, 'respectfulness': .65, 'trust': .54,
            'organization': .79, 'productiveness': .76, 'responsibility': .65, 'anxiety': .90, 'depression': .81, 'emotional_volatility': .80,
            'intellectual_curiosity': .86, 'aesthetic_sensitivity': .58, 'creative_imagination': .64}
# IPIP-NEO facets: Korean IPIP-NEO-300, adult sample (N = 334), 5-factor EFA of facet scores, loading on the facet's own domain (장승민, 2017 연구보고서, 표 7)
LAM_IPIP = {'anxiety': .86, 'anger': .66, 'depression': .62, 'self_consciousness': .58, 'immoderation': .33, 'vulnerability': .71,
            'friendliness': .83, 'gregariousness': .91, 'assertiveness': .58, 'activity_level': .35, 'excitement_seeking': .48, 'cheerfulness': .53,
            'imagination': .72, 'artistic_interest': .63, 'emotionality': .63, 'adventurousness': .52, 'intellect': .66, 'liberalism': .34,
            'trust': .41, 'morality': .69, 'altruism': .60, 'cooperation': .74, 'modesty': .32, 'sympathy': .60,
            'self_efficacy': .55, 'orderliness': .64, 'dutifulness': .47, 'achievement_striving': .70, 'self_discipline': .73, 'cautiousness': .55}
LAM_MBTI, LAM_BAT = 0.7, 0.7     # no Korean source: assumption
# IPIP-NEO facet -> BFI-2 facet measuring the same facet. Without a shared facet part the two correlate only through the domain
# (e.g. assertiveness .55 x .58 = .32); the facet-specific parts are now shared so that the two measures correlate SAME_FACET_R (assumption)
SAME_FACET = {'gregariousness': 'sociability', 'assertiveness': 'assertiveness', 'activity_level': 'energy_level', 'sympathy': 'compassion', 'cooperation': 'respectfulness',
              'trust': 'trust', 'orderliness': 'organization', 'achievement_striving': 'productiveness', 'dutifulness': 'responsibility', 'anxiety': 'anxiety',
              'depression': 'depression', 'anger': 'emotional_volatility', 'intellect': 'intellectual_curiosity', 'artistic_interest': 'aesthetic_sensitivity', 'imagination': 'creative_imagination'}
SAME_FACET_R = 0.6
HT6 = {91: 0, 11: 1, 21: 2, 26: 2, 22: 3, 23: 3, 31: 4, 32: 4, 33: 4, 39: 4, 49: 4, 59: 4}     # census 세대유형 -> KNHANES household type (else 기타)
DEP = [(0, 'Minimal'), (5, 'Mild'), (10, 'Moderate'), (15, 'Moderately severe'), (20, 'Severe')]
ANX = [(0, 'Minimal'), (5, 'Mild'), (10, 'Moderate'), (15, 'Severe')]
SES = ['Low income', 'Lower-middle', 'Middle', 'Upper-middle', 'High income']
USD = ['<$25k', '$25k-50k', '$50k-100k', '$100k-200k', '$200k+']
METRO_SIDO = [11, 23, 31]
IND12 = {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7, 'K': 8, 'L': 9, 'M': 9, 'N': 9, 'O': 10, 'P': 10, 'Q': 10, 'R': 11, 'S': 11, 'T': 11, 'U': 10}
SIZE_OUT = ['1-4 employees', '5-9 employees', '10-29 employees', '30-99 employees', '100-299 employees', '300+ employees']
SIZE_SETS = [['Startup (<50)'], ['Startup (<50)'], ['Startup (<50)'], ['Startup (<50)', 'SMB (50-500)'], ['SMB (50-500)'], ['SMB (50-500)', 'Mid (500-5k)', 'Enterprise (5k+)']]
KR_ADDED = ['english_proficiency', 'lang_english', 'dev_professional_status'] + ['trait_curiosity', 'cog_curiosity', 'trait_creativity', 'trait_appreciation_of_beauty', 'trait_open_mindedness', 'cog_open_mindedness', 'cog_ambiguity_tolerance', 'need_for_closure', 'att_rapid_change', 'pref_novelty_vs_familiarity', 'pref_routine_vs_variety', 'pref_stability_vs_change', 'trait_discipline', 'trait_perseverance', 'trait_prudence', 'trait_self_regulation', 'trait_ambition', 'cog_procrastination', 'lstyle_tidiness', 'pref_plan_vs_spontaneous', 'cog_decision_speed', 'trait_zest', 'trait_leadership', 'skill_leadership', 'interpersonal_agency_dominance', 'cog_assertiveness', 'pref_big_group_vs_one_on_one', 'trait_competitiveness', 'pref_competition_vs_collab', 'trait_kindness', 'trait_humility', 'trait_honesty', 'interpersonal_communion_warmth', 'cog_politeness', 'trait_resilience', 'cog_optimism', 'pref_logic_vs_intuition']   # the last 37 = DERIVED attributes; English ability (KGSS 2018), developer status (EAPS B형 occupation)
# MatrAIx personality-like attributes made from the Korean personality scores instead of the MatrAIx graph.
# (attribute, source score, sign, r, anchors). z = sign * r * source + sqrt(1 - r^2) * noise, cut at the MatrAIx prior proportions (first value = top).
# anchors (attribute, sign): the value may not sit at the opposite extreme of an anchor (sign +1: first value goes with the anchor's first value).
# r is an assumption (0.8 near-identical construct, 0.6-0.7 related construct); 'MBTI_J'/'MBTI_F' anchors are the MBTI letters.
DERIVED = [
    ('trait_curiosity', 'bfi2_facet_intellectual_curiosity', 1, .8, [('bfi2_facet_intellectual_curiosity', 1), ('big5_intellect', 1)]),
    ('cog_curiosity', 'bfi2_facet_intellectual_curiosity', 1, .8, [('bfi2_facet_intellectual_curiosity', 1), ('big5_intellect', 1), ('trait_curiosity', 1)]),
    ('trait_creativity', 'bfi2_facet_creative_imagination', 1, .8, [('bfi2_facet_creative_imagination', 1), ('big5_imagination', 1)]),
    ('trait_appreciation_of_beauty', 'bfi2_facet_aesthetic_sensitivity', 1, .8, [('bfi2_facet_aesthetic_sensitivity', 1), ('big5_artistic_interest', 1)]),
    ('trait_open_mindedness', 'bfi2_domain_open_mindedness', 1, .7, [('bfi2_domain_open_mindedness', 1)]),
    ('cog_open_mindedness', 'bfi2_domain_open_mindedness', 1, .7, [('bfi2_domain_open_mindedness', 1), ('trait_open_mindedness', 1)]),
    ('cog_ambiguity_tolerance', 'bfi2_domain_open_mindedness', 1, .6, [('bfi2_domain_open_mindedness', 1)]),
    ('need_for_closure', 'bfi2_domain_open_mindedness', -1, .6, [('bfi2_domain_open_mindedness', -1), ('cog_ambiguity_tolerance', -1)]),
    ('att_rapid_change', 'big5_adventurousness', 1, .6, [('big5_adventurousness', 1)]),
    ('pref_novelty_vs_familiarity', 'big5_adventurousness', 1, .8, [('big5_adventurousness', 1)]),
    ('pref_routine_vs_variety', 'big5_adventurousness', -1, .8, [('big5_adventurousness', -1), ('pref_novelty_vs_familiarity', -1)]),
    ('pref_stability_vs_change', 'big5_adventurousness', -1, .7, [('big5_adventurousness', -1), ('att_rapid_change', -1), ('pref_routine_vs_variety', 1)]),
    ('trait_discipline', 'big5_self_discipline', 1, .8, [('big5_self_discipline', 1)]),
    ('trait_perseverance', 'bfi2_domain_conscientiousness', 1, .7, [('bfi2_domain_conscientiousness', 1), ('big5_self_discipline', 1)]),
    ('trait_prudence', 'big5_cautiousness', 1, .8, [('big5_cautiousness', 1)]),
    ('trait_self_regulation', 'big5_immoderation', -1, .7, [('big5_immoderation', -1), ('big5_self_discipline', 1)]),
    ('trait_ambition', 'big5_achievement_striving', 1, .8, [('big5_achievement_striving', 1)]),
    ('cog_procrastination', 'bfi2_facet_productiveness', -1, .7, [('bfi2_facet_productiveness', -1), ('big5_self_discipline', -1), ('trait_discipline', -1)]),
    ('lstyle_tidiness', 'bfi2_facet_organization', 1, .8, [('bfi2_facet_organization', 1), ('big5_orderliness', 1)]),
    ('pref_plan_vs_spontaneous', 'mbti_J', 1, .8, [('MBTI_J', 1), ('bfi2_domain_conscientiousness', 1)]),
    ('cog_decision_speed', 'big5_cautiousness', -1, .7, [('big5_cautiousness', -1)]),
    ('trait_zest', 'bfi2_facet_energy_level', 1, .8, [('bfi2_facet_energy_level', 1), ('big5_activity_level', 1)]),
    ('trait_leadership', 'bfi2_facet_assertiveness', 1, .7, [('bfi2_facet_assertiveness', 1), ('big5_assertiveness', 1)]),
    ('skill_leadership', 'bfi2_facet_assertiveness', 1, .6, [('trait_leadership', 1)]),
    ('interpersonal_agency_dominance', 'bfi2_facet_assertiveness', 1, .8, [('bfi2_facet_assertiveness', 1), ('big5_assertiveness', 1)]),
    ('cog_assertiveness', 'bfi2_facet_assertiveness', 1, .8, [('bfi2_facet_assertiveness', 1), ('big5_assertiveness', 1), ('interpersonal_agency_dominance', 1)]),
    ('pref_big_group_vs_one_on_one', 'big5_gregariousness', 1, .8, [('big5_gregariousness', 1), ('bfi2_facet_sociability', 1)]),
    ('trait_competitiveness', 'big5_cooperation', -1, .7, [('big5_cooperation', -1)]),
    ('pref_competition_vs_collab', 'big5_cooperation', -1, .7, [('big5_cooperation', -1), ('trait_competitiveness', 1)]),
    ('trait_kindness', 'big5_altruism', 1, .8, [('big5_altruism', 1), ('bfi2_facet_compassion', 1)]),
    ('trait_humility', 'big5_modesty', 1, .8, [('big5_modesty', 1)]),
    ('trait_honesty', 'big5_morality', 1, .8, [('big5_morality', 1)]),
    ('interpersonal_communion_warmth', 'big5_friendliness', 1, .8, [('big5_friendliness', 1)]),
    ('cog_politeness', 'bfi2_facet_respectfulness', 1, .8, [('bfi2_facet_respectfulness', 1)]),
    ('trait_resilience', 'big5_vulnerability', -1, .8, [('big5_vulnerability', -1)]),
    ('cog_optimism', 'bfi2_facet_depression', -1, .6, [('bfi2_facet_depression', -1)]),
    ('pref_logic_vs_intuition', 'mbti_F', -1, .7, [('MBTI_F', -1)]),
]

EN_LANG = {'Native': 'Native', 'Fluent (C1-C2)': 'Fluent', 'Intermediate (B1-B2)': 'Conversational', 'Basic (A1-A2)': 'Basic', 'None': 'None'}


class Base4(Base):
    """adds the household head's characteristics, employed members, census social activity"""
    def __init__(self, sh, sp, rng):
        super().__init__(sh, sp, rng)
        if core.HOLDOUT:   # mother-child gaps from training households only (same split as the CART synthesis: seed 5, 80%)
            Hr = pd.read_parquet(paths.art('census', 'H.parquet')); tr = set(Hr.hid.sample(frac=0.8, random_state=5))
            rp = pd.read_parquet(REAL_P()); self.gap = aggregates.expand(aggregates.freq_table(aggregates.gaps_from(rp[rp.hid.isin(tr)])))
        P = pd.read_parquet(sp)
        hd = P[P.rel == 1].drop_duplicates('hid').set_index('hid')
        for c in ['sex', 'age', 'mar', 'edu', 'sch', 'econ', 'status', 'occ']:
            self.P['h_' + c] = self.P.hid.map(hd[c]).fillna(self.P[c]).to_numpy()
        self.P['nemp'] = self.P.hid.map(P[P.econ.isin([1, 2, 3])].groupby('hid').size()).fillna(0).to_numpy()
        s = P.set_index(['hid', 'pos'])[['soc_pol', 'soc_soc']]; idx = pd.MultiIndex.from_arrays([self.P.hid, self.P.pos])
        self.P['soc_pol'] = s.soc_pol.reindex(idx).to_numpy(); self.P['soc_soc'] = s.soc_soc.reindex(idx).to_numpy()


class Gen4(Gen0):
    def __init__(self, sh, sp, seed=0, switch_on=True):
        self.rng = np.random.default_rng(seed)
        self.rows, K0, PH, self.ids, self.emit = build()
        self.K = set(K0) | set(SCHWARTZ) | set(VM.VAL_STAR) | set(KR_ADDED)
        self.PH = []; self.PH0 = PH; self.on = switch_on
        self.base = Base4(sh, sp, self.rng)
        self.S = KRSampler(self.K, placeholder=set(), seed=seed, extra_masks=self._extra_masks(), dyn_masks=self._dyn_masks())
        self.V = self.S.values; self.ph_order = []; self.delta = 0.30
        X = core.SUF   # '' = models on all data, '_80' = validation models on 80% of each source
        M = lambda f: paths.inp('models', f)
        self.PP = json.load(open(M(f'pers_params{X}.json'))); self.VP = json.load(open(M(f'value_params{X}.json')))
        self.HM = pickle.load(open(M(f'health_models{X}.pkl'), 'rb')); self.LI = pickle.load(open(M(f'labor_income_models{X}.pkl'), 'rb'))
        self.KG = pickle.load(open(M(f'kgss_models{X}.pkl'), 'rb')); self.FM = pickle.load(open(M(f'family_models{X}.pkl'), 'rb'))
        self.part, self.part_all = KG.census_participation()
        self.KR_NEW = [a for a in self.KG['order'] if a.startswith('kr_')] + ['kr_depression_level', 'kr_anxiety_level']
        self.DM = pickle.load(open(M(f'dev_models{X}.pkl'), 'rb'))
        self.NONDEV_VAL, self.NONDEV_NA = nondev_plan(self.V)
        self.repair_on, self.repair_rounds = True, 40
        self.REP = RepairRules(self.V, self.K, sorted(set(self.V) | self.K | set(self.KR_NEW)))

    def _extra_masks(self):
        return super()._extra_masks() + conflict_masks(graph_values(), self.K)

    def _dyn_masks(self):
        V = graph_values(); return dyn_language_masks(V) + nondev_masks(V) + rule_pair_masks(self.K)

    # ------------------------------------------------------------ developer status, re-draw of contradicting MatrAIx parts
    def _dev(self, work, occ, letter, bgc, s01, rng):
        """1-digit occupation 2 workers: EAPS B형 model -> Professional developer / Non-developer technologist / other; everyone else 'Not a developer'"""
        from krpersona.blocks import dev as DEVM
        n = len(work); cls = np.zeros(n, int); m = work & (occ == 2)
        if m.any():
            ind12 = np.array([IND12.get(l, 0) for l in letter])
            cls[m] = DEVM.draw(self.DM, np.stack([s01, bgc['ag7'], bgc['edu5'], ind12], 1)[m].astype(int), rng)
        sets = [['Professional developer'] if c == 1 else ['Non-developer technologist'] if c == 2 else [] for c in cls]
        self._alloc('dev_professional_status', sets, np.where(cls == 0, NOT_DEV, None).astype(object))
        self.nondev = cls == 0

    def _rest_labels(self, t, rows):
        lab = np.array(list(self.V[t]) + [None], dtype=object)[self.codes[t][rows]]
        if t in self.NONDEV_NA: lab = np.where(self.nondev[rows], NA_DEV, lab)
        return lab

    def _frame(self, rows):
        cols = {t: np.asarray(v, dtype=object)[rows] for t, v in self.out.items()}
        cols.update({t: self._rest_labels(t, rows) for t in self.S.rest if t in self.emit})
        return pd.DataFrame(cols)

    def _repair(self):
        """rows that break a hard rule of the fixing rule sets get the MatrAIx part re-drawn (Korean values fixed), up to repair_rounds times"""
        n = len(self.D); rows = np.arange(n); hist = []
        for r in range(self.repair_rounds + 1):
            bad = self.REP.bad(self._frame(rows)); hist.append(int(bad.sum()))
            rows = rows[bad]
            if r == self.repair_rounds or not len(rows): break
            sub = {k: v[rows] for k, v in self.codes.items() if k in self.K}
            self.S.sample_rest(sub); self._multi(sub)
            for t in self.S.rest: self.codes[t][rows] = sub[t]
        self.repair_hist = hist; self.repair_left = rows

    def _multi(self, codes):
        """multilingualism (a MatrAIx root with no parents) is set from the languages drawn: number of languages at Conversational or better
        (Korean, English from english_proficiency, other lang_*) 1 -> Monolingual, 2 -> Bilingual, 3+ -> Trilingual+"""
        V = self.V
        if not hasattr(self, '_ml'):
            eng = np.array([v in ('Native', 'Fluent (C1-C2)', 'Intermediate (B1-B2)') for v in V['english_proficiency']] + [False])   # last: no-category code
            fo = {t: np.array([v in ('Conversational', 'Fluent', 'Native') for v in V[t]] + [False]) for t in V if t.startswith('lang_') and t not in ('lang_korean', 'lang_english')}
            self._ml = (eng, fo, [V['multilingualism'].index(v) for v in ('Monolingual', 'Bilingual', 'Trilingual+')])
        eng, fo, idx = self._ml
        n = 1 + eng[codes['english_proficiency']].astype(int)
        for t, m in fo.items(): n = n + m[codes[t]]
        codes['multilingualism'] = np.array(idx)[np.minimum(n, 3) - 1]

    def _put(self, t, sets, out=None):
        self._alloc(t, sets, None if out is None else np.asarray(out, dtype=object))

    # ------------------------------------------------------------ main
    def make(self, n):
        D = self.base.draw(n); self.D = D; rng = self.rng
        self.out, self.codes, self.avail = {}, {}, set()
        age = D.age.to_numpy().astype(int); sex = D.sex.to_numpy().astype(int); s01 = (sex == 2).astype(int)
        # ---------------- census values mapped to MatrAIx attributes
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
        letter = np.array([L[int(x)] if w and x == x else '' for x, w in zip(ind, work)], dtype=object)
        DMAP = {'A': ['Agriculture'], 'B': ['Manufacturing'], 'C': ['Manufacturing'], 'D': ['Engineering'], 'E': ['Engineering'], 'F': ['Skilled Trades'],
                'G': ['Business & Management'], 'L': ['Business & Management'], 'N': ['Business & Management'], 'H': ['Business & Management'],
                'I': ['Hospitality'], 'T': ['Hospitality'], 'J': ['Software & AI', 'Media & Journalism'], 'K': ['Finance & Economics'],
                'M': ['Law & Policy', 'Engineering', 'Natural Sciences', 'Social Sciences', 'Business & Management'], 'O': ['Public Sector'], 'U': ['Public Sector'],
                'P': ['Education'], 'Q': ['Healthcare & Medicine'], 'R': ['Arts & Humanities'], 'S': ['Skilled Trades', 'Hospitality']}
        self._alloc('domain', [DMAP[l] if l else [] for l in letter], np.where(work, None, NA).astype(object))
        RCUST = {3: ('Office / clerical', ['Operations', 'Finance', 'HR']), 4: ('Service', ['Operations']), 5: ('Retail sales', ['Sales / GTM']),
                 6: ('Agriculture / fishery', ['Operations']), 7: ('Skilled trades / craft', ['Operations']), 8: ('Machine operation / assembly', ['Operations']),
                 9: ('Elementary labor', ['Operations']), 10: ('Other occupation', ['Operations'])}
        rsets, rout = [], []
        for w_, o_, il in zip(work, occ, letter):
            if not w_: rsets.append([]); rout.append(NA); continue
            o_ = int(o_)
            if o_ == 1: rsets.append(['Executive']); rout.append(None)
            elif o_ == 2:
                rsets.append({'P': ['Teaching'], 'Q': ['Clinical'], 'J': ['Engineering'], 'K': ['Finance'], 'M': ['Research', 'Legal', 'Design', 'Engineering']}.get(il, ['Research', 'Engineering', 'Design'])); rout.append(None)
            else: rsets.append(RCUST[o_][1]); rout.append(RCUST[o_][0])
        self._alloc('role_function', rsets, np.array(rout, dtype=object))
        rel = D.rel.to_numpy(); ten = D.tenure_h.to_numpy(); k = D.k.to_numpy(); hk = D.hkind.to_numpy()
        self._set('lstyle_household_size', np.select([np.isin(hk, [4, 5]), k == 1, k == 2, k <= 4], ['Communal', 'Lives alone', '2 people', '3-4 people'], '5+ people'))
        comm, mode, place = D.comm.to_numpy(), D['mode'].to_numpy(), D.place.to_numpy()
        cm = np.where(np.isin(comm, [1, 2]), pd.Series(mode).map({1: 'Walk', 2: 'Bike', 3: 'Car', 4: 'Public transit', 5: 'Public transit', 6: 'Rideshare', 7: 'Other'}).to_numpy(),
                      np.where(work & (place == 2), 'Remote', NA)).astype(object)
        self._set('lstyle_commute_mode', cm)
        lim = D.lim.fillna(0).astype(int).to_numpy(); nb = np.array([bin(x).count('1') for x in lim])
        # census activity limitation bits: 1 vision, 2 hearing, 4 physical, 8 mental, 16 intellectual/autism, 32 speech
        # (speech alone -> Physical, as in the Korean classification 외부 신체기능 장애; Sensory would contradict normal vision and hearing)
        self._set('demo_disability_status', np.select([lim == 0, nb >= 2, np.isin(lim, [1, 2]), np.isin(lim, [4, 32])], ['No disability', 'Multiple', 'Sensory', 'Physical'], 'Cognitive'))
        self._alloc('health_vision', [['Low vision', 'Blind'] if x & 1 else ['Normal', 'Corrected'] for x in lim])
        self._alloc('health_hearing', [['Moderate loss', 'Deaf / hard of hearing'] if x & 2 else ['Normal', 'Mild loss'] for x in lim])
        self._alloc('health_mobility', [['Moderate limitation', 'Uses mobility aid'] if x & 4 else ['Full', 'Mild limitation'] for x in lim])
        car = D.car.to_numpy(); blind = self.out['health_vision'] == 'Blind'
        self._alloc('demo_driver_status', [['Non-driver', 'Cannot drive'] if b else ['Daily driver'] if m in (3, 5) else (['Occasional driver', 'Licensed, rarely drives', 'Non-driver'] if c == 1 else ['Licensed, rarely drives', 'Non-driver']) for m, c, b in zip(mode, car, blind)])
        r1, r5, bp = D.r1.to_numpy(), D.r5.to_numpy(), D.bplace.to_numpy()
        self._set('lifex_geographic_mobility', np.select([(r1 == 5) | (r5 == 5), bp == 5, bp == 3, r5 == 6, r5 == 4], ['Moved internationally', 'Moved nationally', 'Moved within region', 'Moved nationally', 'Moved within region'], 'Never left hometown'))
        self._set('lstyle_pet_ownership', pd.Series(D.pet.to_numpy()).map({0: 'None', 1: 'Dog', 2: 'Cat', 4: 'Other'}).fillna('Multiple pets').to_numpy())
        ms = self.out['demo_marital_status']; fm = D.fmage.to_numpy(); yrs = np.full(n, np.nan)
        for i in np.flatnonzero(mar == 2):
            lo, hi = FMAGE.get(int(fm[i]), (25, 29)); hi = min(hi, age[i]); lo = min(lo, hi)
            yrs[i] = age[i] - rng.integers(lo, hi + 1) + rng.random()
        rl = [['Not in one'] if m in ('Single', 'Widowed', 'Divorced') else (['Under 1 year', '1-5 years', '5-15 years'] if m in ('In a relationship', 'Engaged') else
              [('Under 1 year' if y < 1 else '1-5 years' if y < 5 else '5-15 years' if y < 15 else '15+ years')]) for m, y in zip(ms, yrs)]
        self._alloc('demo_relationship_length', rl)
        # ---------------- rule-derived attributes
        by = 2020 - age
        self._set('demo_generation', np.select([by <= 1945, by <= 1964, by <= 1980, by <= 1996], ['Silent', 'Boomer', 'Gen X', 'Millennial'], 'Gen Z'))
        dec = [('1960s-70s', 1960, 1979), ('1980s', 1980, 1989), ('1990s', 1990, 1999), ('2000s', 2000, 2009), ('2010s', 2010, 2019), ('2020s', 2020, 2029)]
        fd = []
        for b in by:
            yr_ = np.arange(b + 10, min(b + 25, 2020) + 1); wts = np.array([((yr_ >= a) & (yr_ <= z)).sum() for _, a, z in dec], float)
            fd.append(dec[rng.choice(6, p=wts / wts.sum())][0] if wts.sum() > 0 else '1960s-70s')
        self._set('lifex_formative_decade', fd)
        for t, v in [('lifex_immigration_generation', 'Native multi-generational'), ('demo_ethnicity_broad', 'East Asian'), ('cultural_background', 'Collectivist (East Asian)'),
                     ('cult_south_korea', 'Native'), ('lang_korean', 'Native'), ('primary_language', 'Korean')]:
            self._set(t, np.full(n, v, dtype=object))
        # ---------------- background codes shared by the survey blocks
        bgc = dict(sex=s01, ag7=core.ag7(age), edu5=core.edu5_census(e, s), edu4=core.edu4_census(e, s), mar4=core.mar4(mar), emp5=core.emp5_census(econ, st),
                   emp2=(~work).astype(int), urban2=(D.dong.to_numpy() != 3).astype(int), reg7=D.sido.map(core.SIDO_REG7).fillna(1).astype(int).to_numpy())
        self.bgc = bgc
        # ---------------- family and children
        self._children(D, age, sex, mar, bgc, rng)
        nk = np.array([len(c) for c in self.kids]); mn = np.array([min(c) if c else np.nan for c in self.kids])
        co = np.array([len(o) if isinstance(o, (list, np.ndarray)) else 0 for o in D.own.tolist()]); grand = D.grand.to_numpy()
        self._set('demo_children_count', np.select([nk == 0, mn >= 20, nk == 1, nk == 2], ['None', 'Adult children', '1 child', '2 children'], '3+ children'))
        self._set('demo_parental_status', np.select([grand, nk == 0, mn < 1, mn < 18], ['Grandparent', 'Not a parent', 'New parent', 'Parent of minors'], 'Parent of adults'))
        cores = co > 0
        self._set('lifex_parenting_journey', np.select([nk == 0, mn < 13, mn <= 19, cores], ['No children', 'Young children', 'Teenagers', 'Grown children'], 'Empty nester'))
        emp = self.out['demo_employment_status']; working = np.isin(self.codes['demo_employment_status'], [self.V['demo_employment_status'].index(v) for v in ['Full-time', 'Part-time', 'Self-employed', 'Gig / freelance']])
        jt = D.jten.to_numpy(); ex = []
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
            # fallback (no 'Student' when not enrolled, no 'Early career' with 20+ years of experience)
            if not O: O = ({'Early career'} if exo[i] != '20+' else {'Mid-life'}) if age[i] < 40 else ({'Mid-life'} if age[i] < 60 else {'Retirement'})
            ls.append(sorted(O))
        self._alloc('life_stage', ls)
        # ---------------- labor, income, personality, values, health, KGSS
        wage_lb = self._labor(D, work, st, occ, letter, bgc, rng)
        self._dev(work, occ, letter, bgc, s01, rng)
        self._income(D, rel, ten, k, wage_lb, rng)
        eta, mu = self._personality(sex, age, rng)
        self._values(sex, age, e, s, eta, mu, rng)
        self._health(D, age, bgc, eta, mu, rng)
        self._kgss(D, age, bgc, rng)
        missing = [t for t in self.K if t not in self.out]
        assert not missing, missing
        if self.on:
            self.S.sample_rest(self.codes); self._multi(self.codes)
            self.repair_hist, self.repair_left = [], np.array([], int)
            if self.repair_on: self._repair()
            for t in self.S.rest:
                if not t in self.emit: continue
                if t in self.NONDEV_NA:   # developer-only attribute: 'Not applicable' for non-developers
                    lab = np.array(list(self.V[t]), dtype=object)[self.codes[t]]
                    self.out[t] = pd.Categorical(np.where(self.nondev, NA_DEV, lab), categories=list(self.V[t]) + [NA_DEV])
                else: self.out[t] = pd.Categorical.from_codes(self.codes[t], categories=self.V[t])
        return self.out

    # ------------------------------------------------------------ blocks
    def _children(self, D, age, sex, mar, bgc, rng):
        """co-resident own children + non-co-resident: census births - deaths - co-resident where a fertility record exists;
        KLoSA living-children model for 61+ ever-married persons without one; gaps of 61+ from KLoSA donors (same sex, age band), else census mother-child gaps"""
        n = len(D); own = D.own.tolist(); bi = D.births.to_numpy(); de = D.deaths.fillna(0).to_numpy()
        co = np.array([len(o) if isinstance(o, (list, np.ndarray)) else 0 for o in own])
        men = (age >= 61) & ~(bi == bi) & (mar != 1)          # 61+ ever married without a census fertility record (both sexes)
        tot = np.zeros(n, int)
        if men.any():
            tot[men] = FM.draw_total(self.FM, (sex[men] == 1).astype(int), age[men], bgc['mar4'][men], bgc['edu4'][men], bgc['urban2'][men], co[men], rng)
        kids = []; self.kid_src = np.zeros(n, int)
        for i in range(n):
            c = list(own[i]) if isinstance(own[i], (list, np.ndarray)) else []
            if bi[i] == bi[i]: nn = int(max(0, bi[i] - de[i] - len(c)))
            elif men[i]: nn = int(max(0, tot[i] - len(c))); self.kid_src[i] = 2
            else: nn = 0
            if nn:
                g = FM.draw_gaps(self.FM, sex[i] == 1, age[i], nn, rng) if age[i] >= 61 else None
                if g is not None: c += [int(age[i] - x) for x in g]; self.kid_src[i] = max(self.kid_src[i], 1)
                else:
                    for _ in range(nn):
                        g_ = self.base.gap[self.base.gap <= age[i]]
                        if len(g_): c.append(int(age[i] - rng.choice(g_)))
            if not c and D.grand.iat[i]:
                g_ = self.base.gap[self.base.gap <= age[i] - 15]
                c.append(int(age[i] - rng.choice(g_)) if len(g_) else int(age[i] - 25))
            kids.append(sorted(c))
        self.kids = kids

    def _labor(self, D, work, st, occ, letter, bgc, rng):
        """company size (EAPS A형 model with 시도; own-account -> Solo / freelance, public administration -> Public sector) and own wage band (internal)"""
        Lm = self.LI['labor']; n = len(D)
        occ10 = np.where(work & (occ == occ), np.where(occ == 10, 0, occ), 0).astype(int)
        ind12 = np.array([IND12.get(l, 0) for l in letter])
        emp_l = np.select([st == 1, st == 3, st == 4], [0, 1, 2], 0)
        s17 = D.sido.map({c: i for i, c in enumerate(Lm['sido17'])}).fillna(0).astype(int).to_numpy()
        X = np.stack([bgc['sex'], bgc['ag7'], bgc['edu5'], bgc['mar4'], emp_l, occ10, ind12, s17], 1).astype(int)
        size = core.sample(Lm['size'], X, rng)
        sets, out = [], []
        for i in range(n):
            if not work[i]: sets.append([]); out.append(NA)
            elif st[i] == 2: sets.append(['Solo / freelance']); out.append(None)
            elif letter[i] == 'O': sets.append(['Public sector']); out.append(None)
            else: sets.append(SIZE_SETS[size[i]]); out.append(SIZE_OUT[size[i]])
        self._put('company_size', sets, out)
        wb = core.sample(Lm['wage'], np.column_stack([X, size]), rng)
        lower = np.array(Lm['wage']['lower'])[wb]
        self.wage_band = np.where(work & (st == 1), wb, -1)
        return np.where(work & (st == 1), lower, 0)

    def _income(self, D, rel, ten, k, wage_lb, rng):
        """household income (HFCS model on the head's characteristics; at least 12 x own monthly wage lower bound), USD band, SES quintile rule, mortgage"""
        I = self.LI['income']; n = len(D)
        h_age = D.h_age.to_numpy().astype(int); h_e, h_s = D.h_edu.to_numpy(), D.h_sch.to_numpy()
        h_work = np.isin(D.h_econ.to_numpy(), [1, 2, 3]); h_occ = D.h_occ.to_numpy()
        hh5 = np.minimum(k, 5) - 1; nemp4 = np.minimum(D.nemp.to_numpy(), 3).astype(int); metro = D.sido.isin(METRO_SIDO).to_numpy().astype(int)
        h_emp5 = core.emp5_census(D.h_econ.to_numpy(), D.h_status.to_numpy())
        X = np.stack([(D.h_sex.to_numpy() == 2).astype(int), core.ag7(h_age), core.edu5_census(h_e, h_s), core.mar4(D.h_mar.to_numpy()), h_emp5,
                      np.where(h_work & (h_occ == h_occ), np.where(h_occ == 10, 9, h_occ), 0).astype(int), hh5, nemp4, metro], 1).astype(int)
        nb = len(I['inc_cuts']) + 1; lb_bin = np.digitize(12 * wage_lb, I['inc_cuts'])
        al = (np.arange(nb)[None, :] >= lb_bin[:, None]).astype(float)
        ib = core.sample(I['income'], X, rng, allowed=al)
        amt = np.zeros(n)
        for b in np.unique(ib):
            m = ib == b; v, w = I['donors'].get(int(b), (np.array([I['inc_cuts'][min(b, nb - 2)]]), np.array([1.0])))
            amt[m] = rng.choice(v, m.sum(), p=w / w.sum())
        amt = np.maximum(amt, 12 * wage_lb)
        self.income = amt
        self._set('demo_household_income', np.array(USD, dtype=object)[np.digitize(amt, I['usd_edges'])])
        eq = amt / np.sqrt(np.maximum(k, 1))
        self._set('socioeconomic_band', np.array(SES, dtype=object)[np.digitize(eq, I['ses_cuts'])])
        self.inc4 = np.digitize(eq, I['inc4_cuts'])
        Xm = np.stack([core.ag7(h_age), h_emp5, hh5, metro, np.digitize(amt, I['mort_cuts'])], 1).astype(int)
        mort = core.sample(I['mortgage'], Xm, rng)
        hs = np.where(rel == 3, 'Living with family', np.where(ten == 1, np.where(mort == 1, 'Mortgage', 'Own outright'), 'Renting')).astype(object)
        self._set('demo_housing_status', hs)

    def _personality(self, sex, age, rng):
        n = len(sex); B = np.array(self.PP['B']); Phi = np.array(self.PP['Phi'])
        mu = pers.phi(sex, age) @ B; eta = mu + rng.standard_normal((n, 5)) @ np.linalg.cholesky(Phi).T
        zf = {}; self.fz = {}
        for j, d in enumerate(FACT):
            zz = []; lam = np.array([LAM_BFI2[f] for f in BFI2[d]]); uniq = {}
            for f, l in zip(BFI2[d], lam):
                uniq[f] = rng.standard_normal(n); z = l * eta[:, j] + np.sqrt(1 - l ** 2) * uniq[f]; zz.append(z); self._set(f'bfi2_facet_{f}', level(z)); self.fz[f'bfi2_facet_{f}'] = z
            sd = np.sqrt(lam.sum() ** 2 + (1 - lam ** 2).sum()) / 3; zf[d] = np.mean(zz, axis=0) / sd
            self._set(BFI2_DOMAIN[d], level(zf[d])); self.fz[BFI2_DOMAIN[d]] = zf[d]
            for f in IPIP[d]:
                l = LAM_IPIP[f]; u = rng.standard_normal(n); b = SAME_FACET.get(f)
                if b in uniq:   # same facet in both instruments: the facet-specific parts are shared so that corr = SAME_FACET_R
                    lb = LAM_BFI2[b]; rho = float(np.clip((SAME_FACET_R - lb * l) / np.sqrt((1 - lb ** 2) * (1 - l ** 2)), 0, 1)); u = rho * uniq[b] + np.sqrt(1 - rho ** 2) * u
                z = l * eta[:, j] + np.sqrt(1 - l ** 2) * u; lab = level(z)
                if b in uniq:   # rule sets 2 and 3: the same facet is never at opposite extremes in the two instruments
                    bl = self.out[f'bfi2_facet_{b}']; lab = np.where((bl == 'Very high') & (lab == 'Very low'), 'Low', np.where((bl == 'Very low') & (lab == 'Very high'), 'High', lab)).astype(object)
                self._set(f'big5_{f}', lab); self.fz[f'big5_{f}'] = z
        zl = {k: LAM_MBTI * eta[:, j] + np.sqrt(1 - LAM_MBTI ** 2) * rng.standard_normal(n) for k, j in [('E', 2), ('N', 0), ('F', 3), ('J', 1)]}
        zb = LAM_BAT * eta[:, 2] + np.sqrt(1 - LAM_BAT ** 2) * rng.standard_normal(n)
        Z = np.stack([zf[d] for d in FACT], 1); srt = np.sort(Z, 1); top = Z.argmax(1)
        dom = np.where((srt[:, -1] >= 0.51) & (srt[:, -1] - srt[:, -2] >= self.delta), np.array([DOM[d] for d in FACT], dtype=object)[top], 'Balanced')
        self._set('dominant_trait', dom); self.Z = Z; self.eta = eta
        # social battery: no opposite extreme to the BFI-2 extraversion domain or to an extraversion-dominant profile
        order = ['Strong introvert', 'Introvert', 'Ambivert', 'Extrovert', 'Strong extrovert']
        pri = np.array([self.S.prior['lstyle_social_battery'][self.V['lstyle_social_battery'].index(v)] for v in order])
        bi = np.searchsorted(norm.ppf(np.cumsum(pri)[:-1]), zb)
        e_dom = self.out[BFI2_DOMAIN['E']]
        bi = np.where(e_dom == 'Very high', np.maximum(bi, 1), bi); bi = np.where(e_dom == 'Very low', np.minimum(bi, 3), bi)
        bi = np.where(dom == DOM['E'], np.maximum(bi, 2), bi)
        self._set('lstyle_social_battery', np.array(order, dtype=object)[bi])
        # MBTI letters from the same latent traits; E/I follows the extremes of the BFI-2 domain and the social battery
        E = zl['E'] > 0
        E = np.where((e_dom == 'Very high') | (bi == 4), True, np.where((e_dom == 'Very low') | (bi == 0), False, E))
        N_, F_, J = zl['N'] > 0, zl['F'] > 0, zl['J'] > 0
        typ = np.char.add(np.char.add(np.where(E, 'E', 'I'), np.where(N_, 'N', 'S')), np.char.add(np.where(F_, 'F', 'T'), np.where(J, 'J', 'P')))
        lab = {v.split(' ')[0]: v for v in self.V['mbti_type']}; self._set('mbti_type', np.array([lab[t] for t in typ], dtype=object))
        self.mbti = dict(E=E, N=N_, F=F_, J=J); self.fz['mbti_J'] = zl['J']; self.fz['mbti_F'] = zl['F']; self.fz['eta'] = eta
        self._derived(rng)
        return eta, mu

    def _derived(self, rng):
        """DERIVED attributes from the personality scores (MatrAIx prior proportions, anchors keep them off opposite extremes)"""
        n = len(self.eta); V = self.V
        for t, src, sign, r, anchors in DERIVED:
            z = sign * r * self.fz[src] + np.sqrt(1 - r ** 2) * rng.standard_normal(n)
            p = np.asarray(self.S.prior[t], float); p = p / p.sum(); k = len(p)
            idx = np.searchsorted(norm.ppf(np.cumsum(p)[:-1]), -z)          # 0 = first (top) value
            no_last, no_first = np.zeros(n, bool), np.zeros(n, bool)
            for a, sg in anchors:
                if a in ('MBTI_J', 'MBTI_F'):
                    top = self.mbti[a[-1]]; bot = ~top
                else:
                    o = self.out[a]; top = o == V[a][0]; bot = o == V[a][-1]
                if sg < 0: top, bot = bot, top
                no_last |= top; no_first |= bot
            idx = np.where(no_last, np.minimum(idx, k - 2), idx); idx = np.where(no_first, np.maximum(idx, 1), idx)
            self._set(t, np.array(V[t], dtype=object)[idx])

    def _values(self, sex, age, e, s, eta, mu, rng):
        cat, z = VM.draw(self.VP, sex, 2020 - age, VM.edu3_census(e, s), eta, mu, rng)
        for a, c in cat.items(): self._set(a, np.array(self.V[a], dtype=object)[c])
        self.vz = z
        VP = self.VP; th = np.deg2rad(VP['theta_deg']); Uv = np.stack([np.cos(th), np.sin(th)], 1); sd = np.array(VP['sd'])
        gm = VM.phiV(sex, 2020 - age, VM.edu3_census(e, s)) @ np.array(VP['G']).T @ Uv.T / sd[None, :]
        self.zt = z - gm      # value scores minus group mean: input of the values -> attitudes link

    def _health(self, D, age, bgc, eta, mu, rng):
        H = self.HM; sem = H['sem']; M = H['models']; n = len(D)
        par = dict(bgc); par['inc4'] = self.inc4
        def draw(a):
            m = M[a]; X = np.stack([par[p] for p in m['parents']], 1).astype(int); return core.sample(m, X, rng)
        par['chronic'] = draw('chronic')
        ht = D.gtype.map(HT6).fillna(5).astype(int).to_numpy(); a10 = np.digitize(age, [30, 40, 50, 60, 70])
        cols = [np.ones(n), bgc['sex']] + [(a10 == j).astype(float) for j in range(1, 6)] + [(bgc['mar4'] == j).astype(float) for j in range(1, 4)]
        cols += [(ht == j).astype(float) for j in range(1, 6)] + [(bgc['emp5'] == 4).astype(float)] + [(bgc['edu4'] == j).astype(float) for j in range(1, 4)]
        cols += [(self.inc4 == j).astype(float) for j in range(1, 4)] + [(par['chronic'] == j).astype(float) for j in range(1, 3)]
        chi = np.stack(cols, 1)
        xi = chi @ sem['G'] + np.outer(eta[:, 4] - mu[:, 4], sem['beta_N']) + rng.standard_normal((n, 2)) @ np.linalg.cholesky(sem['Sigma']).T
        self.xi = xi; par['xd'] = np.digitize(xi[:, 0], H['xi_cuts']); par['xa'] = np.digitize(xi[:, 1], H['xi_cuts'])
        for a in ['general', 'smoking', 'alcohol', 'exercise', 'stress']: par[a] = draw(a)
        from krpersona.blocks import health as HMm
        for a in ['chronic', 'general', 'smoking', 'alcohol', 'exercise', 'stress']:
            node, sets, _ = HMm.SETS[a]; self._put(node, [sets[c] for c in par[a]])
        dim = np.array([0] * 9 + [1] * 7); y = (sem['tau'][None, :, :] < (sem['lam'][None, :] * xi[:, dim] + rng.standard_normal((n, 16)))[:, :, None]).sum(2)
        phq, gad = y[:, :9].sum(1), y[:, 9:].sum(1); self.phq, self.gad = phq, gad
        self.out['kr_depression_level'] = np.array([l for _, l in DEP], dtype=object)[np.digitize(phq, [c for c, _ in DEP[1:]])]
        self.out['kr_anxiety_level'] = np.array([l for _, l in ANX], dtype=object)[np.digitize(gad, [c for c, _ in ANX[1:]])]
        good = np.isin(par['general'], [0, 1])
        mh = np.select([(phq >= 20) | (gad >= 15), (phq >= 10) | (gad >= 10), (phq >= 5) | (gad >= 5), good], ['In crisis', 'Struggling', 'Stable', 'Thriving'], 'Stable')
        self._set('health_mental_health', mh.astype(object))
        self.hpar = par

    def _kgss(self, D, age, bgc, rng):
        oob = dict(gh=self.hpar['general'], xd=self.hpar['xd'], party=(D.soc_pol.to_numpy() == 1).astype(int), soc=(D.soc_soc.to_numpy() == 1).astype(int), zt=self.zt,
                   abroad=(self.out['lifex_geographic_mobility'] == 'Moved internationally').astype(int))
        o = self.out; n = len(age); low = lambda t, vals: np.isin(o[t], vals)
        LS = ['Low', 'Very low']; LV_ = ['Minor', 'Irrelevant']
        oob['vp_low'] = {'Achievement': low('val_achievement', LV_) | low('schwartz_value_achievement', LS),
                         'Security': low('val_security_stability', LV_) | low('schwartz_value_security', LS),
                         'Autonomy': low('val_independence', LV_) | low('val_personal_freedom', LV_) | low('schwartz_value_self_direction', LS),
                         'Tradition': low('val_tradition', LV_) | low('schwartz_value_tradition', LS),
                         'Novelty': low('val_adventure', LV_) | low('schwartz_value_stimulation', LS),
                         'Community': low('schwartz_value_benevolence', LS)}   # benevolence alone; val_community is a MatrAIx node: masked later
        oob['out'] = o   # outputs so far, for the KGSS cross-attribute constraints (kgss.KPAIRS)
        Y = KG.generate(self.KG, bgc, age, oob, rng, part_rate=self.part); self.kgss_y = Y
        for a, (out, cond) in KG.outputs(self.KG, Y).items():
            if a == 'values_priority':   # members of a priority set that the person rates low are taken out
                cond = [[m for m in c if not oob['vp_low'][m][i]] or c for i, c in enumerate(cond)]
            if cond is None: self.out[a] = out
            else: self._put(a, cond, out)
        self._set('lang_english', np.array([EN_LANG[v] for v in self.out['english_proficiency']], dtype=object))
