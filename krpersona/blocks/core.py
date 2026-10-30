"""Shared pieces for block estimation: background codes, MatrAIx prior means, fitting and sampling one attribute."""
import sys, os, json, hashlib, numpy as np
# validation mode: KR_HOLDOUT=1 drops a fixed random 20% of every source before estimation; model files get the suffix _80
HOLDOUT = os.environ.get('KR_HOLDOUT') == '1'
SUF = '_80' if HOLDOUT else ''
NO_VLINKS = os.environ.get('KR_NO_VLINKS') == '1'     # validation only: generate without the values -> attitudes links
def holdout_mask(name, n, frac=0.2):
    """fixed 20% hold-out of a source's raw rows (same mask every time for the same source name)"""
    seed = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).random(n) < frac
from krpersona import paths
from krpersona.blocks import pml

# ---------------------------------------------------------------- background variables shared by surveys and the generator
# name -> (labels, MatrAIx parent node, MatrAIx categories per level or None)
BG = {
    'sex': (['남', '여'], 'gender_identity', [['Man'], ['Woman']]),
    'ag7': (['19–24', '25–34', '35–44', '45–54', '55–64', '65–74', '75+'], 'age_bracket',
            [['18-24'], ['25-34'], ['35-44'], ['45-54'], ['55-64'], ['65-74'], ['75-84', '85+']]),
    'edu5': (['중졸 이하', '고졸', '전문대졸', '대졸', '대학원졸'], 'highest_education',
             [['No formal', 'Primary', 'Secondary'], ['Secondary', 'Some college'], ["Associate's"], ["Bachelor's"], ["Master's", 'Doctorate', 'Postdoc']]),
    'edu4': (['초졸 이하', '중졸', '고졸', '대졸 이상'], 'highest_education',
             [['No formal', 'Primary'], ['Secondary'], ['Secondary', 'Some college'], ["Associate's", "Bachelor's", "Master's", 'Doctorate', 'Postdoc']]),
    'mar4': (['미혼', '배우자 있음', '사별', '이혼·별거'], 'demo_marital_status',
             [['Single', 'In a relationship', 'Engaged'], ['Married'], ['Widowed'], ['Divorced', 'Separated']]),
    'emp5': (['임금근로', '고용원 있는 자영', '고용원 없는 자영', '무급가족', '비취업'], 'demo_employment_status',
             [['Full-time', 'Part-time'], ['Self-employed'], ['Self-employed', 'Gig / freelance'], ['Self-employed'], ['Student', 'Unemployed', 'Retired', 'Homemaker']]),
    'emp2': (['취업', '비취업'], 'demo_employment_status',
             [['Full-time', 'Part-time', 'Self-employed', 'Gig / freelance'], ['Student', 'Unemployed', 'Retired', 'Homemaker']]),
    'occ10': (['비취업', '관리자', '전문가', '사무', '서비스', '판매', '농림어업', '기능원', '장치·기계', '단순노무'], 'role_function',
              [None, ['Executive'], ['Research', 'Engineering', 'Design', 'Teaching', 'Clinical', 'Legal', 'Finance'], ['Operations', 'Finance', 'HR'],
               ['Operations'], ['Sales / GTM'], ['Operations'], ['Operations'], ['Operations'], ['Operations']]),
    'urban2': (['동', '읍면'], 'urbanicity', [['Dense urban', 'Suburban'], ['Small town', 'Rural']]),
    'inc4': (['하', '중하', '중상', '상'], 'socioeconomic_band', [['Low income'], ['Lower-middle'], ['Middle', 'Upper-middle'], ['High income']]),
    'reg7': (['서울', '인천경기', '강원', '충청', '경상', '전라', '제주'], None, None),
    'metro': (['비수도권', '수도권'], None, None),
    'hh5': (['1명', '2명', '3명', '4명', '5명 이상'], 'lstyle_household_size', [['Lives alone'], ['2 people'], ['3-4 people'], ['3-4 people'], ['5+ people']]),
    'nemp4': (['0명', '1명', '2명', '3명 이상'], None, None),
}
SIDO_REG7 = {11: 0, 23: 1, 31: 1, 32: 2, 25: 3, 29: 3, 33: 3, 34: 3, 21: 4, 22: 4, 26: 4, 37: 4, 38: 4, 24: 5, 35: 5, 36: 5, 39: 6}

def ag7(age): return np.digitize(np.asarray(age), [25, 35, 45, 55, 65, 75])
def edu5_census(edu, sch):
    """completed level (enrolled, on leave or dropped out at a level counts as the level below): 중졸 이하, 고졸, 전문대졸, 대졸, 대학원졸"""
    e, s = np.asarray(edu), np.asarray(sch); g = s == 1
    return np.select([(e <= 3) | ((e == 4) & ~g), (e == 4) | (np.isin(e, [5, 6]) & ~g), e == 5, e == 6, (e >= 7) & (g | (s == 3))], [0, 1, 2, 3, 4], 3)
def edu4_census(edu, sch):
    """KNHANES levels: 초졸 이하, 중졸, 고졸, 대졸 이상(전문대 포함)"""
    e, s = np.asarray(edu), np.asarray(sch); g = s == 1
    return np.select([(e <= 2) | ((e == 3) & ~g), (e == 3) | ((e == 4) & ~g), (e == 4) | (np.isin(e, [5, 6]) & ~g)], [0, 1, 2], 3)
def mar4(m): return np.select([np.asarray(m) == 1, np.asarray(m) == 2, np.asarray(m) == 3], [0, 1, 2], 3)
def emp5_census(econ, status):
    e, s = np.asarray(econ), np.asarray(status); work = np.isin(e, [1, 2, 3])
    return np.where(~work, 4, np.select([s == 1, s == 3, s == 2], [0, 1, 2], 3))

# ---------------------------------------------------------------- MatrAIx priors
class MX:
    def __init__(self):
        from krpersona.matraix.sampler import KRSampler
        S = KRSampler(set())   # plain MatrAIx compile (nothing fixed)
        self.S = S; self.b = S.b; self.values = S.values; self.prior = S.prior
        g = json.load(open(paths.graph_path())); self.cpts = {c['target']: c for c in g['full_cpts']}
        self.edge = {}
        for t, edges in self.b.in_edges.items():
            for e in edges: self.edge[(e['source'], t)] = e

    def prior_mean(self, target, cats, parents, D, X):
        """target: MatrAIx node or None; cats: list of MatrAIx value sets per outcome category (or None);
        parents: list of (name, MatrAIx node or None, sets per level or None); D/X: design and survey parent codes (for CPT projection)"""
        K = len(cats); M = np.zeros((D.p, K))
        if target is None or target not in self.values or cats is None or any(c is None for c in cats):
            return M, 'no MatrAIx node: intercept from data, effects 0'
        V = self.values[target]; P0 = self.prior[target]
        sel = np.array([[1.0 if v in (s or []) else 0.0 for v in V] for s in cats])      # (K, |V|)
        p0k = np.maximum(sel @ P0, 1e-9); M[0] = np.log(p0k / p0k.sum())
        g = self.b.gamma[target]; notes = []
        cpt = self.cpts.get(target); repl = set(cpt['parents']) if cpt and cpt.get('replace_pairwise_parent_edges') else set()
        for j, (nm, node, sets) in enumerate(parents):
            if node is None or sets is None or node in repl: continue
            e = self.edge.get((node, target))
            if e is None: continue
            L = np.exp(e['logratio'] + self.b.logprior[target][None, :])     # P_e(v | u): (k_node, |V|)
            Pu = self.prior[node]; Vn = self.values[node]
            for lv, us in enumerate(sets):
                if not us: continue
                idx = [Vn.index(u) for u in us if u in Vn]
                if not idx: continue
                pv = (Pu[idx][:, None] * L[idx]).sum(0) / Pu[idx].sum()        # P_e(v | h_j(x))
                pk = np.maximum(sel @ pv, 1e-9); pk /= pk.sum()
                M[D.off[j] + lv] = g * e['weight'] * (np.log(pk) - np.log(p0k / p0k.sum()))
            notes.append(node)
        if cpt:   # CPT: project centered log-probabilities onto main effects (+ pair effects) with the survey's parent mix
            par = cpt['parents']; ours = {node: j for j, (nm, node, sets) in enumerate(parents) if node in par and sets is not None}
            if ours:
                kap = self._cpt_kappa(target, cpt, parents, ours, X, sel)
                if kap is not None:
                    A = D.matrix(X); Am = A[:, :D.main_end]
                    sol = np.linalg.lstsq(Am.toarray(), kap, rcond=None)[0]; M[:D.main_end] = sol
                    if D.pairs:
                        res = kap - Am @ sol; Ap = A[:, D.main_end:]; M[D.main_end:] = np.linalg.lstsq(Ap.toarray(), res, rcond=None)[0]
                    notes.append('CPT(' + ','.join(par) + ')')
        return M, 'MatrAIx edges from ' + ', '.join(notes) if notes else 'MatrAIx prior only (no edges from these parents)'

    def _cpt_kappa(self, target, cpt, parents, ours, X, sel):
        V = self.values[target]; par = cpt['parents']; vt = {p: self.values[p] for p in par}
        rows = {}
        for r in cpt['rows']:
            a = r['parent_assignment']; d = r.get('distribution', {})
            rows[tuple(a.get(p) for p in par)] = np.array([float(d.get(v, 0)) for v in V])
        import itertools
        out = np.zeros((len(X), sel.shape[0]))
        cache = {}
        for i, x in enumerate(map(tuple, X[:, [ours[p] for p in par if p in ours]])):
            if x in cache: out[i] = cache[x]; continue
            fixed = {p: parents[ours[p]][2][lv] for p, lv in zip([p for p in par if p in ours], x)}
            tot = np.zeros(len(V)); wt = 0.0
            for combo in itertools.product(*[(fixed[p] if p in fixed else vt[p]) for p in par]):
                w = np.prod([self.prior[p][vt[p].index(c)] for p, c in zip(par, combo)])
                d = rows.get(combo)
                if d is None or d.sum() <= 0: d = self.prior[target]
                tot += w * d / d.sum(); wt += w
            pk = np.maximum(sel @ (tot / wt), 1e-9); lk = np.log(pk); cache[x] = lk - lk.mean(); out[i] = cache[x]
        return out

# ---------------------------------------------------------------- fit / sample one attribute
def fit_attr(mx, name, target, cats, labels, parents, levels, X, y, w, pairs=(), quick=False, groups=None):
    """returns a model dict; parents: list of (name, node, sets); levels: number of levels per parent"""
    D = pml.Design(levels, pairs); K = len(cats)
    M, note = mx.prior_mean(target, cats, parents, D, X)
    if target is None or target not in mx.values or cats is None or any(c is None for c in cats):
        p = np.bincount(y, weights=w, minlength=K) + 0.5; M[0] = np.log(p / p.sum())
    if quick: B, info = pml.fit(D, X, y, w * len(w) / w.sum(), K, M, 1.0, 0.25), dict(s_main=1.0, s_int=0.25)
    else: B, info = pml.cv_fit(D, X, y, w, K, M, groups=groups)
    P = pml.predict(D, B, X); ll = float((w * np.log(P[np.arange(len(y)), y] + 1e-12)).sum() / w.sum())
    P0 = pml.predict(D, M, X); ll0 = float((w * np.log(P0[np.arange(len(y)), y] + 1e-12)).sum() / w.sum())
    marg = np.bincount(y, weights=w, minlength=K) / w.sum(); llm = float((w * np.log(np.maximum(marg[y], 1e-12))).sum() / w.sum())
    return dict(name=name, target=target, cats=cats, labels=labels, parents=[p[0] for p in parents], levels=list(levels), pairs=list(pairs),
                B=B, M=M, info=info, prior_note=note, n=int(len(y)), ll=ll, ll_prior=ll0, ll_marginal=llm, marg=marg.tolist())

def sample(model, X, rng, allowed=None):
    D = pml.Design(model['levels'], model['pairs']); P = pml.predict(D, model['B'], X)
    if allowed is not None: P = P * allowed; P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    c = P.cumsum(1); u = rng.random(len(X))[:, None] * c[:, -1:]
    return (c < u).sum(1)
