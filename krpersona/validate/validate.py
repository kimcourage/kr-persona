"""Validation: (1) 2/3-way TVD vs held-out data, (2) pMSE, (3) independent contradiction rules, (4) official statistics.
usage: KR_HOLDOUT=1 python validate.py <personas.parquet> <internal.parquet> <out.pkl>"""
import sys, itertools, pickle, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from krpersona.validate import sources as SRC
RNG = np.random.default_rng(2026)
MAX_TRIPLES = 1500

def joint(codes, w, dims):
    idx = np.ravel_multi_index(codes.T, dims); return np.bincount(idx, weights=w, minlength=int(np.prod(dims))) / w.sum()

def n_eff(w): return w.sum() ** 2 / (w ** 2).sum()

def tvd_sets(src, P, label, subsets, syn_scope=None, year_rows=None):
    """per subset: TVD(synthetic, held-out), TVD(MatrAIx, held-out), noise floor (synthetic resampled to the held-out effective size)"""
    real, w, T = src['real'], src['w'], src['to_class']
    S = {a: T[a][0](P[a]) for a in T if a in P}; K = {a: T[a][1] + 1 for a in T}
    scope = np.ones(len(P), bool) if syn_scope is None else syn_scope
    MXc = {a: T[a][0](label[a]) for a in T if a in label} if label is not None else {}
    mscope = src.get('scope')(label) if (label is not None and src.get('scope')) else np.ones(len(label) if label is not None else 0, bool)
    out = []
    for sub in subsets:
        if any(a not in S for a in sub): continue
        rows = np.all(np.stack([real[a].to_numpy() >= 0 for a in sub], 1), 1)
        if year_rows is not None: rows &= year_rows(sub)
        if rows.sum() < 30: out.append((sub, np.nan, np.nan, np.nan, int(rows.sum()))); continue
        dims = [K[a] for a in sub]
        pr = joint(np.stack([real[a].to_numpy()[rows] for a in sub], 1), w[rows], dims)
        sc = np.stack([S[a][scope] for a in sub], 1); ps = joint(sc, np.ones(len(sc)), dims)
        t = 0.5 * np.abs(ps - pr).sum()
        ne = max(int(n_eff(w[rows])), 10); fl = np.mean([0.5 * np.abs(joint(sc[RNG.integers(0, len(sc), ne)], np.ones(ne), dims) - ps).sum() for _ in range(3)])
        tm = np.nan
        if MXc and all(a in MXc for a in sub):
            mc = np.stack([MXc[a][mscope] for a in sub], 1); tm = 0.5 * np.abs(joint(mc, np.ones(len(mc)), dims) - pr).sum()
        out.append((sub, t, tm, fl, int(rows.sum())))
    return pd.DataFrame(out, columns=['속성', '합성', 'MatrAIx', '표본오차 기준', '조사 응답 수']).assign(자료=src['name'], k=lambda d: d.속성.map(len))

def subsets_of(attrs, k):
    allc = list(itertools.combinations(attrs, k))
    if k == 3 and len(allc) > MAX_TRIPLES: allc = [allc[i] for i in RNG.choice(len(allc), MAX_TRIPLES, replace=False)]
    return allc

def pmse(src, P, label, attrs, rows=None, cap=3000):
    """decision tree, 5-fold out-of-fold propensity; synthetic vs held-out, MatrAIx vs held-out, held-out half vs half (baseline)"""
    real, w, T = src['real'], src['w'], src['to_class']
    ok = np.all(np.stack([real[a].to_numpy() >= 0 for a in attrs], 1), 1)
    if rows is not None: ok &= rows
    R = np.stack([real[a].to_numpy()[ok] for a in attrs], 1); wr = w[ok]
    if len(R) < 100: return dict(n=len(R))
    h = min(cap, len(R) // 2)
    def draw_real(idx, m): p = wr[idx] / wr[idx].sum(); return R[RNG.choice(idx, m, p=p)]
    def score(A, Bm):
        X = np.vstack([A, Bm]); y = np.r_[np.zeros(len(A)), np.ones(len(Bm))]
        Xo = pd.get_dummies(pd.DataFrame(X).astype(str)).to_numpy(float)
        p = cross_val_predict(DecisionTreeClassifier(min_samples_leaf=20, max_depth=8, random_state=0), Xo, y, cv=5, method='predict_proba')[:, 1]
        return float(np.mean((p - 0.5) ** 2))
    allr = np.arange(len(R)); perm = RNG.permutation(allr); A, Bh = perm[:len(R) // 2], perm[len(R) // 2:]
    syn = np.stack([T[a][0](P[a]) for a in attrs], 1); scope = src.get('scope')(P) if src.get('scope') else np.ones(len(P), bool); syn = syn[scope]
    out = dict(n=len(R), m=h, synthetic=score(draw_real(allr, h), syn[RNG.integers(0, len(syn), h)]), baseline=score(draw_real(A, h), draw_real(Bh, h)))
    if label is not None and all(a in label for a in attrs):
        mc = np.stack([T[a][0](label[a]) for a in attrs], 1); ms = src.get('scope')(label) if src.get('scope') else np.ones(len(label), bool); mc = mc[ms]
        out['MatrAIx'] = score(draw_real(allr, h), mc[RNG.integers(0, len(mc), h)])
    return out

if __name__ == '__main__':
    P = pd.read_parquet(sys.argv[1]); I = pd.read_parquet(sys.argv[2]); OUT = {}
    srcs = [SRC.census(), SRC.knhanes(), SRC.eaps(), SRC.hfcs(), SRC.kgss()]
    allcols = sorted({a for s in srcs for a in s['to_class']})
    MXA = SRC.matraix_adults([c for c in P.columns if c != 'persona_id'])
    print('MatrAIx adults', len(MXA), flush=True)
    TV, PM = [], {}
    for s in srcs:
        attrs = [a for a in s['to_class'] if a in s['real'].columns]
        scope = s['scope'](P) if s.get('scope') else None
        yr = None
        if 'years' in s:
            Y = s['real'].YEAR.to_numpy(); yrs = s['years']
            def yr(sub, Y=Y, yrs=yrs):
                ks = [a for a in sub if a in yrs]
                if not ks: return np.ones(len(Y), bool)
                common = set.intersection(*[set(yrs[a]) for a in ks])
                return np.isin(Y, list(common)) if common else np.isin(Y, sorted(set().union(*[set(yrs[a]) for a in ks])))
        for k in (2, 3):
            TV.append(tvd_sets(s, P, MXA, subsets_of(attrs, k), syn_scope=scope, year_rows=yr))
            print(s['name'], k, 'done', flush=True)
        if 'years' in s:
            a21 = [a for a in attrs if a in s['years'] and 2021 in s['years'][a]][:25] + ['gender_identity', 'age_bracket', 'highest_education']
            PM[s['name']] = pmse(s, P, MXA, a21, rows=(s['real'].YEAR == 2021).to_numpy())
        else:
            PM[s['name']] = pmse(s, P, MXA, attrs)
        print('pMSE', s['name'], PM[s['name']], flush=True)
    OUT['tvd'] = pd.concat(TV, ignore_index=True); OUT['pmse'] = PM
    # (3) independent rules
    from krpersona.rules import set1 as IR
    t_ours, h_ours = IR.run(P.astype(object).astype(str)); t_mx, h_mx = IR.run(MXA.astype(object).astype(str))
    OUT['rules'] = t_ours.merge(t_mx[['id', 'n_scope', 'violation_rate']], on='id', how='left', suffixes=('', '_MatrAIx')); OUT['any_hard'] = (h_ours, h_mx)
    # (4) official statistics
    c = srcs[0]; e = c['extra']; w = c['w']
    o = dict()
    o['65세 이상 비율(성인)'] = (float(np.mean(I.age >= 65)), float(np.average(e['age'] >= 65, weights=w)), None)
    o['1인 가구 비율(가구 기준, 성인 1/가구원 수 가중)'] = (float(np.average(I.k == 1, weights=1 / I.k)), float(np.average(e['k'] == 1, weights=w / e['k'])), 0.317)
    rel = P.demo_religion_affiliation.astype(str); none = rel.isin(['None', 'Atheist / agnostic', 'Spiritual but unaffiliated'])
    o['종교 있음'] = (float(1 - none.mean()), None, 0.40); o['개신교'] = (float((rel == 'Protestant').mean()), None, 0.18)
    o['불교'] = (float(rel.isin(['Buddhist', 'Won Buddhist']).mean()), None, 0.16); o['천주교'] = (float((rel == 'Catholic').mean()), None, 0.06)
    kn = srcs[1]; ph = kn['extra']['phq']; okp = ~np.isnan(ph)
    o['우울(PHQ-9 ≥ 10)'] = (float(np.mean(I.phq >= 10)), float(np.average(ph[okp] >= 10, weights=kn['w'][okp])), 0.053)
    OUT['official'] = pd.DataFrame([(k, *v) for k, v in o.items()], columns=['지표', '합성', '센서스·조사 20% 분리분', '공식 기준값'])
    pickle.dump(OUT, open(sys.argv[3], 'wb'))
    T_ = OUT['tvd']
    print(T_.groupby(['자료', 'k'])[['합성', 'MatrAIx', '표본오차 기준']].mean().round(4))
    print('any hard', OUT['any_hard']); print(OUT['official'])
