"""Family block from KLoSA 10th wave (2024, ages 61+): living-children count for men without a fertility record, and age gaps of non-co-resident children.
Personas are 2020 ages; KLoSA respondents are matched by birth cohort (2020 age = 2024 age - 4).
The stored donors are microaggregated (microaggregate()): each donor's gaps and weight are the means of a group of at least KMIN respondents
with the same number of non-co-resident children, so the model file holds no respondent's own values."""
import sys, pickle, numpy as np, pandas as pd, pyreadstat, warnings
warnings.filterwarnings('ignore')
from krpersona import paths
from krpersona.blocks import core
F = lambda: str(paths.raw('klosa', 'wave10'))
AB = [61, 65, 70, 75, 80]            # 2020 age bands: 61-64, 65-69, 70-74, 75-79, 80+
NT = ['0명', '1명', '2명', '3명', '4명', '5명 이상']
def ab(age): return np.digitize(np.asarray(age), AB[1:])

def load():
    S = range(1, 9)
    cols = ['w10A002_age', 'w10gender1', 'w10marital', 'w10edu', 'w10region2', 'w10Ba003', 'w10wgt_c'] + [f'w10Ba008_{s:02d}' for s in S] + [f'w10Ba013_{s:02d}' for s in S]
    d, _ = pyreadstat.read_sav(F(), usecols=cols); d['ho'] = core.holdout_mask('klosa', len(d))
    d = d[d.w10wgt_c.notna() & (d.w10Ba003 >= 0)].reset_index(drop=True)
    d['age20'] = d.w10A002_age - 4; d = d[d.age20 >= 61].reset_index(drop=True)
    d['male'] = (d.w10gender1 == 1).astype(int)
    d['mar'] = d.w10marital.map({1: 1, 2: 3, 3: 3, 4: 2, 5: 0}).astype(int)      # mar4: 미혼, 배우자 있음, 사별, 이혼·별거
    d['edu4'] = (d.w10edu - 1).astype(int); d['urban2'] = (d.w10region2 == 2).astype(int)
    d['ntot'] = np.minimum(d.w10Ba003, 5).astype(int)
    gaps, cos = [], []
    for i, r in d.iterrows():
        g = [r.w10A002_age - r[f'w10Ba008_{s:02d}'] for s in S if r[f'w10Ba008_{s:02d}'] == r[f'w10Ba008_{s:02d}'] and r[f'w10Ba008_{s:02d}'] > 0]
        c = [r[f'w10Ba013_{s:02d}'] == 1 for s in S if r[f'w10Ba008_{s:02d}'] == r[f'w10Ba008_{s:02d}'] and r[f'w10Ba008_{s:02d}'] > 0]
        gaps.append(g); cos.append(c)
    d['gaps'] = gaps; d['cores'] = cos
    d['nco'] = [sum(c) for c in cos]
    return d

KMIN = 5   # smallest group of respondents behind a stored donor


def microaggregate(donors, k=KMIN):
    """{(male, band): [(gaps, weight)]} -> same keys and list order, each usable donor replaced by its group's mean gaps and mean weight.
    Groups have >= k donors with the same number of children: consecutive in gap order within sex x age band (a remainder of fewer than k
    joins the last group); classes with fewer than k donors are pooled over the age bands of the same sex; a pool still smaller than k
    joins, donor by donor, the group of the same sex and number of children with the nearest mean gaps (of the other sex if there is none)."""
    use = {c: [(np.sort(np.asarray(g, float)), float(w)) for g, w in pool if np.min(g) >= 15] for c, pool in donors.items()}
    assert max(g.max() for pool in use.values() for g, _ in pool) < 61   # the age filter of draw_gaps never binds for a 61+ persona
    gap = lambda m: use[m[0]][m[1]][0]
    def chunks(mem):
        mem = sorted(mem, key=lambda m: tuple(gap(m))); n = len(mem) // k * k
        G = [mem[i:i + k] for i in range(0, n, k)]; G[-1] += mem[n:]; return G
    def nearest(groups, m):
        return min(groups, key=lambda G: np.abs(np.mean([gap(x) for x in G], 0) - gap(m)).sum())
    by = {}
    for c, pool in use.items():
        for j, (g, _) in enumerate(pool): by.setdefault((c[0], len(g)), {}).setdefault(c[1], []).append((c, j))
    groups, cross = [], {}
    for (male, n), bands in sorted(by.items()):
        own, pool = [], []
        for b, mem in sorted(bands.items()): (own.extend(chunks(mem)) if len(mem) >= k else pool.extend(mem))
        if len(pool) >= k: own.extend(chunks(pool))
        elif own: [nearest(own, m).append(m) for m in pool]
        else: cross.setdefault(n, []).extend(pool)
        groups.extend(own)
    for n, mem in cross.items():
        same = [G for G in groups if len(gap(G[0])) == n]; assert same, f'no group with {n} children'
        [nearest(same, m).append(m) for m in mem]
    out = {c: list(pool) for c, pool in use.items()}
    for G in groups:
        mg = np.round(np.mean([gap(m) for m in G], 0)); mw = float(np.mean([use[m[0]][m[1]][1] for m in G]))
        for c, j in G: out[c][j] = (mg.copy(), mw)
    assert all(len(out[c][j][0]) == len(use[c][j][0]) for c in use for j in range(len(use[c])))
    return out, dict(groups=len(groups), smallest=min(map(len, groups)), donors=sum(map(len, use.values())),
                     removed=sum(map(len, donors.values())) - sum(map(len, use.values())))


def fit():
    d = load(); d = d[~d.ho] if core.HOLDOUT else d; mx = core.MX()
    par = [('ab5', None, None), ('mar4', None, None), ('edu4', None, None), ('urban2', None, None)]
    MS = {}
    for sx in (1, 0):
        m = d[d.male == sx]; X = np.stack([ab(m.age20), m.mar, m.edu4, m.urban2], 1); w = m.w10wgt_c.to_numpy(float); w = w / w.mean()
        MS[sx] = core.fit_attr(mx, 'living_children_' + ('men' if sx else 'women'), None, [None] * 6, NT, par, [5, 4, 4, 2], X, m.ntot.to_numpy(), w, pairs=[(0, 1)])
    M = MS[1]; m = d[d.male == 1]
    # donors for non-co-resident children ages: gaps of the non-co-resident children, by sex x 2020 age band
    don = {}
    for (sx, b), g in d.assign(b=ab(d.age20)).groupby(['male', 'b']):
        rows = [(np.array([x for x, c in zip(r.gaps, r.cores) if not c]), r.w10wgt_c) for r in g.itertuples()]
        don[(int(sx), int(b))] = [(gp, wt) for gp, wt in rows if len(gp)]
    # checks
    from collections import Counter
    for sx in (1, 0): print('living children model', 'men' if sx else 'women', 'n', MS[sx]['n'], 'll', round(MS[sx]['ll'], 3), 'marg', round(MS[sx]['ll_marginal'], 3), 's', MS[sx]['info']['s_main'], MS[sx]['info']['s_int'])
    for mar, lab in [(1, '배우자 있음'), (2, '사별'), (3, '이혼·별거'), (0, '미혼')]:
        k = m.mar == mar
        if k.sum(): print(' men', lab, 'n', int(k.sum()), 'mean living children', round(float(np.average(m.ntot[k], weights=m.w10wgt_c[k])), 2))
    gm = np.concatenate([gp for v in don.values() for gp, _ in v]) if don else np.array([])
    print('gap by sex (non-co-resident)', {s: round(float(np.mean(np.concatenate([gp for (sx, b), v in don.items() if sx == s for gp, _ in v]))), 1) for s in (0, 1)})
    don, info = microaggregate(don); print('donor groups', info)
    out = dict(model=MS[1], models=MS, donors=don, ab=AB, nt=NT)
    pickle.dump(out, open(paths.art('models', f'family_models{core.SUF}.pkl', mkdir=True), 'wb'))
    return out, d

def draw_total(F_, male, age, mar4, edu4, urban2, nco, rng):
    """living children for 61+ persons without a census fertility record (men outside a head-spouse pair, women with the item missing);
    at least the co-resident count"""
    X = np.stack([ab(age), mar4, edu4, urban2], 1).astype(int); out = np.zeros(len(age), int)
    al = (np.arange(6)[None, :] >= np.minimum(nco, 5)[:, None]).astype(float)
    for sx in (1, 0):
        k = np.asarray(male) == sx
        if k.any(): out[k] = core.sample(F_['models'][sx], X[k], rng, allowed=al[k])
    return out

def draw_gaps(F_, male, age, k, rng):
    """ages gaps for k non-co-resident children of a 61+ persona: a donor of the same sex and age band with exactly k such children
    (keeps a realistic spacing between siblings: the mean spacing of a group of >= KMIN respondents); fewer donors -> any donor with >= k, then take k of them"""
    pool = F_['donors'].get((int(male), int(ab(age))), [])
    pool = [(g, w) for g, w in pool if g.min() >= 15 and g.max() <= age]
    ok = [(g, w) for g, w in pool if len(g) == k] or [(g, w) for g, w in pool if len(g) >= k]
    if not ok: return None
    p = np.array([w for _, w in ok]); g = ok[rng.choice(len(ok), p=p / p.sum())][0]
    return np.sort(rng.choice(g, k, replace=False)) if len(g) > k else g

if __name__ == '__main__':
    out, d = fit()
    # compare KLoSA women living children with the census record (births - deaths) for the same cohorts
    R = pd.read_csv(paths.raw('census2020', 'person'), encoding='cp949', dtype=str,
                    usecols=['성별코드', '만연령', '혼인상태코드', '남아출산수', '여아출산수', '남아사망수', '여아사망수', '출생국적_대한민국여부', '현재국적_대한민국여부', '출생지구분코드', '인구가중값'])
    R = R[(R.출생국적_대한민국여부 == '1') & (R.현재국적_대한민국여부 == '1') & (R.출생지구분코드 != '4') & (R.성별코드 == '2') & (R.만연령.astype(int) >= 61) & (R.혼인상태코드 != '1')]
    f = lambda c: pd.to_numeric(R[c], errors='coerce').fillna(0)
    R = R.assign(liv=(f('남아출산수') + f('여아출산수') - f('남아사망수') - f('여아사망수')).clip(0, 5), b=ab(R.만연령.astype(int)))
    W = d[(d.male == 0) & (d.mar != 0)].assign(b=lambda t: ab(t.age20))
    print('ever-married women, mean living children: census 2020 vs KLoSA 2024 (cohort matched)')
    for b in range(5):
        c = R[R.b == b]; k = W[W.b == b]
        print(f'  band {b}: census {np.average(c.liv, weights=c.인구가중값.astype(float)):.2f}  KLoSA {np.average(k.ntot, weights=k.w10wgt_c):.2f}  (n {len(k)})')
