# Sequential CART synthesis of the census 2% sample: household structure -> head -> other members -> dwelling
# usage: python -m krpersona.census.synth <n_households or 0 for same as sample> <seed> [train_frac]   (files in artifacts/census)
import sys, time, pickle, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from krpersona import paths

DIR = paths.art('census'); NSYN = int(sys.argv[1]); SEED = int(sys.argv[2]); FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
MIN_LEAF = 10; TRIES = 30; ALLOW_MIN = 3; SDC_TRIES = 8
rng = np.random.default_rng(SEED); T0 = time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)

H = pd.read_parquet(f'{DIR}/H.parquet'); P = pd.read_parquet(f'{DIR}/P.parquet')
if FRAC < 1:
    keep = H.hid.sample(frac=FRAC, random_state=SEED); H = H[H.hid.isin(keep)].reset_index(drop=True); P = P[P.hid.isin(keep)].reset_index(drop=True)
NSYN = NSYN or len(H)
CNT = [f'n{r:02d}' for r in range(2, 15)]
STRUCT = ['gtype', 'hkind'] + CNT
H1 = ['sido', 'dong'] + STRUCT + ['k']
PV = ['mar', 'nat', 'bplace', 'bsido', 'r5', 'r1', 'edu', 'sch', 'fmage', 'econ', 'status', 'occ', 'ind', 'jten', 'place',
      'comm', 'mode', 'lim', 'births', 'deaths', 'prejob', 'brk', 'soc_pol', 'soc_soc']
NUM = {'age', 'd', 'births', 'deaths'}
HKEY = ['sex', 'age', 'mar', 'edu', 'econ', 'status', 'occ', 'nat', 'bplace', 'r5']
H2 = ['dwell', 'tenure_h', 'car', 'pet']

def gate(v, D):
    """rows where variable v applies, from values already in D"""
    if v == 'sch': return D.edu != 1
    if v == 'lim': return D.age >= 5
    if v == 'comm': return D.age >= 12
    if v == 'mode': return D.comm.isin([1, 2])
    if v in ('econ', 'mar'): return D.age >= 15
    if v in ('status', 'ind', 'occ', 'jten', 'place'): return D.econ.isin([1, 2, 3])
    if v == 'fmage': return D.mar.isin([2, 3, 4])
    if v in ('births', 'deaths', 'prejob'): return (D.sex == 2) & D.mar.isin([2, 3, 4])
    if v == 'brk': return D.prejob == 1
    if v in ('soc_pol', 'soc_soc'): return D.age >= 20
    if v == 'bsido': return D.bplace == 5
    return pd.Series(True, index=D.index)

# allowed combinations (value, keys) seen at least ALLOW_MIN times in the real data
PAIRS = {'mar': [['age']], 'edu': [['age']], 'sch': [['edu', 'age']], 'fmage': [['age']], 'econ': [['age']], 'jten': [['age']],
         'births': [['age']], 'r5': [['age']], 'r1': [['age']], 'comm': [['age']], 'occ': [['status']], 'place': [['status']]}
def enc(D, cols):
    c = np.zeros(len(D), dtype=np.int64)
    for i, col in enumerate(cols):
        x = np.nan_to_num(D[col].to_numpy(dtype=float), nan=-1).astype(np.int64) + 2
        c = c * 512 + x
    return c

def mat(D, cols): return np.nan_to_num(D[cols].to_numpy(dtype=np.float32), nan=-1.0)

class Donor:
    def __init__(s, cat): s.cat = cat
    def fit(s, X, y, w):
        t = (DecisionTreeClassifier if s.cat else DecisionTreeRegressor)(min_samples_leaf=MIN_LEAF, random_state=SEED)
        t.fit(X, y, sample_weight=w); s.t = t
        leaf = t.apply(X); o = np.argsort(leaf, kind='stable'); ls = leaf[o]
        s.idx = o; s.cw = np.cumsum(w[o]); s.st = np.zeros(t.tree_.node_count, np.int64); s.en = np.zeros_like(s.st)
        u, a = np.unique(ls, return_index=True); s.st[u] = a; s.en[u] = np.r_[a[1:], len(ls)]
        s.nleaf = len(u); return s
    def draw(s, X):
        leaf = s.t.apply(X); a, b = s.st[leaf], s.en[leaf]
        c0 = np.where(a > 0, s.cw[np.maximum(a - 1, 0)], 0.0); tot = s.cw[b - 1] - c0
        p = np.searchsorted(s.cw, c0 + rng.random(len(X)) * tot, side='right')
        return s.idx[np.clip(p, a, b - 1)]

STATS = {}
def synth_var(v, Dtr, Dsy, feats, target=None, extra_ok=None, wcol='hw', fix=None):
    """fit CART for v on Dtr (rows where gate holds), sample into Dsy (rows where gate holds); returns nothing (writes Dsy[v])"""
    target = target or v
    gtr = gate(v, Dtr) & Dtr[target].notna(); gsy = gate(v, Dsy)
    Dsy[target] = np.nan
    if gsy.sum() == 0: return
    tr = Dtr[gtr]; m = Donor(target not in NUM)
    m.fit(mat(tr, feats), tr[target].to_numpy(), tr[wcol].to_numpy(dtype=float))
    vals = tr[target].to_numpy(); sy = Dsy[gsy].copy()
    pos = m.draw(mat(sy, feats)); sy[target] = vals[pos]
    allowed = []
    for keys in PAIRS.get(target, []):
        ok = set(pd.Series(enc(tr, [target] + keys)).value_counts().loc[lambda s: s >= ALLOW_MIN].index)
        allowed.append((keys, ok))
    def bad(S):
        b = np.zeros(len(S), bool)
        for keys, ok in allowed: b |= ~np.isin(enc(S, [target] + keys), list(ok))
        if extra_ok is not None: b |= ~np.asarray(extra_ok(S), bool)
        return b
    nb0 = 0; badm = bad(sy); nb0 = badm.sum(); tries = 0
    while badm.any() and tries < TRIES:
        X = mat(sy[badm], feats); sy.loc[badm, target] = vals[m.draw(X)]; badm = bad(sy); tries += 1
    nfb = int(badm.sum())
    if nfb and allowed:   # fallback: real values among records with the same keys
        keys = allowed[0][0]; trk = enc(tr, keys); okv = ~bad(tr) if extra_ok is None else ~bad(tr)
        syk = enc(sy[badm], keys); grp = pd.Series(np.arange(len(tr)))[okv].groupby(trk[okv]).apply(np.array).to_dict()
        new = sy.loc[badm, target].to_numpy().copy()
        for i, kk in enumerate(syk):
            g = grp.get(kk)
            if g is not None: new[i] = vals[rng.choice(g)]
        sy.loc[badm, target] = new
    if fix is not None: sy = fix(sy)
    nres = int(bad(sy).sum())
    Dsy.loc[gsy, target] = sy[target].to_numpy()
    STATS[f'{target}{"" if len(Dsy)==NSYN else "*"}'] = dict(train=int(gtr.sum()), leaves=m.nleaf, first_bad=int(nb0), fallback=nfb, residual=nres)
    log(f'{target:8s} n_train={gtr.sum():7d} leaves={m.nleaf:6d} bad={nb0} fallback={nfb} residual={nres}')

# ---------------- stage 1: region and household structure
w = H.hw.to_numpy(float)
S = pd.DataFrame({'hid': np.arange(NSYN)})
S['sido'] = rng.choice(H.sido.to_numpy(), size=NSYN, p=w / w.sum())
S['hw'] = 1.0
H['k'] = H['k'].astype(float)
synth_var('dong', H, S, ['sido'])
m = Donor(False).fit(mat(H, ['sido', 'dong']), H[['k'] + CNT].to_numpy(float), w)   # structure block copied as a whole
pos = m.draw(mat(S, ['sido', 'dong']))
for c in STRUCT + ['k']: S[c] = H[c].to_numpy()[pos]
log('structure block', 'leaves', m.nleaf)

# ---------------- stage 2: household head
HD = P[P.pos == 0].merge(H[['hid'] + STRUCT + ['k']], on='hid')
SH = S.copy()
feats = H1[:]
for v in ['sex', 'age'] + PV:
    extra = fix = None
    if v == 'mar': extra = lambda D: ~((D.n02 > 0) & (D.mar != 2)); fix = lambda D: D.assign(mar=np.where(D.n02 > 0, 2, D.mar))
    if v == 'deaths': extra = lambda D: D.deaths <= D.births; fix = lambda D: D.assign(deaths=np.minimum(D.deaths, D.births))
    if v == 'age': extra = lambda D: D.age >= 14; fix = lambda D: D.assign(age=D.age.clip(lower=14))
    synth_var(v, HD, SH, feats, extra_ok=extra, fix=fix); feats = feats + [v]
SH['rel'] = 1; SH['pos'] = 0; SH['ridx'] = 0; SH['d'] = 0.0

# ---------------- stage 3: other members
OT = P[P.pos > 0].merge(H[['hid'] + STRUCT + ['k']], on='hid')
hk = P[P.pos == 0].set_index('hid')[HKEY].add_prefix('h_')
OT = OT.join(hk, on='hid'); OT['d'] = OT.age - OT.h_age
sp = P[P.rel == 2].drop_duplicates('hid').set_index('hid').age
OT['sp_d'] = np.where(OT.rel != 2, OT.hid.map(sp) - OT.h_age, np.nan)
OT = OT.sort_values(['hid', 'pos']); OT['prev_d'] = np.where(OT.ridx > 0, OT.groupby('hid').d.shift(1), np.nan)
# synthetic member skeleton from the synthesized counts
rows = []
for r in range(2, 15):
    n = SH[f'n{r:02d}'].to_numpy().astype(int)
    hh = np.repeat(SH.hid.to_numpy(), n); idx = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    rows.append(pd.DataFrame({'hid': hh, 'rel': r, 'ridx': idx}))
SO = pd.concat(rows).sort_values(['hid', 'rel', 'ridx']).reset_index(drop=True)
SO['pos'] = SO.groupby('hid').cumcount() + 1
SO = SO.merge(SH[['hid', 'hw'] + H1], on='hid').join(SH.set_index('hid')[HKEY].add_prefix('h_'), on='hid')
SO['hw'] = 1.0; SO['sex'] = np.nan; SO['d'] = np.nan; SO['age'] = np.nan; SO['sp_d'] = np.nan; SO['prev_d'] = np.nan
fs = H1 + ['rel', 'ridx', 'pos', 'h_sex', 'h_age', 'h_mar']
fd = fs + ['sex', 'sp_d', 'prev_d']
ms = Donor(True).fit(mat(OT, fs), OT.sex.to_numpy(), OT.hw.to_numpy(float))
md = Donor(False).fit(mat(OT, fd), OT.d.to_numpy(), OT.hw.to_numpy(float))
okrd = set(pd.Series(enc(OT, ['rel', 'd'])).value_counts().loc[lambda s: s >= ALLOW_MIN].index)
okra = set(pd.Series(enc(OT, ['rel', 'age'])).value_counts().loc[lambda s: s >= ALLOW_MIN].index)
okss = set(pd.Series(enc(OT, ['rel', 'sex', 'h_sex'])).value_counts().loc[lambda s: s >= ALLOW_MIN].index)
log('member sex/age models', ms.nleaf, md.nleaf)
sxv, dv = OT.sex.to_numpy(), OT.d.to_numpy(); CNTR = {'bad': 0, 'clamp': 0}
def badage(D):
    a = D.h_age + D.d
    return (a < 0) | (a > 85) | ~np.isin(enc(D, ['rel', 'd']), list(okrd)) | \
           ~np.isin(enc(D.assign(age=a), ['rel', 'age']), list(okra)) | ((D.ridx > 0) & (D.d > D.prev_d))
def draw_members(mask):
    """sex and age of non-head members in households selected by mask, position by position"""
    for j in range(1, int(SO.pos.max()) + 1):
        J = mask & (SO.pos == j)
        if not J.any(): continue
        sub = SO[J].copy()
        if j > 1:   # spouse and previous same-relationship member already done
            spd = SO[SO.rel == 2].drop_duplicates('hid').set_index('hid').d
            sub['sp_d'] = np.where(sub.rel != 2, sub.hid.map(spd), np.nan)
            prev = SO[SO.pos == j - 1].set_index('hid').d
            sub['prev_d'] = np.where(sub.ridx > 0, sub.hid.map(prev), np.nan)
        sub['sex'] = sxv[ms.draw(mat(sub, fs))]
        for _ in range(TRIES):
            b = ~np.isin(enc(sub, ['rel', 'sex', 'h_sex']), list(okss))
            if not b.any(): break
            sub.loc[b, 'sex'] = sxv[ms.draw(mat(sub[b], fs))]
        sub['d'] = dv[md.draw(mat(sub, fd))]; b = badage(sub); CNTR['bad'] += int(b.sum())
        for _ in range(TRIES):
            if not b.any(): break
            sub.loc[b, 'd'] = dv[md.draw(mat(sub[b], fd))]; b = badage(sub)
        if b.any():   # clamp the few left: keep ordering and age range
            CNTR['clamp'] += int(b.sum()); dd = sub.loc[b, 'd'].clip(lower=-sub.loc[b, 'h_age'], upper=85 - sub.loc[b, 'h_age'])
            sub.loc[b, 'd'] = np.where(sub.loc[b, 'ridx'] > 0, np.minimum(dd, sub.loc[b, 'prev_d']), dd)
        sub['age'] = sub.h_age + sub.d
        SO.loc[J, ['sex', 'd', 'age', 'sp_d', 'prev_d']] = sub[['sex', 'd', 'age', 'sp_d', 'prev_d']].to_numpy()
draw_members(pd.Series(True, index=SO.index))
log('member sex/age done', 'first bad age', CNTR['bad'], 'clamped', CNTR['clamp']); STATS['member_age'] = dict(first_bad=CNTR['bad'], fallback=CNTR['clamp'])
# disclosure control: a synthetic household whose region and full member list (relationship, sex, age) equals a household
# that is unique in the real data gets its members redrawn
def hsig(hh, pp):
    t = pp.sort_values(['hid', 'rel', 'age']); t = t.assign(s=t.rel.astype(int).astype(str) + ':' + t.sex.astype(int).astype(str) + ':' + t.age.astype(int).astype(str))
    m = t.groupby('hid').s.agg(','.join)
    return hh.set_index('hid').sido.astype(int).astype(str) + '|' + hh.set_index('hid').dong.astype(int).astype(str) + '|' + m.reindex(hh.hid).to_numpy()
rs = hsig(H, P); vc = rs.value_counts(); RUQ = set(vc[vc == 1].index)
SDC = []
for it in range(SDC_TRIES):
    ss = hsig(S, pd.concat([SH[['hid', 'rel', 'sex', 'age']].assign(rel=1), SO[['hid', 'rel', 'sex', 'age']]]))
    hit = ss[ss.isin(RUQ)].index; hitk = hit[S.set_index('hid').loc[hit, 'k'] > 1]
    SDC.append((it, len(hit), len(hitk))); log('SDC round', it, 'households matching a real unique', len(hit), 'redrawable', len(hitk))
    if len(hitk) == 0: break
    draw_members(SO.hid.isin(hitk))
STATS['sdc'] = SDC
feats = H1 + ['rel', 'ridx', 'pos', 'sex', 'age', 'd'] + [f'h_{c}' for c in HKEY]
for v in PV:
    extra = fix = None
    if v == 'mar': extra = lambda D: ~((D.rel == 2) & (D.mar != 2)); fix = lambda D: D.assign(mar=np.where(D.rel == 2, 2, D.mar))
    if v == 'deaths': extra = lambda D: D.deaths <= D.births; fix = lambda D: D.assign(deaths=np.minimum(D.deaths, D.births))
    synth_var(v, OT, SO, feats, extra_ok=extra, fix=fix); feats = feats + [v]

# ---------------- stage 4: dwelling, tenure, car, pets
def summ(Pall):
    X = pd.DataFrame({'hid': Pall.hid.to_numpy(), 'n_emp': Pall.econ.isin([1, 2, 3]).to_numpy(int), 'n_u18': (Pall.age < 18).to_numpy(int),
                      'n_65': (Pall.age >= 65).to_numpy(int), 'n_car': Pall['mode'].isin([3, 5]).to_numpy(int), 'n_frn': (Pall.nat >= 3).to_numpy(int)})
    g = X.groupby('hid').sum(); a = Pall.groupby('hid').age
    return g.assign(max_age=a.max(), min_age=a.min())
HS = ['sex', 'age', 'mar', 'edu', 'econ', 'status', 'occ', 'nat', 'bplace', 'r5', 'mode']
SUM = ['n_emp', 'n_u18', 'n_65', 'n_car', 'n_frn', 'max_age', 'min_age']
Hr = H.join(summ(P), on='hid').join(P[P.pos == 0].set_index('hid')[HS].add_prefix('h_'), on='hid')
Sall = pd.concat([SH[['hid'] + ['sex', 'age'] + PV], SO[['hid'] + ['sex', 'age'] + PV]])
Sr = S.join(summ(Sall), on='hid').join(SH.set_index('hid')[HS].add_prefix('h_'), on='hid')
feats = H1 + [f'h_{c}' for c in HS] + SUM
for v in H2:
    synth_var(v, Hr, Sr, feats); feats = feats + [v]

# ---------------- output
SP = pd.concat([SH.assign(rel=1), SO], ignore_index=True)[['hid', 'pos', 'rel', 'ridx', 'sex', 'age'] + PV].sort_values(['hid', 'pos'])
SHH = Sr[['hid', 'sido', 'dong'] + STRUCT + ['k'] + H2]
tag = f'syn_s{SEED}' + ('' if FRAC == 1 else f'_f{FRAC}')
SHH.to_parquet(f'{DIR}/{tag}_H.parquet'); SP.to_parquet(f'{DIR}/{tag}_P.parquet')
pickle.dump(STATS, open(f'{DIR}/{tag}_stats.pkl', 'wb'))
log('saved', SHH.shape, SP.shape)
