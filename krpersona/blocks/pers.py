"""Personality model from KGSS TIPI items (2009, 2011, 2012): B (sex, age), Phi; checks against OECD PIAAC Korea."""
import sys, json, numpy as np, pandas as pd, pyreadstat, openpyxl
from krpersona import paths
from krpersona.blocks import core
IT = {'E': ('EXTROVRT', 'INTROVRT'), 'A': ('AGREEALE', 'CRITICAL'), 'C': ('CONSCNTS', 'HASTY'), 'N': ('EMOTIONL', 'EVENTMPD'), 'O': ('OPEN', 'CLOSED')}
FACT = ['O', 'C', 'E', 'A', 'N']
# TIPI 6-week test-retest (Gosling, Rentfrow & Swann 2003): E .77, A .71, C .76, Emotional stability .70, O .62
RHO = {'E': .77, 'A': .71, 'C': .76, 'N': .70, 'O': .62}
AGES = [19, 30, 40, 50, 60, 70]   # age groups: 19-29 (ref), 30-39, ..., 70+

def phi(sex, age):
    a = np.asarray(age); X = [np.ones(len(a)), (np.asarray(sex) == 2).astype(float)]
    for lo, hi in zip(AGES[1:], AGES[2:] + [200]): X.append(((a >= lo) & (a < hi)).astype(float))
    return np.stack(X, 1)
NAMES = ['절편', '여성', '30–39세', '40–49세', '50–59세', '60–69세', '70세 이상']

def fit():
    items = [v for p in IT.values() for v in p]
    df, _ = pyreadstat.read_sav(str(paths.raw('kgss', 'all')), usecols=['YEAR', 'AGE', 'SEX', 'FINALWT'] + items)
    df['ho'] = core.holdout_mask('kgss', len(df))
    df = df[df.YEAR.isin([2009, 2011, 2012]) & (df.AGE >= 19) & df.SEX.isin([1, 2]) & ~(df.ho & core.HOLDOUT)]
    df = df[df[items].apply(lambda c: c.between(1, 7)).all(axis=1)].copy()
    w = df.FINALWT.to_numpy(float); w = w / w.mean()
    O = {}
    for d, (pos, neg) in IT.items():
        s = (df[pos] + (8 - df[neg])) / 2
        if d == 'N': s = (df[pos] + (8 - df[neg])) / 2      # EMOTIONL high = more neurotic; EVENTMPD reversed
        z = s.copy()
        for y, g in df.groupby('YEAR'):                     # standardize within year (weighted)
            ww = w[df.YEAR.to_numpy() == y]; m = np.average(g.index.map(s), weights=ww)
            sd = np.sqrt(np.average((s[g.index] - m) ** 2, weights=ww)); z[g.index] = (s[g.index] - m) / sd
        O[d] = z.to_numpy()
    o = np.stack([O[d] for d in FACT], 1)
    X = phi(df.SEX, df.AGE); W = np.diag(w)
    Bobs = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (o * w[:, None]))   # (7, 5)
    res = o - X @ Bobs; C = (res * w[:, None]).T @ res / w.sum()
    rho = np.array([RHO[d] for d in FACT]); D = np.sqrt(rho)
    B = Bobs / D[None, :]
    Phi_raw = (C - np.diag(1 - rho)) / np.outer(D, D)
    # SE of B (weighted OLS, sandwich-free approx)
    XtWX = X.T @ (X * w[:, None]); se = np.sqrt(np.diag(np.linalg.inv(XtWX)))[:, None] * np.sqrt(np.diag(C))[None, :] / D[None, :]
    return dict(B=B, Bobs=Bobs, se=se, Phi_raw=Phi_raw, C=C, n=len(df), corr_obs=np.corrcoef(o.T))

def center(B, Phi_raw):
    """intercept so the census target population mean is 0; diagonal 1 - Var(mu); off-diagonals from the data; PD fix"""
    R = pd.read_csv(paths.raw('census2020', 'person'), encoding='cp949', dtype=str,
                    usecols=['성별코드', '만연령', '출생국적_대한민국여부', '현재국적_대한민국여부', '출생지구분코드', '인구가중값'])
    R = R[(R.출생국적_대한민국여부 == '1') & (R.현재국적_대한민국여부 == '1') & (R.출생지구분코드 != '4') & (R.만연령.astype(int) >= 19)]
    w = R.인구가중값.astype(float).to_numpy(); X = phi(R.성별코드.astype(int), R.만연령.astype(int))
    mu = X @ B; B = B.copy(); B[0] -= np.average(mu, axis=0, weights=w); mu = X @ B
    V = np.array([np.average((mu[:, j] - np.average(mu[:, j], weights=w)) ** 2, weights=w) for j in range(5)])
    Cmu = np.cov(mu.T, aweights=w)
    sd_raw = np.sqrt(np.clip(np.diag(Phi_raw), 1e-6, None)); Rr = Phi_raw / np.outer(sd_raw, sd_raw); np.fill_diagonal(Rr, 1)
    Dg = np.sqrt(1 - V)
    # latent total correlation from the data (residual corr), minus the part from group differences
    Phi = Rr * np.outer(Dg, Dg)
    ev, Q = np.linalg.eigh(Phi / np.outer(Dg, Dg)); fixed = bool((ev < 1e-3).any())
    if fixed:
        Rp = Q @ np.diag(np.maximum(ev, 1e-3)) @ Q.T; Rpp = Rp / np.sqrt(np.outer(np.diag(Rp), np.diag(Rp))); Phi = Rpp * np.outer(Dg, Dg)
    return B, Phi, V, fixed

def oecd_korea():
    """PIAAC cycle 2 Korea, unadjusted standardized differences (observed BFI-2-S scores)"""
    wb = openpyxl.load_workbook(paths.raw('oecd_bfi2', 'xlsx'), read_only=True, data_only=True); out = {}
    doms = ['A', 'C', 'ES', 'E', 'O']
    for fig, key in [('Figure 4.1', '25-34 minus 55-65'), ('Figure 4.3', 'men minus women')]:
        for r in wb[fig].iter_rows(values_only=True):
            v = [x for x in r if x is not None]
            if v and str(v[0]) == 'Korea': out[key] = {d: float(v[1 + 4 * i]) for i, d in enumerate(doms)}
    return out

if __name__ == '__main__':
    f = fit(); B, Phi, V, fixed = center(f['B'], f['Phi_raw'])
    json.dump(dict(B=B.tolist(), Phi=Phi.tolist(), fact=FACT, names=NAMES, rho=RHO, n=f['n'], Bobs=f['Bobs'].tolist(), se=f['se'].tolist(),
                   varmu=V.tolist(), pd_fixed=fixed, oecd=oecd_korea(), corr_obs=f['corr_obs'].tolist()), open(paths.art('models', f'pers_params{core.SUF}.json', mkdir=True), 'w'), ensure_ascii=False, indent=1)
    pd.set_option('display.width', 200)
    print('n', f['n'], 'PD fixed', fixed); print(pd.DataFrame(B, index=NAMES, columns=FACT).round(3)); print(pd.DataFrame(f['se'], index=NAMES, columns=FACT).round(3))
    print('Phi'); print(pd.DataFrame(Phi, index=FACT, columns=FACT).round(3)); print('Var(mu)', np.round(V, 3)); print('OECD Korea', oecd_korea())
