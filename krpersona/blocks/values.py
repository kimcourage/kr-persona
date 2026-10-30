"""Value model from KGSS 2021 PVQ items (22 items -> 10 Schwartz values), circumplex vector model + val_* measurement + personality link.
Angles: theory (36 deg steps) is replaced by MDS angles because the estimated angles predict held-out halves better."""
import sys, json, numpy as np, pandas as pd, pyreadstat
from scipy.optimize import minimize
from krpersona import paths
from krpersona.blocks import core
VALS = ['self_direction', 'stimulation', 'hedonism', 'achievement', 'power', 'security', 'conformity', 'tradition', 'benevolence', 'universalism']
KO = ['자기주도', '자극', '쾌락', '성취', '권력', '안전', '순응', '전통', '박애', '보편주의']
THETA0 = np.deg2rad(np.arange(10) * 36.0)
# KGSS 2021 VALUES1-22 (ESS PVQ-21 order + item 21 "help people and contribute to society")
ITEM_VAL = {1: 'self_direction', 2: 'power', 3: 'universalism', 4: 'achievement', 5: 'security', 6: 'stimulation', 7: 'conformity', 8: 'universalism', 9: 'tradition',
            10: 'hedonism', 11: 'self_direction', 12: 'benevolence', 13: 'achievement', 14: 'security', 15: 'stimulation', 16: 'conformity', 17: 'power', 18: 'benevolence',
            19: 'universalism', 20: 'tradition', 21: 'benevolence', 22: 'hedonism'}
# val_* tied to one Schwartz value: the attribute names the value or a Schwartz Value Survey marker item of that value.
# val_family, val_health, val_spirituality_faith, val_patriotism, val_sustainability are KGSS items (block 4); val_community, val_personal_growth,
# val_knowledge_truth, val_privacy have no single marker and stay with MatrAIx.
VAL_STAR = {'val_creativity_self_expression': 'self_direction', 'val_independence': 'self_direction', 'val_personal_freedom': 'self_direction',
            'val_adventure': 'stimulation', 'val_fun_enjoyment': 'hedonism', 'val_achievement': 'achievement', 'val_career_success': 'achievement',
            'val_power_influence': 'power', 'val_wealth': 'power', 'val_social_status': 'power', 'val_recognition': 'power',
            'val_security_stability': 'security', 'val_order_structure': 'security', 'val_tradition': 'tradition',
            'val_helping_others': 'benevolence', 'val_integrity_honesty': 'benevolence', 'val_loyalty': 'benevolence',
            'val_equality': 'universalism', 'val_justice_fairness': 'universalism', 'val_beauty_aesthetics': 'universalism'}
LAM_VAL = 0.75      # loading of a val_* on its Schwartz value score (assumption, same as IPIP facets)
# Parks-Leduc, Feldman & Bardi (2015) meta-analytic rho; values not reported as significant -> 0 (assumption). columns O C E A N
R_LIT = {'self_direction': dict(O=.52), 'stimulation': dict(O=.36, E=.36), 'hedonism': dict(E=.20), 'achievement': dict(C=.17, E=.31),
         'power': dict(E=.31, A=-.42), 'security': dict(O=-.24, C=.37), 'conformity': dict(O=-.27, C=.27, A=.26), 'tradition': dict(O=-.31, A=.22),
         'benevolence': dict(A=.61), 'universalism': dict(O=.33, A=.39)}
FACT = ['O', 'C', 'E', 'A', 'N']
CUTS = [-1.60, -0.57, 0.57, 1.60]    # z -> Very low, Low, Average, High, Very high
COH = [1950, 1960, 1970, 1980, 1990]  # birth cohorts: <=1949, 1950s, 1960s, 1970s, 1980s, 1990+
NAMES = ['절편', '여성', '1950년대생', '1960년대생', '1970년대생', '1980년대생', '1990년 이후생', '전문대·대졸', '대학원졸']

def phiV(sex, by, edu3):
    by = np.asarray(by); X = [np.ones(len(by)), (np.asarray(sex) == 2).astype(float)]
    c = np.digitize(by, COH)
    for k in range(1, 6): X.append((c == k).astype(float))
    for k in (1, 2): X.append((np.asarray(edu3) == k).astype(float))
    return np.stack(X, 1)
def edu3_census(e, s):
    e, s = np.asarray(e).astype(int), np.asarray(s); g = s == 1
    return np.select([(e >= 7) & (g | (s == 3)), np.isin(e, [5, 6]) & g, e >= 7], [2, 1, 1], 0)

def load():
    it = [f'VALUES{i}' for i in range(1, 23)]
    df, _ = pyreadstat.read_sav(str(paths.raw('kgss', 'all')), usecols=['YEAR', 'AGE', 'SEX', 'EDUC', 'GRADUATE', 'FINALWT'] + it)
    df['ho'] = core.holdout_mask('kgss', len(df))
    df = df[(df.YEAR == 2021) & (df.AGE >= 19) & df.SEX.isin([1, 2]) & ~(df.ho & core.HOLDOUT)]
    df = df[df[it].apply(lambda c: c.between(1, 6)).all(axis=1)].copy()
    w = df.FINALWT.to_numpy(float).copy(); w /= w.mean()
    V = np.stack([np.mean([7 - df[f'VALUES{i}'].to_numpy(float) for i, v in ITEM_VAL.items() if v == val], axis=0) for val in VALS], 1)
    edu = df.EDUC.to_numpy(); grad = df.GRADUATE.to_numpy() == 1
    edu3 = np.select([(edu >= 6) & (edu <= 7) & grad, np.isin(edu, [4, 5]) & grad, (edu >= 6) & (edu <= 7)], [2, 1, 1], 0)
    return V, w, df.SEX.to_numpy(), 2021 - df.AGE.to_numpy(), edu3

def dcenter(V, w):
    vbar = np.average(V, axis=0, weights=w); return V - V.mean(1, keepdims=True) - vbar[None, :] + vbar.mean(), vbar

def mds_angles(Vt):
    """classical MDS of relative-score correlations (d^2 = 2(1-r)), 2-D, rotated/reflected onto theory angles (Procrustes)"""
    R = np.corrcoef(Vt.T); J = np.eye(10) - .1; ev, Q = np.linalg.eigh(J @ R @ J); Y = Q[:, [-1, -2]] * np.sqrt(ev[[-1, -2]])
    T = np.stack([np.cos(THETA0), np.sin(THETA0)], 1); best = None
    for refl in (1, -1):
        Yr = Y * np.array([1, refl]); A, s, Bt = np.linalg.svd(Yr.T @ T); Z = Yr @ A @ Bt
        if best is None or s.sum() > best[0]: best = (s.sum(), np.arctan2(Z[:, 1], Z[:, 0]) % (2 * np.pi), np.hypot(Z[:, 0], Z[:, 1]))
    return best[1], best[2], ev[::-1]

def heldout_test(V, w, reps=50, seed=1):
    def ss(Vt, th):
        X = np.stack([np.ones(10), np.cos(th), np.sin(th)], 1); H = X @ np.linalg.pinv(X); r = Vt - Vt @ H.T; return (r ** 2).sum() / len(Vt)
    rng = np.random.default_rng(seed); n = len(V); out = []; angs = []
    for _ in range(reps):
        m = rng.random(n) < .5; a = mds_angles(dcenter(V[m], w[m])[0])[0]; Vb = dcenter(V[~m], w[~m])[0]
        out.append((ss(Vb, THETA0), ss(Vb, a))); angs.append(a)
    out = np.array(out); A = np.exp(1j * np.array(angs)); Rbar = np.abs(A.mean(0))
    return dict(ss_theory=float(out[:, 0].mean()), ss_mds=float(out[:, 1].mean()), mds_better=float((out[:, 1] < out[:, 0]).mean()),
                half_circ_sd_deg=np.rad2deg(np.sqrt(-2 * np.log(Rbar))).round(1).tolist())

def fit_vector(V, w, sex, by, edu3, theta):
    Vt, vbar = dcenter(V, w)
    X10 = np.stack([np.ones(10), np.cos(theta), np.sin(theta)], 1); P = np.linalg.solve(X10.T @ X10, X10.T)     # (3, 10)
    nu = (Vt @ P.T)[:, 1:]                                                     # per-person regression on [1, cos, sin]
    H = X10 @ P; r = Vt - Vt @ H.T
    psi2_j = np.average(r ** 2, axis=0, weights=w) / (1 - np.diag(H))            # per value, df-corrected
    E = (P[1:] * psi2_j[None, :]) @ P[1:].T                                      # error covariance of nu-hat
    X = phiV(sex, by, edu3)
    G = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (nu * w[:, None])).T      # (2, p)
    res = nu - X @ G.T; Vres = np.cov(res.T, aweights=w)
    sigma2 = float(np.trace(Vres - E) / 2)                                       # isotropic vector variance
    m = V.mean(1); g0 = np.linalg.lstsq(X * np.sqrt(w)[:, None], m * np.sqrt(w), rcond=None)[0]
    s0 = float(np.average((m - X @ g0) ** 2, weights=w) - psi2_j.mean() / 10)
    se = np.sqrt(np.outer(np.diag(np.linalg.inv(X.T @ (X * w[:, None]))), np.diag(Vres)))
    return dict(G=G, sigma2=sigma2, psi2=psi2_j, sigma0_2=s0, vbar=vbar, se=se.T, Vres=Vres, E=E, n=len(V))

def center(G):
    R = pd.read_csv(paths.raw('census2020', 'person'), encoding='cp949', dtype=str,
                    usecols=['성별코드', '만연령', '교육정도코드', '수학구분코드', '출생국적_대한민국여부', '현재국적_대한민국여부', '출생지구분코드', '인구가중값'])
    R = R[(R.출생국적_대한민국여부 == '1') & (R.현재국적_대한민국여부 == '1') & (R.출생지구분코드 != '4') & (R.만연령.astype(int) >= 19)]
    X = phiV(R.성별코드.astype(int), 2020 - R.만연령.astype(int), edu3_census(R.교육정도코드, pd.to_numeric(R.수학구분코드).fillna(1)))
    w = R.인구가중값.astype(float).to_numpy()
    mu = X @ G.T; G = G.copy(); G[:, 0] -= np.average(mu, axis=0, weights=w); mu = X @ G.T
    return G, np.cov(mu.T, aweights=w)

def alpha_items(w):
    """Spearman-Brown reliability of each value score from its mean inter-item correlation (raw item scores)"""
    it = [f'VALUES{i}' for i in range(1, 23)]
    df, _ = pyreadstat.read_sav(str(paths.raw('kgss', 'all')), usecols=['YEAR', 'AGE', 'SEX'] + it)
    df['ho'] = core.holdout_mask('kgss', len(df))
    df = df[(df.YEAR == 2021) & (df.AGE >= 19) & df.SEX.isin([1, 2]) & ~(df.ho & core.HOLDOUT)]; df = df[df[it].apply(lambda c: c.between(1, 6)).all(axis=1)]
    out = []
    for val in VALS:
        ii = [f'VALUES{i}' for i, v in ITEM_VAL.items() if v == val]; C = np.cov(df[ii].to_numpy(float).T, aweights=w); d = np.sqrt(np.diag(C))
        k = len(ii); r = (C / np.outer(d, d))[np.triu_indices(k, 1)].mean(); out.append(k * r / (1 + (k - 1) * r))
    return np.array(out)

def fit_link(Sw, Phi, alpha):
    """personality link on the 10 relative scores: Cov(v~, eta | y) = L Phi.
    target corr = literature rho x sqrt(alpha_j) (observed value score vs latent trait), centred over values (relative scores sum to 0),
    shrunk by one factor c if needed so that Sw - L Phi L' stays PSD"""
    Rl = np.array([[R_LIT[v].get(f, 0.0) for f in FACT] for v in VALS])
    s = np.sqrt(np.diag(Sw)); sp = np.sqrt(np.diag(Phi))
    T = Rl * np.sqrt(alpha)[:, None] * s[:, None] * sp[None, :]
    T = T - T.mean(0, keepdims=True)
    ev, Q = np.linalg.eigh(Sw); keep = ev > 1e-8 * ev.max(); Qk = Q[:, keep]; T = Qk @ (Qk.T @ T)     # stay in the span of Sw
    L = T @ np.linalg.inv(Phi)
    Wh = Qk / np.sqrt(ev[keep])[None, :]; M = Wh.T @ L @ Phi @ L.T @ Wh; lam = np.linalg.eigvalsh(M).max()
    c = min(1.0, 0.95 / np.sqrt(lam)) if lam > 0 else 1.0                                              # keep 5% margin
    L = c * L; imp = (L @ Phi) / s[:, None] / sp[None, :]
    share = np.diag(L @ Phi @ L.T) / np.diag(Sw)
    return L, imp, Rl, float(c), share

def prior_cuts():
    from krpersona.matraix.sampler import KRSampler
    from scipy.stats import norm
    S = KRSampler(set()); out = {}
    for a in VAL_STAR:
        V = S.values[a]; P = np.asarray(S.prior[a], float); P = P / P.sum()
        cum = np.cumsum(P[::-1])[:-1]                          # low -> high (values listed high -> low)
        out[a] = dict(values=V, p0=P.tolist(), cuts=norm.ppf(np.clip(cum, 1e-9, 1 - 1e-9)).tolist())
    sch = {}
    for v in VALS:
        a = 'schwartz_value_' + v; P = np.asarray(S.prior[a], float); sch[a] = dict(values=S.values[a], p0=(P / P.sum()).round(4).tolist())
    return out, sch

# ---------------------------------------------------------------- generation
def draw(p, sex, by, edu3, eta, Bphi, rng):
    """returns (dict attr -> category index in MatrAIx value order (0 = highest), z). eta, Bphi: (n, 5) in FACT order"""
    n = len(sex); X = phiV(sex, by, edu3); G = np.array(p['G']); L = np.array(p['L_pers']); Phi = np.array(p['Phi_pers'])
    th = np.deg2rad(p['theta_deg']); Uv = np.stack([np.cos(th), np.sin(th)], 1)
    R = np.array(p['Sw']) - L @ Phi @ L.T; ev, Q = np.linalg.eigh(R); A = Q * np.sqrt(np.maximum(ev, 0))[None, :]
    v = X @ G.T @ Uv.T + (np.asarray(eta) - np.asarray(Bphi)) @ L.T + rng.standard_normal((n, 10)) @ A.T
    z = v / np.array(p['sd'])[None, :]
    out = {}
    for j, val in enumerate(VALS): out['schwartz_value_' + val] = 4 - np.digitize(z[:, j], CUTS)
    for a, val in p['val_star'].items():
        za = LAM_VAL * z[:, VALS.index(val)] + np.sqrt(1 - LAM_VAL ** 2) * rng.standard_normal(n)
        out[a] = 4 - np.digitize(za, p['val_cuts'][a]['cuts'])
    return out, z

if __name__ == '__main__':
    V, w, sex, by, edu3 = load(); Vt, _ = dcenter(V, w)
    th_mds, radius, eig = mds_angles(Vt); test = heldout_test(V, w)
    f0 = fit_vector(V, w, sex, by, edu3, THETA0); f = fit_vector(V, w, sex, by, edu3, th_mds)
    G, CG = center(f['G']); Uv = np.stack([np.cos(th_mds), np.sin(th_mds)], 1)
    X = phiV(sex, by, edu3); Res = Vt - X @ f['G'].T @ Uv.T; Sw = np.cov(Res.T, aweights=w)          # within-group covariance of relative scores
    sd = np.sqrt(np.diag(Sw) + np.einsum('jk,kl,jl->j', Uv, CG, Uv))
    pers = json.load(open(paths.art('models', f'pers_params{core.SUF}.json'))); Phi = np.array(pers['Phi'])
    alpha = alpha_items(w); L, imp, Rl, cshr, share = fit_link(Sw, Phi, alpha)
    vc, sch = prior_cuts()
    # model check: within-group corr implied by vector + iid e vs the observed one used here
    Cd = f['sigma2'] * Uv @ Uv.T + np.diag(f['psi2']); dd = np.sqrt(np.diag(Cd)); Rd = Cd / np.outer(dd, dd); Rw = Sw / np.outer(np.sqrt(np.diag(Sw)), np.sqrt(np.diag(Sw)))
    out = dict(vals=VALS, ko=KO, theta_deg=np.rad2deg(th_mds).round(2).tolist(), theta_theory_deg=np.rad2deg(THETA0).tolist(), mds_radius=radius.round(3).tolist(),
               mds_eig=eig.round(3).tolist(), heldout=test, G=G.tolist(), C_Gamma=CG.tolist(), names=NAMES, coh=COH, sigma2=f['sigma2'], psi2=f['psi2'].tolist(),
               sigma0_2=f['sigma0_2'], vbar=f['vbar'].tolist(), se=f['se'].tolist(), n=f['n'], Sw=Sw.tolist(), sd=sd.tolist(), cuts=CUTS,
               theory_fit=dict(sigma2=f0['sigma2'], psi2=f0['psi2'].tolist()), alpha=alpha.tolist(),
               L_pers=L.tolist(), Phi_pers=Phi.tolist(), link_shrink=cshr, link_share=share.tolist(), r_lit=Rl.tolist(), r_implied=imp.tolist(),
               val_star=VAL_STAR, lam_val=LAM_VAL, val_cuts=vc, schwartz_prior=sch, item_val=ITEM_VAL, r_within_obs=Rw.tolist(), r_within_design=Rd.tolist())
    json.dump(out, open(paths.art('models', f'value_params{core.SUF}.json', mkdir=True), 'w'), ensure_ascii=False, indent=1)
    print('n', f['n'], '| MDS eig', eig[:4].round(2), '| held-out SS theory %.3f MDS %.3f, MDS better %.0f%%' % (test['ss_theory'], test['ss_mds'], 100 * test['mds_better']))
    print(pd.DataFrame(dict(theory=np.rad2deg(THETA0), mds=np.rad2deg(th_mds).round(), radius=radius.round(2), half_sd=test['half_circ_sd_deg'], psi2=f['psi2'].round(3),
                            alpha=alpha.round(2), sd=sd.round(3), link_share=share.round(3)), index=KO))
    print('sigma2 %.4f (theory angles %.4f)  mean psi2 %.4f  sigma0^2 %.4f  r0 %.3f  link shrink %.3f' % (f['sigma2'], f0['sigma2'], f['psi2'].mean(), f['sigma0_2'], f['sigma2'] / (f['sigma2'] + f['psi2'].mean()), cshr))
    print(pd.DataFrame(G.T, index=NAMES, columns=['nu1', 'nu2']).round(3))
    print('implied vs literature r\n', pd.DataFrame(np.hstack([imp.round(2), Rl]), index=KO, columns=[c + '_mod' for c in FACT] + [c + '_lit' for c in FACT]))
    print('within-group relative-score corr: observed (used) vs design (vector + iid e)')
    for j in range(10):
        k = (j + 1) % 10; print(f'  {KO[j]}-{KO[k]} {Rw[j, k]:.2f} / {Rd[j, k]:.2f}', end='')
        k = (j + 5) % 10; print(f' | {KO[j]}-{KO[k]} {Rw[j, k]:.2f} / {Rd[j, k]:.2f}')
    # simulate population check
    rng = np.random.default_rng(0); m = 20000; idx = rng.choice(len(V), m, p=w / w.sum())
    Bphi = np.zeros((m, 5)); eta = rng.multivariate_normal(np.zeros(5), Phi, m)
    cat, z = draw(out, sex[idx], by[idx], edu3[idx], eta, Bphi, rng)
    print('sim z mean/sd', z.mean(0).round(2), z.std(0).round(2))
    print('sim Very high share', {k[15:]: round(float((v == 0).mean()), 3) for k, v in cat.items() if k.startswith('schwartz')})
    print('sim corr(val_*, its value)', {a: round(float(np.corrcoef(cat[a], cat['schwartz_value_' + v])[0, 1]), 2) for a, v in list(VAL_STAR.items())[:6]})
