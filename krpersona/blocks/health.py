"""Health block (KNHANES 2022): depression/anxiety measurement + structural model (A5), then chronic, general health, smoking, alcohol, exercise, stress."""
import sys, pickle, json, numpy as np, pandas as pd, pyreadstat
from krpersona import paths
from krpersona.blocks import core
R_D, R_A = 0.59, 0.58   # neuroticism - depression / anxiety correlations: Davenport et al. (2024, J Affect Disord Rep 17:100832) Tables 1-2 pooled r
HT6 = ['1인', '부부', '부부+자녀', '한부모+자녀', '3세대 이상', '기타']

def load():
    cols = ['sex', 'age', 'edu', 'marri_1', 'marri_2', 'EC1_1', 'EC_stt_1', 'ho_incm', 'town_t', 'genertn', 'cfam', 'wt_itvex', 'D_1_1', 'BP1', 'BS1_1', 'BS3_1',
            'BD1', 'BD1_11', 'BE3_31', 'DI1_dg', 'DI2_dg', 'DE1_dg', 'DJ4_dg', 'DN1_dg'] + [f'BP_PHQ_{i}' for i in range(1, 10)] + [f'BP_GAD_{i}' for i in range(1, 8)]
    df, _ = pyreadstat.read_sav(str(paths.raw('knhanes2022', 'all')), usecols=cols)
    df['ho'] = core.holdout_mask('knhanes', len(df))
    df = df[(df.age >= 19) & df.wt_itvex.notna() & (df.wt_itvex > 0)].copy()
    d = pd.DataFrame(index=df.index)
    d['sex'] = (df.sex == 2).astype(int); d['ag7'] = core.ag7(df.age); d['age10'] = np.digitize(df.age, [30, 40, 50, 60, 70])
    d['edu4'] = df.edu.map({1: 0, 2: 1, 3: 2, 4: 3})
    d['mar4'] = np.select([df.marri_1 == 2, df.marri_2.isin([1, 2]), df.marri_2 == 3, df.marri_2 == 4], [0, 1, 2, 3], -1)
    d['emp5'] = np.select([df.EC1_1 == 2, df.EC_stt_1 == 1, df.EC_stt_1 == 3, df.EC_stt_1 == 2, df.EC_stt_1 == 4], [4, 0, 1, 2, 3], -1)
    d['inc4'] = df.ho_incm - 1; d['urban2'] = df.town_t - 1
    d['ht6'] = df.genertn.map({1: 0, 2: 1, 4: 2, 5: 3, 7: 4, 3: 5, 6: 5}); d['hh5'] = np.minimum(df.cfam, 5) - 1
    dg = sum((df[v] == 1).astype(int) for v in ['DI1_dg', 'DI2_dg', 'DE1_dg', 'DJ4_dg', 'DN1_dg']); d['chronic'] = np.minimum(dg, 2)
    d['general'] = df.D_1_1.map({1: 0, 2: 1, 3: 2, 4: 3, 5: 3})
    d['stress'] = df.BP1.map({1: 0, 2: 1, 3: 2, 4: 3})
    d['smoking'] = np.select([df.BS1_1 == 3, df.BS3_1 == 3, df.BS3_1 == 2, df.BS3_1 == 1], [0, 1, 2, 3], -1)
    d['alcohol'] = np.select([(df.BD1 == 1) | (df.BD1_11 == 1), df.BD1_11 == 2, df.BD1_11.isin([3, 4]), df.BD1_11 == 5, df.BD1_11 == 6], [0, 1, 2, 3, 4], -1)
    d['exercise'] = np.select([df.BE3_31 == 8, df.BE3_31.between(3, 7), df.BE3_31 == 2, df.BE3_31 == 1], [0, 1, 2, 3], -1)
    for i in range(1, 10): d[f'p{i}'] = df[f'BP_PHQ_{i}'].where(df[f'BP_PHQ_{i}'].between(0, 3))
    for i in range(1, 8): d[f'g{i}'] = df[f'BP_GAD_{i}'].where(df[f'BP_GAD_{i}'].between(0, 3))
    d['w'] = df.wt_itvex; d['ho'] = df.ho
    return d

# ---------------------------------------------------------------- 2-D ordinal probit MIMIC with Gauss-Hermite quadrature
def chi_matrix(d):
    X = [np.ones(len(d)), d.sex.to_numpy()]
    for k in range(1, 6): X.append((d.age10 == k).to_numpy(float))
    for k in range(1, 4): X.append((d.mar4 == k).to_numpy(float))
    for k in range(1, 6): X.append((d.ht6 == k).to_numpy(float))
    X.append((d.emp5 == 4).to_numpy(float))
    for k in range(1, 4): X.append((d.edu4 == k).to_numpy(float))
    for k in range(1, 4): X.append((d.inc4 == k).to_numpy(float))
    for k in range(1, 3): X.append((d.chronic == k).to_numpy(float))
    return np.stack(X, 1)
CHI = ['절편', '여성'] + [f'{a}대' for a in ['30', '40', '50', '60', '70+']] + ['배우자 있음', '사별', '이혼·별거'] + HT6[1:] + ['비취업'] + ['중졸', '고졸', '대졸 이상'] + ['소득 중하', '소득 중상', '소득 상'] + ['만성질환 1개', '만성질환 2개+']

ITEMS = [f'p{i}' for i in range(1, 10)] + [f'g{i}' for i in range(1, 8)]


def sem_rows(d):
    """respondents used by the measurement model: all PHQ-9 and GAD-7 items and every background variable known"""
    return d[d[ITEMS].notna().all(axis=1) & d[['sex', 'age10', 'mar4', 'ht6', 'emp5', 'edu4', 'inc4', 'chronic']].ge(0).all(axis=1)]


def fit_sem(d, Q=20, iters=400):
    import torch
    it = ITEMS
    s = sem_rows(d); X = torch.tensor(chi_matrix(s), dtype=torch.float64); Y = torch.tensor(s[it].to_numpy(), dtype=torch.long)
    w = torch.tensor(s.w.to_numpy() / s.w.mean(), dtype=torch.float64); n, p = X.shape
    gh, ghw = np.polynomial.hermite_e.hermegauss(Q); gh = torch.tensor(gh); ghw = torch.tensor(ghw / ghw.sum())
    dim = torch.tensor([0] * 9 + [1] * 7)
    lam = torch.full((16,), 1.0, dtype=torch.float64, requires_grad=True)
    t0 = torch.tensor(np.tile([0.0, 1.0, 2.0], (16, 1)), dtype=torch.float64, requires_grad=True)
    G = torch.zeros((p - 1, 2), dtype=torch.float64, requires_grad=True); rz = torch.zeros(1, dtype=torch.float64, requires_grad=True)   # no intercept: thresholds carry the level
    Xn = X[:, 1:]
    opt = torch.optim.LBFGS([lam, t0, G, rz], lr=1, max_iter=iters, line_search_fn='strong_wolfe')
    nd = torch.distributions.Normal(0., 1.)
    def nll():
        tau = torch.cumsum(torch.cat([t0[:, :1], torch.nn.functional.softplus(t0[:, 1:])], 1), 1)      # ordered thresholds
        rho = torch.tanh(rz)[0]; mu = Xn @ G                                                             # residual var 1 each, corr rho
        z1 = gh[:, None].expand(Q, Q); z2 = gh[None, :].expand(Q, Q); a = z1; b = rho * z1 + torch.sqrt(1 - rho ** 2) * z2
        nodes = torch.stack([a.reshape(-1), b.reshape(-1)], 1); wq = (ghw[:, None] * ghw[None, :]).reshape(-1)    # (Q2, 2)
        xi = mu[:, None, :] + nodes[None, :, :]                                                          # (n, Q2, 2)
        eta = lam[None, None, :] * xi[:, :, dim]                                                         # (n, Q2, 16)
        up = torch.cat([tau, torch.full((16, 1), 1e6, dtype=torch.float64)], 1); lo = torch.cat([torch.full((16, 1), -1e6, dtype=torch.float64), tau], 1)
        u = up[torch.arange(16)[None, :], Y]; l = lo[torch.arange(16)[None, :], Y]
        pr = nd.cdf(u[:, None, :] - eta) - nd.cdf(l[:, None, :] - eta)
        lp = torch.log(pr.clamp_min(1e-12)).sum(2) + torch.log(wq)[None, :]
        return -(w * torch.logsumexp(lp, 1)).sum() + 1e-4 * (G ** 2).sum()
    def closure():
        opt.zero_grad(); L = nll(); L.backward(); return L
    for _ in range(3): opt.step(closure)
    with torch.no_grad():
        tau = torch.cumsum(torch.cat([t0[:, :1], torch.nn.functional.softplus(t0[:, 1:])], 1), 1).numpy(); rho = float(torch.tanh(rz)[0])
        Gm = np.vstack([np.zeros((1, 2)), G.numpy()]); lamv = lam.numpy().copy(); mu = X.numpy() @ Gm
    # rescale each dimension to population mean 0 / variance 1 using the survey weights
    wn = s.w.to_numpy(); m = np.average(mu, axis=0, weights=wn); vmu = np.array([np.average((mu[:, k] - m[k]) ** 2, weights=wn) for k in range(2)])
    sd = np.sqrt(vmu + 1.0); Gs = Gm / sd[None, :]; Gs[0] -= m / sd
    lam_s = lamv * sd[dim.numpy()]; tau_s = tau - (lamv * m[dim.numpy()])[:, None]
    V = np.array([[1 / sd[0] ** 2, rho / (sd[0] * sd[1])], [rho / (sd[0] * sd[1]), 1 / sd[1] ** 2]])   # Cov(xi | chi) on the new scale
    return dict(lam=lam_s, tau=tau_s, G=Gs, V=V, rho_resid=rho, n=int(n), chi=CHI, items=it)

def posterior_draws(sem, d, M=5, Q=20, rng=None):
    """multiple imputation draws of xi for respondents with all items (grid posterior)"""
    rng = rng or np.random.default_rng(0); from scipy.stats import norm
    it = sem['items']; s = sem_rows(d); X = chi_matrix(s); mu = X @ sem['G']; L = np.linalg.cholesky(sem['V'])
    g = np.linspace(-4, 4, 41); Z = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
    dim = np.array([0] * 9 + [1] * 7); Y = s[it].to_numpy().astype(int)
    up = np.concatenate([sem['tau'], np.full((16, 1), 1e6)], 1); lo = np.concatenate([np.full((16, 1), -1e6), sem['tau']], 1)
    Vi = np.linalg.inv(sem['V']); out = np.zeros((M, len(s), 2))
    for i in range(len(s)):
        xi = mu[i][None, :] + Z
        dz = Z @ Vi * Z; lpri = -0.5 * dz.sum(1)
        eta = sem['lam'][None, :] * xi[:, dim]
        pr = norm.cdf(up[np.arange(16), Y[i]][None, :] - eta) - norm.cdf(lo[np.arange(16), Y[i]][None, :] - eta)
        lp = np.log(np.clip(pr, 1e-12, None)).sum(1) + lpri; p = np.exp(lp - lp.max()); p /= p.sum()
        k = rng.choice(len(Z), size=M, p=p); out[:, i] = xi[k]
    return out

# ---------------------------------------------------------------- block attributes
SETS = {'chronic': ('health_chronic_condition', [['None'], ['Managed'], ['Multiple']], ['진단 0개', '1개', '2개 이상']),
        'general': ('health_general_health', [['Excellent'], ['Good'], ['Fair'], ['Poor']], ['매우 좋음', '좋음', '보통', '나쁨']),
        'smoking': ('lstyle_smoking', [['Never'], ['Former'], ['Occasional'], ['Regular']], ['비흡연', '과거 흡연', '가끔', '매일']),
        'alcohol': ('lstyle_alcohol_use', [['Never'], ['Rarely'], ['Socially'], ['Regularly'], ['Heavily']], ['안 마심', '월 1회 미만', '월 1–4회', '주 2–3회', '주 4회 이상']),
        'exercise': ('lstyle_exercise_freq', [['Daily'], ['Weekly'], ['Monthly'], ['Rarely', 'Never']], ['7일', '2–6일', '1일', '0일']),
        'stress': ('health_stress_level', [['Very high'], ['High'], ['Moderate'], ['Low', 'None']], ['대단히 많이', '많이', '조금', '거의 안 느낌'])}
ORDER = ['chronic', 'general', 'smoking', 'alcohol', 'exercise', 'stress']
BGP = ['sex', 'ag7', 'edu4', 'mar4', 'emp5', 'inc4', 'urban2']
PAIRS = [('sex', 'ag7'), ('ag7', 'edu4'), ('sex', 'mar4'), ('ag7', 'emp5')]
XI_CUTS = [-0.84, -0.25, 0.25, 0.84]    # xi quintile bins as parents

def fit_block(d, sem, mx, quick=False):
    draws = posterior_draws(sem, d)
    base = sem_rows(d).copy(); models = {}
    stack = pd.concat([base.assign(xd=np.digitize(draws[m][:, 0], XI_CUTS), xa=np.digitize(draws[m][:, 1], XI_CUTS), w=base.w / len(draws), rid=base.index.to_numpy()) for m in range(len(draws))])
    parent_def = {k: (k, core.BG[k][1], core.BG[k][2]) for k in BGP}
    lev = {k: len(core.BG[k][0]) for k in BGP}
    for a in ORDER:
        node, cats, labels = SETS[a]
        par = list(BGP) + ([] if a == 'chronic' else ['chronic', 'xd', 'xa']) + ORDER[1:ORDER.index(a)] if a != 'chronic' else list(BGP)
        D_ = (stack if a != 'chronic' else d)
        D_ = D_[(D_[par + [a]] >= 0).all(axis=1) & D_[par + [a]].notna().all(axis=1)]
        P = []
        for p in par:
            if p in parent_def: P.append(parent_def[p])
            elif p in SETS: P.append((p, SETS[p][0], SETS[p][1]))
            else: P.append((p, None, None))
        L = [lev.get(p, len(SETS[p][1]) if p in SETS else 5) for p in par]
        pairs = [(par.index(x), par.index(y)) for x, y in PAIRS if x in par and y in par]
        m = core.fit_attr(mx, a, node, cats, labels, P, L, D_[par].to_numpy(int), D_[a].to_numpy(int), D_.w.to_numpy(float), pairs, quick=quick,
                          groups=(D_.rid.to_numpy() if 'rid' in D_ else None))
        models[a] = m; print(a, 'n', m['n'], 's', m['info'].get('s_main'), m['info'].get('s_int'), 'll', round(m['ll'], 4), 'prior', round(m['ll_prior'], 4), 'marg', round(m['ll_marginal'], 4), m['prior_note'][:60], flush=True)
    return models

if __name__ == '__main__':
    d = load()
    if core.HOLDOUT: d = d[~d.ho]
    print('KNHANES adults', len(d))
    if '--reuse-sem' in sys.argv: sem = pickle.load(open(paths.art('models', f'health_models{core.SUF}.pkl'), 'rb'))['sem']
    else: sem = fit_sem(d)
    print('SEM n', sem['n'], 'rho resid', round(sem['rho_resid'], 3), 'V', np.round(sem['V'], 3)); print('lambda', np.round(sem['lam'], 2))
    print(pd.DataFrame(sem['G'], index=CHI, columns=['우울', '불안']).round(3).to_string())
    # neuroticism link
    P = json.load(open(paths.art('models', f'pers_params{core.SUF}.json'))); PhiNN = np.array(P['Phi'])[4, 4]; V = sem['V']
    beta = np.array([R_D * np.sqrt(V[0, 0]) / np.sqrt(PhiNN), R_A * np.sqrt(V[1, 1]) / np.sqrt(PhiNN)])
    Sig = V - np.outer(beta, beta) * PhiNN; sem.update(beta_N=beta, Sigma=Sig, r=(R_D, R_A), PhiNN=PhiNN)
    print('beta_N', np.round(beta, 3), 'Sigma PD', bool(np.all(np.linalg.eigvalsh(Sig) > 0)))
    mx = core.MX(); models = fit_block(d, sem, mx, quick='--quick' in sys.argv)
    pickle.dump(dict(sem=sem, models=models, xi_cuts=XI_CUTS, bgp=BGP), open(paths.art('models', f'health_models{core.SUF}.pkl', mkdir=True), 'wb'))
