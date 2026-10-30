"""Tables for docs/06 (block models, part A: personality, values, developer status) -> docs/tables/block_a_*.csv (UTF-8 with BOM).

From the saved models (artifacts/models; '100%' = all data, '80%' = KR_HOLDOUT=1 models):
  block_a_pers_coef.csv        personality B (latent, census-centred), Bobs (observed scale), se
  block_a_pers_cov.csv         personality Phi, its correlation matrix R, observed score correlations, Var(mu)
  block_a_pers_loadings.csv    BFI-2 / IPIP-NEO facet loadings (literature), same-facet partner, shared-part rho, implied correlation
  block_a_values_summary.csv   per value: angles, radius, half-sample SD, psi^2, mean, reliability, total SD, link share, group share
  block_a_values_gamma.csv     value group effects Gamma (census-centred) and se
  block_a_values_sw.csv        within-group covariance S_w of the relative value scores
  block_a_values_link.csv      value-personality link L, implied and literature correlations
  block_a_values_corr.csv      within-group correlations: observed (used) and vector + iid error model (design)
  block_a_val_star.csv         the 20 val_* attributes, linked value, MatrAIx prior and cut points
  block_a_value_links.csv      values -> attitudes links of the KGSS block (kgss_models*.pkl['vlinks'])
  block_a_dev_coef.csv         developer model coefficients
  block_a_scalars.csv          scalars: sample sizes, value-model variances and held-out angle test, developer class shares
From the raw data (only if config/paths.toml points to the microdata): fit of the 80% models on the held-out 20%
  block_a_holdout_pers.csv     R^2 of the sex-age model and residual correlations, training vs held-out (KGSS 2009/2011/2012)
  block_a_holdout_values.csv   within-group correlations of relative value scores: held-out vs model S_w vs design model (KGSS 2021)
  block_a_holdout_dev.csv      class shares and log-likelihood among held-out professionals (지역별고용조사 2025 상반기 B형)
usage: python -m krpersona.report.tables_blocks_a
"""
import json, pickle
import numpy as np, pandas as pd
from krpersona import paths
from krpersona.blocks import core, pers as PM, values as VM, dev as DV
from krpersona.generate.generator import LAM_BFI2, LAM_IPIP, SAME_FACET, SAME_FACET_R
from krpersona.generate.base import BFI2, BFI2_DOMAIN, IPIP

OUT = paths.ROOT / 'docs' / 'tables'
MODELS = [('100%', ''), ('80%', '_80')]
SCAL = []   # scalar results of all three blocks -> block_a_scalars.csv
F = PM.FACT


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    print(name, df.shape)


def load(name):
    p = paths.art('models', name)
    return json.load(open(p)) if name.endswith('.json') else pickle.load(open(p, 'rb'))


# ---------------------------------------------------------------- personality
def pers_tables():
    coef, cov = [], []
    for lab, suf in MODELS:
        p = load(f'pers_params{suf}.json')
        for mat in ['B', 'Bobs', 'se']:
            for nm, row in zip(p['names'], p[mat]):
                coef.append(dict(model=lab, matrix=mat, term=nm, **{f: round(v, 4) for f, v in zip(F, row)}))
        Phi = np.array(p['Phi']); d = np.sqrt(np.diag(Phi))
        for mat, M in [('Phi', Phi), ('R', Phi / np.outer(d, d)), ('corr_obs', np.array(p['corr_obs']))]:
            for f, row in zip(F, M):
                cov.append(dict(model=lab, matrix=mat, row=f, **{g: round(v, 4) for g, v in zip(F, row)}))
        cov.append(dict(model=lab, matrix='varmu', row='', **{g: round(v, 4) for g, v in zip(F, p['varmu'])}))
        SCAL.append(dict(block='pers', model=lab, quantity='n', value=p['n'])); SCAL.append(dict(block='pers', model=lab, quantity='pd_fixed', value=p['pd_fixed']))
    save(pd.DataFrame(coef), 'block_a_pers_coef.csv'); save(pd.DataFrame(cov), 'block_a_pers_cov.csv')
    rows = []
    for d in F:
        for f in BFI2[d]:
            rows.append(dict(instrument='BFI-2', domain=d, facet=f, attribute=f'bfi2_facet_{f}', loading=LAM_BFI2[f],
                             source='Choi et al. (2025) Table 3, KOR employee T1 (n = 856)'))
        for f in IPIP[d]:
            b = SAME_FACET.get(f); r = dict(instrument='IPIP-NEO', domain=d, facet=f, attribute=f'big5_{f}', loading=LAM_IPIP[f],
                                            source='장승민 (2017) 표 7, 성인 표본 (N = 334)')
            if b:
                lb, l = LAM_BFI2[b], LAM_IPIP[f]
                rho = float(np.clip((SAME_FACET_R - lb * l) / np.sqrt((1 - lb ** 2) * (1 - l ** 2)), 0, 1))
                r.update(same_facet_partner=f'bfi2_facet_{b}', domain_only_corr=round(lb * l, 3), shared_rho=round(rho, 3),
                         corr_with_partner=round(lb * l + rho * np.sqrt((1 - lb ** 2) * (1 - l ** 2)), 3))
            rows.append(r)
        lam = np.array([LAM_BFI2[f] for f in BFI2[d]]); sd = np.sqrt(lam.sum() ** 2 + (1 - lam ** 2).sum()) / 3
        rows.append(dict(instrument='BFI-2 domain', domain=d, facet='(mean of 3 facets)', attribute=BFI2_DOMAIN[d], loading=round(lam.mean() / sd, 3),
                         source=f'derived: sd = {sd:.4f}, loading = corr(z_d, eta_d)'))
    save(pd.DataFrame(rows), 'block_a_pers_loadings.csv')


# ---------------------------------------------------------------- values
def values_tables():
    summ, gam, sw, link, corr = [], [], [], [], []
    for lab, suf in MODELS:
        p = load(f'value_params{suf}.json')
        th = np.deg2rad(p['theta_deg']); U = np.stack([np.cos(th), np.sin(th)], 1); CG = np.array(p['C_Gamma']); sd = np.array(p['sd'])
        gshare = np.einsum('jk,kl,jl->j', U, CG, U) / sd ** 2
        for j, v in enumerate(p['vals']):
            summ.append(dict(model=lab, value=v, ko=p['ko'][j], theta_theory=p['theta_theory_deg'][j], theta_mds=p['theta_deg'][j], mds_radius=p['mds_radius'][j],
                             half_circ_sd_deg=p['heldout']['half_circ_sd_deg'][j], psi2_mds=round(p['psi2'][j], 4), psi2_theory=round(p['theory_fit']['psi2'][j], 4),
                             vbar=round(p['vbar'][j], 4), alpha=round(p['alpha'][j], 4), sd_total=round(sd[j], 4), link_share=round(p['link_share'][j], 4),
                             group_share=round(float(gshare[j]), 4)))
        se = np.array(p['se'])
        for k, nm in enumerate(p['names']):
            gam.append(dict(model=lab, term=nm, nu1=round(p['G'][0][k], 4), nu2=round(p['G'][1][k], 4), se_nu1=round(se[0, k], 4), se_nu2=round(se[1, k], 4)))
        for j, v in enumerate(p['ko']):
            sw.append(dict(model=lab, value=v, **{w: round(x, 4) for w, x in zip(p['ko'], p['Sw'][j])}))
            link.append(dict(model=lab, value=v, **{f'L_{f}': round(x, 4) for f, x in zip(F, p['L_pers'][j])},
                             **{f'r_model_{f}': round(x, 3) for f, x in zip(F, p['r_implied'][j])}, **{f'r_lit_{f}': x for f, x in zip(F, p['r_lit'][j])}))
        for j in range(10):
            for k in range(j + 1, 10):
                corr.append(dict(model=lab, value_1=p['ko'][j], value_2=p['ko'][k], r_within_obs=round(p['r_within_obs'][j][k], 3),
                                 r_within_design=round(p['r_within_design'][j][k], 3)))
        for q, v in [('n', p['n']), ('sigma2', p['sigma2']), ('sigma2_theory_angles', p['theory_fit']['sigma2']), ('psi2_mean', float(np.mean(p['psi2']))),
                     ('r0', p['sigma2'] / (p['sigma2'] + float(np.mean(p['psi2'])))), ('sigma0_2', p['sigma0_2']), ('heldout_ss_theory', p['heldout']['ss_theory']),
                     ('heldout_ss_mds', p['heldout']['ss_mds']), ('heldout_mds_better', p['heldout']['mds_better']), ('link_shrink', p['link_shrink']),
                     ('mds_eigenvalues', ' '.join(str(x) for x in p['mds_eig'])), ('C_Gamma', json.dumps(np.round(CG, 5).tolist()))]:
            SCAL.append(dict(block='values', model=lab, quantity=q, value=v))
    save(pd.DataFrame(summ), 'block_a_values_summary.csv'); save(pd.DataFrame(gam), 'block_a_values_gamma.csv'); save(pd.DataFrame(sw), 'block_a_values_sw.csv')
    save(pd.DataFrame(link), 'block_a_values_link.csv'); save(pd.DataFrame(corr), 'block_a_values_corr.csv')
    p = load('value_params.json')
    save(pd.DataFrame([dict(attribute=a, value=v, values=' / '.join(p['val_cuts'][a]['values']), prior=' / '.join(f'{x:.3f}' for x in p['val_cuts'][a]['p0']),
                            cuts=' / '.join(f'{x:.4f}' for x in p['val_cuts'][a]['cuts']), loading=p['lam_val']) for a, v in p['val_star'].items()]), 'block_a_val_star.csv')
    rows = []
    for lab, suf in MODELS:
        f = paths.art('models', f'kgss_models{suf}.pkl')
        if not f.exists(): continue
        for a, v in pickle.load(open(f, 'rb'))['vlinks'].items():
            rows.append(dict(model=lab, attribute=a, n=v['n'], categories=v['Bv'].shape[1], s=v['s'], cv_gain_per_person=round(v['cv_gain'], 5),
                             max_abs_Bv=round(float(np.abs(v['Bv']).max()), 4)))
    save(pd.DataFrame(rows), 'block_a_value_links.csv')


# ---------------------------------------------------------------- developer status
def dev_tables():
    terms = ['절편', '여성'] + [f'ag7={x}' for x in core.BG['ag7'][0][1:]] + [f'edu5={x}' for x in core.BG['edu5'][0][1:]]
    from krpersona.blocks.labor_income import IND12_LAB
    terms += [f'ind12={x}' for x in IND12_LAB[1:]]
    rows = []
    for lab, suf in MODELS:
        m = load(f'dev_models{suf}.pkl')
        for t, c in zip(terms, m['coef'].T):
            rows.append(dict(model=lab, term=t, other=round(c[0], 4), developer=round(c[1], 4), technologist=round(c[2], 4)))
        SCAL.append(dict(block='dev', model=lab, quantity='n_professionals', value=m['n']))
        for k, v in m['share_in_prof'].items(): SCAL.append(dict(block='dev', model=lab, quantity=f'share_in_professionals_class{k}', value=v))
        for k, v in m['share_of_employed'].items(): SCAL.append(dict(block='dev', model=lab, quantity=f'share_of_employed_class{k}', value=v))
    save(pd.DataFrame(rows), 'block_a_dev_coef.csv')


# ---------------------------------------------------------------- hold-out fit (real data and model predictions only)
def wcorr(X, w):
    C = np.cov(X.T, aweights=w); d = np.sqrt(np.diag(C)); return C / np.outer(d, d)


def holdout_kgss():
    import pyreadstat
    it_p = [v for p in PM.IT.values() for v in p]; it_v = [f'VALUES{i}' for i in range(1, 23)]
    df, _ = pyreadstat.read_sav(str(paths.raw('kgss', 'all')), usecols=['YEAR', 'AGE', 'SEX', 'FINALWT', 'EDUC', 'GRADUATE'] + it_p + it_v)
    df['ho'] = core.holdout_mask('kgss', len(df))
    # personality: same filters and scoring as pers.fit, scores standardised within year in each part
    p80 = load('pers_params_80.json'); Bobs = np.array(p80['Bobs']); rows = []
    base = df[df.YEAR.isin([2009, 2011, 2012]) & (df.AGE >= 19) & df.SEX.isin([1, 2])]
    base = base[base[it_p].apply(lambda c: c.between(1, 7)).all(axis=1)]
    res = {}
    for part, d in [('training', base[~base.ho]), ('held-out', base[base.ho])]:
        w = d.FINALWT.to_numpy(float); w = w / w.mean(); o = np.zeros((len(d), 5))
        for j, f in enumerate(F):
            pos, neg = PM.IT[f]; s = ((d[pos] + 8 - d[neg]) / 2).to_numpy(float)
            for y in (2009, 2011, 2012):
                m = d.YEAR.to_numpy() == y; mu = np.average(s[m], weights=w[m]); sd = np.sqrt(np.average((s[m] - mu) ** 2, weights=w[m])); o[m, j] = (s[m] - mu) / sd
        e = o - PM.phi(d.SEX, d.AGE) @ Bobs
        r2 = 1 - np.average(e ** 2, axis=0, weights=w) / np.average((o - np.average(o, axis=0, weights=w)) ** 2, axis=0, weights=w)
        res[part] = (len(d), r2, wcorr(e, w))
    for j, f in enumerate(F):
        rows.append(dict(quantity=f'R2 {f}', training=round(res['training'][1][j], 4), held_out=round(res['held-out'][1][j], 4)))
    for j in range(5):
        for k in range(j + 1, 5):
            rows.append(dict(quantity=f'residual corr {F[j]}-{F[k]}', training=round(res['training'][2][j, k], 4), held_out=round(res['held-out'][2][j, k], 4)))
    rows.append(dict(quantity='n', training=res['training'][0], held_out=res['held-out'][0]))
    save(pd.DataFrame(rows), 'block_a_holdout_pers.csv')
    # values: held-out KGSS 2021, relative scores as generated (training vbar), minus the 80% group model; within-group correlations
    v80 = load('value_params_80.json')
    d = df[(df.YEAR == 2021) & (df.AGE >= 19) & df.SEX.isin([1, 2]) & df.ho]; d = d[d[it_v].apply(lambda c: c.between(1, 6)).all(axis=1)]
    w = d.FINALWT.to_numpy(float); w = w / w.mean()
    V = np.stack([np.mean([7 - d[f'VALUES{i}'].to_numpy(float) for i, v in VM.ITEM_VAL.items() if v == val], axis=0) for val in VM.VALS], 1)
    vbar = np.array(v80['vbar']); Vt = V - V.mean(1, keepdims=True) - vbar[None, :] + vbar.mean()
    edu, grad = d.EDUC.to_numpy(), d.GRADUATE.to_numpy() == 1
    edu3 = np.select([(edu >= 6) & (edu <= 7) & grad, np.isin(edu, [4, 5]) & grad, (edu >= 6) & (edu <= 7)], [2, 1, 1], 0)
    th = np.deg2rad(v80['theta_deg']); U = np.stack([np.cos(th), np.sin(th)], 1)
    Rh = wcorr(Vt - VM.phiV(d.SEX.to_numpy(), 2021 - d.AGE.to_numpy(), edu3) @ np.array(v80['G']).T @ U.T, w)
    rows = []
    for j in range(10):
        for k in range(j + 1, 10):
            rows.append(dict(value_1=VM.KO[j], value_2=VM.KO[k], held_out=round(Rh[j, k], 3), model_Sw=round(v80['r_within_obs'][j][k], 3),
                             design_iid=round(v80['r_within_design'][j][k], 3)))
    t = pd.DataFrame(rows)
    t.loc[len(t)] = dict(value_1='(mean |diff| vs held-out)', value_2=f'n = {len(d)}', held_out=None,
                         model_Sw=round((t.model_Sw - t.held_out).abs().mean(), 4), design_iid=round((t.design_iid - t.held_out).abs().mean(), 4))
    save(t, 'block_a_holdout_values.csv')


def holdout_dev():
    cols = ['만연령', '경제활동인구상태코드', '3자리_8차직업소분류코드', '3자리_11차산업소분류코드', '성별코드', '교육정도_학력구분코드', '교육정도_수학구분코드', '시도전국가중값']
    e = pd.read_csv(paths.raw('eaps2025h1', 'type_b'), encoding='cp949', dtype=str, usecols=cols)
    e['ho'] = core.holdout_mask('eaps_b', len(e))
    f = lambda c: pd.to_numeric(e[c], errors='coerce')
    e = e[(f('만연령') >= 19) & (f('경제활동인구상태코드') == 1)].copy(); f = lambda c: pd.to_numeric(e[c], errors='coerce')
    occ = f('3자리_8차직업소분류코드'); div = f('3자리_11차산업소분류코드') // 10
    sec = pd.Series([next((s for a, b, s in DV.KSIC if a <= x <= b), None) if x == x else None for x in div], index=e.index)
    d = pd.DataFrame(index=e.index)
    d['sex'] = (f('성별코드') == 2).astype(int); d['ag7'] = core.ag7(f('만연령'))
    edu = f('교육정도_학력구분코드') + 1; sch = f('교육정도_수학구분코드').map({1: 1, 2: 2, 3: 5, 4: 4, 5: 3})
    d['edu5'] = core.edu5_census(edu.fillna(1), sch.fillna(1)); d['ind12'] = sec.map(DV.IND12)
    d['occ1'] = occ // 100; d['y'] = np.select([occ.isin(DV.DEV), occ.isin(DV.TECH)], [1, 2], 0); d['w'] = f('시도전국가중값'); d['ho'] = e.ho.to_numpy()
    d = d[d.ind12.notna() & d.occ1.notna()]
    m80 = load('dev_models_80.pkl'); rows = []
    for part, dd in [('training', d[~d.ho]), ('held-out', d[d.ho])]:
        p = dd[dd.occ1 == 2]; w = p.w.to_numpy(float); w = w / w.mean()
        L = DV.design(p[list(DV.LEVELS)].to_numpy(int)) @ m80['coef'].T; L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
        y = p.y.to_numpy(); marg = np.array([m80['share_in_prof'][c] for c in (0, 1, 2)])
        r = dict(part=part, n_professionals=len(p))
        for c, nm in enumerate(['other', 'developer', 'technologist']):
            r[f'observed_{nm}'] = round(float(np.average(y == c, weights=w)), 4); r[f'predicted_{nm}'] = round(float(np.average(P[:, c], weights=w)), 4)
        r['loglik_model'] = round(float(np.average(np.log(P[np.arange(len(y)), y]), weights=w)), 4)
        r['loglik_marginal'] = round(float(np.average(np.log(marg[y]), weights=w)), 4)
        for c, nm in [(1, 'developer'), (2, 'technologist')]: r[f'share_of_employed_{nm}'] = round(float((p.w * (p.y == c)).sum() / dd.w.sum()), 5)
        rows.append(r)
    save(pd.DataFrame(rows), 'block_a_holdout_dev.csv')


if __name__ == '__main__':
    pers_tables(); values_tables(); dev_tables(); save(pd.DataFrame(SCAL), 'block_a_scalars.csv')
    try:
        ok =paths.raw('kgss', 'all').exists() and paths.raw('eaps2025h1', 'type_b').exists()
    except RuntimeError:
        ok = False
    if ok:
        holdout_kgss(); holdout_dev()
    else:
        print('raw data not available: hold-out tables skipped')
