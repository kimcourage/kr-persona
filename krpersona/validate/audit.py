# audits written for the MatrAIx diagnosis (step 1, before this generator), applied to MatrAIx Korean-native adults and to generated personas
# usage: python -m krpersona.validate.audit <personas.parquet> <out.pkl>
import sys, os, json, numpy as np, pandas as pd
from krpersona.baseline import demo_checks as dc, analyze as an
from krpersona.baseline.lib import KR as MKR, cols, vals, ci
GEN = [('Silent', 1900, 1945), ('Boomer', 1946, 1964), ('Gen X', 1965, 1980), ('Millennial', 1981, 1996), ('Gen Z', 1997, 2012), ('Gen Alpha', 2013, 2100)]
def make_checks(REF):
    def gen_ok(b):
        lo, hi = dc.AGE_RANGE[b]; b0, b1 = REF - hi, REF - lo; return [g for g, s, e in GEN if s <= b1 and e >= b0]
    def gen_mis(M):
        hit = np.zeros(len(M), bool)
        for b in dc.AGE_RANGE: hit |= dc.m(M, 'age_bracket', [b]) & ~dc.m(M, 'demo_generation', gen_ok(b))
        return hit
    def form_mis(M):
        hit = np.zeros(len(M), bool)
        for b in dc.AGE_RANGE:
            lo, hi = dc.AGE_RANGE[b]; w0, w1 = REF - hi + 10, REF - lo + 25
            ok = [d for d, (s, e) in dc.DEC.items() if s <= w1 and e >= w0]; hit |= dc.m(M, 'age_bracket', [b]) & ~dc.m(M, 'lifex_formative_decade', ok)
        return hit
    R = []
    for r in dc.R:
        if r[0] == 'gen_vs_age': r = r[:5] + (gen_mis,)
        if r[0] == 'formative_vs_age': r = r[:5] + (form_mis,)
        R.append(r)
    return R
def run(M, REF):
    out = []
    any_h = np.zeros(len(M), bool)
    for rid, grp, sev, desc, scope, hit in make_checks(REF):
        s = scope(M); h = hit(M) & s
        if sev == 'hard': any_h |= h
        out.append((rid, grp, sev, desc, int(s.sum()), float(h.sum() / s.sum()) if s.sum() else np.nan))
    return pd.DataFrame(out, columns=['id', '묶음', '종류', '설명', '해당 수', '위반 비율']), float(any_h.mean())
adult = np.isin(MKR[:, ci['age_bracket']], [vals['age_bracket'].index(b) for b in dc.ADULT_BRACKETS])
MA = MKR[adult]
P = pd.read_parquet(sys.argv[1])
M = np.full((len(P), len(cols)), 255, np.uint8)
for j, c in enumerate(cols):
    if c in P.columns:
        lut = {v: i for i, v in enumerate(vals[c])}; M[:, j] = P[c].astype(object).map(lambda v: lut.get(v, 255)).to_numpy(np.uint8)
ra, ha = run(MA, 2026); rb, hb = run(M, 2020)
T = ra[['id', '묶음', '종류', '설명']].copy(); T['MatrAIx 한국 Native 성인'] = ra['위반 비율']; T['생성 표본'] = rb['위반 비율']
T = T[T['MatrAIx 한국 Native 성인'].notna() | T['생성 표본'].notna()]
A, B = an.checks(MA), an.checks(M)
keys = ['lang_korean_native_or_fluent', 'lang_korean_none', 'primary_mandarin', 'lang_mandarin_native', 'also_china_native', 'also_japan_native',
        'cultural_background_collectivist_east_asian', 'ethnicity_east_asian', 'immig_first_gen', 'citizen_by_birth_AND_first_gen_immigrant',
        'monocultural_AND_multiple_native_cultures', 'never_left_hometown_AND_other_native_culture', 'zero_native_languages', 'monolingual_with_2plus_native_langs',
        'age18_24_doctorate_or_postdoc', 'age13_24_exp_11plus_years', 'age13_24_director_vp_csuite', 'employment_retired_AND_age_under_45',
        'dominant_extraversion_AND_bfi2_extraversion_low', 'corr_bfi2_extraversion_vs_ipip_facet_mean', 'corr_bfi2_extraversion_vs_own_facets']
K2 = pd.DataFrame([(k, A[k], B[k]) for k in keys], columns=['점검', 'MatrAIx 한국 Native 성인', '생성 표본'])
pd.to_pickle({'demo': T, 'any_hard': (ha, hb), 'kr': K2, 'n': (len(MA), len(M))}, sys.argv[2])
pd.set_option('display.width', 220); print(T.round(4).to_string(index=False)); print('any hard', round(ha, 4), round(hb, 4)); print(K2.round(4).to_string(index=False))
