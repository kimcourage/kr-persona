"""Tables for docs chapter 6, sections 6.5-6.9 (health, labour, income, KGSS, family blocks), from the saved model files.

docs/tables/ (UTF-8 with BOM):
  block_b_models.csv             health, labour, income and family models: data, parents, penalty, fit (full data and 80% validation models)
  block_b_kgss_models.csv        the 73 KGSS block models: years, generation-year weights, age handling, parents, penalty, fit (full and 80%)
  block_b_kgss_value_links.csv   values -> attitudes links of the KGSS block (s, n, CV gain, calibration terms; full and 80%)
  block_b_health_sem_items.csv   PHQ-9 / GAD-7 measurement model: loadings and thresholds (rescaled; full and 80%)
  block_b_health_sem_gamma.csv   structural model of depression / anxiety: coefficients on the covariates, residual covariance, neuroticism link
  block_b_income_cuts.csv        income cut points (equivalized income quantiles, USD edges, bins)
  block_b_effects.csv            effects of every parent level and of the value links (log-odds against the first category, other parents at the
                                 reference profile REF), estimate and prior mean
usage: python -m krpersona.report.tables_blocks_b
"""
import pickle
import numpy as np, pandas as pd
from krpersona import paths
from krpersona.blocks import core, pml
from krpersona.blocks.labor_income import EMP4
from krpersona.blocks.values import VALS          # row order of the value-link matrices

OUT = paths.ROOT / 'docs' / 'tables'
SIDO = {11: '서울', 21: '부산', 22: '대구', 23: '인천', 24: '광주', 25: '대전', 26: '울산', 29: '세종', 31: '경기', 32: '강원', 33: '충북', 34: '충남',
        35: '전북', 36: '전남', 37: '경북', 38: '경남', 39: '제주'}
Q5 = ['1분위(최저)', '2분위', '3분위', '4분위', '5분위(최고)']
GH = ['매우 좋음', '좋음', '보통', '나쁨']
IB7 = ['2,000 미만', '2,000–3,500', '3,500–5,000', '5,000–7,000', '7,000–10,000', '10,000–15,000', '15,000 이상']
AB5 = ['61–64', '65–69', '70–74', '75–79', '80+']
# reference profile for the effect table (a common profile instead of the first level of every parent, so that contrasts of parents in an
# interaction pair are read at a plausible level of the other parent): man aged 45-54, married, wage worker, etc.; parents not listed: level 0
REF = dict(sex=0, ag7=3, edu5=3, edu4=2, mar4=1, emp5=0, emp2=0, inc4=1, urban2=0, reg7=0, chronic=0, xd=2, xa=2, general=2, smoking=0, alcohol=2,
           exercise=1, emp_l=0, occ10=3, ind12=1, s17=0, size=2, hh5=3, nemp4=1, metro=0, ib7=3, ab5=0, gh=1)


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    print(name, df.shape)


def load(name):
    return pickle.load(open(paths.art('models', name), 'rb'))


def r(x, k=4):
    return None if x is None else round(float(x), k)


def fmt_list(v, k=4):
    return ' / '.join(f'{x:.{k}f}' for x in v)


def lattice_end(m):
    """True when the chosen penalty is at an end of the reachable factor-2 lattice (s 0.015625 or 16, s_int 0.00625 or 3.2)"""
    s, si = m['info']['s_main'], m['info']['s_int']
    return bool(np.isclose(s, 0.015625) or np.isclose(s, 16) or (m['pairs'] and (np.isclose(si, 0.00625) or np.isclose(si, 3.2))))


def improvement(m):
    return 100 * (m['ll'] - m['ll_marginal']) / abs(m['ll_marginal'])


def level_labels(name, L, ctx):
    """labels of the L levels of parent `name` in a model; ctx: dict of extra label lists"""
    if name in ctx: lab = list(ctx[name])
    elif name in core.BG: lab = list(core.BG[name][0])
    else: lab = [str(i) for i in range(L)]
    if len(lab) < L: lab = lab + ['결측'] * (L - len(lab))
    return lab[:L]


def effects(block, model_name, m, ctx, ref=None):
    """log-odds of each category against the first one when one parent moves from its reference level (REF, else 0) to level l, all other
    parents at their reference levels (interaction cells included); first rows per model: log-odds at the reference profile"""
    D = pml.Design(m['levels'], m['pairs']); ref = {**REF, **(ref or {})}
    x0 = np.array([min(ref.get(p, 0), L - 1) for p, L in zip(m['parents'], m['levels'])], int)
    lab = m['labels']; rows = []
    def z(B, x): return (D.matrix(x[None, :]) @ B)[0]
    zb, zm = z(m['B'], x0), z(m['M'], x0)
    refdesc = ', '.join(f'{p}={level_labels(p, L, ctx)[x0[j]]}' for j, (p, L) in enumerate(zip(m['parents'], m['levels'])))
    for k in range(1, len(lab)):
        rows.append(dict(블록=block, 모형=model_name, 부모='(기준 프로필)', 수준=refdesc, 범주=lab[k], 기준범주=lab[0], 추정=r(zb[k] - zb[0]), 사전평균=r(zm[k] - zm[0])))
    for j, (p, L) in enumerate(zip(m['parents'], m['levels'])):
        ll = level_labels(p, L, ctx)
        for l in range(L):
            if l == x0[j]: continue
            x = x0.copy(); x[j] = l; b, mm = z(m['B'], x), z(m['M'], x)
            for k in range(1, len(lab)):
                rows.append(dict(블록=block, 모형=model_name, 부모=p, 수준=f'{ll[l]} (기준 {ll[x0[j]]})', 범주=lab[k], 기준범주=lab[0],
                                 추정=r((b[k] - b[0]) - (zb[k] - zb[0])), 사전평균=r((mm[k] - mm[0]) - (zm[k] - zm[0]))))
    return rows


def model_row(block, name, data, m, m80, extra=None):
    d = dict(블록=block, 모형=name, 자료=data, MatrAIx_노드=m['target'] or '(없음)', 범주수=len(m['labels']), 범주=' / '.join(map(str, m['labels'])),
             부모=', '.join(f'{p}({L})' for p, L in zip(m['parents'], m['levels'])),
             상호작용=', '.join(f"{m['parents'][a]}×{m['parents'][b]}" for a, b in m['pairs']) or '(없음)',
             계수표=f"{m['B'].shape[0]}×{m['B'].shape[1]}", n=m['n'], s=m['info']['s_main'], s_int=m['info']['s_int'] if m['pairs'] else None,
             격자끝=lattice_end(m), ll=r(m['ll']), ll_사전평균=r(m['ll_prior']), ll_주변=r(m['ll_marginal']), 주변대비개선_pct=r(improvement(m), 2),
             가중주변분포=fmt_list(m['marg']), 사전평균_비고=m['prior_note'])
    if extra: d.update(extra)
    if m80 is not None:
        d.update({'n_80': m80['n'], 's_80': m80['info']['s_main'], 's_int_80': m80['info']['s_int'] if m80['pairs'] else None, 'll_80': r(m80['ll']),
                  '주변대비개선_pct_80': r(improvement(m80), 2)})
    return d


def main():
    H, H8 = load('health_models.pkl'), load('health_models_80.pkl')
    LI, LI8 = load('labor_income_models.pkl'), load('labor_income_models_80.pkl')
    K, K8 = load('kgss_models.pkl'), load('kgss_models_80.pkl')
    F, F8 = load('family_models.pkl'), load('family_models_80.pkl')
    rows, eff = [], []
    # ---------------- health
    hctx = {a: H['models'][a]['labels'] for a in H['models']}; hctx.update(xd=Q5, xa=Q5)
    for a, m in H['models'].items():
        rows.append(model_row('건강', a, '국민건강영양조사 2022' + ('' if a == 'chronic' else ' (잠재 점수 5회 대체, 응답자 단위 CV)'), m, H8['models'][a]))
        eff += effects('건강', a, m, hctx)
    # ---------------- labour, income
    lab = LI['labor']; inc = LI['income']
    lctx = dict(emp_l=EMP4, ind12=lab['ind12_lab'], s17=[f'{c} {SIDO[c]}' for c in lab['sido17']], size=lab['size']['labels'], ib7=IB7)
    for k, nm in [('size', 'company_size'), ('wage', '월 임금 구간(내부)')]:
        rows.append(model_row('노동', nm, '지역별고용조사 2025 상반기 A형', lab[k], LI8['labor'][k]))
        eff += effects('노동', nm, lab[k], lctx)
    for k, nm in [('income', '가구 경상소득 구간(내부, 1,000만 원)'), ('mortgage', '주택담보대출(자가 가구)')]:
        rows.append(model_row('소득', nm, '가계금융복지조사 2025 (2024년 소득)', inc[k], LI8['income'][k]))
        eff += effects('소득', nm, inc[k], lctx)
    # ---------------- family
    fctx = dict(ab5=AB5)
    for sx, nm in [(1, '생존 자녀 수(남, 61세 이상)'), (0, '생존 자녀 수(여, 61세 이상)')]:
        rows.append(model_row('가족', nm, '고령화연구패널 10차(2024), 2020년 나이 61세 이상', F['models'][sx], F8['models'][sx]))
        eff += effects('가족', nm, F['models'][sx], fctx)
    save(pd.DataFrame(rows), 'block_b_models.csv')
    # ---------------- KGSS
    ms, ms8 = K['models'], K8['models']; krows = []
    kctx_base = dict(gh=GH, xd=Q5, party=['비당원·소극 참여', '당원(적극 활동)'])
    for i, a in enumerate(K['order']):
        m, m8 = ms[a], ms8.get(a)
        ctx = dict(kctx_base); ctx['year'] = [str(y) for y in m['years']]
        for p in m['par_in']: ctx[p] = ms[p]['labels']
        ex = {'순서': i + 1, '추정 연도': ', '.join(map(str, m['years'])), '생성 연도 가중치': ', '.join(f'{y}: {w:.3f}' for y, w in m['gen_w'].items()),
              '연령 처리': m['age_mode'] + (f" (나이 − {m['cohort_shift']})" if m['cohort_shift'] else ''), '배경 변수': ', '.join(m['bg']),
              '블록 안 부모': ', '.join(m['par_in']) or '(없음)', '블록 밖 부모': ', '.join(m['outb']) or '(없음)', '추정 기간 행 수': m['n_window'],
              '가치 연결': a in K.get('vlinks', {})}
        d = model_row('KGSS', a, '한국종합사회조사', m, m8, ex)
        krows.append({k: d[k] for k in ['순서', '모형', 'MatrAIx_노드', '범주수', '범주', '추정 연도', '생성 연도 가중치', '연령 처리', '배경 변수', '블록 안 부모', '블록 밖 부모',
                                        '상호작용', '계수표', '추정 기간 행 수', 'n', 's', 's_int', '격자끝', 'll', 'll_사전평균', 'll_주변', '주변대비개선_pct', '가중주변분포',
                                        '사전평균_비고', '가치 연결', 'n_80', 's_80', 's_int_80', 'll_80', '주변대비개선_pct_80']})
        eff += effects('KGSS', a, m, ctx)
    po = K['extra']; po8 = K8['extra']
    labels_pe = ms['demo_political_engagement']['labels']
    for k in range(1, len(labels_pe)):
        eff.append(dict(블록='KGSS', 모형='demo_political_engagement', 부모='party(고정 오프셋, 2004·2014)', 수준='당원(기준 비당원)', 범주=labels_pe[k], 기준범주=labels_pe[0],
                        추정=r(po['party_offset'][k] - po['party_offset'][0]), 사전평균=None))
    save(pd.DataFrame(krows).rename(columns={'모형': '속성'}), 'block_b_kgss_models.csv')
    # value links
    vrows = []
    for a in K['order']:
        v, v8 = K['vlinks'].get(a), K8['vlinks'].get(a)
        if v is None and v8 is None: continue
        d = {'속성': a}
        for tag, x in [('', v), ('_80', v8)]:
            d.update({f'n{tag}': x['n'] if x else None, f's{tag}': x['s'] if x else None, f'CV이득_1인당{tag}': r(x['cv_gain'], 5) if x else None,
                      f'max_abs_delta{tag}': r(np.abs(x['delta']).max()) if x else None})
        vrows.append(d)
        if v is not None:
            labs = ms[a]['labels']
            for j, val in enumerate(VALS):
                for k in range(1, len(labs)):
                    eff.append(dict(블록='KGSS', 모형=a, 부모='가치 연결(z̃ 1 표준편차)', 수준=val, 범주=labs[k], 기준범주=labs[0],
                                    추정=r(v['Bv'][j, k] - v['Bv'][j, 0]), 사전평균=0.0))
    save(pd.DataFrame(vrows), 'block_b_kgss_value_links.csv')
    # ---------------- health SEM
    s, s8 = H['sem'], H8['sem']; it = []
    for k, name in enumerate(s['items']):
        it.append({'문항': ('PHQ-9 ' if k < 9 else 'GAD-7 ') + str(k + 1 if k < 9 else k - 8), '변수': ('BP_PHQ_' if k < 9 else 'BP_GAD_') + str(k + 1 if k < 9 else k - 8),
                   '차원': '우울' if k < 9 else '불안', 'lambda': r(s['lam'][k]), 'tau_0|1': r(s['tau'][k, 0]), 'tau_1|2': r(s['tau'][k, 1]), 'tau_2|3': r(s['tau'][k, 2]),
                   'lambda_80': r(s8['lam'][k]), 'tau_0|1_80': r(s8['tau'][k, 0]), 'tau_1|2_80': r(s8['tau'][k, 1]), 'tau_2|3_80': r(s8['tau'][k, 2])})
    save(pd.DataFrame(it), 'block_b_health_sem_items.csv')
    g = [{'항목': c, '우울(xi_D)': r(s['G'][i, 0]), '불안(xi_A)': r(s['G'][i, 1]), '우울_80': r(s8['G'][i, 0]), '불안_80': r(s8['G'][i, 1])} for i, c in enumerate(s['chi'])]
    for nm, a, b in [('V (chi가 주어졌을 때 공분산)', s['V'], s8['V']), ('Sigma (신경증 분리 뒤 잔차 공분산)', s['Sigma'], s8['Sigma'])]:
        g.append({'항목': nm + ' 1행', '우울(xi_D)': r(a[0, 0]), '불안(xi_A)': r(a[0, 1]), '우울_80': r(b[0, 0]), '불안_80': r(b[0, 1])})
        g.append({'항목': nm + ' 2행', '우울(xi_D)': r(a[1, 0]), '불안(xi_A)': r(a[1, 1]), '우울_80': r(b[1, 0]), '불안_80': r(b[1, 1])})
    g.append({'항목': 'beta_N (신경증 1 증가당)', '우울(xi_D)': r(s['beta_N'][0]), '불안(xi_A)': r(s['beta_N'][1]), '우울_80': r(s8['beta_N'][0]), '불안_80': r(s8['beta_N'][1])})
    g.append({'항목': '원척도 잔차 상관 rho', '우울(xi_D)': r(s['rho_resid']), '불안(xi_A)': None, '우울_80': r(s8['rho_resid']), '불안_80': None})
    g.append({'항목': 'Phi_NN (성격 모형)', '우울(xi_D)': r(s['PhiNN']), '불안(xi_A)': None, '우울_80': r(s8['PhiNN']), '불안_80': None})
    g.append({'항목': 'r_D, r_A (문헌)', '우울(xi_D)': s['r'][0], '불안(xi_A)': s['r'][1], '우울_80': s8['r'][0], '불안_80': s8['r'][1]})
    g.append({'항목': 'n', '우울(xi_D)': s['n'], '불안(xi_A)': None, '우울_80': s8['n'], '불안_80': None})
    save(pd.DataFrame(g), 'block_b_health_sem_gamma.csv')
    # ---------------- income cut points
    cr = []
    for key, desc in [('ses_cuts', 'socioeconomic_band 5분위 경계(19세 이상 성인 가중)'), ('ses_cuts_all_persons', '참고: 전체 가구원 가중 5분위 경계'),
                      ('inc4_cuts', '건강 블록 소득 4분위 경계(가구원 가중)'), ('usd_edges', 'demo_household_income 달러 구간 경계(1,300원/달러)'),
                      ('mort_cuts', '주택담보대출 모형의 소득 구간 경계'), ('inc_cuts', '가구소득 모형의 구간 경계(21구간)')]:
        cr.append({'경계': key, '설명': desc, '전체 자료(만 원/년)': ', '.join(f'{x:,.1f}' for x in inc[key]), '80% 자료(만 원/년)': ', '.join(f'{x:,.1f}' for x in LI8['income'][key])})
    save(pd.DataFrame(cr), 'block_b_income_cuts.csv')
    save(pd.DataFrame(eff), 'block_b_effects.csv')


if __name__ == '__main__':
    main()
