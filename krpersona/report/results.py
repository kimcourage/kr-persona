"""Result tables for publication: results/검증_결과.xlsx (one sheet per table, plain formatting) and the same tables as CSV in results/csv/.
Everything is read from artifacts (samples/*/stats.json, census/*, validation/*); nothing is re-estimated here.
Counts of real records below 10 are not published (shown as '<10').
usage: python -m krpersona.report.results
"""
import json, pickle
import numpy as np, pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter
from krpersona import paths

OUT = paths.ROOT / 'results'
V = lambda *p: paths.art('validation', *p)
SAMPLES = [('val_on', '검증 표본(80% 모형)'), ('val_novl', '검증 표본, 가치→태도 연결 없음'), ('out_on', '최종 표본(100% 모형)'), ('out_off', '최종 표본, 한국 속성만')]
MX = 'MatrAIx 한국 Native 성인'
HC, KRC, KOC, MIC, MXC = ('hard 하나 이상(전체)', 'hard 하나 이상: 한국 속성 관련 규칙(한국 속성만 + 섞임)', 'hard 하나 이상: 한국 속성만 규칙',
                          'hard 하나 이상: 섞임 규칙', 'hard 하나 이상: MatrAIx 속성만 규칙')
SOURCES = ['인구주택총조사 2020(20% 가구)', '국민건강영양조사 2022(20%)', '지역별고용조사 2025 상반기 취업자(20%)',
           '가계금융복지조사 2025(20% 가구, 성인 가중)', 'KGSS 2016–2025(20%)']
TABLES = []   # (sheet name, title, notes, DataFrame, {column: number format})
CSV = {'요약': 'summary', '생성': 'generation', '센서스 합성': 'census_tree_stats', '센서스 점검': 'census_distance', '센서스 논리': 'census_logic',
       '센서스 구분': 'census_propensity', '센서스 노출': 'census_disclosure', '센서스 기억': 'census_memorization', '① 분포 거리': 'v1_tvd_summary',
       '① 큰 차이': 'v1_tvd_largest', '② 구분 가능성': 'v2_pmse', '④ 공식 통계': 'v4_official', '⑤ 전달 누락': 'v5_render', '센서스': 'census_summary',
       '영어 능력': 'check_english', '가치 연결': 'check_value_links', '노동 시점': 'check_labour_timing', '언어': 'check_languages', '진단 점검': 'check_matraix_diagnosis',
       '③ 모순율': 'v3_rules_summary', '③ 세트 4 규칙별': 'v3_rules_set4_by_rule', '③ 세트 4 묶음별': 'v3_rules_set4_by_group'}


def add(name, title, df, notes=(), fmt=None):
    TABLES.append((name, title, list(notes), df.reset_index(drop=True), fmt or {}))


def small(x):
    return '<10' if 0 < x < 10 else int(x)


def load(p):
    return pickle.load(open(p, 'rb')) if p.exists() else None


# ------------------------------------------------------------------ generation
def generation():
    rows = []
    for s, lab in SAMPLES:
        p = paths.art('samples', s, 'stats.json')
        if not p.exists(): continue
        st = json.load(open(p))
        rep = [b for b in st.get('repair', []) if b]
        rows.append((lab, s, st['N'], st['tried'], st['acceptance'], st.get('trial_acceptance'), st.get('delta'), 'on' if st.get('switch_on') else 'off',
                     sum(b[0] for b in rep) if rep else None, sum(b[-1] for b in rep) if rep else None, st.get('seconds')))
    add('생성', '표본 생성 통계', pd.DataFrame(rows, columns=['표본', '이름', '인원', '생성 시도', '거부 검사 통과율', '시험 통과율(칸별 가중 계산용)', '우세 성향 δ',
                                                         'MatrAIx 속성', '다시 뽑기 전 규칙 위반 인원', '40번 다시 뽑은 뒤 남은 인원', '소요(초)']),
        ['거부 검사: 0단계 검사 규칙과 MatrAIx 금지 규칙 84개(7장). 다시 뽑기: 규칙 세트 1~3의 hard 규칙(7장). 남은 인원은 버리지 않고 그대로 둔다.'],
        {5: '0.0000', 6: '0.0000'})


# ------------------------------------------------------------------ census synthesis
def census():
    st = load(paths.art('census', 'syn_s1_stats.pkl')); ck = load(paths.art('census', 'syn_s1_check.pkl')); ho = load(paths.art('census', 'syn_s7_f0.5_holdout.pkl'))
    if st:
        rows = [(('가구원' if k.endswith('*') else '가구 또는 가구주'), k.rstrip('*'), v.get('train'), v.get('leaves'), v.get('first_bad'), v.get('fallback'), v.get('residual'))
                for k, v in st.items() if isinstance(v, dict) and 'train' in v]
        T = pd.DataFrame(rows, columns=['단위', '변수', '학습 기록 수', '잎 수', '첫 추출에서 규칙 위반', '대체 기록 사용', '남은 위반'])
        m = st.get('member_age', {})
        T.loc[len(T)] = ('가구원', 'age(나이 차)', None, None, m.get('first_bad'), m.get('fallback'), None)
        add('센서스 합성', '센서스 CART 합성(syn_s1): 변수별 나무와 규칙 위반', T,
            ['학습 기록 수는 가중치 없는 실제 기록 수. 대체 기록: 30번 다시 뽑아도 규칙을 어겨 같은 열쇠 값의 실제 기록으로 바꾼 수. 가구원 나이는 범위로 자른 수(5장).'])
    if ck:
        add('센서스 점검', '센서스 합성 점검: 1·2·3차원 분포 거리(실제 대 합성, %p)',
            pd.concat([ck['uni'].assign(차원='1차원').rename(columns={'변수': '조합'}),
                       ck['bi'].assign(차원='2차원', 조합=lambda d: d['변수 1'] + ' × ' + d['변수 2'])[['차원', '조합', '총변동거리(%p)']],
                       ck['tri'].assign(차원='3차원').rename(columns={'교차표': '조합'})])[['차원', '조합', '총변동거리(%p)']],
            ['2차원은 가장 큰 120개 조합만 저장되어 있다.'], {3: '0.000'})
        lg = ck['logic'].copy(); lg['실제 건수'] = lg['실제 건수'].map(small); lg['합성 건수'] = lg['합성 건수'].map(small)
        add('센서스 논리', '센서스 합성 점검: 논리 규칙 위반(syn_s1)', lg, ['10건 미만은 <10으로 적는다.'], {3: '0.000', 5: '0.000'})
        add('센서스 구분', '센서스 합성 점검: 실제와 합성의 구분 가능성', ck['prop'], [], {2: '0.000', 3: '0.000000'})
        add('센서스 노출', '센서스 합성 점검: 실제의 유일 조합과 같은 합성 기록', ck['disc'], [], {4: '0.00', 6: '0.00'})
    if ck and ho is not None:
        u, b, t, pr, dc = ck['uni'], ck['bi'], ck['tri'], ck['prop'], ck['disc']
        cnt = t[t['교차표'].str.contains('대상자 수')]; t = t[~t['교차표'].str.contains('대상자 수')]   # the last row of 'tri' is a count ratio, not a distance
        rows = [('1차원 분포 거리(%p)', f"{len(u)}개 변수 평균", u['총변동거리(%p)'].mean(), f"최대 {u['총변동거리(%p)'].max():.2f}"),
                ('2차원 분포 거리(%p)', '저장된 상위 120개 조합의 최대', b['총변동거리(%p)'].max(), ''),
                ('3차원 분포 거리(%p)', f"{len(t)}개 교차표 평균", t['총변동거리(%p)'].mean(), f"최대 {t['총변동거리(%p)'].max():.2f}"),
                ('페르소나 대상자 수(합성 ÷ 실제, %)', '국내 출생 내국인 19세 이상', float(cnt['총변동거리(%p)'].iloc[0]) if len(cnt) else None, 'syn_s1_check.pkl의 tri 표 마지막 행'),
                ('구분 가능성 pMSE', '실제 대 합성', float(pr.iloc[0]['pMSE']), f"기준(실제 대 실제) {float(pr.iloc[1]['pMSE']):.6f}"),
                ('구분 가능성 AUC', '실제 대 합성', float(pr.iloc[0]['AUC']), f"기준 {float(pr.iloc[1]['AUC']):.3f}")]
        for _, r in dc.iterrows(): rows.append(('실제의 유일 조합과 같은 합성 기록(%)', r['단위'], float(r['합성 중 비율(%)']), f"실제 중 유일 비율 {float(r['실제 중 비율(%)']):.1f}%"))
        for _, r in ho.iterrows(): rows.append(('기억 여부 시험: 학습 절반 대 보지 않은 절반 일치율 차이(%p)', r['키 변수'], float(r['차이(%p)']), ''))
        add('센서스', '센서스 CART 합성(syn_s1) 점검 요약', pd.DataFrame(rows, columns=['점검', '대상', '값', '참고']),
            ['자세한 표는 csv/census_*.csv. 기억 여부 시험은 절반으로 합성한 syn_s7_f0.5로 한다(5장).'], {3: '0.0000'})
    if ho is not None:
        add('센서스 기억', '기억 여부 시험(syn_s7_f0.5): 학습 절반과 보지 않은 절반의 유일 조합 일치율', ho,
            ['차이가 0에 가까우면 합성이 학습 기록을 외우지 않았다는 뜻이다(5장).'], {k: '0.00' for k in range(2, 7)})


# ------------------------------------------------------------------ validation ①②④⑤ and checks
def validation():
    R = load(V('val_results.pkl'))
    if R is None: return
    T = R['tvd']
    G = T.groupby(['자료', 'k']).agg(조합_수=('합성', 'size'), 계산된_수=('합성', 'count'), 합성=('합성', 'mean'), MatrAIx=('MatrAIx', 'mean'),
                                     표본오차_기준=('표본오차 기준', 'mean')).reset_index()
    G['합성 ÷ 기준'] = G.합성 / G.표본오차_기준; G['MatrAIx ÷ 기준'] = G.MatrAIx / G.표본오차_기준
    G['_o'] = G.자료.map({s: i for i, s in enumerate(SOURCES)}); G = G.sort_values(['_o', 'k']).drop(columns='_o')
    G.columns = [c.replace('_', ' ') for c in G.columns]
    add('① 분포 거리', '① 속성 2개·3개 조합의 평균 총변동거리(떼어 둔 실제 자료 대비)', G,
        ['합성 = 검증 표본(80% 모형). 표본오차 기준 = 합성이 참일 때 분리분 크기의 표본이 보이는 거리. MatrAIx = 1단계 재현 표본의 한국 Native 성인.',
         'MatrAIx에 없는 kr_* 속성이 든 조합은 MatrAIx 칸이 비어 있다. KGSS 3개 조합은 1,500개 무작위.'], {5: '0.000', 6: '0.000', 7: '0.000', 8: '0.00', 9: '0.00'})
    T.to_csv(OUT / 'csv' / 'tvd_all_subsets.csv', index=False, encoding='utf-8-sig')
    T2 = T.assign(초과=T.합성 - T['표본오차 기준']).sort_values('초과', ascending=False).groupby('자료').head(10)
    T2['_o'] = T2.자료.map({s: i for i, s in enumerate(SOURCES)}); T2 = T2.sort_values(['_o', '초과'], ascending=[True, False])
    add('① 큰 차이', '① 표본오차 기준을 가장 많이 넘는 조합(자료별 10개)', T2[['자료', '속성', '합성', 'MatrAIx', '표본오차 기준', '초과', '조사 응답 수']],
        [], {3: '0.000', 4: '0.000', 5: '0.000', 6: '0.000'})
    PM, PX = R['pmse'], load(V('pmse_matraix_subset.pkl')) or {}
    rows = []
    for s in SOURCES:
        p = PM.get(s, {}); q = PX.get(s)
        rows.append((s, '전체 비교 속성', p.get('n'), p.get('m'), p.get('synthetic'), p.get('baseline'), p.get('MatrAIx')))
        if q: rows.append((s, f"MatrAIx에도 있는 속성 {len(q['attrs'])}개", q.get('n'), q.get('m'), q.get('synthetic'), q.get('baseline'), q.get('MatrAIx')))
    add('② 구분 가능성', '② pMSE(0에 가까울수록 실제와 구분되지 않음)', pd.DataFrame(rows, columns=['자료', '속성', '분리분 완전 응답 수', '비교 표본 크기(각)', '합성', '기준: 분리분 반 대 반', 'MatrAIx']),
        ['결정나무(잎 최소 20, 깊이 8), 5겹 교차 예측. 두 집단을 같은 크기로 섞는다.'], {5: '0.0000', 6: '0.0000', 7: '0.0000'})
    add('④ 공식 통계', '④ 공식 통계와 비교', R['official'], ['합성 = 검증 표본. 공식 기준값의 출처는 docs/09_검증.md.'], {2: '0.0%', 3: '0.0%', 4: '0.0%'})
    rr = []
    for f, sheet in [('render_check_val.pkl', '검증 표본'), ('render_check_full.pkl', '최종 표본')]:
        rc = load(V(f))
        if rc is None: continue
        d = rc['R']
        rr.append((sheet, len(d), d.n_attr.mean(), d.kr_missing.mean() / d.n_attr.mean(), d.mx_missing.mean(), d.mx_missing.min(), d.mx_missing.max(),
                   d.mx_missing.mean() / d.n_attr.mean(), rc['template_ok'], rc['ref_chars']))
    if rr:
        add('⑤ 전달 누락', '⑤ 프롬프트 전달 누락(각 표본의 처음 5,000명)', pd.DataFrame(rr, columns=['표본', '인원', '1인 속성 수', '한국 렌더러 누락 비율', 'MatrAIx 렌더러 누락 수(평균)',
            'MatrAIx 렌더러 누락 수(최소)', 'MatrAIx 렌더러 누락 수(최대)', 'MatrAIx 렌더러 누락 비율', '템플릿 줄 점검 통과 비율', '예시 프롬프트 글자 수']),
            ['같은 페르소나를 MatrAIx 렌더러로 바꾸면 기본값과 같은 값, none 류 값이 빠진다(7.10절).'], {4: '0.0%', 5: '0.0', 8: '0.0%', 9: '0.000'})
    en = load(V('english_check.pkl'))
    if en is not None: add('영어 능력', '영어 능력: KGSS 2018 분리분과 검증 표본(출생 연도별)', en, [], {k: '0.000' for k in range(3, 12)})
    vl = load(V('value_link_check.pkl'))
    if vl is not None:
        r1 = np.corrcoef(vl['분리분 상관'], vl['연결 후 생성 상관'])[0, 1]; r0 = np.corrcoef(vl['분리분 상관'], vl['연결 전 생성 상관'])[0, 1]
        add('가치 연결', '가치 → KGSS 태도 연결: 떼어 둔 KGSS 2021 응답자의 가치–태도 상관과 생성 표본의 상관', vl,
            [f'230개 (속성, 가치) 쌍. 분리분 상관과의 상관: 연결 후 {r1:.3f}, 연결 전(val_novl) {r0:.3f}.'], {4: '0.000', 5: '0.000', 6: '0.000'})
    ed = load(V('eaps_decomp.pkl'))
    if ed: add('노동 시점', '취업자 연령·학력: 합성(2020 구성), 2020 센서스 분리분, 2025 지역별고용조사 분리분',
               pd.concat([d.set_axis(['연령', '합성', '2020 센서스 분리분', '2025 지역별고용조사 분리분'], axis=1).assign(표=t) for d, t in [(ed['age'], '취업자 연령 분포'), (ed['ba'], '연령별 대졸 이상 비율')]])[['표', '연령', '합성', '2020 센서스 분리분', '2025 지역별고용조사 분리분']],
               ['페르소나는 2020년 인구 구성이라 2025년 취업자보다 젊고 학력이 낮다(선택 보정을 하지 않은 결과).'], {3: '0.000', 4: '0.000', 5: '0.000'})
    au = load(V('audit_out_on.pkl'))
    if au:
        D = au['kr'].copy(); D.insert(0, '구분', '한국 정체성·성격 점검'); D.insert(2, '설명', None)
        T = au['demo'].rename(columns={'id': '점검'})[['점검', '설명', 'MatrAIx 한국 Native 성인', '생성 표본']].copy(); T.insert(0, '구분', '1단계 페르소나 안 모순 점검')
        T.loc[len(T)] = ('1단계 페르소나 안 모순 점검', 'hard 하나 이상', None, au['any_hard'][0], au['any_hard'][1])
        add('진단 점검', '1단계 MatrAIx 진단 점검을 최종 표본에 적용(MatrAIx 한국 Native 성인 대 최종 표본)', pd.concat([D, T]),
            [f"MatrAIx {au['n'][0]:,}명, 최종 표본 {au['n'][1]:,}명. 상관 행은 비율이 아니라 상관계수다. 점검 정의는 docs/02_MatrAIx와의_관계.md."], {4: '0.0000', 5: '0.0000'})
    lc = load(V('language_check.pkl'))
    if lc:
        rows = []
        for key, lab in [('multilingualism', '다중언어 수준'), ('lifex_languages_lifetime', '평생 언어 수'), ('demo_home_language', '가정 언어'),
                         ('english_proficiency', '영어 능력'), ('other_languages_conversational', '회화 이상인 영어 외 외국어 수')]:
            vals = sorted(set().union(*[set(v[key].index) for v in lc.values()]), key=str)
            for x in vals: rows.append([lab, str(x)] + [float(v[key].get(x, 0)) for v in lc.values()])
        rows.append(['회화 이상인 영어 외 외국어 수', '평균(개)'] + [v['other_languages_mean'] for v in lc.values()])
        add('언어', '언어 속성 분포(3차 보완 전후)', pd.DataFrame(rows, columns=['속성', '값'] + list(lc)),
            ['3차 수정의 규칙에서 만든 금지 규칙이 특정 언어를 강제한 결함과 그 보완(8장).'], {k: '0.000' for k in range(3, 3 + len(lc))})


# ------------------------------------------------------------------ ③ rules
def rules():
    parts = []
    for k in (4, 3, 2, 1):
        RS = load(V(f'rules_set{k}.pkl'))
        if RS is None: continue
        S = RS['summary'].set_index('대상').T.reset_index().rename(columns={'index': '지표'})
        parts.append(S.assign(규칙=f'세트 {k}'))
        T = RS['table']; rate = [c for c in T.columns if c.startswith(('위반율: ', '적용 대상: '))]
        T[['id', 'group', 'severity', '규칙 종류', 'description'] + rate].to_csv(OUT / 'csv' / f'v3_rules_set{k}_by_rule_all_samples.csv', index=False, encoding='utf-8-sig')
        if k == 4:
            lab = [c for c in T.columns if c.startswith('위반율: ')]
            T4 = T[['id', 'group', 'severity', '규칙 종류', 'description'] + lab].copy(); T4.columns = [c.replace('위반율: ', '') for c in T4.columns]
            main = [c for c in T4.columns if '보완 후 검증' in c]
            if main: T4 = T4.sort_values(['severity', main[0]], ascending=[True, False])
            add('③ 세트 4 규칙별', '③ 세트 4: 규칙별 위반율(적용 대상 중)', T4, ['세트 1~3의 규칙별 값과 적용 대상 수는 csv/v3_rules_setN_by_rule_all_samples.csv.'], {c: '0.00%' for c in range(6, 6 + len(lab))})
    if parts:
        D = pd.concat(parts); cols = ['규칙', '지표'] + [c for c in D.columns if c not in ('규칙', '지표')]
        D = D[cols]
        fm = {i: '0.00%' for i in range(3, len(cols) + 1)}
        add('③ 모순율', '③ 독립 규칙 세트로 잰 모순율(위반이 하나 이상인 페르소나 비율)', D,
            ['세트 1~3은 생성기 수정에 썼고, 세트 4는 측정용으로 썼다가 마지막에 규칙 3개를 수정에 썼다. 각 세트의 독립 측정값은 그 세트로 고치기 전 표본의 값이다(8장).',
             '"규칙 수" 행은 비율이 아니라 개수다.'], fm)
    g = load(V('rules4_groups.pkl'))
    if g is not None:
        g = g[~g['규칙 묶음'].str.contains('제외')]
        add('③ 세트 4 묶음별', '③ 세트 4: 규칙 묶음별 hard 위반 비율', g, [], {i: '0.00%' for i in range(2, g.shape[1] + 1)})


# ------------------------------------------------------------------ summary
def summary():
    rows = []
    for s, lab in SAMPLES[:1] + SAMPLES[2:3]:
        p = paths.art('samples', s, 'stats.json')
        if p.exists():
            st = json.load(open(p)); rows.append(('생성', f"{lab} {st['N']:,}명: 거부 검사 통과율", st['acceptance'], None, None))
    R = load(V('val_results.pkl'))
    if R is not None:
        T = R['tvd']; T2 = T[T.k == 2].groupby('자료')[['합성', 'MatrAIx', '표본오차 기준']].mean()
        for s in SOURCES:
            if s in T2.index: rows.append(('① 2개 조합 평균 총변동거리', s, T2.loc[s, '합성'], T2.loc[s, 'MatrAIx'], T2.loc[s, '표본오차 기준']))
        for s in SOURCES:
            p = R['pmse'].get(s)
            if p: rows.append(('② pMSE', s, p['synthetic'], p.get('MatrAIx'), p['baseline']))
    for k in (4, 3, 2, 1):
        RS = load(V(f'rules_set{k}.pkl'))
        if RS is None: continue
        S = RS['summary'].set_index('대상'); main = [l for l in S.index if '보완 후 검증' in l]; fin = [l for l in S.index if '보완 후 최종' in l]
        for c, nm in [(HC, 'hard 하나 이상'), (KRC, '한국 속성 관련'), (MXC, 'MatrAIx 속성만')]:
            rows.append((f'③ 세트 {k}', nm, float(S.loc[main[0], c]) if main else None, float(S.loc[MX, c]) if MX in S.index else None, float(S.loc[fin[0], c]) if fin else None))
    for f in ['render_check_val.pkl']:
        rc = load(V(f))
        if rc: d = rc['R']; rows.append(('⑤ 전달 누락 비율', '한국 렌더러(합성) / MatrAIx 렌더러(같은 페르소나)', d.kr_missing.mean() / d.n_attr.mean(), d.mx_missing.mean() / d.n_attr.mean(), None))
    D = pd.DataFrame(rows, columns=['구분', '항목', 'kr-persona', 'MatrAIx', '기준값 또는 최종 표본'])
    TABLES.insert(0, ('요약', '요약', ['kr-persona = 검증 표본(80% 모형, 2만 명). MatrAIx = 1단계 재현 표본의 한국 문화 Native 성인 37,237명.',
                                       '마지막 열: ①은 표본오차 기준, ②는 분리분 반 대 반, ③은 최종 표본(100% 모형)의 값.'], D, {3: '0.0000', 4: '0.0000', 5: '0.0000'}))


ORDER = ['요약', '생성', '① 분포 거리', '① 큰 차이', '② 구분 가능성', '③ 모순율', '③ 세트 4 규칙별', '③ 세트 4 묶음별', '④ 공식 통계', '⑤ 전달 누락',
         '진단 점검', '언어', '영어 능력', '가치 연결', '노동 시점', '센서스', '센서스 합성', '센서스 점검', '센서스 논리', '센서스 구분', '센서스 노출', '센서스 기억']
EXCEL = {'요약', '생성', '① 분포 거리', '② 구분 가능성', '③ 모순율', '③ 세트 4 규칙별', '④ 공식 통계', '⑤ 전달 누락', '언어', '영어 능력', '노동 시점', '센서스'}   # the rest: CSV only


def write():
    TABLES.sort(key=lambda t: ORDER.index(t[0]))
    wb = openpyxl.Workbook(); wb.remove(wb.active); bold = Font(bold=True)
    for name, title, notes, df, fmt in TABLES:
        df.to_csv(OUT / 'csv' / f'{CSV[name]}.csv', index=False, encoding='utf-8-sig')
        if name not in EXCEL: continue
        ws = wb.create_sheet(name[:31]); ws['A1'] = title; ws['A1'].font = bold
        for i, t in enumerate(notes, 2): ws.cell(row=i, column=1, value=t)
        h = len(notes) + 3
        for j, c in enumerate(df.columns, 1):
            x = ws.cell(row=h, column=j, value=str(c)); x.font = bold; x.alignment = Alignment(wrap_text=True, vertical='top')
        for i, r in enumerate(df.itertuples(index=False), h + 1):
            for j, v in enumerate(r, 1):
                if isinstance(v, (float, np.floating)) and np.isnan(v): v = None
                elif isinstance(v, np.integer): v = int(v)
                elif isinstance(v, np.floating): v = float(v)
                elif isinstance(v, (tuple, list)): v = ' × '.join(map(str, v))
                x = ws.cell(row=i, column=j, value=v)
                if j in fmt and isinstance(v, float): x.number_format = fmt[j]
        for j, c in enumerate(df.columns, 1):
            w = max([len(str(c))] + [len(str(v)) for v in df.iloc[:200, j - 1].tolist()])
            ws.column_dimensions[get_column_letter(j)].width = min(max(8, w * 1.1 + 2), 70)
        ws.freeze_panes = ws.cell(row=h + 1, column=1)
    wb.save(OUT / '검증_결과.xlsx')
    print('sheets', [t[0] for t in TABLES])


if __name__ == '__main__':
    (OUT / 'csv').mkdir(parents=True, exist_ok=True)
    generation(); census(); validation(); rules(); summary(); write()
