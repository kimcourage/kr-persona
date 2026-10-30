"""Contradiction rates by rule set, severity and attribute group.

Usage: python -m krpersona.validate.rules_summary <1|2|3|4> <out.pkl> <label=personas.parquet> [...]
Rule set 1 excludes temporal_2020 because technology attributes describe present-day use.
All four sets informed generator changes; see docs/08_모순_규칙.md for evaluation independence.
"""
import sys, pickle, importlib, numpy as np, pandas as pd
from krpersona.validate import sources as SRC
from krpersona.validate.rule_cols import k_set
SET = sys.argv[1]; R2 = importlib.import_module(f'krpersona.rules.set{int(SET)}')
if SET == '1': R2.RULES[:] = [r for r in R2.RULES if r['group'] != 'temporal_2020']

def referenced(df):
    used = {}; orig = R2._s; cur = {'set': None}
    def spy(d, col):
        if cur['set'] is not None: cur['set'].add(col)
        return orig(d, col)
    R2._s = spy
    for r in R2.RULES:
        cur['set'] = set()
        try: r['scope'](df); r['hit'](df)
        except Exception: pass
        used[r['id']] = sorted(cur['set'])
    R2._s = orig; return used

GRP = {}

def summarize(df, kind, sev):
    H, S, st = R2.violation_matrix(df)
    ran = [c for c in H.columns if st.get(c) == 'ok']
    hard = [c for c in ran if sev[c] == 'hard']; soft = [c for c in ran if sev[c] == 'soft']
    row = {'n': len(df), 'hard 하나 이상(전체)': float(H[hard].any(axis=1).mean()), 'soft 하나 이상': float(H[soft].any(axis=1).mean())}
    pop = [c for c in hard if GRP.get(c) == 'population']
    if pop:   # set 2: rules that check the population definition (age 19+, citizen, born in Korea ...) rather than two attributes
        row['hard 하나 이상(인구 정의 규칙 제외)'] = float(H[[c for c in hard if c not in pop]].any(axis=1).mean())
        row['hard 하나 이상: 인구 정의 규칙'] = float(H[pop].any(axis=1).mean())
    kr = [c for c in hard if kind.get(c) in ('한국 속성만', '섞임')]
    row['hard 하나 이상: 한국 속성 관련 규칙(한국 속성만 + 섞임)'] = float(H[kr].any(axis=1).mean()) if kr else np.nan
    for k in ['한국 속성만', '섞임', 'MatrAIx 속성만']:
        cs = [c for c in hard if kind.get(c) == k]; row[f'hard 하나 이상: {k} 규칙'] = float(H[cs].any(axis=1).mean()) if cs else np.nan; row[f'{k} hard 규칙 수(적용된 것)'] = len(cs)
    rate = {c: (float(H[c].sum() / S[c].sum()) if S[c].sum() else np.nan) for c in H.columns}
    nsc = {c: int(S[c].sum()) for c in H.columns}
    return row, rate, nsc, st

if __name__ == '__main__':
    out_path = sys.argv[2]; targets = [a.split('=', 1) for a in sys.argv[3:]]
    P = {lab: pd.read_parquet(path) for lab, path in targets}
    cols = [c for c in next(iter(P.values())).columns if c != 'persona_id']
    used = referenced(next(iter(P.values())).iloc[:500]); K = k_set()
    kind = {}
    for rid, cs in used.items():
        ink = [c for c in cs if c in K or c.startswith('kr_')]; outk = [c for c in cs if c not in ink]
        kind[rid] = '한국 속성만' if cs and not outk else ('MatrAIx 속성만' if cs and not ink else ('섞임' if cs else '열 없음'))
    sev = {r['id']: r['severity'] for r in R2.RULES}; GRP.update({r['id']: r['group'] for r in R2.RULES})
    P['MatrAIx 한국 Native 성인'] = SRC.matraix_adults(cols)
    rows, T = [], pd.DataFrame([(r['id'], r['group'], r['severity'], kind[r['id']], r['description'], ', '.join(used[r['id']][:8])) for r in R2.RULES],
                               columns=['id', 'group', 'severity', '규칙 종류', 'description', '읽는 열'])
    for lab, df in list(P.items()):
        row, rate, nsc, st = summarize(df, kind, sev); rows.append(dict(대상=lab, **row))
        P[lab] = None   # free memory; rules read every column through _s (astype(str)), so the frame is not copied to strings
        T[f'적용 대상: {lab}'] = T.id.map(nsc); T[f'위반율: {lab}'] = T.id.map(rate)
        T[f'상태: {lab}'] = T.id.map(st)
    Sm = pd.DataFrame(rows)
    pickle.dump(dict(summary=Sm, table=T, kind=kind, used=used, n_rules=len(R2.RULES), n_hard=sum(v == 'hard' for v in sev.values())), open(out_path, 'wb'))
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); print(Sm.T)
    for lab in P:
        x = T[(T.severity == 'hard') & (T[f'위반율: {lab}'] > 0)].sort_values(f'위반율: {lab}', ascending=False)
        print('==', lab); print(x[['id', '규칙 종류', f'적용 대상: {lab}', f'위반율: {lab}', 'description']].head(15).to_string(index=False))
