"""Tables for the documentation (docs/tables/*.csv, UTF-8 with BOM so that Excel opens them correctly).

attributes.csv          every published attribute: MatrAIx category, how it is made, source, output values
derived_attributes.csv  MatrAIx attributes made from the Korean personality scores
nondev_attributes.csv   developer-only attributes and their value for people who are not developers
masks_static.csv        generation-time masks (value of one attribute forbidden given another)
masks_compiled.csv      masks compiled from rule sets 1-3 (artifacts/masks/rule_masks.pkl)
matraix_masks.csv       MatrAIx masks whose target is a Korean attribute: kept as a rejection check or dropped
rules_set{1-4}.csv      independent consistency rules: group, severity, kind (by the columns read), description, columns read
category_mapping.csv    sheet '범주 대응' of the mapping workbook (raw value -> output category, conditioning MatrAIx category)
new_korean_attributes.csv  sheet '새 한국 속성' (adopted and excluded candidates with reasons)
kgss_block.csv          structure of the KGSS block (artifacts/mapping/kgss_analysis.pkl: window, years, age treatment, parents)
block_a_*.csv, block_b_*.csv  block model estimates: krpersona/report/tables_blocks_a.py and tables_blocks_b.py (run by scripts/08_results.sh)
usage: python -m krpersona.report.doc_tables
"""
import json, pickle, inspect
import numpy as np, pandas as pd
from krpersona import paths
from krpersona.matraix.graph import graph_values
from krpersona.generate.assign import build, SCHWARTZ
from krpersona.generate import masks as MK
from krpersona.generate.generator import KR_ADDED, DERIVED, Gen4
from krpersona.generate.base import Gen0
from krpersona.blocks import values as VM

OUT = paths.ROOT / 'docs' / 'tables'


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    print(name, df.shape)


def K_set():
    rows, K0, PH, ids, emit = build()
    return rows, set(K0) | set(SCHWARTZ) | set(VM.VAL_STAR) | set(KR_ADDED)


def attributes():
    rows, K = K_set()
    dims = json.load(open(paths.ROOT / 'krpersona' / 'render' / 'dimensions_kr.json'))['dimensions']
    V = graph_values()
    der = {t: (src, sign, r) for t, src, sign, r, _ in DERIVED}
    val, na = MK.nondev_plan(V)
    kg = pickle.load(open(paths.art('models', 'kgss_models.pkl'), 'rb')) if paths.art('models', 'kgss_models.pkl').exists() else None
    kgss_attrs = set(kg['order']) | {'english_proficiency'} if kg else set()
    n_static = pd.Series([t for t, _, _ in MK.conflict_masks(V, K)]).value_counts()
    cm = paths.art('masks', 'rule_masks.pkl')
    n_comp = pd.Series([t for t, _ in pickle.load(open(cm, 'rb'))]).value_counts() if cm.exists() else pd.Series(dtype=int)
    obs = None
    s = paths.art('samples', 'out_on', 'personas.parquet')
    if s.exists():
        obs = pd.read_parquet(s)
    out = []
    for d in dims:
        a = d['id']; new = a.startswith('kr_')
        how, src, note = '', '', []
        if a in der:
            sc, sign, r = der[a]; how = '(3) 성격 점수에서 파생'; src = f"{sc} ({'+' if sign > 0 else '-'}, r = {r})"
        elif a == 'english_proficiency':
            how = '(4) 조사 추정'; src = 'KGSS 2018 영어 능력 문항(ENGPROF1-3)'
        elif a == 'lang_english':
            how = '(4) 조사 추정'; src = 'english_proficiency와 같은 수준으로 변환'
        elif a == 'dev_professional_status':
            how = '(4) 조사 추정'; src = '지역별고용조사 2025 상반기 B형 직업 소분류'
        elif a in SCHWARTZ:
            how = '(3) 잠재 점수: 가치'; src = 'KGSS 2021 PVQ 문항'
        elif a in VM.VAL_STAR:
            how = '(3) 잠재 점수: 가치 문항'; src = 'KGSS 2021 PVQ 문항(가치 잠재 점수에서)'
        elif new and a in ('kr_depression_level', 'kr_anxiety_level'):
            how = '(4) 조사 추정'; src = '국민건강영양조사 2022(PHQ-9, GAD-7)'
        elif new or a in kgss_attrs and a not in rows:
            how = '(4) 조사 추정'; src = '한국종합사회조사(KGSS)'
        elif a in rows:
            how, src = rows[a][0], rows[a][1]
        else:
            how = '(5) MatrAIx'; src = 'MatrAIx 그래프'
        if a in val: note.append(f"개발자가 아니면 '{val[a]}'")
        if a in na: note.append("개발자가 아니면 'Not applicable'")
        if a == 'multilingualism': note.append('언어별 능력에서 계산(회화 이상 언어 수)')
        if a in n_static.index: note.append(f'정적 금지 규칙 {int(n_static[a])}개')
        if a in n_comp.index: note.append(f'규칙에서 만든 금지 규칙 {int(n_comp[a])}개')
        vals = list(d.get('values') or [])
        if obs is not None and a in obs.columns:
            extra = [v for v in pd.unique(obs[a].astype(object).dropna()) if v not in vals]
            if extra: note.append('MatrAIx 범주 밖 출력 값: ' + ' | '.join(map(str, extra[:12])))
        out.append((d.get('index'), a, d.get('label'), d.get('category'), '새 한국 속성' if new else 'MatrAIx 속성', how, src, ' | '.join(map(str, vals)), '; '.join(note)))
    save(pd.DataFrame(out, columns=['MatrAIx 순번', 'id', '이름', 'MatrAIx 분류', '구분', '생성 방식', '자료 또는 근거', '값 목록', '비고']), 'attributes.csv')


def derived():
    rows = [(t, src, '+' if sign > 0 else '-', r, ', '.join(f"{a} ({'+' if s > 0 else '-'})" for a, s in anc)) for t, src, sign, r, anc in DERIVED]
    save(pd.DataFrame(rows, columns=['속성', '원 점수', '방향', 'r(가정)', '정반대 극단 금지 기준(방향)']), 'derived_attributes.csv')


def nondev():
    val, na = MK.nondev_plan(graph_values())
    rows = [(t, v) for t, v in sorted(val.items())] + [(t, 'Not applicable (출력만 바꿈)') for t in sorted(na)]
    save(pd.DataFrame(rows, columns=['속성', "개발자가 아닌 사람의 값"]), 'nondev_attributes.csv')


def static_masks():
    rows, K = K_set(); V = graph_values()
    src = inspect.getsource(MK.conflict_masks)
    G = Gen0.__new__(Gen0)
    out = []
    for origin, ms in [('0단계 생성기(generate/base.py)', Gen0._extra_masks(G)), ('충돌 수정(generate/masks.py)', MK.conflict_masks(V, K))]:
        for t, cond, mult in ms:
            mult = np.asarray(mult); vt = V[t]
            forb = [v for v, m in zip(vt, mult) if m == 0]
            eff = ('허용: ' + ' | '.join(v for v, m in zip(vt, mult) if m > 0)) if len(forb) > len(vt) / 2 else ('금지: ' + ' | '.join(forb))
            out.append((origin, t, ' 그리고 '.join(f"{p} ∈ {{{', '.join(vs)}}}" for p, vs in cond.items()), eff))
    save(pd.DataFrame(out, columns=['출처', '대상 속성', '조건', '효과']), 'masks_static.csv')


def compiled_masks():
    p = paths.art('masks', 'rule_masks.pkl')
    if not p.exists(): return
    V = graph_values(); F = pickle.load(open(p, 'rb')); out = []
    for (t, c), m in F.items():
        for i, j in zip(*np.nonzero(m[:-1])):
            out.append((t, c, V[c][i], V[t][j]))
    save(pd.DataFrame(out, columns=['대상 속성', '조건 속성', '조건 값', '금지되는 대상 값']), 'masks_compiled.csv')


def matraix_masks():
    p = paths.art('masks', 'mask_class.pkl')
    if p.exists(): save(pd.read_pickle(p), 'matraix_masks.csv')


def rules():
    for k in (1, 2, 3, 4):
        p = paths.art('validation', f'rules_set{k}.pkl')
        if not p.exists(): continue
        R = pickle.load(open(p, 'rb')); T = R['table']
        D = T[['id', 'group', 'severity', '규칙 종류', 'description', '읽는 열']].rename(columns={'group': '묶음', 'severity': '강도', 'description': '설명'})
        save(D, f'rules_set{k}.csv')   # definitions only; violation rates are in results/csv


def mapping_sheets():
    import openpyxl
    wb = openpyxl.load_workbook(paths.MAPPING, read_only=True)
    for sheet, name in [('범주 대응', 'category_mapping.csv'), ('새 한국 속성', 'new_korean_attributes.csv')]:
        rows = list(wb[sheet].iter_rows(values_only=True)); head = rows[3]
        df = pd.DataFrame([r[:len(head)] for r in rows[4:] if any(v is not None for v in r)], columns=head)
        if sheet == '범주 대응': df[['블록', 'MatrAIx 속성', '자료', '원자료 변수']] = df[['블록', 'MatrAIx 속성', '자료', '원자료 변수']].ffill()
        save(df, name)
    p = paths.art('mapping', 'kgss_analysis.pkl')
    if p.exists(): save(pickle.load(open(p, 'rb'))['P'], 'kgss_block.csv')


if __name__ == '__main__':
    attributes(); derived(); nondev(); static_masks(); compiled_masks(); matraix_masks(); rules(); mapping_sheets()
