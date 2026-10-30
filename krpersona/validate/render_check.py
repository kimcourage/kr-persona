"""Renderer checks: (1) no omission, (2) final prompt line count via the real template, (3) fixed example, (4) MatrAIx renderer omissions.
usage: python -m krpersona.validate.render_check <personas.parquet> <out.pkl>"""
import sys, importlib.util, pickle, numpy as np, pandas as pd
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from krpersona import paths
from krpersona.render import render as KR
MX = str(paths.matraix())
spec = importlib.util.spec_from_file_location('mx_catalog', f'{MX}/src/matraix/persona_dimension_catalog.py'); MC = importlib.util.module_from_spec(spec); spec.loader.exec_module(MC)
TPL = Path(f'{MX}/environment/agents/matraix/agents/persona/templates')

def dims_of(row):
    return {k: v for k, v in row.items() if k != 'persona_id' and v is not None and not (isinstance(v, float) and np.isnan(v))}

def prompt(d):
    env = Environment(loader=FileSystemLoader(TPL), undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    ctx = dict(schema_version=None, persona_id='x', version=None, display_name=None, summary=None, system_prompt=None, dimensions=d,
               demographics=None, psychology=None, communication=None, preferences=None, behavior=None)
    return env.get_template('persona_system.md.j2').render(**ctx, **KR.build_template_context_extras(d)).strip()

if __name__ == '__main__':
    P = pd.read_parquet(sys.argv[1]).astype(object); n = min(5000, len(P)); S = P.iloc[:n]
    rows = []
    for _, r in S.iterrows():
        d = dims_of(r.to_dict())
        g, used = KR.collect_dimension_items(d, return_ids=True)
        mx = MC.collect_dimension_items(d, catalog_path=f'{MX}/persona/schema/dimensions.json')
        mx_ids = {i for items in mx.values() for i, _, _ in items}
        rows.append(dict(n_attr=len(d), kr_used=len(used), kr_missing=len(set(d) - set(used)), mx_used=len(mx_ids) + 0, mx_missing=len([k for k in d if k not in mx_ids])))
    R = pd.DataFrame(rows)
    # (2) real template: "- label: value" lines in the final prompt = number of attributes
    chk2 = []
    for _, r in S.iloc[:200].iterrows():
        d = dims_of(r.to_dict()); t = prompt(d); chk2.append(sum(1 for l in t.splitlines() if l.startswith('- ')) == len(d))
    # (3) fixed example: a woman aged 45-54 working as a teacher (born 1966-1975; reference prompt in artifacts/validation/)
    ex = P[(P.age_bracket.astype(str) == '45-54') & (P.role_function.astype(str) == 'Teaching') & (P.gender_identity.astype(str) == 'Woman')]
    ref = prompt(dims_of(ex.iloc[0].to_dict())) if len(ex) else ''
    (Path(sys.argv[2]).parent / 'reference_teacher_45-54.md').write_text(ref, encoding='utf-8')
    # MatrAIx: which attributes go missing most often
    miss = {}
    for _, r in S.iloc[:2000].iterrows():
        d = dims_of(r.to_dict()); mx = MC.collect_dimension_items(d, catalog_path=f'{MX}/persona/schema/dimensions.json')
        ids = {i for items in mx.values() for i, _, _ in items}
        for k in d:
            if k not in ids: miss[k] = miss.get(k, 0) + 1
    top = pd.Series(miss).sort_values(ascending=False) / min(2000, n)
    out = dict(R=R, template_ok=float(np.mean(chk2)), ref_id=(ex.iloc[0].persona_id if len(ex) else None), ref_chars=len(ref), top_missing=top)
    pickle.dump(out, open(sys.argv[2], 'wb'))
    print('personas', n, 'attributes per persona', R.n_attr.mean(), '| Korean renderer missing', R.kr_missing.sum(), '| template line check', out['template_ok'])
    print('MatrAIx renderer: omitted per persona mean', round(R.mx_missing.mean(), 1), 'share', round((R.mx_missing / R.n_attr).mean(), 4), 'reference chars', len(ref))
    print(top.head(15).round(3).to_dict())
