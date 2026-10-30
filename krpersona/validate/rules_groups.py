"""share of personas with at least one hard violation of a rule set, by rule group (and MatrAIx)
usage: python rules_groups.py <set 2|3|4> <out.pkl> <label>=<personas.parquet> ..."""
import sys, pickle, pandas as pd
import importlib
from krpersona.validate import sources as SRC
R2 = importlib.import_module(f'krpersona.rules.set{int(sys.argv[1])}')
sev = {r['id']: r['severity'] for r in R2.RULES}; grp = {r['id']: r['group'] for r in R2.RULES}
out = {}
targets = [a.split('=', 1) for a in sys.argv[3:]]
cols = [c for c in pd.read_parquet(targets[0][1]).columns if c != 'persona_id']
for lab, df in [(l, pd.read_parquet(p)) for l, p in targets] + [('MatrAIx 한국 Native 성인', SRC.matraix_adults(cols))]:
    H, S, st = R2.violation_matrix(df); hard = [c for c in H.columns if sev[c] == 'hard' and st[c] == 'ok']
    G = {g: float(H[[c for c in hard if grp[c] == g]].any(axis=1).mean()) for g in sorted(set(grp.values())) if any(grp[c] == g for c in hard)}
    ex = [c for c in hard if grp[c] not in ('tech_tools', 'developer_ai')]
    G['(개발·AI·기술 도구 묶음 제외 전체)'] = float(H[ex].any(axis=1).mean()); G['(전체)'] = float(H[hard].any(axis=1).mean())
    out[lab] = G; del df, H, S
T = pd.DataFrame(out); T.index.name = '규칙 묶음'; T = T.reset_index()
pickle.dump(T, open(sys.argv[2], 'wb')); print(T.round(3).to_string(index=False))
