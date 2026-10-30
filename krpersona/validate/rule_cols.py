"""columns referenced by each independent rule (instrumented run) and violation rates by rule type"""
import sys, pickle, numpy as np, pandas as pd
from krpersona.rules import set1 as IR
def referenced(df):
    used = {}; orig = IR._s; cur = {'set': None}
    def spy(d, col):
        if cur['set'] is not None: cur['set'].add(col)
        return orig(d, col)
    IR._s = spy
    for r in IR.RULES:
        cur['set'] = set()
        try: r['scope'](df); r['hit'](df)
        except Exception: pass
        used[r['id']] = sorted(cur['set'])
    IR._s = orig; return used
def k_set():
    from krpersona.generate.assign import build, SCHWARTZ
    from krpersona.blocks import values as VM
    from krpersona.generate.generator import KR_ADDED   # step-5 Korean attributes (English, developer status, derived personality-like ones)
    rows, K0, PH, ids, emit = build(); return set(K0) | set(SCHWARTZ) | set(VM.VAL_STAR) | set(KR_ADDED)
if __name__ == '__main__':
    P = pd.read_parquet(sys.argv[1]).astype(object).astype(str); used = referenced(P.iloc[:500]); K = k_set()
    kind = {}
    for rid, cols in used.items():
        ink = [c for c in cols if c in K or c.startswith('kr_')]; out = [c for c in cols if c not in ink]
        kind[rid] = '한국 속성만' if cols and not out else ('MatrAIx 속성만' if cols and not ink else ('섞임' if cols else '열 없음'))
    pickle.dump(dict(used=used, kind=kind), open(sys.argv[2], 'wb'))
    print(pd.Series(kind).value_counts())
