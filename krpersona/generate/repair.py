"""Re-drawing of MatrAIx attributes for personas that break a hard rule of the fixing rule sets (sets 1-3)."""
import numpy as np, pandas as pd
from krpersona.rules import set1 as R1, set2 as R2, set3 as R3


class RepairRules:
    """hard rules of the independent rule sets used for fixing (set 1 without the 2020 product rules, sets 2 and 3 without the population-definition
    rules), keeping only rules that read at least one MatrAIx (5) column: violations of those are re-drawn (MatrAIx part only, Korean values fixed)"""
    def __init__(self, V, K, cols):
        one = pd.DataFrame({c: [V[c][0] if c in V else 'x'] for c in cols})
        self.sets = []
        for M, skip in [(R1, 'temporal_2020'), (R2, 'population'), (R3, 'population')]:
            rules = []
            for r in M.RULES:
                if r['severity'] != 'hard' or r['group'] == skip: continue
                used = set(); orig = M._s
                def spy(d, col, used=used, orig=orig): used.add(col); return orig(d, col)
                M._s = spy
                try: r['scope'](one); r['hit'](one)
                except Exception: pass
                M._s = orig
                if used and any(c not in K for c in used): rules.append(r)
            self.sets.append((M, rules))
        self.n_rules = [len(r) for _, r in self.sets]

    def bad(self, F):
        b = np.zeros(len(F), bool)
        for M, rules in self.sets:
            H, S, st = M.violation_matrix(F, rules); b |= H.any(axis=1).to_numpy()
        return b
