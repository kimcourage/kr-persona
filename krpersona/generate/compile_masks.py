"""step 5 round 3: compile the pairwise parts of the fixing rule sets (hard rules of sets 1-3) into generation-time masks.
For every rule and every pair of columns it reads (at least one a MatrAIx (5) node), all value pairs are tried on a baseline row where the rule
does not fire; a value pair is forbidden when it makes the rule fire and neither value fires with every value of the other column (those fire because
of a third column or on their own, and are left to the re-draw step). Baselines per rule (union): a 'neutral' row and three rows of the baseline
sample where the rule does not fire. The neutral row sets each column to the value with the lowest firing rate when imposed on MatrAIx
Korean-native rows, half where the rule fires and half where it does not (up to 300 each: values that stop it and do not start it), so that 'any member of a family' rules do not already fire through another member;
if the rule never fires there, a greedy search is used (each column to the value leaving the fewest one-column changes that make the rule fire).
output: rule_masks.pkl = {(target, condition): forbidden matrix (len(V[cond]) + 1, len(V[target]))}; the last row is the "no category" sentinel.
usage: python -m krpersona.generate.compile_masks <baseline personas.parquet>   (the round-3 masks used artifacts/samples/test5/personas.parquet)"""
import sys, itertools, pickle, numpy as np, pandas as pd
from krpersona import paths
from krpersona.generate import generator as K4
from krpersona.generate.assign import build, SCHWARTZ
from krpersona.matraix.graph import graph_values
from krpersona.blocks import values as VM
from krpersona.rules import set1 as R1, set2 as R2, set3 as R3
from krpersona.validate import sources as SRC

# Attributes about languages. Rules on them count languages ("at least two at Conversational"); compiled pair by pair on one baseline row they
# force a particular language (round 3 forced French: Monolingual 0%, French Conversational+ 83%), so they are not compiled.
LANG = lambda c: c.startswith('lang_') or c in ('multilingualism', 'lifex_languages_lifetime', 'demo_home_language', 'primary_language', 'english_proficiency')


def fires(M, r, rows): return M.violation_matrix(rows, [r])[0][r['id']].to_numpy()

def neutral(M, r, row, cols, V, passes=2):
    """greedy: each column to the value with the fewest one-column changes that make the rule fire (the row itself must not fire)"""
    row = row.copy()
    def singles(cands):   # for each candidate row, every one-column change
        out = []
        for c in cols:
            x = pd.DataFrame(np.repeat(cands.to_numpy(), len(V[c]), 0), columns=cols); x[c] = np.tile(V[c], len(cands)); out.append(x)
        return pd.concat(out, ignore_index=True), sum(len(V[c]) for c in cols)
    for _ in range(passes):
        for c in cols:
            cand = pd.DataFrame(np.repeat(row.to_numpy(), len(V[c]), 0), columns=cols); cand[c] = V[c]
            f0 = fires(M, r, cand); S, k = singles(cand)
            hs = fires(M, r, S)
            off = 0; pot = np.zeros(len(cand))
            for c2 in cols:
                n2 = len(V[c2]); blk = hs[off: off + len(cand) * n2].reshape(len(cand), n2); pot += blk.sum(1); off += len(cand) * n2
            pot[f0] = np.inf
            row[c] = V[c][int(np.argmin(pot))]
    return row

def neutral_mx(M, r, row, cols, V, MXF):
    """each column to the value with the lowest firing rate when imposed on firing MatrAIx rows MXF"""
    row = row.copy()
    for c in cols:
        x = pd.concat([MXF.assign(**{c: v}) for v in V[c]], ignore_index=True)
        row[c] = V[c][int(np.argmin(fires(M, r, x).reshape(len(V[c]), len(MXF)).mean(1)))]
    return row

def used_cols(M, r, one):
    used = set(); orig = M._s
    def spy(d, col): used.add(col); return orig(d, col)
    M._s = spy
    try: r['scope'](one); r['hit'](one)
    except Exception: pass
    M._s = orig; return sorted(used)

if __name__ == '__main__':
    V = graph_values(); K = set(build()[1]) | set(SCHWARTZ) | set(VM.VAL_STAR) | set(K4.KR_ADDED)
    P = pd.read_parquet(sys.argv[1]); cols_all = [c for c in P.columns if c != 'persona_id']
    one = pd.DataFrame({c: [str(P[c].iloc[0])] for c in cols_all})
    MX = SRC.matraix_adults(cols_all)
    F = {}; nrules = 0; npairs = 0
    for M, skip in [(R1, 'temporal_2020'), (R2, 'population'), (R3, 'population')]:
        for r in M.RULES:
            if r['severity'] != 'hard' or r['group'] == skip: continue
            cols = [c for c in used_cols(M, r, one) if c in V]
            if len(cols) < 2 or all(c in K for c in cols): continue
            H = M.violation_matrix(P[cols], [r])[0][r['id']].to_numpy()
            ok = np.flatnonzero(~H)
            if not len(ok): continue
            bases = [P[cols].iloc[[ok[i]]].astype(str).reset_index(drop=True) for i in sorted({0, len(ok) // 2, len(ok) - 1})]
            hm = fires(M, r, MX[cols].astype(str)) if all(c in MX for c in cols) else np.zeros(0, bool)
            if hm.sum() >= 20:
                k_ = min(300, int(hm.sum())); mxf = pd.concat([MX[cols][hm].sample(k_, random_state=0), MX[cols][~hm].sample(k_, random_state=0)]).astype(str).reset_index(drop=True)
                neu = neutral_mx(M, r, bases[0], cols, V, mxf)
            else: neu = neutral(M, r, bases[0], cols, V)
            if not fires(M, r, neu)[0]: bases.insert(0, neu)
            nrules += 1
            if nrules % 20 == 0: print('rules', nrules, flush=True)
            fam = lambda c: next((f for f in ('cult_', 'lang_', 'hob_', 'sport_', 'prog_', 'tool_', 'fam_', 'filmg_', 'musg_', 'cuis_', 'topic_', 'acad_', 'bookg_', 'ai_task_') if c.startswith(f)), c)
            big = {f for f in map(fam, cols) if sum(fam(c) == f for c in cols) > 12}
            for (c1, c2), base in itertools.product(itertools.combinations(cols, 2), bases):
                if c1 in K and c2 in K: continue
                if LANG(c1) or LANG(c2): continue   # language counts: handled by masks.dyn_language_masks and generator._multi, not pairwise
                if fam(c1) == fam(c2) and fam(c1) in big: continue   # members of one large family (cult_*, hob_*, lang_* ...) are not constrained pairwise
                v1, v2 = V[c1], V[c2]
                rows = pd.DataFrame(np.repeat(base.to_numpy(), len(v1) * len(v2), 0), columns=cols)
                rows[c1] = np.repeat(v1, len(v2)); rows[c2] = np.tile(v2, len(v1))
                h = M.violation_matrix(rows, [r])[0][r['id']].to_numpy().reshape(len(v1), len(v2))
                if not h.any(): continue
                alone1 = h.all(1); alone2 = h.all(0)   # fires whatever the other column is: not a constraint between these two columns
                hh = h & ~alone1[:, None] & ~alone2[None, :]
                if not hh.any(): continue
                npairs += 1
                for t, c, mat in [(c2, c1, hh), (c1, c2, hh.T)]:
                    if t in K: continue
                    m = F.setdefault((t, c), np.zeros((len(V[c]) + 1, len(V[t])), bool)); m[:-1] |= mat
    pickle.dump(F, open(paths.art('masks', 'rule_masks.pkl', mkdir=True), 'wb'))
    print('rules used', nrules, '(column pair, baseline) hits with forbidden value pairs', npairs, 'masks (target, condition)', len(F), 'forbidden cells', int(sum(m.sum() for m in F.values())))
