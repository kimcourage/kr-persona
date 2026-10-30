"""Generate Full-DAG synthetic personas with the official MatrAIx sampler,
apply the published quality-filter rules, and keep:
  - every row with cult_south_korea == "Native"  (KR-native subset)
  - a uniform random baseline of all rows        (global baseline)
Outputs uint8 code matrices + column/value metadata as .npz in artifacts/baseline/.
usage: python -m krpersona.baseline.gen [N total, default 1,000,000]
"""
import json, sys, time
import numpy as np
from krpersona import paths

REPO = paths.matraix()
OUT = paths.art('baseline')
N_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
BATCH = 100_000
BASELINE_PER_BATCH = 5_000
paths.use_matraix()
from persona.synthesis.sampler import PersonaForwardSampler, SamplingConfig, DEFAULT_GRAPH_PATH  # noqa

s = PersonaForwardSampler(DEFAULT_GRAPH_PATH, SamplingConfig(seed=20261014))
cols = list(s.emit_nodes)
vals = {c: s.values[c] for c in cols}
ci = {c: i for i, c in enumerate(cols)}

qf = json.load(open(REPO / "persona/post_process/quality_filter/contradictions.json"))["conditional_masks"]
rules = []
for m in qf:
    if m["target"] not in ci or any(p not in ci for p in m["condition"]):
        print("skip rule (field not emitted):", m["mask_id"]); continue
    conds = [(ci[p], np.array([vals[p].index(v) for v in allowed if v in vals[p]])) for p, allowed in m["condition"].items()]
    bad = np.array([vals[m["target"]].index(v) for v in m["bad_values"] if v in vals[m["target"]]])
    rules.append((m["mask_id"], ci[m["target"]], conds, bad))

kr_col = ci["cult_south_korea"]; native_code = vals["cult_south_korea"].index("Native")
rng = np.random.default_rng(7)
kr_parts, base_parts = [], []
stats = {"generated": 0, "rejected_by_quality_filter": 0, "rule_hits": {}}
t0 = time.time()
done = 0
while done < N_TOTAL:
    n = min(BATCH, N_TOTAL - done)
    idx = s.sample_indices(n)
    M = np.stack([idx[c].astype(np.uint8) for c in cols], axis=1)
    reject = np.zeros(n, dtype=bool)
    for mid, t, conds, bad in rules:
        hit = np.isin(M[:, t], bad)
        for p, allowed in conds:
            hit &= np.isin(M[:, p], allowed)
        stats["rule_hits"][mid] = stats["rule_hits"].get(mid, 0) + int(hit.sum())
        reject |= hit
    stats["generated"] += n
    stats["rejected_by_quality_filter"] += int(reject.sum())
    M = M[~reject]
    kr_parts.append(M[M[:, kr_col] == native_code])
    base_parts.append(M[rng.choice(len(M), size=BASELINE_PER_BATCH, replace=False)])
    done += n
    print(f"{done:,} rows  kr_native so far {sum(len(p) for p in kr_parts):,}  {time.time()-t0:.0f}s", flush=True)

KR = np.concatenate(kr_parts); BASE = np.concatenate(base_parts)
stats["kept_after_filter"] = stats["generated"] - stats["rejected_by_quality_filter"]
stats["kr_native_rows"] = int(len(KR)); stats["baseline_rows"] = int(len(BASE))
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / "codes.npz", kr=KR, base=BASE)
json.dump({"columns": cols, "values": vals}, open(OUT / "meta.json", "w"), ensure_ascii=False)
json.dump(stats, open(OUT / "gen_stats.json", "w"), indent=1)
print(json.dumps({k: v for k, v in stats.items() if k != "rule_hits"}, indent=1))
