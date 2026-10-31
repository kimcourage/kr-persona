"""Generate a persona sample: calibration of delta (dominant trait) and of acceptance by cell -> generation with rejection -> outputs.
usage: python -m krpersona.generate.run <N> <seed> <on|off> <sample name> [calibration batch size, default 50000]
on = MatrAIx attributes are sampled (conditioned on the Korean values); off = only the Korean attributes.
KR_HOLDOUT=1 uses the validation models (80% of every source) and the census base synthesized from 80% of households.
Outputs in artifacts/samples/<sample name>/: personas.parquet (published attributes), internal.parquet (internal values such as exact age and 시도,
not for publication), personas_matraix.parquet (MatrAIx unified dataset format), persona_codes.schema.json, stats.json."""
import argparse, json, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from krpersona import paths
from krpersona.generate.resources import check_inputs
from krpersona.blocks import core

parser = argparse.ArgumentParser(description='사전 추정 모형으로 한국 성인 페르소나를 생성합니다.')
parser.add_argument('count', type=int, help='생성 인원(양의 정수)')
parser.add_argument('seed', type=int, help='난수 시드(0 이상의 정수)')
parser.add_argument('mode', choices=['on', 'off'], help='on: 전체 속성, off: 한국 속성만')
parser.add_argument('name', help='artifacts/samples 아래 결과 폴더 이름')
parser.add_argument('calibration', type=int, nargs='?', default=50000, help='사전 보정 표본 수(기본 50000)')
args = parser.parse_args()
if args.count <= 0 or args.seed < 0 or args.calibration <= 0:
    parser.error('인원과 사전 보정 표본 수는 양수, 시드는 0 이상이어야 합니다.')
if args.name in {'.', '..'} or '/' in args.name or '\\' in args.name:
    parser.error('결과 이름에는 경로 구분자를 사용할 수 없습니다.')
check_inputs(core.HOLDOUT)
from krpersona.generate.generator import Gen4 as Gen0, FACT
from krpersona.matraix.own_masks import korean_masks, rules
N, SEED, ON = args.count, args.seed, args.mode == 'on'
OUT = paths.art('samples', args.name); OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time(); log = lambda *a: print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)
TAG = 'syn_s5_f0.8' if core.HOLDOUT else 'syn_s1'   # validation: census base synthesized from 80% of households
log('loading generation inputs')
G = Gen0(paths.inp('census', TAG + '_H.parquet'), paths.inp('census', TAG + '_P.parquet'), seed=SEED, switch_on=True)
cls = pd.read_pickle(paths.inp('masks', 'mask_class.pkl')); keep_ids = set(cls.mask_id[cls['처리'] == '모순 검사로 유지'])
KEPT = [m for m in korean_masks(G.K) if m['mask_id'] in keep_ids]
# ---- delta so that Balanced = 26% (MatrAIx prior), and acceptance by cell
G.repair_on = False   # calibration (delta, trial acceptance) without the step-5 re-draw
B = args.calibration; log('calibration', B); G.make(B); Z = G.Z; srt = np.sort(Z, 1)
def bal(d): return np.mean(~((srt[:, -1] >= 0.51) & (srt[:, -1] - srt[:, -2] >= d)))
lo, hi = 0.0, 2.0
for _ in range(40):
    mid = (lo + hi) / 2; lo, hi = (mid, hi) if bal(mid) < 0.26 else (lo, mid)
G.delta = round((lo + hi) / 2, 4); log('delta', G.delta, 'balanced', round(bal(G.delta), 4))
G.make(B); V = rules(G, KEPT); ok = ~V.any(axis=1).to_numpy()
acc = pd.Series(ok).groupby(G.D.cell.to_numpy()).agg(['mean', 'size']); a_all = ok.mean()
a_hat = np.where(acc['size'] >= 30, acc['mean'], a_all); a_hat = pd.Series(np.maximum(a_hat, 0.5), index=acc.index)
G.base.P['w'] = G.base.P.w0 / G.base.P.cell.map(a_hat).fillna(a_all)
log('trial acceptance', round(a_all, 5), 'cells', len(acc))
# ---- generation with rejection
G.on = ON; G.repair_on = True; keep, tried, viol, nacc = [], 0, None, 0; keep_int = []; rep = []
while sum(len(k) for k in keep) < N:
    m = min(25000, int((N - sum(len(k) for k in keep)) / max(a_all, 0.5) * 1.05) + 100)
    out = G.make(m); V = rules(G, KEPT if ON else [k for k in KEPT]); ok = ~V.any(axis=1).to_numpy(); tried += m
    viol = V.sum() if viol is None else viol + V.sum()
    df = pd.DataFrame({k: (v if isinstance(v, pd.Categorical) else pd.Categorical(v)) for k, v in out.items()}); df['_cell'] = G.D.cell.to_numpy()
    keep.append(df[ok]); nacc += int(ok.sum()); rep.append(G.repair_hist); log('batch', m, 'accepted', int(ok.sum()), 'repair (rows breaking a fixing rule per round)', G.repair_hist)
    inn = pd.DataFrame({'age': G.D.age.to_numpy(), 'sex': G.D.sex.to_numpy(), 'sido': G.D.sido.to_numpy(), 'k': G.D.k.to_numpy(), 'rel': G.D.rel.to_numpy(), 'hkind': G.D.hkind.to_numpy(), 'income': G.income, 'wage_band': G.wage_band, 'inc4': G.inc4,
                        'xi_d': G.xi[:, 0], 'xi_a': G.xi[:, 1], 'phq': G.phq, 'gad': G.gad, 'kid_src': G.kid_src, 'n_kids': [len(c) for c in G.kids],
                        **{'h_' + a: G.hpar[a] for a in ['chronic', 'general', 'smoking', 'alcohol', 'exercise', 'stress', 'xd', 'xa']},
                        **{'k_' + a: v for a, v in G.kgss_y.items()}, **{'e_' + d: G.eta[:, j] for j, d in enumerate(FACT)}, **{'v_' + str(j): G.vz[:, j] for j in range(10)}})
    keep_int.append(inn[ok])
P = pd.concat(keep, ignore_index=True).iloc[:N]; INT = pd.concat(keep_int, ignore_index=True).iloc[:N]
cells = P.pop('_cell')
KR_NEW = G.KR_NEW
P.insert(0, 'persona_id', [f'krp-{SEED}-{i:07d}' for i in range(len(P))])
P.to_parquet(OUT / 'personas.parquet'); INT.insert(0, 'persona_id', P.persona_id.to_numpy()); INT.to_parquet(OUT / 'internal.parquet')
# MatrAIx unified format (codes + out-of-category overrides) for MatrAIx tools
paths.use_matraix()
from persona.post_process.unified_dataset.schema import AttributeCodec, UNIFIED_SCHEMA
cols = list(G.emit)
schema = OUT / 'persona_codes.schema.json'; json.dump({'format': 'persona_codes', 'columns': [{'id': c, 'values': G.V[c]} for c in cols]}, open(schema, 'w'), ensure_ascii=False)
codec = AttributeCodec.from_codes_schema(schema)
enc = []; PX = P.reindex(columns=cols)
for lo_ in range(0, len(P), 20000):
    for r in PX.iloc[lo_:lo_ + 20000].astype(object).to_dict('records'):
        enc.append(codec.encode_mapping({k: v for k, v in r.items() if isinstance(v, str)}))
tab = pa.table({'source': ['kr_persona'] * len(P), 'source_row_index': pa.array(np.arange(len(P)), pa.uint64()), 'source_record_id': P.persona_id.tolist(),
                'attributes': pa.array([e[0] for e in enc], pa.binary(len(enc[0][0]))), 'null_bitmap': pa.array([e[1] for e in enc], UNIFIED_SCHEMA.field('null_bitmap').type),
                'attribute_overrides': pa.array([e[2] for e in enc], UNIFIED_SCHEMA.field('attribute_overrides').type),
                'has_description': [False] * len(P), 'descriptions': pa.array([None] * len(P), UNIFIED_SCHEMA.field('descriptions').type),
                'grounding': pa.array([None] * len(P), UNIFIED_SCHEMA.field('grounding').type), 'metadata_json': pa.array([None] * len(P), pa.string())}, schema=UNIFIED_SCHEMA)
pq.write_table(tab, OUT / 'personas_matraix.parquet', compression='zstd')
st = dict(N=len(P), tried=tried, acceptance=nacc / tried, delta=G.delta, trial_acceptance=float(a_all), switch_on=ON, seconds=round(time.time() - T0),
          violations={k: int(v) for k, v in viol.items() if v}, repair=rep, repair_rules=G.REP.n_rules, kept_masks=len(KEPT), reversed_edges=len(G.S.reversed), K=len(G.K), placeholders=len(G.PH))
json.dump(st, open(OUT / 'stats.json', 'w'), ensure_ascii=False, indent=1); log(json.dumps({k: v for k, v in st.items() if k != 'violations'}))
