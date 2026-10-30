"""MatrAIx graph sampler with Korean attributes fixed first.

- Korean nodes K are set by the Korean generator (fixed codes per row, or drawn from MatrAIx q inside an allowed set).
  A code equal to len(values) is the sentinel "no MatrAIx category" (out-of-category Korean value): every link from it is factor 1.
- Edges K -> K are not used to generate K; they enter only q of a K node drawn inside an allowed set (h^-1(k), Omega) or a stage-0 placeholder.
- Edges b -> a with b not in K and a in K are reversed: b is drawn later with factor P_e(x_a | v_b)^(gamma_b w_e).
- Masks whose target is in K are not used, except for stage-0 placeholder nodes drawn from MatrAIx q.
- CPTs whose target is in K: q of that node only, with not-yet-drawn parents averaged over their prior.
- Everything else (edges, CPTs, masks into non-K nodes) is used as in MatrAIx, with K values as inputs.
"""
import math
from collections import defaultdict
import numpy as np
from krpersona import paths

paths.use_matraix()
from persona.synthesis.sampler import PersonaForwardSampler, SamplingConfig, DEFAULT_GRAPH_PATH  # noqa: E402


class KRSampler:
    def __init__(self, K, placeholder=(), seed=0, extra_masks=(), graph_path=DEFAULT_GRAPH_PATH, dyn_masks=()):
        b = PersonaForwardSampler(graph_path, SamplingConfig(seed=seed, emit_only=False))
        self.b = b; self.values = b.values; self.vtoi = b.vtoi; self.prior = b.prior; self.logprior = b.logprior
        self.K = set(K); self.ph = set(placeholder); self.rng = np.random.default_rng(seed)
        self.emit = set(b.emit_nodes); self.k = {n: len(v) for n, v in self.values.items()}
        topo = [n for n in b.topological_order if n in b.nodes]
        seen = set(); self.topo = [n for n in topo if not (n in seen or seen.add(n))] + [n for n in b.nodes if n not in set(topo)]
        pad = lambda T: np.concatenate([T, np.zeros((T.shape[0], 1), np.float32)], axis=1)   # sentinel column
        self.in_mod = defaultdict(list)   # non-K target -> [(source, w, table (k_t, k_s+1))]
        self.in_K = defaultdict(list)     # K target -> [(source in K, w, table)]
        self.reversed = []
        for t, edges in b.in_edges.items():
            repl = b.replaced_parents.get(t, set())
            for e in edges:
                s, w, L = e['source'], e['weight'], e['logratio'].astype(np.float32)   # L: (k_s, k_t) = log P(t|s) - log P0(t)
                if s in repl: continue
                if t in self.K:
                    if s in self.K: self.in_K[t].append((s, w, pad(L.T)))
                    else: self.in_mod[s].append((t, w, pad(L))); self.reversed.append((s, t))   # log P_e(x_t | v_s) up to a constant
                else:
                    self.in_mod[t].append((s, w, pad(L.T)))
        self.cpt = defaultdict(list)
        for t, cpts in b.full_cpts.items():
            for c in cpts:
                lut = np.zeros((c['code_space'] + 1, self.k[t]), np.float32)
                for code, lr in c['lookup'].items(): lut[code] = lr
                self.cpt[t].append(dict(parents=c['parents'], mult=c['multipliers'], space=c['code_space'], w=c['weight'], lut=lut))
        self.masks = defaultdict(list)
        for t, ms in b.masks.items():
            for m in ms: self.masks[t].append(([(p, a) for p, a in m['condition']], m['value_mult'].astype(np.float32)))
        for t, cond, mult in extra_masks:
            self.masks[t].append(([(p, np.array([self.vtoi[p][v] for v in vs], np.int64)) for p, vs in cond.items()], np.asarray(mult, np.float32)))
        # row-wise masks computed from what is already drawn: fn(codes, avail) -> (k_t, n) multipliers or None (e.g. counts over many nodes)
        self.dyn = defaultdict(list)
        for t, fn in dyn_masks: self.dyn[t].append(fn)
        # gamma: K nodes keep the MatrAIx value; other nodes are recomputed over the links they now receive (incl. reversed)
        self.gamma = {}
        for n in b.nodes:
            if n in self.K: self.gamma[n] = b.gamma[n]; continue
            ws = [c['w'] for c in self.cpt.get(n, [])] + [w for _, w, _ in self.in_mod.get(n, [])]
            self.gamma[n] = 1.0 / max(1.0, math.sqrt(max(sum(w * w for w in ws), 1e-12)))
        self.rest = [n for n in self.topo if n not in self.K]

    # ---------------------------------------------------------------- helpers
    def code_of(self, node, labels):
        """labels: array of value strings (None/unknown -> sentinel)"""
        m = self.vtoi[node]; s = self.k[node]
        return np.array([m.get(v, s) if v is not None else s for v in labels], np.int16)

    def _cpt_term(self, c, t, codes, avail):
        par = c['parents']; n = len(next(iter(codes.values())))
        if all(p in avail for p in par):
            code = np.zeros(n, np.int64); bad = np.zeros(n, bool)
            for p, mu in zip(par, c['mult']):
                x = codes[p].astype(np.int64); bad |= x >= self.k[p]; code += np.minimum(x, self.k[p] - 1) * mu
            code[bad] = c['space']
            return c['lut'][code].T   # (k_t, n)
        if t not in self.K: return None
        # K target with some parents not drawn yet: average over their prior
        dims = [self.k[p] for p in par]
        P = np.exp(c['lut'][:c['space']] + self.logprior[t][None, :])   # (space, k_t) probabilities
        P = P.reshape(dims[::-1] + [self.k[t]])                          # axis j <-> parent par[len-1-j]
        axes_par = par[::-1]
        for j in reversed(range(len(axes_par))):
            p = axes_par[j]
            if p not in avail:
                P = np.tensordot(P, self.prior[p], axes=([j], [0]))
                axes_par = axes_par[:j] + axes_par[j + 1:]
        if not axes_par: return np.repeat((np.log(np.maximum(P, 1e-12)) - self.logprior[t])[:, None], n, axis=1).astype(np.float32)
        lr = np.log(np.maximum(P, 1e-12)) - self.logprior[t]          # shape dims(avail, reversed order) + [k_t]
        rem = axes_par[::-1]; mult = np.cumprod([1] + [self.k[p] for p in rem[:-1]])
        flat = lr.reshape(-1, self.k[t]); flat = np.concatenate([flat, np.zeros((1, self.k[t]))])
        code = np.zeros(n, np.int64); bad = np.zeros(n, bool)
        for p, mu in zip(rem, mult):
            x = codes[p].astype(np.int64); bad |= x >= self.k[p]; code += np.minimum(x, self.k[p] - 1) * mu
        code[bad] = flat.shape[0] - 1
        return flat[code].T.astype(np.float32)

    def _logits(self, t, codes, avail, use_masks):
        n = len(next(iter(codes.values())))
        g = self.gamma[t]; lg = np.repeat(self.logprior[t].astype(np.float32)[:, None], n, axis=1)
        for c in self.cpt.get(t, []):
            term = self._cpt_term(c, t, codes, avail)
            if term is not None: lg += g * c['w'] * term
        for s, w, T in (self.in_K[t] if t in self.K else self.in_mod[t]):
            if s in avail: lg += (g * w) * T[:, codes[s]]
        lg -= lg.max(axis=0, keepdims=True); pr = np.exp(lg)
        if use_masks:
            for cond, mult in self.masks.get(t, []):
                if not all(p in avail for p, _ in cond): continue
                rows = np.ones(n, bool)
                for p, allowed in cond: rows &= np.isin(codes[p], allowed)
                if rows.any():
                    sub = pr[:, rows] * mult[:, None]; dead = sub.sum(0) <= 0; sub[:, dead] = pr[:, rows][:, dead]; pr[:, rows] = sub
            for fn in self.dyn.get(t, []):
                m = fn(codes, avail)
                if m is None: continue
                sub = pr * m; dead = sub.sum(0) <= 0; sub[:, dead] = pr[:, dead]; pr = sub
        return pr

    def _draw(self, pr):
        cdf = np.cumsum(pr, axis=0); u = self.rng.random(pr.shape[1]) * cdf[-1]
        return np.minimum((cdf < u).sum(axis=0), pr.shape[0] - 1).astype(np.int16)

    # ---------------------------------------------------------------- public
    def draw_korean(self, t, codes, avail, allowed=None):
        """draw K node t from MatrAIx q given already-drawn Korean parents; allowed: (n, k) bool or None"""
        pr = self._logits(t, codes, avail, use_masks=True)   # MatrAIx masks act inside the allowed set; if they empty it, the set wins
        if allowed is not None:
            al = allowed.T.astype(np.float32); p2 = pr * al; z = p2.sum(0) <= 0
            p2[:, z] = (np.exp(self.logprior[t])[:, None] * al)[:, z]; pr = p2
        return self._draw(pr)

    def sample_rest(self, codes, nodes=None):
        """draw every non-K node (attributes and latent nodes) in MatrAIx order; codes holds all K codes"""
        avail = set(codes)
        for t in (nodes or self.rest):
            codes[t] = self._draw(self._logits(t, codes, avail, use_masks=True)); avail.add(t)
        return codes
