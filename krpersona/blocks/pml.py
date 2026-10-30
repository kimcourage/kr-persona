"""Penalized multinomial logit with a prior mean.
logit_v = b0_v + sum_j b_j[v, x_j] + sum_(j,k) b_jk[v, x_j, x_k];  penalty ||main - prior_main||^2/(2 s^2) + ||int - prior_int||^2/(2 s_int^2)
"""
import numpy as np, scipy.sparse as sp
from scipy.optimize import minimize

class Design:
    def __init__(self, levels, pairs=()):
        """levels: list of number of levels per parent; pairs: list of (j, k) parent index pairs"""
        self.levels = list(levels); self.pairs = [tuple(p) for p in pairs]
        off = [1]
        for L in self.levels: off.append(off[-1] + L)
        self.main_end = off[-1]; self.off = off[:-1]
        poff = [];  o = self.main_end
        for j, k in self.pairs: poff.append(o); o += self.levels[j] * self.levels[k]
        self.poff = poff; self.p = o
    def matrix(self, X):
        n = len(X); rows = [np.arange(n)]; cols = [np.zeros(n, np.int64)]
        for j, L in enumerate(self.levels): rows.append(np.arange(n)); cols.append(self.off[j] + X[:, j])
        for (j, k), o in zip(self.pairs, self.poff): rows.append(np.arange(n)); cols.append(o + X[:, j] * self.levels[k] + X[:, k])
        r = np.concatenate(rows); c = np.concatenate(cols)
        return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, self.p))

def fit(D, X, y, w, K, M, s_main, s_int, B0=None, maxiter=500):
    """D: Design; X (n, J) int codes; y (n,) int in [0, K); w (n,) weights normalized to sum n; M (p, K) prior mean"""
    A = D.matrix(X); n = len(y); Y = np.zeros((n, K)); Y[np.arange(n), y] = 1
    prec = np.full(D.p, 1 / s_main ** 2); prec[D.main_end:] = 1 / s_int ** 2
    def f(b):
        B = b.reshape(D.p, K); Z = A @ B; Z -= Z.max(1, keepdims=True); E = np.exp(Z); S = E.sum(1, keepdims=True); P = E / S
        ll = (w * (np.log(P[np.arange(n), y] + 1e-300))).sum(); dB = B - M
        g = A.T @ (w[:, None] * (P - Y)) + prec[:, None] * dB
        return -ll + 0.5 * (prec[:, None] * dB ** 2).sum(), g.ravel()
    r = minimize(f, (M if B0 is None else B0).ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter))
    return r.x.reshape(D.p, K)

def predict(D, B, X):
    Z = D.matrix(X) @ B; Z -= Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

def cv_fit(D, X, y, w, K, M, s0=(0.25, 0.1), folds=5, seed=0, groups=None, max_steps=10, lo=(0.01, 0.005), hi=(16.0, 4.0)):
    """choose (s_main, s_int) by weighted held-out log-likelihood (5-fold; rows of the same group stay in one fold):
    coordinate search on a factor-2 grid from s0 (halve/double each scale while the CV log-likelihood improves), then refit on all"""
    w = w * len(w) / w.sum(); rng = np.random.default_rng(seed)
    if groups is None: fold = rng.integers(0, folds, len(y))
    else: u, inv = np.unique(groups, return_inverse=True); fold = rng.integers(0, folds, len(u))[inv]
    cache = {}
    def cvll(sm, si):
        key = (round(sm, 6), round(si, 6))
        if key in cache: return cache[key]
        tot = 0.0
        for f_ in range(folds):
            tr, te = fold != f_, fold == f_
            B = fit(D, X[tr], y[tr], w[tr] * tr.sum() / w[tr].sum(), K, M, sm, si, maxiter=200)
            P = predict(D, B, X[te]); tot += (w[te] * np.log(P[np.arange(te.sum()), y[te]] + 1e-12)).sum()
        cache[key] = tot; return tot
    cur = (s0[0], s0[1] if D.pairs else 1.0); best = cvll(*cur)
    for _ in range(max_steps):
        cands = [(cur[0] * 2, cur[1]), (cur[0] / 2, cur[1])] + ([(cur[0], cur[1] * 2), (cur[0], cur[1] / 2)] if D.pairs else [])
        cands = [c for c in cands if lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1]]
        sc = {c: cvll(*c) for c in cands}; cb = max(sc, key=sc.get)
        if sc[cb] <= best: break
        cur, best = cb, sc[cb]
    B = fit(D, X, y, w, K, M, cur[0], cur[1])
    return B, dict(s_main=cur[0], s_int=cur[1], cv={f'{a}/{b}': v for (a, b), v in cache.items()}, at_bound=bool(cur[0] in lo + hi or (D.pairs and cur[1] in lo + hi)))
