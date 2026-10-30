"""KRSampler with nothing fixed reproduces MatrAIx's own sampler: per-attribute TVD between the two, against two MatrAIx draws.
Run: python tests/test_sampler.py (needs config/paths.toml with matraix_dir)."""
if __name__ == '__main__':
    import time
    import numpy as np
    from krpersona.matraix.sampler import KRSampler

    t = time.time(); S = KRSampler(K=set(), seed=1); print('init', round(time.time() - t, 1), 'rest', len(S.rest))
    n = 50000; t = time.time(); codes = S.sample_rest({'_dummy': np.zeros(n, np.int64)}); print('sample', round(time.time() - t, 1))
    b = S.b; b.rng = np.random.default_rng(2); B = b.sample_indices(n)
    tv = []
    for c in S.emit:
        k = S.k[c]; a = np.bincount(codes[c], minlength=k)[:k] / n; bb = np.bincount(B[c].astype(int), minlength=k)[:k] / n; tv.append((0.5 * np.abs(a - bb).sum(), c))
    tv.sort(reverse=True); print('max tvd', tv[:5], 'median', np.median([x for x, _ in tv]))
    b.rng = np.random.default_rng(3); B2 = b.sample_indices(n)   # null: two MatrAIx draws
    tv0 = sorted([0.5 * np.abs(np.bincount(B[c].astype(int), minlength=S.k[c])[:S.k[c]] / n - np.bincount(B2[c].astype(int), minlength=S.k[c])[:S.k[c]] / n).sum() for c in S.emit], reverse=True)
    print('null max', tv0[:5], 'median', np.median(tv0))
