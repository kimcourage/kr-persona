"""How the estimation stores respondent data: KLoSA donors (blocks/family.py) and mother-child gaps (census/aggregates.py)."""
import numpy as np
from krpersona.blocks.family import microaggregate, KMIN
from krpersona.census.aggregates import freq_table


def _donors(seed=0):
    rng = np.random.default_rng(seed); D = {}
    for male in (0, 1):
        for band in range(5):
            pool = []
            for _ in range(int(rng.integers(20, 60))):
                k = int(rng.choice([1, 2, 3, 4, 5, 6], p=[.2, .4, .25, .1, .04, .01]))
                pool.append((np.sort(rng.integers(16, 45, k)).astype(float), float(rng.uniform(100, 3000))))
            pool.append((np.array([12.0, 30.0]), 500.0))   # a gap under 15: never drawn, removed
            D[(male, band)] = pool
    return D


def test_microaggregate_keeps_structure_and_group_size():
    D = _donors(); out, info = microaggregate(D)
    assert info['removed'] == 10 and info['smallest'] >= KMIN
    for c, pool in D.items():
        use = [(g, w) for g, w in pool if g.min() >= 15]
        assert len(out[c]) == len(use)
        assert all(len(a[0]) == len(b[0]) for a, b in zip(use, out[c]))
        assert all(g.min() >= 15 for g, _ in out[c])
    cnt = {}
    for pool in out.values():
        for g, w in pool: cnt[(tuple(g), w)] = cnt.get((tuple(g), w), 0) + 1
    assert min(cnt.values()) >= KMIN


def test_freq_table_merges_small_ends():
    v = np.repeat([15, 16, 17, 30, 47, 48, 49], [3, 4, 50, 100, 33, 4, 2])
    t = freq_table(v)
    assert t['value'].tolist() == [17, 30, 47] and t['count'].tolist() == [57, 100, 39]
    assert t['count'].sum() == len(v)
