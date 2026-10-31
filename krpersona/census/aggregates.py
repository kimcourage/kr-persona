"""Census aggregates used at generation time, so that generation does not need the raw census files.

poststrat      person-weight totals of the persona population (domestic-born Korean nationals aged 19+) by sex x 5-year age (85+ pooled) x 시도;
               keys 'sex_age5_sido' as in Base (post-stratification targets)
gap            mother-child age gaps (mothers who are head or spouse, co-resident children; 15-50 years) as a frequency table
               {'value', 'count'}; values held by fewer than 10 records at either end are merged into the nearest value with 10 or more.
               load() expands it in ascending order. Used to give ages to non-co-resident children
participation  P(사회분야단체 참여 | sex, ag7, edu5) among the same population aged 20+, person-weighted (dict cell -> rate) and the overall rate;
               used by the KGSS block for volunteering
build() computes them from the raw census and writes artifacts/census/aggregates.pkl; load() reads it through paths.inp (artifacts or bundle).
usage: python -m krpersona.census.aggregates
"""
import pickle
import numpy as np, pandas as pd
from krpersona import paths

CELL = lambda s, a, d: s.astype(int).astype(str) + '_' + (np.minimum(a.astype(int), 85) // 5).astype(str) + '_' + d.astype(int).astype(str)
_CACHE = {}


def poststrat_from_raw():
    R = pd.read_csv(paths.raw('census2020', 'person'), encoding='cp949', dtype=str, usecols=['성별코드', '만연령', '행정구역시도코드', '출생국적_대한민국여부', '현재국적_대한민국여부', '출생지구분코드', '인구가중값'])
    R = R[(R.출생국적_대한민국여부 == '1') & (R.현재국적_대한민국여부 == '1') & (R.출생지구분코드 != '4') & (R.만연령.astype(int) >= 19)]
    return pd.Series(R.인구가중값.astype(float).to_numpy()).groupby(CELL(R.성별코드, R.만연령, R.행정구역시도코드).to_numpy()).sum()


def gaps_from(rp):
    mo = rp[rp.rel.isin([1, 2]) & (rp.sex == 2)][['hid', 'age']].rename(columns={'age': 'ma'})
    ch = rp[rp.rel == 3][['hid', 'age']].merge(mo, on='hid'); gap = (ch.ma - ch.age)
    return gap[(gap >= 15) & (gap <= 50)].to_numpy()


def freq_table(v, kmin=10):
    """values -> (value, count) with counts below kmin at either end merged into the nearest value with kmin or more"""
    val, cnt = np.unique(np.asarray(v), return_counts=True); val, cnt = list(val), list(cnt)
    while cnt[0] < kmin: x = cnt.pop(0); val.pop(0); cnt[0] += x
    while cnt[-1] < kmin: x = cnt.pop(); val.pop(); cnt[-1] += x
    assert min(cnt) >= kmin
    return dict(value=np.array(val, dtype=np.asarray(v).dtype), count=np.array(cnt))


def expand(t):
    """frequency table -> values in ascending order"""
    return np.repeat(t['value'], t['count'])


def participation_from_raw():
    from krpersona.blocks import core
    R = pd.read_csv(paths.raw('census2020', 'person'), encoding='cp949', dtype=str,
                    usecols=['성별코드', '만연령', '교육정도코드', '수학구분코드', '출생국적_대한민국여부', '현재국적_대한민국여부', '출생지구분코드', '인구가중값',
                             '사회활동_사회분야단체_참여및활동여부'])
    R = R[(R.출생국적_대한민국여부 == '1') & (R.현재국적_대한민국여부 == '1') & (R.출생지구분코드 != '4') & (R.만연령.astype(int) >= 20)]
    y = R.사회활동_사회분야단체_참여및활동여부.astype(float).to_numpy() == 1
    cell = (R.성별코드.astype(int).to_numpy() - 1) * 100 + core.ag7(R.만연령.astype(int).to_numpy()) * 10 + core.edu5_census(R.교육정도코드.astype(int), pd.to_numeric(R.수학구분코드).fillna(1).to_numpy())
    w = R.인구가중값.astype(float).to_numpy()
    t = pd.DataFrame(dict(c=cell, y=y * w, w=w)).groupby('c').sum(); return (t.y / t.w).to_dict(), float((y * w).sum() / w.sum())


def build():
    out = dict(poststrat=poststrat_from_raw(), gap=freq_table(gaps_from(pd.read_parquet(paths.art('census', 'P.parquet')))), participation=participation_from_raw())
    pickle.dump(out, open(paths.art('census', 'aggregates.pkl', mkdir=True), 'wb'))
    return out


def load():
    if 'a' not in _CACHE:
        A = pickle.load(open(paths.inp('census', 'aggregates.pkl'), 'rb'))
        A['gap'] = expand(A['gap'])
        _CACHE['a'] = A
    return _CACHE['a']


if __name__ == '__main__':
    a = build(); print('cells', len(a['poststrat']), 'gaps', int(a['gap']['count'].sum()), 'gap values', len(a['gap']['value']), 'participation cells', len(a['participation'][0]), 'overall', round(a['participation'][1], 4))
