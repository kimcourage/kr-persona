"""Language attributes of generated samples: multilingualism, languages in a lifetime, home language, English level, and the number and
share of languages spoken at Conversational or better. Written after round 3, whose compiled masks forced particular languages.
usage: python -m krpersona.validate.language_check <out.pkl> <label=personas.parquet> [<label=personas.parquet> ...]
"""
import sys, pickle
import numpy as np, pandas as pd

GOOD = {'Conversational', 'Fluent', 'Native'}
SCALARS = ['multilingualism', 'lifex_languages_lifetime', 'demo_home_language', 'english_proficiency']


def summary(P):
    lang = [c for c in P.columns if c.startswith('lang_')]
    other = [c for c in lang if c not in ('lang_korean', 'lang_english')]
    good = pd.DataFrame({c: P[c].astype(str).isin(GOOD) for c in other})
    n_other = good.sum(axis=1)
    out = {'n': len(P)}
    for c in SCALARS:
        out[c] = P[c].astype(str).value_counts(normalize=True)
    out['other_languages_conversational'] = pd.cut(n_other, [-1, 0, 1, 2, 4, 9, 10 ** 6], labels=['0', '1', '2', '3-4', '5-9', '10+']).value_counts(normalize=True).sort_index()
    out['other_languages_mean'] = float(n_other.mean())
    out['share_conversational_by_language'] = good.mean().sort_values(ascending=False)
    return out


if __name__ == '__main__':
    res = {}
    for arg in sys.argv[2:]:
        label, path = arg.split('=', 1)
        res[label] = summary(pd.read_parquet(path))
    pickle.dump(res, open(sys.argv[1], 'wb'))
    for key in SCALARS + ['other_languages_conversational']:
        print('\n' + key)
        print(pd.DataFrame({k: v[key] for k, v in res.items()}).fillna(0).round(3).to_string())
    print('\nmean number of other languages at Conversational+')
    print(pd.Series({k: round(v['other_languages_mean'], 2) for k, v in res.items()}).to_string())
    print('\nshare at Conversational+ (top 8 per sample)')
    print(pd.DataFrame({k: v['share_conversational_by_language'].head(8).round(3) for k, v in res.items()}).fillna('').to_string())
