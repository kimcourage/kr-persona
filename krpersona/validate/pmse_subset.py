"""pMSE again on the attributes MatrAIx also has (KNHANES; KGSS 2021 respondents), so the MatrAIx value is comparable.
usage: KR_HOLDOUT=1 python pmse_subset.py <personas.parquet> <out.pkl>"""
import sys, pickle, pandas as pd
from krpersona.validate import sources as SRC, validate as VA

if __name__ == '__main__':
    P = pd.read_parquet(sys.argv[1]); MXA = SRC.matraix_adults([c for c in P.columns if c != 'persona_id']); out = {}
    kn = SRC.knhanes(); attrs = [a for a in kn['to_class'] if a in kn['real'].columns and a in MXA.columns]
    out[kn['name']] = dict(VA.pmse(kn, P, MXA, attrs), attrs=attrs)
    kg = SRC.kgss(); attrs = [a for a in kg['to_class'] if a in kg['real'].columns]
    # the 10 attributes MatrAIx also has with the most 2021 held-out answers (complete rows stay near all 2021 respondents)
    y21 = (kg['real'].YEAR == 2021).to_numpy()
    cand = [a for a in attrs if a in kg['years'] and 2021 in kg['years'][a] and a in MXA.columns]
    cnt = {a: int(((kg['real'][a] >= 0).to_numpy() & y21).sum()) for a in cand}
    a21 = sorted(cand, key=lambda a: -cnt[a])[:10] + ['gender_identity', 'age_bracket', 'highest_education']
    out[kg['name']] = dict(VA.pmse(kg, P, MXA, a21, rows=(kg['real'].YEAR == 2021).to_numpy()), attrs=a21)
    pickle.dump(out, open(sys.argv[2], 'wb')); print(out)
