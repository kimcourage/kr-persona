"""Where the labor-survey gap in (1) comes from: workers' age and education, synthetic vs 2020 census held-out vs 2025 EAPS held-out.
usage: KR_HOLDOUT=1 python eaps_decomp.py <personas.parquet> <out.pkl>"""
import sys, pickle, numpy as np, pandas as pd
from krpersona.validate import sources as SRC
AG = ['19-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']

def share(x, w, k): return np.bincount(x, weights=w, minlength=k)[:k] / w.sum()

if __name__ == '__main__':
    P = pd.read_parquet(sys.argv[1], columns=['age_bracket', 'highest_education', 'demo_employment_status'])
    e = SRC.eaps(); T = e['to_class']; sc = e['scope'](P); S = P[sc]
    ag_s = np.minimum(T['age_bracket'][0](S.age_bracket), 6); ed_s = T['highest_education'][0](S.highest_education)
    r = e['real']; w = e['w']; ag_e = np.minimum(r.age_bracket.to_numpy(), 6); ed_e = r.highest_education.to_numpy()
    c = SRC.census(); rc = c['real']; wc = c['w']; m = np.isin(rc.demo_employment_status.to_numpy(), [0, 1, 2])
    ag_c = np.minimum(rc.age_bracket.to_numpy()[m], 6); ed_c = rc.highest_education.to_numpy()[m]; wc = wc[m]
    # census education classes -> the EAPS comparison classes (bachelor's or more = class >= 2 in both)
    ba_c = np.isin(ed_c, [c['to_class']['highest_education'][0](pd.Series([v])).item() for v in ["Bachelor's", "Master's", 'Doctorate']])
    A = pd.DataFrame({'연령': AG, '합성 취업자': share(ag_s, np.ones(len(ag_s)), 7), '2020 센서스 분리분 취업자': share(ag_c, wc, 7), '2025 지역별고용조사 분리분 취업자': share(ag_e, w, 7)})
    B = pd.DataFrame({'연령': AG,
                      '합성': [np.mean(ed_s[ag_s == i] >= 2) for i in range(7)],
                      '2020 센서스 분리분': [np.average(ba_c[ag_c == i], weights=wc[ag_c == i]) for i in range(7)],
                      '2025 지역별고용조사 분리분': [np.average(ed_e[ag_e == i] >= 2, weights=w[ag_e == i]) for i in range(7)]})
    pickle.dump(dict(age=A, ba=B), open(sys.argv[2], 'wb')); pd.set_option('display.width', 200); print(A.round(3)); print(B.round(3))
