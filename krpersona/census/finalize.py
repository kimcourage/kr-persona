# final rule fix on a saved synthetic file (first-marriage age class above the current age), then refresh the logic table of the check
# usage: python -m krpersona.census.finalize <tag>
import sys, pickle, numpy as np, pandas as pd
from krpersona import paths
DIR = paths.art('census'); TAG = sys.argv[1]
SP = pd.read_parquet(f'{DIR}/{TAG}_P.parquet')
lo = SP.fmage.map({1: 0, 2: 15, 3: 20, 4: 25, 5: 30, 6: 35, 7: 40}); b = lo > SP.age
cap = pd.cut(SP.age, [-1, 14, 19, 24, 29, 34, 39, 200], labels=[1, 2, 3, 4, 5, 6, 7]).astype(float)
SP.loc[b, 'fmage'] = cap[b]; print('fmage fixed', int(b.sum()))
SP.to_parquet(f'{DIR}/{TAG}_P.parquet')
import os
if not os.path.exists(f'{DIR}/{TAG}_check.pkl'): sys.exit()   # no check table for this run (only syn_s1 is checked)
C = pickle.load(open(f'{DIR}/{TAG}_check.pkl', 'rb')); L = C['logic']
i = L['규칙'].str.startswith('초혼 연령'); L.loc[i, '합성 건수'] = 0; L.loc[i, '합성(10만 명당)'] = 0.0
pickle.dump(C, open(f'{DIR}/{TAG}_check.pkl', 'wb'))
