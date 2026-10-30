"""MatrAIx baseline sample (artifacts/baseline): KR = Korean-culture natives, BASE = uniform sample of all MatrAIx personas."""
import json, numpy as np
from krpersona import paths
D=np.load(paths.art('baseline','codes.npz')); KR=D['kr']; BASE=D['base']
meta=json.load(open(paths.art('baseline','meta.json'))); cols=meta['columns']; vals=meta['values']; ci={c:i for i,c in enumerate(cols)}
def dist(M,c):
    k=len(vals[c]); x=np.bincount(M[:,ci[c]],minlength=k)[:k]; return x/x.sum()
def show(c, M1=None, M2=None, label=('KR-native','global')):
    M1=KR if M1 is None else M1; M2=BASE if M2 is None else M2
    a,b=dist(M1,c),dist(M2,c)
    print(f'-- {c}')
    for v,x,y in zip(vals[c],a,b): print(f'   {v:38s} {x*100:6.1f}%  {y*100:6.1f}%')
def tvd(c,M1=None,M2=None):
    M1=KR if M1 is None else M1; M2=BASE if M2 is None else M2
    return 0.5*np.abs(dist(M1,c)-dist(M2,c)).sum()
def mask(M,c,vs): return np.isin(M[:,ci[c]],[vals[c].index(v) for v in vs])
