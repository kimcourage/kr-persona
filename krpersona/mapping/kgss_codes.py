"""KGSS attribute codes: the KGSS variables of each attribute (sheet '범주 대응' of the mapping workbook) and the combined items built from several variables."""
import re,numpy as np,pandas as pd,openpyxl
from krpersona import paths
EXTRA=['LOTR1','LOTR2','LOTR3','LOTR4','LOTR5','LOTR6']
def attr_vars(xlsx=None):
    ws=openpyxl.load_workbook(xlsx or paths.MAPPING)['범주 대응']; A={}
    for r in ws.iter_rows(min_row=5,values_only=True):
        if r[1] and r[2]=='KGSS' and not str(r[1]).startswith(('Schwartz','BFI')):
            A[r[1]]=[v for v in re.findall(r'\b[A-Z][A-Z0-9_]{2,}\b',str(r[3])) if v not in('KGSS','ISSP')]
    A['trait_hope_optimism']=EXTRA
    return A
def validity(df,m,allv):
    out={}
    for v in allv:
        miss=[k for k,l in m.variable_value_labels.get(v,{}).items() if re.search('모르|무응답|비해당|DK|Refusal|Not Applicable|선택할 수 없',str(l))]
        s=df[v]; out[v]=s.notna()&(s>=0)&~s.isin(miss)
    return out
def build(df,V,A):
    g=lambda v: df[v].where(V[v])
    C={}
    for a,vs in A.items(): C[a]=g(vs[0])
    rel,att,ten=g('RELIG'),g('ATTEND'),g('RELITEN'); r=pd.Series(np.nan,index=df.index); has=rel.notna()&(rel!=4)
    r[rel==4]=0; r[has&att.between(5,8)]=3; r[has&att.between(1,4)]=2; r[has&att.between(1,2)&(ten==1)]=1; C['religiosity']=r
    t=pd.concat([g(v) for v in ['CONGOVT','CONLEGIS','CONJUDGE']],axis=1).mean(axis=1); C['kr_trust_in_institutions']=pd.cut(t,[0,1.5,2.34,3.1],labels=False)
    t=pd.concat([g(v) for v in ['CONPRESS','CONTV']],axis=1).mean(axis=1); C['kr_trust_in_news_media']=pd.cut(t,[0,1.5,2.0,3.1],labels=False)
    t=pd.concat([g(v) for v in ['POLEFF11','POLEFF18']],axis=1).mean(axis=1); C['kr_political_efficacy']=pd.cut(t,[0,2.0,3.5,5.1],labels=False)
    T=pd.concat([g(v) for v in A['lifex_travel_breadth']],axis=1); k=T.notna().all(1); C['lifex_travel_breadth']=pd.cut((T==1).sum(1),[-1,0,2,9],labels=False).where(k)
    L=pd.concat([g(v) if v not in('LOTR2','LOTR4','LOTR5') else 6-g(v) for v in EXTRA],axis=1); s=L.sum(1).where(L.notna().all(1))
    C['trait_hope_optimism']=pd.qcut(s.rank(method='first'),5,labels=False).where(s.notna())
    s=(g('OTHERS1')+g('OTHERS2')); C['trait_empathy']=pd.qcut(s.rank(method='first'),5,labels=False).where(s.notna())
    o=g('SIBNO1')+g('SIBNO2'); y=g('SIBNO4')+g('SIBNO5'); b=pd.Series(np.nan,index=df.index)
    b[(o==0)&(y==0)]=0; b[(o==0)&(y>0)]=1; b[(o>0)&(y==0)]=2; b[(o>0)&(y>0)]=3; C['demo_birth_order']=b
    P=pd.concat([g(v) for v in ['SIGNPET','JOINDEM','ATTRALLY']],axis=1); e=pd.Series(np.nan,index=df.index); k=P.notna().all(1)
    e[k&(P==4).all(1)]=3; e[k&(P==3).any(axis=1)]=2; e[k&P.isin([1,2]).any(axis=1)]=1; e[k&((P.JOINDEM==1)|(P.ATTRALLY==1))]=0; C['demo_political_engagement']=e
    C['att_authority']=g('PATRACH1').fillna(8-g('PATRACH123'))
    C['att_immigration']=g('LETIN1').fillna(g('FORNWKER'))
    C['att_traditional_gender_roles']=g('HUBBYWRK').fillna(g('HBBYWK08'))
    return pd.DataFrame(C)
