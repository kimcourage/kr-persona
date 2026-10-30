"""Structure of the KGSS block (step 1).
A: for attributes asked both in 2003-2010 and 2016-2025, age-group TVD of the 2016-2025 distribution under three fits (all years with main effects,
   all years with age x period, 2016-2025 only).  B: 2003-2010 distributions carried forward with no change, age fixed, or birth cohort fixed.
P: per attribute the estimation window (2016-2025 if >= 1,000 valid answers there, else all years), the years used, the year used at generation,
   the age treatment, and up to 3 KGSS parents chosen greedily (log-likelihood gain >= 2%, background sex, age group, education, year).
Output: artifacts/mapping/kgss_analysis.pkl ({'A', 'B', 'P'}). Usage: python -m krpersona.mapping.kgss_analysis"""
import sys,pyreadstat,pandas as pd,numpy as np,warnings,pickle,time
warnings.filterwarnings('ignore')
from krpersona import paths
from krpersona.mapping.kgss_codes import attr_vars,validity,build
from sklearn.linear_model import LogisticRegression
t0=time.time()
A=attr_vars(); allv=sorted({v for vs in A.values() for v in vs})
df,m=pyreadstat.read_sav(str(paths.raw('kgss','all')),usecols=['YEAR','AGE','SEX','EDUC','FINALWT']+allv)
df=df[(df.AGE>=19)&(df.AGE<=100)&df.SEX.isin([1,2])].reset_index(drop=True); df['YEAR']=df.YEAR.astype(int)
df['ag']=pd.cut(df.AGE,[18,29,39,49,59,200],labels=False); df['ed']=np.select([df.EDUC.isin([0,1,2,8]),df.EDUC==3,df.EDUC.between(4,7)],[0,1,2],-1)
df['coh']=pd.cut(df.YEAR-df.AGE,[0,1939,1949,1959,1969,1979,2100],labels=False)
V=validity(df,m,allv); C=build(df,V,A); O=C.notna()
LATE=df.YEAR>=2016; EARLY=df.YEAR<=2010
yrs={a:sorted(df.loc[O[a],'YEAR'].unique()) for a in A}
def ll(X,y):
    clf=LogisticRegression(C=10,max_iter=3000).fit(X,y); P=clf.predict_proba(X)
    return np.log(P[np.arange(len(y)),np.searchsorted(clf.classes_,y)]).sum()
def oh(cols,d): return pd.get_dummies(d[cols].astype(int).astype(str),drop_first=True).values.astype(float)
def tvd_age(P,cls,y,w,ag):
    out=[]
    for g_ in range(5):
        k=ag==g_
        if k.sum()<50: continue
        obs=np.array([np.average(y[k]==c,weights=w[k]) for c in cls]); out.append(0.5*np.abs(obs-np.average(P[k],axis=0,weights=w[k])).sum()*100)
    return np.mean(out)
# ===== A and B: year handling
RA,RB=[],[]
for a in A:
    if (O[a]&EARLY).sum()<1000 or (O[a]&LATE).sum()<1000: continue
    d=df[O[a]].copy(); y=C.loc[O[a],a].astype(int).values; L=(d.YEAR>=2016).values
    d['per']=np.where(d.YEAR>=2016,2,np.where(d.YEAR<=2010,0,1)); d['agper']=d.ag*10+d.per
    dl=d[L]; yl=y[L]; w=dl.FINALWT.values; ag=dl.ag.values
    def fp(X,yy,keep=None):
        clf=LogisticRegression(C=10,max_iter=3000).fit(X,yy); P=clf.predict_proba(X); return (P if keep is None else P[keep]),clf.classes_
    P1,c1=fp(oh(['SEX','ag','ed','YEAR'],d),y,L); P2,c2=fp(oh(['SEX','ag','ed','YEAR','agper'],d),y,L); P3,c3=fp(oh(['SEX','ag','ed','YEAR'],dl),yl)
    RA.append((a,int((O[a]&EARLY).sum()),int((O[a]&LATE).sum()),tvd_age(P1,c1,yl,w,ag),tvd_age(P2,c2,yl,w,ag),tvd_age(P3,c3,yl,w,ag)))
    de=d[d.per==0]; ye=y[(d.per==0).values]; cls=sorted(set(ye))
    def tab(key):
        t=pd.crosstab(de[key],ye,values=de.FINALWT,aggfunc='sum').reindex(columns=cls).fillna(0); return t.div(t.sum(1),axis=0)
    ta,tc=tab('ag'),tab('coh'); cl=dl.coh.clip(tc.index.min(),tc.index.max())
    P0=np.tile([np.average(ye==c,weights=de.FINALWT) for c in cls],(len(dl),1))
    RB.append((a,tvd_age(P0,cls,yl,w,ag),tvd_age(ta.reindex(dl.ag).values,cls,yl,w,ag),tvd_age(np.nan_to_num(tc.reindex(cl).values),cls,yl,w,ag)))
RA=pd.DataFrame(RA,columns=['속성','2003–2010 응답','2016–2025 응답','전 연도·주효과','전 연도·연령×시기','2016–2025만'])
RB=pd.DataFrame(RB,columns=['속성','변화 없음','연령 고정','출생 코호트 고정'])
print('7A n',len(RA),RA.iloc[:,3:].median().round(2).to_dict(),'main>2pp',(RA['전 연도·주효과']>2).mean().round(2),round(time.time()-t0))
print('7B',RB.iloc[:,1:].median().round(2).to_dict(),'cohort<age',(RB['출생 코호트 고정']<RB['연령 고정']).mean().round(2))
# ===== P: parents within the KGSS block
WIN={a:(LATE if (O[a]&LATE).sum()>=1000 else pd.Series(True,index=df.index)) for a in A}
first=['demo_religion_affiliation','religiosity','political_lean','kr_subjective_class','kr_view_of_north_korea','kr_unification_need','kr_trust_in_institutions','kr_trust_in_news_media','kr_trust_in_big_business','att_organized_religion','att_labor_unions','val_patriotism','kr_country_felt_closest']
Nw={a:int((O[a]&WIN[a]).sum()) for a in A}
order=first+sorted([a for a in A if a not in first],key=lambda a:-Nw[a])
must={'religiosity':['demo_religion_affiliation'],'att_organized_religion':['demo_religion_affiliation','religiosity'],'topic_spirituality':['demo_religion_affiliation','religiosity'],'lifex_faith_journey':['demo_religion_affiliation','religiosity'],'val_spirituality_faith':['demo_religion_affiliation','religiosity'],'kr_unification_need':['kr_view_of_north_korea'],'kr_happiness':['kr_life_satisfaction'],'demo_political_engagement':['political_lean'],'topic_politics':['political_lean']}
S=(df.SEX.astype(int).astype(str)+df.ag.astype(str)+df.ed.astype(str)+df.YEAR.astype(str))
def Hc(y,s):
    t=pd.crosstab(s,y).values.astype(float); p=t/t.sum(); ps=p.sum(1,keepdims=True)
    with np.errstate(divide='ignore',invalid='ignore'): return -np.nansum(p*np.log(p/ps))
def cmi(y,x,s):
    k=y.notna()&x.notna(); y,x,s=y[k],x[k],s[k]; n=len(y)
    if n<300: return -1
    dfree=sum((g.y.nunique()-1)*(g.x.nunique()-1) for _,g in pd.DataFrame({'y':y,'x':x,'s':s}).groupby('s'))
    return Hc(y,s)-Hc(y,s+'|'+x.astype(int).astype(str))-dfree/(2*n)
def select(a,R,elig,crit):
    y=C[a]; sel=[b for b in must.get(a,[]) if b in elig]; gains=[]
    def design(par,mk):
        d=df.loc[mk,['SEX','ag','ed','YEAR']].astype(int).astype(str)
        for b in par: d[b]=C.loc[mk,b].astype(int).astype(str)
        return pd.get_dummies(d,drop_first=True).values.astype(float)
    for j,b in enumerate(sel):
        mk=R.copy()
        for c in sel[:j+1]: mk&=O[c]
        yy=y[mk].astype(int).values; l0=ll(design(sel[:j],mk),yy); l1=ll(design(sel[:j+1],mk),yy); gains.append(f"{b} +{(l1-l0)/abs(l0)*100:.1f}%(논리)")
    scr=sorted([(cmi(y[R],C.loc[R,b],S[R]),b) for b in elig if b not in sel],reverse=True); cand=[b for c,b in scr[:8] if c>0]
    K=y[R].nunique(); base=R.copy()
    for b in sel: base&=O[b]
    while len(sel)<3:
        best=None
        for b in [c for c in cand if c not in sel]:
            mk=base&O[b]
            if mk.sum()<0.9*R.sum(): continue
            yy=y[mk].astype(int).values; l0=ll(design(sel,mk),yy); l1=ll(design(sel+[b],mk),yy)
            k=(K-1)*(C.loc[mk,b].nunique()-1); g=(l1-l0)/abs(l0)*100
            crit_v=2*(l1-l0)-k*(np.log(mk.sum()) if crit=='BIC' else 2)
            if crit=='GAIN2' and g<2: continue
            if best is None or crit_v>best[0]: best=(crit_v,b,g)
        if best is None or best[0]<=0: break
        sel.append(best[1]); gains.append(f"{best[1]} +{best[2]:.1f}%"); base&=O[best[1]]
    return sel,gains,scr
rows=[]
for i,a in enumerate(order):
    R=O[a]&WIN[a]
    elig=[b for b in order[:i] if (O[b]&R).sum()/R.sum()>=0.9]
    sel,gains,scr=select(a,R,elig,'GAIN2')
    selB,_,_=select(a,R,elig,'BIC')
    yw=[t for t in yrs[a] if (WIN[a] is not LATE) or t>=2016]
    lo=[t for t in yw if t<=2020]; hi=[t for t in yw if t>=2020]
    gen=(f"{max(lo)}·{min(hi)} 보간({min(hi)-2020}:{2020-max(lo)})" if lo and hi and min(hi)-max(lo)<=7 and max(lo)!=min(hi) else f"{min(yw,key=lambda t:(abs(t-2020),-t))}년")
    old=max(yw)<=2011; late_any=(O[a]&LATE).sum()>0
    agep='출생 코호트 맞춤' if old else ('연령 그대로' if WIN[a] is LATE else '연령×시기 상호작용')
    rows.append((i+1,a,'2016–2025' if WIN[a] is LATE else '전 연도',Nw[a],', '.join(map(str,yw)),gen,agep,len(elig),', '.join(sel) if sel else '(배경 변수만)',', '.join(gains),', '.join(selB) if selB else '(배경 변수만)'))
    if i%12==0: print(i,a,round(time.time()-t0),flush=True)
P=pd.DataFrame(rows,columns=['순서','속성','추정 기간','유효 응답(추정 기간)','추정에 쓰는 연도','생성 시 연도','연령 처리','부모 자격 속성 수','선택 부모(최대 3)','부모별 로그우도 개선','참고: BIC 기준 선택'])
pickle.dump({'A':RA,'B':RB,'P':P},open(paths.art('mapping','kgss_analysis.pkl',mkdir=True),'wb'))
pd.set_option('display.width',400); pd.set_option('display.max_colwidth',120)
print(P[['순서','속성','추정 기간','생성 시 연도','연령 처리','부모 자격 속성 수','부모별 로그우도 개선','참고: BIC 기준 선택']].to_string(index=False))
print('no parent',(P['선택 부모(최대 3)']=='(배경 변수만)').sum(),'BIC none',(P['참고: BIC 기준 선택']=='(배경 변수만)').sum(),'done',round(time.time()-t0))
