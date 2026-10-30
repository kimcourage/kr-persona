"""Build the analysis tables (household H, person P) from the 2020 census 2% sample.
Output: artifacts/census/H.parquet, P.parquet (real microdata extract: keep with the raw data). Usage: python -m krpersona.census.prep"""
import sys,pandas as pd,numpy as np
from krpersona import paths
OUT=paths.art('census','x',mkdir=True).parent
p=pd.read_csv(paths.raw('census2020','person'),encoding='cp949',dtype=str)
h=pd.read_csv(paths.raw('census2020','household'),encoding='cp949',dtype=str)
f=lambda s: pd.to_numeric(s,errors='coerce')
H=pd.DataFrame({'hid':h.가구일련번호,'sido':f(h.행정구역시도코드),'dong':f(h.동읍면부_구분코드),'gtype':f(h.세대유형코드),'hkind':f(h.가구구분코드),
  'dwell':f(h.거처종류코드),'tenure_h':f(h.점유형태코드),'car':f(h.자동차주차장소_자동차보유여부),'hw':f(h.가구가중값)})
yes=lambda s:(f(s)==1).astype(int)
H['pet']=yes(h.반려동물_개유무)+2*yes(h.반려동물_고양이유무)+4*yes(h.반려동물_기타유무)
P=pd.DataFrame({'hid':p.가구일련번호,'mno':f(p.가구원번호),'rel':f(p.가구주관계코드),'sex':f(p.성별코드),'age':f(p.만연령)})
bn,cn=f(p.출생국적_대한민국여부),f(p.현재국적_대한민국여부)
P['nat']=np.select([(bn==1)&(cn==1),(bn==2)&(cn==1),(bn==2)&(cn==2)],[1,2,3],4)
sido=f(p.행정구역시도코드); bp=f(p.출생지구분코드); bs=f(p.출생지_타행정구역시도코드)
P['bplace']=np.where((bp==3)&(bs!=sido),5,bp); P['bsido']=np.where(P.bplace==5,bs,np.nan)
r5=f(p['5년전거주지_구분코드']); s5=f(p['5년전거주지_타시도행정구역코드'])
P['r5']=np.where((r5==4)&s5.notna()&(s5!=sido),6,r5); P['r1']=f(p['1년전거주지_구분코드'])
P['edu']=f(p.교육정도코드); P['sch']=f(p.수학구분코드)
lim=[('시각장애',1),('청각장애',2),('육체적제약',4),('정신적제약',8),('지적자폐성장애',16),('언어장애',32)]
L=sum(b*(f(p[f'활동제약_{n}_지속여부'])==1).astype(int) for n,b in lim); P['lim']=np.where(f(p.활동제약_시각장애_지속여부).notna(),L,np.nan)
P['comm']=f(p.통근통학구분코드)
fl={k:f(p[c]).notna() for k,c in [('W','교통수단_도보여부'),('B','교통수단_자전거_이용여부'),('C','교통수단_승용차및소형승합차_이용여부'),('T','교통수단_트럭_이용여부'),('Bu','교통수단_시내좌석마을버스_이용여부'),('Cb','교통수단_통근통학버스이용_이용여부'),('Ib','교통수단_고속시외버스_이용여부'),('S','교통수단_전철지하철_이용여부'),('Tr','교통수단_기차_이용여부'),('Tx','교통수단_택시_이용여부'),('O','교통수단_기타_이용여부')]}
pub=fl['Bu']|fl['Cb']|fl['Ib']|fl['S']|fl['Tr']; car=fl['C']|fl['T']
mode=np.select([pub&car,pub,car,fl['Tx'],fl['B'],fl['W'],fl['O']],[5,4,3,6,2,1,7],0)
P['mode']=np.where(P.comm.isin([1,2]),mode,np.nan)
P['econ']=f(p.경제활동상태코드); emp=P.econ.isin([1,2,3])
P['status']=np.where(emp,f(p.종사상지위코드),np.nan)
P['ind']=np.where(emp,p.산업대분류코드.map({c:i+1 for i,c in enumerate('ABCDEFGHIJKLMNOPQRSTU')}),np.nan)
P['occ']=np.where(emp,p.직업대분류코드.map({**{str(i):i for i in range(1,10)},'A':10}),np.nan)
P['jten']=np.where(emp,f(p.현직업근무년수코드),np.nan); P['place']=np.where(emp,f(p.근로장소코드),np.nan)
P['mar']=f(p.혼인상태코드); ever=P.mar.isin([2,3,4]); P['fmage']=np.where(ever,f(p.초혼연령코드),np.nan)
fw=(P.sex==2)&ever
P['births']=np.where(fw,f(p.남아출산수).fillna(0)+f(p.여아출산수).fillna(0),np.nan)
P['deaths']=np.where(fw,f(p.남아사망수).fillna(0)+f(p.여아사망수).fillna(0),np.nan)
P['prejob']=np.where(fw,f(p.결혼전취업유무),np.nan); P['brk']=np.where(P.prejob==1,f(p.경력단절유무),np.nan)
a20=P.age>=20
P['soc_pol']=np.where(a20,yes(p.사회활동_정치분야단체_참여및활동여부),np.nan); P['soc_soc']=np.where(a20,yes(p.사회활동_사회분야단체_참여및활동여부),np.nan)
# order members: head first, then by relationship code, older first within the same relationship
P=P.sort_values(['hid','rel','age'],ascending=[True,True,False]).reset_index(drop=True)
P['pos']=P.groupby('hid').cumcount(); P['ridx']=P.groupby(['hid','rel']).cumcount()
cnt=P[P.rel>1].pivot_table(index='hid',columns='rel',values='age',aggfunc='size',fill_value=0)
for r in range(2,15): H[f'n{r:02d}']=H.hid.map(cnt[r] if r in cnt else {}).fillna(0).astype(int)
H['k']=H.hid.map(P.groupby('hid').size())
assert (H.k==1+H[[f'n{r:02d}' for r in range(2,15)]].sum(1)).all()
P=P.merge(H[['hid','hw','sido','dong']],on='hid')
H.to_parquet(f'{OUT}/H.parquet'); P.to_parquet(f'{OUT}/P.parquet')
print(H.shape,P.shape); print(P.isna().mean().round(3).to_dict())
for c in ['nat','bplace','r5','mode','lim']: print(c,P[c].value_counts(dropna=False).sort_index().head(12).to_dict())
print('pet',H.pet.value_counts().sort_index().to_dict(),'gtype n',H.gtype.nunique())
# person-weight vs household-weight check (sex x age5 x sido, persons)
pw=f(p.인구가중값); q=pd.DataFrame({'hid':p.가구일련번호,'sex':f(p.성별코드),'a5':(f(p.만연령)//5).clip(upper=17),'sido':sido,'pw':pw}).merge(H[['hid','hw']],on='hid')
g1=q.groupby(['sex','a5','sido']).pw.sum(); g2=q.groupby(['sex','a5','sido']).hw.sum()
print('TVD person vs hh weights (sex x age5 x sido) %',round(0.5*np.abs(g1/g1.sum()-g2/g2.sum()).sum()*100,2), 'totals',round(pw.sum()),round(q.hw.sum()))
