"""Rejection checks applied to every generated persona: MatrAIx masks that target Korean attributes and are kept as checks, plus fixed logical rules."""
import json, numpy as np, pandas as pd
from krpersona import paths
EDU_MIN = {"Associate's": 19, "Bachelor's": 20, "Master's": 22, 'Doctorate': 25, 'Postdoc': 27}
EXP_LB = {'0-2': 0, '3-5': 3, '6-10': 6, '11-20': 11, '20+': 20}
WORKING = ['Full-time', 'Part-time', 'Self-employed', 'Gig / freelance', 'Unpaid family worker']
DOMAIN_OF = {'High openness': 'bfi2_domain_open_mindedness', 'High conscientiousness': 'bfi2_domain_conscientiousness',
             'High extraversion': 'bfi2_domain_extraversion', 'High agreeableness': 'bfi2_domain_agreeableness', 'High neuroticism': 'bfi2_domain_negative_emotionality'}

def korean_masks(K):
    g = json.load(open(paths.graph_path())); return [m for m in g['conditional_masks'] if m['target'] in K and m.get('bad_values')]

def mask_hits(out, m):
    hit = np.isin(out[m['target']], m['bad_values'])
    for p, allowed in m['condition'].items():
        if p not in out: return None
        hit &= np.isin(out[p], allowed)
    return hit

def rules(G, kept_masks):
    o, D, n = G.out, G.D, len(G.D); age = D.age.to_numpy(); R = {}
    R['경력 하한 > 나이 − 15'] = np.array([EXP_LB[x] for x in o['years_experience']]) > age - 15
    R['학력별 최소 나이 미달'] = np.array([age[i] < EDU_MIN.get(x, 0) for i, x in enumerate(o['highest_education'])])
    R['자녀와 나이 차 15세 미만'] = np.array([any(age[i] - c < 15 for c in G.kids[i]) for i in range(n)])
    cc, ps, pj = o['demo_children_count'], o['demo_parental_status'], o['lifex_parenting_journey']
    R['자녀 수·부모 상태·육아 경험 불일치'] = ((cc == 'None') != (ps == 'Not a parent')) | ((cc == 'None') != (pj == 'No children'))
    dom = o['dominant_trait']; bad = np.zeros(n, bool)
    for v, d in DOMAIN_OF.items(): bad |= (dom == v) & ~np.isin(o[d], ['High', 'Very high'])
    R['우세 성격과 요인 범주 불일치'] = bad
    R['국내 출생 내국인인데 한국어 Native 아님'] = o['lang_korean'] != 'Native'
    R['종교 없음인데 종교성 Observant·Devout'] = np.isin(o['demo_religion_affiliation'], ['None', 'Atheist / agnostic']) & np.isin(o['religiosity'], ['Observant', 'Devout'])
    R['시각 Blind인데 운전'] = (o['health_vision'] == 'Blind') & ~np.isin(o['demo_driver_status'], ['Non-driver', 'Cannot drive'])
    ms, rl = o['demo_marital_status'], o['demo_relationship_length']
    R['혼인 상태와 관계 기간 불일치'] = (np.isin(ms, ['Married', 'In a relationship', 'Engaged']) & (rl == 'Not in one')) | ((ms == 'Single') & (rl != 'Not in one'))
    if 'lifex_relationship_history' in o:
        R['미혼인데 연애 이력이 결혼·재혼·사별'] = np.isin(ms, ['Single', 'In a relationship', 'Engaged']) & np.isin(o['lifex_relationship_history'], ['Married once', 'Remarried', 'Widowed'])
        R['사별인데 상실 경험에 배우자·자녀 없음'] = (ms == 'Widowed') & ~np.isin(o['lifex_loss_experience'], ['Lost a partner / child', 'Multiple bereavements'])
        R['석사 미만인데 학업 경로 Advanced degrees'] = ~np.isin(o['highest_education'], ["Master's", 'Doctorate', 'Postdoc']) & (o['lifex_education_journey'] == 'Advanced degrees')
        w = np.isin(o['demo_employment_status'], WORKING)
        R['취업·비취업과 근무 일정 불일치'] = (w & (o['lstyle_work_schedule'] == 'Unemployed')) | (~w & (o['lstyle_work_schedule'] != 'Unemployed'))
    for m in kept_masks:
        h = mask_hits(o, m)
        if h is not None: R[f"MatrAIx 규칙 {m['mask_id']}"] = h
    return pd.DataFrame(R)
