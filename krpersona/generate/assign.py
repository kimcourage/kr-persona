"""Assignment table: which method makes each MatrAIx attribute ((1) census, (2) rule, (3) latent score, (4) survey model, (5) MatrAIx), and how the stage-0 generator fills it."""
import json, openpyxl, re
from krpersona import paths

CENSUS = ['age_bracket', 'gender_identity', 'region', 'urbanicity', 'highest_education', 'demo_marital_status', 'demo_employment_status',
          'demo_citizenship_status', 'demo_housing_status', 'lstyle_household_size', 'lstyle_commute_mode', 'demo_disability_status',
          'health_vision', 'health_hearing', 'health_mobility', 'domain', 'role_function', 'demo_relationship_length',
          'lifex_geographic_mobility', 'lstyle_pet_ownership', 'demo_driver_status']
RULE = ['demo_generation', 'lifex_formative_decade', 'lifex_immigration_generation', 'demo_ethnicity_broad', 'cultural_background',
        'cult_south_korea', 'lang_korean', 'primary_language', 'demo_children_count', 'demo_parental_status', 'lifex_parenting_journey',
        'life_stage', 'dominant_trait', 'years_experience', 'seniority', 'health_mental_health', 'socioeconomic_band']
RULE_NEEDS_SURVEY = {'health_mental_health': '증상 수준(건강 블록)', 'socioeconomic_band': '원화 가구소득(소득 블록)'}
SCHWARTZ = ['schwartz_value_self_direction', 'schwartz_value_stimulation', 'schwartz_value_hedonism', 'schwartz_value_achievement',
            'schwartz_value_power', 'schwartz_value_security', 'schwartz_value_conformity', 'schwartz_value_tradition',
            'schwartz_value_benevolence', 'schwartz_value_universalism']

def build():
    g = json.load(open(paths.graph_path())); nodes = g['nodes']
    ids = [n['id'] for n in nodes]; emit = [n['id'] for n in nodes if n.get('emit', True) is not False]
    PERS = [i for i in ids if i.startswith('bfi2_') or i.startswith('big5_')] + ['mbti_type', 'lstyle_social_battery']
    ws = openpyxl.load_workbook(paths.MAPPING)['범주 대응']; survey = {}; blk = None
    for r in ws.iter_rows(min_row=5, values_only=True):
        if r[1] and r[0] in ('노동 블록', '소득 블록', '건강 블록', '태도/가치관/사회인식 블록'):
            a = str(r[1]).split(' ')[0]
            if a in ids and a not in CENSUS and a not in RULE: survey[a] = (r[0], r[2])
    survey.pop('health_sleep_quality', None)          # 대응표에서 "MatrAIx에 맡김"
    rows = {}
    for a in emit:
        if a in CENSUS: rows[a] = ('(1) 센서스', '인구주택총조사(합성 기본 속성)', '0단계에도 같음')
        elif a in RULE:
            rows[a] = ('(2) 규칙', '규칙 문서', f'0단계는 MatrAIx 확률(필요한 {RULE_NEEDS_SURVEY[a]}이 아직 없음)' if a in RULE_NEEDS_SURVEY else '0단계에도 같음')
        elif a in PERS: rows[a] = ('(3) 잠재 점수: 성격', 'KGSS TIPI 10문항(4단계 추정; 대응표에는 BFI-10으로 적힘)', '0단계는 임시 모수(성, 연령 효과 0, 요인 간 상관 0, 부하량 0.8)')
        elif a in SCHWARTZ: rows[a] = ('(3) 잠재 점수: 가치', 'KGSS 2021 Schwartz(4단계 추정)', '0단계는 (5) MatrAIx')
        elif a in survey: rows[a] = ('(4) 조사 추정', f'{survey[a][1]}({survey[a][0]})', '0단계는 MatrAIx 확률(한국 속성 먼저, 아직 뽑지 않은 부모 연결은 인수 1)')
        else: rows[a] = ('(5) MatrAIx', 'MatrAIx 그래프', '입력값은 한국 값. 스위치를 끄면 만들지 않음')
    K0 = [a for a, r in rows.items() if r[0][:3] in ('(1)', '(2)', '(3)') and 'Schwartz' not in r[1]]
    PH = [a for a, r in rows.items() if r[0].startswith('(4)')] + list(RULE_NEEDS_SURVEY)
    K0 = sorted(set(K0) | set(PH))
    return rows, K0, PH, ids, emit

if __name__ == '__main__':
    rows, K0, PH, ids, emit = build()
    import collections
    print(collections.Counter(r[0] for r in rows.values()), 'K0', len(K0), 'placeholders', len(PH))
