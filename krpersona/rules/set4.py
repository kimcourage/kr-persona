"""Consistency rules (set 4) for synthetic personas of South Korean adults.

Written only from the attribute catalog definitions and real-world knowledge.

Time assumptions (from the catalog's population definition):
  * population = domestic-born Korean citizens aged 19+ (age_bracket '18-24' means 19-24);
  * composition and birth-year-derived attributes use a 2020 reference: birth year = 2020 - age;
  * technology / products / AI tools describe life today (2026), so products released after
    2020 are never treated as contradictions.

API
  RULES                       list of dicts: id, group, severity, description, scope, hit
  violation_matrix(df, rules) -> (hits, scopes, status)
  run(df)                     -> (table, any_hard_rate)
  check_values(catalog_path)  -> list of problems (empty when every referenced id/value exists)

Every column is read only through the module-level accessor ``_s``.
Developer-only attributes may hold 'Not applicable' (and dev_professional_status
'Not a developer'); such values never count as a contradiction by themselves.

These rules also informed generator changes; evaluation independence is limited.
See docs/08_모순_규칙.md.
"""
import json

import numpy as np
import pandas as pd


class MissingColumn(KeyError):
    """Raised by ``_s`` when a rule needs a column that is not in the frame."""


def _s(df, col):
    """The single column accessor: stripped string view of one column."""
    if col not in df.columns:
        raise MissingColumn(col)
    return df[col].astype(str).str.strip()


# ---------------------------------------------------------------------------
# value-reference registry (used by check_values)
# ---------------------------------------------------------------------------
_REFS = {}          # col -> set of referenced values


def _ref(col, vals):
    _REFS.setdefault(col, set()).update(vals)


def _in(df, col, *vals):
    """Column value is one of vals (missing values -> False)."""
    _ref(col, vals)
    return _s(df, col).isin(list(vals))


def _ni(df, col, *vals):
    """Column holds a valid-looking value that is NOT one of vals.

    A value is treated as missing (-> False) when it is 'nan', 'None', '' or '<NA>'."""
    _ref(col, vals)
    x = _s(df, col)
    return (~x.isin(list(vals)) & ~x.isin(_MISSING) & x.notna()).fillna(False).astype(bool)


_MISSING = ['nan', 'None', '', '<NA>', 'NaN', 'none']

_ORDERS = {}        # name -> (col, [ordered values])


def _order(col, values, name=None):
    _ref(col, values)
    _ORDERS[name or col] = (col, list(values))
    return name or col


def _rank(df, name):
    col, values = _ORDERS[name]
    m = {v: float(i) for i, v in enumerate(values)}
    return _s(df, col).map(m).astype(float)


def _bool(x, index=None):
    if isinstance(x, pd.Series):
        return x.fillna(False).astype(bool)
    return pd.Series(bool(x), index=index)


def _any(*xs):
    out = xs[0].copy()
    for x in xs[1:]:
        out = out | x
    return out


def _all(*xs):
    out = xs[0].copy()
    for x in xs[1:]:
        out = out & x
    return out


# ---------------------------------------------------------------------------
# rule registration
# ---------------------------------------------------------------------------
RULES = []
_IDS = set()


def _rule(rid, group, severity, description, scope, hit):
    assert rid.startswith('R4-'), rid
    assert rid not in _IDS, rid
    assert severity in ('hard', 'soft'), severity
    _IDS.add(rid)
    RULES.append(dict(id=rid, group=group, severity=severity,
                      description=description, scope=scope, hit=hit))


def _pair(rid, group, severity, description, col_a, vals_a, col_b, vals_b):
    """Scope: col_a in vals_a.  Hit: col_b in vals_b."""
    vals_a, vals_b = tuple(vals_a), tuple(vals_b)
    _ref(col_a, vals_a)
    _ref(col_b, vals_b)
    _rule(rid, group, severity, description,
          lambda df: _in(df, col_a, *vals_a),
          lambda df: _in(df, col_b, *vals_b))


# age brackets (population is 19+, so '18-24' = 19-24); birth year = 2020 - age
AGE = _order('age_bracket', ['Under 5', '5-12', '13-17', '18-24', '25-34', '35-44',
                             '45-54', '55-64', '65-74', '75-84', '85+'])
ADULT_AGES = ('18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+')


# ===========================================================================
# RULES
# ===========================================================================

# --- population definition --------------------------------------------------
_rule('R4-POP-H01', 'population', 'hard',
      '모집단은 19세 이상인데 연령대가 미성년(5세 미만·5-12·13-17)',
      lambda df: _ni(df, 'age_bracket'),
      lambda df: _in(df, 'age_bracket', 'Under 5', '5-12', '13-17'))

_rule('R4-POP-H02', 'population', 'hard',
      '모집단은 한국 국적자인데 시민권 상태가 영주권자·비자 소지자·미등록 체류자',
      lambda df: _ni(df, 'demo_citizenship_status'),
      lambda df: _in(df, 'demo_citizenship_status', 'Permanent resident', 'Visa holder', 'Undocumented'))

_rule('R4-POP-H03', 'population', 'hard',
      '모집단은 국내 거주 한국인(인구총조사)인데 거주 지역이 동아시아가 아님',
      lambda df: _ni(df, 'region'),
      lambda df: _ni(df, 'region', 'East Asia'))

_rule('R4-POP-H04', 'population', 'hard',
      '모집단은 국내 출생자인데 이민 세대가 1세대 이민자',
      lambda df: _ni(df, 'lifex_immigration_generation'),
      lambda df: _in(df, 'lifex_immigration_generation', 'First-generation immigrant'))

_rule('R4-POP-S01', 'population', 'soft',
      '국내 출생 한국 국민이 귀화 시민인 경우는 극히 드묾(국내 출생 화교 등 일부)',
      lambda df: _ni(df, 'demo_citizenship_status'),
      lambda df: _in(df, 'demo_citizenship_status', 'Naturalized citizen'))

_rule('R4-POP-S02', 'population', 'soft',
      '국내 출생 한국 국민 성인 중 동아시아계·혼혈이 아닌 민족 배경은 1% 미만(다문화 가정 성인 자녀도 극소수)',
      lambda df: _ni(df, 'demo_ethnicity_broad'),
      lambda df: _ni(df, 'demo_ethnicity_broad', 'East Asian', 'Multiracial'))

# --- age arithmetic (birth year = 2020 - age) --------------------------------
# generation windows with +-2..3 years tolerance; a bracket is allowed a generation
# when its birth-year span overlaps the tolerant window.
_GEN_ALLOWED = {
    '18-24': ('Gen Z', 'Millennial'),           # born 1996-2001
    '25-34': ('Millennial', 'Gen Z'),           # born 1986-1995
    '35-44': ('Millennial', 'Gen X'),           # born 1976-1985
    '45-54': ('Gen X', 'Boomer'),               # born 1966-1975
    '55-64': ('Boomer', 'Gen X'),               # born 1956-1965
    '65-74': ('Boomer', 'Silent'),              # born 1946-1955
    '75-84': ('Silent', 'Boomer'),              # born 1936-1945
    '85+': ('Silent',),                         # born <=1935
}
_GEN_ALL = ('Gen Alpha', 'Gen Z', 'Millennial', 'Gen X', 'Boomer', 'Silent')
_ref('demo_generation', _GEN_ALL)
_ref('age_bracket', tuple(_GEN_ALLOWED))


def _gen_bad(df):
    a = _s(df, 'age_bracket')
    g = _s(df, 'demo_generation')
    bad = pd.Series(False, index=df.index)
    for br, ok in _GEN_ALLOWED.items():
        bad |= (a == br) & g.isin(_GEN_ALL) & ~g.isin(ok)
    return bad


_rule('R4-AGE-H01', 'age_arithmetic', 'hard',
      '출생연도(2020-나이)로 본 세대와 세대 코호트 값이 2세대 이상 어긋남(허용오차 ±2~3년)',
      lambda df: _in(df, 'age_bracket', *ADULT_AGES) & _in(df, 'demo_generation', *_GEN_ALL),
      _gen_bad)

# formative decade must be a decade in which the person was aged roughly 5-35
_FORM_BAD = {
    '18-24': ('1960s-70s', '1980s', '1990s'),
    '25-34': ('1960s-70s', '1980s'),
    '35-44': ('1960s-70s',),
    '45-54': ('2020s',),
    '55-64': ('2010s', '2020s'),
    '65-74': ('2000s', '2010s', '2020s'),
    '75-84': ('1990s', '2000s', '2010s', '2020s'),
    '85+': ('1980s', '1990s', '2000s', '2010s', '2020s'),
}
_ref('lifex_formative_decade', ('1960s-70s', '1980s', '1990s', '2000s', '2010s', '2020s'))


def _form_bad(df):
    a = _s(df, 'age_bracket')
    d = _s(df, 'lifex_formative_decade')
    bad = pd.Series(False, index=df.index)
    for br, badv in _FORM_BAD.items():
        bad |= (a == br) & d.isin(badv)
    return bad


_rule('R4-AGE-H02', 'age_arithmetic', 'hard',
      '형성기 10년대가 출생연도 기준으로 5세 이전이거나 35세 이후인 시기',
      lambda df: _in(df, 'age_bracket', *ADULT_AGES) & _ni(df, 'lifex_formative_decade'),
      _form_bad)

_pair('R4-AGE-H03', 'age_arithmetic', 'hard',
      '19-24세인데 교제 기간이 15년 이상(9세 이전부터 교제)',
      'age_bracket', ['18-24'], 'demo_relationship_length', ['15+ years'])

_rule('R4-AGE-H04', 'age_arithmetic', 'hard',
      '19-24세인데 성인 자녀·성인 자녀의 부모·손주(5세 전후 출산 필요)',
      lambda df: _in(df, 'age_bracket', '18-24'),
      lambda df: _in(df, 'demo_children_count', 'Adult children')
      | _in(df, 'demo_parental_status', 'Parent of adults', 'Grandparent')
      | _in(df, 'lifex_parenting_journey', 'Grown children', 'Empty nester', 'Raising grandchildren'))

_rule('R4-AGE-S01', 'age_arithmetic', 'soft',
      '25-34세인데 성인 자녀(만 19세 이상) 보유 — 15세 이전 출산이 필요해 극히 드묾',
      lambda df: _in(df, 'age_bracket', '25-34'),
      lambda df: _in(df, 'demo_children_count', 'Adult children')
      | _in(df, 'demo_parental_status', 'Parent of adults')
      | _in(df, 'lifex_parenting_journey', 'Grown children', 'Empty nester'))

_rule('R4-AGE-S02', 'age_arithmetic', 'soft',
      '25-44세 조부모·손주 양육 — 만혼 사회인 한국에서 1% 미만',
      lambda df: _in(df, 'age_bracket', '25-34', '35-44'),
      lambda df: _in(df, 'demo_parental_status', 'Grandparent')
      | _in(df, 'lifex_parenting_journey', 'Raising grandchildren'))

_pair('R4-AGE-S03', 'age_arithmetic', 'soft',
      '19-24세 사별·이혼 — 2020 인구총조사에서 20대 초반 유배우 경험자 자체가 드물어 1% 미만',
      'age_bracket', ['18-24'], 'demo_marital_status', ['Widowed', 'Divorced'])

_pair('R4-AGE-S04', 'age_arithmetic', 'soft',
      '19-34세 은퇴 상태 — 한국 청년층에서 1% 미만',
      'age_bracket', ['18-24', '25-34'], 'demo_employment_status', ['Retired'])

_pair('R4-AGE-S06', 'age_arithmetic', 'soft',
      '75세 이상인데 고용 상태가 학생 — 1% 미만',
      'age_bracket', ['75-84', '85+'], 'demo_employment_status', ['Student'])

# --- family ----------------------------------------------------------------
_pair('R4-FAM-H01', 'family', 'hard',
      '자녀 없음인데 부모 상태가 신생아 부모·미성년/성인 자녀 부모·조부모',
      'demo_children_count', ['None'],
      'demo_parental_status', ['New parent', 'Parent of minors', 'Parent of adults', 'Grandparent'])

_pair('R4-FAM-H02', 'family', 'hard',
      '자녀 수가 1명 이상(또는 성인 자녀)인데 부모 상태가 부모 아님',
      'demo_children_count', ['1 child', '2 children', '3+ children', 'Adult children'],
      'demo_parental_status', ['Not a parent'])

_pair('R4-FAM-H03', 'family', 'hard',
      '육아 여정이 자녀 없음인데 자녀 수가 1명 이상',
      'lifex_parenting_journey', ['No children'],
      'demo_children_count', ['1 child', '2 children', '3+ children', 'Adult children'])

_pair('R4-FAM-H04', 'family', 'hard',
      '자녀 수 없음인데 육아 여정이 어린 자녀·10대 자녀·장성한 자녀·빈 둥지·손주 양육',
      'demo_children_count', ['None'],
      'lifex_parenting_journey', ['Young children', 'Teenagers', 'Grown children', 'Empty nester',
                                  'Raising grandchildren'])

_pair('R4-FAM-H05', 'family', 'hard',
      '부모 아님인데 육아 여정이 자녀 양육 단계(어린·10대·장성·빈 둥지·손주 양육)',
      'demo_parental_status', ['Not a parent'],
      'lifex_parenting_journey', ['Young children', 'Teenagers', 'Grown children', 'Empty nester',
                                  'Raising grandchildren'])

_rule('R4-FAM-H06', 'family', 'hard',
      '생애 단계가 어린 자녀 부모·빈 둥지인데 자녀 없음/부모 아님',
      lambda df: _in(df, 'life_stage', 'Parent of young kids', 'Empty nester'),
      lambda df: _in(df, 'demo_children_count', 'None')
      | _in(df, 'demo_parental_status', 'Not a parent')
      | _in(df, 'lifex_parenting_journey', 'No children'))

_pair('R4-FAM-H07', 'family', 'hard',
      '혼인 상태가 교제 중·약혼·기혼·동거인데 교제 기간이 교제 안 함',
      'demo_marital_status', ['In a relationship', 'Engaged', 'Married', 'Domestic partnership'],
      'demo_relationship_length', ['Not in one'])

_pair('R4-FAM-H08', 'family', 'hard',
      '혼인 상태가 미혼(인구총조사 미혼=혼인 경험 없음)인데 관계 이력이 결혼 1회·재혼·사별',
      'demo_marital_status', ['Single'],
      'lifex_relationship_history', ['Married once', 'Remarried', 'Widowed'])

_pair('R4-FAM-H09', 'family', 'hard',
      '1인 가구인데 주거 형태가 가족과 함께 거주',
      'lstyle_household_size', ['Lives alone'], 'demo_housing_status', ['Living with family'])

_rule('R4-FAM-H10', 'family', 'hard',
      '사별(혼인 상태 또는 관계 이력)인데 상실 경험이 큰 상실 없음',
      lambda df: _in(df, 'demo_marital_status', 'Widowed') | _in(df, 'lifex_relationship_history', 'Widowed'),
      lambda df: _in(df, 'lifex_loss_experience', 'No major loss'))

# --- life events vs histories ----------------------------------------------
_pair('R4-LIF-H01', 'life_events', 'hard',
      '고향을 떠난 적이 없는데 고향과의 관계가 자주 방문·가끔 귀향·거의 안 감·연결 없음',
      'lifex_geographic_mobility', ['Never left hometown'],
      'lifex_hometown_tie', ['Visits often', 'Occasional return', 'Rarely returns', 'No connection'])

_pair('R4-LIF-H02', 'life_events', 'hard',
      '해외로 이주한 적이 있는데 여행 범위가 출국 경험 없음',
      'lifex_geographic_mobility', ['Moved internationally'],
      'lifex_travel_breadth', ['Never left country'])

_pair('R4-LIF-H03', 'life_events', 'hard',
      '해외 거주·글로벌 노마드인데 고향을 떠난 적 없음',
      'lifex_travel_breadth', ['Lived abroad', 'Global nomad'],
      'lifex_geographic_mobility', ['Never left hometown'])

_rule('R4-LIF-H04', 'life_events', 'hard',
      '귀국자(리터니)인데 출국 경험 없음 또는 고향을 떠난 적 없음',
      lambda df: _in(df, 'lifex_immigration_generation', 'Returnee / repatriate'),
      lambda df: _in(df, 'lifex_travel_breadth', 'Never left country')
      | _in(df, 'lifex_geographic_mobility', 'Never left hometown'))

_pair('R4-LIF-H05', 'life_events', 'hard',
      '제3문화 아동(부모와 다른 문화권에서 성장)인데 출국 경험 없음',
      'lifex_cultural_exposure', ['Third-culture kid'], 'lifex_travel_breadth', ['Never left country'])

_pair('R4-LIF-H06', 'life_events', 'hard',
      '삶을 규정한 사건이 이주·주요 이사인데 고향을 떠난 적 없음',
      'major_life_events', ['Migration / immigration', 'Major relocation'],
      'lifex_geographic_mobility', ['Never left hometown'])

_pair('R4-LIF-H07', 'life_events', 'hard',
      '삶을 규정한 사건이 사별인데 상실 경험이 큰 상실 없음',
      'major_life_events', ['Bereavement'], 'lifex_loss_experience', ['No major loss'])

_pair('R4-LIF-H08', 'life_events', 'hard',
      '삶을 규정한 사건이 건강 문제인데 건강 여정이 큰 사건 없음',
      'major_life_events', ['Health journey'], 'lifex_health_journey', ['No major events'])

_pair('R4-LIF-H09', 'life_events', 'hard',
      '삶을 규정한 사건이 창업인데 창업 이력이 고려한 적 없음·고려만 함',
      'major_life_events', ['Started a business'],
      'lifex_entrepreneurship_history', ['Never considered', 'Considered it'])

_pair('R4-LIF-H10', 'life_events', 'hard',
      '경력 형태가 창업가형인데 창업 이력이 고려한 적 없음',
      'lifex_career_path_shape', ['Entrepreneurial'],
      'lifex_entrepreneurship_history', ['Never considered'])

_pair('R4-LIF-S02', 'life_events', 'soft',
      '삶을 규정한 사건이 군 복무인데 군 이력이 복무한 적 없음 — 본인 경험이 아닌 경우는 드묾',
      'major_life_events', ['Military service'], 'lifex_military_history', ['Never served'])

# --- military ----------------------------------------------------------------
_rule('R4-MIL-H01', 'military', 'hard',
      '현역·제대 군인인데 군 이력이 복무한 적 없음',
      lambda df: _in(df, 'demo_veteran_status', 'Active duty', 'Veteran'),
      lambda df: _in(df, 'lifex_military_history', 'Never served'))

_pair('R4-MIL-H02', 'military', 'hard',
      '군 이력이 직업 군인·참전 군인인데 제대군인 상태가 민간인(군 경력 없음)',
      'lifex_military_history', ['Career veteran', 'Combat veteran'],
      'demo_veteran_status', ['Civilian'])

_pair('R4-MIL-H03', 'military', 'hard',
      '65세 이상 현역 군인 — 군인 연령정년 최고 63세(대장)',
      'age_bracket', ['65-74', '75-84', '85+'], 'demo_veteran_status', ['Active duty'])

_pair('R4-MIL-H04', 'military', 'hard',
      '현역 군인인데 고용 상태가 은퇴·전업주부·실업',
      'demo_veteran_status', ['Active duty'],
      'demo_employment_status', ['Retired', 'Homemaker', 'Unemployed'])

_rule('R4-MIL-S01', 'military', 'soft',
      '여성 현역·제대 군인 또는 직업/단기 복무 이력 — 여군은 성인 여성의 0.1% 수준',
      lambda df: _in(df, 'gender_identity', 'Woman'),
      lambda df: _in(df, 'demo_veteran_status', 'Active duty', 'Veteran')
      | _in(df, 'lifex_military_history', 'Served briefly', 'Career veteran', 'Combat veteran'))

_pair('R4-MIL-S02', 'military', 'soft',
      '64세 이하 참전 군인 — 베트남전(1964-73) 이후 한국군 실전 참전자는 극소수',
      'age_bracket', ['18-24', '25-34', '35-44', '45-54', '55-64'],
      'lifex_military_history', ['Combat veteran'])

_pair('R4-MIL-S03', 'military', 'soft',
      '45세 이상 예비군 — 예비군 편성은 만 40세까지(지원 예비군 제외 극소수)',
      'age_bracket', ['45-54', '55-64', '65-74', '75-84', '85+'],
      'demo_veteran_status', ['Reserve / guard'])

# --- education -----------------------------------------------------------------
_EDU_BELOW_MASTER = ('No formal', 'Primary', 'Secondary', 'Vocational / cert', 'Some college',
                     "Associate's", "Bachelor's")
_EDU_DEGREE = ("Associate's", "Bachelor's", "Master's", 'Doctorate', 'Postdoc')

_pair('R4-EDU-H01', 'education', 'hard',
      '19-24세인데 최종 학력이 박사·박사후 — 학사(22-23세) 후 최소 4-5년 필요',
      'age_bracket', ['18-24'], 'highest_education', ['Doctorate', 'Postdoc'])

_pair('R4-EDU-S01', 'education', 'soft',
      '19-24세 석사 학위 — 학사 후 2년이 필요해 24세 이하 석사는 1% 미만',
      'age_bracket', ['18-24'], 'highest_education', ["Master's"])

_pair('R4-EDU-H02', 'education', 'hard',
      '교육 여정이 대학원 학위 취득인데 최종 학력이 석사 미만',
      'lifex_education_journey', ['Advanced degrees'], 'highest_education', _EDU_BELOW_MASTER)

_pair('R4-EDU-H03', 'education', 'hard',
      '최종 학력이 무학·초등인데 교육기관 유형이 연구중심·일반 대학·전문대',
      'highest_education', ['No formal', 'Primary'],
      'institution_tier', ['Top-tier research', 'Mid-tier university', 'Community college'])

_pair('R4-EDU-S02', 'education', 'soft',
      '최종 학력이 고졸(중등)인데 교육기관 유형이 4년제 대학 — 중퇴자는 Some college로 표기되어야 하므로 드묾',
      'highest_education', ['Secondary'],
      'institution_tier', ['Top-tier research', 'Mid-tier university'])

_pair('R4-EDU-H04', 'education', 'hard',
      '학위(전문학사 이상) 소지자인데 교육기관 유형이 독학',
      'highest_education', _EDU_DEGREE, 'institution_tier', ['Self-taught'])

_pair('R4-EDU-H05', 'education', 'hard',
      '최종 학력이 무학·초등·중등인데 연구 실적이 학위논문만 있음',
      'highest_education', ['No formal', 'Primary', 'Secondary'], 'research_output', ['Thesis only'])

_pair('R4-EDU-S05', 'education', 'soft',
      '최종 학력이 고졸 이하인데 직무가 임상 — 의사·간호사·치료사 등 면허는 대학 교육 필요',
      'highest_education', ['No formal', 'Primary', 'Secondary'], 'role_function', ['Clinical'])

# --- employment and career ------------------------------------------------------
_rule('R4-EMP-H01', 'employment', 'hard',
      '19-24세인데 해당 분야 경력 11년 이상 — 13세 이전부터 근무해야 함',
      lambda df: _in(df, 'age_bracket', '18-24'),
      lambda df: _in(df, 'years_experience', '11-20', '20+'))

_pair('R4-EMP-S01', 'employment', 'soft',
      '25-34세인데 분야 경력 20년 이상 — 14세 이전부터 근무해야 해 극히 드묾',
      'age_bracket', ['25-34'], 'years_experience', ['20+'])

_pair('R4-EMP-H02', 'employment', 'hard',
      '취업 상태(전일·시간제·자영·프리랜서·무급가족종사)인데 근무 일정이 실업',
      'demo_employment_status', ['Full-time', 'Part-time', 'Self-employed', 'Gig / freelance',
                                 'Unpaid family worker'],
      'lstyle_work_schedule', ['Unemployed'])

_pair('R4-EMP-H03', 'employment', 'hard',
      '실업 상태인데 근무 일정이 9-to-5·교대·온콜 근무',
      'demo_employment_status', ['Unemployed'],
      'lstyle_work_schedule', ['9-to-5', 'Shift work', 'On-call'])

_pair('R4-EMP-S03', 'employment', 'soft',
      '19-24세 임원(VP·C-suite·Director) — 1% 미만',
      'age_bracket', ['18-24'], 'seniority', ['Director', 'VP', 'C-suite'])

_pair('R4-EMP-S04', 'employment', 'soft',
      '19-34세인데 경력 단계가 은퇴 — 1% 미만',
      'age_bracket', ['18-24', '25-34'], 'seniority', ['Retired'])

_IND_COLS = ('ind_retail', 'ind_e_commerce', 'ind_automotive', 'ind_oil_gas', 'ind_utilities',
             'ind_construction', 'ind_travel_tourism', 'ind_restaurants', 'ind_agriculture',
             'ind_food_beverage', 'ind_pharmaceuticals', 'ind_biotech', 'ind_entertainment',
             'ind_publishing', 'ind_advertising', 'ind_government', 'ind_defense', 'ind_nonprofit',
             'ind_transportation', 'ind_shipping', 'ind_chemicals', 'ind_apparel',
             'ind_consumer_electronics', 'ind_semiconductors', 'ind_legal_services',
             'ind_consulting', 'ind_fitness_wellness', 'ind_beauty_cosmetics', 'ind_finance',
             'ind_banking', 'ind_accounting', 'ind_aerospace', 'ind_fine_art', 'ind_healthcare',
             'ind_sports', 'ind_gaming', 'ind_technology', 'ind_marketing_agencies',
             'ind_logistics', 'ind_manufacturing', 'ind_real_estate', 'ind_insurance',
             'ind_hospitality', 'ind_textiles', 'ind_media', 'ind_music', 'ind_energy',
             'ind_mining', 'ind_education', 'ind_telecommunications')


def _n_ind_veteran(df):
    """Number of industries with 'Veteran' experience (needs every ind_* column)."""
    n = pd.Series(0, index=df.index)
    for c in _IND_COLS:
        n = n + _in(df, c, 'Veteran').astype(int)
    return n


_rule('R4-EMP-S05', 'employment', 'soft',
      '19-24세인데 어떤 산업에서 베테랑(10년 이상 수준) 경력 — 1% 미만',
      lambda df: _in(df, 'age_bracket', '18-24'),
      lambda df: _n_ind_veteran(df) >= 1)

# --- income, class and housing ------------------------------------------------------
_rule('R4-INC-H01', 'income', 'hard',
      '가구소득 최하위(<$25k)인데 사회경제 계층이 고소득, 또는 최상위($200k+)인데 저소득',
      lambda df: _in(df, 'demo_household_income', '<$25k', '$200k+'),
      lambda df: (_in(df, 'demo_household_income', '<$25k') & _in(df, 'socioeconomic_band', 'High income'))
      | (_in(df, 'demo_household_income', '$200k+') & _in(df, 'socioeconomic_band', 'Low income')))

_rule('R4-INC-S01', 'income', 'soft',
      '객관적 계층(소득·사회경제)과 주관적 계층이 정반대 극단(저소득↔상층, 고소득↔하층) — 매우 드묾',
      lambda df: _in(df, 'socioeconomic_band', 'Low income', 'High income'),
      lambda df: (_in(df, 'socioeconomic_band', 'Low income') & _in(df, 'kr_subjective_class', 'Upper class'))
      | (_in(df, 'socioeconomic_band', 'High income') & _in(df, 'kr_subjective_class', 'Lower class')))

_pair('R4-LIF-H12', 'life_events', 'hard',
      '겪은 역경이 보호받음(역경 없음)인데 성장 환경이 고난·격동',
      'lifex_adversity_level', ['Sheltered'], 'lifex_childhood_environment', ['Hardship', 'Turbulent'])

# --- languages ------------------------------------------------------------------
_LANG_COLS = ('lang_korean', 'lang_cantonese', 'lang_italian', 'lang_turkish', 'lang_vietnamese',
              'lang_thai', 'lang_indonesian', 'lang_malay', 'lang_dutch', 'lang_polish',
              'lang_ukrainian', 'lang_persian', 'lang_hebrew', 'lang_greek', 'lang_czech',
              'lang_hungarian', 'lang_romanian', 'lang_swedish', 'lang_norwegian', 'lang_danish',
              'lang_finnish', 'lang_tagalog', 'lang_urdu', 'lang_tamil', 'lang_telugu',
              'lang_marathi', 'lang_punjabi', 'lang_gujarati', 'lang_hausa', 'lang_yoruba',
              'lang_igbo', 'lang_amharic', 'lang_zulu', 'lang_afrikaans', 'lang_serbian',
              'lang_croatian', 'lang_bulgarian', 'lang_slovak', 'lang_mandarin', 'lang_spanish',
              'lang_hindi', 'lang_arabic', 'lang_french', 'lang_portuguese', 'lang_bengali',
              'lang_russian', 'lang_japanese', 'lang_german', 'lang_swahili', 'lang_english')
_LANG_LEVELS = ('Native', 'Fluent', 'Conversational', 'Basic', 'None')
for _c in _LANG_COLS:
    _ref(_c, _LANG_LEVELS)


def _n_lang(df, *levels):
    """Number of languages spoken at one of the given levels (needs every lang_* column)."""
    n = pd.Series(0, index=df.index)
    for c in _LANG_COLS:
        n = n + _in(df, c, *levels).astype(int)
    return n


_PRIMARY_TO_COL = {'English': 'lang_english', 'Mandarin': 'lang_mandarin', 'Spanish': 'lang_spanish',
                   'Hindi': 'lang_hindi', 'Arabic': 'lang_arabic', 'French': 'lang_french',
                   'Portuguese': 'lang_portuguese', 'Bengali': 'lang_bengali',
                   'Russian': 'lang_russian', 'Japanese': 'lang_japanese', 'German': 'lang_german',
                   'Swahili': 'lang_swahili', 'Korean': 'lang_korean'}
_ref('primary_language', tuple(_PRIMARY_TO_COL))


def _primary_weak(df):
    p = _s(df, 'primary_language')
    bad = pd.Series(False, index=df.index)
    for lang, col in _PRIMARY_TO_COL.items():
        bad |= (p == lang) & _in(df, col, 'Basic', 'None')
    return bad


_rule('R4-LAN-H01', 'language', 'hard',
      '제1(주)언어로 지정된 언어의 구사 수준이 기초·없음',
      lambda df: _in(df, 'primary_language', *_PRIMARY_TO_COL),
      _primary_weak)

_rule('R4-POP-S03', 'population', 'soft',
      '국내 출생 한국 국민 성인의 주언어가 한국어가 아님 — 1% 미만',
      lambda df: _ni(df, 'primary_language'),
      lambda df: _ni(df, 'primary_language', 'Korean'))

_rule('R4-POP-S04', 'population', 'soft',
      '국내 출생 한국 국민 성인의 한국어 구사가 원어민·유창 수준 미만 — 1% 미만',
      lambda df: _ni(df, 'lang_korean'),
      lambda df: _in(df, 'lang_korean', 'Conversational', 'Basic', 'None'))

_rule('R4-POP-S05', 'population', 'soft',
      '국내 출생 한국인이 한국어 외 언어를 원어민 수준으로 구사 — 1% 미만',
      lambda df: _ni(df, 'lang_korean'),
      lambda df: _n_lang(df, 'Native') - _in(df, 'lang_korean', 'Native').astype(int) >= 1)

_ENG_PROF = _order('english_proficiency', ['Native', 'Fluent (C1-C2)', 'Intermediate (B1-B2)',
                                           'Basic (A1-A2)', 'None'])
_ENG_LANG = _order('lang_english', list(_LANG_LEVELS), name='lang_english_rank')

_rule('R4-LAN-H02', 'language', 'hard',
      '영어 능력(english_proficiency)과 영어 구사(lang_english)가 3단계 이상 어긋남(예: 원어민↔기초, 유창↔없음)',
      lambda df: _rank(df, _ENG_PROF).notna() & _rank(df, _ENG_LANG).notna(),
      lambda df: (_rank(df, _ENG_PROF) - _rank(df, _ENG_LANG)).abs() >= 3)

_rule('R4-LAN-H03', 'language', 'hard',
      '단일언어 사용자인데 두 개 이상 언어를 원어민·유창 수준으로 구사',
      lambda df: _in(df, 'multilingualism', 'Monolingual'),
      lambda df: _n_lang(df, 'Native', 'Fluent') >= 2)

_rule('R4-LAN-H04', 'language', 'hard',
      '이중·다중언어 사용자인데 회화 이상 수준으로 구사하는 언어가 하나뿐',
      lambda df: _in(df, 'multilingualism', 'Bilingual', 'Trilingual+'),
      lambda df: _n_lang(df, 'Native', 'Fluent', 'Conversational') <= 1)

_rule('R4-LAN-H05', 'language', 'hard',
      '평생 배운 언어가 하나뿐인데 회화 이상으로 구사하는 언어가 둘 이상이거나 다중언어 사용자',
      lambda df: _in(df, 'lifex_languages_lifetime', 'One'),
      lambda df: (_n_lang(df, 'Native', 'Fluent', 'Conversational') >= 2)
      | _in(df, 'multilingualism', 'Bilingual', 'Trilingual+'))

# --- personality: same trait, different instruments (opposite extremes only) -------
_EXT = {   # scale family -> (top extreme, bottom extreme)
    'vh': ('Very high', 'Very low'),        # BFI-2 / IPIP / interpersonal (Very high..Very low)
    'cog': ('Very high', 'None'),           # cog_* / health_* (Very high..None)
    'va': ('Signature', 'Absent'),          # trait_* character strengths
}


def _opp(rid, group, description, col_a, fam_a, col_b, fam_b, reverse=False, severity='hard'):
    """Hit when the two attributes sit at opposite extremes of the same trait.

    reverse=True for reverse-keyed pairs (e.g. self-regulation vs immoderation)."""
    ta, ba = _EXT[fam_a]
    tb, bb = _EXT[fam_b]
    _ref(col_a, (ta, ba))
    _ref(col_b, (tb, bb))
    if reverse:
        tb, bb = bb, tb

    def scope(df):
        return _in(df, col_a, ta, ba) | _in(df, col_b, tb, bb)

    def hit(df):
        return (_in(df, col_a, ta) & _in(df, col_b, bb)) | (_in(df, col_a, ba) & _in(df, col_b, tb))
    _rule(rid, group, severity, description, scope, hit)


def _opp_domain(rid, description, domain, facets):
    """BFI-2 domain score is the mean of its facets: domain at one extreme with any facet at
    the other extreme is arithmetically impossible."""
    _ref(domain, ('Very high', 'Very low'))
    for f in facets:
        _ref(f, ('Very high', 'Very low'))

    def scope(df):
        return _in(df, domain, 'Very high', 'Very low')

    def hit(df):
        hi = _in(df, domain, 'Very high')
        lo = _in(df, domain, 'Very low')
        out = pd.Series(False, index=df.index)
        for f in facets:
            out |= (hi & _in(df, f, 'Very low')) | (lo & _in(df, f, 'Very high'))
        return out
    _rule(rid, 'personality', 'hard', description, scope, hit)


_opp_domain('R4-PER-H01', 'BFI-2 개방성 영역 점수와 하위 facet이 정반대 극단(영역=facet 평균이라 불가능)',
            'bfi2_domain_open_mindedness', ('bfi2_facet_intellectual_curiosity',
                                            'bfi2_facet_aesthetic_sensitivity',
                                            'bfi2_facet_creative_imagination'))
_opp_domain('R4-PER-H02', 'BFI-2 성실성 영역 점수와 하위 facet이 정반대 극단',
            'bfi2_domain_conscientiousness', ('bfi2_facet_organization', 'bfi2_facet_productiveness',
                                              'bfi2_facet_responsibility'))
_opp_domain('R4-PER-H03', 'BFI-2 외향성 영역 점수와 하위 facet이 정반대 극단',
            'bfi2_domain_extraversion', ('bfi2_facet_sociability', 'bfi2_facet_assertiveness',
                                         'bfi2_facet_energy_level'))
_opp_domain('R4-PER-H04', 'BFI-2 우호성 영역 점수와 하위 facet이 정반대 극단',
            'bfi2_domain_agreeableness', ('bfi2_facet_compassion', 'bfi2_facet_respectfulness',
                                          'bfi2_facet_trust'))
_opp_domain('R4-PER-H05', 'BFI-2 부정적 정서성 영역 점수와 하위 facet이 정반대 극단',
            'bfi2_domain_negative_emotionality', ('bfi2_facet_anxiety', 'bfi2_facet_depression',
                                                  'bfi2_facet_emotional_volatility'))

_opp('R4-PER-H06', 'personality', '지적 호기심: BFI-2 facet과 IPIP 지성(Intellect)이 정반대 극단',
     'bfi2_facet_intellectual_curiosity', 'vh', 'big5_intellect', 'vh')
_opp('R4-PER-H07', 'personality', '심미적 민감성: BFI-2 facet과 IPIP 예술적 관심이 정반대 극단',
     'bfi2_facet_aesthetic_sensitivity', 'vh', 'big5_artistic_interest', 'vh')
_opp('R4-PER-H08', 'personality', '상상력: BFI-2 창의적 상상력과 IPIP 상상력이 정반대 극단',
     'bfi2_facet_creative_imagination', 'vh', 'big5_imagination', 'vh')
_opp('R4-PER-H09', 'personality', '조직성: BFI-2 Organization과 IPIP 질서정연함이 정반대 극단',
     'bfi2_facet_organization', 'vh', 'big5_orderliness', 'vh')
_opp('R4-PER-H10', 'personality', '책임감: BFI-2 Responsibility와 IPIP 의무감이 정반대 극단',
     'bfi2_facet_responsibility', 'vh', 'big5_dutifulness', 'vh')
_opp('R4-PER-H11', 'personality', '사교성: BFI-2 Sociability와 IPIP 군거성(Gregariousness)이 정반대 극단',
     'bfi2_facet_sociability', 'vh', 'big5_gregariousness', 'vh')
_opp('R4-PER-H12', 'personality', '주장성: BFI-2 Assertiveness와 IPIP Assertiveness가 정반대 극단',
     'bfi2_facet_assertiveness', 'vh', 'big5_assertiveness', 'vh')
_opp('R4-PER-H14', 'personality', '활동 에너지: BFI-2 Energy Level과 IPIP 활동 수준이 정반대 극단',
     'bfi2_facet_energy_level', 'vh', 'big5_activity_level', 'vh')
_opp('R4-PER-H15', 'personality', '동정심: BFI-2 Compassion과 IPIP 공감(Sympathy)이 정반대 극단',
     'bfi2_facet_compassion', 'vh', 'big5_sympathy', 'vh')
_opp('R4-PER-H16', 'personality', '신뢰: BFI-2 Trust와 IPIP Trust가 정반대 극단',
     'bfi2_facet_trust', 'vh', 'big5_trust', 'vh')
_opp('R4-PER-H17', 'personality', '불안: BFI-2 Anxiety와 IPIP Anxiety가 정반대 극단',
     'bfi2_facet_anxiety', 'vh', 'big5_anxiety', 'vh')
_opp('R4-PER-H18', 'personality', '우울: BFI-2 Depression과 IPIP Depression이 정반대 극단',
     'bfi2_facet_depression', 'vh', 'big5_depression', 'vh')
_opp('R4-PER-H20', 'personality', '자기통제: IPIP 자기절제(Self-discipline)와 성격강점 절제(Discipline)가 정반대 극단',
     'big5_self_discipline', 'vh', 'trait_discipline', 'va')

_opp('R4-PER-H21', 'personality', '호기심: 성격강점 호기심과 인지성향 호기심(cog)이 정반대 극단',
     'trait_curiosity', 'va', 'cog_curiosity', 'cog')
_opp('R4-PER-H22', 'personality', '개방적 사고: 성격강점 개방성과 인지성향 개방성(cog)이 정반대 극단',
     'trait_open_mindedness', 'va', 'cog_open_mindedness', 'cog')
_opp('R4-PER-H23', 'personality', '낙관: 성격강점 희망·낙관과 인지성향 낙관(cog)이 정반대 극단',
     'trait_hope_optimism', 'va', 'cog_optimism', 'cog')
_opp('R4-PER-H24', 'personality', '겸손: 성격강점 겸손과 IPIP 겸손(Modesty)이 정반대 극단',
     'trait_humility', 'va', 'big5_modesty', 'vh')
_opp('R4-PER-H25', 'personality', '정직: 성격강점 정직과 IPIP 도덕성(Morality)이 정반대 극단',
     'trait_honesty', 'va', 'big5_morality', 'vh')
_opp('R4-PER-H26', 'personality', '신중함: 성격강점 신중함(Prudence)과 IPIP 신중성(Cautiousness)이 정반대 극단',
     'trait_prudence', 'va', 'big5_cautiousness', 'vh')
_opp('R4-PER-H27', 'personality', '창의성: 성격강점 창의성과 BFI-2 창의적 상상력이 정반대 극단',
     'trait_creativity', 'va', 'bfi2_facet_creative_imagination', 'vh')
_opp('R4-PER-H28', 'personality', '심미안: 성격강점 아름다움 감상과 BFI-2 심미적 민감성이 정반대 극단',
     'trait_appreciation_of_beauty', 'va', 'bfi2_facet_aesthetic_sensitivity', 'vh')
_opp('R4-PER-H29', 'personality', '친절: 성격강점 친절과 IPIP 이타성(Altruism)이 정반대 극단',
     'trait_kindness', 'va', 'big5_altruism', 'vh')
_opp('R4-PER-H30', 'personality', '공감: 성격강점 공감과 IPIP 공감(Sympathy)이 정반대 극단',
     'trait_empathy', 'va', 'big5_sympathy', 'vh')
_opp('R4-PER-H31', 'personality', '자기조절 강점이 최상인데 IPIP 무절제(Immoderation)도 최상(또는 둘 다 최하) — 역방향 척도',
     'trait_self_regulation', 'va', 'big5_immoderation', 'vh', reverse=True)
_opp('R4-PER-H32', 'personality', '회복탄력성 강점이 최상인데 IPIP 취약성(Vulnerability)도 최상(또는 둘 다 최하) — 역방향 척도',
     'trait_resilience', 'va', 'big5_vulnerability', 'vh', reverse=True)
_opp('R4-PER-H34', 'personality', '미루기 성향이 최상인데 IPIP 자기절제도 최상(또는 둘 다 최하) — 역방향 척도',
     'cog_procrastination', 'cog', 'big5_self_discipline', 'vh', reverse=True)
_opp('R4-PER-H35', 'personality', '종결 욕구가 최상인데 모호함 인내도 최상(또는 둘 다 최하) — 역방향 척도',
     'need_for_closure', 'vh', 'cog_ambiguity_tolerance', 'cog', reverse=True)

_rule('R4-PER-H36', 'personality', 'hard',
      '세부 지향성(cog)과 큰 그림/세부 선호가 정반대 극단(세부 지향 없음↔세부 집착, 최상↔큰 그림만)',
      lambda df: _in(df, 'cog_detail_orientation', 'Very high', 'None'),
      lambda df: (_in(df, 'cog_detail_orientation', 'None') & _in(df, 'cog_big_picture_vs_detail', 'Detail-obsessed'))
      | (_in(df, 'cog_detail_orientation', 'Very high') & _in(df, 'cog_big_picture_vs_detail', 'Big-picture only')))

_pair('R4-PER-H37', 'personality', 'hard',
      '의사결정 방식이 숙고형(Deliberative)인데 결정 속도가 즉흥적 결정(Snap decisions)',
      'decision_style', ['Deliberative'], 'cog_decision_speed', ['Snap decisions'])

_DOM_MAP = {'High openness': 'bfi2_domain_open_mindedness',
            'High conscientiousness': 'bfi2_domain_conscientiousness',
            'High extraversion': 'bfi2_domain_extraversion',
            'High agreeableness': 'bfi2_domain_agreeableness',
            'High neuroticism': 'bfi2_domain_negative_emotionality'}
_ref('dominant_trait', tuple(_DOM_MAP))
for _c in _DOM_MAP.values():
    _ref(_c, ('Very low',))


def _dominant_bad(df):
    d = _s(df, 'dominant_trait')
    bad = pd.Series(False, index=df.index)
    for v, col in _DOM_MAP.items():
        bad |= (d == v) & _in(df, col, 'Very low')
    return bad


_rule('R4-PER-H38', 'personality', 'hard',
      '대표 성격 특질이 "높은 X"인데 BFI-2의 해당 X 영역 점수가 매우 낮음',
      lambda df: _in(df, 'dominant_trait', *_DOM_MAP),
      _dominant_bad)

_rule('R4-PER-H39', 'personality', 'hard',
      '신경유형이 불안 성향(Anxiety-prone)인데 BFI-2·IPIP 불안 점수가 매우 낮음',
      lambda df: _in(df, 'neurotype', 'Anxiety-prone'),
      lambda df: _in(df, 'bfi2_facet_anxiety', 'Very low') | _in(df, 'big5_anxiety', 'Very low'))

_MBTI = ('INTJ — Architect', 'INTP — Logician', 'ENTJ — Commander', 'ENTP — Debater',
         'INFJ — Advocate', 'INFP — Mediator', 'ENFJ — Protagonist', 'ENFP — Campaigner',
         'ISTJ — Logistician', 'ISFJ — Defender', 'ESTJ — Executive', 'ESFJ — Consul',
         'ISTP — Virtuoso', 'ISFP — Adventurer', 'ESTP — Entrepreneur', 'ESFP — Entertainer')
_ref('mbti_type', _MBTI)


def _mbti_letter(df, pos, letter):
    m = [v for v in _MBTI if v[pos] == letter]
    return _in(df, 'mbti_type', *m)


_rule('R4-PER-S01', 'personality', 'soft',
      'MBTI 내향(I)인데 BFI-2 외향성 매우 높음, 또는 외향(E)인데 매우 낮음 — E/I와 외향성은 강하게 상관(r≈.7)',
      lambda df: _in(df, 'mbti_type', *_MBTI),
      lambda df: (_mbti_letter(df, 0, 'I') & _in(df, 'bfi2_domain_extraversion', 'Very high'))
      | (_mbti_letter(df, 0, 'E') & _in(df, 'bfi2_domain_extraversion', 'Very low')))

_rule('R4-PER-S02', 'personality', 'soft',
      'MBTI 직관(N)인데 BFI-2 개방성 매우 낮음, 또는 감각(S)인데 매우 높음 — S/N과 개방성은 강하게 상관',
      lambda df: _in(df, 'mbti_type', *_MBTI),
      lambda df: (_mbti_letter(df, 1, 'N') & _in(df, 'bfi2_domain_open_mindedness', 'Very low'))
      | (_mbti_letter(df, 1, 'S') & _in(df, 'bfi2_domain_open_mindedness', 'Very high')))

# --- health and disability ---------------------------------------------------------
_pair('R4-HEA-H01', 'health', 'hard',
      '시각이 실명인데 운전 상태가 매일·가끔 운전 — 운전면허 시력 기준 미달',
      'health_vision', ['Blind'], 'demo_driver_status', ['Daily driver', 'Occasional driver'])

_pair('R4-HEA-S01', 'health', 'soft',
      '저시력인데 매일 운전 — 면허 시력 기준상 매우 드묾',
      'health_vision', ['Low vision'], 'demo_driver_status', ['Daily driver'])

_pair('R4-HEA-H02', 'health', 'hard',
      '시각이 실명인데 장애 상태가 장애 없음 — 실명은 감각(시각) 장애',
      'health_vision', ['Blind'], 'demo_disability_status', ['No disability'])

_rule('R4-HEA-S02', 'health', 'soft',
      '장애 유형이 감각 장애인데 시력·청력이 모두 정상 — 다른 감각 장애는 극히 드묾',
      lambda df: _in(df, 'demo_disability_status', 'Sensory'),
      lambda df: _in(df, 'health_vision', 'Normal', 'Corrected') & _in(df, 'health_hearing', 'Normal'))

_rule('R4-HEA-H03', 'health', 'hard',
      '만성질환 없음인데 장애 상태가 만성질환이거나 건강 여정이 만성질환 관리 중',
      lambda df: _in(df, 'health_chronic_condition', 'None'),
      lambda df: _in(df, 'demo_disability_status', 'Chronic illness')
      | _in(df, 'lifex_health_journey', 'Manages chronic condition'))

_rule('R4-HEA-H04', 'health', 'hard',
      '정신건강이 번영(Thriving)인데 PHQ-9 중등도-심각 이상 우울 또는 GAD-7 심각 불안',
      lambda df: _in(df, 'health_mental_health', 'Thriving'),
      lambda df: _in(df, 'kr_depression_level', 'Moderately severe', 'Severe')
      | _in(df, 'kr_anxiety_level', 'Severe'))

_pair('R4-HEA-H05', 'health', 'hard',
      'PHQ-9 심각 우울(우울 기분·흥미 상실이 거의 매일)인데 전반적 행복이 매우 행복',
      'kr_depression_level', ['Severe'], 'kr_happiness', ['Very happy'])

_rule('R4-HEA-S05', 'health', 'soft',
      '심각 우울(PHQ-9)인데 삶에 매우 만족하거나 에너지 수준이 매우 높음 — 피로·불만족이 핵심 증상이라 매우 드묾',
      lambda df: _in(df, 'kr_depression_level', 'Severe'),
      lambda df: _in(df, 'kr_life_satisfaction', 'Very satisfied') | _in(df, 'health_energy_level', 'Very high'))

_rule('R4-HEA-H06', 'health', 'hard',
      '신경유형(neurotype)과 신경다양성(health_neurodivergence)이 서로 다름: 한쪽은 ADHD·자폐·난독, 다른 쪽은 정형',
      lambda df: _in(df, 'neurotype', 'ADHD', 'Autistic', 'Dyslexic')
      | _in(df, 'health_neurodivergence', 'ADHD', 'Autistic', 'Dyslexic'),
      lambda df: (_in(df, 'neurotype', 'ADHD', 'Autistic', 'Dyslexic') & _in(df, 'health_neurodivergence', 'Neurotypical'))
      | (_in(df, 'health_neurodivergence', 'ADHD', 'Autistic', 'Dyslexic') & _in(df, 'neurotype', 'Neurotypical')))

_rule('R4-HEA-H07', 'health', 'hard',
      'ADHD(신경유형·신경다양성)인데 주의력 상태 없음, 또는 주의력 질환 진단인데 신경 정형(두 항목 모두)',
      lambda df: _in(df, 'health_neurodivergence', 'ADHD') | _in(df, 'health_attention_condition', 'Diagnosed'),
      lambda df: (_in(df, 'health_neurodivergence', 'ADHD') & _in(df, 'health_attention_condition', 'None'))
      | (_in(df, 'health_attention_condition', 'Diagnosed') & _in(df, 'health_neurodivergence', 'Neurotypical')
         & _in(df, 'neurotype', 'Neurotypical')))

_pair('R4-HEA-H08', 'health', 'hard',
      '운동을 전혀 안 하는데 체력 수준이 운동선수급',
      'lstyle_exercise_freq', ['Never'], 'health_fitness_level', ['Athlete'])

_pair('R4-HEA-S07', 'health', 'soft',
      '운동을 거의 안 하는데 체력 수준이 운동선수급 — 매우 드묾',
      'lstyle_exercise_freq', ['Rarely'], 'health_fitness_level', ['Athlete'])

_pair('R4-HEA-H09', 'health', 'hard',
      '매일 운동하는데 체력 수준이 좌식(Sedentary)',
      'lstyle_exercise_freq', ['Daily'], 'health_fitness_level', ['Sedentary'])

_pair('R4-HEA-S09', 'health', 'soft',
      '85세 이상인데 체력 수준이 운동선수급 — 1% 미만',
      'age_bracket', ['85+'], 'health_fitness_level', ['Athlete'])

_pair('R4-HEA-S10', 'health', 'soft',
      '한국 국민인데 건강보험 미가입 — 국민건강보험·의료급여로 사실상 전 국민 적용',
      'demo_citizenship_status', ['Citizen by birth', 'Dual national'], 'health_insurance_status', ['Uninsured'])

# --- religion ------------------------------------------------------------------------
_REL_AFFIL = ('Christian', 'Muslim', 'Hindu', 'Buddhist', 'Jewish', 'Sikh', 'Catholic', 'Protestant',
              'Won Buddhist', 'Korean new religion')
_REL_NONE = ('None', 'Atheist / agnostic')

_pair('R4-REL-H01', 'religion', 'hard',
      '종교성이 독실·계율 준수인데 종교 소속이 없음·무신론/불가지론',
      'religiosity', ['Devout', 'Observant'], 'demo_religion_affiliation', _REL_NONE)

_pair('R4-REL-H02', 'religion', 'hard',
      '신앙 여정이 평생 세속적인데 특정 종교(개신교·천주교·불교 등)에 소속',
      'lifex_faith_journey', ['Always secular'], 'demo_religion_affiliation', _REL_AFFIL)

_pair('R4-REL-H03', 'religion', 'hard',
      '신앙 여정이 평생 세속적인데 종교성이 독실·계율 준수',
      'lifex_faith_journey', ['Always secular'], 'religiosity', ['Devout', 'Observant'])

_rule('R4-REL-H04', 'religion', 'hard',
      '신앙 여정이 평생 신앙·신앙 회복인데 종교 없음·무신론이거나 종교성이 세속적',
      lambda df: _in(df, 'lifex_faith_journey', 'Lifelong faith', 'Returned to faith'),
      lambda df: _in(df, 'demo_religion_affiliation', *_REL_NONE) | _in(df, 'religiosity', 'Secular'))

_pair('R4-REL-S01', 'religion', 'soft',
      '신앙 여정이 개종인데 현재 종교 없음·무신론 — 탈종교는 Lapsed로 표기되므로 드묾',
      'lifex_faith_journey', ['Converted'], 'demo_religion_affiliation', _REL_NONE)

_pair('R4-REL-S02', 'religion', 'soft',
      '종교성이 세속적인데 특정 종교에 소속 — 명목상 신자는 Nominal로 표기되므로 드묾',
      'religiosity', ['Secular'], 'demo_religion_affiliation', _REL_AFFIL)

_rule('R4-REL-H05', 'religion', 'hard',
      '종교성이 독실한데 영성·신앙 가치가 무관하거나 성격강점 영성이 전무',
      lambda df: _in(df, 'religiosity', 'Devout'),
      lambda df: _in(df, 'val_spirituality_faith', 'Irrelevant') | _in(df, 'trait_spirituality', 'Absent'))

_pair('R4-REL-S04', 'religion', 'soft',
      '종교 소속이 유교인데 유교적 생활 방식을 따르지 않음 — 매우 드묾',
      'demo_religion_affiliation', ['Confucian'], 'kr_confucian_way_of_life', ['No'])

_rule('R4-POP-S06', 'population', 'soft',
      '국내 출생 한국인이 이슬람·힌두·시크·유대교 신자 — 1% 미만',
      lambda df: _ni(df, 'demo_religion_affiliation'),
      lambda df: _in(df, 'demo_religion_affiliation', 'Muslim', 'Hindu', 'Sikh', 'Jewish'))

# --- values: same value measured twice (opposite extremes only) ------------------------
_VAL_SCHWARTZ = [
    ('R4-VAL-H01', 'val_achievement', 'schwartz_value_achievement', '성취'),
    ('R4-VAL-H02', 'val_power_influence', 'schwartz_value_power', '권력'),
    ('R4-VAL-H03', 'val_security_stability', 'schwartz_value_security', '안전·안정'),
    ('R4-VAL-H04', 'val_tradition', 'schwartz_value_tradition', '전통'),
    ('R4-VAL-H05', 'val_helping_others', 'schwartz_value_benevolence', '타인 돕기·박애'),
    ('R4-VAL-H06', 'val_independence', 'schwartz_value_self_direction', '독립·자기주도'),
    ('R4-VAL-H07', 'val_adventure', 'schwartz_value_stimulation', '모험·자극'),
    ('R4-VAL-H08', 'val_fun_enjoyment', 'schwartz_value_hedonism', '즐거움·쾌락'),
    ('R4-VAL-H09', 'val_equality', 'schwartz_value_universalism', '평등·보편주의'),
]


def _val_opp(rid, vcol, scol, name):
    _ref(vcol, ('Core value', 'Irrelevant'))
    _ref(scol, ('Very high', 'Very low'))
    _rule(rid, 'values', 'hard',
          '%s 가치: 가치 우선순위(val)와 Schwartz 가치가 정반대 극단(핵심 가치↔매우 낮음, 무관↔매우 높음)' % name,
          lambda df: _in(df, vcol, 'Core value', 'Irrelevant') | _in(df, scol, 'Very high', 'Very low'),
          lambda df: (_in(df, vcol, 'Core value') & _in(df, scol, 'Very low'))
          | (_in(df, vcol, 'Irrelevant') & _in(df, scol, 'Very high')))


for _rid, _v, _sc, _nm in _VAL_SCHWARTZ:
    _val_opp(_rid, _v, _sc, _nm)

_PRIO_MAP = {'Achievement': ('val_achievement', 'schwartz_value_achievement'),
             'Security': ('val_security_stability', 'schwartz_value_security'),
             'Autonomy': ('val_independence', 'schwartz_value_self_direction'),
             'Community': ('val_community', 'schwartz_value_benevolence'),
             'Novelty': ('val_adventure', 'schwartz_value_stimulation'),
             'Tradition': ('val_tradition', 'schwartz_value_tradition')}
_ref('values_priority', tuple(_PRIO_MAP))
for _vc, _scc in _PRIO_MAP.values():
    _ref(_vc, ('Irrelevant',))
    _ref(_scc, ('Very low',))


def _prio_bad(df):
    p = _s(df, 'values_priority')
    bad = pd.Series(False, index=df.index)
    for v, (vc, sc) in _PRIO_MAP.items():
        bad |= (p == v) & (_in(df, vc, 'Irrelevant') | _in(df, sc, 'Very low'))
    return bad


_rule('R4-VAL-H10', 'values', 'hard',
      '최우선 가치(values_priority)로 꼽은 가치가 가치 목록에서 무관하거나 Schwartz 점수가 매우 낮음',
      lambda df: _in(df, 'values_priority', *_PRIO_MAP),
      _prio_bad)

_pair('R4-VAL-S01', 'values', 'soft',
      '영성·신앙이 핵심 가치인데 무신론/불가지론자 — 매우 드묾',
      'val_spirituality_faith', ['Core value'], 'demo_religion_affiliation', ['Atheist / agnostic'])

# --- attitudes vs behaviours / preferences ------------------------------------------------
_rule('R4-ATT-H01', 'attitude_behaviour', 'hard',
      '원격근무 태도와 근무 장소 선호가 정반대(원격근무 반대↔강한 원격 선호, 열광↔강한 사무실 선호)',
      lambda df: _in(df, 'att_remote_work', 'Opposed', 'Enthusiast'),
      lambda df: (_in(df, 'att_remote_work', 'Opposed') & _in(df, 'pref_work_location', 'Strongly remote'))
      | (_in(df, 'att_remote_work', 'Enthusiast') & _in(df, 'pref_work_location', 'Strongly office')))

_rule('R4-ATT-H02', 'attitude_behaviour', 'hard',
      '사무실 출근 태도와 근무 장소 선호가 정반대(사무실 근무 반대↔강한 사무실 선호, 열광↔강한 원격 선호)',
      lambda df: _in(df, 'att_working_from_office', 'Opposed', 'Enthusiast'),
      lambda df: (_in(df, 'att_working_from_office', 'Opposed') & _in(df, 'pref_work_location', 'Strongly office'))
      | (_in(df, 'att_working_from_office', 'Enthusiast') & _in(df, 'pref_work_location', 'Strongly remote')))

_rule('R4-ATT-H03', 'attitude_behaviour', 'hard',
      '위험 감수 태도와 위험 성향이 정반대(위험 감수 열광↔위험 회피, 반대↔위험 추구)',
      lambda df: _in(df, 'att_risk_taking', 'Enthusiast', 'Opposed'),
      lambda df: (_in(df, 'att_risk_taking', 'Enthusiast') & _in(df, 'risk_tolerance', 'Risk-averse'))
      | (_in(df, 'att_risk_taking', 'Opposed') & _in(df, 'risk_tolerance', 'Risk-seeking')))

_DOSPERT = ('dospert_financial_risk_tolerance', 'dospert_ethical_risk_tolerance',
            'dospert_recreational_risk_tolerance', 'dospert_social_risk_tolerance',
            'dospert_health_safety_risk_tolerance')
for _c in _DOSPERT:
    _ref(_c, ('Very high', 'Very low'))


def _all_dospert(df, level):
    return _all(*[_in(df, c, level) for c in _DOSPERT])


_rule('R4-ATT-H04', 'attitude_behaviour', 'hard',
      '전반적 위험 성향이 위험 추구인데 DOSPERT 5개 영역 모두 매우 낮음(또는 위험 회피인데 모두 매우 높음)',
      lambda df: _in(df, 'risk_tolerance', 'Risk-seeking', 'Risk-averse'),
      lambda df: (_in(df, 'risk_tolerance', 'Risk-seeking') & _all_dospert(df, 'Very low'))
      | (_in(df, 'risk_tolerance', 'Risk-averse') & _all_dospert(df, 'Very high')))

_pair('R4-ATT-H05', 'attitude_behaviour', 'hard',
      '비건 식단인데 비건주의에 반대(Opposed)',
      'lstyle_diet_type', ['Vegan'], 'att_veganism', ['Opposed'])

_pair('R4-ATT-H06', 'attitude_behaviour', 'hard',
      '정치 성향이 비정치적(Apolitical)인데 정치 참여가 활동가(Activist)',
      'political_lean', ['Apolitical'], 'demo_political_engagement', ['Activist'])

_pair('R4-ATT-H07', 'attitude_behaviour', 'hard',
      '신기술 수용이 최첨단(Bleeding edge)인데 신기술에 반대',
      'pref_early_vs_late', ['Bleeding edge'], 'att_new_technology', ['Opposed'])

_pair('R4-ATT-S03', 'attitude_behaviour', 'soft',
      '미디어 소비가 소셜미디어 중심인데 소셜미디어에 반대 — 매우 드묾',
      'media_diet', ['Social-media-heavy'], 'att_social_media', ['Opposed'])

_pair('R4-ATT-S05', 'attitude_behaviour', 'soft',
      '동성애자(게이·레즈비언)인데 동성 관계가 항상 잘못이라고 응답 — 매우 드묾',
      'demo_sexual_orientation', ['Gay / lesbian'], 'kr_attitude_same_sex_relations', ['Always wrong'])

# --- preferences that restate personality / habits -------------------------------------------
_rule('R4-PRF-H01', 'preferences', 'hard',
      '일상 선호가 정반대: 루틴 갈망인데 항상 새로움, 또는 다양성 갈망인데 항상 익숙함',
      lambda df: _in(df, 'pref_routine_vs_variety', 'Craves routine', 'Craves variety'),
      lambda df: (_in(df, 'pref_routine_vs_variety', 'Craves routine') & _in(df, 'pref_novelty_vs_familiarity', 'Always novel'))
      | (_in(df, 'pref_routine_vs_variety', 'Craves variety') & _in(df, 'pref_novelty_vs_familiarity', 'Always familiar')))

_rule('R4-PRF-H02', 'preferences', 'hard',
      '논리/직관 선호와 의사결정 방식이 정반대(순수 논리↔직관형, 순수 직관↔분석형)',
      lambda df: _in(df, 'pref_logic_vs_intuition', 'Pure logic', 'Pure intuition'),
      lambda df: (_in(df, 'pref_logic_vs_intuition', 'Pure logic') & _in(df, 'decision_style', 'Intuitive'))
      | (_in(df, 'pref_logic_vs_intuition', 'Pure intuition') & _in(df, 'decision_style', 'Analytical')))

_rule('R4-PRF-H03', 'preferences', 'hard',
      '리더/팔로워 선호와 리더십 강점이 정반대(항상 주도↔리더십 전무, 따르기 선호↔리더십 대표 강점)',
      lambda df: _in(df, 'pref_lead_vs_follow', 'Always leads', 'Prefers to follow'),
      lambda df: (_in(df, 'pref_lead_vs_follow', 'Always leads') & _in(df, 'trait_leadership', 'Absent'))
      | (_in(df, 'pref_lead_vs_follow', 'Prefers to follow') & _in(df, 'trait_leadership', 'Signature')))

_pair('R4-PRF-H04', 'preferences', 'hard',
      '매우 경쟁 지향인데 성격강점 경쟁심이 전무',
      'pref_competition_vs_collab', ['Highly competitive'], 'trait_competitiveness', ['Absent'])

_rule('R4-PRF-H05', 'preferences', 'hard',
      '대규모 모임 선호와 IPIP 군거성이 정반대(큰 모임 좋아함↔매우 낮음, 1:1만↔매우 높음)',
      lambda df: _in(df, 'pref_big_group_vs_one_on_one', 'Loves big groups', 'One-on-one only'),
      lambda df: (_in(df, 'pref_big_group_vs_one_on_one', 'Loves big groups') & _in(df, 'big5_gregariousness', 'Very low'))
      | (_in(df, 'pref_big_group_vs_one_on_one', 'One-on-one only') & _in(df, 'big5_gregariousness', 'Very high')))

_rule('R4-PRF-H06', 'preferences', 'hard',
      '정보 상세도 선호와 큰 그림/세부 성향이 정반대(요지만↔세부 집착, 모든 세부↔큰 그림만)',
      lambda df: _in(df, 'pref_detail_brief_vs_full', 'Just the gist', 'Every detail'),
      lambda df: (_in(df, 'pref_detail_brief_vs_full', 'Just the gist') & _in(df, 'cog_big_picture_vs_detail', 'Detail-obsessed'))
      | (_in(df, 'pref_detail_brief_vs_full', 'Every detail') & _in(df, 'cog_big_picture_vs_detail', 'Big-picture only')))

_pair('R4-PRF-H07', 'preferences', 'hard',
      '미디어 소비가 최소(Minimal)인데 매일 둠스크롤링',
      'media_diet', ['Minimal'], 'habit_doomscrolling', ['Daily'])

_pair('R4-PRF-H08', 'preferences', 'hard',
      '실명인데 접근성 요구가 없음',
      'health_vision', ['Blind'], 'accessibility_needs', ['None'])

# --- wellbeing ------------------------------------------------------------------------------
_rule('R4-WEL-S01', 'wellbeing', 'soft',
      '행복감과 삶의 만족도가 정반대 극단(매우 행복↔매우 불만족, 매우 불행↔매우 만족) — 매우 드묾',
      lambda df: _in(df, 'kr_happiness', 'Very happy', 'Very unhappy'),
      lambda df: (_in(df, 'kr_happiness', 'Very happy') & _in(df, 'kr_life_satisfaction', 'Very dissatisfied'))
      | (_in(df, 'kr_happiness', 'Very unhappy') & _in(df, 'kr_life_satisfaction', 'Very satisfied')))

# --- developer / AI tooling --------------------------------------------------------------------
# 'Not applicable' / 'Not a developer' are never a contradiction by themselves; they are not used
# as hit values below.
_CODE_AI_NONE = ('Never used', 'Actively avoids')
_AI_TASK_CODING = ('ai_task_code_generation', 'ai_task_code_review', 'ai_task_codebase_learning',
                   'ai_task_debugging_fixing', 'ai_task_deployment_monitoring', 'ai_task_documentation',
                   'ai_task_refactoring', 'ai_task_testing')
for _c in _AI_TASK_CODING:
    _ref(_c, ('Currently mostly AI-assisted', 'Currently partially AI-assisted'))

_rule('R4-DEV-H01', 'developer_ai', 'hard',
      '코딩 AI를 써본 적 없음·적극 회피인데 코딩 작업(생성·리뷰·디버깅·리팩터링·테스트 등)을 현재 AI 보조로 수행',
      lambda df: _in(df, 'coding_ai_usage_frequency', *_CODE_AI_NONE),
      lambda df: _any(*[_in(df, c, 'Currently mostly AI-assisted', 'Currently partially AI-assisted')
                        for c in _AI_TASK_CODING]))

_AI_FREQ = _order('coding_ai_usage_frequency', ['Daily', 'Weekly', 'Monthly or infrequent',
                                                'Tried but not active', 'Never used'], name='ai_freq')
_AGENT_FREQ = _order('coding_agent_usage_frequency', ['Daily', 'Weekly', 'Monthly or infrequent',
                                                      'Experimental only', 'Not yet but interested'],
                     name='agent_freq')
_ref('coding_ai_usage_frequency', ('Actively avoids',))
_ref('coding_agent_usage_frequency', ('No plans',))


def _ai_rank(df):
    r = _rank(df, _AI_FREQ)
    return r.where(~_in(df, 'coding_ai_usage_frequency', 'Actively avoids'), 4.0)


def _agent_rank(df):
    r = _rank(df, _AGENT_FREQ)
    return r.where(~_in(df, 'coding_agent_usage_frequency', 'No plans'), 4.0)


_rule('R4-DEV-H02', 'developer_ai', 'hard',
      '코딩 에이전트(코딩 AI의 하위 형태) 사용 빈도가 코딩 AI 전체 사용 빈도보다 잦음',
      lambda df: _agent_rank(df).notna() & _ai_rank(df).notna(),
      lambda df: _agent_rank(df) < _ai_rank(df))

_pair('R4-DEV-H03', 'developer_ai', 'hard',
      '에이전트 실패 허용도가 "에이전트 미사용"인데 코딩 에이전트를 매일·매주·매월 사용',
      'coding_agent_failure_tolerance', ['Does not use agents'],
      'coding_agent_usage_frequency', ['Daily', 'Weekly', 'Monthly or infrequent'])

_pair('R4-DEV-H04', 'developer_ai', 'hard',
      '코딩 AI를 써본 적 없음·회피인데 AI 도구 학습 경로가 "AI 코딩 도구 자체 사용"',
      'coding_ai_usage_frequency', _CODE_AI_NONE, 'coding_ai_learning_source', ['AI-enabled coding tools'])

_pair('R4-DEV-H05', 'developer_ai', 'hard',
      '코딩 AI를 써본 적 없음·회피인데 에이전트로 생산성이 크게·중간 정도 향상',
      'coding_ai_usage_frequency', _CODE_AI_NONE,
      'coding_agent_workflow_impact', ['Major productivity increase', 'Moderate productivity increase'])

_pair('R4-DEV-H06', 'developer_ai', 'hard',
      '코딩 AI를 써본 적 없음·회피인데 개발 도움 요청 시 AI에게 먼저 물음',
      'coding_ai_usage_frequency', _CODE_AI_NONE, 'developer_help_seeking_preference', ['Ask AI first'])

_rule('R4-DEV-H07', 'developer_ai', 'hard',
      'GitHub Copilot 파워유저·정기 사용자인데 코딩 AI를 써본 적 없음·회피',
      lambda df: _in(df, 'tool_github_copilot', 'Power user', 'Regular'),
      lambda df: _in(df, 'coding_ai_usage_frequency', *_CODE_AI_NONE))

_rule('R4-DEV-H08', 'developer_ai', 'hard',
      '코딩 AI 호감도와 산출물 신뢰가 정반대 극단(매우 비호의↔가벼운 검토로 강하게 신뢰, 매우 호의↔전혀 불신)',
      lambda df: _in(df, 'coding_ai_sentiment', 'Very unfavorable', 'Very favorable'),
      lambda df: (_in(df, 'coding_ai_sentiment', 'Very unfavorable') & _in(df, 'coding_ai_output_trust', 'Strongly trusts with light review'))
      | (_in(df, 'coding_ai_sentiment', 'Very favorable') & _in(df, 'coding_ai_output_trust', 'Does not trust')))

_rule('R4-DEV-H09', 'developer_ai', 'hard',
      '에이전트 자율성 선호와 통제 기대가 정반대(자율 실행↔읽기 전용 선호, 자율성 없음↔광범위 자율 작업 허용)',
      lambda df: _in(df, 'coding_agent_autonomy_preference', 'Autonomous execution', 'No agent autonomy'),
      lambda df: (_in(df, 'coding_agent_autonomy_preference', 'Autonomous execution')
                  & _in(df, 'coding_agent_control_expectation', 'Read-only assistance preferred'))
      | (_in(df, 'coding_agent_autonomy_preference', 'No agent autonomy')
         & _in(df, 'coding_agent_control_expectation', 'Agent may run broad tasks autonomously')))

_pair('R4-DEV-H10', 'developer_ai', 'hard',
      '개발자 아님(Not a developer)인데 개발자 역할 유형이 프론트·백엔드·풀스택·모바일·데이터/ML·DevOps 등 개발 직무',
      'dev_professional_status', ['Not a developer'],
      'dev_role_archetype', ['Front-end developer', 'Back-end developer', 'Full-stack developer',
                             'Mobile developer', 'Data / ML developer', 'DevOps / SRE / infrastructure',
                             'Security engineer', 'Game developer', 'Embedded / systems developer'])

_rule('R4-DEV-H11', 'developer_ai', 'hard',
      '전문 개발자인데 코딩 활동 없음 또는 코딩 기술 없음',
      lambda df: _in(df, 'dev_professional_status', 'Professional developer'),
      lambda df: _in(df, 'open_source_activity_level', 'No coding activity') | _in(df, 'skill_coding', 'None'))

_rule('R4-DEV-H12', 'developer_ai', 'hard',
      '코딩 활동 없음인데 GitHub에서 코드 푸시·PR 기여·저장소 관리 또는 코딩 AI 매일 사용',
      lambda df: _in(df, 'open_source_activity_level', 'No coding activity'),
      lambda df: _in(df, 'github_contribution_mode', 'Code pusher', 'Pull request contributor', 'Repository maintainer')
      | _in(df, 'coding_ai_usage_frequency', 'Daily'))

_pair('R4-DEV-H13', 'developer_ai', 'hard',
      'PR을 거의 쓰지 않는데 GitHub 기여 방식이 PR 기여자',
      'pr_collaboration_style', ['Rare PR usage'], 'github_contribution_mode', ['Pull request contributor'])

_pair('R4-DEV-H14', 'developer_ai', 'hard',
      'Stack Overflow를 써본 적 없는데 참여 방식이 읽기·투표·댓글·질문·답변·관리',
      'stackoverflow_usage_pattern', ['Never used'],
      'stackoverflow_participation_style', ['Reads / searches only', 'Votes / bookmarks', 'Comments / discusses',
                                            'Asks questions', 'Answers questions', 'Moderates / curates'])

_rule('R4-DEV-H15', 'developer_ai', 'hard',
      'IC/관리자 지향과 개발자 역할 유형이 충돌(인사 관리자↔학생·학습자, 학생·학습자↔엔지니어링 매니저)',
      lambda df: _in(df, 'dev_ic_manager_orientation', 'People manager', 'Student / learner'),
      lambda df: (_in(df, 'dev_ic_manager_orientation', 'People manager') & _in(df, 'dev_role_archetype', 'Student / learner'))
      | (_in(df, 'dev_ic_manager_orientation', 'Student / learner') & _in(df, 'dev_role_archetype', 'Engineering manager / lead')))

_rule('R4-DEV-H16', 'developer_ai', 'hard',
      '테스트 접근(TDD·포괄적 단위테스트)과 버그 수정 시 테스트 행동(테스트 없음)이 정반대, 또는 그 역',
      lambda df: _in(df, 'code_testing_approach', 'Test-driven development (TDD)', 'Comprehensive unit test coverage', 'No tests'),
      lambda df: (_in(df, 'code_testing_approach', 'Test-driven development (TDD)', 'Comprehensive unit test coverage')
                  & _in(df, 'bug_fix_test_behavior', 'No tests'))
      | (_in(df, 'code_testing_approach', 'No tests') & _in(df, 'bug_fix_test_behavior', 'Always adds regression tests')))

_TOOL_PROG = (('tool_python', 'prog_python'), ('tool_r', 'prog_r'), ('tool_sql', 'prog_sql'),
              ('tool_matlab', 'prog_matlab'))
for _t, _pg in _TOOL_PROG:
    _ref(_t, ('Power user', 'Never used'))
    _ref(_pg, ('Expert', 'None'))


def _tool_prog_bad(df):
    out = pd.Series(False, index=df.index)
    for t, pg in _TOOL_PROG:
        out |= (_in(df, t, 'Power user') & _in(df, pg, 'None')) | (_in(df, pg, 'Expert') & _in(df, t, 'Never used'))
    return out


_rule('R4-DEV-H17', 'developer_ai', 'hard',
      '같은 언어의 도구 사용과 언어 숙련이 정반대(Python·R·SQL·MATLAB: 파워유저↔숙련 없음, 전문가↔사용 경험 없음)',
      lambda df: _any(*[_in(df, t, 'Power user', 'Never used') | _in(df, pg, 'Expert', 'None') for t, pg in _TOOL_PROG]),
      _tool_prog_bad)

# --- culture, mobility and devices ---------------------------------------------------------------
_CULT_FOREIGN = ('cult_mexico', 'cult_brazil', 'cult_argentina', 'cult_united_kingdom', 'cult_france',
                 'cult_germany', 'cult_italy', 'cult_spain', 'cult_netherlands', 'cult_sweden',
                 'cult_poland', 'cult_russia', 'cult_turkey', 'cult_egypt', 'cult_saudi_arabia',
                 'cult_uae', 'cult_israel', 'cult_iran', 'cult_nigeria', 'cult_kenya',
                 'cult_south_africa', 'cult_ethiopia', 'cult_india', 'cult_pakistan',
                 'cult_bangladesh', 'cult_china', 'cult_japan', 'cult_vietnam', 'cult_thailand',
                 'cult_indonesia', 'cult_philippines', 'cult_australia', 'cult_new_zealand',
                 'cult_singapore', 'cult_malaysia', 'cult_greece', 'cult_portugal',
                 'cult_united_states', 'cult_canada')


def _cult_any(df, *levels):
    """Any foreign culture at one of the levels (needs every cult_* column)."""
    return _any(*[_in(df, c, *levels) for c in _CULT_FOREIGN])


_rule('R4-POP-H05', 'population', 'hard',
      '국내 출생·거주 한국인인데 한국 문화 친숙도가 방문·학습·생소함',
      lambda df: _ni(df, 'cult_south_korea'),
      lambda df: _in(df, 'cult_south_korea', 'Visited', 'Studied', 'Unfamiliar'))

_rule('R4-POP-S07', 'population', 'soft',
      '국내 출생 한국인이 외국 문화를 모국(Native) 수준으로 보유 — 1% 미만',
      lambda df: _ni(df, 'cult_south_korea'),
      lambda df: _cult_any(df, 'Native'))

_rule('R4-CUL-H01', 'culture', 'hard',
      '출국 경험이 없는데 외국에 거주·방문한 경험이 있음',
      lambda df: _in(df, 'lifex_travel_breadth', 'Never left country'),
      lambda df: _cult_any(df, 'Lived there', 'Visited'))

_pair('R4-DVC-H01', 'devices', 'hard',
      '주 메신저가 iMessage(애플 기기 전용)인데 기기 생태계가 안드로이드·윈도우·리눅스',
      'lstyle_primary_messenger', ['iMessage'], 'lstyle_device_ecosystem', ['Android/Google', 'Windows', 'Linux'])

_pair('R4-DVC-H02', 'devices', 'hard',
      '주 브라우저가 Safari(애플 기기 전용, 윈도우판 2012년 중단)인데 기기 생태계가 안드로이드·윈도우·리눅스',
      'lstyle_browser', ['Safari'], 'lstyle_device_ecosystem', ['Android/Google', 'Windows', 'Linux'])

_pair('R4-DVC-H03', 'devices', 'hard',
      '주간 스트리밍 16시간 이상(하루 2.3시간 이상)인데 하루 화면 사용 시간 2시간 미만',
      'lstyle_streaming_hours', ['16+'], 'lstyle_screen_time', ['<2 hrs'])

_pair('R4-DVC-H04', 'devices', 'hard',
      '주 소셜미디어가 없음(None)인데 미디어 소비가 소셜미디어 중심',
      'lstyle_primary_social', ['None'], 'media_diet', ['Social-media-heavy'])

_pair('R4-TEC-H01', 'technology', 'hard',
      '기술 숙련도가 회피형(Avoidant)인데 신기술 수용이 최첨단·얼리어답터',
      'tech_savviness', ['Avoidant'], 'pref_early_vs_late', ['Bleeding edge', 'Early adopter'])

_pair('R4-TEC-S01', 'technology', 'soft',
      '65세 이상(1955년 이전 출생)인데 기술 숙련도가 디지털 네이티브 — 디지털 네이티브는 1980년 이후 출생 세대 개념',
      'age_bracket', ['65-74', '75-84', '85+'], 'tech_savviness', ['Digital native'])

_rule('R4-TEC-S02', 'technology', 'soft',
      'ChatGPT·Claude 파워유저인데 AI에 반대(Opposed) — 매우 드묾',
      lambda df: _in(df, 'tool_chatgpt', 'Power user') | _in(df, 'tool_claude', 'Power user'),
      lambda df: _in(df, 'att_ai', 'Opposed'))

# --- lifestyle and habits -----------------------------------------------------------------------
_pair('R4-LST-H01', 'lifestyle', 'hard',
      '은행 계좌가 없는데(Unbanked) 주 결제 수단이 신용·체크카드',
      'lstyle_banking_style', ['Unbanked'], 'lstyle_payment_pref', ['Credit card', 'Debit card'])

_rule('R4-LST-H02', 'lifestyle', 'hard',
      '암호화폐에 반대하거나 혐오하는데 주 결제 수단이 암호화폐이거나 투자가 암호화폐 위주',
      lambda df: _in(df, 'att_cryptocurrency', 'Opposed') | _in(df, 'topic_cryptocurrency', 'Averse'),
      lambda df: _in(df, 'lstyle_payment_pref', 'Crypto') | _in(df, 'lstyle_investment_style', 'Crypto-heavy'))

_rule('R4-LST-H03', 'lifestyle', 'hard',
      '사회적 에너지(내향/외향)와 BFI-2 외향성이 정반대 극단(강한 내향↔매우 높음, 강한 외향↔매우 낮음)',
      lambda df: _in(df, 'lstyle_social_battery', 'Strong introvert', 'Strong extrovert'),
      lambda df: (_in(df, 'lstyle_social_battery', 'Strong introvert') & _in(df, 'bfi2_domain_extraversion', 'Very high'))
      | (_in(df, 'lstyle_social_battery', 'Strong extrovert') & _in(df, 'bfi2_domain_extraversion', 'Very low')))

_pair('R4-LST-H04', 'lifestyle', 'hard',
      '매일·매주 자원봉사하는데 봉사 이력이 없음',
      'lstyle_volunteering', ['Daily', 'Weekly'], 'lifex_service_history', ['None'])

_pair('R4-LST-H05', 'lifestyle', 'hard',
      '독서를 전혀 하지 않는데 매일·매주 자기 전 독서 습관',
      'lstyle_reading_freq', ['Never'], 'habit_reading_before_bed', ['Daily', 'Weekly'])

_pair('R4-LST-H06', 'lifestyle', 'hard',
      '요리를 전혀 하지 않는데 매일·매주 밀프렙(식사 준비) 습관',
      'lstyle_cooking_freq', ['Never'], 'habit_meal_prepping', ['Daily', 'Weekly'])

_rule('R4-LST-H07', 'lifestyle', 'hard',
      '정리정돈 상태와 IPIP 질서정연함이 정반대 극단(혼돈↔매우 높음, 티끌 없음↔매우 낮음)',
      lambda df: _in(df, 'lstyle_tidiness', 'Chaotic', 'Spotless'),
      lambda df: (_in(df, 'lstyle_tidiness', 'Chaotic') & _in(df, 'big5_orderliness', 'Very high'))
      | (_in(df, 'lstyle_tidiness', 'Spotless') & _in(df, 'big5_orderliness', 'Very low')))

_rule('R4-LST-H08', 'lifestyle', 'hard',
      '검소함과 저축/소비 성향이 정반대(검소한 저축가↔마음껏 소비, 과소비형↔철저한 저축)',
      lambda df: _in(df, 'lstyle_frugality', 'Frugal saver', 'Splurger'),
      lambda df: (_in(df, 'lstyle_frugality', 'Frugal saver') & _in(df, 'pref_save_vs_spend', 'Free spender'))
      | (_in(df, 'lstyle_frugality', 'Splurger') & _in(df, 'pref_save_vs_spend', 'Hard saver')))

_rule('R4-LST-H09', 'lifestyle', 'hard',
      '게임에 혐오인데 매일 게임, 또는 e스포츠를 직접 하는데 게임을 전혀 안 함',
      lambda df: _in(df, 'topic_gaming', 'Averse') | _in(df, 'sport_esports', 'Play'),
      lambda df: (_in(df, 'topic_gaming', 'Averse') & _in(df, 'lstyle_gaming_freq', 'Daily'))
      | (_in(df, 'sport_esports', 'Play') & _in(df, 'lstyle_gaming_freq', 'Never')))

_rule('R4-LST-S01', 'lifestyle', 'soft',
      '와인·수제맥주에 열정적 관심인데 음주를 전혀 하지 않음 — 매우 드묾',
      lambda df: _in(df, 'topic_wine', 'Passionate') | _in(df, 'topic_craft_beer', 'Passionate'),
      lambda df: _in(df, 'lstyle_alcohol_use', 'Never'))

# --- interests vs activities -------------------------------------------------------------------
_TOPIC_HOBBY = (('topic_knitting', 'hob_knitting'), ('topic_woodworking', 'hob_woodworking'),
                ('topic_calligraphy', 'hob_calligraphy'), ('topic_birdwatching', 'hob_birdwatching'),
                ('topic_genealogy', 'hob_genealogy'), ('topic_stand_up_comedy', 'hob_stand_up_comedy'),
                ('topic_baking', 'hob_bread_baking'), ('topic_gardening', 'hob_vegetable_gardening'),
                ('topic_astronomy', 'hob_stargazing'))
for _t, _h in _TOPIC_HOBBY:
    _ref(_t, ('Averse',))
    _ref(_h, ('Avid',))

_rule('R4-INT-H01', 'interests', 'hard',
      '같은 활동에 대해 관심 주제는 혐오(Averse)인데 취미로는 열성적(Avid) — 뜨개질·목공·서예·탐조·족보·스탠드업·제빵·텃밭·별보기',
      lambda df: _any(*[_in(df, t, 'Averse') for t, _ in _TOPIC_HOBBY]),
      lambda df: _any(*[_in(df, t, 'Averse') & _in(df, h, 'Avid') for t, h in _TOPIC_HOBBY]))

_ACAD_TOPIC = (('acad_astronomy', 'topic_astronomy'), ('acad_philosophy', 'topic_philosophy'),
               ('acad_literature', 'topic_literature'), ('acad_visual_art', 'topic_visual_art'),
               ('acad_music', 'topic_music'), ('acad_world_history', 'topic_history'))
for _a, _t in _ACAD_TOPIC:
    _ref(_a, ('Passionate', 'Averse'))
    _ref(_t, ('Passionate', 'Averse'))

_rule('R4-INT-H02', 'interests', 'hard',
      '같은 분야의 학과목 관심과 주제 관심이 정반대 극단(열정↔혐오) — 천문·철학·문학·미술·음악·역사',
      lambda df: _any(*[_in(df, a, 'Passionate', 'Averse') for a, _ in _ACAD_TOPIC]),
      lambda df: _any(*[(_in(df, a, 'Passionate') & _in(df, t, 'Averse')) | (_in(df, a, 'Averse') & _in(df, t, 'Passionate'))
                        for a, t in _ACAD_TOPIC]))

_pair('R4-INT-H03', 'interests', 'hard',
      '명상 주제를 혐오하는데 매일 명상 습관',
      'topic_meditation', ['Averse'], 'habit_meditation', ['Daily'])

_rule('R4-FOD-H01', 'food', 'hard',
      '비건인데 비건 요리를 기피하거나, 채식·비건인데 채식 요리를 기피',
      lambda df: _in(df, 'lstyle_diet_type', 'Vegan', 'Vegetarian'),
      lambda df: (_in(df, 'lstyle_diet_type', 'Vegan') & _in(df, 'cuis_vegan', 'Avoid'))
      | _in(df, 'cuis_vegetarian', 'Avoid'))

_rule('R4-FOD-S01', 'food', 'soft',
      '비건인데 해산물 요리·미국식 바비큐를 매우 좋아함(Love) — 매우 드묾',
      lambda df: _in(df, 'lstyle_diet_type', 'Vegan'),
      lambda df: _in(df, 'cuis_seafood', 'Love') | _in(df, 'cuis_american_bbq', 'Love'))

_rule('R4-POP-S08', 'population', 'soft',
      '국내 출생 한국인이 한식을 기피(Avoid) — 1% 미만',
      lambda df: _ni(df, 'cuis_korean'),
      lambda df: _in(df, 'cuis_korean', 'Avoid'))

# --- knowledge vs practical skills ---------------------------------------------------------------
_rule('R4-SKL-H01', 'skills', 'hard',
      '운전 불가인데 운전 기술 고급·마스터, 또는 매일 운전하는데 운전 기술 없음',
      lambda df: _in(df, 'demo_driver_status', 'Cannot drive', 'Daily driver'),
      lambda df: (_in(df, 'demo_driver_status', 'Cannot drive') & _in(df, 'skill_driving', 'Master', 'Advanced'))
      | (_in(df, 'demo_driver_status', 'Daily driver') & _in(df, 'skill_driving', 'None')))

_PROG_COLS = ('prog_ruby', 'prog_php', 'prog_kotlin', 'prog_objective_c', 'prog_haskell', 'prog_elixir',
              'prog_erlang', 'prog_clojure', 'prog_lua', 'prog_perl', 'prog_r', 'prog_julia',
              'prog_powershell', 'prog_dart', 'prog_f', 'prog_ocaml', 'prog_assembly', 'prog_cobol',
              'prog_fortran', 'prog_graphql', 'prog_matlab', 'prog_python', 'prog_javascript',
              'prog_java', 'prog_c', 'prog_go', 'prog_bash', 'prog_typescript', 'prog_rust',
              'prog_swift', 'prog_scala', 'prog_sql', 'prog_solidity')
for _c in _PROG_COLS:
    _ref(_c, ('Expert', 'Proficient', 'None'))

_rule('R4-SKL-H02', 'skills', 'hard',
      '코딩 기술이 고급·마스터인데 모든 프로그래밍 언어 숙련 없음, 또는 코딩 기술 없음인데 어떤 언어 전문가',
      lambda df: _in(df, 'skill_coding', 'Master', 'Advanced', 'None'),
      lambda df: (_in(df, 'skill_coding', 'Master', 'Advanced') & _all(*[_in(df, c, 'None') for c in _PROG_COLS]))
      | (_in(df, 'skill_coding', 'None') & _any(*[_in(df, c, 'Expert') for c in _PROG_COLS])))

_rule('R4-SKL-H03', 'skills', 'hard',
      '번역·통역 기술이 고급·마스터인데 단일언어 사용자',
      lambda df: _in(df, 'skill_translation', 'Master', 'Advanced') | _in(df, 'skill_interpretation', 'Master', 'Advanced'),
      lambda df: _in(df, 'multilingualism', 'Monolingual'))

_SKILL_FAM = (('skill_statistical_modeling', 'fam_statistics'),
              ('skill_project_management', 'fam_project_management'),
              ('skill_photography', 'fam_photography'),
              ('skill_data_analysis', 'fam_data_science'))
for _sk, _fm in _SKILL_FAM:
    _ref(_sk, ('Master', 'Advanced'))
    _ref(_fm, ('None',))

_rule('R4-SKL-H04', 'skills', 'hard',
      '실무 기술이 고급·마스터인데 같은 분야 지식 친숙도가 전혀 없음(통계 모델링·프로젝트 관리·사진·데이터 분석)',
      lambda df: _any(*[_in(df, sk, 'Master', 'Advanced') for sk, _ in _SKILL_FAM]),
      lambda df: _any(*[_in(df, sk, 'Master', 'Advanced') & _in(df, fm, 'None') for sk, fm in _SKILL_FAM]))


# ===========================================================================
# engine
# ===========================================================================
def violation_matrix(df, rules=None):
    rules = RULES if rules is None else rules
    idx = df.index
    hits, scopes, status = {}, {}, {}
    for r in rules:
        rid = r['id']
        try:
            sc = _bool(r['scope'](df), idx)
            ht = _bool(r['hit'](df), idx) & sc
            scopes[rid], hits[rid], status[rid] = sc, ht, 'ok'
        except MissingColumn as e:
            col = e.args[0] if e.args else '?'
            scopes[rid] = pd.Series(False, index=idx)
            hits[rid] = pd.Series(False, index=idx)
            status[rid] = 'skipped: missing %s' % col
        except Exception as e:  # noqa: BLE001 - never raise from a rule
            scopes[rid] = pd.Series(False, index=idx)
            hits[rid] = pd.Series(False, index=idx)
            status[rid] = 'error: %s: %s' % (type(e).__name__, e)
    cols = [r['id'] for r in rules]
    hits = pd.DataFrame(hits, index=idx, columns=cols).astype(bool)
    scopes = pd.DataFrame(scopes, index=idx, columns=cols).astype(bool)
    return hits, scopes, status


def run(df):
    hits, scopes, status = violation_matrix(df)
    rows = []
    for r in RULES:
        rid = r['id']
        n_scope = int(scopes[rid].sum())
        n_hit = int(hits[rid].sum())
        rows.append(dict(id=rid, group=r['group'], severity=r['severity'],
                         description=r['description'], n_scope=n_scope, n_hit=n_hit,
                         violation_rate=(n_hit / n_scope) if n_scope else np.nan,
                         status=status[rid]))
    table = pd.DataFrame(rows, columns=['id', 'group', 'severity', 'description', 'n_scope',
                                        'n_hit', 'violation_rate', 'status'])
    hard_ok = [r['id'] for r in RULES if r['severity'] == 'hard' and status[r['id']] == 'ok']
    if len(df) == 0 or not hard_ok:
        any_hard_rate = np.nan
    else:
        any_hard_rate = float(hits[hard_ok].any(axis=1).mean())
    return table, any_hard_rate


def check_values(catalog_path):
    """Return a list of problems with attribute ids / value strings referenced by RULES."""
    with open(catalog_path, encoding='utf-8') as f:
        cat = json.load(f)
    allowed = {}
    for a in cat['attributes']:
        allowed[a['id']] = set(a['values']) | set(a.get('extra_values_not_in_matrAIx', []))
        if a['id'].startswith('dev_') or a['category'].startswith('Developer'):
            allowed[a['id']] |= {'Not applicable'}
    allowed.setdefault('dev_professional_status', set()).add('Not a developer')
    # run every rule on a one-row frame with every catalog column so that all value
    # references made at call time are recorded
    first = {k: (sorted(v)[0] if v else '') for k, v in allowed.items()}
    dummy = pd.DataFrame([first])
    problems = []
    _, _, status = violation_matrix(dummy)
    for rid, st in status.items():
        if st != 'ok':
            problems.append('%s: %s' % (rid, st))
    for col, vals in sorted(_REFS.items()):
        if col not in allowed:
            problems.append('unknown attribute id: %s' % col)
            continue
        for v in sorted(vals):
            if v not in allowed[col]:
                problems.append('unknown value for %s: %r' % (col, v))
    return problems
