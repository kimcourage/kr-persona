"""Within-persona contradictions among age / gender / family / work / role / education fields."""
import json, sys
import numpy as np
from krpersona import paths
from krpersona.baseline.lib import KR, BASE, cols, vals, ci, mask

KIDS = ["Under 5", "5-12"]
MINOR = ["Under 5", "5-12", "13-17"]
ADULT_BRACKETS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"]
# 2026 reference. birth-year ranges per bracket -> allowed generation labels
GEN_OK = {"Under 5": ["Gen Alpha"], "5-12": ["Gen Alpha"], "13-17": ["Gen Alpha", "Gen Z"], "18-24": ["Gen Z"],
          "25-34": ["Gen Z", "Millennial"], "35-44": ["Millennial"], "45-54": ["Millennial", "Gen X"],
          "55-64": ["Gen X", "Boomer"], "65-74": ["Boomer"], "75-84": ["Boomer", "Silent"], "85+": ["Silent"]}
# formative window = ages 10-25 of the bracket's birth range; decades outside it are impossible
DEC = {"1960s-70s": (1960, 1979), "1980s": (1980, 1989), "1990s": (1990, 1999), "2000s": (2000, 2009),
       "2010s": (2010, 2019), "2020s": (2020, 2029)}
AGE_RANGE = {"18-24": (18, 24), "25-34": (25, 34), "35-44": (35, 44), "45-54": (45, 54), "55-64": (55, 64),
             "65-74": (65, 74), "75-84": (75, 84), "85+": (85, 100)}


def formative_ok(age):
    lo, hi = AGE_RANGE[age]
    b0, b1 = 2026 - hi, 2026 - lo
    w0, w1 = b0 + 10, b1 + 25
    return [d for d, (s, e) in DEC.items() if s <= w1 and e >= w0]


def m(M, c, vs):
    return mask(M, c, vs)


def gen_mismatch(M):
    hit = np.zeros(len(M), bool)
    for age, ok in GEN_OK.items():
        hit |= m(M, "age_bracket", [age]) & ~m(M, "demo_generation", ok)
    return hit


def formative_mismatch(M):
    hit = np.zeros(len(M), bool)
    for age in AGE_RANGE:
        ok = formative_ok(age)
        hit |= m(M, "age_bracket", [age]) & ~m(M, "lifex_formative_decade", ok)
    return hit


def any_ind_veteran(M):
    ind = [c for c in cols if c.startswith("ind_") and "Veteran" in vals[c]]
    return np.any(np.stack([m(M, c, ["Veteran"]) for c in ind]), axis=0)


# (id, group, severity, description, scope_fn, hit_fn)   severity: hard = logically impossible, soft = implausible
R = [
 ("gen_vs_age", "나이×시대", "hard", "세대(demo_generation)가 나이와 안 맞음", lambda M: np.ones(len(M), bool), gen_mismatch),
 ("formative_vs_age", "나이×시대", "hard", "성장기 연대(lifex_formative_decade)가 나이상 불가능 (성인)",
  lambda M: m(M, "age_bracket", ADULT_BRACKETS), formative_mismatch),
 ("teen_divorced_widowed", "나이×가족", "hard", "13–17세인데 이혼·사별·별거", lambda M: m(M, "age_bracket", ["13-17"]),
  lambda M: m(M, "demo_marital_status", ["Divorced", "Widowed", "Separated"])),
 ("teen_married", "나이×가족", "soft", "13–17세인데 기혼·약혼·동거", lambda M: m(M, "age_bracket", ["13-17"]),
  lambda M: m(M, "demo_marital_status", ["Married", "Domestic partnership", "Engaged"])),
 ("young_adult_children", "나이×가족", "hard", "13–24세인데 성인 자녀·손주 있음", lambda M: m(M, "age_bracket", ["13-17", "18-24"]),
  lambda M: m(M, "demo_children_count", ["Adult children"]) | m(M, "demo_parental_status", ["Parent of adults", "Grandparent"])),
 ("u35_grandparent", "나이×가족", "soft", "25–34세인데 성인 자녀·조부모", lambda M: m(M, "age_bracket", ["25-34"]),
  lambda M: m(M, "demo_children_count", ["Adult children"]) | m(M, "demo_parental_status", ["Parent of adults", "Grandparent"])),
 ("parenting_journey_vs_children", "가족×가족", "hard", "자녀 없음인데 육아 경험이 '어린 자녀/10대/성인 자녀/빈 둥지'",
  lambda M: m(M, "demo_children_count", ["None"]),
  lambda M: m(M, "lifex_parenting_journey", ["Young children", "Teenagers", "Grown children", "Empty nester"])),
 ("children_vs_no_children_journey", "가족×가족", "hard", "자녀 1명 이상인데 육아 경험 '자녀 없음'",
  lambda M: m(M, "demo_children_count", ["1 child", "2 children", "3+ children", "Adult children"]),
  lambda M: m(M, "lifex_parenting_journey", ["No children"])),
 ("young_grown_children_journey", "나이×가족", "hard", "13–24세인데 육아 경험 '성인 자녀/빈 둥지/손주 양육'",
  lambda M: m(M, "age_bracket", ["13-17", "18-24"]),
  lambda M: m(M, "lifex_parenting_journey", ["Grown children", "Empty nester", "Raising grandchildren"])),
 ("kid_parenting_journey", "나이×가족", "hard", "13세 미만인데 육아 경험이 '자녀 없음' 외",
  lambda M: m(M, "age_bracket", KIDS), lambda M: ~m(M, "lifex_parenting_journey", ["No children"])),
 ("kid_relationship_history", "나이×가족", "hard", "13세 미만인데 연애사 '장기 연애/결혼/재혼/사별'",
  lambda M: m(M, "age_bracket", KIDS),
  lambda M: m(M, "lifex_relationship_history", ["Long-term partnership", "Married once", "Remarried", "Widowed"])),
 ("widowed_history_vs_single", "가족×가족", "soft", "결혼 상태 '사별/이혼/별거'인데 연애사 'Limited'",
  lambda M: m(M, "demo_marital_status", ["Widowed", "Divorced", "Separated"]), lambda M: m(M, "lifex_relationship_history", ["Limited"])),
 ("yexp_18_24", "나이×경력", "hard", "18–24세인데 경력 11년 이상", lambda M: m(M, "age_bracket", ["18-24"]),
  lambda M: m(M, "years_experience", ["11-20", "20+"])),
 ("yexp_25_34", "나이×경력", "hard", "25–34세인데 경력 20년 이상", lambda M: m(M, "age_bracket", ["25-34"]), lambda M: m(M, "years_experience", ["20+"])),
 ("young_exec", "나이×직급", "soft", "18–24세인데 이사·VP·C레벨", lambda M: m(M, "age_bracket", ["18-24"]),
  lambda M: m(M, "seniority", ["Director", "VP", "C-suite"])),
 ("young_retired", "나이×고용", "hard", "35세 미만인데 은퇴(고용 상태 또는 직급)", lambda M: m(M, "age_bracket", ["13-17", "18-24", "25-34"]),
  lambda M: m(M, "demo_employment_status", ["Retired"]) | m(M, "seniority", ["Retired"])),
 ("old_early_life_stage", "나이×생애단계", "soft", "65세 이상인데 생애단계 '학생/사회초년생/어린 자녀 부모'",
  lambda M: m(M, "age_bracket", ["65-74", "75-84", "85+"]), lambda M: m(M, "life_stage", ["Student", "Early career", "Parent of young kids"])),
 ("teen_late_life_stage", "나이×생애단계", "hard", "13–24세인데 생애단계 '빈 둥지/은퇴'",
  lambda M: m(M, "age_bracket", ["13-17", "18-24"]), lambda M: m(M, "life_stage", ["Empty nester", "Retirement"])),
 ("kid_role_exec_etc", "나이×역할", "hard", "13세 미만인데 직무가 경영진·임상·법무·재무·HR·영업",
  lambda M: m(M, "age_bracket", KIDS), lambda M: m(M, "role_function", ["Executive", "Clinical", "Legal", "Finance", "HR", "Sales / GTM"])),
 ("kid_company", "나이×역할", "hard", "13세 미만인데 소속 조직이 기업·공공·학계·NGO",
  lambda M: m(M, "age_bracket", KIDS), lambda M: ~m(M, "company_size", ["Solo / freelance"])),
 ("kid_domain_practitioner", "나이×역할", "hard", "13세 미만인데 분야 성격이 '최전선 연구자/실무자/교육자'",
  lambda M: m(M, "age_bracket", KIDS), lambda M: m(M, "domain_characteristics", ["Cutting-edge researcher", "Practitioner", "Educator"])),
 ("minor_research_output", "나이×역할", "hard", "18세 미만인데 논문·백서 저자", lambda M: m(M, "age_bracket", MINOR),
  lambda M: m(M, "research_output", ["Prolific publisher", "Occasional author", "Industry whitepapers"])),
 ("kid_institution", "나이×학력", "hard", "13세 미만인데 소속 교육기관이 대학·전문대·부트캠프", lambda M: m(M, "age_bracket", KIDS),
  lambda M: m(M, "institution_tier", ["Top-tier research", "Mid-tier university", "Community college", "Online / bootcamp"])),
 ("kid_linkedin", "나이×역할", "hard", "13세 미만인데 링크드인 활동(사고 리더·네트워커·구직·리크루터)", lambda M: m(M, "age_bracket", KIDS),
  lambda M: m(M, "linkedin_activity", ["Thought leader", "Active networker", "Job seeker", "Recruiter"])),
 ("kid_career_path", "나이×경력", "hard", "13세 미만인데 커리어 경로가 '막 시작' 외", lambda M: m(M, "age_bracket", KIDS),
  lambda M: ~m(M, "lifex_career_path_shape", ["Just starting"])),
 ("kid_entrepreneur", "나이×경력", "hard", "13세 미만인데 창업·엑싯 경험", lambda M: m(M, "age_bracket", KIDS),
  lambda M: m(M, "lifex_entrepreneurship_history", ["Founded once", "Serial founder", "Exited a company"])),
 ("minor_military", "나이×경력", "hard", "18세 미만인데 직업군인·참전 경력", lambda M: m(M, "age_bracket", MINOR),
  lambda M: m(M, "lifex_military_history", ["Career veteran", "Combat veteran"])),
 ("minor_edu_journey", "나이×학력", "hard", "18세 미만인데 학업 경로 '대학원 학위/성인 재입학'", lambda M: m(M, "age_bracket", MINOR),
  lambda M: m(M, "lifex_education_journey", ["Advanced degrees", "Returned as adult"])),
 ("minor_public_office", "나이×경력", "hard", "18세 미만인데 공직 경력", lambda M: m(M, "age_bracket", MINOR),
  lambda M: m(M, "lifex_service_history", ["Public office"])),
 ("kid_industry_veteran", "나이×경력", "hard", "13세 미만인데 어떤 산업이든 'Veteran'", lambda M: m(M, "age_bracket", KIDS), any_ind_veteran),
 ("kid_people_manager", "나이×역할", "hard", "13세 미만인데 피플 매니저·창업자", lambda M: m(M, "age_bracket", KIDS),
  lambda M: m(M, "dev_ic_manager_orientation", ["People manager", "Founder / owner"])),
 ("toddler_caregiver", "나이×가족", "hard", "5세 미만인데 돌봄 제공자", lambda M: m(M, "age_bracket", ["Under 5"]),
  lambda M: ~m(M, "health_caregiver_status", ["Not a caregiver"])),
 ("retired_but_student_stage", "고용×생애단계", "hard", "고용 '학생'인데 생애단계 '은퇴' (또는 반대)", lambda M: np.ones(len(M), bool),
  lambda M: (m(M, "demo_employment_status", ["Student"]) & m(M, "life_stage", ["Retirement"])) |
            (m(M, "demo_employment_status", ["Retired"]) & m(M, "life_stage", ["Student", "Early career"]))),
 ("seniority_retired_fulltime", "고용×직급", "soft", "직급 '은퇴'인데 고용 '정규직/학생'", lambda M: m(M, "seniority", ["Retired"]),
  lambda M: m(M, "demo_employment_status", ["Full-time", "Student"])),
 ("exec_0_2_years", "직급×경력", "soft", "이사·VP·C레벨인데 경력 0–2년", lambda M: m(M, "seniority", ["Director", "VP", "C-suite"]),
  lambda M: m(M, "years_experience", ["0-2"])),
 ("intern_20_years", "직급×경력", "hard", "학생/인턴인데 경력 11년 이상", lambda M: m(M, "seniority", ["Student / intern"]),
  lambda M: m(M, "years_experience", ["11-20", "20+"])),
 ("solo_manager", "조직×직급", "soft", "1인/프리랜서인데 매니저·이사·VP 또는 피플 매니저", lambda M: m(M, "company_size", ["Solo / freelance"]),
  lambda M: m(M, "seniority", ["Manager", "Director", "VP"]) | m(M, "dev_ic_manager_orientation", ["People manager"])),
 ("clinical_outside_health", "역할×분야", "soft", "직무 '임상'인데 분야가 의료 외", lambda M: m(M, "role_function", ["Clinical"]),
  lambda M: ~m(M, "domain", ["Healthcare & Medicine"])),
 ("legal_outside_law", "역할×분야", "soft", "직무 '법무'인데 분야가 법·정책 외", lambda M: m(M, "role_function", ["Legal"]),
  lambda M: ~m(M, "domain", ["Law & Policy"])),
 ("low_edu_top_university", "학력×학력", "hard", "최종학력 무학·초졸인데 연구중심/일반 대학 소속",
  lambda M: m(M, "highest_education", ["No formal", "Primary"]), lambda M: m(M, "institution_tier", ["Top-tier research", "Mid-tier university"])),
 ("low_edu_prolific", "학력×역할", "hard", "최종학력 고졸 이하인데 다작 연구자",
  lambda M: m(M, "highest_education", ["No formal", "Primary", "Secondary"]), lambda M: m(M, "research_output", ["Prolific publisher"])),
 ("adv_degree_journey_vs_edu", "학력×학력", "hard", "학업 경로 '대학원 학위'인데 최종학력 학사 미만",
  lambda M: m(M, "lifex_education_journey", ["Advanced degrees"]),
  lambda M: ~m(M, "highest_education", ["Master's", "Doctorate", "Postdoc"])),
 ("income_vs_ses", "소득×계층", "hard", "계층 '고소득'인데 가구소득 $25k 미만 (또는 저소득인데 $200k+)", lambda M: np.ones(len(M), bool),
  lambda M: (m(M, "socioeconomic_band", ["High income"]) & m(M, "demo_household_income", ["<$25k"])) |
            (m(M, "socioeconomic_band", ["Low income"]) & m(M, "demo_household_income", ["$200k+"]))),
]


def run(M):
    out = {}
    for rid, grp, sev, desc, scope, hit in R:
        s = scope(M); h = hit(M) & s
        out[rid] = dict(group=grp, severity=sev, desc=desc, scope_n=int(s.sum()), hits=int(h.sum()),
                        rate_in_scope=float(h.sum() / s.sum()) if s.sum() else None, rate_all=float(h.mean()))
    return out


if __name__ == "__main__":
    res = {"kr": run(KR), "global": run(BASE)}
    # any-hit per persona
    for name, M in [("kr", KR), ("global", BASE)]:
        hard = np.zeros(len(M), bool); anyh = np.zeros(len(M), bool)
        for rid, grp, sev, desc, scope, hit in R:
            h = hit(M) & scope(M); anyh |= h
            if sev == "hard":
                hard |= h
        res[name + "_persona_with_any_hard"] = float(hard.mean())
        res[name + "_persona_with_any"] = float(anyh.mean())
        res[name + "_minor_share"] = float(m(M, "age_bracket", MINOR).mean())
        res[name + "_adult_with_any_hard"] = float(hard[~m(M, "age_bracket", MINOR)].mean())
    json.dump(res, open(paths.art('baseline', 'demo_results.json'), 'w'), ensure_ascii=False, indent=1)
    for rid, grp, sev, desc, *_ in R:
        a, b = res["kr"][rid], res["global"][rid]
        f = lambda x: "  -  " if x is None else f"{100*x:5.1f}%"
        print(f"{sev:4s} {grp:10s} {desc:52s} 해당 {a['scope_n']:6d}명 중 {f(a['rate_in_scope'])} | 전체 {f(a['rate_all'])} | global {f(b['rate_in_scope'])}")
    for k in ["kr_persona_with_any_hard", "kr_persona_with_any", "kr_minor_share", "kr_adult_with_any_hard", "global_persona_with_any_hard"]:
        print(k, round(100 * res[k], 1))
