"""Contradiction / plausibility checks on the KR-native synthetic subset."""
import json
import numpy as np
from krpersona import paths
from krpersona.baseline.lib import KR, BASE, cols, vals, ci, dist, mask

LIK = {"Very high": 5, "High": 4, "Average": 3, "Low": 2, "Very low": 1}


def score(M, c):
    lut = np.array([LIK.get(v, np.nan) for v in vals[c]], dtype=float)
    return lut[M[:, ci[c]]]


def rate(m):
    return float(m.mean())


def checks(M):
    r = {}
    n = len(M)
    r["n"] = n
    # ---- identity / culture -------------------------------------------------
    r["lang_korean_native_or_fluent"] = rate(mask(M, "lang_korean", ["Native", "Fluent"]))
    r["lang_korean_none"] = rate(mask(M, "lang_korean", ["None"]))
    r["lang_korean_none_or_basic"] = rate(mask(M, "lang_korean", ["None", "Basic"]))
    r["primary_mandarin"] = rate(mask(M, "primary_language", ["Mandarin"]))
    r["lang_mandarin_native"] = rate(mask(M, "lang_mandarin", ["Native"]))
    cn = mask(M, "cult_china", ["Native"]); jp = mask(M, "cult_japan", ["Native"])
    r["also_china_native"] = rate(cn); r["also_japan_native"] = rate(jp)
    r["china_and_japan_native_too"] = rate(cn & jp); r["any_other_native_culture"] = rate(cn | jp)
    others = [c for c in cols if c.startswith("cult_") and c != "cult_south_korea"]
    nat_other = np.zeros(n, dtype=int)
    for c in others:
        nat_other += mask(M, c, ["Native"])
    r["other_native_cultures_max"] = int(nat_other.max())
    r["cultural_background_collectivist_east_asian"] = rate(mask(M, "cultural_background", ["Collectivist (East Asian)"]))
    r["cultural_background_non_east_asian_frame"] = rate(~mask(M, "cultural_background", ["Collectivist (East Asian)", "Mixed / diaspora"]))
    r["ethnicity_east_asian"] = rate(mask(M, "demo_ethnicity_broad", ["East Asian"]))
    r["immig_first_gen"] = rate(mask(M, "lifex_immigration_generation", ["First-generation immigrant"]))
    r["citizen_by_birth_AND_first_gen_immigrant"] = rate(mask(M, "demo_citizenship_status", ["Citizen by birth"]) & mask(M, "lifex_immigration_generation", ["First-generation immigrant"]))
    mono = mask(M, "lifex_cultural_exposure", ["Monocultural"])
    r["monocultural_AND_multiple_native_cultures"] = rate(mono & (nat_other >= 1))
    never_left = mask(M, "lifex_geographic_mobility", ["Never left hometown"])
    r["never_left_hometown_AND_other_native_culture"] = rate(never_left & (nat_other >= 1))
    r["cuis_korean_avoid"] = rate(mask(M, "cuis_korean", ["Avoid"]))
    r["cuis_korean_love_or_like"] = rate(mask(M, "cuis_korean", ["Love", "Like"]))
    # language budget: native languages vs multilingualism
    langs = [c for c in cols if c.startswith("lang_") and "Native" in vals[c]]
    nat_lang = np.zeros(n, dtype=int)
    for c in langs:
        nat_lang += mask(M, c, ["Native"])
    r["zero_native_languages"] = rate(nat_lang == 0)
    r["monolingual_with_2plus_native_langs"] = rate(mask(M, "multilingualism", ["Monolingual"]) & (nat_lang >= 2))
    # ---- life course ---------------------------------------------------------
    young = mask(M, "age_bracket", ["Under 5", "5-12"])
    r["age_under_13"] = rate(young)
    r["under13_python_expert_or_proficient"] = rate(young & mask(M, "prog_python", ["Expert", "Proficient"]))
    r["under13_github_power_or_regular"] = rate(young & mask(M, "tool_github", ["Power user", "Regular"]))
    r["under13_professional_developer"] = rate(young & mask(M, "dev_professional_status", ["Professional developer"]))
    r["under13_household_income_200k_plus"] = rate(young & mask(M, "demo_household_income", ["$200k+"]))
    a1824 = mask(M, "age_bracket", ["18-24"])
    r["age18_24_doctorate_or_postdoc"] = rate(a1824 & mask(M, "highest_education", ["Doctorate", "Postdoc"]))
    r["age13_24_exp_11plus_years"] = rate(mask(M, "age_bracket", ["13-17", "18-24"]) & mask(M, "years_experience", ["11-20", "20+"]))
    r["age13_24_director_vp_csuite"] = rate(mask(M, "age_bracket", ["13-17", "18-24"]) & mask(M, "seniority", ["Director", "VP", "C-suite"]))
    r["employment_student_AND_seniority_csuite_vp_director"] = rate(mask(M, "demo_employment_status", ["Student"]) & mask(M, "seniority", ["Director", "VP", "C-suite"]))
    r["employment_retired_AND_age_under_45"] = rate(mask(M, "demo_employment_status", ["Retired"]) & mask(M, "age_bracket", ["13-17", "18-24", "25-34", "35-44"]))
    r["life_stage_retirement_AND_age_under_35"] = rate(mask(M, "life_stage", ["Retirement"]) & mask(M, "age_bracket", ["13-17", "18-24", "25-34"]))
    r["nonprogrammer_status_but_python_expert"] = rate(mask(M, "dev_professional_status", ["Non-developer technologist"]) & mask(M, "prog_python", ["Expert"]))
    # ---- psychometrics -------------------------------------------------------
    ext_dom = score(M, "bfi2_domain_extraversion")
    ext_fac = np.nanmean(np.stack([score(M, c) for c in ["big5_friendliness", "big5_gregariousness", "big5_assertiveness", "big5_activity_level", "big5_excitement_seeking", "big5_cheerfulness"]]), axis=0)
    r["bfi2_extraversion_vs_ipip_facets_gap_ge2"] = rate(np.abs(ext_dom - ext_fac) >= 2)
    r["corr_bfi2_extraversion_vs_ipip_facet_mean"] = float(np.corrcoef(ext_dom, ext_fac)[0, 1])
    bf = np.stack([score(M, c) for c in ["bfi2_facet_sociability", "bfi2_facet_assertiveness", "bfi2_facet_energy_level"]])
    r["corr_bfi2_extraversion_vs_own_facets"] = float(np.corrcoef(ext_dom, bf.mean(0))[0, 1])
    de = mask(M, "dominant_trait", ["High extraversion"])
    r["dominant_extraversion_AND_bfi2_extraversion_low"] = rate(de & (ext_dom <= 2))
    r["share_dominant_extraversion"] = rate(de)
    pairs = [("power", "universalism"), ("achievement", "benevolence"), ("self_direction", "conformity"), ("stimulation", "security"), ("hedonism", "tradition")]
    r["schwartz_opposing_corr"] = {f"{a}~{b}": float(np.corrcoef(score(M, f"schwartz_value_{a}"), score(M, f"schwartz_value_{b}"))[0, 1]) for a, b in pairs}
    r["schwartz_power_AND_universalism_both_very_high"] = rate((score(M, "schwartz_value_power") == 5) & (score(M, "schwartz_value_universalism") == 5))
    sv = np.stack([score(M, f"schwartz_value_{a}") for a, b in pairs for a in (a, b)])
    r["schwartz_all10_very_high_or_high"] = rate((sv >= 4).all(0))
    return r


if __name__ == "__main__":
    out = {"kr_native": checks(KR), "global": checks(BASE)}
    # distributions worth reporting
    out["dist"] = {c: {"values": vals[c], "kr": dist(KR, c).tolist(), "global": dist(BASE, c).tolist()}
                   for c in ["age_bracket", "urbanicity", "demo_religion_affiliation", "religiosity", "primary_language", "lang_korean",
                             "cultural_background", "cuis_korean", "sport_baseball", "lifex_immigration_generation", "demo_citizenship_status",
                             "political_lean", "lstyle_household_size", "demo_household_income"]}
    json.dump(out, open(paths.art('baseline', 'results.json'), 'w'), indent=1, ensure_ascii=False)
    for k in out["kr_native"]:
        a, b = out["kr_native"][k], out["global"][k]
        if isinstance(a, dict):
            print(k); [print(f"    {kk:28s} {a[kk]:+.3f}  {b[kk]:+.3f}") for kk in a]
        elif isinstance(a, float):
            print(f"{k:55s} {a*100 if 'corr' not in k else a:8.2f}  {b*100 if 'corr' not in k else b:8.2f}")
        else:
            print(f"{k:55s} {a:>8}  {b:>8}")
