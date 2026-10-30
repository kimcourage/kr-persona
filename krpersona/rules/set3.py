# -*- coding: utf-8 -*-
"""Consistency rules (set 3) for synthetic South Korean adult personas.

Written only from the attribute definitions in attribute_catalog.json and from real-world
knowledge about South Korea.  It does not depend on any generator or on earlier rule sets.

Conventions taken from the population definition
  * Population: domestic-born Korean citizens aged 19+.  age_bracket '18-24' means 19-24.
  * Birth-year-derived attributes (generation, formative decade, life-course arithmetic)
    use birth year = 2020 - age (November 2020 census reference).
  * Technology, products and AI tools describe life in 2026, so tools released after 2020
    are never treated as contradictions.

Every rule is a dict with
  id, group, severity ('hard' | 'soft'), description (one Korean line),
  scope(df) -> bool Series   rows the rule applies to
  hit(df)   -> bool Series   violation (counted only inside scope)

All column access goes through `_s(df, col)`.  Value literals live in module-level
constants and are verified against the catalog by `check_values`.

These rules also informed generator changes; evaluation independence is limited.
See docs/08_모순_규칙.md.
"""
import json

import numpy as np
import pandas as pd

__all__ = ["RULES", "MissingColumn", "violation_matrix", "run", "check_values"]


# =============================================================================
# Column accessor and small helpers (every helper reads columns only via _s)
# =============================================================================
class MissingColumn(KeyError):
    """Raised when a rule needs a column that the frame does not have."""


def _s(df, col):
    """The only column accessor: stripped string view of one column."""
    if col not in df.columns:
        raise MissingColumn(col)
    return df[col].astype(str).str.strip()


# (column, value) pairs referenced at run time; check_values verifies them against the catalog.
_SEEN = set()


def _in(df, col, vals):
    """Row value of `col` is one of `vals`."""
    if isinstance(vals, str):
        vals = (vals,)
    s = _s(df, col)
    for v in vals:
        _SEEN.add((col, v))
    return pd.Series(s.isin(list(vals)).to_numpy(dtype=bool), index=df.index)


def _rank(df, col, scale):
    """Position of the value in an ordered scale (low -> high); NaN when not on the scale."""
    s = _s(df, col)
    for v in scale:
        _SEEN.add((col, v))
    m = {v: float(i) for i, v in enumerate(scale)}
    return pd.Series([m.get(v, np.nan) for v in s.tolist()], index=df.index, dtype=float)


def _rows(df, col):
    """All rows (the column must exist)."""
    _s(df, col)
    return pd.Series(True, index=df.index)


def _false(df):
    return pd.Series(False, index=df.index)


def _or(*xs):
    out = xs[0]
    for x in xs[1:]:
        out = out | x
    return out


def _and(*xs):
    out = xs[0]
    for x in xs[1:]:
        out = out & x
    return out


def _any_col(df, cols, vals):
    out = _false(df)
    for c in cols:
        out = out | _in(df, c, vals)
    return out


def _all_col(df, cols, vals):
    out = pd.Series(True, index=df.index)
    for c in cols:
        out = out & _in(df, c, vals)
    return out


def _count_col(df, cols, vals):
    out = pd.Series(0, index=df.index)
    for c in cols:
        out = out + _in(df, c, vals).astype(int)
    return out


def _opp(df, a, sa, b, sb, rev=False):
    """a and b at opposite ends of their scales (rev=True: same ends, for reverse-keyed pairs)."""
    ra, rb = _rank(df, a, sa), _rank(df, b, sb)
    a_hi, a_lo = ra == len(sa) - 1, ra == 0
    b_hi, b_lo = rb == len(sb) - 1, rb == 0
    if rev:
        return (a_hi & b_hi) | (a_lo & b_lo)
    return (a_hi & b_lo) | (a_lo & b_hi)


def _ext(df, a, sa, b, sb):
    """Either a or b is at an end of its scale."""
    ra, rb = _rank(df, a, sa), _rank(df, b, sb)
    return (ra == 0) | (ra == len(sa) - 1) | (rb == 0) | (rb == len(sb) - 1)


def _opp_pairs(df, pairs):
    out = _false(df)
    for p in pairs:
        out = out | _opp(df, *p)
    return out


def _ext_pairs(df, pairs):
    out = _false(df)
    for p in pairs:
        out = out | _ext(df, *p[:4])
    return out


# =============================================================================
# Value constants (exact catalog spellings)
# =============================================================================
LV = ("Very low", "Low", "Average", "High", "Very high")
COG = ("None", "Low", "Moderate", "High", "Very high")
VIA = ("Absent", "Slight", "Moderate", "Strong", "Signature")
VALS = ("Irrelevant", "Minor", "Moderate", "Important", "Core value")
INTR = ("Averse", "Indifferent", "Neutral", "Interested", "Passionate")
ATTS = ("Opposed", "Skeptical", "Neutral", "Positive", "Enthusiast")
FAMS = ("None", "Aware", "Familiar", "Proficient", "Expert")
SKL = ("None", "Beginner", "Intermediate", "Advanced", "Master")
PRG = ("None", "Familiar", "Proficient", "Expert")
TOOL = ("Never used", "Tried it", "Occasional", "Regular", "Power user")
LNG = ("None", "Basic", "Conversational", "Fluent", "Native")
ENG = ("None", "Basic (A1-A2)", "Intermediate (B1-B2)", "Fluent (C1-C2)", "Native")
POLITE = ("Rude", "Brusque", "Neutral", "Polite", "Very polite")

ADV = ("Advanced", "Master")
PROF = ("Proficient", "Expert")
AVID = ("Avid", "Active")
HEAVY_TOOL = ("Regular", "Power user")
DAILY_WEEKLY = ("Daily", "Weekly")

# ---- age -------------------------------------------------------------------
AGES = ("18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+")
MINOR_AGES = ("Under 5", "5-12", "13-17")
A19 = ("18-24",)
A25 = ("25-34",)
A35 = ("35-44",)
A_LE34 = ("18-24", "25-34")
A_LE44 = ("18-24", "25-34", "35-44")
A_LE64 = ("18-24", "25-34", "35-44", "45-54", "55-64")
A55P = ("55-64", "65-74", "75-84", "85+")
A65P = ("65-74", "75-84", "85+")
A75P = ("75-84", "85+")
S01_AGES = ("18-24", "25-34", "35-44", "75-84", "85+")

GENERATIONS = ("Gen Alpha", "Gen Z", "Millennial", "Gen X", "Boomer", "Silent")
# Allowed cohort labels per bracket (birth year 2020-age, +-2 years at cohort borders,
# Korean MZ usage 1980-1994 / 1995+ and the Korean 2nd baby boom 1968-74 included).
GEN_OK = {
    "18-24": ("Gen Z", "Millennial"),
    "25-34": ("Millennial", "Gen Z"),
    "35-44": ("Millennial", "Gen X"),
    "45-54": ("Gen X", "Boomer"),
    "55-64": ("Boomer", "Gen X"),
    "65-74": ("Boomer", "Silent"),
    "75-84": ("Silent", "Boomer"),
    "85+": ("Silent",),
}
DECADES = ("1960s-70s", "1980s", "1990s", "2000s", "2010s", "2020s")
# Formative years taken generously as ages 5-30; '1960s-70s' is the earliest bucket.
DECADE_OK = {
    "18-24": ("2000s", "2010s", "2020s"),
    "25-34": ("1990s", "2000s", "2010s", "2020s"),
    "35-44": ("1980s", "1990s", "2000s", "2010s"),
    "45-54": ("1960s-70s", "1980s", "1990s", "2000s"),
    "55-64": ("1960s-70s", "1980s", "1990s"),
    "65-74": ("1960s-70s", "1980s"),
    "75-84": ("1960s-70s",),
    "85+": ("1960s-70s",),
}

# ---- demographics ------------------------------------------------------------
NOT_WOMAN = ("Man", "Non-binary", "Self-described", "Prefer not to say")
REGIONS_NOT_KOREA = ("North America", "Latin America", "Western Europe", "Eastern Europe",
                     "Sub-Saharan Africa", "MENA", "South Asia", "Southeast Asia", "Oceania")
NON_CITIZEN = ("Permanent resident", "Visa holder", "Undocumented")
CULT_NOT_NATIVE_KR = ("Visited", "Studied", "Unfamiliar")
ETHNIC_RARE = ("White / European", "Black / African", "Hispanic / Latino", "South Asian",
               "Southeast Asian", "Middle Eastern", "Indigenous", "Pacific Islander")
CULTURE_RARE = ("South Asian", "Latin", "African", "Middle Eastern", "Indigenous")

PARTNERED = ("Married", "Engaged", "In a relationship", "Domestic partnership")
HAS_KIDS = ("1 child", "2 children", "3+ children", "Adult children")
PARENT4 = ("New parent", "Parent of minors", "Parent of adults", "Grandparent")
PARENTAL_NOT_STEP = ("Not a parent", "New parent", "Parent of minors", "Parent of adults", "Grandparent")
JOURNEY_ALL = ("No children", "Young children", "Teenagers", "Grown children", "Empty nester",
               "Raising grandchildren")
JOURNEY_KIDS = JOURNEY_ALL[1:]
ADULT_CHILD_PARENTAL = ("Parent of adults", "Grandparent")
ADULT_CHILD_JOURNEY = ("Grown children", "Empty nester", "Raising grandchildren")
EVER_MARRIED_HIST = ("Married once", "Remarried", "Widowed")

SERVED = ("Served briefly", "Career veteran", "Combat veteran")
EMPLOYED = ("Full-time", "Part-time", "Self-employed", "Gig / freelance", "Unpaid family worker")
WORK_SCHED_ACTIVE = ("9-to-5", "Shift work", "On-call", "Freelance")
EXEC = ("Director", "VP", "C-suite")
NO_VENTURE = ("Never considered", "Considered it")

# ---- education ---------------------------------------------------------------
BELOW_GRAD = ("No formal", "Primary", "Secondary", "Vocational / cert", "Some college",
              "Associate's", "Bachelor's")
GRAD = ("Master's", "Doctorate", "Postdoc")
DEGREE = ("Associate's", "Bachelor's", "Master's", "Doctorate", "Postdoc")
UNI_TIER = ("Top-tier research", "Mid-tier university", "Community college")
LOW_EDU = ("No formal", "Primary")
NO_TERTIARY = ("No formal", "Primary", "Secondary")
MED_FAM = ("fam_cardiology", "fam_neurology", "fam_oncology", "fam_pediatrics", "fam_psychiatry",
           "fam_radiology", "fam_surgery", "fam_immunology", "fam_nursing", "fam_pharmacology",
           "fam_public_health", "fam_epidemiology", "fam_nutrition", "fam_veterinary_medicine")
LAW_FAM = ("fam_constitutional_law", "fam_contract_law", "fam_criminal_law", "fam_tax_law",
           "fam_intellectual_property")

# ---- column families ---------------------------------------------------------
LANG_COLS = (
    "lang_korean", "lang_cantonese", "lang_italian", "lang_turkish", "lang_vietnamese", "lang_thai",
    "lang_indonesian", "lang_malay", "lang_dutch", "lang_polish", "lang_ukrainian", "lang_persian",
    "lang_hebrew", "lang_greek", "lang_czech", "lang_hungarian", "lang_romanian", "lang_swedish",
    "lang_norwegian", "lang_danish", "lang_finnish", "lang_tagalog", "lang_urdu", "lang_tamil",
    "lang_telugu", "lang_marathi", "lang_punjabi", "lang_gujarati", "lang_hausa", "lang_yoruba",
    "lang_igbo", "lang_amharic", "lang_zulu", "lang_afrikaans", "lang_serbian", "lang_croatian",
    "lang_bulgarian", "lang_slovak", "lang_mandarin", "lang_spanish", "lang_hindi", "lang_arabic",
    "lang_french", "lang_portuguese", "lang_bengali", "lang_russian", "lang_japanese", "lang_german",
    "lang_swahili", "lang_english")
FOREIGN_LANG_COLS = tuple(c for c in LANG_COLS if c != "lang_korean")
PRIMARY_LANG_COL = {
    "English": "lang_english", "Mandarin": "lang_mandarin", "Spanish": "lang_spanish",
    "Hindi": "lang_hindi", "Arabic": "lang_arabic", "French": "lang_french",
    "Portuguese": "lang_portuguese", "Bengali": "lang_bengali", "Russian": "lang_russian",
    "Japanese": "lang_japanese", "German": "lang_german", "Swahili": "lang_swahili",
    "Korean": "lang_korean"}
NON_KOREAN_PRIMARY = tuple(k for k in PRIMARY_LANG_COL if k != "Korean")
LANG_WEAK = ("None", "Basic", "Conversational")
ENG_WEAK = ("None", "Basic (A1-A2)", "Intermediate (B1-B2)")

FOREIGN_CULT_COLS = (
    "cult_mexico", "cult_brazil", "cult_argentina", "cult_united_kingdom", "cult_france",
    "cult_germany", "cult_italy", "cult_spain", "cult_netherlands", "cult_sweden", "cult_poland",
    "cult_russia", "cult_turkey", "cult_egypt", "cult_saudi_arabia", "cult_uae", "cult_israel",
    "cult_iran", "cult_nigeria", "cult_kenya", "cult_south_africa", "cult_ethiopia", "cult_india",
    "cult_pakistan", "cult_bangladesh", "cult_china", "cult_japan", "cult_vietnam", "cult_thailand",
    "cult_indonesia", "cult_philippines", "cult_australia", "cult_new_zealand", "cult_singapore",
    "cult_malaysia", "cult_greece", "cult_portugal", "cult_united_states", "cult_canada")

HOB_COLS = (
    "hob_karaoke", "hob_crocheting", "hob_pottery", "hob_metalworking", "hob_leatherworking",
    "hob_candle_making", "hob_origami", "hob_scrapbooking", "hob_quilting", "hob_embroidery",
    "hob_jewelry_making", "hob_model_building", "hob_antiquing", "hob_bonsai", "hob_aquariums",
    "hob_beekeeping", "hob_stargazing", "hob_geocaching", "hob_rock_climbing", "hob_bouldering",
    "hob_kayaking", "hob_paddleboarding", "hob_scuba_diving", "hob_snorkeling",
    "hob_horseback_riding", "hob_foraging", "hob_winemaking", "hob_cheesemaking",
    "hob_salsa_dancing", "hob_improv", "hob_juggling", "hob_whittling", "hob_letterpress",
    "hob_metal_detecting", "hob_cosplay", "hob_skydiving", "hob_paragliding",
    "hob_urban_sketching", "hob_home_brewing", "hob_birdwatching", "hob_genealogy",
    "hob_coin_collecting", "hob_stamp_collecting", "hob_knitting", "hob_woodworking",
    "hob_calligraphy", "hob_ballroom_dance", "hob_stand_up_comedy", "hob_bread_baking",
    "hob_vegetable_gardening")
# Karaoke (noraebang) is a routine social outing in Korea, not necessarily a hobby.
HOBBY_COLS_FOR_INTENSITY = tuple(c for c in HOB_COLS if c != "hob_karaoke")
ACTIVE_HOBBIES = ("hob_rock_climbing", "hob_bouldering", "hob_kayaking", "hob_paddleboarding",
                  "hob_scuba_diving", "hob_horseback_riding", "hob_salsa_dancing",
                  "hob_ballroom_dance")

SPORT_COLS = (
    "sport_volleyball", "sport_wrestling", "sport_track_field", "sport_yoga", "sport_weightlifting",
    "sport_triathlon", "sport_soccer", "sport_basketball", "sport_american_football",
    "sport_baseball", "sport_tennis", "sport_golf", "sport_cricket", "sport_rugby", "sport_hockey",
    "sport_swimming", "sport_running", "sport_cycling", "sport_boxing", "sport_mma", "sport_skiing",
    "sport_snowboarding", "sport_surfing", "sport_skateboarding", "sport_climbing",
    "sport_gymnastics", "sport_badminton", "sport_table_tennis", "sport_squash", "sport_sailing",
    "sport_rowing", "sport_martial_arts", "sport_pilates", "sport_crossfit", "sport_esports",
    "sport_darts", "sport_bowling", "sport_archery", "sport_equestrian", "sport_fencing")
# Playing these is physical exercise (esports, darts, bowling and screen-golf-heavy golf excluded).
PHYS_SPORTS = tuple(c for c in SPORT_COLS
                    if c not in ("sport_esports", "sport_darts", "sport_bowling", "sport_golf"))
SPECTATOR_SPORTS = tuple(c for c in SPORT_COLS
                         if c not in ("sport_esports", "sport_yoga", "sport_pilates"))
MOBILITY_HEAVY_SPORTS = ("sport_running", "sport_soccer", "sport_basketball", "sport_tennis",
                         "sport_volleyball", "sport_badminton", "sport_squash", "sport_rugby",
                         "sport_hockey", "sport_american_football", "sport_triathlon",
                         "sport_crossfit", "sport_skiing", "sport_snowboarding",
                         "sport_skateboarding", "sport_climbing", "sport_track_field",
                         "sport_gymnastics", "sport_surfing")

FILM_COLS = (
    "filmg_thriller", "filmg_sci_fi", "filmg_romance", "filmg_documentary", "filmg_war",
    "filmg_western", "filmg_superhero", "filmg_action", "filmg_adventure", "filmg_comedy",
    "filmg_drama", "filmg_horror", "filmg_fantasy", "filmg_animation", "filmg_crime",
    "filmg_mystery", "filmg_historical", "filmg_musical", "filmg_noir", "filmg_indie_film",
    "filmg_art_house", "filmg_biopic", "filmg_comedy_drama", "filmg_disaster")
MUSIC_COLS = (
    "musg_pop", "musg_rock", "musg_classical", "musg_techno", "musg_trance", "musg_soul", "musg_ska",
    "musg_synthwave", "musg_trap", "musg_hip_hop", "musg_r_b", "musg_jazz", "musg_blues",
    "musg_opera", "musg_country", "musg_folk", "musg_reggae", "musg_reggaeton", "musg_electronic",
    "musg_house", "musg_drum_bass", "musg_metal", "musg_punk", "musg_indie", "musg_k_pop",
    "musg_j_pop", "musg_latin", "musg_afrobeats", "musg_gospel", "musg_funk", "musg_disco",
    "musg_ambient", "musg_lo_fi", "musg_bluegrass", "musg_bollywood")
BOOK_COLS = (
    "bookg_literary_fiction", "bookg_science_fiction", "bookg_fantasy", "bookg_mystery",
    "bookg_thriller", "bookg_romance", "bookg_historical_fiction", "bookg_horror",
    "bookg_biography", "bookg_memoir", "bookg_self_help", "bookg_business",
    "bookg_popular_science", "bookg_history", "bookg_philosophy", "bookg_poetry",
    "bookg_young_adult", "bookg_graphic_novels", "bookg_true_crime", "bookg_travel_writing",
    "bookg_cookbooks", "bookg_essays")

IND_COLS = (
    "ind_retail", "ind_e_commerce", "ind_automotive", "ind_oil_gas", "ind_utilities",
    "ind_construction", "ind_travel_tourism", "ind_restaurants", "ind_agriculture",
    "ind_food_beverage", "ind_pharmaceuticals", "ind_biotech", "ind_telecommunications",
    "ind_entertainment", "ind_publishing", "ind_advertising", "ind_government", "ind_defense",
    "ind_nonprofit", "ind_transportation", "ind_shipping", "ind_chemicals", "ind_apparel",
    "ind_consumer_electronics", "ind_semiconductors", "ind_legal_services", "ind_consulting",
    "ind_fitness_wellness", "ind_beauty_cosmetics", "ind_finance", "ind_banking", "ind_accounting",
    "ind_education", "ind_aerospace", "ind_fine_art", "ind_healthcare", "ind_sports", "ind_gaming",
    "ind_technology", "ind_marketing_agencies", "ind_logistics", "ind_manufacturing",
    "ind_real_estate", "ind_insurance", "ind_hospitality", "ind_textiles", "ind_media", "ind_music",
    "ind_energy", "ind_mining")

PROG_COLS = (
    "prog_ruby", "prog_php", "prog_kotlin", "prog_objective_c", "prog_haskell", "prog_elixir",
    "prog_erlang", "prog_clojure", "prog_lua", "prog_perl", "prog_r", "prog_julia",
    "prog_powershell", "prog_dart", "prog_f", "prog_ocaml", "prog_assembly", "prog_cobol",
    "prog_fortran", "prog_graphql", "prog_matlab", "prog_python", "prog_javascript", "prog_java",
    "prog_c", "prog_go", "prog_bash", "prog_typescript", "prog_rust", "prog_swift", "prog_scala",
    "prog_sql", "prog_solidity")

AI_TASK_COLS = (
    "ai_task_code_generation", "ai_task_code_review", "ai_task_codebase_learning",
    "ai_task_data_generation_analytics", "ai_task_debugging_fixing",
    "ai_task_deployment_monitoring", "ai_task_documentation", "ai_task_learning_concepts",
    "ai_task_project_planning", "ai_task_refactoring", "ai_task_search_answers", "ai_task_testing")
AI_CURRENT = ("Currently mostly AI-assisted", "Currently partially AI-assisted")
AI_NOT_CURRENT = ("Plans mostly AI use", "Plans partial AI use", "Does not plan AI use",
                  "Not applicable")
AI_NONUSE = ("Never used", "Actively avoids")
AGENT_NONUSE = ("Not yet but interested", "No plans")
AGENT_IMPACT_USED = ("Major productivity increase", "Moderate productivity increase",
                     "Small targeted improvement", "Negative / slowdown")
GITHUB_ACTIVE = ("Code pusher", "Pull request contributor", "Issue reporter",
                 "Reviewer / commenter", "Repository maintainer", "Star / fork explorer")
GITHUB_CODE = ("Code pusher", "Pull request contributor", "Repository maintainer")
PR_AUTHORING = ("Small focused PRs", "Moderate scoped PRs", "Large feature PRs",
                "Draft / iterative PRs")
RELEASE_INVOLVED = ("Active release maintainer", "Occasional release contributor",
                    "Dependency / update maintainer", "Internal-only maintainer")
SO_PARTICIPATES = ("Reads / searches only", "Votes / bookmarks", "Comments / discusses",
                   "Asks questions", "Answers questions", "Moderates / curates")
DEV_TOOLS = ("tool_git", "tool_docker", "tool_kubernetes", "tool_terraform", "tool_linux_cli",
             "tool_vim", "tool_jetbrains_ides", "tool_vs_code", "tool_aws", "tool_azure",
             "tool_google_cloud", "tool_github", "tool_gitlab", "tool_jupyter", "tool_postman")
NON_APPLE = ("Android/Google", "Windows", "Linux")

MBTI_TYPES = ("INTJ — Architect", "INTP — Logician", "ENTJ — Commander", "ENTP — Debater",
              "INFJ — Advocate", "INFP — Mediator", "ENFJ — Protagonist", "ENFP — Campaigner",
              "ISTJ — Logistician", "ISFJ — Defender", "ESTJ — Executive", "ESFJ — Consul",
              "ISTP — Virtuoso", "ISFP — Adventurer", "ESTP — Entrepreneur", "ESFP — Entertainer")
MBTI_E = tuple(t for t in MBTI_TYPES if t[0] == "E")
MBTI_I = tuple(t for t in MBTI_TYPES if t[0] == "I")
MBTI_N = tuple(t for t in MBTI_TYPES if t[1] == "N")
MBTI_S = tuple(t for t in MBTI_TYPES if t[1] == "S")
MBTI_J = tuple(t for t in MBTI_TYPES if t[3] == "J")
MBTI_P = tuple(t for t in MBTI_TYPES if t[3] == "P")

BFI2_DOMAINS = {
    "bfi2_domain_open_mindedness": ("bfi2_facet_intellectual_curiosity",
                                    "bfi2_facet_aesthetic_sensitivity",
                                    "bfi2_facet_creative_imagination"),
    "bfi2_domain_conscientiousness": ("bfi2_facet_organization", "bfi2_facet_productiveness",
                                      "bfi2_facet_responsibility"),
    "bfi2_domain_extraversion": ("bfi2_facet_sociability", "bfi2_facet_assertiveness",
                                 "bfi2_facet_energy_level"),
    "bfi2_domain_agreeableness": ("bfi2_facet_compassion", "bfi2_facet_respectfulness",
                                  "bfi2_facet_trust"),
    "bfi2_domain_negative_emotionality": ("bfi2_facet_anxiety", "bfi2_facet_depression",
                                          "bfi2_facet_emotional_volatility"),
}
DOMINANT_TO_DOMAIN = {
    "High openness": "bfi2_domain_open_mindedness",
    "High conscientiousness": "bfi2_domain_conscientiousness",
    "High extraversion": "bfi2_domain_extraversion",
    "High agreeableness": "bfi2_domain_agreeableness",
    "High neuroticism": "bfi2_domain_negative_emotionality",
}

# BFI-2 facet <-> IPIP-NEO facet measuring the same construct, by Big Five domain.
FACET_PAIRS = {
    "N": (("bfi2_facet_anxiety", LV, "big5_anxiety", LV),
          ("bfi2_facet_depression", LV, "big5_depression", LV),
          ("bfi2_facet_emotional_volatility", LV, "big5_anger", LV)),
    "E": (("bfi2_facet_sociability", LV, "big5_gregariousness", LV),
          ("bfi2_facet_assertiveness", LV, "big5_assertiveness", LV),
          ("bfi2_facet_energy_level", LV, "big5_activity_level", LV)),
    "A": (("bfi2_facet_compassion", LV, "big5_sympathy", LV),
          ("bfi2_facet_respectfulness", LV, "big5_cooperation", LV),
          ("bfi2_facet_trust", LV, "big5_trust", LV)),
    "C": (("bfi2_facet_organization", LV, "big5_orderliness", LV),
          ("bfi2_facet_productiveness", LV, "big5_self_discipline", LV),
          ("bfi2_facet_responsibility", LV, "big5_dutifulness", LV)),
    "O": (("bfi2_facet_intellectual_curiosity", LV, "big5_intellect", LV),
          ("bfi2_facet_aesthetic_sensitivity", LV, "big5_artistic_interest", LV),
          ("bfi2_facet_creative_imagination", LV, "big5_imagination", LV)),
}
# VIA character strengths <-> personality facets of the same construct.
VIA_OC_PAIRS = (("trait_curiosity", VIA, "bfi2_facet_intellectual_curiosity", LV),
                ("trait_creativity", VIA, "bfi2_facet_creative_imagination", LV),
                ("trait_appreciation_of_beauty", VIA, "bfi2_facet_aesthetic_sensitivity", LV),
                ("trait_prudence", VIA, "big5_cautiousness", LV),
                ("trait_discipline", VIA, "big5_self_discipline", LV),
                ("trait_ambition", VIA, "big5_achievement_striving", LV),
                ("trait_perseverance", VIA, "bfi2_facet_productiveness", LV))
VIA_AEN_PAIRS = (("trait_humility", VIA, "big5_modesty", LV),
                 ("trait_kindness", VIA, "big5_altruism", LV),
                 ("trait_honesty", VIA, "big5_morality", LV),
                 ("trait_zest", VIA, "bfi2_facet_energy_level", LV),
                 ("trait_self_regulation", VIA, "big5_immoderation", LV, True),
                 ("trait_resilience", VIA, "big5_vulnerability", LV, True))
COG_VIA_PAIRS = (("cog_curiosity", COG, "trait_curiosity", VIA),
                 ("cog_curiosity", COG, "bfi2_facet_intellectual_curiosity", LV),
                 ("cog_open_mindedness", COG, "trait_open_mindedness", VIA),
                 ("cog_optimism", COG, "trait_hope_optimism", VIA))
STYLE_PAIRS = (("cog_assertiveness", COG, "bfi2_facet_assertiveness", LV),
               ("interpersonal_agency_dominance", LV, "bfi2_facet_assertiveness", LV),
               ("cog_politeness", POLITE, "bfi2_facet_respectfulness", LV),
               ("interpersonal_communion_warmth", LV, "big5_friendliness", LV))
SCHWARTZ_PAIRS_1 = (("schwartz_value_self_direction", LV, "val_independence", VALS),
                    ("schwartz_value_stimulation", LV, "val_adventure", VALS),
                    ("schwartz_value_hedonism", LV, "val_fun_enjoyment", VALS),
                    ("schwartz_value_achievement", LV, "val_achievement", VALS),
                    ("schwartz_value_power", LV, "val_power_influence", VALS))
SCHWARTZ_PAIRS_2 = (("schwartz_value_security", LV, "val_security_stability", VALS),
                    ("schwartz_value_tradition", LV, "val_tradition", VALS),
                    ("schwartz_value_benevolence", LV, "val_helping_others", VALS),
                    ("schwartz_value_universalism", LV, "val_equality", VALS))
PRIORITY_TO_VAL = {"Achievement": ("val_achievement",),
                   "Security": ("val_security_stability",),
                   "Autonomy": ("val_independence", "val_personal_freedom"),
                   "Community": ("val_community",),
                   "Novelty": ("val_adventure",),
                   "Tradition": ("val_tradition",)}
PRIORITY_TO_SCHWARTZ = {"Achievement": ("schwartz_value_achievement",),
                        "Security": ("schwartz_value_security",),
                        "Autonomy": ("schwartz_value_self_direction",),
                        "Community": ("schwartz_value_benevolence", "schwartz_value_universalism"),
                        "Novelty": ("schwartz_value_stimulation",),
                        "Tradition": ("schwartz_value_tradition",)}
MFT_VALUE_PAIRS = (("mft_loyalty_betrayal", LV, "val_loyalty", VALS),
                   ("mft_fairness_cheating", LV, "val_justice_fairness", VALS),
                   ("mft_liberty_oppression", LV, "val_personal_freedom", VALS),
                   ("mft_care_harm", LV, "val_helping_others", VALS),
                   ("trait_honesty", VIA, "val_integrity_honesty", VALS),
                   ("trait_loyalty", VIA, "val_loyalty", VALS))

# ---- expertise ----------------------------------------------------------------
SUBFIELD_PAIRS = (("fam_deep_learning", "fam_machine_learning"),
                  ("fam_astrophysics", "fam_astronomy"),
                  ("fam_auditing", "fam_accounting"),
                  ("fam_actuarial_science", "fam_statistics"))
RELATED_FIELD_PAIRS = (("fam_computer_vision", "fam_machine_learning"),
                       ("fam_natural_language_processing", "fam_machine_learning"),
                       ("fam_data_science", "fam_statistics"),
                       ("fam_epidemiology", "fam_statistics"),
                       ("fam_lean_manufacturing", "fam_operations_management"),
                       ("fam_ux_research", "fam_human_computer_interaction"))
SKILL_FAM_PAIRS = (("skill_photography", ("fam_photography",)),
                   ("skill_project_management", ("fam_project_management",)),
                   ("skill_statistical_modeling", ("fam_statistics",)),
                   ("skill_audio_production", ("fam_sound_engineering", "fam_music_production")),
                   ("skill_financial_modeling", ("fam_corporate_finance",)))
SPECIALTY_FAM = {"Machine learning": "fam_machine_learning", "Cardiology": "fam_cardiology",
                 "Constitutional law": "fam_constitutional_law",
                 "Quant trading": "fam_quantitative_trading",
                 "Curriculum design": "fam_curriculum_design",
                 "Structural engineering": "fam_structural_engineering",
                 "Molecular biology": "fam_molecular_biology",
                 "Behavioral economics": "fam_behavioral_economics",
                 "Comparative literature": "fam_comparative_literature",
                 "Operations": "fam_operations_management", "Agronomy": "fam_agronomy",
                 "Robotics": "fam_robotics", "Investigative reporting": "fam_journalism",
                 "Urban planning": "fam_urban_planning", "Culinary arts": "fam_culinary_arts"}
DOMAINS = ("Software & AI", "Healthcare & Medicine", "Law & Policy", "Finance & Economics",
           "Education", "Engineering", "Natural Sciences", "Social Sciences", "Arts & Humanities",
           "Business & Management", "Agriculture", "Manufacturing", "Media & Journalism",
           "Public Sector", "Hospitality", "Skilled Trades")  # all except 'Not applicable'
SPECIALTY_DOMAINS = {
    "Cardiology": ("Healthcare & Medicine", "Natural Sciences", "Education"),
    "Constitutional law": ("Law & Policy", "Public Sector", "Education", "Social Sciences"),
    "Quant trading": ("Finance & Economics", "Software & AI", "Business & Management",
                      "Natural Sciences", "Education"),
    "Molecular biology": ("Natural Sciences", "Healthcare & Medicine", "Education", "Agriculture",
                          "Engineering", "Manufacturing"),
    "Agronomy": ("Agriculture", "Natural Sciences", "Education", "Public Sector"),
    "Comparative literature": ("Arts & Humanities", "Education", "Media & Journalism",
                               "Social Sciences"),
    "Investigative reporting": ("Media & Journalism", "Social Sciences", "Arts & Humanities",
                                "Law & Policy", "Public Sector", "Education"),
    "Structural engineering": ("Engineering", "Skilled Trades", "Manufacturing", "Public Sector",
                               "Education", "Natural Sciences"),
}
IND_FAM_PAIRS = (("ind_accounting", "fam_accounting"), ("ind_real_estate", "fam_real_estate"),
                 ("ind_insurance", "fam_insurance"), ("ind_logistics", "fam_logistics"),
                 ("ind_mining", "fam_mining"), ("ind_textiles", "fam_textiles"))
QUANT_SKILLS = ("skill_mathematics", "skill_mental_arithmetic", "skill_statistical_modeling",
                "skill_financial_modeling")
VISUAL_SKILLS = ("skill_drawing", "skill_painting", "skill_sketching", "skill_photography",
                 "skill_videography", "skill_video_editing")

# ---- interests ----------------------------------------------------------------
HOBBY_TOPIC_PAIRS = (("hob_birdwatching", "topic_birdwatching"), ("hob_genealogy", "topic_genealogy"),
                     ("hob_knitting", "topic_knitting"), ("hob_woodworking", "topic_woodworking"),
                     ("hob_calligraphy", "topic_calligraphy"),
                     ("hob_stand_up_comedy", "topic_stand_up_comedy"),
                     ("hob_bread_baking", "topic_baking"),
                     ("hob_vegetable_gardening", "topic_gardening"),
                     ("hob_stargazing", "topic_astronomy"),
                     ("hob_coin_collecting", "topic_collecting"),
                     ("hob_stamp_collecting", "topic_collecting"),
                     ("hob_salsa_dancing", "topic_dance"), ("hob_ballroom_dance", "topic_dance"),
                     ("hob_winemaking", "topic_wine"), ("hob_home_brewing", "topic_craft_beer"))
BOOK_TOPIC_PAIRS = (("bookg_literary_fiction", "topic_literature"), ("bookg_poetry", "topic_poetry"),
                    ("bookg_history", "topic_history"), ("bookg_philosophy", "topic_philosophy"),
                    ("bookg_true_crime", "topic_true_crime"),
                    ("bookg_popular_science", "topic_science"),
                    ("bookg_travel_writing", "topic_travel"),
                    ("bookg_graphic_novels", "topic_comics"),
                    ("bookg_self_help", "topic_self_improvement"))
ACAD_TOPIC_PAIRS = (("acad_literature", INTR, "topic_literature", INTR),
                    ("acad_philosophy", INTR, "topic_philosophy", INTR),
                    ("acad_visual_art", INTR, "topic_visual_art", INTR),
                    ("acad_music", INTR, "topic_music", INTR),
                    ("acad_astronomy", INTR, "topic_astronomy", INTR),
                    ("acad_world_history", INTR, "topic_history", INTR),
                    ("acad_drama", INTR, "topic_theater", INTR),
                    ("acad_environmental_science", INTR, "topic_environment", INTR))
ATT_TOPIC_PAIRS = (("att_cryptocurrency", "topic_cryptocurrency"),
                   ("att_space_exploration", "topic_space"),
                   ("att_social_media", "topic_social_media"),
                   ("att_new_technology", "topic_technology"),
                   ("att_organized_religion", "topic_religion"))

RELIGIOUS = ("Christian", "Muslim", "Hindu", "Buddhist", "Jewish", "Sikh", "Catholic", "Protestant",
             "Won Buddhist", "Korean new religion", "Other religion")
NO_RELIGION = ("Atheist / agnostic", "None")

# Column-family lists checked statically by check_values.
_COLUMN_LISTS = {
    "LANG_COLS": LANG_COLS, "FOREIGN_CULT_COLS": FOREIGN_CULT_COLS, "HOB_COLS": HOB_COLS,
    "SPORT_COLS": SPORT_COLS, "FILM_COLS": FILM_COLS, "MUSIC_COLS": MUSIC_COLS,
    "BOOK_COLS": BOOK_COLS, "IND_COLS": IND_COLS, "PROG_COLS": PROG_COLS,
    "AI_TASK_COLS": AI_TASK_COLS, "DEV_TOOLS": DEV_TOOLS, "MED_FAM": MED_FAM, "LAW_FAM": LAW_FAM,
    "QUANT_SKILLS": QUANT_SKILLS, "VISUAL_SKILLS": VISUAL_SKILLS,
    "PRIMARY_LANG_COL": tuple(PRIMARY_LANG_COL.values()),
    "SPECIALTY_FAM": tuple(SPECIALTY_FAM.values()),
    "BFI2_DOMAINS": tuple(BFI2_DOMAINS) + tuple(c for v in BFI2_DOMAINS.values() for c in v),
}

# =============================================================================
# Rule registry
# =============================================================================
RULES = []


def _rule(rid, group, severity, description, scope, hit):
    RULES.append({"id": rid, "group": group, "severity": severity,
                  "description": description, "scope": scope, "hit": hit})


# -----------------------------------------------------------------------------
# population (population definition only: 19+, Korean citizen, born and living in Korea)
# -----------------------------------------------------------------------------
_rule("R3-POP-H01", "population", "hard",
      "연령대가 19세 미만(모집단은 19세 이상 성인)",
      lambda d: _rows(d, "age_bracket"),
      lambda d: _in(d, "age_bracket", MINOR_AGES))
_rule("R3-POP-H02", "population", "hard",
      "시민권 상태가 영주권·비자·미등록 등 비국민(모집단은 한국 국적자)",
      lambda d: _rows(d, "demo_citizenship_status"),
      lambda d: _in(d, "demo_citizenship_status", NON_CITIZEN))
_rule("R3-POP-H03", "population", "hard",
      "이민 1세대(해외 출생)로 국내 출생 모집단과 불일치",
      lambda d: _rows(d, "lifex_immigration_generation"),
      lambda d: _in(d, "lifex_immigration_generation", "First-generation immigrant"))
_rule("R3-POP-H04", "population", "hard",
      "거주 세계 지역이 동아시아가 아님(국내 거주 한국인 모집단과 불일치)",
      lambda d: _rows(d, "region"),
      lambda d: _in(d, "region", REGIONS_NOT_KOREA))
_rule("R3-POP-H05", "population", "hard",
      "국내 출생·거주 한국인인데 한국 문화 친숙도가 방문/학습/생소",
      lambda d: _rows(d, "cult_south_korea"),
      lambda d: _in(d, "cult_south_korea", CULT_NOT_NATIVE_KR))
_rule("R3-POP-S01", "population", "soft",
      "국내 출생 국민인데 귀화 시민 또는 이민 2세(2020년 성인 중 1% 미만)",
      lambda d: _rows(d, "demo_citizenship_status"),
      lambda d: _in(d, "demo_citizenship_status", "Naturalized citizen")
      | _in(d, "lifex_immigration_generation", "Second-generation"))
_rule("R3-POP-S02", "population", "soft",
      "국내 출생 한국인인데 민족·문화 배경이 비동아시아계(다문화 출신 성인은 1% 미만)",
      lambda d: _rows(d, "demo_ethnicity_broad"),
      lambda d: _in(d, "demo_ethnicity_broad", ETHNIC_RARE)
      | _in(d, "cultural_background", CULTURE_RARE))


# -----------------------------------------------------------------------------
# age (life-course arithmetic, birth year = 2020 - age)
# -----------------------------------------------------------------------------
def _h_age_generation(d):
    out = _false(d)
    for b, ok in GEN_OK.items():
        bad = tuple(g for g in GENERATIONS if g not in ok)
        out = out | (_in(d, "age_bracket", b) & _in(d, "demo_generation", bad))
    return out


def _h_age_decade(d):
    out = _false(d)
    for b, ok in DECADE_OK.items():
        bad = tuple(x for x in DECADES if x not in ok)
        out = out | (_in(d, "age_bracket", b) & _in(d, "lifex_formative_decade", bad))
    return out


def _adult_child_signal(d):
    return _or(_in(d, "demo_children_count", "Adult children"),
               _in(d, "demo_parental_status", ADULT_CHILD_PARENTAL),
               _in(d, "lifex_parenting_journey", ADULT_CHILD_JOURNEY),
               _in(d, "life_stage", "Empty nester"))


def _h_female_fertility(d):
    return _in(d, "gender_identity", "Woman") & _or(
        _in(d, "age_bracket", A55P) & _in(d, "demo_children_count", "Expecting"),
        _in(d, "age_bracket", A65P) & (_in(d, "demo_parental_status", "New parent")
                                       | _in(d, "lifex_parenting_journey", "Young children")),
        _in(d, "age_bracket", A75P) & (_in(d, "demo_parental_status", "Parent of minors")
                                       | _in(d, "lifex_parenting_journey", "Teenagers")))


def _h_family_timeline(d):
    return _or(
        _in(d, "age_bracket", A25) & _adult_child_signal(d),
        _in(d, "age_bracket", A35) & (_in(d, "demo_parental_status", "Grandparent")
                                      | _in(d, "lifex_parenting_journey", "Raising grandchildren")),
        _in(d, "age_bracket", A19) & _or(
            _in(d, "demo_marital_status", ("Widowed", "Divorced", "Separated")),
            _in(d, "lifex_relationship_history", ("Remarried", "Widowed")),
            _in(d, "demo_children_count", ("2 children", "3+ children"))),
        _in(d, "age_bracket", A75P) & _in(d, "gender_identity", NOT_WOMAN) & _or(
            _in(d, "demo_children_count", "Expecting"),
            _in(d, "demo_parental_status", ("New parent", "Parent of minors")),
            _in(d, "lifex_parenting_journey", ("Young children", "Teenagers"))))


def _h_career_timing(d):
    return _or(
        _in(d, "age_bracket", A19) & _or(
            _in(d, "highest_education", ("Doctorate", "Postdoc")),
            _in(d, "years_experience", "11-20"),
            _in(d, "seniority", EXEC),
            _any_col(d, IND_COLS, "Veteran")),
        _in(d, "age_bracket", A25) & _in(d, "years_experience", "20+"),
        _in(d, "age_bracket", A_LE34) & _or(_in(d, "demo_employment_status", "Retired"),
                                           _in(d, "life_stage", "Retirement"),
                                           _in(d, "seniority", "Retired")),
        _in(d, "age_bracket", A65P) & (_in(d, "demo_employment_status", "Student")
                                       | _in(d, "life_stage", "Student")))


_rule("R3-AGE-H01", "age", "hard",
      "세대 구분이 연령대(출생연도=2020-나이, 경계 ±2년 허용)와 맞지 않음",
      lambda d: _in(d, "age_bracket", AGES), _h_age_generation)
_rule("R3-AGE-H02", "age", "hard",
      "성장기(형성기) 연대가 출생연도상 5~30세 구간과 겹치지 않음",
      lambda d: _in(d, "age_bracket", AGES), _h_age_decade)
_rule("R3-AGE-H03", "age", "hard",
      "19-24세인데 성인 자녀·손주·빈 둥지(자녀 독립) 상태",
      lambda d: _in(d, "age_bracket", A19), _adult_child_signal)
_rule("R3-AGE-H04", "age", "hard",
      "19-24세인데 경력 20년 이상, 관계 15년 이상, 또는 직업군인 경력 완료(기간 산술상 불가)",
      lambda d: _in(d, "age_bracket", A19),
      lambda d: _or(_in(d, "years_experience", "20+"),
                    _in(d, "demo_relationship_length", "15+ years"),
                    _in(d, "lifex_military_history", "Career veteran")))
_rule("R3-AGE-H05", "age", "hard",
      "여성 연령상 임신·출산 불가: 55세 이상 임신 중, 65세 이상 신생아·영유아, 75세 이상 미성년·10대 자녀",
      lambda d: _in(d, "gender_identity", "Woman") & _in(d, "age_bracket", A55P),
      _h_female_fertility)
_rule("R3-AGE-S01", "age", "soft",
      "연령 대비 가족 이력 비현실(25-34세 성인 자녀/손주, 35-44세 손주, 19-24세 사별·이혼·재혼·2자녀+, 75세+ 남성 영유아·미성년 자녀; 각 1% 미만)",
      lambda d: _in(d, "age_bracket", S01_AGES), _h_family_timeline)
_rule("R3-AGE-S02", "age", "soft",
      "연령 대비 학력·경력 시점 비현실(19-24세 박사·경력 11년+·임원·업계 베테랑, 25-34세 경력 20년+, 34세 이하 은퇴, 65세+ 학생; 각 1% 미만)",
      lambda d: _rows(d, "age_bracket"), _h_career_timing)


# -----------------------------------------------------------------------------
# family
# -----------------------------------------------------------------------------
_rule("R3-FAM-H01", "family", "hard",
      "결혼·약혼·연애·동거 중인데 관계 기간이 '관계 없음'",
      lambda d: _in(d, "demo_marital_status", PARTNERED),
      lambda d: _in(d, "demo_relationship_length", "Not in one"))
_rule("R3-FAM-H02", "family", "hard",
      "자녀 수와 부모 상태 불일치(자녀 없음인데 부모·조부모, 자녀 있음인데 비부모)",
      lambda d: _in(d, "demo_children_count", "None") | _in(d, "demo_parental_status", "Not a parent"),
      lambda d: (_in(d, "demo_children_count", "None") & _in(d, "demo_parental_status", PARENT4))
      | (_in(d, "demo_children_count", HAS_KIDS) & _in(d, "demo_parental_status", "Not a parent")))
_rule("R3-FAM-H03", "family", "hard",
      "양육 여정이 자녀 수·부모 상태와 모순(자녀 없음 여정인데 부모, 양육 여정인데 비부모·자녀 없음)",
      lambda d: _in(d, "lifex_parenting_journey", JOURNEY_ALL),
      lambda d: _or(
          _in(d, "lifex_parenting_journey", "No children")
          & (_in(d, "demo_parental_status", PARENT4) | _in(d, "demo_children_count", HAS_KIDS)),
          _in(d, "lifex_parenting_journey", JOURNEY_KIDS)
          & _in(d, "demo_parental_status", "Not a parent"),
          _in(d, "lifex_parenting_journey", JOURNEY_KIDS) & _in(d, "demo_children_count", "None")
          & _in(d, "demo_parental_status", PARENTAL_NOT_STEP)))
_rule("R3-FAM-H04", "family", "hard",
      "생애 단계가 '어린 자녀 양육'·'빈 둥지'인데 자녀 없음(의붓·위탁 제외)",
      lambda d: _in(d, "life_stage", ("Parent of young kids", "Empty nester")),
      lambda d: _or(_in(d, "demo_children_count", "None"),
                    _in(d, "demo_parental_status", "Not a parent"),
                    _in(d, "lifex_parenting_journey", "No children"))
      & _in(d, "demo_parental_status", PARENTAL_NOT_STEP))
_rule("R3-FAM-H05", "family", "hard",
      "1인 가구인데 주거 형태가 '가족과 함께 거주'",
      lambda d: _in(d, "lstyle_household_size", "Lives alone"),
      lambda d: _in(d, "demo_housing_status", "Living with family"))
_rule("R3-FAM-S01", "family", "soft",
      "혼인·가구 이력 부정합(미혼인데 결혼/재혼/사별 이력, 사별인데 상실 경험이 조부모·부모뿐, 1인 가구인데 공유 주거; 드묾)",
      lambda d: _in(d, "demo_marital_status", ("Single", "Widowed"))
      | _in(d, "lstyle_household_size", "Lives alone"),
      lambda d: _or(
          _in(d, "demo_marital_status", "Single") & _in(d, "lifex_relationship_history", EVER_MARRIED_HIST),
          _in(d, "demo_marital_status", "Widowed")
          & _in(d, "lifex_loss_experience", ("Lost a grandparent", "Lost a parent")),
          _in(d, "lstyle_household_size", "Lives alone") & _in(d, "demo_housing_status", "Shared housing")))


# -----------------------------------------------------------------------------
# military (Korean conscription and Military Personnel Management Act)
# -----------------------------------------------------------------------------
_rule("R3-MIL-H01", "military", "hard",
      "복무 경험 '없음'인데 현역·제대군인이거나 군 복무가 주요 생애 사건",
      lambda d: _in(d, "lifex_military_history", "Never served"),
      lambda d: _in(d, "demo_veteran_status", ("Veteran", "Active duty"))
      | _in(d, "major_life_events", "Military service"))
_rule("R3-MIL-H02", "military", "hard",
      "65세 이상 현역 군인(군인사법 연령정년 최고 63세)",
      lambda d: _in(d, "demo_veteran_status", "Active duty"),
      lambda d: _in(d, "age_bracket", A65P))
_rule("R3-MIL-H03", "military", "hard",
      "현역 군인인데 고용 상태가 은퇴·전업주부이거나 직급이 은퇴",
      lambda d: _in(d, "demo_veteran_status", "Active duty"),
      lambda d: _in(d, "demo_employment_status", ("Retired", "Homemaker")) | _in(d, "seniority", "Retired"))
_rule("R3-MIL-S01", "military", "soft",
      "병역 이력 비현실(여성 군 복무·제대, 예비군인데 미복무, 직업·참전 군인인데 '민간인', 34세 이하 참전, 65세+ 예비군; 한국 여성 복무자 1% 미만, 예비군 의무 40세까지)",
      lambda d: _rows(d, "lifex_military_history"),
      lambda d: _or(
          _in(d, "gender_identity", "Woman") & _or(
              _in(d, "lifex_military_history", SERVED),
              _in(d, "demo_veteran_status", ("Veteran", "Active duty", "Reserve / guard")),
              _in(d, "major_life_events", "Military service")),
          _in(d, "demo_veteran_status", "Reserve / guard") & _in(d, "lifex_military_history", "Never served"),
          _in(d, "lifex_military_history", ("Career veteran", "Combat veteran"))
          & _in(d, "demo_veteran_status", "Civilian"),
          _in(d, "lifex_military_history", "Combat veteran") & _in(d, "age_bracket", A_LE34),
          _in(d, "demo_veteran_status", "Reserve / guard") & _in(d, "age_bracket", A65P)))


# -----------------------------------------------------------------------------
# education
# -----------------------------------------------------------------------------
_rule("R3-EDU-H01", "education", "hard",
      "교육 여정과 최종 학력 모순(대학원 학위 여정인데 석사 미만, 무학인데 정규 과정)",
      lambda d: _in(d, "lifex_education_journey", "Advanced degrees") | _in(d, "highest_education", "No formal"),
      lambda d: (_in(d, "lifex_education_journey", "Advanced degrees") & _in(d, "highest_education", BELOW_GRAD))
      | (_in(d, "highest_education", "No formal") & _in(d, "lifex_education_journey", "Traditional track")))
_rule("R3-EDU-H02", "education", "hard",
      "교육기관 유형과 학력 모순(학위 보유자인데 기관 '해당 없음', 무학·초졸인데 대학 기관)",
      lambda d: _in(d, "institution_tier", "Not applicable") | _in(d, "highest_education", LOW_EDU),
      lambda d: (_in(d, "institution_tier", "Not applicable") & _in(d, "highest_education", DEGREE))
      | (_in(d, "highest_education", LOW_EDU) & _in(d, "institution_tier", UNI_TIER)))
_rule("R3-EDU-H03", "education", "hard",
      "연구 실적과 학력 모순(고졸 이하인데 학위논문만, 박사인데 학위논문조차 없음)",
      lambda d: _in(d, "research_output", "Thesis only") | _in(d, "highest_education", ("Doctorate", "Postdoc")),
      lambda d: (_in(d, "research_output", "Thesis only") & _in(d, "highest_education", NO_TERTIARY))
      | (_in(d, "highest_education", ("Doctorate", "Postdoc")) & _in(d, "research_output", "None")))
_rule("R3-EDU-H04", "education", "hard",
      "의학·법학 학위 전공인데 해당 분야 세부 지식이 전부 '없음'",
      lambda d: _in(d, "highest_education", DEGREE) & _in(d, "academic_field", ("Medicine", "Law")),
      lambda d: (_in(d, "academic_field", "Medicine") & _all_col(d, MED_FAM, "None"))
      | (_in(d, "academic_field", "Law") & _all_col(d, LAW_FAM, "None")))
_rule("R3-EDU-S01", "education", "soft",
      "44세 이하인데 무학·초졸(의무교육 세대의 1% 미만)",
      lambda d: _in(d, "age_bracket", A_LE44),
      lambda d: _in(d, "highest_education", LOW_EDU))
_rule("R3-EDU-S02", "education", "soft",
      "학력 조합 비현실(중퇴 여정인데 대학원 학위, 비학생 고졸인데 대학 기관, 독학인데 대학원, 고졸 이하인데 다작 연구자·최첨단 연구자·학술지 주 정보원)",
      lambda d: _rows(d, "highest_education"),
      lambda d: _or(
          _in(d, "lifex_education_journey", "Dropped out") & _in(d, "highest_education", GRAD),
          _in(d, "highest_education", "Secondary")
          & _in(d, "institution_tier", ("Top-tier research", "Mid-tier university"))
          & ~_in(d, "demo_employment_status", "Student"),
          _in(d, "institution_tier", "Self-taught") & _in(d, "highest_education", GRAD),
          _in(d, "highest_education", NO_TERTIARY) & _or(
              _in(d, "research_output", "Prolific publisher"),
              _in(d, "domain_characteristics", "Cutting-edge researcher"),
              _in(d, "media_diet", "Academic journals"))))


# -----------------------------------------------------------------------------
# career
# -----------------------------------------------------------------------------
_rule("R3-CAR-H01", "career", "hard",
      "고용 상태와 근무 형태 모순(실업자인데 정규·교대·대기·프리랜서 근무, 취업자인데 근무 형태 '실업')",
      lambda d: _in(d, "demo_employment_status", "Unemployed") | _in(d, "lstyle_work_schedule", "Unemployed"),
      lambda d: (_in(d, "demo_employment_status", "Unemployed") & _in(d, "lstyle_work_schedule", WORK_SCHED_ACTIVE))
      | (_in(d, "lstyle_work_schedule", "Unemployed") & _in(d, "demo_employment_status", EMPLOYED)))
_rule("R3-CAR-H02", "career", "hard",
      "취업 중인데 직무가 '해당 없음'이거나 피고용자인데 소속 조직 규모가 '해당 없음'",
      lambda d: _in(d, "demo_employment_status", EMPLOYED),
      lambda d: _in(d, "role_function", "Not applicable")
      | (_in(d, "demo_employment_status", ("Full-time", "Part-time")) & _in(d, "company_size", "Not applicable")))
_rule("R3-CAR-H03", "career", "hard",
      "경력 시작 단계인데 해당 분야 경력 11년 이상(초기 경력인데 20년 이상)",
      lambda d: _in(d, "lifex_career_path_shape", "Just starting") | _in(d, "life_stage", "Early career"),
      lambda d: (_in(d, "lifex_career_path_shape", "Just starting") & _in(d, "years_experience", ("11-20", "20+")))
      | (_in(d, "life_stage", "Early career") & _in(d, "years_experience", "20+")))
_rule("R3-CAR-H04", "career", "hard",
      "창업자 직급·창업형 경력·창업 생애사건인데 창업 이력이 '고려 안 함/고려만 함'",
      lambda d: _in(d, "lifex_entrepreneurship_history", NO_VENTURE),
      lambda d: _or(_in(d, "seniority", "Founder"),
                    _in(d, "lifex_career_path_shape", "Entrepreneurial"),
                    _in(d, "major_life_events", "Started a business")))
_rule("R3-CAR-H05", "career", "hard",
      "직무가 임상(의료)인데 의료 업계 근무 경험이 '없음'(임상 업무 자체가 의료 경험)",
      lambda d: _in(d, "role_function", "Clinical"),
      lambda d: _in(d, "ind_healthcare", "None"))
_rule("R3-CAR-H06", "career", "hard",
      "사람을 관리하는 관리자(People manager)인데 인력 관리 역량이 '없음'",
      lambda d: _in(d, "dev_ic_manager_orientation", "People manager"),
      lambda d: _in(d, "skill_people_management", "None"))
_rule("R3-CAR-S01", "career", "soft",
      "직업 상태 조합 비현실(은퇴자의 정규·교대·대기 근무, 창업주·사업주인데 창업 고려조차 안 함, 농어업 직무인데 농업 경험 없음, 교육 직무인데 교육 업계 경험 없음; 기업 강사 등 예외 드묾)",
      lambda d: _rows(d, "demo_employment_status"),
      lambda d: _or(
          _in(d, "demo_employment_status", "Retired") & _in(d, "lstyle_work_schedule", ("9-to-5", "Shift work", "On-call")),
          _in(d, "dev_ic_manager_orientation", "Founder / owner") & _in(d, "lifex_entrepreneurship_history", "Never considered"),
          _in(d, "role_function", "Agriculture / fishery") & _in(d, "ind_agriculture", "None"),
          _in(d, "role_function", "Teaching") & _in(d, "ind_education", "None")))
_rule("R3-CAR-S02", "career", "soft",
      "직급과 경력·역량 불균형(경력 0-2년 임원, 경력 20년+ 인턴, 경력 시작 단계 임원·은퇴, 관리자 이상인데 인력 관리 역량 없음)",
      lambda d: _rows(d, "seniority"),
      lambda d: _or(
          _in(d, "years_experience", "0-2") & _in(d, "seniority", EXEC),
          _in(d, "seniority", "Student / intern") & _in(d, "years_experience", "20+"),
          _in(d, "lifex_career_path_shape", "Just starting") & _in(d, "seniority", EXEC + ("Retired",)),
          _in(d, "seniority", ("Manager",) + EXEC) & _in(d, "skill_people_management", "None")))


# -----------------------------------------------------------------------------
# income, housing and money
# -----------------------------------------------------------------------------
_rule("R3-INC-H01", "income", "hard",
      "가구소득과 소득 계층이 정반대 극단(<$25k인데 고소득층, $200k+인데 저소득층)",
      lambda d: _in(d, "demo_household_income", ("<$25k", "$200k+")),
      lambda d: (_in(d, "demo_household_income", "<$25k") & _in(d, "socioeconomic_band", "High income"))
      | (_in(d, "demo_household_income", "$200k+") & _in(d, "socioeconomic_band", "Low income")))
_rule("R3-INC-H02", "income", "hard",
      "은행 계좌가 없는데(Unbanked) 주 결제 수단이 체크·신용카드",
      lambda d: _in(d, "lstyle_banking_style", "Unbanked"),
      lambda d: _in(d, "lstyle_payment_pref", ("Debit card", "Credit card")))
_rule("R3-INC-H03", "income", "hard",
      "저축·소비 성향 두 문항이 정반대 극단(강한 저축가↔과소비, 자유 소비↔검소한 저축가)",
      lambda d: _in(d, "pref_save_vs_spend", ("Hard saver", "Free spender"))
      | _in(d, "lstyle_frugality", ("Frugal saver", "Splurger")),
      lambda d: (_in(d, "pref_save_vs_spend", "Hard saver") & _in(d, "lstyle_frugality", "Splurger"))
      | (_in(d, "pref_save_vs_spend", "Free spender") & _in(d, "lstyle_frugality", "Frugal saver")))
_rule("R3-INC-S01", "income", "soft",
      "소득·계층 지표 간 큰 괴리(소득과 계층 3단계 차이, 주관적 상류층인데 저소득, 주관적 하층인데 고소득, 상속 자산인데 저소득층)",
      lambda d: _rows(d, "socioeconomic_band"),
      lambda d: _or(
          _in(d, "demo_household_income", "<$25k") & _in(d, "socioeconomic_band", "Upper-middle"),
          _in(d, "demo_household_income", "$100k-200k") & _in(d, "socioeconomic_band", "Low income"),
          _in(d, "demo_household_income", "$200k+") & _in(d, "socioeconomic_band", "Lower-middle"),
          _in(d, "kr_subjective_class", "Upper class")
          & (_in(d, "socioeconomic_band", "Low income") | _in(d, "demo_household_income", "<$25k")),
          _in(d, "kr_subjective_class", "Lower class")
          & (_in(d, "socioeconomic_band", "High income") | _in(d, "demo_household_income", "$200k+")),
          _in(d, "lifex_financial_trajectory", "Inherited wealth") & _in(d, "socioeconomic_band", "Low income")))
_rule("R3-INC-S02", "income", "soft",
      "은행 계좌가 없는데 모바일 간편결제·후불결제·증권 투자 이용(국내 서비스는 계좌 연동 필요)",
      lambda d: _in(d, "lstyle_banking_style", "Unbanked"),
      lambda d: _in(d, "lstyle_payment_pref", ("Mobile wallet", "BNPL"))
      | _in(d, "lstyle_investment_style", ("Index investor", "Active trader")))


# -----------------------------------------------------------------------------
# language
# -----------------------------------------------------------------------------
def _h_primary_lang(d):
    out = _false(d)
    for lang, col in PRIMARY_LANG_COL.items():
        out = out | (_in(d, "primary_language", lang) & _in(d, col, LANG_WEAK))
    return out | (_in(d, "primary_language", "English") & _in(d, "english_proficiency", ENG_WEAK))


def _lang_levels(d):
    cols = {}
    for c in LANG_COLS:
        r = _rank(d, c, LNG)
        if c == "lang_english":
            r = pd.Series(np.fmax(r.to_numpy(), _rank(d, "english_proficiency", ENG).to_numpy()),
                          index=d.index)
        cols[c] = r
    return pd.DataFrame(cols, index=d.index)


def _h_multilingual(d):
    lv = _lang_levels(d)
    n_fluent = (lv >= 3).sum(axis=1)
    n_conv = (lv >= 2).sum(axis=1)
    return (_in(d, "multilingualism", "Monolingual") & (n_fluent >= 2)) \
        | (_in(d, "multilingualism", ("Bilingual", "Trilingual+")) & (n_conv <= 1))


def _h_lifetime_langs(d):
    lv = _lang_levels(d)
    n_conv = (lv >= 2).sum(axis=1)
    cap = (_in(d, "lifex_languages_lifetime", "One").astype(float) * 1
           + _in(d, "lifex_languages_lifetime", "Two").astype(float) * 2
           + _in(d, "lifex_languages_lifetime", "Three").astype(float) * 3)
    capped = cap > 0
    return (capped & (n_conv > cap)) \
        | (_in(d, "lifex_languages_lifetime", "One") & _in(d, "multilingualism", ("Bilingual", "Trilingual+"))) \
        | (_in(d, "lifex_languages_lifetime", "Two") & _in(d, "multilingualism", "Trilingual+"))


def _scope_eng(d):
    return _rank(d, "english_proficiency", ENG).notna() & _rank(d, "lang_english", LNG).notna()


def _h_eng(d):
    return (_rank(d, "english_proficiency", ENG) - _rank(d, "lang_english", LNG)).abs() >= 2


_rule("R3-LAN-H01", "language", "hard",
      "주 사용 언어(모국어·지배 언어)인데 해당 언어 능력이 회화·기초·없음 수준",
      lambda d: _rows(d, "primary_language"), _h_primary_lang)
_rule("R3-LAN-H02", "language", "hard",
      "영어 능력 두 문항(english_proficiency·lang_english)이 2단계 이상 차이",
      _scope_eng, _h_eng)
_rule("R3-LAN-H03", "language", "hard",
      "다언어 수준과 언어별 능력 모순(단일 언어인데 유창 2개 이상, 이중·삼중 언어인데 회화 이상 1개 이하)",
      lambda d: _in(d, "multilingualism", ("Monolingual", "Bilingual", "Trilingual+")), _h_multilingual)
_rule("R3-LAN-H04", "language", "hard",
      "평생 배운 언어 수가 현재 회화 이상 언어 수 또는 다언어 수준보다 적음",
      lambda d: _in(d, "lifex_languages_lifetime", ("One", "Two", "Three")), _h_lifetime_langs)
_rule("R3-LAN-S01", "language", "soft",
      "국내 출생 한국인인데 주 언어가 한국어가 아니거나 한국어가 회화 이하(1% 미만)",
      lambda d: _rows(d, "primary_language"),
      lambda d: _in(d, "primary_language", NON_KOREAN_PRIMARY) | _in(d, "lang_korean", LANG_WEAK))
_rule("R3-LAN-S02", "language", "soft",
      "국내 출생 한국인인데 외국어를 원어민 수준으로 구사하거나 외국 문화가 '본토'(이중문화 성장 1% 미만)",
      lambda d: _rows(d, "lang_korean"),
      lambda d: _any_col(d, FOREIGN_LANG_COLS, "Native") | _in(d, "english_proficiency", "Native")
      | _any_col(d, FOREIGN_CULT_COLS, "Native"))


# -----------------------------------------------------------------------------
# mobility, travel and culture
# -----------------------------------------------------------------------------
_rule("R3-MOB-H01", "mobility", "hard",
      "출국 경험이 없는데 외국을 방문했거나 거주한 적 있음",
      lambda d: _in(d, "lifex_travel_breadth", "Never left country"),
      lambda d: _any_col(d, FOREIGN_CULT_COLS, ("Lived there", "Visited")))
_rule("R3-MOB-H02", "mobility", "hard",
      "출국 경험이 없는데 해외 이주·귀국자·제3문화 아동·여러 문화권 거주 이력",
      lambda d: _in(d, "lifex_travel_breadth", "Never left country"),
      lambda d: _or(_in(d, "lifex_geographic_mobility", "Moved internationally"),
                    _in(d, "lifex_immigration_generation", "Returnee / repatriate"),
                    _in(d, "lifex_cultural_exposure", ("Lived across cultures", "Third-culture kid"))))
_rule("R3-MOB-H03", "mobility", "hard",
      "고향을 떠난 적이 없는데 고향과의 관계가 '자주 방문/가끔 귀향/거의 안 감/연고 없음'",
      lambda d: _in(d, "lifex_geographic_mobility", "Never left hometown"),
      lambda d: _in(d, "lifex_hometown_tie", ("Visits often", "Occasional return", "Rarely returns", "No connection")))
_rule("R3-MOB-H04", "mobility", "hard",
      "고향을 떠난 적이 없는데 해외 거주·귀국자·제3문화 아동이거나 이주·대규모 이사가 주요 사건",
      lambda d: _in(d, "lifex_geographic_mobility", "Never left hometown"),
      lambda d: _or(_in(d, "lifex_travel_breadth", ("Lived abroad", "Global nomad")),
                    _any_col(d, FOREIGN_CULT_COLS, "Lived there"),
                    _in(d, "lifex_immigration_generation", "Returnee / repatriate"),
                    _in(d, "lifex_cultural_exposure", "Third-culture kid"),
                    _in(d, "major_life_events", ("Major relocation", "Migration / immigration"))))
_rule("R3-MOB-S01", "mobility", "soft",
      "이동 이력 조합 비현실(해외 거주했는데 단일문화 노출, 글로벌 노마드인데 집순이, 출국 경험 없는데 잦은 비행기 이용)",
      lambda d: _rows(d, "lifex_travel_breadth"),
      lambda d: _or(
          _in(d, "lifex_cultural_exposure", "Monocultural") & _or(
              _in(d, "lifex_travel_breadth", ("Lived abroad", "Global nomad")),
              _in(d, "lifex_geographic_mobility", "Moved internationally"),
              _any_col(d, FOREIGN_CULT_COLS, "Lived there")),
          _in(d, "lifex_travel_breadth", "Global nomad") & _in(d, "lstyle_travel_freq", "Homebody"),
          _in(d, "lifex_travel_breadth", "Never left country") & _in(d, "lstyle_travel_freq", "Frequent flyer")))


# -----------------------------------------------------------------------------
# religion
# -----------------------------------------------------------------------------
_rule("R3-REL-H01", "religion", "hard",
      "신앙심이 독실·계율 준수인데 종교가 무신론/불가지론 또는 없음",
      lambda d: _in(d, "religiosity", ("Devout", "Observant")),
      lambda d: _in(d, "demo_religion_affiliation", NO_RELIGION))
_rule("R3-REL-H02", "religion", "hard",
      "종교 없음·무신론인데 신앙 여정이 '평생 신앙'·'신앙 복귀'",
      lambda d: _in(d, "demo_religion_affiliation", NO_RELIGION),
      lambda d: _in(d, "lifex_faith_journey", ("Lifelong faith", "Returned to faith")))
_rule("R3-REL-H03", "religion", "hard",
      "독실·계율 준수인데 신앙 여정이 '항상 세속적'·'신앙 이탈'",
      lambda d: _in(d, "religiosity", ("Devout", "Observant")),
      lambda d: _in(d, "lifex_faith_journey", ("Always secular", "Lapsed / left faith")))
_rule("R3-REL-H04", "religion", "hard",
      "독실한 신자이거나 신앙을 핵심 가치로 두면서 영성 가치 '무관'·영성 강점 '없음'",
      lambda d: _in(d, "religiosity", "Devout") | _in(d, "val_spirituality_faith", "Core value"),
      lambda d: (_in(d, "religiosity", "Devout")
                 & (_in(d, "val_spirituality_faith", "Irrelevant") | _in(d, "trait_spirituality", "Absent")))
      | (_in(d, "val_spirituality_faith", "Core value") & _in(d, "trait_spirituality", "Absent")))
_rule("R3-REL-H05", "religion", "hard",
      "종교적 식이 제한이 있는데 종교가 없음·무신론",
      lambda d: _in(d, "health_dietary_restriction", "Religious"),
      lambda d: _in(d, "demo_religion_affiliation", NO_RELIGION))
_rule("R3-REL-S01", "religion", "soft",
      "종교성 지표 간 부정합(항상 세속적인데 특정 종교 소속, 세속적인데 영적 각성 전환점·신앙이 핵심 가치, 독실한데 제도 종교 반대·종교 관심 혐오)",
      lambda d: _rows(d, "religiosity"),
      lambda d: _or(
          _in(d, "lifex_faith_journey", "Always secular") & _in(d, "demo_religion_affiliation", RELIGIOUS),
          _in(d, "religiosity", "Secular") & (_in(d, "lifex_turning_point", "Spiritual awakening")
                                              | _in(d, "val_spirituality_faith", "Core value")),
          _in(d, "religiosity", "Devout") & (_in(d, "att_organized_religion", "Opposed")
                                             | _in(d, "topic_religion", "Averse"))))


# -----------------------------------------------------------------------------
# health, disability and accessibility
# -----------------------------------------------------------------------------
_rule("R3-HLT-H01", "health", "hard",
      "시각 '실명'인데 장애 상태가 '장애 없음'",
      lambda d: _in(d, "health_vision", "Blind"),
      lambda d: _in(d, "demo_disability_status", "No disability"))
_rule("R3-HLT-H02", "health", "hard",
      "감각 장애인데 시력(정상·교정)과 청력(정상) 모두 정상",
      lambda d: _in(d, "demo_disability_status", "Sensory"),
      lambda d: _in(d, "health_vision", ("Normal", "Corrected")) & _in(d, "health_hearing", "Normal"))
_rule("R3-HLT-H03", "health", "hard",
      "만성질환 '없음'인데 장애 유형이 만성질환이거나 건강 여정이 '만성질환 관리'",
      lambda d: _in(d, "health_chronic_condition", "None"),
      lambda d: _in(d, "demo_disability_status", "Chronic illness")
      | _in(d, "lifex_health_journey", "Manages chronic condition"))
_rule("R3-HLT-H04", "health", "hard",
      "접근성 요구(시각·청각·운동)나 접근성 도구 사용 맥락인데 대응하는 손상·보조기기가 전혀 없음",
      lambda d: _in(d, "accessibility_needs", ("Visual", "Hearing", "Motor"))
      | _in(d, "device_context", "Accessibility tool"),
      lambda d: _or(
          _in(d, "accessibility_needs", "Visual") & _in(d, "health_vision", ("Normal", "Corrected"))
          & _in(d, "health_color_vision", "Typical") & _in(d, "health_contrast_need", "No")
          & _in(d, "health_text_size_need", "No") & _in(d, "health_motion_sensitivity", "None")
          & _in(d, "health_assistive_tech", "None"),
          _in(d, "accessibility_needs", "Hearing") & _in(d, "health_hearing", "Normal"),
          _in(d, "accessibility_needs", "Motor") & _in(d, "health_mobility", "Full")
          & _in(d, "health_dexterity", "Full"),
          _in(d, "device_context", "Accessibility tool") & _in(d, "health_assistive_tech", "None")
          & _in(d, "accessibility_needs", "None")))
_rule("R3-HLT-H05", "health", "hard",
      "신경유형 두 문항 불일치(한쪽은 ADHD·자폐·난독, 다른 쪽은 신경전형)",
      lambda d: _in(d, "neurotype", ("ADHD", "Autistic", "Dyslexic"))
      | _in(d, "health_neurodivergence", ("ADHD", "Autistic", "Dyslexic")),
      lambda d: (_in(d, "neurotype", ("ADHD", "Autistic", "Dyslexic")) & _in(d, "health_neurodivergence", "Neurotypical"))
      | (_in(d, "health_neurodivergence", ("ADHD", "Autistic", "Dyslexic")) & _in(d, "neurotype", "Neurotypical")))
_rule("R3-HLT-H06", "health", "hard",
      "주의력 장애 진단인데 두 신경유형 문항 모두 신경전형, 또는 ADHD인데 주의력 문제 '없음'",
      lambda d: _in(d, "health_attention_condition", "Diagnosed") | _in(d, "health_neurodivergence", "ADHD")
      | _in(d, "neurotype", "ADHD"),
      lambda d: (_in(d, "health_attention_condition", "Diagnosed") & _in(d, "health_neurodivergence", "Neurotypical")
                 & _in(d, "neurotype", "Neurotypical"))
      | ((_in(d, "health_neurodivergence", "ADHD") | _in(d, "neurotype", "ADHD"))
         & _in(d, "health_attention_condition", "None")))
_rule("R3-HLT-H07", "health", "hard",
      "체력 수준과 운동 빈도 모순(운동선수급인데 월 1회 이하, 좌식 생활인데 매일 운동·철인3종·크로스핏)",
      lambda d: _in(d, "health_fitness_level", ("Athlete", "Sedentary")),
      lambda d: (_in(d, "health_fitness_level", "Athlete") & _in(d, "lstyle_exercise_freq", ("Monthly", "Rarely", "Never")))
      | (_in(d, "health_fitness_level", "Sedentary") & _or(_in(d, "lstyle_exercise_freq", "Daily"),
                                                          _in(d, "sport_triathlon", "Play"),
                                                          _in(d, "sport_crossfit", "Play"))))
_rule("R3-HLT-H08", "health", "hard",
      "운동을 전혀 안 하는데 신체 활동 스포츠를 직접 하거나 활동적 취미(등반·카약·다이빙·승마·댄스)에 적극 참여",
      lambda d: _in(d, "lstyle_exercise_freq", "Never"),
      lambda d: _any_col(d, PHYS_SPORTS, "Play") | _any_col(d, ACTIVE_HOBBIES, AVID))
_rule("R3-HLT-H09", "health", "hard",
      "실명인데 매일·가끔 직접 운전(시각 기준 미달로 면허 불가)",
      lambda d: _in(d, "health_vision", "Blind"),
      lambda d: _in(d, "demo_driver_status", ("Daily driver", "Occasional driver")))
_rule("R3-HLT-S01", "health", "soft",
      "손상·편의 요구와 장애 상태 불일치(농·난청·저시력인데 장애 없음, 65세 미만 보행보조기 사용인데 장애 없음, 인지 편의 요구·화면낭독기·확대기 사용인데 해당 조건 없음)",
      lambda d: _rows(d, "demo_disability_status"),
      lambda d: _or(
          _in(d, "demo_disability_status", "No disability")
          & (_in(d, "health_hearing", "Deaf / hard of hearing") | _in(d, "health_vision", "Low vision")),
          _in(d, "health_mobility", "Uses mobility aid") & _in(d, "demo_disability_status", "No disability")
          & _in(d, "age_bracket", A_LE64),
          _in(d, "accessibility_needs", "Cognitive / neurodivergent")
          & _in(d, "health_neurodivergence", "Neurotypical") & _in(d, "neurotype", "Neurotypical")
          & _in(d, "health_attention_condition", "None")
          & ~_in(d, "demo_disability_status", ("Cognitive", "Multiple")),
          _in(d, "health_assistive_tech", "Screen reader") & _in(d, "health_vision", ("Normal", "Corrected"))
          & ~_in(d, "health_neurodivergence", "Dyslexic") & ~_in(d, "neurotype", "Dyslexic"),
          _in(d, "health_assistive_tech", "Magnifier") & _in(d, "health_vision", "Normal")))
_rule("R3-HLT-S02", "health", "soft",
      "손상과 그 기능이 필요한 활동의 조합 비현실(보행보조기 사용 운동선수·구기 종목 참여, 실명인데 시각예술·촬영 고급, 농·난청인데 통화만 선호, 저시력인데 매일 운전)",
      lambda d: _in(d, "health_mobility", "Uses mobility aid") | _in(d, "health_vision", ("Blind", "Low vision"))
      | _in(d, "health_hearing", "Deaf / hard of hearing"),
      lambda d: _or(
          _in(d, "health_mobility", "Uses mobility aid")
          & (_in(d, "health_fitness_level", "Athlete") | _any_col(d, MOBILITY_HEAVY_SPORTS, "Play")),
          _in(d, "health_vision", "Blind") & _any_col(d, VISUAL_SKILLS, ADV),
          _in(d, "health_hearing", "Deaf / hard of hearing") & _in(d, "pref_text_vs_call", "Calls only"),
          _in(d, "health_vision", "Low vision") & _in(d, "demo_driver_status", "Daily driver")))
_rule("R3-HLT-S03", "health", "soft",
      "운동을 전혀 안 한다면서 매일 아침 걷기·스트레칭을 하거나 피트니스에 열정적",
      lambda d: _in(d, "lstyle_exercise_freq", "Never"),
      lambda d: _or(_in(d, "habit_stretching_mobility", "Daily"), _in(d, "habit_morning_walk", "Daily"),
                    _in(d, "topic_fitness", "Passionate")))
_rule("R3-HLT-S04", "health", "soft",
      "건강 지표 조합 비현실(만성질환 없는데 하루 여러 번 복약, 전반적 건강 '최상'인데 심한 만성통증, 건강 '나쁨'인데 운동선수급 체력)",
      lambda d: _rows(d, "health_general_health"),
      lambda d: _or(
          _in(d, "health_chronic_condition", "None") & _in(d, "health_medication_use", "Multiple daily"),
          _in(d, "health_general_health", "Excellent") & _in(d, "health_pain_level", "Severe"),
          _in(d, "health_general_health", "Poor") & _in(d, "health_fitness_level", "Athlete")))


# -----------------------------------------------------------------------------
# lifestyle, habits, food
# -----------------------------------------------------------------------------
_rule("R3-LIF-H01", "lifestyle", "hard",
      "취미 몰입도 '없음'인데 특정 취미에 열성·적극 참여(노래방 제외)",
      lambda d: _in(d, "lstyle_hobby_intensity", "None"),
      lambda d: _any_col(d, HOBBY_COLS_FOR_INTENSITY, AVID))
_rule("R3-LIF-H02", "lifestyle", "hard",
      "활동 빈도 '전혀 안 함'인데 그 활동에 의존하는 습관(취침 전 독서·밀프렙·e스포츠 플레이)이 주 1회 이상",
      lambda d: _in(d, "lstyle_reading_freq", "Never") | _in(d, "lstyle_cooking_freq", "Never")
      | _in(d, "lstyle_gaming_freq", "Never"),
      lambda d: _or(
          _in(d, "lstyle_reading_freq", "Never") & _in(d, "habit_reading_before_bed", DAILY_WEEKLY),
          _in(d, "lstyle_cooking_freq", "Never") & _in(d, "habit_meal_prepping", DAILY_WEEKLY),
          _in(d, "lstyle_gaming_freq", "Never") & _in(d, "sport_esports", "Play")))
_rule("R3-LIF-H03", "lifestyle", "hard",
      "정기적으로 하는 활동의 역량이 '없음'(매일 요리, 제빵·텃밭·어반스케치 취미 적극, 매일·가끔 운전)",
      lambda d: _or(_in(d, "lstyle_cooking_freq", "Daily"), _in(d, "hob_bread_baking", AVID),
                    _in(d, "hob_vegetable_gardening", AVID), _in(d, "hob_urban_sketching", AVID),
                    _in(d, "demo_driver_status", ("Daily driver", "Occasional driver"))),
      lambda d: _or(
          _in(d, "lstyle_cooking_freq", "Daily") & _in(d, "skill_cooking", "None"),
          _in(d, "hob_bread_baking", AVID) & _in(d, "skill_baking", "None"),
          _in(d, "hob_vegetable_gardening", AVID) & _in(d, "skill_gardening", "None"),
          _in(d, "hob_urban_sketching", AVID) & _in(d, "skill_sketching", "None"),
          _in(d, "demo_driver_status", ("Daily driver", "Occasional driver")) & _in(d, "skill_driving", "None")))
_rule("R3-LIF-H04", "lifestyle", "hard",
      "암벽등반·볼더링 취미에 적극적인데 스포츠 클라이밍과의 관계가 '없음'",
      lambda d: _in(d, "hob_rock_climbing", AVID) | _in(d, "hob_bouldering", AVID),
      lambda d: _in(d, "sport_climbing", "None"))
_rule("R3-LIF-H05", "lifestyle", "hard",
      "비건·채식 식단인데 비건주의에 반대하거나 비건·채식 요리를 기피",
      lambda d: _in(d, "lstyle_diet_type", ("Vegan", "Vegetarian")),
      lambda d: (_in(d, "lstyle_diet_type", "Vegan")
                 & _or(_in(d, "att_veganism", "Opposed"), _in(d, "cuis_vegan", "Avoid"),
                       _in(d, "cuis_vegetarian", "Avoid")))
      | (_in(d, "lstyle_diet_type", "Vegetarian") & _in(d, "cuis_vegetarian", "Avoid")))
_rule("R3-LIF-H06", "lifestyle", "hard",
      "하위 요리를 매우 좋아하면서 상위 요리를 기피(스시↔일식, 쓰촨·광둥↔중식, 타파스↔스페인, 레바논↔중동)",
      lambda d: _in(d, "cuis_japanese", "Avoid") | _in(d, "cuis_chinese", "Avoid")
      | _in(d, "cuis_spanish", "Avoid") | _in(d, "cuis_middle_eastern", "Avoid"),
      lambda d: _or(
          _in(d, "cuis_sushi", "Love") & _in(d, "cuis_japanese", "Avoid"),
          (_in(d, "cuis_sichuan", "Love") | _in(d, "cuis_cantonese", "Love")) & _in(d, "cuis_chinese", "Avoid"),
          _in(d, "cuis_spanish_tapas", "Love") & _in(d, "cuis_spanish", "Avoid"),
          _in(d, "cuis_lebanese", "Love") & _in(d, "cuis_middle_eastern", "Avoid")))
_rule("R3-LIF-H07", "lifestyle", "hard",
      "주 정보원과 소비 행태 모순(뉴스가 주 정보원인데 뉴스 회피, SNS 위주인데 주 이용 SNS 없음)",
      lambda d: _in(d, "media_diet", ("News", "Social-media-heavy")),
      lambda d: (_in(d, "media_diet", "News") & _in(d, "lstyle_news_freq", "Avoids news"))
      | (_in(d, "media_diet", "Social-media-heavy") & _in(d, "lstyle_primary_social", "None")))
_rule("R3-LIF-H08", "lifestyle", "hard",
      "주 1회 이상 봉사활동을 하는데 봉사·공동체 활동 이력이 '없음'",
      lambda d: _in(d, "lstyle_volunteering", DAILY_WEEKLY),
      lambda d: _in(d, "lifex_service_history", "None"))
_rule("R3-LIF-S01", "lifestyle", "soft",
      "절제·식단과 기호 모순(비건·채식인데 해산물·바비큐를 매우 좋아함, 비건인데 식이 제한 없음, 금주인데 와인·맥주 열정, 카페인 0인데 커피 습관·열정)",
      lambda d: _rows(d, "lstyle_diet_type"),
      lambda d: _or(
          _in(d, "lstyle_diet_type", ("Vegan", "Vegetarian"))
          & (_in(d, "cuis_seafood", "Love") | _in(d, "cuis_american_bbq", "Love")),
          _in(d, "lstyle_diet_type", "Pescatarian") & _in(d, "cuis_american_bbq", "Love"),
          _in(d, "lstyle_diet_type", "Vegan") & _in(d, "health_dietary_restriction", "None"),
          _in(d, "lstyle_alcohol_use", "Never") & _or(
              _in(d, "topic_wine", "Passionate"), _in(d, "topic_craft_beer", "Passionate"),
              _in(d, "hob_winemaking", "Avid"), _in(d, "hob_home_brewing", "Avid"),
              _in(d, "fam_sommelier_knowledge", "Expert")),
          _in(d, "lstyle_caffeine", "None")
          & (_in(d, "lstyle_coffee_ritual", ("Home brew", "Café regular", "Office coffee"))
             | _in(d, "topic_coffee", "Passionate"))))
_rule("R3-LIF-S02", "lifestyle", "soft",
      "하루 화면 이용 2시간 미만인데 주당 스트리밍 16시간 이상(하루 평균 2.3시간 이상)",
      lambda d: _in(d, "lstyle_screen_time", "<2 hrs"),
      lambda d: _in(d, "lstyle_streaming_hours", "16+"))
_rule("R3-LIF-S03", "lifestyle", "soft",
      "독서를 전혀 안 하는데 책 장르 2개 이상을 매우 좋아하거나 읽기를 강하게 선호",
      lambda d: _in(d, "lstyle_reading_freq", "Never"),
      lambda d: (_count_col(d, BOOK_COLS, "Love") >= 2)
      | _in(d, "cog_reading_vs_watching", "Strongly prefers reading"))
_rule("R3-LIF-S04", "lifestyle", "soft",
      "국내 거주 한국인인데 한식을 기피(1% 미만)",
      lambda d: _rows(d, "cuis_korean"),
      lambda d: _in(d, "cuis_korean", "Avoid"))


# -----------------------------------------------------------------------------
# devices and platforms
# -----------------------------------------------------------------------------
_rule("R3-DEV-H01", "devices", "hard",
      "주 메신저가 iMessage인데 기기 생태계가 안드로이드·윈도우·리눅스(애플 기기 전용)",
      lambda d: _in(d, "lstyle_primary_messenger", "iMessage"),
      lambda d: _in(d, "lstyle_device_ecosystem", NON_APPLE))
_rule("R3-DEV-H02", "devices", "hard",
      "주 브라우저가 Safari인데 기기 생태계가 안드로이드·윈도우·리눅스(Safari는 애플 전용)",
      lambda d: _in(d, "lstyle_browser", "Safari"),
      lambda d: _in(d, "lstyle_device_ecosystem", NON_APPLE))
_rule("R3-DEV-H03", "devices", "hard",
      "최첨단 얼리어답터인데 기술 회피형이거나 신기술에 반대",
      lambda d: _in(d, "pref_early_vs_late", "Bleeding edge"),
      lambda d: _in(d, "tech_savviness", "Avoidant") | _in(d, "att_new_technology", "Opposed"))
_rule("R3-DEV-H04", "devices", "hard",
      "기술 회피형인데 직업 개발자",
      lambda d: _in(d, "tech_savviness", "Avoidant"),
      lambda d: _in(d, "dev_professional_status", "Professional developer"))
_rule("R3-DEV-S01", "devices", "soft",
      "기술 회피·거부형인데 리눅스 생태계 사용자이거나 개발 도구 파워유저",
      lambda d: _in(d, "tech_savviness", ("Avoidant", "Reluctant")),
      lambda d: _in(d, "lstyle_device_ecosystem", "Linux") | _any_col(d, DEV_TOOLS, "Power user"))
_rule("R3-DEV-S02", "devices", "soft",
      "55세 이상(1965년 이전 출생)인데 '디지털 네이티브'(디지털 환경에서 성장하지 않은 세대)",
      lambda d: _in(d, "tech_savviness", "Digital native"),
      lambda d: _in(d, "age_bracket", A55P))


# -----------------------------------------------------------------------------
# tech: developer and AI-tool attributes
# -----------------------------------------------------------------------------
_rule("R3-TEC-H01", "tech", "hard",
      "코딩 AI 사용 빈도와 과업별 AI 활용 모순(미사용·회피인데 현재 AI 활용 과업 있음, 매일·매주 사용인데 현재 AI 활용 과업 없음)",
      lambda d: _in(d, "coding_ai_usage_frequency", AI_NONUSE + DAILY_WEEKLY),
      lambda d: (_in(d, "coding_ai_usage_frequency", AI_NONUSE) & _any_col(d, AI_TASK_COLS, AI_CURRENT))
      | (_in(d, "coding_ai_usage_frequency", DAILY_WEEKLY) & _all_col(d, AI_TASK_COLS, AI_NOT_CURRENT)))
_rule("R3-TEC-H02", "tech", "hard",
      "코딩 AI를 써본 적 없음·회피인데 GitHub Copilot 또는 코딩 에이전트를 사용",
      lambda d: _in(d, "coding_ai_usage_frequency", AI_NONUSE),
      lambda d: (_in(d, "coding_ai_usage_frequency", "Never used")
                 & (_in(d, "tool_github_copilot", ("Tried it", "Occasional", "Regular", "Power user"))
                    | _in(d, "coding_agent_usage_frequency",
                          ("Daily", "Weekly", "Monthly or infrequent", "Experimental only"))))
      | (_in(d, "coding_ai_usage_frequency", "Actively avoids")
         & (_in(d, "tool_github_copilot", HEAVY_TOOL) | _in(d, "coding_agent_usage_frequency", DAILY_WEEKLY))))
_rule("R3-TEC-H03", "tech", "hard",
      "코딩 에이전트 사용 여부와 에이전트 경험 답변 모순(매일·매주 사용인데 '에이전트 안 씀', 미사용인데 작업 방식 변화 보고)",
      lambda d: _in(d, "coding_agent_usage_frequency", DAILY_WEEKLY + AGENT_NONUSE)
      | _in(d, "coding_agent_failure_tolerance", "Does not use agents"),
      lambda d: (_in(d, "coding_agent_usage_frequency", DAILY_WEEKLY)
                 & _in(d, "coding_agent_failure_tolerance", "Does not use agents"))
      | ((_in(d, "coding_agent_usage_frequency", AGENT_NONUSE)
          | _in(d, "coding_agent_failure_tolerance", "Does not use agents"))
         & _in(d, "coding_agent_workflow_impact", AGENT_IMPACT_USED)))
_rule("R3-TEC-H04", "tech", "hard",
      "도움 요청·학습 경로가 써본 적 없는 자원(AI 먼저 묻기·AI 도구로 학습인데 AI 미사용, Q&A 먼저 검색인데 Stack Overflow류 미사용)",
      lambda d: _in(d, "developer_help_seeking_preference", ("Ask AI first", "Search Stack Overflow / Q&A first"))
      | _in(d, "coding_ai_learning_source", "AI-enabled coding tools"),
      lambda d: _or(
          _in(d, "developer_help_seeking_preference", "Ask AI first") & _in(d, "coding_ai_usage_frequency", AI_NONUSE),
          _in(d, "coding_ai_learning_source", "AI-enabled coding tools")
          & _in(d, "coding_ai_usage_frequency", "Never used"),
          _in(d, "developer_help_seeking_preference", "Search Stack Overflow / Q&A first")
          & _in(d, "stackoverflow_usage_pattern", "Never used")))
_rule("R3-TEC-H05", "tech", "hard",
      "Stack Overflow류를 써본 적 없는데 읽기·투표·질문·답변 등 참여 방식이 있음",
      lambda d: _in(d, "stackoverflow_usage_pattern", "Never used"),
      lambda d: _in(d, "stackoverflow_participation_style", SO_PARTICIPATES))
_rule("R3-TEC-H06", "tech", "hard",
      "코딩 활동 '없음'인데 직업 개발자·매일 코딩 AI 사용·코드 푸시/PR 작성·릴리스 관리·잦은 코드리뷰",
      lambda d: _in(d, "open_source_activity_level", "No coding activity"),
      lambda d: _or(_in(d, "dev_professional_status", "Professional developer"),
                    _in(d, "coding_ai_usage_frequency", "Daily"),
                    _in(d, "github_contribution_mode", GITHUB_CODE),
                    _in(d, "pr_collaboration_style", PR_AUTHORING),
                    _in(d, "release_maintenance_orientation", RELEASE_INVOLVED),
                    _in(d, "code_review_participation_level", "Frequent reviewer")))
_rule("R3-TEC-H07", "tech", "hard",
      "오픈소스·GitHub 기여 주장과 비참여 모순(주 기여 맥락이 오픈소스인데 비공개·코딩 없음, GitHub 활동 유형이 있는데 GitHub·GitLab 미사용)",
      lambda d: _in(d, "dev_contribution_context", "Open-source / community work")
      | _in(d, "github_contribution_mode", GITHUB_ACTIVE),
      lambda d: (_in(d, "dev_contribution_context", "Open-source / community work")
                 & _in(d, "open_source_activity_level", ("Private-only developer", "No coding activity")))
      | (_in(d, "github_contribution_mode", GITHUB_ACTIVE) & _in(d, "tool_github", "Never used")
         & _in(d, "tool_gitlab", "Never used")))
_rule("R3-TEC-H08", "tech", "hard",
      "코딩 역량과 언어·리뷰 숙련 모순(코딩 '없음'인데 언어 숙련·코드리뷰 고급, 코딩 고급인데 모든 언어 '없음', 잦은 리뷰어인데 리뷰 역량 없음)",
      lambda d: _in(d, "skill_coding", ("None",) + ADV)
      | _in(d, "code_review_participation_level", "Frequent reviewer"),
      lambda d: _or(
          _in(d, "skill_coding", "None") & (_any_col(d, PROG_COLS, PROF) | _in(d, "skill_code_review", ADV)),
          _in(d, "skill_coding", ADV) & _all_col(d, PROG_COLS, "None"),
          _in(d, "code_review_participation_level", "Frequent reviewer") & _in(d, "skill_code_review", "None")))
_rule("R3-TEC-H09", "tech", "hard",
      "직업 개발자인데 코딩 역량 없음·모든 프로그래밍 언어 없음·은퇴 상태",
      lambda d: _in(d, "dev_professional_status", "Professional developer"),
      lambda d: _or(_in(d, "skill_coding", "None"), _all_col(d, PROG_COLS, "None"),
                    _in(d, "demo_employment_status", "Retired")))


def _lang_tool_pairs(d):
    pairs = (("prog_python", "tool_python"), ("prog_r", "tool_r"), ("prog_sql", "tool_sql"),
             ("prog_matlab", "tool_matlab"))
    sc, hit = _false(d), _false(d)
    for p, t in pairs:
        sc = sc | _in(d, p, PROF) | _in(d, t, HEAVY_TOOL)
        hit = hit | (_in(d, p, PROF) & _in(d, t, "Never used")) | (_in(d, t, HEAVY_TOOL) & _in(d, p, "None"))
    return sc, hit


_rule("R3-TEC-H10", "tech", "hard",
      "같은 언어의 숙련도와 도구 사용 경험 모순(Python·R·SQL·MATLAB: 숙련인데 사용 경험 없음, 정기 사용인데 숙련 '없음')",
      lambda d: _lang_tool_pairs(d)[0], lambda d: _lang_tool_pairs(d)[1])
_rule("R3-TEC-H11", "tech", "hard",
      "선행 언어 없이 상위 언어·도구 숙련(TypeScript인데 JavaScript 없음, Objective-C인데 C 없음, Jupyter 정기 사용인데 Python·R·Julia 없음)",
      lambda d: _or(_in(d, "prog_typescript", PROF), _in(d, "prog_objective_c", PROF),
                    _in(d, "tool_jupyter", HEAVY_TOOL)),
      lambda d: _or(
          _in(d, "prog_typescript", PROF) & _in(d, "prog_javascript", "None"),
          _in(d, "prog_objective_c", PROF) & _in(d, "prog_c", "None"),
          _in(d, "tool_jupyter", HEAVY_TOOL) & _all_col(d, ("prog_python", "prog_r", "prog_julia"), "None")))
_rule("R3-TEC-H12", "tech", "hard",
      "코딩 관행 답변끼리 모순(버그 수정 시 항상 회귀 테스트↔테스트 없음, TDD↔버그 테스트 없음, 주석 없음↔항상 TLDR, 리팩터링 안 함↔광범위 리팩터링)",
      lambda d: _or(_in(d, "bug_fix_test_behavior", ("Always adds regression tests", "No tests")),
                    _in(d, "code_comment_style", "No comments"),
                    _in(d, "code_refactoring_frequency", "Never refactors")),
      lambda d: _or(
          _in(d, "bug_fix_test_behavior", "Always adds regression tests") & _in(d, "code_testing_approach", "No tests"),
          _in(d, "bug_fix_test_behavior", "No tests") & _in(d, "code_testing_approach", "Test-driven development (TDD)"),
          _in(d, "code_comment_style", "No comments")
          & _in(d, "code_summary_documentation", ("Always includes file-level TLDR", "Always includes function-level TLDR")),
          _in(d, "code_refactoring_frequency", "Never refactors")
          & _in(d, "bug_fix_scope_preference", "Broad refactor while fixing")))
_rule("R3-TEC-H13", "tech", "hard",
      "코딩 에이전트 자율성 선호와 통제 기대가 정반대(완전 자율 실행↔읽기 전용·매 편집 승인, 자율성 없음·제안만↔광범위 자율 실행)",
      lambda d: _in(d, "coding_agent_autonomy_preference",
                    ("Autonomous execution", "No agent autonomy", "Suggest-only assistant")),
      lambda d: _or(
          _in(d, "coding_agent_autonomy_preference", "Autonomous execution")
          & _in(d, "coding_agent_control_expectation",
                ("Read-only assistance preferred", "Requires explicit approval before edits")),
          _in(d, "coding_agent_autonomy_preference", ("No agent autonomy", "Suggest-only assistant"))
          & _in(d, "coding_agent_control_expectation", "Agent may run broad tasks autonomously"),
          _in(d, "coding_agent_autonomy_preference", "No agent autonomy")
          & _in(d, "coding_agent_control_expectation", "Agent may edit within scoped files")))
_rule("R3-TEC-H14", "tech", "hard",
      "에이전트 도입의 주 장벽이 보안·개인정보인데 도구 평가에서 보안·개인정보는 '전혀 걱정 안 함'",
      lambda d: _in(d, "coding_agent_adoption_barrier", "Security / privacy / compliance"),
      lambda d: _in(d, "coding_tool_security_privacy_blocker", "Not a concern"))
_rule("R3-TEC-S01", "tech", "soft",
      "AI 태도·감정과 AI 도구 사용·신뢰의 극단적 괴리(AI 반대인데 코딩 AI 매우 호의·챗봇 파워유저, AI 열광인데 코딩 AI 매우 비호의·회피, 매우 비호의인데 강한 신뢰·AI 기능 필수, 미사용인데 복잡 작업 위임 편안)",
      lambda d: _rows(d, "att_ai"),
      lambda d: _or(
          _in(d, "att_ai", "Opposed") & _or(_in(d, "coding_ai_sentiment", "Very favorable"),
                                            _in(d, "tool_chatgpt", "Power user"),
                                            _in(d, "tool_claude", "Power user")),
          _in(d, "att_ai", "Enthusiast") & (_in(d, "coding_ai_sentiment", "Very unfavorable")
                                            | _in(d, "coding_ai_usage_frequency", "Actively avoids")),
          _in(d, "coding_ai_sentiment", "Very unfavorable")
          & (_in(d, "coding_ai_output_trust", "Strongly trusts with light review")
             | _in(d, "coding_tool_ai_capability_importance", "Critical")),
          _in(d, "coding_ai_usage_frequency", "Actively avoids") & _in(d, "coding_tool_ai_capability_importance", "Critical"),
          _in(d, "coding_ai_usage_frequency", "Never used")
          & _in(d, "coding_ai_complexity_comfort", "Comfortable delegating complex tasks")))
_rule("R3-TEC-S02", "tech", "soft",
      "오픈소스 메인테이너인데 GitHub 거의 안 씀·릴리스 무관·오픈소스 반대",
      lambda d: _in(d, "open_source_activity_level", "Maintainer / core contributor"),
      lambda d: _or(_in(d, "github_contribution_mode", "Rare GitHub user"),
                    _in(d, "release_maintenance_orientation", "Not involved in releases"),
                    _in(d, "att_open_source", "Opposed")))
_rule("R3-TEC-S03", "tech", "soft",
      "코딩 입문자(학습 중·초급)인데 특정 언어 전문가 수준",
      lambda d: _in(d, "dev_professional_status", "Learning to code") | _in(d, "skill_coding", "Beginner"),
      lambda d: _any_col(d, PROG_COLS, "Expert"))
_rule("R3-TEC-S04", "tech", "soft",
      "도구 의존 관계 비현실(git 미사용인데 GitHub·GitLab 정기 사용, Kubernetes 파워유저인데 Docker 미사용, Bash 전문가인데 리눅스 CLI 미사용, 코딩 역량 없는 Copilot 파워유저, 매일 코딩 AI·에이전트인데 ChatGPT·Claude·Copilot 모두 미사용)",
      lambda d: _rows(d, "tool_git"),
      lambda d: _or(
          _in(d, "tool_git", "Never used") & (_in(d, "tool_github", HEAVY_TOOL) | _in(d, "tool_gitlab", HEAVY_TOOL)),
          _in(d, "tool_github_copilot", "Power user") & _in(d, "skill_coding", "None"),
          _in(d, "tool_kubernetes", "Power user") & _in(d, "tool_docker", "Never used"),
          _in(d, "prog_bash", "Expert") & _in(d, "tool_linux_cli", "Never used"),
          (_in(d, "coding_ai_usage_frequency", "Daily") | _in(d, "coding_agent_usage_frequency", "Daily"))
          & _all_col(d, ("tool_chatgpt", "tool_claude", "tool_github_copilot"), "Never used")))


# -----------------------------------------------------------------------------
# expertise: familiarity, skills, tools, industry
# -----------------------------------------------------------------------------
def _gap(d, hi_col, hi_scale, lo_cols, lo_scale):
    """rank(hi) - max(rank(lo...)) on scales of equal length."""
    lo = None
    for c in lo_cols:
        r = _rank(d, c, lo_scale)
        lo = r if lo is None else pd.Series(np.fmax(lo.to_numpy(), r.to_numpy()), index=d.index)
    return _rank(d, hi_col, hi_scale) - lo


def _subfield(d, pairs):
    sc, hit = _false(d), _false(d)
    for sub, parent in pairs:
        sc = sc | (_rank(d, sub, FAMS) >= 3)
        hit = hit | (_gap(d, sub, FAMS, (parent,), FAMS) >= 3)
    return sc, hit


def _skill_fam(d):
    sc, hit = _false(d), _false(d)
    for skill, fams in SKILL_FAM_PAIRS:
        sc = sc | _in(d, skill, ADV)
        hit = hit | (_gap(d, skill, SKL, fams, FAMS) >= 3)
    return sc, hit


def _h_specialty_fam(d):
    out = _false(d)
    for spec, col in SPECIALTY_FAM.items():
        out = out | (_in(d, "subject_specialty", spec) & _in(d, col, "None"))
    return out


def _h_specialty_domain(d):
    out = _false(d)
    for spec, ok in SPECIALTY_DOMAINS.items():
        bad = tuple(x for x in DOMAINS if x not in ok)
        out = out | (_in(d, "subject_specialty", spec) & _in(d, "domain", bad))
    return out


def _h_ind_fam(d):
    out = _false(d)
    for ind, fam in IND_FAM_PAIRS:
        out = out | (_in(d, ind, "Veteran") & _in(d, fam, "None"))
    return out


_rule("R3-EXP-H01", "expertise", "hard",
      "하위 분야 지식이 상위 분야보다 3단계 이상 높음(딥러닝>머신러닝, 천체물리>천문학, 감사>회계, 보험계리>통계)",
      lambda d: _subfield(d, SUBFIELD_PAIRS)[0], lambda d: _subfield(d, SUBFIELD_PAIRS)[1])
_rule("R3-EXP-H02", "expertise", "hard",
      "실무 역량이 같은 주제 지식보다 3단계 이상 높음(사진·프로젝트관리·통계모델링·오디오제작·재무모델링)",
      lambda d: _skill_fam(d)[0], lambda d: _skill_fam(d)[1])
_rule("R3-EXP-H03", "expertise", "hard",
      "세부 전공(subject_specialty) 분야에 대한 지식이 '없음'(분야 해당 없음 제외)",
      lambda d: _in(d, "domain", DOMAINS), _h_specialty_fam)
_rule("R3-EXP-H04", "expertise", "hard",
      "해당 업계 베테랑인데 그 업계 핵심 지식이 '없음'(회계·부동산·보험·물류·광업·섬유)",
      lambda d: _any_col(d, tuple(p[0] for p in IND_FAM_PAIRS), "Veteran"), _h_ind_fam)
_rule("R3-EXP-H05", "expertise", "hard",
      "수리 감각 '없음'인데 수학·암산·통계모델링·재무모델링 고급 이상",
      lambda d: _in(d, "cog_numeracy_comfort", "None"),
      lambda d: _any_col(d, QUANT_SKILLS, ADV))
_rule("R3-EXP-H06", "expertise", "hard",
      "분석·영상·3D 도구 파워유저인데 해당 역량이 '없음'(BI→시각화, Excel·Sheets→스프레드시트 모델링, Premiere→영상편집, Blender→3D 모델링)",
      lambda d: _or(_any_col(d, ("tool_tableau", "tool_power_bi", "tool_looker", "tool_excel",
                                 "tool_google_sheets", "tool_premiere_pro", "tool_blender"), "Power user")),
      lambda d: _or(
          _any_col(d, ("tool_tableau", "tool_power_bi", "tool_looker"), "Power user")
          & _in(d, "skill_data_visualization", "None"),
          _any_col(d, ("tool_excel", "tool_google_sheets"), "Power user") & _in(d, "skill_spreadsheet_modeling", "None"),
          _in(d, "tool_premiere_pro", "Power user") & _in(d, "skill_video_editing", "None"),
          _in(d, "tool_blender", "Power user") & _in(d, "fam_3d_modeling", "None")))
_rule("R3-EXP-H07", "expertise", "hard",
      "전문 소프트웨어 파워유저인데 그 분야 지식이 '없음'(클라우드→클라우드 인프라, SPSS·Stata·SAS→통계, QuickBooks·Xero→회계)",
      lambda d: _any_col(d, ("tool_aws", "tool_azure", "tool_google_cloud", "tool_spss", "tool_stata",
                             "tool_sas", "tool_quickbooks", "tool_xero"), "Power user"),
      lambda d: _or(
          _any_col(d, ("tool_aws", "tool_azure", "tool_google_cloud"), "Power user")
          & _in(d, "fam_cloud_infrastructure", "None"),
          _any_col(d, ("tool_spss", "tool_stata", "tool_sas"), "Power user") & _in(d, "fam_statistics", "None"),
          _any_col(d, ("tool_quickbooks", "tool_xero"), "Power user") & _in(d, "fam_accounting", "None")))
_rule("R3-EXP-S01", "expertise", "soft",
      "세부 전공과 주 분야가 양립하기 어려움(예: 심장학 전공인데 농업·제조 분야, 헌법학인데 숙박업)",
      lambda d: _in(d, "subject_specialty", tuple(SPECIALTY_DOMAINS)) & _in(d, "domain", DOMAINS),
      _h_specialty_domain)
_rule("R3-EXP-S02", "expertise", "soft",
      "밀접한 분야 간 지식 격차 3단계 이상(컴퓨터비전·NLP↔머신러닝, 데이터과학·역학↔통계, 린↔운영관리, UX리서치↔HCI) 또는 Unity·Unreal 파워유저인데 게임개발 지식 없음",
      lambda d: _subfield(d, RELATED_FIELD_PAIRS)[0]
      | _any_col(d, ("tool_unity", "tool_unreal_engine"), "Power user"),
      lambda d: _subfield(d, RELATED_FIELD_PAIRS)[1]
      | (_any_col(d, ("tool_unity", "tool_unreal_engine"), "Power user") & _in(d, "fam_game_development", "None")))


# -----------------------------------------------------------------------------
# personality instruments (only opposite extremes for instruments of the same construct)
# -----------------------------------------------------------------------------
def _facet_rule(key):
    pairs = FACET_PAIRS[key]
    return (lambda d: _ext_pairs(d, pairs)), (lambda d: _opp_pairs(d, pairs))


def _domain_vs_facets(d):
    sc, hit = _false(d), _false(d)
    for dom, facets in BFI2_DOMAINS.items():
        r = _rank(d, dom, LV)
        total = sum(_rank(d, f, LV) for f in facets)
        sc = sc | (r == 4) | (r == 0)
        hit = hit | ((r == 4) & (total <= 5)) | ((r == 0) & (total >= 7))
    return sc, hit


def _h_dominant(d):
    out = _false(d)
    for trait, dom in DOMINANT_TO_DOMAIN.items():
        out = out | (_in(d, "dominant_trait", trait) & _in(d, dom, ("Low", "Very low")))
    return out | (_in(d, "dominant_trait", "High extraversion")
                  & _in(d, "lstyle_social_battery", ("Strong introvert", "Introvert")))


def _h_extraversion_markers(d):
    e = "bfi2_domain_extraversion"
    return _or(
        _in(d, "mbti_type", MBTI_E) & (_in(d, e, "Very low") | _in(d, "lstyle_social_battery", "Strong introvert")),
        _in(d, "mbti_type", MBTI_I) & (_in(d, e, "Very high") | _in(d, "lstyle_social_battery", "Strong extrovert")),
        _in(d, "lstyle_social_battery", "Strong introvert") & _in(d, e, "Very high"),
        _in(d, "lstyle_social_battery", "Strong extrovert") & _in(d, e, "Very low"))


def _h_neuro_anxiety(d):
    return _in(d, "neurotype", "Anxiety-prone") & (_in(d, "bfi2_facet_anxiety", "Very low")
                                                   | _in(d, "big5_anxiety", "Very low"))


_rule("R3-PER-H01", "personality", "hard",
      "BFI-2와 IPIP-NEO의 같은 부정적 정서성 하위요인(불안·우울·변덕/분노)이 정반대 극단, 또는 불안 성향 신경유형인데 불안 요인 매우 낮음",
      lambda d: _ext_pairs(d, FACET_PAIRS["N"]) | _in(d, "neurotype", "Anxiety-prone"),
      lambda d: _opp_pairs(d, FACET_PAIRS["N"]) | _h_neuro_anxiety(d))
for _k, _num, _name in (("E", 2, "외향성(사교성·주장성·활력)"), ("A", 3, "친화성(연민·존중·신뢰)"),
                        ("C", 4, "성실성(조직성·생산성·책임감)"), ("O", 5, "개방성(지적호기심·심미성·상상력)")):
    _sc, _ht = _facet_rule(_k)
    _rule(f"R3-PER-H0{_num}", "personality", "hard",
          f"BFI-2와 IPIP-NEO의 같은 {_name} 하위요인이 정반대 극단(매우 높음↔매우 낮음)",
          _sc, _ht)

_rule("R3-PER-H06", "personality", "hard",
      "BFI-2 영역 점수가 세 하위요인 평균과 산술적으로 양립 불가(영역 매우 높음인데 하위요인 합 ≤5/12, 매우 낮음인데 ≥7/12)",
      lambda d: _domain_vs_facets(d)[0], lambda d: _domain_vs_facets(d)[1])
_rule("R3-PER-H07", "personality", "hard",
      "가장 두드러진 특성(dominant_trait)의 BFI-2 영역이 낮음·매우 낮음(외향 우세인데 내향 성향 포함)",
      lambda d: _in(d, "dominant_trait", tuple(DOMINANT_TO_DOMAIN)), _h_dominant)
_rule("R3-PER-H08", "personality", "hard",
      "외향성 지표 간 정반대(MBTI E/I·사회적 배터리·BFI-2 외향성 중 하나는 극단 외향, 다른 하나는 극단 내향)",
      lambda d: _in(d, "bfi2_domain_extraversion", ("Very low", "Very high"))
      | _in(d, "lstyle_social_battery", ("Strong introvert", "Strong extrovert")),
      _h_extraversion_markers)
_rule("R3-PER-H09", "personality", "hard",
      "MBTI 판단(J)인데 '매우 즉흥적', 인식(P)인데 '매우 계획적'(같은 구성개념의 반대 극단)",
      lambda d: _in(d, "pref_plan_vs_spontaneous", ("Highly planned", "Highly spontaneous")),
      lambda d: (_in(d, "mbti_type", MBTI_J) & _in(d, "pref_plan_vs_spontaneous", "Highly spontaneous"))
      | (_in(d, "mbti_type", MBTI_P) & _in(d, "pref_plan_vs_spontaneous", "Highly planned")))
_rule("R3-PER-H10", "personality", "hard",
      "VIA 강점과 같은 구성개념의 개방성·성실성 요인이 정반대(호기심·창의성·심미안·신중함·절제·야망·끈기)",
      lambda d: _ext_pairs(d, VIA_OC_PAIRS), lambda d: _opp_pairs(d, VIA_OC_PAIRS))
_rule("R3-PER-H11", "personality", "hard",
      "VIA 강점과 같은 구성개념의 친화성·외향성·신경성 요인이 정반대(겸손↔겸양, 친절↔이타, 정직↔도덕성, 열정↔활력, 자기조절↔무절제 역, 회복탄력성↔취약성 역)",
      lambda d: _ext_pairs(d, VIA_AEN_PAIRS), lambda d: _opp_pairs(d, VIA_AEN_PAIRS))
_rule("R3-PER-H12", "personality", "hard",
      "자기서술(호기심·열린 마음·낙관)과 같은 이름의 VIA 강점·BFI-2 지적호기심이 정반대 극단",
      lambda d: _ext_pairs(d, COG_VIA_PAIRS), lambda d: _opp_pairs(d, COG_VIA_PAIRS))
_rule("R3-PER-H13", "personality", "hard",
      "대인 스타일 지표가 정반대(자기주장·대인 주도성↔BFI-2 주장성, 공손함↔존중, 대인 온정↔IPIP 친근함)",
      lambda d: _ext_pairs(d, STYLE_PAIRS), lambda d: _opp_pairs(d, STYLE_PAIRS))
_rule("R3-PER-H14", "personality", "hard",
      "성실성 행동 지표가 정반대(미루기 매우 높음↔자기통제·생산성 매우 높음 등, 정돈 상태 극단↔조직성·질서 반대 극단)",
      lambda d: _or(_in(d, "cog_procrastination", ("Very high", "None")),
                    _in(d, "lstyle_tidiness", ("Chaotic", "Spotless"))),
      lambda d: _or(
          _in(d, "cog_procrastination", "Very high")
          & (_in(d, "big5_self_discipline", "Very high") | _in(d, "bfi2_facet_productiveness", "Very high")),
          _in(d, "cog_procrastination", "None")
          & (_in(d, "big5_self_discipline", "Very low") | _in(d, "bfi2_facet_productiveness", "Very low")),
          _in(d, "lstyle_tidiness", "Chaotic")
          & (_in(d, "bfi2_facet_organization", "Very high") | _in(d, "big5_orderliness", "Very high")),
          _in(d, "lstyle_tidiness", "Spotless")
          & (_in(d, "bfi2_facet_organization", "Very low") | _in(d, "big5_orderliness", "Very low"))))
_rule("R3-PER-H15", "personality", "hard",
      "모호함 인내가 매우 높은데 종결 욕구도 매우 높음(또는 둘 다 최저) — 역방향 동일 구성개념",
      lambda d: _ext(d, "cog_ambiguity_tolerance", COG, "need_for_closure", LV),
      lambda d: _opp(d, "cog_ambiguity_tolerance", COG, "need_for_closure", LV, True))
_rule("R3-PER-H16", "personality", "hard",
      "세부 지향성 매우 높음인데 '큰 그림만', 세부 지향성 없음인데 '세부 집착'",
      lambda d: _in(d, "cog_detail_orientation", ("Very high", "None")),
      lambda d: (_in(d, "cog_detail_orientation", "Very high") & _in(d, "cog_big_picture_vs_detail", "Big-picture only"))
      | (_in(d, "cog_detail_orientation", "None") & _in(d, "cog_big_picture_vs_detail", "Detail-obsessed")))
_rule("R3-PER-H17", "personality", "hard",
      "의사결정 방식 지표가 정반대(신중성 매우 높음↔즉석 결정, 매우 낮음↔고민만 함, 분석형↔순수 직관, 직관형↔순수 논리)",
      lambda d: _or(_in(d, "big5_cautiousness", ("Very high", "Very low")),
                    _in(d, "decision_style", ("Analytical", "Intuitive"))),
      lambda d: _or(
          _in(d, "big5_cautiousness", "Very high") & _in(d, "cog_decision_speed", "Snap decisions"),
          _in(d, "big5_cautiousness", "Very low") & _in(d, "cog_decision_speed", "Agonizes"),
          _in(d, "decision_style", "Analytical") & _in(d, "pref_logic_vs_intuition", "Pure intuition"),
          _in(d, "decision_style", "Intuitive") & _in(d, "pref_logic_vs_intuition", "Pure logic")))
_rule("R3-PER-H18", "personality", "hard",
      "군집성(gregariousness) 매우 높음인데 일대일만 선호, 매우 낮음인데 대규모 모임을 매우 좋아함",
      lambda d: _in(d, "big5_gregariousness", ("Very high", "Very low")),
      lambda d: (_in(d, "big5_gregariousness", "Very high") & _in(d, "pref_big_group_vs_one_on_one", "One-on-one only"))
      | (_in(d, "big5_gregariousness", "Very low") & _in(d, "pref_big_group_vs_one_on_one", "Loves big groups")))
_rule("R3-PER-H19", "personality", "hard",
      "새로움·변화 선호 지표가 정반대(항상 새로움↔루틴 갈망, 항상 익숙함↔다양성 갈망, 모험성 매우 낮음/높음과 반대 극단 선호)",
      lambda d: _or(_in(d, "pref_novelty_vs_familiarity", ("Always novel", "Always familiar")),
                    _in(d, "pref_routine_vs_variety", ("Craves routine", "Craves variety")),
                    _in(d, "big5_adventurousness", ("Very low", "Very high"))),
      lambda d: _or(
          _in(d, "pref_novelty_vs_familiarity", "Always novel") & _in(d, "pref_routine_vs_variety", "Craves routine"),
          _in(d, "pref_novelty_vs_familiarity", "Always familiar") & _in(d, "pref_routine_vs_variety", "Craves variety"),
          _in(d, "big5_adventurousness", "Very low")
          & (_in(d, "pref_novelty_vs_familiarity", "Always novel") | _in(d, "pref_routine_vs_variety", "Craves variety")),
          _in(d, "big5_adventurousness", "Very high")
          & (_in(d, "pref_novelty_vs_familiarity", "Always familiar") | _in(d, "pref_routine_vs_variety", "Craves routine"))))
_rule("R3-PER-H20", "personality", "hard",
      "IPIP 자유주의(진보 투표 성향) 매우 높음인데 정치 성향 우파, 매우 낮음인데 좌파",
      lambda d: _in(d, "big5_liberalism", ("Very high", "Very low")),
      lambda d: (_in(d, "big5_liberalism", "Very high") & _in(d, "political_lean", "Right"))
      | (_in(d, "big5_liberalism", "Very low") & _in(d, "political_lean", "Left")))
_rule("R3-PER-H21", "personality", "hard",
      "리더십·경쟁심 강점과 역량·선호가 정반대(리더십 대표강점인데 역량 없음, 리더십 없음인데 마스터·항상 주도, 경쟁심 없음↔매우 경쟁적, 대표강점↔매우 협력적)",
      lambda d: _in(d, "trait_leadership", ("Signature", "Absent"))
      | _in(d, "trait_competitiveness", ("Signature", "Absent")),
      lambda d: _or(
          _in(d, "trait_leadership", "Signature") & _in(d, "skill_leadership", "None"),
          _in(d, "trait_leadership", "Absent")
          & (_in(d, "skill_leadership", "Master") | _in(d, "pref_lead_vs_follow", "Always leads")),
          _in(d, "trait_competitiveness", "Absent") & _in(d, "pref_competition_vs_collab", "Highly competitive"),
          _in(d, "trait_competitiveness", "Signature") & _in(d, "pref_competition_vs_collab", "Highly collaborative")))
_rule("R3-PER-H22", "personality", "hard",
      "격식성 매우 높음인데 기본 말투가 구어체, 은어·속어 위주인데 기본 말투가 격식·표준어",
      lambda d: _in(d, "cog_formality", ("Very formal", "Slangy")),
      lambda d: (_in(d, "cog_formality", "Very formal") & _in(d, "register", "Colloquial"))
      | (_in(d, "cog_formality", "Slangy") & _in(d, "register", "Formal / standard")))
_rule("R3-PER-S01", "personality", "soft",
      "MBTI 선호와 Big Five 극단의 괴리(N↔개방성 매우 낮음, S↔매우 높음, J↔성실성 매우 낮음, P↔매우 높음; 상관 .5~.7)",
      lambda d: _in(d, "bfi2_domain_open_mindedness", ("Very low", "Very high"))
      | _in(d, "bfi2_domain_conscientiousness", ("Very low", "Very high")),
      lambda d: _or(
          _in(d, "mbti_type", MBTI_N) & _in(d, "bfi2_domain_open_mindedness", "Very low"),
          _in(d, "mbti_type", MBTI_S) & _in(d, "bfi2_domain_open_mindedness", "Very high"),
          _in(d, "mbti_type", MBTI_J) & _in(d, "bfi2_domain_conscientiousness", "Very low"),
          _in(d, "mbti_type", MBTI_P) & _in(d, "bfi2_domain_conscientiousness", "Very high")))
_rule("R3-PER-S02", "personality", "soft",
      "자극 추구와 레저 위험 행동의 반대 극단(자극 추구 매우 낮음인데 스카이다이빙·패러글라이딩 적극·레저 위험 감수 매우 높음, 또는 그 반대)",
      lambda d: _in(d, "big5_excitement_seeking", ("Very low", "Very high")),
      lambda d: (_in(d, "big5_excitement_seeking", "Very low")
                 & _or(_in(d, "hob_skydiving", AVID), _in(d, "hob_paragliding", AVID),
                       _in(d, "dospert_recreational_risk_tolerance", "Very high")))
      | (_in(d, "big5_excitement_seeking", "Very high") & _in(d, "dospert_recreational_risk_tolerance", "Very low")))
_rule("R3-PER-S03", "personality", "soft",
      "VIA 강점과 관련 자기서술의 반대 극단(공감 대표강점↔연민 매우 낮음·공감 표현/역량 없음, 팀워크↔강한 단독 선호, 학구열↔인지욕구 매우 낮음, 감사 없음↔매일 감사 실천)",
      lambda d: _rows(d, "trait_empathy"),
      lambda d: _or(
          _in(d, "trait_empathy", "Signature") & _or(_in(d, "bfi2_facet_compassion", "Very low"),
                                                     _in(d, "cog_empathy_expression", "None"),
                                                     _in(d, "skill_empathy", "None")),
          _in(d, "trait_empathy", "Absent") & (_in(d, "bfi2_facet_compassion", "Very high")
                                               | _in(d, "skill_empathy", "Master")),
          _in(d, "trait_teamwork", "Signature") & _in(d, "pref_team_vs_solo", "Strongly solo"),
          _in(d, "trait_teamwork", "Absent") & _in(d, "pref_team_vs_solo", "Strongly team"),
          _in(d, "trait_love_of_learning", "Signature") & _in(d, "need_for_cognition", "Very low"),
          _in(d, "trait_gratitude", "Absent") & _in(d, "habit_gratitude_practice", "Daily")))
_rule("R3-PER-S04", "personality", "soft",
      "사회적 배터리 극단과 모임 규모 선호의 반대 극단(강한 내향인데 대규모 모임 매우 좋아함, 강한 외향인데 일대일만)",
      lambda d: _in(d, "lstyle_social_battery", ("Strong introvert", "Strong extrovert")),
      lambda d: (_in(d, "lstyle_social_battery", "Strong introvert") & _in(d, "pref_big_group_vs_one_on_one", "Loves big groups"))
      | (_in(d, "lstyle_social_battery", "Strong extrovert") & _in(d, "pref_big_group_vs_one_on_one", "One-on-one only")))


# -----------------------------------------------------------------------------
# values and value priorities
# -----------------------------------------------------------------------------
def _h_priority(d, mapping, low):
    out = _false(d)
    for top, cols in mapping.items():
        out = out | (_in(d, "values_priority", top) & _all_col(d, cols, low))
    return out


_rule("R3-VAL-H01", "values", "hard",
      "Schwartz 가치(자기주도·자극·쾌락·성취·권력)와 같은 가치 항목이 정반대(매우 높음↔무관, 매우 낮음↔핵심 가치)",
      lambda d: _ext_pairs(d, SCHWARTZ_PAIRS_1), lambda d: _opp_pairs(d, SCHWARTZ_PAIRS_1))
_rule("R3-VAL-H02", "values", "hard",
      "Schwartz 가치(안전·전통·박애·보편주의)와 같은 가치 항목(안정·전통·타인 돕기·평등)이 정반대",
      lambda d: _ext_pairs(d, SCHWARTZ_PAIRS_2), lambda d: _opp_pairs(d, SCHWARTZ_PAIRS_2))
_rule("R3-VAL-H03", "values", "hard",
      "최우선 가치(values_priority)에 해당하는 가치 항목이 모두 '사소함·무관'",
      lambda d: _rows(d, "values_priority"),
      lambda d: _h_priority(d, PRIORITY_TO_VAL, ("Minor", "Irrelevant")))
_rule("R3-VAL-H04", "values", "hard",
      "최우선 가치에 해당하는 Schwartz 가치가 모두 낮음·매우 낮음",
      lambda d: _rows(d, "values_priority"),
      lambda d: _h_priority(d, PRIORITY_TO_SCHWARTZ, ("Low", "Very low")))
_rule("R3-VAL-H05", "values", "hard",
      "권위에 대한 태도와 도덕기반 권위 민감성이 정반대(열광↔매우 낮음, 반대↔매우 높음)",
      lambda d: _ext(d, "att_authority", ATTS, "mft_authority_subversion", LV),
      lambda d: _opp(d, "att_authority", ATTS, "mft_authority_subversion", LV))
_rule("R3-VAL-H06", "values", "hard",
      "데이터 프라이버시 태도와 프라이버시 가치가 정반대(반대↔핵심 가치, 열광↔무관)",
      lambda d: _ext(d, "att_data_privacy", ATTS, "val_privacy", VALS),
      lambda d: _opp(d, "att_data_privacy", ATTS, "val_privacy", VALS))
_rule("R3-VAL-S01", "values", "soft",
      "도덕기반·VIA 강점과 같은 가치 항목의 반대 극단(충성·공정·자유·배려, 정직·충성) 또는 유교적 삶인데 전통 가치 최저",
      lambda d: _rows(d, "val_loyalty"),
      lambda d: _opp_pairs(d, MFT_VALUE_PAIRS)
      | (_in(d, "kr_confucian_way_of_life", "Yes")
         & (_in(d, "schwartz_value_tradition", "Very low") | _in(d, "val_tradition", "Irrelevant"))))
_rule("R3-VAL-S02", "values", "soft",
      "동성애자·양성애자 등인데 동성 관계를 '항상 잘못'으로 평가(내면화된 혐오는 드묾)",
      lambda d: _in(d, "demo_sexual_orientation", ("Gay / lesbian", "Bisexual", "Pansexual", "Queer")),
      lambda d: _in(d, "kr_attitude_same_sex_relations", "Always wrong"))


# -----------------------------------------------------------------------------
# interests vs activities
# -----------------------------------------------------------------------------
def _pairs_hit(d, pairs, a_vals, b_vals):
    out = _false(d)
    for a, b in pairs:
        out = out | (_in(d, a, a_vals) & _in(d, b, b_vals))
    return out


_rule("R3-INT-H01", "interests", "hard",
      "스포츠에 대한 관심이 '혐오'인데 특정 스포츠를 팬으로 챙겨 봄",
      lambda d: _in(d, "topic_sports", "Averse"),
      lambda d: _any_col(d, SPECTATOR_SPORTS, "Follow"))
_rule("R3-INT-H02", "interests", "hard",
      "취미에 열성·적극 참여하는데 같은 주제 관심이 '혐오'(탐조·족보·뜨개·목공·서예·제빵·텃밭·별보기·수집·댄스·와인·홈브루)",
      lambda d: _any_col(d, tuple(p[0] for p in HOBBY_TOPIC_PAIRS), AVID),
      lambda d: _pairs_hit(d, HOBBY_TOPIC_PAIRS, AVID, "Averse"))
_rule("R3-INT-H03", "interests", "hard",
      "영화(또는 음악) 관심이 '혐오'인데 특정 영화(음악) 장르를 매우 좋아함",
      lambda d: _in(d, "topic_film", "Averse") | _in(d, "topic_music", "Averse"),
      lambda d: (_in(d, "topic_film", "Averse") & _any_col(d, FILM_COLS, "Love"))
      | (_in(d, "topic_music", "Averse") & _any_col(d, MUSIC_COLS, "Love")))
_rule("R3-INT-H04", "interests", "hard",
      "같은 주제의 책 장르를 매우 좋아하는데 주제 관심이 '혐오'(문학·시·역사·철학·실화범죄·과학·여행·만화·자기계발)",
      lambda d: _any_col(d, tuple(p[1] for p in BOOK_TOPIC_PAIRS), "Averse"),
      lambda d: _pairs_hit(d, BOOK_TOPIC_PAIRS, "Love", "Averse"))
_rule("R3-INT-H05", "interests", "hard",
      "같은 과목의 학문적 관심과 주제 관심이 정반대(열정↔혐오: 문학·철학·미술·음악·천문·역사·연극·환경)",
      lambda d: _ext_pairs(d, ACAD_TOPIC_PAIRS), lambda d: _opp_pairs(d, ACAD_TOPIC_PAIRS))
_rule("R3-INT-H06", "interests", "hard",
      "매일 명상을 하는데 명상에 대한 관심이 '혐오'",
      lambda d: _in(d, "habit_meditation", "Daily"),
      lambda d: _in(d, "topic_meditation", "Averse"))
_rule("R3-INT-S01", "interests", "soft",
      "달리기·사이클·요가를 직접 하면서 그 주제를 혐오(건강 목적이라도 드묾)",
      lambda d: _any_col(d, ("sport_running", "sport_cycling", "sport_yoga"), "Play"),
      lambda d: _pairs_hit(d, (("sport_running", "topic_running"), ("sport_cycling", "topic_cycling"),
                                ("sport_yoga", "topic_yoga")), "Play", "Averse"))
_rule("R3-INT-S02", "interests", "soft",
      "이용 빈도·취향과 관심의 극단적 괴리(게임 안 함↔게임 열정, 매일 게임↔혐오, 팟캐스트 안 들음↔열정, 음악 거의 안 들음↔음악·공연 열정, 영화·드라마 혐오인데 스트리밍 16시간+, 주 1회+ 봉사↔봉사 혐오, 아니메 열정↔애니메이션 영화 싫어함)",
      lambda d: _rows(d, "lstyle_gaming_freq"),
      lambda d: _or(
          _in(d, "topic_anime", "Passionate") & _in(d, "filmg_animation", "Dislike"),
          _in(d, "lstyle_gaming_freq", "Never") & _in(d, "topic_gaming", "Passionate"),
          _in(d, "lstyle_gaming_freq", "Daily") & _in(d, "topic_gaming", "Averse"),
          _in(d, "lstyle_podcast_listening", "Never") & _in(d, "topic_podcasts", "Passionate"),
          _in(d, "lstyle_music_listening", "Rarely")
          & (_in(d, "topic_music", "Passionate") | _in(d, "topic_live_concerts", "Passionate")),
          _in(d, "topic_tv_series", "Averse") & _in(d, "topic_film", "Averse") & _in(d, "lstyle_streaming_hours", "16+"),
          _in(d, "lstyle_volunteering", DAILY_WEEKLY) & _in(d, "topic_volunteering", "Averse")))


# -----------------------------------------------------------------------------
# attitudes vs behaviours
# -----------------------------------------------------------------------------
_rule("R3-ATT-H01", "attitudes", "hard",
      "위험 감수에 대한 태도와 위험 성향이 정반대(반대↔위험 추구, 열광↔위험 회피)",
      lambda d: _in(d, "att_risk_taking", ("Opposed", "Enthusiast")),
      lambda d: (_in(d, "att_risk_taking", "Opposed") & _in(d, "risk_tolerance", "Risk-seeking"))
      | (_in(d, "att_risk_taking", "Enthusiast") & _in(d, "risk_tolerance", "Risk-averse")))
_rule("R3-ATT-H02", "attitudes", "hard",
      "재택·사무실 근무 태도와 근무지 선호가 정반대(재택 반대인데 강한 재택 선호 등)",
      lambda d: _in(d, "pref_work_location", ("Strongly office", "Strongly remote")),
      lambda d: _or(
          _in(d, "att_remote_work", "Opposed") & _in(d, "pref_work_location", "Strongly remote"),
          _in(d, "att_remote_work", "Enthusiast") & _in(d, "pref_work_location", "Strongly office"),
          _in(d, "att_working_from_office", "Opposed") & _in(d, "pref_work_location", "Strongly office"),
          _in(d, "att_working_from_office", "Enthusiast") & _in(d, "pref_work_location", "Strongly remote")))
_rule("R3-ATT-H03", "attitudes", "hard",
      "급격한 변화에 열광하면서 안정을 갈망, 또는 반대하면서 변화를 갈망",
      lambda d: _in(d, "att_rapid_change", ("Enthusiast", "Opposed")),
      lambda d: (_in(d, "att_rapid_change", "Enthusiast") & _in(d, "pref_stability_vs_change", "Craves stability"))
      | (_in(d, "att_rapid_change", "Opposed") & _in(d, "pref_stability_vs_change", "Craves change")))
_rule("R3-ATT-H04", "attitudes", "hard",
      "브랜드 충성에 반대하는데 쇼핑 스타일이 '브랜드 충성'",
      lambda d: _in(d, "lstyle_shopping_style", "Brand loyal"),
      lambda d: _in(d, "att_brand_loyalty", "Opposed"))
_rule("R3-ATT-H05", "attitudes", "hard",
      "오픈소스에 반대하는데 도구 선택에서 오픈소스 여부가 '결정적'",
      lambda d: _in(d, "att_open_source", "Opposed"),
      lambda d: _in(d, "coding_tool_open_source_importance", "Critical"))
_rule("R3-ATT-H06", "attitudes", "hard",
      "같은 대상에 대해 태도는 '열광'인데 관심은 '혐오'(암호화폐·우주탐사·소셜미디어·신기술·제도 종교)",
      lambda d: _any_col(d, tuple(p[0] for p in ATT_TOPIC_PAIRS), "Enthusiast"),
      lambda d: _pairs_hit(d, ATT_TOPIC_PAIRS, "Enthusiast", "Averse"))
_rule("R3-ATT-H07", "attitudes", "hard",
      "정치 활동가인데 정치 또는 사회운동에 대한 관심이 '혐오'",
      lambda d: _in(d, "demo_political_engagement", "Activist"),
      lambda d: _in(d, "topic_politics", "Averse") | _in(d, "topic_activism", "Averse"))
_rule("R3-ATT-H08", "attitudes", "hard",
      "스카이다이빙·패러글라이딩에 적극적인데 DOSPERT 레저 위험 감수(바로 그런 활동 의향)가 매우 낮음",
      lambda d: _in(d, "hob_skydiving", AVID) | _in(d, "hob_paragliding", AVID),
      lambda d: _in(d, "dospert_recreational_risk_tolerance", "Very low"))
_rule("R3-ATT-S01", "attitudes", "soft",
      "반대하는 대상에 생활이 의존(대중교통 반대인데 대중교통 통근, 구독 반대인데 10개+, 긱경제 반대인데 긱 노동, 주택소유·부채 반대인데 자가·대출, 암호화폐 반대인데 투자·결제, SNS 반대인데 SNS 위주, 미니멀리즘 반대 미니멀 쇼핑, 소비주의 반대 과소비)",
      lambda d: _rows(d, "att_public_transit"),
      lambda d: _or(
          _in(d, "att_public_transit", "Opposed") & _in(d, "lstyle_commute_mode", "Public transit"),
          _in(d, "att_subscription_services", "Opposed") & _in(d, "lstyle_subscription_count", "10+"),
          _in(d, "att_the_gig_economy", "Opposed") & _in(d, "demo_employment_status", "Gig / freelance"),
          _in(d, "att_homeownership", "Opposed") & _in(d, "demo_housing_status", ("Own outright", "Mortgage")),
          _in(d, "att_taking_on_debt", "Opposed") & _in(d, "demo_housing_status", "Mortgage"),
          _in(d, "att_cryptocurrency", "Opposed")
          & (_in(d, "lstyle_investment_style", "Crypto-heavy") | _in(d, "lstyle_payment_pref", "Crypto")),
          _in(d, "att_social_media", "Opposed") & _in(d, "media_diet", "Social-media-heavy"),
          _in(d, "att_minimalism", "Opposed") & _in(d, "lstyle_shopping_style", "Minimalist"),
          _in(d, "att_consumerism", "Opposed") & _in(d, "lstyle_frugality", "Splurger")))
_rule("R3-ATT-S02", "attitudes", "soft",
      "기후 행동에 반대하면서 지속가능성을 핵심 가치로 두거나 지속가능성·환경에 열정적",
      lambda d: _in(d, "att_climate_action", "Opposed"),
      lambda d: _or(_in(d, "val_sustainability", "Core value"), _in(d, "topic_sustainability", "Passionate"),
                    _in(d, "topic_environment", "Passionate")))
_rule("R3-ATT-S03", "attitudes", "soft",
      "금융 위험 감수 매우 낮음인데 암호화폐 위주·적극 매매 투자",
      lambda d: _in(d, "dospert_financial_risk_tolerance", "Very low"),
      lambda d: _in(d, "lstyle_investment_style", ("Crypto-heavy", "Active trader")))
_rule("R3-ATT-S04", "attitudes", "soft",
      "정치 참여 조합 비현실(탈정치 성향인데 활동가·시위 참여, 활동가인데 봉사·사회운동 이력 없음)",
      lambda d: _in(d, "demo_political_engagement", ("Activist", "Protest participant")),
      lambda d: (_in(d, "political_lean", "Apolitical"))
      | (_in(d, "demo_political_engagement", "Activist") & _in(d, "lifex_service_history", "None")))
_rule("R3-ATT-S05", "attitudes", "soft",
      "가장 가깝게 느끼는 나라가 북한인데 북한을 '적'으로 인식",
      lambda d: _in(d, "kr_country_felt_closest", "North Korea"),
      lambda d: _in(d, "kr_view_of_north_korea", "Enemy"))


# -----------------------------------------------------------------------------
# wellbeing (Korean wellbeing and mental-health measures)
# -----------------------------------------------------------------------------
_rule("R3-WEL-H01", "wellbeing", "hard",
      "정신건강 '번영'인데 최근 2주 우울(PHQ-9) 중등도-중증 이상 또는 불안(GAD-7) 중증",
      lambda d: _in(d, "health_mental_health", "Thriving"),
      lambda d: _in(d, "kr_depression_level", ("Moderately severe", "Severe")) | _in(d, "kr_anxiety_level", "Severe"))
_rule("R3-WEL-H02", "wellbeing", "hard",
      "최근 2주 중증 우울(거의 매일 우울·흥미 상실)인데 전반적 행복이 '매우 행복'",
      lambda d: _in(d, "kr_depression_level", "Severe"),
      lambda d: _in(d, "kr_happiness", "Very happy"))
_rule("R3-WEL-S01", "wellbeing", "soft",
      "행복·삶의 만족·정신건강의 반대 극단(매우 행복↔매우 불만족, 매우 불행↔매우 만족, 중증 우울↔매우 만족, 정신적 위기↔스트레스 없음·매우 만족·매우 행복)",
      lambda d: _rows(d, "kr_happiness"),
      lambda d: _or(
          _in(d, "kr_happiness", "Very happy") & _in(d, "kr_life_satisfaction", "Very dissatisfied"),
          _in(d, "kr_happiness", "Very unhappy") & _in(d, "kr_life_satisfaction", "Very satisfied"),
          _in(d, "kr_depression_level", "Severe") & _in(d, "kr_life_satisfaction", "Very satisfied"),
          _in(d, "health_mental_health", "In crisis")
          & _or(_in(d, "health_stress_level", "None"), _in(d, "kr_life_satisfaction", "Very satisfied"),
                _in(d, "kr_happiness", "Very happy"))))
_rule("R3-WEL-S02", "wellbeing", "soft",
      "현재 중증 우울·불안인데 해당 성격 요인(우울·불안 성향)이 매우 낮거나 활력이 매우 높음",
      lambda d: _in(d, "kr_depression_level", "Severe") | _in(d, "kr_anxiety_level", "Severe"),
      lambda d: (_in(d, "kr_depression_level", "Severe")
                 & _or(_in(d, "bfi2_facet_depression", "Very low"), _in(d, "big5_depression", "Very low"),
                       _in(d, "health_energy_level", "Very high")))
      | (_in(d, "kr_anxiety_level", "Severe")
         & (_in(d, "bfi2_facet_anxiety", "Very low") | _in(d, "big5_anxiety", "Very low"))))


# -----------------------------------------------------------------------------
# life events vs histories
# -----------------------------------------------------------------------------
_rule("R3-EVT-H01", "life_events", "hard",
      "사별(혼인 상태·관계 이력)이나 사별이 주요 사건인데 상실 경험이 '큰 상실 없음'",
      lambda d: _in(d, "lifex_loss_experience", "No major loss"),
      lambda d: _or(_in(d, "demo_marital_status", "Widowed"), _in(d, "lifex_relationship_history", "Widowed"),
                    _in(d, "major_life_events", "Bereavement")))
_rule("R3-EVT-H02", "life_events", "hard",
      "건강 문제가 주요 생애 사건인데 건강 여정이 '큰 사건 없음'",
      lambda d: _in(d, "major_life_events", "Health journey"),
      lambda d: _in(d, "lifex_health_journey", "No major events"))
_rule("R3-EVT-H03", "life_events", "hard",
      "역경 경험이 '보호받음(역경 없음)'인데 성장 환경이 '고난'",
      lambda d: _in(d, "lifex_adversity_level", "Sheltered"),
      lambda d: _in(d, "lifex_childhood_environment", "Hardship"))
_rule("R3-EVT-S01", "life_events", "soft",
      "역경 없이 보호받았다면서 좌절 후 재기·배우자/자녀 상실·다중 사별·불안정하거나 잦은 이주의 성장기",
      lambda d: _in(d, "lifex_adversity_level", "Sheltered"),
      lambda d: _or(_in(d, "lifex_financial_trajectory", "Rebuilt after setback"),
                    _in(d, "lifex_loss_experience", ("Lost a partner / child", "Multiple bereavements")),
                    _in(d, "lifex_childhood_environment", ("Turbulent", "Frequently uprooted"))))


# =============================================================================
# Evaluation API
# =============================================================================
def _as_bool(x, idx):
    if isinstance(x, pd.Series):
        if len(x) != len(idx):
            raise ValueError("rule returned a Series of the wrong length")
        vals = x.to_numpy()
    else:
        vals = np.broadcast_to(np.asarray(x), (len(idx),))
    if vals.dtype != bool:
        vals = np.array([bool(v) if not pd.isna(v) else False for v in vals], dtype=bool)
    return pd.Series(vals, index=idx, dtype=bool)


def violation_matrix(df, rules=None):
    """Evaluate rules on df.

    Returns (hits, scopes, status): hits and scopes are bool DataFrames (rows x rule ids,
    hits already ANDed with scope); status maps rule id -> 'ok', 'skipped: missing <col>'
    or 'error: <msg>'.  Never raises for a failing rule.
    """
    rules = RULES if rules is None else rules
    idx = df.index
    hits, scopes, status = {}, {}, {}
    for r in rules:
        rid = r["id"]
        try:
            sc = _as_bool(r["scope"](df), idx)
            h = _as_bool(r["hit"](df), idx) & sc
            st = "ok"
        except MissingColumn as e:
            sc = h = pd.Series(False, index=idx, dtype=bool)
            st = f"skipped: missing {e.args[0] if e.args else '?'}"
        except Exception as e:  # noqa: BLE001 - report, never raise
            sc = h = pd.Series(False, index=idx, dtype=bool)
            st = f"error: {type(e).__name__}: {e}"
        hits[rid], scopes[rid], status[rid] = h.to_numpy(), sc.to_numpy(), st
    cols = [r["id"] for r in rules]
    hits_df = pd.DataFrame({c: hits[c] for c in cols}, index=idx, columns=cols).astype(bool)
    scopes_df = pd.DataFrame({c: scopes[c] for c in cols}, index=idx, columns=cols).astype(bool)
    return hits_df, scopes_df, status


def run(df):
    """Per-rule summary table and the share of rows violating at least one hard rule that ran."""
    hits, scopes, status = violation_matrix(df)
    rows = []
    for r in RULES:
        rid = r["id"]
        n_scope = int(scopes[rid].sum())
        n_hit = int(hits[rid].sum())
        rows.append({"id": rid, "group": r["group"], "severity": r["severity"],
                     "description": r["description"], "n_scope": n_scope, "n_hit": n_hit,
                     "violation_rate": (n_hit / n_scope) if n_scope else np.nan,
                     "status": status[rid]})
    table = pd.DataFrame(rows, columns=["id", "group", "severity", "description", "n_scope",
                                        "n_hit", "violation_rate", "status"])
    hard_ok = [r["id"] for r in RULES if r["severity"] == "hard" and status[r["id"]] == "ok"]
    if len(df) == 0 or not hard_ok:
        any_hard_rate = float("nan")
    else:
        any_hard_rate = float(hits[hard_ok].any(axis=1).mean())
    return table, any_hard_rate


# =============================================================================
# Catalog checks
# =============================================================================
def _load_allowed(catalog_path):
    with open(catalog_path, encoding="utf-8") as fh:
        cat = json.load(fh)
    allowed = {}
    for a in cat.get("attributes", []):
        allowed[a["id"]] = set(a.get("values", [])) | set(a.get("extra_values_not_in_matrAIx", []))
    return allowed


def _catalog_frame(allowed):
    """Small frame that cycles through every catalog value of every attribute."""
    n = max(len(v) for v in allowed.values())
    data = {}
    for col, vals in allowed.items():
        vals = sorted(vals)
        data[col] = [vals[i % len(vals)] for i in range(n)]
    return pd.DataFrame(data)


def columns_by_rule(df, rules=None):
    """Columns each rule reads, found by wrapping the module accessor `_s`."""
    rules = RULES if rules is None else rules
    g = globals()
    orig = g["_s"]
    out = {}
    current = set()

    def recording(frame, col):
        current.add(col)
        return orig(frame, col)

    g["_s"] = recording
    try:
        for r in rules:
            current.clear()
            try:
                r["scope"](df)
                r["hit"](df)
            except Exception:  # noqa: BLE001
                pass
            out[r["id"]] = set(current)
    finally:
        g["_s"] = orig
    return out


def check_values(catalog_path):
    """Verify every attribute id and value string used by the rules exists in the catalog.

    Returns a list of problem strings (empty when everything matches).
    """
    allowed = _load_allowed(catalog_path)
    problems = []

    # 1. rule metadata
    ids = [r["id"] for r in RULES]
    for rid in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate rule id {rid}")
    for r in RULES:
        if not str(r["id"]).startswith("R3-"):
            problems.append(f"{r['id']}: id must start with R3-")
        if r["severity"] not in ("hard", "soft"):
            problems.append(f"{r['id']}: bad severity {r['severity']!r}")
        if not r.get("group") or not r.get("description"):
            problems.append(f"{r['id']}: missing group or description")

    # 2. static column lists
    for name, cols in _COLUMN_LISTS.items():
        for c in cols:
            if c not in allowed:
                problems.append(f"{name}: unknown attribute id {c!r}")

    # 3. dynamic: run every rule on a catalog frame and verify referenced columns and values
    frame = _catalog_frame(allowed)
    _SEEN.clear()
    used = columns_by_rule(frame)
    _, _, status = violation_matrix(frame)
    for rid, st in status.items():
        if st != "ok":
            problems.append(f"{rid}: {st}")
    for rid, cols in used.items():
        if not cols:
            problems.append(f"{rid}: reads no column")
        for c in sorted(cols):
            if c not in allowed:
                problems.append(f"{rid}: unknown attribute id {c!r}")
    for col, v in sorted(_SEEN):
        if col not in allowed:
            problems.append(f"unknown attribute id {col!r}")
        elif v not in allowed[col]:
            problems.append(f"{col}: unknown value {v!r}")
    return sorted(set(problems))
