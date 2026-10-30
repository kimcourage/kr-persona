# -*- coding: utf-8 -*-
"""
Consistency rules for synthetic personas of South Korean adults.

Population assumed: domestic-born Korean citizens aged 19+, reference date
November 2020 (each persona is read as "as of 2020-11").

The rules were written a priori from the meaning of each attribute
(attribute_catalog.json / dimensions.json) and from real-world knowledge.
They do not use the generator, its masks, or any generated data.

Each rule is a dict:
    id          unique id, e.g. "KR-H01" (H = hard, S = soft)
    group       thematic group
    severity    'hard' (logically / legally impossible) or 'soft' (very implausible)
    description one-line Korean description (soft rules include the reason)
    scope(df)   -> bool Series: rows the rule applies to
    hit(df)     -> bool Series: violation (only counted inside scope)

run(df) -> (table, any_hard_rate)
    table: one row per rule with id, group, severity, description, n_scope,
           n_hit, violation_rate (= n_hit / n_scope), status.
    any_hard_rate: share of all rows that violate at least one hard rule
           (also stored in table.attrs['any_hard_rate']).

Robustness: every value is compared as a stripped string; a rule whose
required columns are missing is skipped (status says which column); any other
error inside a rule is caught and reported in `status` instead of raising.

Age-bracket arithmetic (reference 2020-11): bracket -> birth-year range
    18-24 (19-24 in this population) 1995-2001, 25-34 1985-1995,
    35-44 1975-1985, 45-54 1965-1975, 55-64 1955-1965, 65-74 1945-1955,
    75-84 1935-1945, 85+ <=1935.

These rules also informed generator changes; evaluation independence is limited.
See docs/08_모순_규칙.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCE_YEAR = 2020


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
class MissingColumn(KeyError):
    """Raised inside a rule when a required column is absent."""


_CACHE = {"active": False, "df_id": None, "cols": {}}
# registry of (column -> values) referenced by rules; used for self-checks
_REG: dict = {}


def _reg(col, vals):
    _REG.setdefault(col, set()).update(vals)


def _conv(series: pd.Series) -> pd.Series:
    notna = series.notna().to_numpy()
    vals = series.astype(object).to_numpy()
    out = np.empty(len(vals), dtype=object)
    for i, (v, ok) in enumerate(zip(vals, notna)):
        out[i] = str(v).strip() if ok else ""
    return pd.Series(out, index=series.index, dtype=object)


def _s(df, col) -> pd.Series:
    """Required column as stripped strings ('' for missing values)."""
    if col not in df.columns:
        raise MissingColumn(col)
    if _CACHE["active"] and _CACHE["df_id"] == id(df):
        c = _CACHE["cols"].get(col)
        if c is None:
            c = _conv(df[col])
            _CACHE["cols"][col] = c
        return c
    return _conv(df[col])


def _o(df, col) -> pd.Series:
    """Optional column: '' everywhere if the column is missing."""
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return _s(df, col)


def _T(df):
    return pd.Series(True, index=df.index)


def _F(df):
    return pd.Series(False, index=df.index)


def _L(vals):
    return [vals] if isinstance(vals, str) else list(vals)


def _in(df, col, vals):
    vals = _L(vals)
    _reg(col, vals)
    return _s(df, col).isin(vals)


def _oin(df, col, vals):
    """Optional-column membership (False everywhere if column missing)."""
    vals = _L(vals)
    _reg(col, vals)
    return _o(df, col).isin(vals)


def _nin(df, col, vals):
    vals = _L(vals)
    _reg(col, vals)
    s = _s(df, col)
    return (s != "") & ~s.isin(vals)


def _ne(df, col):
    return _s(df, col) != ""


def _rank(df, col, mapping) -> pd.Series:
    _reg(col, mapping.keys())
    return _s(df, col).map(mapping).astype(float)


def _need(df, cols):
    for c in _L(cols):
        if c not in df.columns:
            raise MissingColumn(c)


def _need_any(df, cols):
    cols = _L(cols)
    if not any(c in df.columns for c in cols):
        raise MissingColumn(" | ".join(cols[:3]) + (" ..." if len(cols) > 3 else ""))


def _any(df, parts):
    """OR of sub-conditions. Parts whose columns are missing are skipped;
    if every part is missing the rule is skipped."""
    out = None
    for p in parts:
        try:
            r = p(df)
        except MissingColumn:
            continue
        r = pd.Series(r, index=df.index).fillna(False).astype(bool)
        out = r if out is None else (out | r)
    if out is None:
        raise MissingColumn("all sub-conditions")
    return out


def _cols(df, prefix, exclude=()):
    return [c for c in df.columns if str(c).startswith(prefix) and c not in exclude]


def _req_any(cols, fn):
    """Wrap fn so that it raises MissingColumn unless >= 1 of `cols` is present.
    `cols` may be a list or a callable df -> list."""
    def f(df):
        cs = cols(df) if callable(cols) else cols
        _need_any(df, cs if cs else ["<none>"])
        return fn(df)
    return f


def _or_all(series_list, df):
    out = _F(df)
    for s in series_list:
        out = out | pd.Series(s, index=df.index).fillna(False).astype(bool)
    return out


def _any_col_in(df, cols, vals):
    present = [c for c in cols if c in df.columns]
    if not present:
        raise MissingColumn(cols[0] if cols else "?")
    out = _F(df)
    for c in present:
        out = out | _in(df, c, vals)
    return out


def _count_in(df, cols, vals):
    present = [c for c in cols if c in df.columns]
    if not present:
        raise MissingColumn(cols[0] if cols else "?")
    out = pd.Series(0, index=df.index)
    for c in present:
        out = out + _in(df, c, vals).astype(int)
    return out


def _bad_pair(df, a, b, bad):
    """True where (value of a, value of b) is in the set `bad`."""
    _reg(a, {x for x, _ in bad})
    _reg(b, {y for _, y in bad})
    A = _s(df, a).to_numpy()
    B = _s(df, b).to_numpy()
    return pd.Series([(x, y) in bad for x, y in zip(A, B)], index=df.index)


# ---------------------------------------------------------------------------
# value orders / scales (all strings exist in attribute_catalog.json)
# ---------------------------------------------------------------------------
AGE_ORDER = ["Under 5", "5-12", "13-17", "18-24", "25-34", "35-44", "45-54",
             "55-64", "65-74", "75-84", "85+"]
ADULT = AGE_ORDER[3:]
MINOR = AGE_ORDER[:3]

BIRTH = {"18-24": (1995, 2001), "25-34": (1985, 1995), "35-44": (1975, 1985),
         "45-54": (1965, 1975), "55-64": (1955, 1965), "65-74": (1945, 1955),
         "75-84": (1935, 1945), "85+": (1900, 1935)}
# generous (outer) and strict (Pew) generation birth-year ranges
GEN_GENEROUS = {"Silent": (1925, 1946), "Boomer": (1943, 1966), "Gen X": (1960, 1982),
                "Millennial": (1979, 2000), "Gen Z": (1994, 2012), "Gen Alpha": (2010, 2025)}
GEN_STRICT = {"Silent": (1928, 1945), "Boomer": (1946, 1964), "Gen X": (1965, 1980),
              "Millennial": (1981, 1996), "Gen Z": (1997, 2012), "Gen Alpha": (2013, 2025)}
# formative decade -> calendar span (2020s truncated at reference year)
FORMATIVE = {"1960s-70s": (1960, 1979), "1980s": (1980, 1989), "1990s": (1990, 1999),
             "2000s": (2000, 2009), "2010s": (2010, 2019), "2020s": (2020, REFERENCE_YEAR)}
FORMATIVE_AGE = (5, 30)  # generous formative-years window (age)


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def _form_birth_window(decade):
    s, e = FORMATIVE[decade]
    return (s - FORMATIVE_AGE[1], e - FORMATIVE_AGE[0])


_GEN_AGE_HARD = {(g, a) for g, gr in GEN_GENEROUS.items() for a, br in BIRTH.items()
                 if not _overlap(gr, br)}
_GEN_AGE_SOFT = {(g, a) for g, gr in GEN_STRICT.items() for a, br in BIRTH.items()
                 if not _overlap(gr, br) and (g, a) not in _GEN_AGE_HARD}
_FORM_AGE_HARD = {(d, a) for d in FORMATIVE for a, br in BIRTH.items()
                  if not _overlap(_form_birth_window(d), br)}
_GEN_FORM_HARD = {(g, d) for g, gr in GEN_GENEROUS.items() for d in FORMATIVE
                  if not _overlap(gr, _form_birth_window(d))}

L5 = {"Very low": 1, "Low": 2, "Average": 3, "High": 4, "Very high": 5}
VAL = {"Irrelevant": 1, "Minor": 2, "Moderate": 3, "Important": 4, "Core value": 5}
TRAIT = {"Absent": 1, "Slight": 2, "Moderate": 3, "Strong": 4, "Signature": 5}
COG = {"None": 1, "Low": 2, "Moderate": 3, "High": 4, "Very high": 5}
LANG = {"None": 0, "Basic": 1, "Conversational": 2, "Fluent": 3, "Native": 4}
ENG = {"None": 0, "Basic (A1-A2)": 1, "Intermediate (B1-B2)": 2, "Fluent (C1-C2)": 3, "Native": 4}
RISK = {"Risk-averse": 1, "Cautious": 2, "Balanced": 3, "Risk-tolerant": 4, "Risk-seeking": 5}
ATT = {"Opposed": 1, "Skeptical": 2, "Neutral": 3, "Positive": 4, "Enthusiast": 5}
INCOME = {"<$25k": 1, "$25k-50k": 2, "$50k-100k": 3, "$100k-200k": 4, "$200k+": 5}
SES = {"Low income": 1, "Lower-middle": 2, "Middle": 3, "Upper-middle": 4, "High income": 5}
SUBJ_CLASS = {"Lower class": 1, "Lower-middle": 2, "Middle": 3, "Upper-middle": 4, "Upper class": 5}

WORKING = ["Full-time", "Part-time", "Self-employed", "Gig / freelance", "Unpaid family worker"]
PARENT4 = ["New parent", "Parent of minors", "Parent of adults", "Grandparent"]
CHILD_COUNTED = ["1 child", "2 children", "3+ children", "Adult children"]
EDU_LOW = ["No formal", "Primary", "Secondary"]
EDU_COLLEGE = ["Some college", "Associate's", "Bachelor's", "Master's", "Doctorate", "Postdoc"]
EDU_BA_PLUS = ["Bachelor's", "Master's", "Doctorate", "Postdoc"]
EDU_GRAD = ["Master's", "Doctorate", "Postdoc"]
NONREL = ["None", "Atheist / agnostic"]
AFFIL = ["Christian", "Muslim", "Hindu", "Buddhist", "Jewish", "Sikh", "Folk / traditional",
         "Catholic", "Confucian", "Korean new religion", "Other religion", "Protestant",
         "Won Buddhist"]
AI_TASK_CURRENT = ["Currently mostly AI-assisted", "Currently partially AI-assisted"]
POST_2020_AI_TOOLS = ["tool_chatgpt", "tool_claude", "tool_midjourney", "tool_github_copilot"]

ADV_FAM = ["fam_particle_physics", "fam_astrophysics", "fam_molecular_biology",
           "fam_biochemistry", "fam_organic_chemistry", "fam_physical_chemistry",
           "fam_nanotechnology", "fam_materials_science", "fam_nuclear_engineering",
           "fam_epidemiology", "fam_actuarial_science", "fam_compilers", "fam_cryptography",
           "fam_cardiology", "fam_neurology", "fam_oncology", "fam_radiology", "fam_surgery",
           "fam_immunology", "fam_pharmacology", "fam_constitutional_law", "fam_tax_law",
           "fam_quantitative_trading", "fam_deep_learning", "fam_distributed_systems"]

# Big Five: BFI-2 domain, its 3 BFI-2 facets, the 6 IPIP-NEO facets of the same domain
DOMAINS = {
    "O": ("bfi2_domain_open_mindedness",
          ["bfi2_facet_intellectual_curiosity", "bfi2_facet_aesthetic_sensitivity",
           "bfi2_facet_creative_imagination"],
          ["big5_imagination", "big5_artistic_interest", "big5_emotionality",
           "big5_adventurousness", "big5_intellect", "big5_liberalism"]),
    "C": ("bfi2_domain_conscientiousness",
          ["bfi2_facet_organization", "bfi2_facet_productiveness", "bfi2_facet_responsibility"],
          ["big5_self_efficacy", "big5_orderliness", "big5_dutifulness",
           "big5_achievement_striving", "big5_self_discipline", "big5_cautiousness"]),
    "E": ("bfi2_domain_extraversion",
          ["bfi2_facet_sociability", "bfi2_facet_assertiveness", "bfi2_facet_energy_level"],
          ["big5_friendliness", "big5_gregariousness", "big5_assertiveness",
           "big5_activity_level", "big5_excitement_seeking", "big5_cheerfulness"]),
    "A": ("bfi2_domain_agreeableness",
          ["bfi2_facet_compassion", "bfi2_facet_respectfulness", "bfi2_facet_trust"],
          ["big5_trust", "big5_morality", "big5_altruism", "big5_cooperation",
           "big5_modesty", "big5_sympathy"]),
    "N": ("bfi2_domain_negative_emotionality",
          ["bfi2_facet_anxiety", "bfi2_facet_depression", "bfi2_facet_emotional_volatility"],
          ["big5_anxiety", "big5_anger", "big5_depression", "big5_self_consciousness",
           "big5_immoderation", "big5_vulnerability"]),
}

# same-construct facet pairs, BFI-2 <-> IPIP-NEO
BF_FACET_PAIRS = [
    ("bfi2_facet_anxiety", L5, "big5_anxiety", L5, False),
    ("bfi2_facet_depression", L5, "big5_depression", L5, False),
    ("bfi2_facet_emotional_volatility", L5, "big5_anger", L5, False),
    ("bfi2_facet_assertiveness", L5, "big5_assertiveness", L5, False),
    ("bfi2_facet_sociability", L5, "big5_gregariousness", L5, False),
    ("bfi2_facet_energy_level", L5, "big5_activity_level", L5, False),
    ("bfi2_facet_trust", L5, "big5_trust", L5, False),
    ("bfi2_facet_compassion", L5, "big5_sympathy", L5, False),
    ("bfi2_facet_respectfulness", L5, "big5_cooperation", L5, False),
    ("bfi2_facet_organization", L5, "big5_orderliness", L5, False),
    ("bfi2_facet_responsibility", L5, "big5_dutifulness", L5, False),
    ("bfi2_facet_creative_imagination", L5, "big5_imagination", L5, False),
    ("bfi2_facet_aesthetic_sensitivity", L5, "big5_artistic_interest", L5, False),
    ("bfi2_facet_intellectual_curiosity", L5, "big5_intellect", L5, False),
]
# VIA-style character strengths <-> Big Five facets
CHAR_PAIRS = [
    ("trait_curiosity", TRAIT, "bfi2_facet_intellectual_curiosity", L5, False),
    ("trait_creativity", TRAIT, "bfi2_facet_creative_imagination", L5, False),
    ("trait_appreciation_of_beauty", TRAIT, "bfi2_facet_aesthetic_sensitivity", L5, False),
    ("trait_love_of_learning", TRAIT, "need_for_cognition", L5, False),
    ("trait_kindness", TRAIT, "bfi2_facet_compassion", L5, False),
    ("trait_humility", TRAIT, "big5_modesty", L5, False),
    ("trait_honesty", TRAIT, "big5_morality", L5, False),
    ("trait_discipline", TRAIT, "big5_self_discipline", L5, False),
    ("trait_prudence", TRAIT, "big5_cautiousness", L5, False),
    ("trait_zest", TRAIT, "bfi2_facet_energy_level", L5, False),
    ("trait_leadership", TRAIT, "bfi2_facet_assertiveness", L5, False),
]
# communication / interpersonal self-report <-> personality (inverse=True: opposite poles expected)
COMM_PAIRS = [
    ("interpersonal_communion_warmth", L5, "bfi2_domain_agreeableness", L5, False),
    ("interpersonal_agency_dominance", L5, "bfi2_facet_assertiveness", L5, False),
    ("cog_assertiveness", COG, "bfi2_facet_assertiveness", L5, False),
    ("cog_curiosity", COG, "trait_curiosity", TRAIT, False),
    ("cog_open_mindedness", COG, "trait_open_mindedness", TRAIT, False),
    ("cog_optimism", COG, "trait_hope_optimism", TRAIT, False),
    ("cog_empathy_expression", COG, "trait_empathy", TRAIT, False),
    ("cog_procrastination", COG, "big5_self_discipline", L5, True),
]

SCHWARTZ_GROUPS = {
    "VA-S01": ("변화개방(자율·자극·쾌락)", [
        ("schwartz_value_self_direction",
         ["val_independence", "val_creativity_self_expression", "val_personal_freedom"]),
        ("schwartz_value_stimulation", ["val_adventure"]),
        ("schwartz_value_hedonism", ["val_fun_enjoyment"])]),
    "VA-S02": ("자기고양(성취·권력)", [
        ("schwartz_value_achievement", ["val_achievement", "val_career_success"]),
        ("schwartz_value_power",
         ["val_power_influence", "val_wealth", "val_social_status", "val_recognition"])]),
    "VA-S03": ("보존(안전·전통)", [
        ("schwartz_value_security", ["val_security_stability", "val_order_structure"]),
        ("schwartz_value_tradition", ["val_tradition"])]),
    "VA-S04": ("자기초월(박애·보편)", [
        ("schwartz_value_benevolence", ["val_helping_others", "val_loyalty"]),
        ("schwartz_value_universalism",
         ["val_equality", "val_justice_fairness", "val_sustainability"])]),
}
PRIORITY_MAP = {  # values_priority -> (val_* item, schwartz value)
    "Achievement": ("val_achievement", "schwartz_value_achievement"),
    "Security": ("val_security_stability", "schwartz_value_security"),
    "Autonomy": ("val_independence", "schwartz_value_self_direction"),
    "Community": ("val_community", "schwartz_value_benevolence"),
    "Novelty": ("val_adventure", "schwartz_value_stimulation"),
    "Tradition": ("val_tradition", "schwartz_value_tradition"),
}
DOMINANT_MAP = {
    "High openness": "bfi2_domain_open_mindedness",
    "High conscientiousness": "bfi2_domain_conscientiousness",
    "High extraversion": "bfi2_domain_extraversion",
    "High agreeableness": "bfi2_domain_agreeableness",
    "High neuroticism": "bfi2_domain_negative_emotionality",
}
DOSPERT = ["dospert_financial_risk_tolerance", "dospert_ethical_risk_tolerance",
           "dospert_recreational_risk_tolerance", "dospert_social_risk_tolerance",
           "dospert_health_safety_risk_tolerance"]

# subject_specialty -> plausible primary domains ('Operations' is cross-cutting: no check)
SPECIALTY_DOMAINS = {
    "Machine learning": {"Software & AI", "Engineering", "Natural Sciences", "Finance & Economics",
                         "Social Sciences", "Healthcare & Medicine", "Manufacturing",
                         "Business & Management", "Education", "Media & Journalism",
                         "Public Sector", "Agriculture"},
    "Cardiology": {"Healthcare & Medicine", "Natural Sciences", "Education"},
    "Constitutional law": {"Law & Policy", "Public Sector", "Social Sciences", "Education",
                           "Media & Journalism"},
    "Quant trading": {"Finance & Economics", "Software & AI", "Business & Management",
                      "Natural Sciences", "Engineering"},
    "Curriculum design": {"Education", "Public Sector", "Arts & Humanities", "Social Sciences",
                          "Business & Management", "Healthcare & Medicine", "Software & AI"},
    "Structural engineering": {"Engineering", "Skilled Trades", "Manufacturing", "Public Sector",
                               "Education"},
    "Molecular biology": {"Natural Sciences", "Healthcare & Medicine", "Agriculture", "Education",
                          "Engineering"},
    "Behavioral economics": {"Social Sciences", "Finance & Economics", "Business & Management",
                             "Law & Policy", "Public Sector", "Education", "Software & AI"},
    "Comparative literature": {"Arts & Humanities", "Education", "Media & Journalism",
                               "Social Sciences"},
    "Agronomy": {"Agriculture", "Natural Sciences", "Public Sector", "Education", "Manufacturing"},
    "Robotics": {"Manufacturing", "Engineering", "Software & AI", "Natural Sciences", "Education",
                 "Healthcare & Medicine"},
    "Investigative reporting": {"Media & Journalism", "Law & Policy", "Public Sector",
                                "Social Sciences"},
    "Urban planning": {"Public Sector", "Engineering", "Social Sciences", "Law & Policy",
                       "Arts & Humanities", "Business & Management", "Education"},
    "Culinary arts": {"Hospitality", "Arts & Humanities", "Education", "Business & Management",
                      "Skilled Trades", "Agriculture"},
    "Electrical work": {"Skilled Trades", "Engineering", "Manufacturing", "Public Sector",
                        "Agriculture", "Hospitality"},
}
ALL_DOMAINS = ["Software & AI", "Healthcare & Medicine", "Law & Policy", "Finance & Economics",
               "Education", "Engineering", "Natural Sciences", "Social Sciences",
               "Arts & Humanities", "Business & Management", "Agriculture", "Manufacturing",
               "Media & Journalism", "Public Sector", "Hospitality", "Skilled Trades"]


# ---------------------------------------------------------------------------
# domain-specific helpers
# ---------------------------------------------------------------------------
def _age_in(df, vals):
    return _in(df, "age_bracket", vals)


def _age_ge(df, b):
    return _age_in(df, ADULT[ADULT.index(b):])


def _age_le(df, b):
    return _age_in(df, ADULT[:ADULT.index(b) + 1])


def _adult(df):
    return _age_in(df, ADULT)


def _woman(df):
    return _in(df, "gender_identity", "Woman")


def _foreign_lang_cols(df):
    return _cols(df, "lang_", exclude=("lang_korean",))


def _n_foreign(df, min_level):
    """Number of non-Korean languages at >= min_level (LANG scale).
    English counts if either lang_english or english_proficiency reaches the level."""
    cols = _foreign_lang_cols(df)
    has_eng = "english_proficiency" in df.columns
    if not cols and not has_eng:
        raise MissingColumn("lang_* / english_proficiency")
    total = pd.Series(0, index=df.index)
    eng_ok = (_rank(df, "english_proficiency", ENG) >= min_level) if has_eng else _F(df)
    for c in cols:
        ok = _rank(df, c, LANG) >= min_level
        if c == "lang_english":
            ok = ok | eng_ok
        total = total + ok.astype(int)
    if has_eng and "lang_english" not in cols:
        total = total + eng_ok.astype(int)
    return total


def _foreign_native(df):
    parts = [lambda d: _any_col_in(d, _foreign_lang_cols(d), "Native"),
             lambda d: _in(d, "english_proficiency", "Native")]
    return _any(df, parts)


def _gap_outside(t, O, gap=3):
    """target rank lies >= gap levels above ALL others or below ALL others."""
    mx = O.max(axis=1, skipna=False)
    mn = O.min(axis=1, skipna=False)
    return ((t - mx) >= gap) | ((mn - t) >= gap)


def _domain_parts(df, key):
    dom, bfi, neo = DOMAINS[key]
    t = _rank(df, dom, L5)
    parts = []
    for fac in (bfi, neo):
        if all(c in df.columns for c in fac):
            O = pd.concat([_rank(df, c, L5) for c in fac], axis=1)
            parts.append((O.notna().all(axis=1) & t.notna(), _gap_outside(t, O)))
    if not parts:
        raise MissingColumn(bfi[0])
    return parts


def _domain_scope(key):
    def f(df):
        out = _F(df)
        for sc, _ in _domain_parts(df, key):
            out = out | sc
        return out
    return f


def _domain_hit(key):
    def f(df):
        out = _F(df)
        for sc, h in _domain_parts(df, key):
            out = out | (sc & h)
        return out
    return f


def _pairs_eval(df, pairs):
    scope = None
    hit = None
    for a, am, b, bm, inverse in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        ra, rb = _rank(df, a, am), _rank(df, b, bm)
        amax, amin, bmax, bmin = max(am.values()), min(am.values()), max(bm.values()), min(bm.values())
        if inverse:
            h = ((ra == amax) & (rb == bmax)) | ((ra == amin) & (rb == bmin))
        else:
            h = ((ra == amax) & (rb == bmin)) | ((ra == amin) & (rb == bmax))
        sc = ra.notna() & rb.notna()
        scope = sc if scope is None else (scope | sc)
        hit = h if hit is None else (hit | h)
    if scope is None:
        raise MissingColumn(pairs[0][0])
    return scope, hit


def _pairs_scope(pairs):
    return lambda df: _pairs_eval(df, pairs)[0]


def _pairs_hit(pairs):
    return lambda df: _pairs_eval(df, pairs)[1]


def _schwartz_eval(df, items):
    scope, hit = None, None
    for sch, vals in items:
        present = [v for v in vals if v in df.columns]
        if sch not in df.columns or not present:
            continue
        t = _rank(df, sch, L5)
        O = pd.concat([_rank(df, v, VAL) for v in present], axis=1)
        sc = t.notna() & O.notna().all(axis=1)
        h = sc & _gap_outside(t, O)
        scope = sc if scope is None else (scope | sc)
        hit = h if hit is None else (hit | h)
    if scope is None:
        raise MissingColumn(items[0][0])
    return scope, hit


def _mbti_letter(df, pos):
    return _s(df, "mbti_type").str[pos:pos + 1]


# ---------------------------------------------------------------------------
# rule registry
# ---------------------------------------------------------------------------
RULES: list = []


def _rule(rid, group, severity, description, scope, hit):
    assert severity in ("hard", "soft")
    RULES.append(dict(id=rid, group=group, severity=severity,
                      description=description, scope=scope, hit=hit))


# ===========================================================================
# 1. korea_identity : domestic-born Korean citizens living in Korea
# ===========================================================================
G = "korea_identity"
_rule("KR-H01", G, "hard", "모집단은 19세 이상 성인인데 연령대가 18세 미만 구간(Under 5/5-12/13-17)",
      lambda d: _ne(d, "age_bracket"),
      lambda d: _age_in(d, MINOR))
_rule("KR-H02", G, "hard", "국내 거주 한국 국민인데 거주 권역(region)이 'East Asia'가 아님",
      lambda d: _ne(d, "region"),
      lambda d: _nin(d, "region", "East Asia"))
_rule("KR-H03", G, "hard", "한국 국민 모집단인데 시민권 상태가 영주권자·비자 소지자·미등록(비국민)",
      lambda d: _ne(d, "demo_citizenship_status"),
      lambda d: _in(d, "demo_citizenship_status", ["Permanent resident", "Visa holder", "Undocumented"]))
_rule("KR-H04", G, "hard", "국내 출생 한국 국민인데 민족 배경이 동아시아계도 혼혈도 아님",
      lambda d: _ne(d, "demo_ethnicity_broad"),
      lambda d: _nin(d, "demo_ethnicity_broad", ["East Asian", "Multiracial"]))
_rule("KR-H05", G, "hard", "국내 출생 한국인의 제1언어(primary_language)가 한국어가 아님",
      lambda d: _ne(d, "primary_language"),
      lambda d: _nin(d, "primary_language", "Korean"))
_rule("KR-H06", G, "hard", "국내 출생 한국인의 한국어 능력이 회화 수준 이하(Conversational/Basic/None)",
      lambda d: _ne(d, "lang_korean"),
      lambda d: _in(d, "lang_korean", ["Conversational", "Basic", "None"]))
_rule("KR-H07", G, "hard", "한국에서 나고 사는데 한국 문화 친숙도가 '방문/학습/생소'(Visited/Studied/Unfamiliar)",
      lambda d: _ne(d, "cult_south_korea"),
      lambda d: _in(d, "cult_south_korea", ["Visited", "Studied", "Unfamiliar"]))
_rule("KR-H08", G, "hard", "국내 출생자인데 이민 세대가 '이민 1세대'(First-generation immigrant)",
      lambda d: _ne(d, "lifex_immigration_generation"),
      lambda d: _in(d, "lifex_immigration_generation", "First-generation immigrant"))
_rule("KR-S01", G, "soft", "귀화·혼혈·이민 2/3세대 배경 — 2020년 국내 출생 성인 국민 중 극소수(다문화가정 성인 자녀·화교 등)",
      lambda d: _any(d, [lambda x: _ne(x, "demo_citizenship_status"),
                         lambda x: _ne(x, "demo_ethnicity_broad"),
                         lambda x: _ne(x, "lifex_immigration_generation")]),
      lambda d: _any(d, [lambda x: _in(x, "demo_citizenship_status", "Naturalized citizen"),
                         lambda x: _in(x, "demo_ethnicity_broad", "Multiracial"),
                         lambda x: _in(x, "lifex_immigration_generation",
                                       ["Second-generation", "Third+ generation"])]))
_rule("KR-S02", G, "soft", "국내 출생 한국인의 한국어가 'Native'가 아닌 'Fluent' — 해외 성장 등 예외적 경우만 가능",
      lambda d: _ne(d, "lang_korean"),
      lambda d: _in(d, "lang_korean", "Fluent"))
_rule("KR-S03", G, "soft", "한국인이 한식(cuis_korean)을 회피(Avoid) — 매우 이례적",
      lambda d: _ne(d, "cuis_korean"),
      lambda d: _in(d, "cuis_korean", "Avoid"))
_rule("KR-S04", G, "soft", "주 메신저가 WhatsApp/WeChat/Signal — 한국은 카카오톡이 사실상 독점이라 매우 이례적",
      lambda d: _ne(d, "lstyle_primary_messenger"),
      lambda d: _in(d, "lstyle_primary_messenger", ["WhatsApp", "WeChat", "Signal"]))
_rule("KR-S05", G, "soft", "종교가 이슬람/힌두/유대/시크 — 국내 출생 한국인 신자는 극소수(<0.1%)",
      lambda d: _ne(d, "demo_religion_affiliation"),
      lambda d: _in(d, "demo_religion_affiliation", ["Muslim", "Hindu", "Jewish", "Sikh"]))

# ===========================================================================
# 2. age_lifecycle : age bracket (ref. 2020-11) vs cohort / life stage
# ===========================================================================
G = "age_lifecycle"
_rule("AG-H01", G, "hard", "세대(demo_generation)가 2020년 기준 연령대와 양립 불가(넓은 세대 정의로도 출생연도 불일치, 예: Gen Z인데 35세 이상)",
      lambda d: _adult(d) & _in(d, "demo_generation", list(GEN_GENEROUS)),
      lambda d: _bad_pair(d, "demo_generation", "age_bracket", _GEN_AGE_HARD))
_rule("AG-H02", G, "hard", "형성기 10년대(lifex_formative_decade)가 연령대와 양립 불가(그 10년 동안 5~30세였을 수 없음)",
      lambda d: _adult(d) & _in(d, "lifex_formative_decade", list(FORMATIVE)),
      lambda d: _bad_pair(d, "lifex_formative_decade", "age_bracket", _FORM_AGE_HARD))
_rule("AG-H03", G, "hard", "세대와 형성기 10년대가 양립 불가(예: Boomer인데 형성기가 2000년대 이후)",
      lambda d: _in(d, "demo_generation", list(GEN_GENEROUS)) & _in(d, "lifex_formative_decade", list(FORMATIVE)),
      lambda d: _bad_pair(d, "demo_generation", "lifex_formative_decade", _GEN_FORM_HARD))
_rule("AG-H04", G, "hard", "19~24세인데 최종학력이 박사/박사후 — 한국 학제상 최소 27세 이상",
      lambda d: _age_in(d, "18-24") & _ne(d, "highest_education"),
      lambda d: _in(d, "highest_education", ["Doctorate", "Postdoc"]))
_rule("AG-H05", G, "hard", "19~24세인데 은퇴(취업상태·생애단계·직급) 또는 경력 20년 이상",
      lambda d: _age_in(d, "18-24"),
      lambda d: _any(d, [lambda x: _in(x, "demo_employment_status", "Retired"),
                         lambda x: _in(x, "life_stage", "Retirement"),
                         lambda x: _in(x, "seniority", "Retired"),
                         lambda x: _in(x, "years_experience", "20+")]))
_rule("AG-H06", G, "hard", "19~24세인데 성인·10대 자녀 또는 손주가 있음(자녀 출산 시 부모 나이가 11세 이하가 됨)",
      lambda d: _age_in(d, "18-24"),
      lambda d: _any(d, [lambda x: _in(x, "demo_parental_status", ["Parent of adults", "Grandparent"]),
                         lambda x: _in(x, "demo_children_count", "Adult children"),
                         lambda x: _in(x, "lifex_parenting_journey",
                                       ["Teenagers", "Grown children", "Empty nester", "Raising grandchildren"]),
                         lambda x: _in(x, "life_stage", "Empty nester")]))
_rule("AG-H07", G, "hard", "19~24세인데 생애단계가 중년(Mid-life)",
      lambda d: _age_in(d, "18-24") & _ne(d, "life_stage"),
      lambda d: _in(d, "life_stage", "Mid-life"))
_rule("AG-H08", G, "hard", "19~24세인데 현재 관계 기간이 15년 이상(9세 이하부터 교제)",
      lambda d: _age_in(d, "18-24") & _ne(d, "demo_relationship_length"),
      lambda d: _in(d, "demo_relationship_length", "15+ years"))
_rule("AG-S01", G, "soft", "세대가 표준(Pew) 출생연도 정의 밖의 연령대(예: Boomer인데 45~54세, Gen Z인데 25~34세)",
      lambda d: _adult(d) & _in(d, "demo_generation", list(GEN_STRICT)),
      lambda d: _bad_pair(d, "demo_generation", "age_bracket", _GEN_AGE_SOFT))
_rule("AG-S02", G, "soft", "형성기 10년대가 '2020s' — 기준시점(2020-11)에 2020년대는 11개월만 경과",
      lambda d: _ne(d, "lifex_formative_decade"),
      lambda d: _in(d, "lifex_formative_decade", "2020s"))
_rule("AG-S03", G, "soft", "연령 대비 이례적 생애단계 — 25~44세 은퇴, 또는 65세 이상 학생·초기경력·인턴",
      lambda d: _age_ge(d, "25-34"),
      lambda d: _any(d, [
          lambda x: _age_in(x, ["25-34", "35-44"]) & _in(x, "demo_employment_status", "Retired"),
          lambda x: _age_in(x, ["25-34", "35-44"]) & _in(x, "life_stage", "Retirement"),
          lambda x: _age_ge(x, "65-74") & _in(x, "demo_employment_status", "Student"),
          lambda x: _age_ge(x, "65-74") & _in(x, "life_stage", ["Student", "Early career"]),
          lambda x: _age_ge(x, "65-74") & _in(x, "seniority", "Student / intern")]))
_rule("AG-S04", G, "soft", "경력 연수가 연령 대비 과도 — 19~24세 11~20년, 25~34세 20년 이상(10대 초중반부터 종사)",
      lambda d: _age_in(d, ["18-24", "25-34"]) & _ne(d, "years_experience"),
      lambda d: (_age_in(d, "18-24") & _in(d, "years_experience", "11-20"))
      | (_age_in(d, "25-34") & _in(d, "years_experience", "20+")))
_rule("AG-S05", G, "soft", "연령 대비 이례적 자녀 상태 — 25~44세 조부모, 25~34세 성인자녀, 여성 55+/전체 75+의 영유아 부모",
      lambda d: _age_ge(d, "25-34"),
      lambda d: _any(d, [
          lambda x: _age_in(x, ["25-34", "35-44"]) & _in(x, "demo_parental_status", "Grandparent"),
          lambda x: _age_in(x, "25-34") & _in(x, "demo_parental_status", "Parent of adults"),
          lambda x: _age_in(x, "25-34") & _in(x, "demo_children_count", "Adult children"),
          lambda x: _age_in(x, "25-34") & _in(x, "lifex_parenting_journey",
                                              ["Grown children", "Empty nester", "Raising grandchildren"]),
          lambda x: ((_woman(x) & _age_ge(x, "55-64")) | _age_ge(x, "75-84"))
          & _in(x, "demo_parental_status", "New parent"),
          lambda x: ((_woman(x) & _age_ge(x, "55-64")) | _age_ge(x, "75-84"))
          & _in(x, "life_stage", "Parent of young kids"),
          lambda x: ((_woman(x) & _age_ge(x, "55-64")) | _age_ge(x, "75-84"))
          & _in(x, "lifex_parenting_journey", "Young children")]))
_rule("AG-S06", G, "soft", "의무교육 확대 이후 세대의 무학/초졸 — 34세 이하 무학·초졸, 35~54세 무학은 극히 드묾",
      lambda d: _age_le(d, "45-54") & _ne(d, "highest_education"),
      lambda d: (_age_le(d, "25-34") & _in(d, "highest_education", ["No formal", "Primary"]))
      | (_age_in(d, ["35-44", "45-54"]) & _in(d, "highest_education", "No formal")))
_rule("AG-S07", G, "soft", "75세 이상인데 큰 상실 경험 없음 — 부모가 생존해 있을 가능성이 거의 없음",
      lambda d: _age_ge(d, "75-84") & _ne(d, "lifex_loss_experience"),
      lambda d: _in(d, "lifex_loss_experience", "No major loss"))

# ===========================================================================
# 3. family : parenthood / children / household
# ===========================================================================
G = "family"
_rule("FA-H01", G, "hard", "부모 상태와 자녀 수 상충 — '부모 아님'인데 자녀 있음, 또는 부모·조부모인데 자녀 없음",
      lambda d: _ne(d, "demo_parental_status") & _ne(d, "demo_children_count"),
      lambda d: (_in(d, "demo_parental_status", "Not a parent") & _in(d, "demo_children_count", CHILD_COUNTED))
      | (_in(d, "demo_parental_status", PARENT4) & _in(d, "demo_children_count", "None")))
_rule("FA-H02", G, "hard", "양육 여정과 부모 상태/자녀 수 상충 — '자녀 없음'인데 부모·자녀 있음, 또는 양육 중/완료인데 부모 아님·자녀 없음",
      lambda d: _ne(d, "lifex_parenting_journey"),
      lambda d: _any(d, [
          lambda x: _in(x, "lifex_parenting_journey", "No children") & _in(x, "demo_parental_status", PARENT4),
          lambda x: _in(x, "lifex_parenting_journey", "No children") & _in(x, "demo_children_count", CHILD_COUNTED),
          lambda x: _in(x, "lifex_parenting_journey",
                        ["Young children", "Teenagers", "Grown children", "Empty nester", "Raising grandchildren"])
          & _in(x, "demo_parental_status", "Not a parent"),
          lambda x: _in(x, "lifex_parenting_journey",
                        ["Young children", "Teenagers", "Grown children", "Empty nester"])
          & _in(x, "demo_children_count", "None")]))
_rule("FA-H03", G, "hard", "생애단계가 '영유아 부모/빈 둥지'인데 부모 아님 또는 자녀 없음",
      lambda d: _in(d, "life_stage", ["Parent of young kids", "Empty nester"]),
      lambda d: _any(d, [lambda x: _in(x, "demo_parental_status", "Not a parent"),
                         lambda x: _in(x, "demo_children_count", "None")]))
_rule("FA-H04", G, "hard", "55세 이상 여성인데 임신 중(Expecting)",
      lambda d: _woman(d) & _age_ge(d, "55-64"),
      lambda d: _in(d, "demo_children_count", "Expecting"))
_rule("FA-H05", G, "hard", "1인 가구인데 주거 형태가 '가족과 거주' 또는 손주를 양육 중",
      lambda d: _in(d, "lstyle_household_size", "Lives alone"),
      lambda d: _any(d, [lambda x: _in(x, "demo_housing_status", "Living with family"),
                         lambda x: _in(x, "lifex_parenting_journey", "Raising grandchildren")]))
_rule("FA-S01", G, "soft", "자녀 세부 상태 불일치 — 미성년/영유아 자녀 부모인데 자녀 항목은 '성인 자녀', 의붓·위탁 부모인데 자녀 없음",
      lambda d: _ne(d, "demo_children_count"),
      lambda d: _any(d, [
          lambda x: _in(x, "demo_parental_status", "Parent of minors") & _in(x, "demo_children_count", "Adult children"),
          lambda x: _in(x, "lifex_parenting_journey", "Young children") & _in(x, "demo_children_count", "Adult children"),
          lambda x: _in(x, "demo_parental_status", "Step / foster parent") & _in(x, "demo_children_count", "None")]))
_rule("FA-S02", G, "soft", "1인 가구인데 영유아 부모이거나 공유 주거 — 동거인이 있는 정황",
      lambda d: _in(d, "lstyle_household_size", "Lives alone"),
      lambda d: _any(d, [lambda x: _in(x, "life_stage", "Parent of young kids"),
                         lambda x: _in(x, "demo_parental_status", "New parent"),
                         lambda x: _in(x, "demo_housing_status", "Shared housing")]))

# ===========================================================================
# 4. marital : marital status vs relationship history / length
# ===========================================================================
G = "marital"
_rule("MA-H01", G, "hard", "기혼·동거·약혼·연애 중인데 관계 기간이 '관계 없음'(Not in one)",
      lambda d: _in(d, "demo_marital_status", ["Married", "Domestic partnership", "Engaged", "In a relationship"]),
      lambda d: _in(d, "demo_relationship_length", "Not in one"))
_rule("MA-H02", G, "hard", "혼인상태 '미혼(Single)'인데 결혼/재혼/사별 이력 — 이혼·사별 상태값이 별도로 존재",
      lambda d: _in(d, "demo_marital_status", "Single"),
      lambda d: _in(d, "lifex_relationship_history", ["Married once", "Remarried", "Widowed"]))
_rule("MA-H03", G, "hard", "사별했는데(혼인상태 또는 관계 이력) 큰 상실 경험이 없음",
      lambda d: _any(d, [lambda x: _in(x, "demo_marital_status", "Widowed"),
                         lambda x: _in(x, "lifex_relationship_history", "Widowed")]),
      lambda d: _in(d, "lifex_loss_experience", "No major loss"))
_rule("MA-S01", G, "soft", "'Single'인데 현재 관계 기간이 있음 — 'In a relationship' 값이 별도로 존재",
      lambda d: _in(d, "demo_marital_status", "Single"),
      lambda d: _in(d, "demo_relationship_length", ["Under 1 year", "1-5 years", "5-15 years", "15+ years"]))
_rule("MA-S02", G, "soft", "기혼인데 성적 지향이 동성애 — 한국은 동성혼 불인정(2020)이라 이성혼만 가능",
      lambda d: _in(d, "demo_marital_status", "Married"),
      lambda d: _in(d, "demo_sexual_orientation", "Gay / lesbian"))

# ===========================================================================
# 5. employment : employment status vs work attributes
# ===========================================================================
G = "employment"
_rule("EM-H01", G, "hard", "취업 상태와 근무 일정 상충 — 취업자인데 일정 '실업', 또는 실업자인데 9-to-5/교대/대기 근무",
      lambda d: _ne(d, "demo_employment_status") & _ne(d, "lstyle_work_schedule"),
      lambda d: (_in(d, "demo_employment_status", WORKING) & _in(d, "lstyle_work_schedule", "Unemployed"))
      | (_in(d, "demo_employment_status", "Unemployed")
         & _in(d, "lstyle_work_schedule", ["9-to-5", "Shift work", "On-call"])))
_rule("EM-H02", G, "hard", "취업자인데 직무가 '해당 없음', 또는 전일제인데 소속 조직 규모 '해당 없음'",
      lambda d: _in(d, "demo_employment_status", WORKING),
      lambda d: _any(d, [lambda x: _in(x, "role_function", "Not applicable"),
                         lambda x: _in(x, "demo_employment_status", "Full-time")
                         & _in(x, "company_size", "Not applicable")]))
_rule("EM-S01", G, "soft", "비취업자의 근무 일정 — 실업자인데 유연/프리랜서 일정, 은퇴자·전업주부인데 9-to-5/교대/대기 근무",
      lambda d: _in(d, "demo_employment_status", ["Unemployed", "Retired", "Homemaker"]),
      lambda d: (_in(d, "demo_employment_status", "Unemployed")
                 & _in(d, "lstyle_work_schedule", ["Flexible hours", "Freelance"]))
      | (_in(d, "demo_employment_status", ["Retired", "Homemaker"])
         & _in(d, "lstyle_work_schedule", ["9-to-5", "Shift work", "On-call"])))
_rule("EM-S02", G, "soft", "은퇴자인데 생애단계가 학생 또는 초기 경력",
      lambda d: _in(d, "demo_employment_status", "Retired"),
      lambda d: _in(d, "life_stage", ["Student", "Early career"]))

# ===========================================================================
# 6. career : career stage / entrepreneurship
# ===========================================================================
G = "career"
_rule("CA-H01", G, "hard", "경력 초입인데 장기 경력 — 경력 경로 '이제 시작'인데 11년+, 생애단계 '초기 경력'인데 20년+",
      lambda d: _ne(d, "years_experience") & _any(d, [lambda x: _in(x, "lifex_career_path_shape", "Just starting"),
                                                     lambda x: _in(x, "life_stage", "Early career")]),
      lambda d: _any(d, [lambda x: _in(x, "lifex_career_path_shape", "Just starting")
                         & _in(x, "years_experience", ["11-20", "20+"]),
                         lambda x: _in(x, "life_stage", "Early career") & _in(x, "years_experience", "20+")]))
_rule("CA-H02", G, "hard", "창업자·창업형 경력·창업이 주요 사건인데 창업 이력이 '고려 안 함/고려만 함'",
      lambda d: _any(d, [lambda x: _in(x, "seniority", "Founder"),
                         lambda x: _in(x, "lifex_career_path_shape", "Entrepreneurial"),
                         lambda x: _in(x, "major_life_events", "Started a business")]),
      lambda d: _in(d, "lifex_entrepreneurship_history", ["Never considered", "Considered it"]))
_rule("CA-S01", G, "soft", "경력 0~2년인데 리드/디렉터/VP/C-레벨 직급(창업자 제외)",
      lambda d: _in(d, "years_experience", "0-2"),
      lambda d: _in(d, "seniority", ["Lead / Principal", "Director", "VP", "C-suite"]))
_rule("CA-S02", G, "soft", "직무(임상/교육/농어업)에 해당하는 산업 경험이 전무(None)",
      lambda d: _in(d, "role_function", ["Clinical", "Teaching", "Agriculture / fishery"]),
      lambda d: _any(d, [
          lambda x: _in(x, "role_function", "Clinical") & _in(x, "ind_healthcare", "None"),
          lambda x: _in(x, "role_function", "Teaching") & _in(x, "ind_education", "None"),
          lambda x: _in(x, "role_function", "Agriculture / fishery") & _in(x, "ind_agriculture", "None")]))

# ===========================================================================
# 7. education
# ===========================================================================
G = "education"
def _is_student(df):
    """Currently a student (highest *completed* level may still be Secondary)."""
    return _oin(df, "demo_employment_status", "Student") | _oin(df, "life_stage", "Student")


_COLLEGE_TIERS = ["Top-tier research", "Mid-tier university", "Community college"]
_rule("ED-H01", G, "hard", "학력과 교육기관 유형 상충 — 무학/초졸(또는 비재학 고졸)인데 대학급 기관, 또는 대학 이상인데 기관 '해당 없음/독학'",
      lambda d: _ne(d, "highest_education") & _ne(d, "institution_tier"),
      lambda d: (_in(d, "highest_education", ["No formal", "Primary"]) & _in(d, "institution_tier", _COLLEGE_TIERS))
      | (_in(d, "highest_education", "Secondary") & _in(d, "institution_tier", _COLLEGE_TIERS) & ~_is_student(d))
      | (_in(d, "highest_education", EDU_COLLEGE) & _in(d, "institution_tier", ["Not applicable", "Self-taught"])))
_rule("ED-H02", G, "hard", "교육 여정이 '대학원 학위'인데 최종학력이 석사 미만",
      lambda d: _in(d, "lifex_education_journey", "Advanced degrees"),
      lambda d: _nin(d, "highest_education", EDU_GRAD))
_rule("ED-H03", G, "hard", "연구실적이 '학위논문만'인데 최종학력이 무학/초졸(또는 비재학 고졸)",
      lambda d: _in(d, "research_output", "Thesis only"),
      lambda d: _in(d, "highest_education", ["No formal", "Primary"])
      | (_in(d, "highest_education", "Secondary") & ~_is_student(d)))
_rule("ED-S01", G, "soft", "박사/박사후인데 연구실적 '없음' — 박사학위는 학위논문이 필수",
      lambda d: _in(d, "highest_education", ["Doctorate", "Postdoc"]),
      lambda d: _in(d, "research_output", "None"))
_rule("ED-S02", G, "soft", "학력과 교육기관·교육 여정 이례 — 대학원 학력+전문대, 전문학사+최상위 연구중심대, 박사+중퇴/독학/직업훈련 여정",
      lambda d: _ne(d, "highest_education"),
      lambda d: _any(d, [
          lambda x: _in(x, "highest_education", EDU_GRAD) & _in(x, "institution_tier", "Community college"),
          lambda x: _in(x, "highest_education", "Associate's") & _in(x, "institution_tier", "Top-tier research"),
          lambda x: _in(x, "highest_education", ["Doctorate", "Postdoc"])
          & _in(x, "lifex_education_journey", ["Dropped out", "Largely self-taught", "Vocational / trade"])]))
_rule("ED-S03", G, "soft", "무학/초졸인데 고등 학문 분야(입자물리·의학 세부과·헌법 등)에 전문가(Expert) 수준 지식",
      lambda d: _in(d, "highest_education", ["No formal", "Primary"]),
      lambda d: _any_col_in(d, ADV_FAM, "Expert"))

# ===========================================================================
# 8. religion
# ===========================================================================
G = "religion"
_rule("RE-H01", G, "hard", "무종교/무신론인데 종교성이 신앙실천·독실 또는 신앙 여정이 '평생 신앙/신앙 복귀'",
      lambda d: _in(d, "demo_religion_affiliation", NONREL),
      lambda d: _any(d, [lambda x: _in(x, "religiosity", ["Observant", "Devout"]),
                         lambda x: _in(x, "lifex_faith_journey", ["Lifelong faith", "Returned to faith"])]))
_rule("RE-H02", G, "hard", "신앙 여정이 '신앙 이탈' 또는 '평생 세속'인데 현재 신앙실천·독실",
      lambda d: _in(d, "lifex_faith_journey", ["Lapsed / left faith", "Always secular"]),
      lambda d: _in(d, "religiosity", ["Observant", "Devout"]))
_rule("RE-S01", G, "soft", "특정 종교 소속인데 종교성 '세속적' 또는 신앙 여정 '평생 세속' — 명목 신자는 'Nominal' 값이 별도 존재",
      lambda d: _in(d, "demo_religion_affiliation", AFFIL),
      lambda d: _any(d, [lambda x: _in(x, "religiosity", "Secular"),
                         lambda x: _in(x, "lifex_faith_journey", "Always secular")]))
_rule("RE-S02", G, "soft", "무종교·세속적인데 '영성/신앙'이 핵심 가치(Core value)",
      lambda d: _in(d, "demo_religion_affiliation", NONREL) & _in(d, "religiosity", "Secular"),
      lambda d: _in(d, "val_spirituality_faith", "Core value"))
_rule("RE-S03", G, "soft", "종교가 유교인데 유교적 생활방식을 따르지 않음(kr_confucian_way_of_life=No)",
      lambda d: _in(d, "demo_religion_affiliation", "Confucian"),
      lambda d: _in(d, "kr_confucian_way_of_life", "No"))
_rule("RE-S04", G, "soft", "식이 제한 사유가 '종교적'인데 무종교/무신론",
      lambda d: _in(d, "health_dietary_restriction", "Religious"),
      lambda d: _in(d, "demo_religion_affiliation", NONREL))

# ===========================================================================
# 9. health
# ===========================================================================
G = "health"
_rule("HE-H01", G, "hard", "만성질환 '없음'인데 장애 유형이 만성질환이거나 건강 여정이 '만성질환 관리 중'",
      lambda d: _in(d, "health_chronic_condition", "None"),
      lambda d: _any(d, [lambda x: _in(x, "demo_disability_status", "Chronic illness"),
                         lambda x: _in(x, "lifex_health_journey", "Manages chronic condition")]))
_rule("HE-H02", G, "hard", "전맹(Blind)인데 운전을 하거나 장애 '없음'",
      lambda d: _in(d, "health_vision", "Blind"),
      lambda d: _any(d, [lambda x: _in(x, "demo_driver_status", ["Daily driver", "Occasional driver"]),
                         lambda x: _in(x, "demo_disability_status", "No disability")]))
_rule("HE-H03", G, "hard", "체력 수준이 '운동선수'인데 운동을 전혀 하지 않음(Never)",
      lambda d: _in(d, "health_fitness_level", "Athlete"),
      lambda d: _in(d, "lstyle_exercise_freq", "Never"))
_rule("HE-H04", G, "hard", "신경다양성 두 항목 상충 — 한쪽은 신경전형(Neurotypical), 다른 쪽은 ADHD/자폐/난독",
      lambda d: _ne(d, "health_neurodivergence") & _ne(d, "neurotype"),
      lambda d: (_in(d, "health_neurodivergence", "Neurotypical") & _in(d, "neurotype", ["ADHD", "Autistic", "Dyslexic"]))
      | (_in(d, "health_neurodivergence", ["ADHD", "Autistic", "Dyslexic"]) & _in(d, "neurotype", "Neurotypical")))
_rule("HE-S01", G, "soft", "저시력(Low vision)인데 매일 운전 — 운전면허 시력 기준(양안 0.5 이상) 충족 어려움",
      lambda d: _in(d, "health_vision", "Low vision"),
      lambda d: _in(d, "demo_driver_status", "Daily driver"))
_rule("HE-S02", G, "soft", "기능 상태와 장애 여부 불일치 — 농·난청/보조기구/보조수단 필요인데 장애 없음, 감각장애인데 시력·청력 정상 등",
      lambda d: _ne(d, "demo_disability_status"),
      lambda d: _any(d, [
          lambda x: _in(x, "demo_disability_status", "No disability") & _in(x, "health_hearing", "Deaf / hard of hearing"),
          lambda x: _in(x, "demo_disability_status", "No disability") & _in(x, "health_mobility", "Uses mobility aid"),
          lambda x: _in(x, "demo_disability_status", "No disability") & _in(x, "health_dexterity", "Assistive needed"),
          lambda x: _in(x, "demo_disability_status", "Sensory") & _in(x, "health_vision", ["Normal", "Corrected"])
          & _in(x, "health_hearing", "Normal"),
          lambda x: _in(x, "demo_disability_status", "Physical") & _in(x, "health_mobility", "Full")
          & _in(x, "health_dexterity", "Full")]))
_rule("HE-S03", G, "soft", "전반적 건강 '최상'인데 심한 만성통증·다중 만성질환·이동보조기구 사용",
      lambda d: _in(d, "health_general_health", "Excellent"),
      lambda d: _any(d, [lambda x: _in(x, "health_pain_level", "Severe"),
                         lambda x: _in(x, "health_chronic_condition", "Multiple"),
                         lambda x: _in(x, "health_mobility", "Uses mobility aid")]))
_rule("HE-S04", G, "soft", "체력 수준과 운동 빈도 불일치 — 운동선수급인데 월1회/드물게, 좌식생활인데 매일 운동",
      lambda d: _in(d, "health_fitness_level", ["Athlete", "Sedentary"]),
      lambda d: (_in(d, "health_fitness_level", "Athlete") & _in(d, "lstyle_exercise_freq", ["Monthly", "Rarely"]))
      | (_in(d, "health_fitness_level", "Sedentary") & _in(d, "lstyle_exercise_freq", "Daily")))
_rule("HE-S05", G, "soft", "주의력 진단과 ADHD 여부 불일치 — 주의력 질환 '진단'인데 신경전형, ADHD인데 주의력 문제 '없음'",
      lambda d: _ne(d, "health_attention_condition"),
      lambda d: _any(d, [
          lambda x: _in(x, "health_attention_condition", "Diagnosed") & _in(x, "health_neurodivergence", "Neurotypical")
          & ~_oin(x, "neurotype", "ADHD"),
          lambda x: _in(x, "health_neurodivergence", "ADHD") & _in(x, "health_attention_condition", "None"),
          lambda x: _in(x, "neurotype", "ADHD") & _in(x, "health_attention_condition", "None")]))
_rule("HE-S06", G, "soft", "만성질환 '다수(Multiple)'인데 복용 약 '없음'",
      lambda d: _in(d, "health_chronic_condition", "Multiple"),
      lambda d: _in(d, "health_medication_use", "None"))

# ===========================================================================
# 10. life_events : defining life event vs the matching life-history attribute
# ===========================================================================
G = "life_events"
_rule("LE-H01", G, "hard", "주요 생애사건이 '사별(Bereavement)'인데 큰 상실 경험 없음",
      lambda d: _in(d, "major_life_events", "Bereavement"),
      lambda d: _in(d, "lifex_loss_experience", "No major loss"))
_rule("LE-H02", G, "hard", "주요 생애사건이 '건강 여정'인데 건강 여정이 '큰 사건 없음'",
      lambda d: _in(d, "major_life_events", "Health journey"),
      lambda d: _in(d, "lifex_health_journey", "No major events"))

# ===========================================================================
# 11. mental_health : PHQ-9 / GAD-7 bands vs wellbeing descriptions
# ===========================================================================
G = "mental_health"
_rule("MH-H01", G, "hard", "정신건강 '번영(Thriving)'인데 PHQ-9 중등도-중증 이상 우울 또는 GAD-7 중증 불안",
      lambda d: _in(d, "health_mental_health", "Thriving"),
      lambda d: _any(d, [lambda x: _in(x, "kr_depression_level", ["Moderately severe", "Severe"]),
                         lambda x: _in(x, "kr_anxiety_level", "Severe")]))
_rule("MH-S01", G, "soft", "정신건강 '위기(In crisis)'인데 우울·불안 증상이 모두 최소(Minimal)",
      lambda d: _in(d, "health_mental_health", "In crisis"),
      lambda d: _in(d, "kr_depression_level", "Minimal") & _in(d, "kr_anxiety_level", "Minimal"))
_rule("MH-S02", G, "soft", "PHQ-9 중증 우울인데 '매우 행복' 또는 삶에 '매우 만족' — 우울 기분·무쾌감과 상충",
      lambda d: _in(d, "kr_depression_level", "Severe"),
      lambda d: _any(d, [lambda x: _in(x, "kr_happiness", "Very happy"),
                         lambda x: _in(x, "kr_life_satisfaction", "Very satisfied")]))
_rule("MH-S03", G, "soft", "행복감과 삶의 만족도가 정반대 극단(매우 행복↔매우 불만족, 매우 불행↔매우 만족)",
      lambda d: _ne(d, "kr_happiness") & _ne(d, "kr_life_satisfaction"),
      lambda d: (_in(d, "kr_happiness", "Very happy") & _in(d, "kr_life_satisfaction", "Very dissatisfied"))
      | (_in(d, "kr_happiness", "Very unhappy") & _in(d, "kr_life_satisfaction", "Very satisfied")))
_rule("MH-S04", G, "soft", "정신건강 상태와 스트레스·행복·만족이 반대 극단(위기인데 스트레스 없음/매우 만족, 번영인데 매우 불행, 중증 증상인데 스트레스 없음)",
      lambda d: _any(d, [lambda x: _in(x, "health_mental_health", ["In crisis", "Thriving"]),
                         lambda x: _in(x, "kr_depression_level", "Severe"),
                         lambda x: _in(x, "kr_anxiety_level", "Severe")]),
      lambda d: _any(d, [
          lambda x: _in(x, "health_mental_health", "In crisis") & _in(x, "health_stress_level", "None"),
          lambda x: _in(x, "health_mental_health", "In crisis") & _in(x, "kr_life_satisfaction", "Very satisfied"),
          lambda x: _in(x, "health_mental_health", "Thriving") & _in(x, "kr_happiness", "Very unhappy"),
          lambda x: _in(x, "health_mental_health", "Thriving") & _in(x, "kr_life_satisfaction", "Very dissatisfied"),
          lambda x: (_in(x, "kr_depression_level", "Severe") | _oin(x, "kr_anxiety_level", "Severe"))
          & _in(x, "health_stress_level", "None"),
          lambda x: _in(x, "kr_anxiety_level", "Severe") & _in(x, "health_stress_level", "None")]))
_rule("MH-S05", G, "soft", "현재 중증 우울/불안(PHQ-9/GAD-7)인데 성격 측면 우울/불안이 '매우 낮음'",
      lambda d: _any(d, [lambda x: _in(x, "kr_depression_level", "Severe"),
                         lambda x: _in(x, "kr_anxiety_level", "Severe")]),
      lambda d: _any(d, [
          lambda x: _in(x, "kr_depression_level", "Severe") & _in(x, "bfi2_facet_depression", "Very low"),
          lambda x: _in(x, "kr_anxiety_level", "Severe") & _in(x, "bfi2_facet_anxiety", "Very low")]))

# ===========================================================================
# 12. mobility_travel
# ===========================================================================
G = "mobility_travel"
_rule("MO-H01", G, "hard", "출국 경험이 없는데(Never left country) 해외 이주·해외 거주·제3문화 성장·귀국자 이력",
      lambda d: _in(d, "lifex_travel_breadth", "Never left country"),
      lambda d: _any(d, [
          lambda x: _in(x, "lifex_geographic_mobility", "Moved internationally"),
          lambda x: _in(x, "lifex_cultural_exposure", "Third-culture kid"),
          lambda x: _in(x, "major_life_events", "Migration / immigration"),
          lambda x: _in(x, "lifex_immigration_generation", "Returnee / repatriate"),
          lambda x: _any_col_in(x, _cols(x, "cult_", exclude=("cult_south_korea",)), "Lived there")]))
_rule("MO-H02", G, "hard", "고향을 떠난 적 없는데 고향과의 관계가 '방문/귀향/단절', 해외 거주, 또는 대이동이 주요 사건",
      lambda d: _in(d, "lifex_geographic_mobility", "Never left hometown"),
      lambda d: _any(d, [
          lambda x: _in(x, "lifex_hometown_tie", ["Visits often", "Occasional return", "Rarely returns", "No connection"]),
          lambda x: _in(x, "lifex_travel_breadth", ["Lived abroad", "Global nomad"]),
          lambda x: _in(x, "major_life_events", "Major relocation")]))

# ===========================================================================
# 13. language
# ===========================================================================
G = "language"
_LIFETIME_MAX = {"One": 1, "Two": 2, "Three": 3}
_MULTI_MIN = {"Monolingual": 1, "Bilingual": 2, "Trilingual+": 3}
_rule("LA-H01", G, "hard", "평생 습득 언어 수(One/Two/Three)보다 회화 이상 가능한 언어(한국어 포함)가 더 많음",
      lambda d: _in(d, "lifex_languages_lifetime", list(_LIFETIME_MAX)),
      lambda d: (1 + _n_foreign(d, LANG["Conversational"])) > _s(d, "lifex_languages_lifetime").map(_LIFETIME_MAX).astype(float))
_rule("LA-H02", G, "hard", "평생 습득 언어 수보다 업무 사용 언어 수(multilingualism)가 더 많음(예: 1개 언어인데 Bilingual)",
      lambda d: _in(d, "lifex_languages_lifetime", list(_LIFETIME_MAX)) & _in(d, "multilingualism", list(_MULTI_MIN)),
      lambda d: _s(d, "multilingualism").map(_MULTI_MIN).astype(float)
      > _s(d, "lifex_languages_lifetime").map(_LIFETIME_MAX).astype(float))
_rule("LA-H03", G, "hard", "외국어 원어민인데 가정 언어는 한국어뿐이고 출국 경험도 없음 — 원어민 습득 경로가 없음",
      lambda d: _in(d, "demo_home_language", "Same as primary") & _in(d, "lifex_travel_breadth", "Never left country"),
      lambda d: _foreign_native(d))
_rule("LA-S01", G, "soft", "영어 능력 두 항목(english_proficiency vs lang_english)이 2단계 이상 차이",
      lambda d: _in(d, "english_proficiency", list(ENG)) & _in(d, "lang_english", list(LANG)),
      lambda d: (_rank(d, "english_proficiency", ENG) - _rank(d, "lang_english", LANG)).abs() >= 2)
_rule("LA-S02", G, "soft", "다중언어 자기평가와 개별 언어 수준 불일치(3개 국어 이상인데 외국어 회화 1개 이하, 단일언어인데 외국어 유창 등)",
      lambda d: _any(d, [lambda x: _ne(x, "multilingualism"), lambda x: _ne(x, "lifex_languages_lifetime")]),
      lambda d: _any(d, [
          lambda x: _in(x, "multilingualism", "Trilingual+") & (_n_foreign(x, 2) < 2),
          lambda x: _in(x, "multilingualism", "Bilingual") & (_n_foreign(x, 2) < 1),
          lambda x: _in(x, "multilingualism", "Monolingual") & (_n_foreign(x, 3) >= 1),
          lambda x: _in(x, "lifex_languages_lifetime", ["Four+", "Polyglot"]) & (_n_foreign(x, 2) < 2)]))
_rule("LA-S03", G, "soft", "이중언어/계승어/혼합언어 가정인데 회화 가능한 외국어가 하나도 없음",
      lambda d: _in(d, "demo_home_language", ["Bilingual home", "Heritage language", "Mixed languages"]),
      lambda d: _n_foreign(d, 2) == 0)
_rule("LA-S04", G, "soft", "외국어 또는 타국 문화가 원어민(Native) 수준 — 국내 출생 한국인에겐 다문화·해외 성장 외 극히 드묾",
      lambda d: _T(d),
      lambda d: _any(d, [_foreign_native,
                         lambda x: _any_col_in(x, _cols(x, "cult_", exclude=("cult_south_korea",)), "Native")]))
_rule("LA-S05", G, "soft", "대졸 이상인데 영어 '전혀 못함(None)' — 한국은 중학교부터 대학까지 영어 필수 이수",
      lambda d: _in(d, "highest_education", EDU_BA_PLUS),
      lambda d: _any(d, [lambda x: _in(x, "english_proficiency", "None"),
                         lambda x: _in(x, "lang_english", "None")]))

# ===========================================================================
# 14. military (Korean conscription context)
# ===========================================================================
G = "military"
_rule("MI-H01", G, "hard", "군 복무 이력 '없음'인데 퇴역·현역·예비군이거나 군 복무가 주요 생애사건",
      lambda d: _in(d, "lifex_military_history", "Never served"),
      lambda d: _any(d, [lambda x: _in(x, "demo_veteran_status", ["Veteran", "Active duty", "Reserve / guard"]),
                         lambda x: _in(x, "major_life_events", "Military service")]))
_rule("MI-H02", G, "hard", "현역 군인인데 65세 이상(한국군 계급정년 최대 63세) 또는 취업상태 '은퇴'",
      lambda d: _in(d, "demo_veteran_status", "Active duty"),
      lambda d: _any(d, [lambda x: _age_ge(x, "65-74"),
                         lambda x: _in(x, "demo_employment_status", "Retired")]))
_rule("MI-H03", G, "hard", "19~24세인데 군 이력이 '직업군인 퇴역(Career veteran)'",
      lambda d: _age_in(d, "18-24") & _ne(d, "lifex_military_history"),
      lambda d: _in(d, "lifex_military_history", "Career veteran"))
_rule("MI-S01", G, "soft", "65세 미만 참전용사 — 한국의 대규모 참전(6·25, 베트남전 1964~73)은 현재 65세 이상",
      lambda d: _in(d, "lifex_military_history", "Combat veteran") & _adult(d),
      lambda d: _age_le(d, "55-64"))

# ===========================================================================
# 15. income
# ===========================================================================
G = "income"
_rule("IN-H01", G, "hard", "가구소득 밴드와 사회경제 계층이 정반대 극단(<$25k↔High income, $200k+↔Low income)",
      lambda d: _in(d, "demo_household_income", list(INCOME)) & _in(d, "socioeconomic_band", list(SES)),
      lambda d: (_rank(d, "demo_household_income", INCOME) - _rank(d, "socioeconomic_band", SES)).abs() >= 4)
_rule("IN-S01", G, "soft", "가구소득 밴드와 사회경제 계층이 3단계 차이(예: <$25k인데 Upper-middle)",
      lambda d: _in(d, "demo_household_income", list(INCOME)) & _in(d, "socioeconomic_band", list(SES)),
      lambda d: (_rank(d, "demo_household_income", INCOME) - _rank(d, "socioeconomic_band", SES)).abs() == 3)
_rule("IN-S02", G, "soft", "주관적 계층과 사회경제 계층이 정반대 극단(하층↔고소득, 상층↔저소득)",
      lambda d: _in(d, "kr_subjective_class", list(SUBJ_CLASS)) & _in(d, "socioeconomic_band", list(SES)),
      lambda d: (_rank(d, "kr_subjective_class", SUBJ_CLASS) - _rank(d, "socioeconomic_band", SES)).abs() >= 4)

# ===========================================================================
# 16. lifestyle
# ===========================================================================
G = "lifestyle"
_PHYSICAL_SPORT_EXCLUDE = ("sport_esports", "sport_darts")
_rule("LS-H01", G, "hard", "안드로이드/리눅스 생태계인데 애플 전용 앱(iMessage 주 메신저, Safari 주 브라우저) 사용",
      lambda d: _in(d, "lstyle_device_ecosystem", ["Android/Google", "Linux"]),
      lambda d: _any(d, [lambda x: _in(x, "lstyle_primary_messenger", "iMessage"),
                         lambda x: _in(x, "lstyle_browser", "Safari")]))
_rule("LS-H02", G, "hard", "은행 계좌가 없는데(Unbanked) 체크/신용카드가 주 결제수단 — 국내 카드는 결제계좌 필수",
      lambda d: _in(d, "lstyle_banking_style", "Unbanked"),
      lambda d: _in(d, "lstyle_payment_pref", ["Debit card", "Credit card"]))
_rule("LS-H03", G, "hard", "주 소셜 플랫폼이 LinkedIn인데 LinkedIn 활동이 '비활동(Inactive)'",
      lambda d: _in(d, "lstyle_primary_social", "LinkedIn"),
      lambda d: _in(d, "linkedin_activity", "Inactive"))
_rule("LS-H04", G, "hard", "운전을 하는데(매일/가끔) 운전 기술이 '없음'",
      lambda d: _in(d, "demo_driver_status", ["Daily driver", "Occasional driver"]),
      lambda d: _in(d, "skill_driving", "None"))
_rule("LS-S01", G, "soft", "하루 화면 사용 2시간 미만인데 주 16시간 이상 스트리밍(하루 2.3시간+) — 산술적으로 상충",
      lambda d: _in(d, "lstyle_screen_time", "<2 hrs"),
      lambda d: _in(d, "lstyle_streaming_hours", "16+"))
_rule("LS-S02", G, "soft", "운동을 전혀 안 하는데(Never) 신체 스포츠를 직접 함(Play)",
      lambda d: _in(d, "lstyle_exercise_freq", "Never"),
      lambda d: _any_col_in(d, _cols(d, "sport_", exclude=_PHYSICAL_SPORT_EXCLUDE), "Play"))
_rule("LS-S03", G, "soft", "술을 전혀 안 마시는데 와인/수제맥주 열정·소믈리에 전문지식·와인/맥주 양조 취미",
      lambda d: _in(d, "lstyle_alcohol_use", "Never"),
      lambda d: _any(d, [lambda x: _in(x, "topic_wine", "Passionate"),
                         lambda x: _in(x, "topic_craft_beer", "Passionate"),
                         lambda x: _in(x, "fam_sommelier_knowledge", "Expert"),
                         lambda x: _in(x, "hob_winemaking", "Avid"),
                         lambda x: _in(x, "hob_home_brewing", "Avid")]))
_rule("LS-S04", G, "soft", "저축/소비 성향 두 항목이 정반대 극단(Hard saver↔Splurger, Free spender↔Frugal saver)",
      lambda d: _ne(d, "pref_save_vs_spend") & _ne(d, "lstyle_frugality"),
      lambda d: (_in(d, "pref_save_vs_spend", "Hard saver") & _in(d, "lstyle_frugality", "Splurger"))
      | (_in(d, "pref_save_vs_spend", "Free spender") & _in(d, "lstyle_frugality", "Frugal saver")))
_rule("LS-S05", G, "soft", "채식/비건인데 비거니즘에 반대하거나 해당 채식 요리를 회피",
      lambda d: _in(d, "lstyle_diet_type", ["Vegan", "Vegetarian"]),
      lambda d: _any(d, [lambda x: _in(x, "att_veganism", "Opposed"),
                         lambda x: _in(x, "lstyle_diet_type", "Vegan") & _in(x, "cuis_vegan", "Avoid"),
                         lambda x: _in(x, "lstyle_diet_type", "Vegetarian") & _in(x, "cuis_vegetarian", "Avoid")]))

# ===========================================================================
# 17. temporal_2020 : products that did not exist at the reference date
# ===========================================================================
G = "temporal_2020"
_rule("TM-H01", G, "hard", "2020년 11월 기준 미출시 AI 제품 사용 이력(ChatGPT 2022.11, Claude 2023.3, Midjourney 2022, GitHub Copilot 2021.6)",
      lambda d: _any(d, [lambda x, c=c: _ne(x, c) for c in POST_2020_AI_TOOLS]),
      lambda d: _any(d, [lambda x, c=c: _nin(x, c, "Never used") for c in POST_2020_AI_TOOLS]))
_rule("TM-S01", G, "soft", "2020년 11월 기준으로 코딩 에이전트 주간 이상 사용 또는 개발 작업 대부분이 AI 주도 — LLM 코딩 도구 대중화 이전",
      lambda d: _any(d, [lambda x: _ne(x, "coding_agent_usage_frequency")]
                     + [lambda x, c=c: _ne(x, c) for c in _cols(d, "ai_task_")]),
      lambda d: _any(d, [lambda x: _in(x, "coding_agent_usage_frequency", ["Daily", "Weekly"])]
                     + [lambda x, c=c: _in(x, c, "Currently mostly AI-assisted") for c in _cols(d, "ai_task_")]))

# ===========================================================================
# 18. developer
# ===========================================================================
G = "developer"
_rule("DV-H01", G, "hard", "코딩 에이전트를 사용하는데(일/주/월) 실패 허용도 항목은 '에이전트 미사용'",
      lambda d: _in(d, "coding_agent_usage_frequency", ["Daily", "Weekly", "Monthly or infrequent"]),
      lambda d: _in(d, "coding_agent_failure_tolerance", "Does not use agents"))
_rule("DV-H02", G, "hard", "개발 AI 도구를 '사용한 적 없음/회피'인데 코딩 에이전트를 쓰거나 현재 AI 보조로 작업",
      lambda d: _in(d, "coding_ai_usage_frequency", ["Never used", "Actively avoids"]),
      lambda d: _any(d, [lambda x: _in(x, "coding_agent_usage_frequency", ["Daily", "Weekly", "Monthly or infrequent"])]
                     + [lambda x, c=c: _in(x, c, AI_TASK_CURRENT) for c in _cols(d, "ai_task_")]))
_rule("DV-S01", G, "soft", "전문 개발자·OSS 기여자·언어 전문가인데 코딩 기술 '없음', 또는 전문 개발자인데 코딩 활동 없음",
      lambda d: _any(d, [lambda x: _ne(x, "skill_coding"), lambda x: _ne(x, "open_source_activity_level")]),
      lambda d: _any(d, [
          lambda x: _in(x, "dev_professional_status", "Professional developer") & _in(x, "skill_coding", "None"),
          lambda x: _in(x, "open_source_activity_level", ["Maintainer / core contributor", "Regular contributor"])
          & _in(x, "skill_coding", "None"),
          lambda x: _any_col_in(x, _prog_level_cols(x), "Expert") & _in(x, "skill_coding", "None"),
          lambda x: _in(x, "dev_professional_status", "Professional developer")
          & _in(x, "open_source_activity_level", "No coding activity")]))
_rule("DV-S02", G, "soft", "코딩 기술이 고급/마스터인데 모든 프로그래밍 언어가 'None'",
      lambda d: _in(d, "skill_coding", ["Master", "Advanced"]),
      lambda d: (_count_in(d, _prog_level_cols(d), "None") == len(_prog_level_cols(d)))
      & (len(_prog_level_cols(d)) >= 5))
_TOOL_PROG = [("tool_python", "prog_python"), ("tool_sql", "prog_sql"),
              ("tool_r", "prog_r"), ("tool_matlab", "prog_matlab")]


def _tool_prog_hit(df):
    parts = []
    for t, p in _TOOL_PROG:
        parts.append(lambda x, t=t, p=p: (_in(x, t, ["Power user", "Regular"]) & _in(x, p, "None"))
                     | (_in(x, p, ["Expert", "Proficient"]) & _in(x, t, "Never used")))
    return _any(df, parts)


def _tool_prog_scope(df):
    return _any(df, [lambda x, t=t, p=p: _ne(x, t) & _ne(x, p) for t, p in _TOOL_PROG])


_rule("DV-S03", G, "soft", "같은 언어의 도구 사용과 프로그래밍 능력 상충(Python/SQL/R/MATLAB 파워유저인데 언어 능력 없음, 또는 그 반대)",
      _tool_prog_scope, _tool_prog_hit)

# ===========================================================================
# 19. personality : same concept measured with different instruments (soft)
# ===========================================================================
G = "personality"
_DOMAIN_KO = {"O": "개방성", "C": "성실성", "E": "외향성", "A": "친화성", "N": "부정적 정서성"}
for _k in ["O", "C", "E", "A", "N"]:
    _rule(f"PE-S0{['O', 'C', 'E', 'A', 'N'].index(_k) + 1}", G, "soft",
          f"BFI-2 {_DOMAIN_KO[_k]} 영역 점수가 모든 하위측면(BFI-2 3개 또는 IPIP-NEO 6개)보다 3단계 이상 높거나 낮음 — 영역은 측면들의 평균",
          _domain_scope(_k), _domain_hit(_k))
_rule("PE-S06", G, "soft", "동일 구성개념의 BFI-2 측면과 IPIP-NEO 측면이 정반대 극단(Very high↔Very low, 예: 불안 vs 불안)",
      _pairs_scope(BF_FACET_PAIRS), _pairs_hit(BF_FACET_PAIRS))
_rule("PE-S07", G, "soft", "성격 강점(호기심·창의성·친절·겸손·절제 등)이 '대표 강점'인데 대응 Big Five 측면이 '매우 낮음'(또는 그 반대)",
      _pairs_scope(CHAR_PAIRS), _pairs_hit(CHAR_PAIRS))
_rule("PE-S08", G, "soft", "대인·의사소통 자기평가(온정·지배성·주장성·낙관 등)가 대응 성격 지표와 정반대 극단(미루기↔자기통제는 동일 극단)",
      _pairs_scope(COMM_PAIRS), _pairs_hit(COMM_PAIRS))
_rule("PE-S09", G, "soft", "외향성 지표 상충 — MBTI I/E, 사회적 배터리, BFI-2 외향성, 대규모 모임 선호가 정반대 극단",
      lambda d: _any(d, [lambda x: _ne(x, "mbti_type"), lambda x: _ne(x, "lstyle_social_battery")]),
      lambda d: _any(d, [
          lambda x: (_mbti_letter(x, 0) == "I") & _in(x, "bfi2_domain_extraversion", "Very high"),
          lambda x: (_mbti_letter(x, 0) == "E") & _in(x, "bfi2_domain_extraversion", "Very low"),
          lambda x: _in(x, "lstyle_social_battery", "Strong introvert") & _in(x, "bfi2_domain_extraversion", "Very high"),
          lambda x: _in(x, "lstyle_social_battery", "Strong extrovert") & _in(x, "bfi2_domain_extraversion", "Very low"),
          lambda x: _in(x, "lstyle_social_battery", "Strong introvert") & (_mbti_letter(x, 0) == "E"),
          lambda x: _in(x, "lstyle_social_battery", "Strong extrovert") & (_mbti_letter(x, 0) == "I"),
          lambda x: _in(x, "lstyle_social_battery", "Strong introvert")
          & _in(x, "pref_big_group_vs_one_on_one", "Loves big groups"),
          lambda x: _in(x, "lstyle_social_battery", "Strong extrovert")
          & _in(x, "pref_big_group_vs_one_on_one", "One-on-one only")]))
_rule("PE-S10", G, "soft", "MBTI 감각(S)인데 개방성 '매우 높음', 또는 직관(N)인데 개방성 '매우 낮음'",
      lambda d: _ne(d, "mbti_type") & _ne(d, "bfi2_domain_open_mindedness"),
      lambda d: ((_mbti_letter(d, 1) == "S") & _in(d, "bfi2_domain_open_mindedness", "Very high"))
      | ((_mbti_letter(d, 1) == "N") & _in(d, "bfi2_domain_open_mindedness", "Very low")))


def _dominant_hit(df):
    dom = _s(df, "dominant_trait")
    _reg("dominant_trait", DOMINANT_MAP.keys())
    out = _F(df)
    for val, col in DOMINANT_MAP.items():
        if col in df.columns:
            out = out | ((dom == val) & _in(df, col, ["Low", "Very low"]))
    return out


_rule("PE-S11", G, "soft", "가장 두드러진 특질(dominant_trait)로 지목된 Big Five 영역이 '낮음/매우 낮음'",
      _req_any(list(DOMINANT_MAP.values()), lambda d: _in(d, "dominant_trait", list(DOMINANT_MAP))),
      _dominant_hit)


def _risk_parts(df):
    parts = []
    if "risk_tolerance" in df.columns:
        t = _rank(df, "risk_tolerance", RISK)
        present = [c for c in DOSPERT if c in df.columns]
        if present:
            O = pd.concat([_rank(df, c, L5) for c in present], axis=1)
            sc = t.notna() & O.notna().all(axis=1)
            parts.append((sc, sc & _gap_outside(t, O)))
        if "att_risk_taking" in df.columns:
            a = _rank(df, "att_risk_taking", ATT)
            sc = t.notna() & a.notna()
            parts.append((sc, ((t == 5) & (a == 1)) | ((t == 1) & (a == 5))))
    if not parts:
        raise MissingColumn("risk_tolerance")
    return parts


_rule("PE-S12", G, "soft", "전반적 위험감수 성향이 모든 DOSPERT 영역보다 3단계 이상 높거나 낮음, 또는 위험감수 태도와 정반대 극단",
      lambda d: _or_all([sc for sc, _ in _risk_parts(d)], d),
      lambda d: _or_all([h for _, h in _risk_parts(d)], d))
_rule("PE-S13", G, "soft", "종결욕구↔모호성 내성, 결정 속도↔결정 스타일이 상충(종결욕구 매우 높음+모호성 내성 매우 높음, 즉흥 결정+숙고형 등)",
      lambda d: _any(d, [lambda x: _ne(x, "need_for_closure"), lambda x: _ne(x, "cog_decision_speed")]),
      lambda d: _any(d, [
          lambda x: _in(x, "need_for_closure", "Very high") & _in(x, "cog_ambiguity_tolerance", "Very high"),
          lambda x: _in(x, "need_for_closure", "Very low") & _in(x, "cog_ambiguity_tolerance", "None"),
          lambda x: _in(x, "cog_decision_speed", "Snap decisions") & _in(x, "decision_style", "Deliberative")]))

# ===========================================================================
# 20. values : Schwartz value levels vs value-importance items vs top value
# ===========================================================================
G = "values"
for _rid, (_label, _items) in SCHWARTZ_GROUPS.items():
    _rule(_rid, G, "soft",
          f"Schwartz {_label} 가치 수준이 관련 가치 중요도 문항 전부보다 3단계 이상 높거나 낮음(예: 전통 Very high인데 val_tradition Minor 이하)",
          (lambda items: (lambda d: _schwartz_eval(d, items)[0]))(_items),
          (lambda items: (lambda d: _schwartz_eval(d, items)[1]))(_items))


def _priority_eval(df):
    pr = _s(df, "values_priority")
    _reg("values_priority", PRIORITY_MAP.keys())
    scope, hit = _F(df), _F(df)
    any_col = False
    for val, (item, sch) in PRIORITY_MAP.items():
        is_v = pr == val
        sub = []
        if item in df.columns:
            sub.append(_in(df, item, ["Minor", "Irrelevant"]))
        if sch in df.columns:
            sub.append(_in(df, sch, ["Low", "Very low"]))
        if not sub:
            continue
        any_col = True
        scope = scope | is_v
        h = sub[0]
        for s_ in sub[1:]:
            h = h | s_
        hit = hit | (is_v & h)
    if not any_col:
        raise MissingColumn("val_* / schwartz_value_*")
    return scope, hit


_rule("VA-S05", G, "soft", "최우선 가치(values_priority)에 대응하는 가치 문항이 '부차적/무관' 또는 Schwartz 가치가 '낮음/매우 낮음'",
      lambda d: _priority_eval(d)[0], lambda d: _priority_eval(d)[1])

# ===========================================================================
# 21. breadth / expertise
# ===========================================================================
G = "expertise"
_rule("EX-S01", G, "soft", "전문성 범위가 비현실적으로 넓음 — 지식 분야 Expert 20개+, 기술 Master 15개+, 프로그래밍 언어 Expert 10개+ 중 하나",
      _req_any(lambda d: _cols(d, "fam_") + _cols(d, "skill_") + _cols(d, "prog_"), _T),
      lambda d: _any(d, [lambda x: _count_in(x, _cols(x, "fam_"), "Expert") >= 20,
                         lambda x: _count_in(x, _cols(x, "skill_"), "Master") >= 15,
                         lambda x: _count_in(x, _prog_level_cols(x), "Expert") >= 10]))


def _specialty_hit(df):
    dom = _s(df, "domain").to_numpy()
    spec = _s(df, "subject_specialty").to_numpy()
    _reg("subject_specialty", SPECIALTY_DOMAINS.keys())
    for v in SPECIALTY_DOMAINS.values():
        _reg("domain", v)
    out = [bool(sp in SPECIALTY_DOMAINS and dm in ALL_DOMAINS and dm not in SPECIALTY_DOMAINS[sp])
           for dm, sp in zip(dom, spec)]
    return pd.Series(out, index=df.index)


_rule("EX-S02", G, "soft", "세부 전문분야(subject_specialty)가 주 분야(domain)와 동떨어짐(예: 분야 'Hospitality'인데 'Constitutional law')",
      lambda d: _in(d, "domain", ALL_DOMAINS) & _in(d, "subject_specialty", list(SPECIALTY_DOMAINS)),
      _specialty_hit)


# programming-language level columns (prog_*: Expert/Proficient/Familiar/None scale)
def _prog_level_cols(df):
    return _cols(df, "prog_")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _as_bool(x, index):
    s = pd.Series(x, index=index) if not isinstance(x, pd.Series) else x
    return s.reindex(index).fillna(False).astype(bool)


def violation_matrix(df: pd.DataFrame, rules=None):
    """Return (hits, scopes, status): bool DataFrames rows x rule-id, and a dict id -> status."""
    rules = RULES if rules is None else rules
    hits, scopes, status = {}, {}, {}
    _CACHE.update(active=True, df_id=id(df), cols={})
    try:
        for r in rules:
            try:
                sc = _as_bool(r["scope"](df), df.index)
                ht = _as_bool(r["hit"](df), df.index) & sc
                scopes[r["id"]], hits[r["id"]] = sc, ht
                status[r["id"]] = "ok"
            except MissingColumn as e:
                status[r["id"]] = f"skipped (missing column: {e.args[0] if e.args else e})"
            except Exception as e:  # never crash the whole run
                status[r["id"]] = f"error: {type(e).__name__}: {e}"
    finally:
        _CACHE.update(active=False, df_id=None, cols={})
    return pd.DataFrame(hits, index=df.index), pd.DataFrame(scopes, index=df.index), status


def run(df: pd.DataFrame, exclude_groups_from_any_hard=None):
    """Evaluate all rules.

    Returns (table, any_hard_rate):
      table          one row per rule: id, group, severity, description, n_scope,
                     n_hit, violation_rate, status (skipped/error rules have NaN rate)
      any_hard_rate  fraction of all rows violating >= 1 hard rule
                     (optionally ignoring groups in exclude_groups_from_any_hard,
                      e.g. {'temporal_2020'} if personas are not meant to be time-anchored)
    """
    excl = set(exclude_groups_from_any_hard or ())
    hits, scopes, status = violation_matrix(df)
    rows = []
    hard_any = pd.Series(False, index=df.index)
    for r in RULES:
        rid = r["id"]
        if rid in hits.columns:
            n_scope = int(scopes[rid].sum())
            n_hit = int(hits[rid].sum())
            rate = (n_hit / n_scope) if n_scope else np.nan
            if r["severity"] == "hard" and r["group"] not in excl:
                hard_any = hard_any | hits[rid]
        else:
            n_scope, n_hit, rate = 0, 0, np.nan
        rows.append(dict(id=rid, group=r["group"], severity=r["severity"],
                         description=r["description"], n_scope=n_scope, n_hit=n_hit,
                         violation_rate=rate, status=status.get(rid, "not run")))
    table = pd.DataFrame(rows, columns=["id", "group", "severity", "description", "n_scope",
                                        "n_hit", "violation_rate", "status"])
    any_hard_rate = float(hard_any.mean()) if len(df) else float("nan")
    table.attrs["any_hard_rate"] = any_hard_rate
    return table, any_hard_rate


def referenced_values():
    """(column -> set of values) referenced by the rules (filled while rules run)."""
    return {k: set(v) for k, v in _REG.items()}


__all__ = ["RULES", "run", "violation_matrix", "referenced_values", "MissingColumn"]
