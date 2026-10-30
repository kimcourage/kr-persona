"""Consistency rules (set R2) for synthetic South Korean adult personas.

Written only from the attribute catalog (ids, labels, descriptions, value lists)
and real-world facts about Korea.  Time assumptions follow the catalog:
  * composition / birth-year attributes use a 2020 reference (birth = 2020 - age);
  * technology, products and AI tools describe life today (2026), so release
    dates of products are never treated as contradictions.
Population: domestic-born Korean citizens aged 19+ (age_bracket '18-24' = 19-24).

Public API
  RULES                      list of rule dicts (id, group, severity, description, scope, hit)
  MissingColumn              KeyError subclass raised by _s for absent columns
  violation_matrix(df, rules=None) -> (hits, scopes, status)
  run(df, rules=None)        -> (table, any_hard_rate)
  check_values(catalog_path) -> list of problems (empty == every id/value exists)

Every column is read through _s(df, col) only (looked up at call time, so it
can be wrapped from outside to discover the columns each rule reads).
Every value string a rule compares against is registered in _REF via _V/_VC,
so check_values can verify exact spelling against the catalog.

These rules also informed generator changes; evaluation independence is limited.
See docs/08_모순_규칙.md.
"""

import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# accessor
# ---------------------------------------------------------------------------
class MissingColumn(KeyError):
    """Raised when a rule needs a column that the frame does not have."""


def _s(df, col):
    if col not in df.columns:
        raise MissingColumn(col)
    return df[col].astype(str).str.strip()


# ---------------------------------------------------------------------------
# value registry (for check_values)
# ---------------------------------------------------------------------------
_REF = {}


def _V(col, *vals):
    """Register values of one column and return them as a list."""
    _REF.setdefault(col, set()).update(vals)
    return list(vals)


def _VC(cols, *vals):
    """Register the same values for several columns."""
    for c in cols:
        _V(c, *vals)
    return list(vals)


def _cols(*cols):
    """Register columns that are read (values registered elsewhere or none)."""
    for c in cols:
        _REF.setdefault(c, set())
    return list(cols)


# ---------------------------------------------------------------------------
# array helpers (all return numpy bool / numeric arrays, never NaN booleans)
# ---------------------------------------------------------------------------
def _T(df):
    return np.ones(len(df), dtype=bool)


def _F(df):
    return np.zeros(len(df), dtype=bool)


def _in(df, col, vals):
    return np.array(_s(df, col).isin(list(vals)).to_numpy(dtype=bool), dtype=bool, copy=True)


def _eq(df, col, val):
    return np.array((_s(df, col) == val).to_numpy(dtype=bool), dtype=bool, copy=True)


def _any_in(df, cols, vals):
    out = _F(df)
    for c in cols:
        out |= _in(df, c, vals)
    return out


def _all_in(df, cols, vals):
    out = _T(df)
    for c in cols:
        out &= _in(df, c, vals)
    return out


def _lvl(df, col, mapping):
    """Ordinal level as float (NaN for values outside the mapping)."""
    return np.array(_s(df, col).map(mapping).to_numpy(dtype=float), dtype=float, copy=True)


def _ser(df, x):
    a = x.to_numpy() if isinstance(x, pd.Series) else np.asarray(x)
    if a.ndim == 0:
        a = np.full(len(df), bool(a))
    if a.shape[0] != len(df):
        raise ValueError("rule returned %d values for %d rows" % (a.shape[0], len(df)))
    if a.dtype != bool:
        a = pd.Series(a).fillna(False).to_numpy().astype(bool)
    return pd.Series(a, index=df.index, dtype=bool)


RULES = []


def _rule(rid, group, severity, description, scope, hit):
    assert severity in ("hard", "soft")
    RULES.append({
        "id": rid,
        "group": group,
        "severity": severity,
        "description": description,
        "scope": (lambda df, f=scope: _ser(df, f(df))),
        "hit": (lambda df, f=hit: _ser(df, f(df))),
    })


# ===========================================================================
# shared value constants
# ===========================================================================
AGE = "age_bracket"
AGE_ALL = _V(AGE, "Under 5", "5-12", "13-17", "18-24", "25-34", "35-44",
             "45-54", "55-64", "65-74", "75-84", "85+")
AGE_U19 = _V(AGE, "Under 5", "5-12", "13-17")
AGE_ADULT = _V(AGE, "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+")
A1824 = "18-24"
A2534 = "25-34"
A3544 = "35-44"
AGE_LE34 = _V(AGE, "18-24", "25-34")
AGE_LT65 = _V(AGE, "18-24", "25-34", "35-44", "45-54", "55-64")
AGE_55P = _V(AGE, "55-64", "65-74", "75-84", "85+")
AGE_65P = _V(AGE, "65-74", "75-84", "85+")
AGE_75P = _V(AGE, "75-84", "85+")

GENDER = "gender_identity"
WOMAN = _V(GENDER, "Woman")[0]

EMP = "demo_employment_status"
EMP_ALL = _V(EMP, "Full-time", "Part-time", "Self-employed", "Gig / freelance", "Student",
             "Unemployed", "Retired", "Homemaker", "Unpaid family worker")
EMP_WORKING = _V(EMP, "Full-time", "Part-time", "Self-employed", "Gig / freelance",
                 "Unpaid family worker")
EMP_EMPLOYEE = _V(EMP, "Full-time", "Part-time")
EMP_FT = _V(EMP, "Full-time")[0]
EMP_RETIRED = _V(EMP, "Retired")[0]
EMP_STUDENT = _V(EMP, "Student")[0]
EMP_UNEMP = _V(EMP, "Unemployed")[0]

LIFE = "life_stage"
LIFE_STUDENT = _V(LIFE, "Student")[0]
LIFE_EARLY = _V(LIFE, "Early career")[0]
LIFE_YOUNGKIDS = _V(LIFE, "Parent of young kids")[0]
LIFE_EMPTY = _V(LIFE, "Empty nester")[0]
LIFE_RETIRE = _V(LIFE, "Retirement")[0]

SEN = "seniority"
SEN_EXEC = _V(SEN, "Director", "VP", "C-suite")
SEN_INTERN = _V(SEN, "Student / intern")[0]
SEN_FOUNDER = _V(SEN, "Founder")[0]
SEN_RETIRED = _V(SEN, "Retired")[0]

YEXP = "years_experience"
YEXP_0_2 = _V(YEXP, "0-2")[0]
YEXP_11_20 = _V(YEXP, "11-20")[0]
YEXP_20P = _V(YEXP, "20+")[0]

ENTRE = "lifex_entrepreneurship_history"
ENTRE_NONE = _V(ENTRE, "Never considered", "Considered it")
ENTRE_SERIAL = _V(ENTRE, "Serial founder", "Exited a company")

IND_COLS = _cols(
    "ind_retail", "ind_e_commerce", "ind_automotive", "ind_oil_gas", "ind_utilities",
    "ind_construction", "ind_travel_tourism", "ind_restaurants", "ind_agriculture",
    "ind_food_beverage", "ind_pharmaceuticals", "ind_biotech", "ind_telecommunications",
    "ind_entertainment", "ind_publishing", "ind_advertising", "ind_government", "ind_defense",
    "ind_nonprofit", "ind_transportation", "ind_shipping", "ind_chemicals", "ind_apparel",
    "ind_consumer_electronics", "ind_semiconductors", "ind_legal_services", "ind_consulting",
    "ind_fitness_wellness", "ind_beauty_cosmetics", "ind_finance", "ind_banking",
    "ind_accounting", "ind_education", "ind_aerospace", "ind_fine_art", "ind_healthcare",
    "ind_sports", "ind_gaming", "ind_technology", "ind_marketing_agencies", "ind_logistics",
    "ind_manufacturing", "ind_real_estate", "ind_insurance", "ind_hospitality", "ind_textiles",
    "ind_media", "ind_music", "ind_energy", "ind_mining")
IND_VETERAN = _VC(IND_COLS, "Veteran")[0]

# --- family -----------------------------------------------------------------
MAR = "demo_marital_status"
MAR_ALL = _V(MAR, "Single", "In a relationship", "Engaged", "Married", "Domestic partnership",
             "Separated", "Divorced", "Widowed")
MAR_PARTNERED = _V(MAR, "In a relationship", "Engaged", "Married", "Domestic partnership")
MAR_SINGLE = _V(MAR, "Single")[0]
MAR_WIDOWED = _V(MAR, "Widowed")[0]
MAR_WID_DIV = _V(MAR, "Widowed", "Divorced")

RLEN = "demo_relationship_length"
RLEN_NONE = _V(RLEN, "Not in one")[0]
RLEN_15P = _V(RLEN, "15+ years")[0]

RHIST = "lifex_relationship_history"
RHIST_EVER_MARRIED = _V(RHIST, "Married once", "Remarried", "Widowed")
RHIST_WIDOWED = _V(RHIST, "Widowed")[0]
RHIST_REM_WID = _V(RHIST, "Remarried", "Widowed")

KIDS = "demo_children_count"
KIDS_ALL = _V(KIDS, "None", "Expecting", "1 child", "2 children", "3+ children", "Adult children")
KIDS_NONE = _V(KIDS, "None")[0]
KIDS_HAS = _V(KIDS, "1 child", "2 children", "3+ children", "Adult children")
KIDS_ADULT = _V(KIDS, "Adult children")[0]
KIDS_EXPECT = _V(KIDS, "Expecting")[0]

PAR = "demo_parental_status"
PAR_ALL = _V(PAR, "Not a parent", "New parent", "Parent of minors", "Parent of adults",
             "Grandparent", "Step / foster parent")
PAR_NOT = _V(PAR, "Not a parent")[0]
PAR_YES = _V(PAR, "New parent", "Parent of minors", "Parent of adults", "Grandparent")
PAR_ADULT_GRAND = _V(PAR, "Parent of adults", "Grandparent")
PAR_GRAND = _V(PAR, "Grandparent")[0]
PAR_NEW = _V(PAR, "New parent")[0]

JOUR = "lifex_parenting_journey"
JOUR_ALL = _V(JOUR, "No children", "Young children", "Teenagers", "Grown children",
              "Empty nester", "Raising grandchildren")
JOUR_NONE = _V(JOUR, "No children")[0]
JOUR_KIDS = _V(JOUR, "Young children", "Teenagers", "Grown children", "Empty nester",
               "Raising grandchildren")
JOUR_YOUNG = _V(JOUR, "Young children")[0]
JOUR_AT_HOME = _V(JOUR, "Young children", "Teenagers")
JOUR_TEEN_PLUS = _V(JOUR, "Teenagers", "Grown children", "Empty nester", "Raising grandchildren")
JOUR_GROWN_PLUS = _V(JOUR, "Grown children", "Empty nester", "Raising grandchildren")
JOUR_EMPTY = _V(JOUR, "Empty nester")[0]
JOUR_RAISE_GRAND = _V(JOUR, "Raising grandchildren")[0]

HH = "lstyle_household_size"
HH_ALONE = _V(HH, "Lives alone")[0]
HOUSING = "demo_housing_status"
HOUSING_FAMILY = _V(HOUSING, "Living with family")[0]
HOUSING_MORTGAGE = _V(HOUSING, "Mortgage")[0]

LOSS = "lifex_loss_experience"
LOSS_NONE = _V(LOSS, "No major loss")[0]
LOSS_ELDER_ONLY = _V(LOSS, "Lost a grandparent", "Lost a parent")
LOSS_SEVERE = _V(LOSS, "Lost a partner / child", "Multiple bereavements")

# --- military ---------------------------------------------------------------
VET = "demo_veteran_status"
VET_SERVING_OR_VET = _V(VET, "Active duty", "Veteran")
VET_ACTIVE = _V(VET, "Active duty")[0]
VET_CIVILIAN = _V(VET, "Civilian")[0]
VET_MIL_ANY = _V(VET, "Active duty", "Veteran", "Reserve / guard")
MILH = "lifex_military_history"
MILH_NEVER = _V(MILH, "Never served")[0]
MILH_SERVED = _V(MILH, "Served briefly", "Career veteran", "Combat veteran")
MILH_CAREER_COMBAT = _V(MILH, "Career veteran", "Combat veteran")
MILH_COMBAT = _V(MILH, "Combat veteran")[0]

MLE = "major_life_events"
MLE_MILITARY = _V(MLE, "Military service")[0]
MLE_BUSINESS = _V(MLE, "Started a business")[0]
MLE_BEREAVE = _V(MLE, "Bereavement")[0]
MLE_HEALTH = _V(MLE, "Health journey")[0]
MLE_MOVE = _V(MLE, "Major relocation", "Migration / immigration")

# --- education ----------------------------------------------------------------
EDU = "highest_education"
EDU_ALL = _V(EDU, "No formal", "Primary", "Secondary", "Vocational / cert", "Some college",
             "Associate's", "Bachelor's", "Master's", "Doctorate", "Postdoc")
EDU_BELOW_BACHELOR = _V(EDU, "No formal", "Primary", "Secondary", "Vocational / cert",
                        "Some college", "Associate's")
EDU_BACHELOR_PLUS = _V(EDU, "Bachelor's", "Master's", "Doctorate", "Postdoc")
EDU_LE_PRIMARY = _V(EDU, "No formal", "Primary")
EDU_LE_SECONDARY = _V(EDU, "No formal", "Primary", "Secondary")
EDU_LE_VOC = _V(EDU, "No formal", "Primary", "Secondary", "Vocational / cert")
EDU_DOCTOR = _V(EDU, "Doctorate", "Postdoc")
EJOUR = "lifex_education_journey"
EJOUR_ADV = _V(EJOUR, "Advanced degrees")[0]
INST = "institution_tier"
INST_NONE = _V(INST, "Not applicable", "Self-taught")
INST_COLLEGE = _V(INST, "Top-tier research", "Mid-tier university", "Community college")
RESOUT = "research_output"
RES_THESIS = _V(RESOUT, "Thesis only")[0]
RES_PROLIFIC = _V(RESOUT, "Prolific publisher")[0]
SKILL_RESEARCH = "skill_research"
SKILL_NONE = _V(SKILL_RESEARCH, "None")[0]
DOMCHAR = "domain_characteristics"
DOMCHAR_CUTTING = _V(DOMCHAR, "Cutting-edge researcher")[0]
FAM_CREDENTIAL_COLS = _cols(
    "fam_particle_physics", "fam_astrophysics", "fam_molecular_biology", "fam_genetics",
    "fam_biochemistry", "fam_organic_chemistry", "fam_physical_chemistry", "fam_cardiology",
    "fam_neurology", "fam_oncology", "fam_radiology", "fam_surgery", "fam_immunology",
    "fam_pharmacology", "fam_epidemiology", "fam_constitutional_law", "fam_compilers",
    "fam_nuclear_engineering", "fam_aerospace_engineering")
FAM_EXPERT = _VC(FAM_CREDENTIAL_COLS, "Expert")[0]

# --- expertise: subject specialty must sit inside the domain -----------------
DOMAIN = "domain"
SPEC = "subject_specialty"
DOMAINS = _V(DOMAIN, "Software & AI", "Healthcare & Medicine", "Law & Policy",
             "Finance & Economics", "Education", "Engineering", "Natural Sciences",
             "Social Sciences", "Arts & Humanities", "Business & Management", "Agriculture",
             "Manufacturing", "Media & Journalism", "Public Sector", "Hospitality",
             "Skilled Trades")
_EDUCATION_DOMAIN = "Education"   # teaching any specialty is allowed
SPEC_DOMAIN_OK = {
    "Machine learning": ["Software & AI", "Engineering", "Natural Sciences", "Social Sciences",
                         "Finance & Economics", "Healthcare & Medicine", "Business & Management",
                         "Manufacturing", "Public Sector", "Media & Journalism", "Agriculture"],
    "Cardiology": ["Healthcare & Medicine", "Natural Sciences"],
    "Constitutional law": ["Law & Policy", "Public Sector", "Social Sciences"],
    "Quant trading": ["Finance & Economics", "Software & AI", "Business & Management",
                      "Natural Sciences"],
    "Curriculum design": ["Public Sector", "Social Sciences", "Arts & Humanities",
                          "Business & Management", "Software & AI"],
    "Structural engineering": ["Engineering", "Skilled Trades", "Manufacturing", "Public Sector"],
    "Molecular biology": ["Natural Sciences", "Healthcare & Medicine", "Agriculture", "Engineering",
                          "Manufacturing", "Software & AI"],
    "Behavioral economics": ["Finance & Economics", "Social Sciences", "Business & Management",
                             "Public Sector", "Law & Policy"],
    "Comparative literature": ["Arts & Humanities", "Media & Journalism", "Social Sciences"],
    "Operations": list(DOMAINS),
    "Agronomy": ["Agriculture", "Natural Sciences", "Public Sector", "Engineering"],
    "Robotics": ["Engineering", "Software & AI", "Manufacturing", "Natural Sciences",
                 "Healthcare & Medicine", "Agriculture", "Skilled Trades"],
    "Investigative reporting": ["Media & Journalism", "Social Sciences", "Arts & Humanities",
                                "Law & Policy"],
    "Urban planning": ["Public Sector", "Engineering", "Social Sciences", "Law & Policy",
                       "Arts & Humanities", "Business & Management", "Natural Sciences"],
    "Culinary arts": ["Hospitality", "Skilled Trades", "Arts & Humanities", "Business & Management",
                      "Manufacturing", "Agriculture", "Healthcare & Medicine", "Media & Journalism",
                      "Public Sector"],
    "Electrical work": ["Skilled Trades", "Engineering", "Manufacturing", "Public Sector"],
}
_V(SPEC, *SPEC_DOMAIN_OK.keys())
for _d in SPEC_DOMAIN_OK.values():
    _V(DOMAIN, *_d)
_V(DOMAIN, _EDUCATION_DOMAIN)

# --- employment / career ------------------------------------------------------
SCHED = "lstyle_work_schedule"
SCHED_UNEMP = _V(SCHED, "Unemployed")[0]
SCHED_FIXED = _V(SCHED, "9-to-5", "Shift work", "On-call")
SCHED_REGULAR = _V(SCHED, "9-to-5", "Shift work")
ROLE = "role_function"
ROLE_NA = _V(ROLE, "Not applicable")[0]
CSIZE = "company_size"
CSIZE_NA = _V(CSIZE, "Not applicable")[0]
CAREER = "lifex_career_path_shape"
CAREER_START = _V(CAREER, "Just starting")[0]
YEXP_11P = _V(YEXP, "11-20", "20+")
ICM = "dev_ic_manager_orientation"

# --- economy ------------------------------------------------------------------
INC = "demo_household_income"
INC_ALL = _V(INC, "<$25k", "$25k-50k", "$50k-100k", "$100k-200k", "$200k+")
INC_100P = _V(INC, "$100k-200k", "$200k+")
INC_LT50 = _V(INC, "<$25k", "$25k-50k")
INC_200P = _V(INC, "$200k+")[0]
SES = "socioeconomic_band"
SES_ALL = _V(SES, "Low income", "Lower-middle", "Middle", "Upper-middle", "High income")
SES_LOW = _V(SES, "Low income")[0]
SES_HIGH = _V(SES, "High income")[0]
SES_LOWMID = _V(SES, "Lower-middle")[0]
BANK = "lstyle_banking_style"
BANK_UNBANKED = _V(BANK, "Unbanked")[0]
BANK_CASH = _V(BANK, "Mostly cash")[0]
BANK_LOW = _V(BANK, "Unbanked", "Mostly cash")
PAY = "lstyle_payment_pref"
PAY_CARD = _V(PAY, "Credit card", "Debit card")
PAY_NONCASH = _V(PAY, "Credit card", "Debit card", "Mobile wallet", "BNPL")
PAY_APP = _V(PAY, "Mobile wallet", "BNPL")
PAY_CRYPTO = _V(PAY, "Crypto")[0]
INVEST = "lstyle_investment_style"
INVEST_CRYPTO = _V(INVEST, "Crypto-heavy")[0]
ATT_CRYPTO = "att_cryptocurrency"
OPPOSED = "Opposed"
ENTHUSIAST = "Enthusiast"
_V(ATT_CRYPTO, OPPOSED)
SAVE = "pref_save_vs_spend"
FRUG = "lstyle_frugality"
SAVE_HARD = _V(SAVE, "Hard saver")[0]
SAVE_FREE = _V(SAVE, "Free spender")[0]
FRUG_SPLURGE = _V(FRUG, "Splurger")[0]
FRUG_SAVER = _V(FRUG, "Frugal saver")[0]

# --- language -------------------------------------------------------------------
LANG_COLS = _cols(
    "lang_korean", "lang_cantonese", "lang_italian", "lang_turkish", "lang_vietnamese",
    "lang_thai", "lang_indonesian", "lang_malay", "lang_dutch", "lang_polish", "lang_ukrainian",
    "lang_persian", "lang_hebrew", "lang_greek", "lang_czech", "lang_hungarian", "lang_romanian",
    "lang_swedish", "lang_norwegian", "lang_danish", "lang_finnish", "lang_tagalog", "lang_urdu",
    "lang_tamil", "lang_telugu", "lang_marathi", "lang_punjabi", "lang_gujarati", "lang_hausa",
    "lang_yoruba", "lang_igbo", "lang_amharic", "lang_zulu", "lang_afrikaans", "lang_serbian",
    "lang_croatian", "lang_bulgarian", "lang_slovak", "lang_mandarin", "lang_spanish",
    "lang_hindi", "lang_arabic", "lang_french", "lang_portuguese", "lang_bengali",
    "lang_russian", "lang_japanese", "lang_german", "lang_swahili", "lang_english")
LANG_EN = "lang_english"
LANG_KO = "lang_korean"
ENGP = "english_proficiency"
_lang_vals = _VC(LANG_COLS, "Native", "Fluent", "Conversational", "Basic", "None")
LANG_LEVEL = dict(zip(_lang_vals, [4, 3, 2, 1, 0]))
_eng_vals = _V(ENGP, "Native", "Fluent (C1-C2)", "Intermediate (B1-B2)", "Basic (A1-A2)", "None")
ENG_LEVEL = dict(zip(_eng_vals, [4, 3, 2, 1, 0]))
LANG_KO_WEAK = _V(LANG_KO, "Conversational", "Basic", "None")
PRIMARY = "primary_language"
PRIMARY_COL = {
    "English": "lang_english", "Mandarin": "lang_mandarin", "Spanish": "lang_spanish",
    "Hindi": "lang_hindi", "Arabic": "lang_arabic", "French": "lang_french",
    "Portuguese": "lang_portuguese", "Bengali": "lang_bengali", "Russian": "lang_russian",
    "Japanese": "lang_japanese", "German": "lang_german", "Swahili": "lang_swahili",
    "Korean": "lang_korean",
}
_V(PRIMARY, *PRIMARY_COL.keys())
PRIMARY_KO = _V(PRIMARY, "Korean")[0]
PRIMARY_NON_KO = _V(PRIMARY, *[k for k in PRIMARY_COL if k != "Korean"])
HEAR = "health_hearing"
HEAR_HEARING = _V(HEAR, "Normal", "Mild loss", "Moderate loss")
MULTI = "multilingualism"
MULTI_MONO = _V(MULTI, "Monolingual")[0]
MULTI_BI = _V(MULTI, "Bilingual")[0]
MULTI_TRI = _V(MULTI, "Trilingual+")[0]
MULTI_ALL = [MULTI_MONO, MULTI_BI, MULTI_TRI]
LIFELANG = "lifex_languages_lifetime"
LIFELANG_N = dict(zip(_V(LIFELANG, "One", "Two", "Three"), [1, 2, 3]))
LIFELANG_ONE = "One"
LIFELANG_LE2 = _V(LIFELANG, "One", "Two")
HOMELANG = "demo_home_language"
HOMELANG_MULTI = _V(HOMELANG, "Bilingual home", "Heritage language", "Mixed languages")
REGISTER = "register"
REGISTER_JARGON = _V(REGISTER, "Technical jargon")[0]
JARGON = "cog_use_of_jargon"
JARGON_AVOID = _V(JARGON, "Avoids jargon")[0]

# --- population frame -----------------------------------------------------------
CIT = "demo_citizenship_status"
CIT_ALL = _V(CIT, "Citizen by birth", "Naturalized citizen", "Permanent resident", "Visa holder",
             "Dual national", "Undocumented")
CIT_NONCITIZEN = _V(CIT, "Permanent resident", "Visa holder", "Undocumented")
CIT_NATURALIZED = _V(CIT, "Naturalized citizen")[0]
IMM = "lifex_immigration_generation"
IMM_ALL = _V(IMM, "First-generation immigrant", "Second-generation", "Third+ generation",
             "Native multi-generational", "Returnee / repatriate")
IMM_FIRST = _V(IMM, "First-generation immigrant")[0]
IMM_2_3 = _V(IMM, "Second-generation", "Third+ generation")
IMM_RETURNEE = _V(IMM, "Returnee / repatriate")[0]
REGION = "region"
REGION_ALL = _V(REGION, "North America", "Latin America", "Western Europe", "Eastern Europe",
                "Sub-Saharan Africa", "MENA", "South Asia", "East Asia", "Southeast Asia",
                "Oceania")
REGION_NOT_EA = [r for r in REGION_ALL if r != "East Asia"]
ETH = "demo_ethnicity_broad"
ETH_ALL = _V(ETH, "White / European", "Black / African", "Hispanic / Latino", "East Asian",
             "South Asian", "Southeast Asian", "Middle Eastern", "Indigenous", "Pacific Islander",
             "Multiracial")
ETH_NOT_EA = [e for e in ETH_ALL if e != "East Asian"]
CULT_KR = "cult_south_korea"
FOREIGN_CULT_COLS = _cols(
    "cult_mexico", "cult_brazil", "cult_argentina", "cult_united_kingdom", "cult_france",
    "cult_germany", "cult_italy", "cult_spain", "cult_netherlands", "cult_sweden", "cult_poland",
    "cult_russia", "cult_turkey", "cult_egypt", "cult_saudi_arabia", "cult_uae", "cult_israel",
    "cult_iran", "cult_nigeria", "cult_kenya", "cult_south_africa", "cult_ethiopia", "cult_india",
    "cult_pakistan", "cult_bangladesh", "cult_china", "cult_japan", "cult_vietnam",
    "cult_thailand", "cult_indonesia", "cult_philippines", "cult_australia", "cult_new_zealand",
    "cult_singapore", "cult_malaysia", "cult_greece", "cult_portugal", "cult_united_states",
    "cult_canada")
CULT_ALL = _VC(FOREIGN_CULT_COLS + [CULT_KR], "Native", "Lived there", "Visited", "Studied",
               "Unfamiliar")
CULT_NATIVE = "Native"
CULT_BEEN = ["Lived there", "Visited"]
CULT_LIVED = "Lived there"
CULT_KR_IMPOSSIBLE = _V(CULT_KR, "Unfamiliar", "Visited")
CULT_KR_ODD = _V(CULT_KR, "Studied", "Lived there")

# --- mobility -------------------------------------------------------------------
TRAVEL = "lifex_travel_breadth"
TRAVEL_NEVER = _V(TRAVEL, "Never left country")[0]
TRAVEL_ABROAD = _V(TRAVEL, "Lived abroad", "Global nomad")
TRAVEL_NOMAD = _V(TRAVEL, "Global nomad")[0]
GEOMOB = "lifex_geographic_mobility"
GEOMOB_HOMETOWN = _V(GEOMOB, "Never left hometown")[0]
GEOMOB_INTL = _V(GEOMOB, "Moved internationally")[0]
CEXP = "lifex_cultural_exposure"
CEXP_ABROAD = _V(CEXP, "Lived across cultures", "Third-culture kid")
CEXP_MONO = _V(CEXP, "Monocultural")[0]
HOMETIE = "lifex_hometown_tie"
HOMETIE_AWAY = _V(HOMETIE, "Visits often", "Occasional return", "Rarely returns", "No connection")
TFREQ = "lstyle_travel_freq"
TFREQ_HOMEBODY = _V(TFREQ, "Homebody")[0]
TFREQ_FLYER = _V(TFREQ, "Frequent flyer")[0]
CHILDHOOD = "lifex_childhood_environment"
CHILD_HARDSHIP = _V(CHILDHOOD, "Hardship")[0]
CHILD_ROUGH = _V(CHILDHOOD, "Turbulent", "Frequently uprooted")
ADV = "lifex_adversity_level"
ADV_SHELTERED = _V(ADV, "Sheltered")[0]

# --- religion -------------------------------------------------------------------
RELIG = "demo_religion_affiliation"
REL_NONE = _V(RELIG, "None", "Atheist / agnostic")
REL_FAITHS = _V(RELIG, "Christian", "Muslim", "Hindu", "Buddhist", "Jewish", "Sikh", "Catholic",
                "Protestant", "Won Buddhist", "Korean new religion", "Other religion")
RLTY = "religiosity"
RLTY_STRONG = _V(RLTY, "Devout", "Observant")
RLTY_DEVOUT = _V(RLTY, "Devout")[0]
RLTY_SECULAR = _V(RLTY, "Secular")[0]
FAITH = "lifex_faith_journey"
FAITH_BELIEVING = _V(FAITH, "Lifelong faith", "Returned to faith")
FAITH_NONBELIEF = _V(FAITH, "Always secular", "Lapsed / left faith")
FAITH_SECULAR = _V(FAITH, "Always secular")[0]
FAITH_CONVERTED = _V(FAITH, "Converted")[0]
VAL_SPIRIT = "val_spirituality_faith"
VAL_SPIRIT_IRREL = _V(VAL_SPIRIT, "Irrelevant")[0]
TRAIT_SPIRIT = "trait_spirituality"
TRAIT_SPIRIT_ABSENT = _V(TRAIT_SPIRIT, "Absent")[0]
DIETR = "health_dietary_restriction"
DIETR_RELIGIOUS = _V(DIETR, "Religious")[0]
DIETR_NONE = _V(DIETR, "None")[0]

# --- health -----------------------------------------------------------------------
DIS = "demo_disability_status"
DIS_NONE = _V(DIS, "No disability")[0]
DIS_SENSORY = _V(DIS, "Sensory")[0]
DIS_CHRONIC = _V(DIS, "Chronic illness")[0]
VIS = "health_vision"
VIS_ALL = _V(VIS, "Normal", "Corrected", "Low vision", "Blind")
VIS_BLIND = _V(VIS, "Blind")[0]
VIS_NORMAL = _V(VIS, "Normal")[0]
VIS_OK = _V(VIS, "Normal", "Corrected")
HEAR_NORMAL = _V(HEAR, "Normal")[0]
HEAR_DEAF = _V(HEAR, "Deaf / hard of hearing")[0]
MOBIL = "health_mobility"
MOBIL_FULL = _V(MOBIL, "Full")[0]
MOBIL_AID = _V(MOBIL, "Uses mobility aid")[0]
DEXT = "health_dexterity"
DEXT_FULL = _V(DEXT, "Full")[0]
DEXT_ASSIST = _V(DEXT, "Assistive needed")[0]
COLORV = "health_color_vision"
COLORV_TYPICAL = _V(COLORV, "Typical")[0]
TEXTSZ = "health_text_size_need"
CONTRAST = "health_contrast_need"
NEED_NO = _VC([TEXTSZ, CONTRAST], "No")[0]
NEED_REQ = _VC([TEXTSZ, CONTRAST], "Requires")[0]
AT = "health_assistive_tech"
AT_ALL = _V(AT, "None", "Screen reader", "Switch control", "Voice control", "Magnifier")
AT_NONE = _V(AT, "None")[0]
AT_HEAVY = _V(AT, "Screen reader", "Switch control")
AT_VISUAL = _V(AT, "Screen reader", "Magnifier")
ACC = "accessibility_needs"
ACC_NONE = _V(ACC, "None")[0]
ACC_VISUAL = _V(ACC, "Visual")[0]
ACC_HEARING = _V(ACC, "Hearing")[0]
ACC_MOTOR = _V(ACC, "Motor")[0]
DEVCTX = "device_context"
DEVCTX_ACC = _V(DEVCTX, "Accessibility tool")[0]
DRIVER = "demo_driver_status"
DRIVER_DRIVES = _V(DRIVER, "Daily driver", "Occasional driver")
SKILL_DRIVING = "skill_driving"
SKILL_DRIVING_NONE = _V(SKILL_DRIVING, "None")[0]
CHRONIC = "health_chronic_condition"
CHRONIC_ALL = _V(CHRONIC, "None", "Managed", "Multiple", "Undiagnosed concerns")
CHRONIC_NONE = _V(CHRONIC, "None")[0]
CHRONIC_MULTI = _V(CHRONIC, "Multiple")[0]
HJOUR = "lifex_health_journey"
HJOUR_CHRONIC = _V(HJOUR, "Manages chronic condition")[0]
HJOUR_NONE = _V(HJOUR, "No major events")[0]
NEURO = "neurotype"
NEURODIV = "health_neurodivergence"
ND_TYPICAL = _VC([NEURO, NEURODIV], "Neurotypical")[0]
ND_SPECIFIC = _VC([NEURO, NEURODIV], "ADHD", "Autistic", "Dyslexic")
ND_ADHD = "ADHD"
ATTN = "health_attention_condition"
ATTN_NONE = _V(ATTN, "None")[0]
ATTN_DIAG = _V(ATTN, "Diagnosed")[0]
FIT = "health_fitness_level"
FIT_ATHLETE = _V(FIT, "Athlete")[0]
FIT_SEDENTARY = _V(FIT, "Sedentary")[0]
EXER = "lstyle_exercise_freq"
EXER_ALL = _V(EXER, "Daily", "Weekly", "Monthly", "Rarely", "Never")
EXER_LOW = _V(EXER, "Monthly", "Rarely", "Never")
EXER_NEVER = _V(EXER, "Never")[0]
EXER_DAILY = _V(EXER, "Daily")[0]
# sports that are physical exercise when played (esports, darts, bowling,
# archery and golf -- incl. Korean screen golf -- are excluded on purpose)
PHYS_SPORT_COLS = _cols(
    "sport_volleyball", "sport_wrestling", "sport_track_field", "sport_yoga",
    "sport_weightlifting", "sport_triathlon", "sport_soccer", "sport_basketball",
    "sport_american_football", "sport_baseball", "sport_tennis", "sport_cricket", "sport_rugby",
    "sport_hockey", "sport_swimming", "sport_running", "sport_cycling", "sport_boxing",
    "sport_mma", "sport_skiing", "sport_snowboarding", "sport_surfing", "sport_skateboarding",
    "sport_climbing", "sport_gymnastics", "sport_badminton", "sport_table_tennis", "sport_squash",
    "sport_sailing", "sport_rowing", "sport_martial_arts", "sport_pilates", "sport_crossfit",
    "sport_equestrian", "sport_fencing")
SPORT_PLAY = _VC(PHYS_SPORT_COLS, "Play")[0]
CLIMB_HOB_COLS = _cols("hob_rock_climbing", "hob_bouldering")
HOB_ACTIVE = _VC(CLIMB_HOB_COLS, "Avid", "Active")
MENTAL = "health_mental_health"
MENTAL_THRIVING = _V(MENTAL, "Thriving")[0]
DEP = "kr_depression_level"
DEP_HIGH = _V(DEP, "Moderately severe", "Severe")
ANX = "kr_anxiety_level"
ANX_SEVERE = _V(ANX, "Severe")[0]
HAPPY = "kr_happiness"
SATIS = "kr_life_satisfaction"
HAPPY_VH = _V(HAPPY, "Very happy")[0]
HAPPY_VU = _V(HAPPY, "Very unhappy")[0]
SATIS_VS = _V(SATIS, "Very satisfied")[0]
SATIS_VD = _V(SATIS, "Very dissatisfied")[0]
PAIN = "health_pain_level"
PAIN_SEVERE = _V(PAIN, "Severe")[0]
MEDS = "health_medication_use"
MEDS_ALL = _V(MEDS, "None", "Occasional", "Daily", "Multiple daily")
MEDS_MULTI = _V(MEDS, "Multiple daily")[0]
MEDS_NONE = _V(MEDS, "None")[0]
INSUR = "health_insurance_status"
INSUR_ALL = _V(INSUR, "Comprehensive", "Basic", "Minimal", "Uninsured")
INSUR_NONE = _V(INSUR, "Uninsured")[0]

# --- life events --------------------------------------------------------------------
VOLUNT = "lstyle_volunteering"
VOLUNT_REG = _V(VOLUNT, "Daily", "Weekly")
SERVICE = "lifex_service_history"
SERVICE_NONE = _V(SERVICE, "None")[0]
SERVICE_ACTIVIST = _V(SERVICE, "Activist / organizer")[0]

# --- lifestyle -----------------------------------------------------------------------
ECO = "lstyle_device_ecosystem"
ECO_ANDROID = _V(ECO, "Android/Google")[0]
ECO_NONAPPLE = _V(ECO, "Android/Google", "Windows", "Linux")
ECO_WINLINUX = _V(ECO, "Windows", "Linux")
MSGR = "lstyle_primary_messenger"
MSGR_IMESSAGE = _V(MSGR, "iMessage")[0]
BROWSER = "lstyle_browser"
BROWSER_SAFARI = _V(BROWSER, "Safari")[0]
KEYNOTE = "tool_keynote"
TOOL_ALL = ["Power user", "Regular", "Occasional", "Tried it", "Never used"]
TOOL_HEAVY = ["Power user", "Regular"]
TOOL_POWER = "Power user"
TOOL_NEVER = "Never used"
_V(KEYNOTE, *TOOL_HEAVY)
SOCIAL = "lstyle_primary_social"
SOCIAL_ALL = _V(SOCIAL, "Instagram", "TikTok", "X / Twitter", "Facebook", "LinkedIn", "YouTube",
                "Reddit", "None")
SOCIAL_LINKEDIN = _V(SOCIAL, "LinkedIn")[0]
SOCIAL_NONE = _V(SOCIAL, "None")[0]
LINKEDIN = "linkedin_activity"
LINKEDIN_INACTIVE = _V(LINKEDIN, "Inactive")[0]
LINKEDIN_ACTIVE = _V(LINKEDIN, "Thought leader", "Active networker")
MEDIA = "media_diet"
MEDIA_SOCIAL = _V(MEDIA, "Social-media-heavy")[0]
MEDIA_NEWS = _V(MEDIA, "News")[0]
MEDIA_MINIMAL = _V(MEDIA, "Minimal")[0]
NEWS = "lstyle_news_freq"
NEWS_AVOID = _V(NEWS, "Avoids news")[0]
NEWS_CONSTANT = _V(NEWS, "Constant")[0]
SCREEN = "lstyle_screen_time"
SCREEN_LT2 = _V(SCREEN, "<2 hrs")[0]
STREAM = "lstyle_streaming_hours"
STREAM_16P = _V(STREAM, "16+")[0]
READ = "lstyle_reading_freq"
READ_NEVER = _V(READ, "Never")[0]
BEDREAD = "habit_reading_before_bed"
BEDREAD_REG = _V(BEDREAD, "Daily", "Weekly")
HOBINT = "lstyle_hobby_intensity"
HOBINT_NONE = _V(HOBINT, "None")[0]
HOB_COLS = _cols(
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
HOB_AVID = _VC(HOB_COLS, "Avid")[0]
DIET = "lstyle_diet_type"
DIET_VEGAN = _V(DIET, "Vegan")[0]
ATT_VEGAN = "att_veganism"
_V(ATT_VEGAN, OPPOSED)
GAMING = "lstyle_gaming_freq"
GAMING_NEVER = _V(GAMING, "Never")[0]
ESPORTS = "sport_esports"
ESPORTS_PLAY = _V(ESPORTS, "Play")[0]
PASSIONATE = "Passionate"
AVERSE = "Averse"
ALC = "lstyle_alcohol_use"
ALC_NEVER = _V(ALC, "Never")[0]
_V("topic_wine", PASSIONATE)
_V("topic_craft_beer", PASSIONATE)
_VC(["hob_winemaking", "hob_home_brewing"], "Avid")
_V("fam_sommelier_knowledge", "Expert")

# --- tech tools / programming ---------------------------------------------------------
PROG_COLS = _cols(
    "prog_ruby", "prog_php", "prog_kotlin", "prog_objective_c", "prog_haskell", "prog_elixir",
    "prog_erlang", "prog_clojure", "prog_lua", "prog_perl", "prog_r", "prog_julia",
    "prog_powershell", "prog_dart", "prog_f", "prog_ocaml", "prog_assembly", "prog_cobol",
    "prog_fortran", "prog_graphql", "prog_matlab", "prog_python", "prog_javascript", "prog_java",
    "prog_c", "prog_go", "prog_bash", "prog_typescript", "prog_rust", "prog_swift", "prog_scala",
    "prog_sql", "prog_solidity")
PROG_GOOD = _VC(PROG_COLS, "Expert", "Proficient")
PROG_NONE = _VC(PROG_COLS, "None")[0]
PROG_EXPERT = "Expert"
PROG_TOOL_PAIRS = [("prog_python", "tool_python"), ("prog_sql", "tool_sql"),
                   ("prog_r", "tool_r"), ("prog_matlab", "tool_matlab")]
for _p, _t in PROG_TOOL_PAIRS:
    _V(_t, *TOOL_ALL)
SKILL_CODING = "skill_coding"
SKILL_CODING_NONE = _V(SKILL_CODING, "None")[0]
SKILL_CODING_MASTER = _V(SKILL_CODING, "Master")[0]
SKILL_CODING_BEGINNER = _V(SKILL_CODING, "Beginner")[0]
SUPERSET_PAIRS = [("prog_typescript", "prog_javascript"), ("prog_objective_c", "prog_c")]
# power-user tools whose use presupposes a core competence (tool cols, competence col, 'None')
TOOL_DOMAIN_RULES = [
    (["tool_sql"], "fam_databases"),
    (["tool_aws", "tool_azure", "tool_google_cloud"], "fam_cloud_infrastructure"),
    (["tool_tableau", "tool_power_bi", "tool_looker"], "skill_data_visualization"),
    (["tool_spss", "tool_stata", "tool_sas"], "fam_statistics"),
    (["tool_blender"], "fam_3d_modeling"),
]
for _tc, _comp in TOOL_DOMAIN_RULES:
    _VC(_tc, TOOL_POWER)
    _V(_comp, "None")
_V("prog_sql", PROG_EXPERT)
CODE_TOOLS = _cols("tool_jetbrains_ides", "tool_github_copilot")
_VC(CODE_TOOLS, *TOOL_HEAVY)
JUPYTER = "tool_jupyter"
_V(JUPYTER, *TOOL_HEAVY)
JUPYTER_LANGS = _cols("prog_python", "prog_r", "prog_julia")
TECH = "tech_savviness"
TECH_AVOIDANT = _V(TECH, "Avoidant")[0]
ADOPT = "pref_early_vs_late"
ADOPT_BLEEDING = _V(ADOPT, "Bleeding edge")[0]
DEVSTAT = "dev_professional_status"
DEVSTAT_PRO = _V(DEVSTAT, "Professional developer")[0]
DEVSTAT_CODERS = _V(DEVSTAT, "Professional developer", "Hobbyist / side-project coder")
K8S, DOCKER, GITHUB, GIT, GITLAB = ("tool_kubernetes", "tool_docker", "tool_github", "tool_git",
                                    "tool_gitlab")
_VC([K8S, GITHUB], TOOL_POWER)
_VC([DOCKER, GIT, GITLAB, GITHUB], TOOL_NEVER)
SHEETS_TOOLS = _cols("tool_excel", "tool_google_sheets")
_VC(SHEETS_TOOLS, TOOL_POWER)
SKILL_SHEETS = "skill_spreadsheet_modeling"
_V(SKILL_SHEETS, "None")

# --- developer / AI --------------------------------------------------------------------
AIUSE = "coding_ai_usage_frequency"
AIUSE_RANK = dict(zip(_V(AIUSE, "Daily", "Weekly", "Monthly or infrequent", "Tried but not active",
                         "Never used", "Actively avoids"), [5, 4, 3, 2, 1, 1]))
AIUSE_NONE = _V(AIUSE, "Never used", "Actively avoids")
AIUSE_LOW = _V(AIUSE, "Never used", "Actively avoids", "Tried but not active")
AIUSE_NEVER = _V(AIUSE, "Never used")[0]
AIUSE_AVOID = _V(AIUSE, "Actively avoids")[0]
AIUSE_TRIED = _V(AIUSE, "Tried but not active")[0]
AIUSE_MIDDLE = _V(AIUSE, "Weekly", "Monthly or infrequent")
# tasks that only exist inside software development (learning concepts, search for answers,
# data analytics and project planning are left out: a non-coder can use AI for them)
AI_TASK_CODING_COLS = _cols(
    "ai_task_code_generation", "ai_task_code_review", "ai_task_codebase_learning",
    "ai_task_debugging_fixing", "ai_task_deployment_monitoring", "ai_task_documentation",
    "ai_task_refactoring", "ai_task_testing")
AI_TASK_CURRENT = _VC(AI_TASK_CODING_COLS, "Currently mostly AI-assisted",
                      "Currently partially AI-assisted")
COPILOT = "tool_github_copilot"
COPILOT_ANY = _V(COPILOT, "Power user", "Regular", "Occasional", "Tried it")
AGENT = "coding_agent_usage_frequency"
AGENT_RANK = dict(zip(_V(AGENT, "Daily", "Weekly", "Monthly or infrequent", "Experimental only",
                         "Not yet but interested", "No plans"), [5, 4, 3, 2, 1, 1]))
AGENT_REGULAR = _V(AGENT, "Daily", "Weekly", "Monthly or infrequent")
AGENT_FREQ = _V(AGENT, "Daily", "Weekly")
AGENT_NONUSER = _V(AGENT, "Not yet but interested", "No plans")
AGFAIL = "coding_agent_failure_tolerance"
AGFAIL_NOUSE = _V(AGFAIL, "Does not use agents")[0]
AGIMPACT = "coding_agent_workflow_impact"
AGIMPACT_CHANGED = _V(AGIMPACT, "Major productivity increase", "Moderate productivity increase",
                      "Small targeted improvement", "Negative / slowdown")
AUTON = "coding_agent_autonomy_preference"
AUTON_LOW = _V(AUTON, "No agent autonomy", "Suggest-only assistant")
AUTON_FULL = _V(AUTON, "Autonomous execution")[0]
CONTROL = "coding_agent_control_expectation"
CONTROL_EDITS = _V(CONTROL, "Agent may run broad tasks autonomously",
                   "Agent may edit within scoped files")
CONTROL_READONLY = _V(CONTROL, "Read-only assistance preferred")[0]
OSS = "open_source_activity_level"
OSS_CONTRIB = _V(OSS, "Maintainer / core contributor", "Regular contributor")
OSS_NOCODE = _V(OSS, "No coding activity")[0]
GHMODE = "github_contribution_mode"
GHMODE_CODE = _V(GHMODE, "Code pusher", "Pull request contributor", "Repository maintainer")
RELEASE = "release_maintenance_orientation"
RELEASE_ACTIVE = _V(RELEASE, "Active release maintainer")[0]
AITRUST = "coding_ai_output_trust"
AITRUST_STRONG = _V(AITRUST, "Strongly trusts with light review")[0]
AITRUST_NONE = _V(AITRUST, "Does not trust")[0]
AISENT = "coding_ai_sentiment"
AISENT_VF = _V(AISENT, "Very favorable")[0]
AISENT_VU = _V(AISENT, "Very unfavorable")[0]
ATT_AI = "att_ai"
_V(ATT_AI, OPPOSED, ENTHUSIAST)
CHATBOT_TOOLS = _cols("tool_chatgpt", "tool_claude")
_VC(CHATBOT_TOOLS, TOOL_POWER)

# --- personality & values ----------------------------------------------------------------
L5 = ["Very high", "High", "Average", "Low", "Very low"]
L5_MAP = dict(zip(L5, [4, 3, 2, 1, 0]))
COG5 = ["Very high", "High", "Moderate", "Low", "None"]
COG5_MAP = dict(zip(COG5, [4, 3, 2, 1, 0]))
TRAIT5 = ["Signature", "Strong", "Moderate", "Slight", "Absent"]
TRAIT5_MAP = dict(zip(TRAIT5, [4, 3, 2, 1, 0]))
VAL5 = ["Core value", "Important", "Moderate", "Minor", "Irrelevant"]
VAL5_MAP = dict(zip(VAL5, [4, 3, 2, 1, 0]))


def _scale_for(col):
    if col.startswith(("bfi2_", "big5_", "schwartz_", "dospert_")):
        return L5_MAP
    if col.startswith("cog_"):
        return COG5_MAP
    if col.startswith("trait_"):
        return TRAIT5_MAP
    if col.startswith("val_"):
        return VAL5_MAP
    raise ValueError(col)


def _reg_scale(*cols):
    for c in cols:
        _V(c, *_scale_for(c).keys())
    return list(cols)


BFI2_DOMAINS = {
    "bfi2_domain_open_mindedness": ["bfi2_facet_intellectual_curiosity",
                                    "bfi2_facet_aesthetic_sensitivity",
                                    "bfi2_facet_creative_imagination"],
    "bfi2_domain_conscientiousness": ["bfi2_facet_organization", "bfi2_facet_productiveness",
                                      "bfi2_facet_responsibility"],
    "bfi2_domain_extraversion": ["bfi2_facet_sociability", "bfi2_facet_assertiveness",
                                 "bfi2_facet_energy_level"],
    "bfi2_domain_agreeableness": ["bfi2_facet_compassion", "bfi2_facet_respectfulness",
                                  "bfi2_facet_trust"],
    "bfi2_domain_negative_emotionality": ["bfi2_facet_anxiety", "bfi2_facet_depression",
                                          "bfi2_facet_emotional_volatility"],
}
for _dom, _facets in BFI2_DOMAINS.items():
    _reg_scale(_dom, *_facets)
DOMTRAIT = "dominant_trait"
DOMTRAIT_MAP = {
    "High openness": "bfi2_domain_open_mindedness",
    "High conscientiousness": "bfi2_domain_conscientiousness",
    "High extraversion": "bfi2_domain_extraversion",
    "High agreeableness": "bfi2_domain_agreeableness",
    "High neuroticism": "bfi2_domain_negative_emotionality",
}
_V(DOMTRAIT, *DOMTRAIT_MAP.keys())
# same facet measured by two instruments (BFI-2 vs IPIP-NEO style big5_*)
FACET_DUP_PAIRS = [
    ("bfi2_facet_assertiveness", "big5_assertiveness"),
    ("bfi2_facet_anxiety", "big5_anxiety"),
    ("bfi2_facet_depression", "big5_depression"),
    ("bfi2_facet_trust", "big5_trust"),
    ("bfi2_facet_organization", "big5_orderliness"),
    ("bfi2_facet_creative_imagination", "big5_imagination"),
    ("bfi2_facet_aesthetic_sensitivity", "big5_artistic_interest"),
    ("bfi2_facet_intellectual_curiosity", "big5_intellect"),
    ("bfi2_facet_compassion", "big5_sympathy"),
    ("bfi2_facet_sociability", "big5_gregariousness"),
    ("bfi2_facet_responsibility", "big5_dutifulness"),
]
# character strength / cognitive style vs. the same construct elsewhere
TRAIT_DUP_PAIRS = [
    ("trait_curiosity", "cog_curiosity"),
    ("trait_open_mindedness", "cog_open_mindedness"),
    ("trait_hope_optimism", "cog_optimism"),
    ("trait_discipline", "big5_self_discipline"),
    ("trait_humility", "big5_modesty"),
    ("trait_creativity", "bfi2_facet_creative_imagination"),
    ("trait_appreciation_of_beauty", "bfi2_facet_aesthetic_sensitivity"),
    ("cog_assertiveness", "big5_assertiveness"),
]
# value importance (val_*) vs. the Schwartz value with the same content
VALUE_DUP_PAIRS = [
    ("val_tradition", "schwartz_value_tradition"),
    ("val_achievement", "schwartz_value_achievement"),
    ("val_power_influence", "schwartz_value_power"),
    ("val_security_stability", "schwartz_value_security"),
    ("val_independence", "schwartz_value_self_direction"),
    ("val_helping_others", "schwartz_value_benevolence"),
    ("val_fun_enjoyment", "schwartz_value_hedonism"),
    ("val_adventure", "schwartz_value_stimulation"),
]
for _a, _b in FACET_DUP_PAIRS + TRAIT_DUP_PAIRS + VALUE_DUP_PAIRS:
    _reg_scale(_a, _b)
VPRIO = "values_priority"
VPRIO_MAP = {
    "Achievement": "val_achievement",
    "Security": "val_security_stability",
    "Autonomy": "val_independence",
    "Community": "val_community",
    "Tradition": "val_tradition",
    "Novelty": "schwartz_value_stimulation",
}
_V(VPRIO, *VPRIO_MAP.keys())
_reg_scale(*VPRIO_MAP.values())
RISK = "risk_tolerance"
RISK_AVERSE = _V(RISK, "Risk-averse")[0]
RISK_SEEKING = _V(RISK, "Risk-seeking")[0]
ATT_RISK = "att_risk_taking"
_V(ATT_RISK, OPPOSED, ENTHUSIAST)
DOSPERT_COLS = _reg_scale("dospert_financial_risk_tolerance", "dospert_ethical_risk_tolerance",
                          "dospert_health_safety_risk_tolerance",
                          "dospert_recreational_risk_tolerance", "dospert_social_risk_tolerance")
TIDY = "lstyle_tidiness"
TIDY_CHAOTIC = _V(TIDY, "Chaotic")[0]
TIDY_SPOTLESS = _V(TIDY, "Spotless")[0]
ORDER_COLS = _reg_scale("big5_orderliness", "bfi2_facet_organization")
BATTERY = "lstyle_social_battery"
BATTERY_SI = _V(BATTERY, "Strong introvert")[0]
BATTERY_SE = _V(BATTERY, "Strong extrovert")[0]
EXTRAV = "bfi2_domain_extraversion"
BIGPIC = "cog_big_picture_vs_detail"
BIGPIC_ONLY = _V(BIGPIC, "Big-picture only")[0]
BIGPIC_OBSESSED = _V(BIGPIC, "Detail-obsessed")[0]
DETAIL = "cog_detail_orientation"
_reg_scale(DETAIL)
DSPEED = "cog_decision_speed"
DSPEED_SNAP = _V(DSPEED, "Snap decisions")[0]
DSTYLE = "decision_style"
DSTYLE_DELIB = _V(DSTYLE, "Deliberative")[0]
MBTI = "mbti_type"
MBTI_ALL = _V(MBTI, "INTJ — Architect", "INTP — Logician", "ENTJ — Commander", "ENTP — Debater",
              "INFJ — Advocate", "INFP — Mediator", "ENFJ — Protagonist", "ENFP — Campaigner",
              "ISTJ — Logistician", "ISFJ — Defender", "ESTJ — Executive", "ESFJ — Consul",
              "ISTP — Virtuoso", "ISFP — Adventurer", "ESTP — Entrepreneur", "ESFP — Entertainer")
MBTI_E = [m for m in MBTI_ALL if m.startswith("E")]
MBTI_I = [m for m in MBTI_ALL if m.startswith("I")]

# --- interests -----------------------------------------------------------------------------
TOPIC_ACAD_PAIRS = [("topic_philosophy", "acad_philosophy"), ("topic_literature", "acad_literature"),
                    ("topic_astronomy", "acad_astronomy"), ("topic_visual_art", "acad_visual_art"),
                    ("topic_music", "acad_music"), ("topic_languages", "acad_foreign_languages"),
                    ("topic_theater", "acad_drama")]
for _a, _b in TOPIC_ACAD_PAIRS:
    _VC([_a, _b], PASSIONATE, AVERSE)
TOPIC_BOOK_PAIRS = [("topic_poetry", "bookg_poetry"), ("topic_philosophy", "bookg_philosophy"),
                    ("topic_true_crime", "bookg_true_crime"), ("topic_history", "bookg_history")]
BOOK_LOVE = "Love"
for _a, _b in TOPIC_BOOK_PAIRS:
    _V(_a, AVERSE)
    _V(_b, BOOK_LOVE)
# (topic, activity column, top engagement value)
TOPIC_ACTIVITY_TOP = [
    ("topic_knitting", "hob_knitting", "Avid"), ("topic_woodworking", "hob_woodworking", "Avid"),
    ("topic_calligraphy", "hob_calligraphy", "Avid"), ("topic_genealogy", "hob_genealogy", "Avid"),
    ("topic_birdwatching", "hob_birdwatching", "Avid"),
    ("topic_stand_up_comedy", "hob_stand_up_comedy", "Avid"),
    ("topic_baking", "hob_bread_baking", "Avid"),
    ("topic_gardening", "hob_vegetable_gardening", "Avid"),
    ("topic_dance", "hob_salsa_dancing", "Avid"), ("topic_dance", "hob_ballroom_dance", "Avid"),
    ("topic_collecting", "hob_coin_collecting", "Avid"),
    ("topic_collecting", "hob_stamp_collecting", "Avid"),
    ("topic_astronomy", "hob_stargazing", "Avid"), ("topic_craft_beer", "hob_home_brewing", "Avid"),
    ("topic_wine", "hob_winemaking", "Avid"),
    ("topic_yoga", "sport_yoga", "Play"), ("topic_running", "sport_running", "Play"),
    ("topic_cycling", "sport_cycling", "Play"),
    ("topic_meditation", "habit_meditation", "Daily"),
    ("topic_gaming", "lstyle_gaming_freq", "Daily"),
    ("topic_podcasts", "lstyle_podcast_listening", "Daily"),
]
# (topic, activity column, zero engagement value)
TOPIC_ACTIVITY_ZERO = [
    ("topic_podcasts", "lstyle_podcast_listening", "Never"),
    ("topic_gaming", "lstyle_gaming_freq", "Never"),
    ("topic_meditation", "habit_meditation", "Never"),
    ("topic_cooking", "lstyle_cooking_freq", "Never"),
    ("topic_literature", "lstyle_reading_freq", "Never"),
    ("topic_fitness", "lstyle_exercise_freq", "Never"),
    ("topic_music", "lstyle_music_listening", "Rarely"),
    ("topic_knitting", "hob_knitting", "Never"), ("topic_woodworking", "hob_woodworking", "Never"),
    ("topic_calligraphy", "hob_calligraphy", "Never"),
    ("topic_birdwatching", "hob_birdwatching", "Never"),
    ("topic_yoga", "sport_yoga", "None"),
]
for _t, _a, _v in TOPIC_ACTIVITY_TOP:
    _V(_t, AVERSE)
    _V(_a, _v)
for _t, _a, _v in TOPIC_ACTIVITY_ZERO:
    _V(_t, PASSIONATE)
    _V(_a, _v)

# --- attitudes & politics --------------------------------------------------------------------
LEAN = "political_lean"
LEAN_APOLITICAL = _V(LEAN, "Apolitical")[0]
ENGAGE = "demo_political_engagement"
ENGAGE_ACTIVIST = _V(ENGAGE, "Activist")[0]
TOPIC_POL = "topic_politics"
_V(TOPIC_POL, AVERSE)
TOPIC_ACTIVISM = "topic_activism"
_V(TOPIC_ACTIVISM, AVERSE)
WORKLOC = "pref_work_location"
WORKLOC_OFFICE = _V(WORKLOC, "Strongly office")[0]
WORKLOC_REMOTE = _V(WORKLOC, "Strongly remote")[0]
ATT_REMOTE = "att_remote_work"
ATT_OFFICE = "att_working_from_office"
_VC([ATT_REMOTE, ATT_OFFICE], OPPOSED, ENTHUSIAST)

# --- age-cohort tables (2020 reference: birth year = 2020 - age) -------------------------------
# birth years:  18-24 -> 1996-2001, 25-34 -> 1986-95, 35-44 -> 1976-85, 45-54 -> 1966-75,
#               55-64 -> 1956-65, 65-74 -> 1946-55, 75-84 -> 1936-45, 85+ -> <=1935
# Generations (Pew: Silent 1928-45, Boomer 46-64, X 65-80, Millennial 81-96, Z 97-2012, Alpha 2013+)
# with adjacent cohorts allowed where a mainstream or Korean definition puts the bracket edge there
# (e.g. Strauss-Howe boomers from 1943, McCrindle Gen Z from 1995, Korean 2nd baby boom 1968-74).
GEN = "demo_generation"
GEN_ALL = _V(GEN, "Gen Alpha", "Gen Z", "Millennial", "Gen X", "Boomer", "Silent")
GEN_OK = {
    "18-24": ["Gen Z", "Millennial"],
    "25-34": ["Millennial", "Gen Z"],
    "35-44": ["Gen X", "Millennial"],
    "45-54": ["Gen X", "Boomer"],
    "55-64": ["Boomer", "Gen X"],
    "65-74": ["Boomer", "Silent"],
    "75-84": ["Silent", "Boomer"],
    "85+": ["Silent"],
}
# Formative decade must overlap ages 8-30 for some birth year in the bracket
# ('1960s-70s' is the earliest bucket, so it is the only option for 75+).
DECADE = "lifex_formative_decade"
DECADE_ALL = _V(DECADE, "1960s-70s", "1980s", "1990s", "2000s", "2010s", "2020s")
DECADE_OK = {
    "18-24": ["2000s", "2010s", "2020s"],
    "25-34": ["1990s", "2000s", "2010s", "2020s"],
    "35-44": ["1980s", "1990s", "2000s", "2010s"],
    "45-54": ["1960s-70s", "1980s", "1990s", "2000s"],
    "55-64": ["1960s-70s", "1980s", "1990s"],
    "65-74": ["1960s-70s", "1980s"],
    "75-84": ["1960s-70s"],
    "85+": ["1960s-70s"],
}


def _bracket_table_hit(df, value_col, table, all_values):
    age = _s(df, AGE)
    val = _s(df, value_col)
    known = val.isin(all_values).to_numpy(dtype=bool)
    bad = _F(df)
    for bracket, ok in table.items():
        bad |= (age == bracket).to_numpy(dtype=bool) & ~val.isin(ok).to_numpy(dtype=bool)
    return bad & known


def _bracket_table_scope(df, value_col, all_values):
    return _in(df, AGE, AGE_ADULT) & _in(df, value_col, all_values)


# --- language profile -----------------------------------------------------------------------------
def _lang_profile(df):
    """Per-row language counts.  English is taken as the min ('lo') or max ('hi') of
    lang_english and english_proficiency so that every count rule is conservative."""
    lv = {c: _lvl(df, c, LANG_LEVEL) for c in LANG_COLS}
    eng_p = _lvl(df, ENGP, ENG_LEVEL)
    valid = np.all([~np.isnan(v) for v in lv.values()], axis=0) & ~np.isnan(eng_p)
    eng_lo = np.fmin(lv[LANG_EN], eng_p)
    eng_hi = np.fmax(lv[LANG_EN], eng_p)
    others = np.vstack([np.nan_to_num(v, nan=0.0) for c, v in lv.items() if c != LANG_EN])
    lo = np.nan_to_num(eng_lo, nan=0.0)
    hi = np.nan_to_num(eng_hi, nan=0.0)

    def cnt(th, e):
        return (others >= th).sum(axis=0) + (e >= th)

    return {
        "valid": valid, "lv": lv, "eng_lo": lo, "eng_hi": hi,
        "n_fluent_lo": cnt(3, lo), "n_conv_lo": cnt(2, lo), "n_conv_hi": cnt(2, hi),
        "n_native_lo": cnt(4, lo),
    }


def _primary_level(df, use_hi):
    p = _lang_profile(df)
    prim = _s(df, PRIMARY)
    out = np.full(len(df), np.nan)
    for lang, col in PRIMARY_COL.items():
        m = (prim == lang).to_numpy(dtype=bool)
        if col == LANG_EN:
            v = p["eng_hi"] if use_hi else p["eng_lo"]
        else:
            v = np.nan_to_num(p["lv"][col], nan=-1.0)
        out = np.where(m, v, out)
    out = np.where(p["valid"], out, np.nan)
    return out


def _pairs_extreme(df, pairs, hi_a=4, lo_b=0):
    """True where any pair sits at opposite extremes (a top & b bottom, or a bottom & b top)."""
    out = _F(df)
    for a, b in pairs:
        la = _lvl(df, a, _scale_for(a))
        lb = _lvl(df, b, _scale_for(b))
        out |= ((la == 4) & (lb == 0)) | ((la == 0) & (lb == 4))
    return out


def _pairs_scope(df, pairs):
    out = _F(df)
    for a, b in pairs:
        la = _lvl(df, a, _scale_for(a))
        lb = _lvl(df, b, _scale_for(b))
        out |= ~np.isnan(la) & ~np.isnan(lb)
    return out


# =====================================================================================
# RULES
# =====================================================================================
G = "population"
_rule("R2-PO-H01", G, "hard", "모집단은 19세 이상 성인인데 연령대가 19세 미만",
      lambda d: _in(d, AGE, AGE_ALL), lambda d: _in(d, AGE, AGE_U19))
_rule("R2-PO-H02", G, "hard", "모집단은 한국 국적자인데 시민권 상태가 영주권자·비자 소지자·미등록",
      lambda d: _in(d, CIT, CIT_ALL), lambda d: _in(d, CIT, CIT_NONCITIZEN))
_rule("R2-PO-H03", G, "hard", "모집단은 국내 출생자인데 이민 1세대(해외 출생 이주자)로 표기",
      lambda d: _in(d, IMM, IMM_ALL), lambda d: _eq(d, IMM, IMM_FIRST))
_rule("R2-PO-H04", G, "hard", "한국 거주자인데 거주 권역(region)이 동아시아가 아님",
      lambda d: _in(d, REGION, REGION_ALL), lambda d: _in(d, REGION, REGION_NOT_EA))
_rule("R2-PO-H05", G, "hard", "한국에서 태어나 사는 사람이 한국 문화에 '모름' 또는 '방문만 함'",
      lambda d: _in(d, CULT_KR, CULT_ALL), lambda d: _in(d, CULT_KR, CULT_KR_IMPOSSIBLE))
_rule("R2-PO-S01", G, "soft", "국내 출생 국민이 귀화자이거나 이민 2·3세대: 성인 인구의 1% 미만",
      lambda d: _in(d, CIT, CIT_ALL) & _in(d, IMM, IMM_ALL),
      lambda d: _eq(d, CIT, CIT_NATURALIZED) | _in(d, IMM, IMM_2_3))
_rule("R2-PO-S02", G, "soft", "국내 출생 한국 국민인데 동아시아계가 아님: 다문화 가정 출신 성인은 1% 미만",
      lambda d: _in(d, ETH, ETH_ALL), lambda d: _in(d, ETH, ETH_NOT_EA))
_rule("R2-PO-S03", G, "soft",
      "국내 출생 한국인의 주 언어가 한국어가 아니거나 한국어 말하기가 회화 이하(청각장애 제외) 또는 영어 원어민: 1% 미만",
      lambda d: _in(d, PRIMARY, list(PRIMARY_COL)) & _in(d, LANG_KO, LANG_LEVEL) & _in(d, HEAR, HEAR_HEARING + [HEAR_DEAF]),
      lambda d: _in(d, PRIMARY, PRIMARY_NON_KO)
      | (_in(d, LANG_KO, LANG_KO_WEAK) & _in(d, HEAR, HEAR_HEARING))
      | _eq(d, ENGP, "Native") | _eq(d, LANG_EN, "Native"))
_rule("R2-PO-S04", G, "soft", "국내 출생자의 한국 문화 친숙도가 Native가 아니거나(학습·거주만) 외국 문화가 Native: 1% 미만",
      lambda d: _in(d, CULT_KR, CULT_ALL),
      lambda d: _in(d, CULT_KR, CULT_KR_ODD) | _any_in(d, FOREIGN_CULT_COLS, [CULT_NATIVE]))

G = "age"
_rule("R2-AG-H01", G, "hard", "출생연도(2020−나이) 기준 불가능한 세대 구분 (인접 세대 경계는 허용)",
      lambda d: _bracket_table_scope(d, GEN, GEN_ALL),
      lambda d: _bracket_table_hit(d, GEN, GEN_OK, GEN_ALL))
_rule("R2-AG-H02", G, "hard", "출생연도(2020−나이) 기준 8~30세 시기와 겹치지 않는 형성기 연대",
      lambda d: _bracket_table_scope(d, DECADE, DECADE_ALL),
      lambda d: _bracket_table_hit(d, DECADE, DECADE_OK, DECADE_ALL))
_rule("R2-AG-H03", G, "hard", "19~24세인데 해당 분야 경력 20년 이상(4세 이전 시작)",
      lambda d: _eq(d, AGE, A1824), lambda d: _eq(d, YEXP, YEXP_20P))
_rule("R2-AG-H04", G, "hard", "19~24세인데 현재 관계 기간 15년 이상(9세 이전 시작)",
      lambda d: _eq(d, AGE, A1824), lambda d: _eq(d, RLEN, RLEN_15P))
_rule("R2-AG-S01", G, "soft", "경력 시작이 15세 이전이어야 하는 경력연수(19~24세 11년+, 25~34세 20년+; 15세 미만 고용 금지)",
      lambda d: _in(d, AGE, AGE_LE34),
      lambda d: (_eq(d, AGE, A1824) & _eq(d, YEXP, YEXP_11_20)) | (_eq(d, AGE, A2534) & _eq(d, YEXP, YEXP_20P)))
_rule("R2-AG-S02", G, "soft", "34세 이하인데 은퇴(고용상태·생애단계·직급 중 하나): 1% 미만",
      lambda d: _in(d, AGE, AGE_LE34),
      lambda d: _eq(d, EMP, EMP_RETIRED) | _eq(d, LIFE, LIFE_RETIRE) | _eq(d, SEN, SEN_RETIRED))
_rule("R2-AG-S03", G, "soft", "65세 이상인데 주된 상태가 학생(고용상태 또는 생애단계): 1% 미만",
      lambda d: _in(d, AGE, AGE_65P),
      lambda d: _eq(d, EMP, EMP_STUDENT) | _eq(d, LIFE, LIFE_STUDENT))
_rule("R2-AG-S04", G, "soft", "19~24세인데 임원급 직급·연쇄창업/엑시트·업계 베테랑: 경력 기간상 거의 불가능",
      lambda d: _eq(d, AGE, A1824),
      lambda d: _in(d, SEN, SEN_EXEC) | _in(d, ENTRE, ENTRE_SERIAL) | _any_in(d, IND_COLS, [IND_VETERAN]))

G = "family"
_rule("R2-FM-H01", G, "hard", "연애·약혼·결혼·동거 중인데 관계 기간이 '관계 없음'",
      lambda d: _in(d, MAR, MAR_PARTNERED), lambda d: _eq(d, RLEN, RLEN_NONE))
_rule("R2-FM-H02", G, "hard", "혼인상태가 미혼(Single)인데 결혼·재혼·사별 이력 (이혼·사별·별거는 별도 값이 있음)",
      lambda d: _eq(d, MAR, MAR_SINGLE), lambda d: _in(d, RHIST, RHIST_EVER_MARRIED))
_rule("R2-FM-H03", G, "hard", "자녀 수와 부모 상태 불일치 (자녀 없음인데 부모/조부모, 부모 아님인데 자녀 있음)",
      lambda d: _in(d, KIDS, KIDS_ALL) & _in(d, PAR, PAR_ALL),
      lambda d: (_eq(d, KIDS, KIDS_NONE) & _in(d, PAR, PAR_YES)) | (_eq(d, PAR, PAR_NOT) & _in(d, KIDS, KIDS_HAS)))
_rule("R2-FM-H04", G, "hard", "자녀 수와 양육 단계 불일치 (자녀 없음인데 자녀 양육 단계, '자녀 없음' 단계인데 자녀 있음)",
      lambda d: _in(d, KIDS, KIDS_ALL) & _in(d, JOUR, JOUR_ALL),
      lambda d: (_eq(d, KIDS, KIDS_NONE) & _in(d, JOUR, JOUR_KIDS)) | (_eq(d, JOUR, JOUR_NONE) & _in(d, KIDS, KIDS_HAS)))
_rule("R2-FM-H05", G, "hard", "부모 상태와 양육 단계 불일치 (부모 아님인데 자녀·손주 양육, '자녀 없음'인데 부모/조부모)",
      lambda d: _in(d, PAR, PAR_ALL) & _in(d, JOUR, JOUR_ALL),
      lambda d: (_eq(d, PAR, PAR_NOT) & _in(d, JOUR, JOUR_KIDS)) | (_eq(d, JOUR, JOUR_NONE) & _in(d, PAR, PAR_YES)))
_rule("R2-FM-H06", G, "hard", "생애단계(영유아 부모/빈 둥지)와 자녀 상태 불일치",
      lambda d: _eq(d, LIFE, LIFE_YOUNGKIDS) | _eq(d, LIFE, LIFE_EMPTY),
      lambda d: (_eq(d, LIFE, LIFE_YOUNGKIDS)
                 & (_eq(d, KIDS, KIDS_NONE) | _eq(d, PAR, PAR_NOT) | _eq(d, JOUR, JOUR_NONE) | _eq(d, JOUR, JOUR_EMPTY)))
      | (_eq(d, LIFE, LIFE_EMPTY)
         & (_eq(d, KIDS, KIDS_NONE) | _eq(d, PAR, PAR_NOT) | _eq(d, JOUR, JOUR_NONE) | _in(d, JOUR, JOUR_AT_HOME))))
_rule("R2-FM-H07", G, "hard", "19~24세인데 10대·성인 자녀나 손주가 있음(11세 이전 출산 필요)",
      lambda d: _eq(d, AGE, A1824),
      lambda d: _in(d, PAR, PAR_ADULT_GRAND) | _eq(d, KIDS, KIDS_ADULT) | _in(d, JOUR, JOUR_TEEN_PLUS)
      | _eq(d, LIFE, LIFE_EMPTY))
_rule("R2-FM-H08", G, "hard", "55세 이상 여성이 임신 중(Expecting): 생물학적으로 사실상 불가능",
      lambda d: _eq(d, GENDER, WOMAN) & _in(d, AGE, AGE_55P), lambda d: _eq(d, KIDS, KIDS_EXPECT))
_rule("R2-FM-H09", G, "hard", "1인가구인데 주거 상태가 '가족과 거주'이거나 손주를 양육",
      lambda d: _eq(d, HH, HH_ALONE),
      lambda d: _eq(d, HOUSING, HOUSING_FAMILY) | _eq(d, JOUR, JOUR_RAISE_GRAND))
_rule("R2-FM-H10", G, "hard", "배우자와 사별했는데 상실 경험이 '큰 상실 없음'",
      lambda d: _eq(d, MAR, MAR_WIDOWED) | _eq(d, RHIST, RHIST_WIDOWED), lambda d: _eq(d, LOSS, LOSS_NONE))
_rule("R2-FM-S01", G, "soft", "25~34세에 성인 자녀(15세 이하 출산 필요) 또는 35~44세에 손주: 1% 미만",
      lambda d: _eq(d, AGE, A2534) | _eq(d, AGE, A3544),
      lambda d: (_eq(d, AGE, A2534) & (_in(d, PAR, PAR_ADULT_GRAND) | _eq(d, KIDS, KIDS_ADULT)
                                      | _in(d, JOUR, JOUR_GROWN_PLUS) | _eq(d, LIFE, LIFE_EMPTY)))
      | (_eq(d, AGE, A3544) & (_eq(d, PAR, PAR_GRAND) | _eq(d, JOUR, JOUR_RAISE_GRAND))))
_rule("R2-FM-S02", G, "soft", "75세 이상인데 임신·신생아 부모·영유아 양육: 1% 미만",
      lambda d: _in(d, AGE, AGE_75P),
      lambda d: _eq(d, KIDS, KIDS_EXPECT) | _eq(d, PAR, PAR_NEW) | _eq(d, JOUR, JOUR_YOUNG) | _eq(d, LIFE, LIFE_YOUNGKIDS))
_rule("R2-FM-S03", G, "soft", "19~24세인데 사별·이혼 또는 재혼·사별 이력: 1% 미만",
      lambda d: _eq(d, AGE, A1824),
      lambda d: _in(d, MAR, MAR_WID_DIV) | _in(d, RHIST, RHIST_REM_WID))
_rule("R2-FM-S04", G, "soft", "배우자 사별자의 상실 경험이 조부모/부모 상실로만 표기(배우자 상실 누락)",
      lambda d: _eq(d, MAR, MAR_WIDOWED) | _eq(d, RHIST, RHIST_WIDOWED), lambda d: _in(d, LOSS, LOSS_ELDER_ONLY))

G = "military"
_rule("R2-MI-H01", G, "hard", "현역 또는 전역 군인인데 군 복무 경험 '없음'",
      lambda d: _in(d, VET, VET_SERVING_OR_VET), lambda d: _eq(d, MILH, MILH_NEVER))
_rule("R2-MI-H02", G, "hard", "65세 이상 현역 군인: 군인사법 연령정년(최고 대장 63세) 초과",
      lambda d: _eq(d, VET, VET_ACTIVE), lambda d: _in(d, AGE, AGE_65P))
_rule("R2-MI-H03", G, "hard", "인생을 형성한 경험이 군 복무인데 복무 경험 '없음'",
      lambda d: _eq(d, MLE, MLE_MILITARY), lambda d: _eq(d, MILH, MILH_NEVER))
_rule("R2-MI-S01", G, "soft", "여성인데 군 복무·전역·예비군 이력: 여성은 징집 대상이 아니며 여군 경험자는 여성의 1% 미만",
      lambda d: _eq(d, GENDER, WOMAN),
      lambda d: _in(d, MILH, MILH_SERVED) | _in(d, VET, VET_MIL_ANY))
_rule("R2-MI-S02", G, "soft", "65세 미만 참전용사: 한국군 참전(6·25·베트남전)은 2020년 기준 65세 이상, 이후 교전 경험자는 극소수",
      lambda d: _eq(d, MILH, MILH_COMBAT), lambda d: _in(d, AGE, AGE_LT65))
_rule("R2-MI-S03", G, "soft", "직업군인·참전 이력인데 군 관련 상태가 '민간인'(전역군인 값 별도)",
      lambda d: _eq(d, VET, VET_CIVILIAN), lambda d: _in(d, MILH, MILH_CAREER_COMBAT))

G = "education"
_rule("R2-ED-H01", G, "hard", "교육 여정이 '대학원 학위'인데 최종학력이 학사 미만",
      lambda d: _eq(d, EJOUR, EJOUR_ADV), lambda d: _in(d, EDU, EDU_BELOW_BACHELOR))
_rule("R2-ED-H02", G, "hard", "학사 이상 학위가 있는데 교육기관 유형이 '해당없음/독학'",
      lambda d: _in(d, EDU, EDU_BACHELOR_PLUS), lambda d: _in(d, INST, INST_NONE))
_rule("R2-ED-H03", G, "hard", "최종학력 무학/초졸인데 대학(전문대 포함) 교육기관 표기",
      lambda d: _in(d, EDU, EDU_LE_PRIMARY), lambda d: _in(d, INST, INST_COLLEGE))
_rule("R2-ED-H04", G, "hard", "학위논문 실적인데 최종학력 고졸·직업자격 이하, 또는 다작 연구자인데 연구 역량 없음",
      lambda d: _eq(d, RESOUT, RES_THESIS) | _eq(d, RESOUT, RES_PROLIFIC),
      lambda d: (_eq(d, RESOUT, RES_THESIS) & _in(d, EDU, EDU_LE_VOC))
      | (_eq(d, RESOUT, RES_PROLIFIC) & _eq(d, SKILL_RESEARCH, SKILL_NONE)))
_rule("R2-ED-S01", G, "soft", "19~24세 박사/박사후: 통상 27세 이후 취득, 극소수",
      lambda d: _eq(d, AGE, A1824), lambda d: _in(d, EDU, EDU_DOCTOR))
_rule("R2-ED-S02", G, "soft", "고졸 이하인데 학위 기반 분야(의학·물리·법학 등) 전문가·다작 연구자·최첨단 연구자: 1% 미만",
      lambda d: _in(d, EDU, EDU_LE_SECONDARY),
      lambda d: _any_in(d, FAM_CREDENTIAL_COLS, [FAM_EXPERT]) | _eq(d, RESOUT, RES_PROLIFIC)
      | _eq(d, DOMCHAR, DOMCHAR_CUTTING))

G = "expertise"


def _spec_domain_hit(d):
    spec = _s(d, SPEC)
    dom = _s(d, DOMAIN)
    bad = _F(d)
    for sp, ok in SPEC_DOMAIN_OK.items():
        bad |= (spec == sp).to_numpy(dtype=bool) & ~dom.isin(ok + [_EDUCATION_DOMAIN]).to_numpy(dtype=bool)
    return bad


_rule("R2-EX-H01", G, "hard", "세부 전공(subject_specialty)이 주 분야(domain)에 속하지 않음 (교육 분야와 Operations는 모두 허용)",
      lambda d: _in(d, SPEC, list(SPEC_DOMAIN_OK)) & _in(d, DOMAIN, DOMAINS), _spec_domain_hit)

G = "employment"
_rule("R2-EM-H01", G, "hard", "근무 일정과 고용 상태 불일치 (취업자인데 '실업' 일정, 실업자인데 정규·교대·대기 근무)",
      lambda d: _eq(d, SCHED, SCHED_UNEMP) | _eq(d, EMP, EMP_UNEMP),
      lambda d: (_eq(d, SCHED, SCHED_UNEMP) & _in(d, EMP, EMP_WORKING))
      | (_eq(d, EMP, EMP_UNEMP) & _in(d, SCHED, SCHED_FIXED)))
_rule("R2-EM-H02", G, "hard", "취업자인데 직무가 '해당없음', 또는 피고용자인데 회사 규모 '해당없음'",
      lambda d: _in(d, EMP, EMP_WORKING),
      lambda d: _eq(d, ROLE, ROLE_NA) | (_in(d, EMP, EMP_EMPLOYEE) & _eq(d, CSIZE, CSIZE_NA)))
_rule("R2-EM-H03", G, "hard", "직급이 '은퇴'인데 전일제 취업 중",
      lambda d: _eq(d, SEN, SEN_RETIRED), lambda d: _eq(d, EMP, EMP_FT))
_rule("R2-EM-H04", G, "hard", "은퇴자인데 경력 경로가 '막 시작' 또는 생애단계 '초기 경력'",
      lambda d: _eq(d, EMP, EMP_RETIRED), lambda d: _eq(d, CAREER, CAREER_START) | _eq(d, LIFE, LIFE_EARLY))
_rule("R2-EM-H05", G, "hard", "경력이 막 시작됐는데 해당 분야 경력 11년 이상",
      lambda d: _eq(d, CAREER, CAREER_START), lambda d: _in(d, YEXP, YEXP_11P))
_rule("R2-EM-H06", G, "hard", "직급이 창업자(Founder)인데 창업 이력이 '고려 안 함/고려만 함'",
      lambda d: _eq(d, SEN, SEN_FOUNDER), lambda d: _in(d, ENTRE, ENTRE_NONE))
_rule("R2-EM-S01", G, "soft", "직급과 경력연수 불일치 (임원급인데 0~2년, 인턴인데 20년 이상)",
      lambda d: _in(d, SEN, SEN_EXEC) | _eq(d, SEN, SEN_INTERN),
      lambda d: (_in(d, SEN, SEN_EXEC) & _eq(d, YEXP, YEXP_0_2)) | (_eq(d, SEN, SEN_INTERN) & _eq(d, YEXP, YEXP_20P)))
_rule("R2-EM-S02", G, "soft", "은퇴 상태인데 전일제 취업(생애단계 은퇴) 또는 정규·교대 근무 일정(고용상태 은퇴)",
      lambda d: _eq(d, EMP, EMP_RETIRED) | _eq(d, LIFE, LIFE_RETIRE),
      lambda d: (_eq(d, LIFE, LIFE_RETIRE) & _eq(d, EMP, EMP_FT))
      | (_eq(d, EMP, EMP_RETIRED) & _in(d, SCHED, SCHED_REGULAR)))

G = "economy_housing"
_rule("R2-EC-H01", G, "hard", "가구소득 $100k(약 1.3억원) 이상인데 소득·자산 계층이 '저소득'",
      lambda d: _in(d, INC, INC_ALL) & _in(d, SES, SES_ALL),
      lambda d: _in(d, INC, INC_100P) & _eq(d, SES, SES_LOW))
_rule("R2-EC-H02", G, "hard", "은행 계좌가 없는데(Unbanked) 주 결제수단이 신용/체크카드",
      lambda d: _eq(d, BANK, BANK_UNBANKED), lambda d: _in(d, PAY, PAY_CARD))
_rule("R2-EC-H03", G, "hard", "암호화폐로 주로 결제하거나 암호화폐 중심 투자를 하면서 암호화폐에 반대",
      lambda d: _eq(d, PAY, PAY_CRYPTO) | _eq(d, INVEST, INVEST_CRYPTO), lambda d: _eq(d, ATT_CRYPTO, OPPOSED))
_rule("R2-EC-H04", G, "hard", "저축·소비 성향을 재는 두 속성이 정반대 (극단 저축가↔과소비, 자유 소비↔검소한 저축)",
      lambda d: _in(d, SAVE, [SAVE_HARD, SAVE_FREE]),
      lambda d: (_eq(d, SAVE, SAVE_HARD) & _eq(d, FRUG, FRUG_SPLURGE)) | (_eq(d, SAVE, SAVE_FREE) & _eq(d, FRUG, FRUG_SAVER)))
_rule("R2-EC-S01", G, "soft", "가구소득과 소득계층이 크게 어긋남 ($50k 미만인데 고소득층, $200k+인데 중하층)",
      lambda d: _in(d, INC, INC_ALL) & _in(d, SES, SES_ALL),
      lambda d: (_in(d, INC, INC_LT50) & _eq(d, SES, SES_HIGH)) | (_eq(d, INC, INC_200P) & _eq(d, SES, SES_LOWMID)))
_rule("R2-EC-S02", G, "soft", "현금 위주인데 주 결제가 카드·간편결제, 또는 무계좌인데 간편결제·주택담보대출",
      lambda d: _in(d, BANK, BANK_LOW),
      lambda d: (_eq(d, BANK, BANK_CASH) & _in(d, PAY, PAY_NONCASH))
      | (_eq(d, BANK, BANK_UNBANKED) & (_in(d, PAY, PAY_APP) | _eq(d, HOUSING, HOUSING_MORTGAGE))))

G = "language"
_rule("R2-LA-H01", G, "hard", "주 언어(primary_language)의 말하기 수준이 기초 이하",
      lambda d: ~np.isnan(_primary_level(d, use_hi=True)),
      lambda d: _primary_level(d, use_hi=True) <= 1)
_rule("R2-LA-H02", G, "hard", "영어 능력을 재는 두 속성(english_proficiency, lang_english)이 2단계 이상 차이",
      lambda d: ~np.isnan(_lvl(d, ENGP, ENG_LEVEL)) & ~np.isnan(_lvl(d, LANG_EN, LANG_LEVEL)),
      lambda d: np.abs(_lvl(d, ENGP, ENG_LEVEL) - _lvl(d, LANG_EN, LANG_LEVEL)) >= 2)


def _multi_scope(d):
    return _lang_profile(d)["valid"] & _in(d, MULTI, MULTI_ALL)


def _multi_hit(d):
    p = _lang_profile(d)
    return ((_eq(d, MULTI, MULTI_MONO) & (p["n_fluent_lo"] >= 2))
            | (_eq(d, MULTI, MULTI_BI) & ((p["n_conv_hi"] < 2) | (p["n_fluent_lo"] >= 3)))
            | (_eq(d, MULTI, MULTI_TRI) & (p["n_conv_hi"] < 3)))


_rule("R2-LA-H03", G, "hard", "다중언어 수준(단일/이중/삼중+)과 언어별 숙련도 개수가 불일치",
      _multi_scope, _multi_hit)


def _lifelang_scope(d):
    return _lang_profile(d)["valid"] & _in(d, LIFELANG, list(LIFELANG_N)) & _in(d, MULTI, MULTI_ALL)


def _lifelang_hit(d):
    p = _lang_profile(d)
    learned = _lvl(d, LIFELANG, LIFELANG_N)
    return ((learned < p["n_conv_lo"])
            | (_eq(d, MULTI, MULTI_BI) & _eq(d, LIFELANG, LIFELANG_ONE))
            | (_eq(d, MULTI, MULTI_TRI) & _in(d, LIFELANG, LIFELANG_LE2)))


_rule("R2-LA-H04", G, "hard", "평생 배운 언어 수가 현재 회화 이상 언어 수 또는 작업 언어 수보다 적음",
      _lifelang_scope, _lifelang_hit)
_rule("R2-LA-H05", G, "hard", "기본 말투가 '전문용어'인데 전문용어 사용은 '회피'",
      lambda d: _eq(d, REGISTER, REGISTER_JARGON), lambda d: _eq(d, JARGON, JARGON_AVOID))
_rule("R2-LA-S01", G, "soft", "주 언어의 말하기 수준이 '회화' 수준에 그침(모국어·지배 언어는 유창 이상이 정상)",
      lambda d: ~np.isnan(_primary_level(d, use_hi=True)),
      lambda d: _primary_level(d, use_hi=True) == 2)
_rule("R2-LA-S02", G, "soft", "영어 두 속성 중 하나만 '원어민'(1단계 차이지만 모국어 여부가 엇갈림)",
      lambda d: ~np.isnan(_lvl(d, ENGP, ENG_LEVEL)) & ~np.isnan(_lvl(d, LANG_EN, LANG_LEVEL)),
      lambda d: (np.abs(_lvl(d, ENGP, ENG_LEVEL) - _lvl(d, LANG_EN, LANG_LEVEL)) == 1)
      & ((_lvl(d, ENGP, ENG_LEVEL) == 4) | (_lvl(d, LANG_EN, LANG_LEVEL) == 4)))
_rule("R2-LA-S03", G, "soft", "가정에서 두 언어 이상(이중언어·계승어·혼합)을 쓰는데 본인은 단일언어",
      lambda d: _in(d, HOMELANG, HOMELANG_MULTI), lambda d: _eq(d, MULTI, MULTI_MONO))

G = "mobility_culture"
_rule("R2-MO-H01", G, "hard", "출국 경험이 없는데 외국 문화에 '거주/방문' 경험",
      lambda d: _eq(d, TRAVEL, TRAVEL_NEVER), lambda d: _any_in(d, FOREIGN_CULT_COLS, CULT_BEEN))
_rule("R2-MO-H02", G, "hard", "출국 경험이 없는데 국제 이주·타문화권 거주·제3문화 아동·귀국자",
      lambda d: _eq(d, TRAVEL, TRAVEL_NEVER),
      lambda d: _eq(d, GEOMOB, GEOMOB_INTL) | _in(d, CEXP, CEXP_ABROAD) | _eq(d, IMM, IMM_RETURNEE))
_rule("R2-MO-H03", G, "hard", "고향을 떠난 적 없는데 고향과의 관계가 '자주 방문/가끔 귀향/거의 안 감/단절'",
      lambda d: _eq(d, GEOMOB, GEOMOB_HOMETOWN), lambda d: _in(d, HOMETIE, HOMETIE_AWAY))
_rule("R2-MO-H04", G, "hard", "고향을 떠난 적 없는데 해외 거주·타문화권 거주·외국 거주·귀국자 이력",
      lambda d: _eq(d, GEOMOB, GEOMOB_HOMETOWN),
      lambda d: _in(d, TRAVEL, TRAVEL_ABROAD) | _in(d, CEXP, CEXP_ABROAD) | _eq(d, IMM, IMM_RETURNEE)
      | _any_in(d, FOREIGN_CULT_COLS, [CULT_LIVED]))
_rule("R2-MO-H05", G, "hard", "글로벌 노마드인데 여행 빈도가 '집에만 있음(Homebody)'",
      lambda d: _eq(d, TRAVEL, TRAVEL_NOMAD), lambda d: _eq(d, TFREQ, TFREQ_HOMEBODY))
_rule("R2-MO-S01", G, "soft", "출국 경험이 없는데 비행기를 자주 탐(국내선만으로 잦은 비행은 드묾)",
      lambda d: _eq(d, TRAVEL, TRAVEL_NEVER), lambda d: _eq(d, TFREQ, TFREQ_FLYER))
_rule("R2-MO-S02", G, "soft", "해외 거주 경험이 있는데 문화 노출이 '단일문화'",
      lambda d: _eq(d, CEXP, CEXP_MONO),
      lambda d: _in(d, TRAVEL, TRAVEL_ABROAD) | _any_in(d, FOREIGN_CULT_COLS, [CULT_LIVED]))

G = "religion"
_rule("R2-RE-H01", G, "hard", "종교 없음/무신론인데 종교성이 '독실/실천'",
      lambda d: _in(d, RELIG, REL_NONE), lambda d: _in(d, RLTY, RLTY_STRONG))
_rule("R2-RE-H02", G, "hard", "종교 없음/무신론인데 신앙 여정이 '평생 신앙/신앙 복귀'",
      lambda d: _in(d, RELIG, REL_NONE), lambda d: _in(d, FAITH, FAITH_BELIEVING))
_rule("R2-RE-H03", G, "hard", "신앙 여정과 현재 종교성이 정반대 (항상 세속·신앙 떠남인데 독실, 평생 신앙·복귀인데 세속)",
      lambda d: _in(d, FAITH, FAITH_NONBELIEF + FAITH_BELIEVING),
      lambda d: (_in(d, FAITH, FAITH_NONBELIEF) & _in(d, RLTY, RLTY_STRONG))
      | (_in(d, FAITH, FAITH_BELIEVING) & _eq(d, RLTY, RLTY_SECULAR)))
_rule("R2-RE-S01", G, "soft", "항상 세속적이었는데 특정 종교 소속, 또는 개종했는데 무종교",
      lambda d: _eq(d, FAITH, FAITH_SECULAR) | _eq(d, FAITH, FAITH_CONVERTED),
      lambda d: (_eq(d, FAITH, FAITH_SECULAR) & _in(d, RELIG, REL_FAITHS))
      | (_eq(d, FAITH, FAITH_CONVERTED) & _in(d, RELIG, REL_NONE)))
_rule("R2-RE-S02", G, "soft", "독실한 신자인데 영성·신앙 가치가 '무관'하거나 영성 특성이 '없음'",
      lambda d: _eq(d, RLTY, RLTY_DEVOUT),
      lambda d: _eq(d, VAL_SPIRIT, VAL_SPIRIT_IRREL) | _eq(d, TRAIT_SPIRIT, TRAIT_SPIRIT_ABSENT))
_rule("R2-RE-S03", G, "soft", "종교적 식이제한이 있는데 종교 없음/무신론",
      lambda d: _eq(d, DIETR, DIETR_RELIGIOUS), lambda d: _in(d, RELIG, REL_NONE))

G = "health"
_rule("R2-HE-H01", G, "hard", "실명(Blind)인데 장애 상태가 '장애 없음'",
      lambda d: _eq(d, DIS, DIS_NONE), lambda d: _eq(d, VIS, VIS_BLIND))
_rule("R2-HE-H02", G, "hard", "감각 장애인데 시각·청각이 모두 정상",
      lambda d: _eq(d, DIS, DIS_SENSORY), lambda d: _in(d, VIS, VIS_OK) & _eq(d, HEAR, HEAR_NORMAL))
_rule("R2-HE-H03", G, "hard", "만성질환 '없음'인데 만성질환 장애 또는 만성질환 관리 중",
      lambda d: _eq(d, CHRONIC, CHRONIC_NONE),
      lambda d: _eq(d, DIS, DIS_CHRONIC) | _eq(d, HJOUR, HJOUR_CHRONIC))
_rule("R2-HE-H04", G, "hard", "실명인데 운전함(매일/가끔)",
      lambda d: _eq(d, VIS, VIS_BLIND), lambda d: _in(d, DRIVER, DRIVER_DRIVES))
_rule("R2-HE-H05", G, "hard", "필요 편의(accessibility_needs) 유형과 해당 신체 기능이 불일치 (실명인데 없음, 청각/시각/운동 편의인데 해당 기능 정상)",
      lambda d: _in(d, ACC, [ACC_NONE, ACC_VISUAL, ACC_HEARING, ACC_MOTOR]),
      lambda d: (_eq(d, ACC, ACC_NONE) & _eq(d, VIS, VIS_BLIND))
      | (_eq(d, ACC, ACC_HEARING) & _eq(d, HEAR, HEAR_NORMAL))
      | (_eq(d, ACC, ACC_VISUAL) & _in(d, VIS, VIS_OK) & _eq(d, COLORV, COLORV_TYPICAL)
         & _eq(d, TEXTSZ, NEED_NO) & _eq(d, CONTRAST, NEED_NO))
      | (_eq(d, ACC, ACC_MOTOR) & _eq(d, MOBIL, MOBIL_FULL) & _eq(d, DEXT, DEXT_FULL)))
_rule("R2-HE-H06", G, "hard", "접근성 도구로 이용 중인데 보조기술·접근성 요구가 모두 '없음'",
      lambda d: _eq(d, DEVCTX, DEVCTX_ACC), lambda d: _eq(d, AT, AT_NONE) & _eq(d, ACC, ACC_NONE))
_rule("R2-HE-H07", G, "hard", "신경발달 표기 불일치 (neurotype↔neurodivergence 비정형/정형, ADHD↔주의력 상태)",
      lambda d: _in(d, NEURO, ND_SPECIFIC + [ND_TYPICAL]) | _in(d, NEURODIV, ND_SPECIFIC + [ND_TYPICAL]),
      lambda d: (_in(d, NEURO, ND_SPECIFIC) & _eq(d, NEURODIV, ND_TYPICAL))
      | (_in(d, NEURODIV, ND_SPECIFIC) & _eq(d, NEURO, ND_TYPICAL))
      | (_eq(d, ATTN, ATTN_NONE) & (_eq(d, NEURO, ND_ADHD) | _eq(d, NEURODIV, ND_ADHD)))
      | (_eq(d, ATTN, ATTN_DIAG) & _eq(d, NEURO, ND_TYPICAL) & _eq(d, NEURODIV, ND_TYPICAL)))
_rule("R2-HE-H08", G, "hard", "운동 빈도와 체력·종목 참여 모순 (운동선수급인데 월 1회 이하, 운동 안 함인데 신체 스포츠·클라이밍을 직접 함)",
      lambda d: _in(d, EXER, EXER_ALL),
      lambda d: (_eq(d, FIT, FIT_ATHLETE) & _in(d, EXER, EXER_LOW))
      | (_eq(d, EXER, EXER_NEVER) & (_any_in(d, PHYS_SPORT_COLS, [SPORT_PLAY]) | _any_in(d, CLIMB_HOB_COLS, HOB_ACTIVE))))
_rule("R2-HE-H09", G, "hard", "정신건강 '번영(Thriving)'인데 PHQ-9 중등도-중증 이상 우울 또는 GAD-7 중증 불안",
      lambda d: _eq(d, MENTAL, MENTAL_THRIVING), lambda d: _in(d, DEP, DEP_HIGH) | _eq(d, ANX, ANX_SEVERE))
_rule("R2-HE-S01", G, "soft", "장애 없음인데 농/난청·보행보조기·손기능 보조·스크린리더/스위치 사용: 드묾",
      lambda d: _eq(d, DIS, DIS_NONE),
      lambda d: _eq(d, HEAR, HEAR_DEAF) | _eq(d, MOBIL, MOBIL_AID) | _eq(d, DEXT, DEXT_ASSIST) | _in(d, AT, AT_HEAVY))
_rule("R2-HE-S02", G, "soft", "시력 정상인데 스크린리더·돋보기·고대비/큰 글자 필수, 또는 실명인데 보조기술 없음",
      lambda d: _in(d, VIS, [VIS_NORMAL, VIS_BLIND]),
      lambda d: (_eq(d, VIS, VIS_NORMAL) & (_in(d, AT, AT_VISUAL) | _eq(d, CONTRAST, NEED_REQ) | _eq(d, TEXTSZ, NEED_REQ)))
      | (_eq(d, VIS, VIS_BLIND) & _eq(d, AT, AT_NONE)))
_rule("R2-HE-S03", G, "soft", "체력 수준이 '좌식 생활'인데 매일 운동",
      lambda d: _eq(d, FIT, FIT_SEDENTARY), lambda d: _eq(d, EXER, EXER_DAILY))
_rule("R2-HE-S04", G, "soft", "정신건강 '번영'인데 매우 불행하거나 삶에 매우 불만족",
      lambda d: _eq(d, MENTAL, MENTAL_THRIVING), lambda d: _eq(d, HAPPY, HAPPY_VU) | _eq(d, SATIS, SATIS_VD))
_rule("R2-HE-S05", G, "soft", "행복감과 삶의 만족도가 정반대 극단 (매우 행복↔매우 불만족, 매우 불행↔매우 만족)",
      lambda d: _in(d, HAPPY, [HAPPY_VH, HAPPY_VU]),
      lambda d: (_eq(d, HAPPY, HAPPY_VH) & _eq(d, SATIS, SATIS_VD)) | (_eq(d, HAPPY, HAPPY_VU) & _eq(d, SATIS, SATIS_VS)))
_rule("R2-HE-S06", G, "soft", "중증 만성통증인데 만성질환 '없음'(만성통증 자체가 만성질환으로 분류됨)",
      lambda d: _eq(d, PAIN, PAIN_SEVERE), lambda d: _eq(d, CHRONIC, CHRONIC_NONE))
_rule("R2-HE-S07", G, "soft", "복약과 만성질환 불일치 (질환 없이 하루 여러 번 복약, 다중 만성질환인데 복약 없음)",
      lambda d: _in(d, MEDS, MEDS_ALL) & _in(d, CHRONIC, CHRONIC_ALL),
      lambda d: (_eq(d, MEDS, MEDS_MULTI) & _eq(d, CHRONIC, CHRONIC_NONE)) | (_eq(d, CHRONIC, CHRONIC_MULTI) & _eq(d, MEDS, MEDS_NONE)))
_rule("R2-HE-S08", G, "soft", "건강보험 미가입: 국민건강보험·의료급여가 국내 거주 국민 전체에 의무 적용",
      lambda d: _in(d, INSUR, INSUR_ALL), lambda d: _eq(d, INSUR, INSUR_NONE))

G = "life_events"
_rule("R2-EV-H01", G, "hard", "인생을 형성한 경험이 창업인데 창업 이력이 '고려 안 함/고려만 함'",
      lambda d: _eq(d, MLE, MLE_BUSINESS), lambda d: _in(d, ENTRE, ENTRE_NONE))
_rule("R2-EV-H02", G, "hard", "인생을 형성한 경험이 사별인데 상실 경험이 '큰 상실 없음'",
      lambda d: _eq(d, MLE, MLE_BEREAVE), lambda d: _eq(d, LOSS, LOSS_NONE))
_rule("R2-EV-H03", G, "hard", "인생을 형성한 경험이 건강 문제인데 건강 여정이 '큰 사건 없음'",
      lambda d: _eq(d, MLE, MLE_HEALTH), lambda d: _eq(d, HJOUR, HJOUR_NONE))
_rule("R2-EV-H04", G, "hard", "인생을 형성한 경험이 큰 이주·이사인데 고향을 떠난 적 없음",
      lambda d: _in(d, MLE, MLE_MOVE), lambda d: _eq(d, GEOMOB, GEOMOB_HOMETOWN))
_rule("R2-EV-H05", G, "hard", "어린 시절이 '궁핍(Hardship)'인데 겪은 역경이 '보호받음(Sheltered)'",
      lambda d: _eq(d, CHILDHOOD, CHILD_HARDSHIP), lambda d: _eq(d, ADV, ADV_SHELTERED))
_rule("R2-EV-H06", G, "hard", "매일/매주 자원봉사하는데 봉사 이력이 '없음'",
      lambda d: _in(d, VOLUNT, VOLUNT_REG), lambda d: _eq(d, SERVICE, SERVICE_NONE))
_rule("R2-EV-S01", G, "soft", "어린 시절이 불안정·잦은 이사인데 역경 수준이 '보호받음'",
      lambda d: _in(d, CHILDHOOD, CHILD_ROUGH), lambda d: _eq(d, ADV, ADV_SHELTERED))
_rule("R2-EV-S02", G, "soft", "75세 이상(부모가 95세 이상)인데 상실 경험이 '큰 상실 없음'",
      lambda d: _in(d, AGE, AGE_75P), lambda d: _eq(d, LOSS, LOSS_NONE))
G = "lifestyle"
_rule("R2-LS-H01", G, "hard", "Android/Google 생태계인데 애플 기기 전용 iMessage·Safari를 주로 사용",
      lambda d: _eq(d, ECO, ECO_ANDROID), lambda d: _eq(d, MSGR, MSGR_IMESSAGE) | _eq(d, BROWSER, BROWSER_SAFARI))
_rule("R2-LS-H02", G, "hard", "주 SNS 표기와 SNS 활동 불일치 (주 SNS가 LinkedIn인데 LinkedIn 비활동, SNS 없음인데 LinkedIn 활발·SNS 중심 정보 소비)",
      lambda d: _in(d, SOCIAL, SOCIAL_ALL),
      lambda d: (_eq(d, SOCIAL, SOCIAL_LINKEDIN) & _eq(d, LINKEDIN, LINKEDIN_INACTIVE))
      | (_eq(d, SOCIAL, SOCIAL_NONE) & (_in(d, LINKEDIN, LINKEDIN_ACTIVE) | _eq(d, MEDIA, MEDIA_SOCIAL))))
_rule("R2-LS-H03", G, "hard", "뉴스 소비 빈도와 주 정보원이 정반대 (뉴스 회피인데 뉴스가 주 정보원, 뉴스 상시 소비인데 미디어 최소)",
      lambda d: _in(d, NEWS, [NEWS_AVOID, NEWS_CONSTANT]),
      lambda d: (_eq(d, NEWS, NEWS_AVOID) & _eq(d, MEDIA, MEDIA_NEWS)) | (_eq(d, NEWS, NEWS_CONSTANT) & _eq(d, MEDIA, MEDIA_MINIMAL)))
_rule("R2-LS-H04", G, "hard", "하루 화면 시간 2시간 미만인데 주 16시간 이상 스트리밍(하루 평균 2.3시간 이상)",
      lambda d: _eq(d, SCREEN, SCREEN_LT2), lambda d: _eq(d, STREAM, STREAM_16P))
_rule("R2-LS-H05", G, "hard", "독서를 전혀 안 하는데 자기 전 독서가 매일/매주 습관",
      lambda d: _eq(d, READ, READ_NEVER), lambda d: _in(d, BEDREAD, BEDREAD_REG))
_rule("R2-LS-H06", G, "hard", "취미 강도가 '없음'인데 열성(Avid) 취미가 있음",
      lambda d: _eq(d, HOBINT, HOBINT_NONE), lambda d: _any_in(d, HOB_COLS, [HOB_AVID]))
_rule("R2-LS-H07", G, "hard", "비건 식단을 하면서 비거니즘에 반대",
      lambda d: _eq(d, DIET, DIET_VEGAN), lambda d: _eq(d, ATT_VEGAN, OPPOSED))
_rule("R2-LS-H08", G, "hard", "운전을 하는데(매일/가끔) 운전 실력이 'None'",
      lambda d: _in(d, DRIVER, DRIVER_DRIVES), lambda d: _eq(d, SKILL_DRIVING, SKILL_DRIVING_NONE))
_rule("R2-LS-H09", G, "hard", "게임을 전혀 안 하는데 e스포츠를 직접 함",
      lambda d: _eq(d, GAMING, GAMING_NEVER), lambda d: _eq(d, ESPORTS, ESPORTS_PLAY))
_rule("R2-LS-S01", G, "soft", "애플 기기 없이(Windows/Linux/Android) iMessage·Safari·Keynote를 주력 사용",
      lambda d: _in(d, ECO, ECO_NONAPPLE),
      lambda d: (_in(d, ECO, ECO_WINLINUX) & (_eq(d, MSGR, MSGR_IMESSAGE) | _eq(d, BROWSER, BROWSER_SAFARI)))
      | _in(d, KEYNOTE, TOOL_HEAVY))
_rule("R2-LS-S03", G, "soft", "술을 전혀 안 마시는데 와인·크래프트 맥주 열정, 와인·맥주 자가 양조 열성, 소믈리에 전문가",
      lambda d: _eq(d, ALC, ALC_NEVER),
      lambda d: _eq(d, "topic_wine", PASSIONATE) | _eq(d, "topic_craft_beer", PASSIONATE)
      | _eq(d, "hob_winemaking", "Avid") | _eq(d, "hob_home_brewing", "Avid")
      | _eq(d, "fam_sommelier_knowledge", "Expert"))

G = "tech_tools"


def _prog_tool_hit(d):
    out = _F(d)
    for p, t in PROG_TOOL_PAIRS:
        out |= (_in(d, p, PROG_GOOD) & _eq(d, t, TOOL_NEVER)) | (_eq(d, p, PROG_NONE) & _in(d, t, TOOL_HEAVY))
    return out


def _prog_tool_scope(d):
    out = _F(d)
    for p, t in PROG_TOOL_PAIRS:
        out |= _in(d, p, PROG_GOOD + [PROG_NONE]) & _in(d, t, TOOL_ALL)
    return out


_rule("R2-TE-H01", G, "hard", "같은 언어를 재는 prog_*와 tool_*(Python·SQL·R·MATLAB)가 모순 (숙련인데 사용 경험 없음, None인데 상시 사용)",
      _prog_tool_scope, _prog_tool_hit)
_rule("R2-TE-H02", G, "hard", "코딩 스킬 'None'인데 어떤 프로그래밍 언어에 숙련/전문가",
      lambda d: _eq(d, SKILL_CODING, SKILL_CODING_NONE), lambda d: _any_in(d, PROG_COLS, PROG_GOOD))


def _superset_hit(d):
    out = _F(d)
    for sup, base in SUPERSET_PAIRS:
        out |= _in(d, sup, PROG_GOOD) & _eq(d, base, PROG_NONE)
    return out


_rule("R2-TE-H03", G, "hard", "상위집합 언어(TypeScript⊃JavaScript, Objective-C⊃C)에 숙련인데 기반 언어는 None",
      lambda d: _any_in(d, [s for s, _ in SUPERSET_PAIRS], PROG_GOOD), _superset_hit)


def _tool_domain_scope(d):
    out = _eq(d, "prog_sql", PROG_EXPERT)
    for tools, _comp in TOOL_DOMAIN_RULES:
        out |= _any_in(d, tools, [TOOL_POWER])
    return out


def _tool_domain_hit(d):
    out = _eq(d, "prog_sql", PROG_EXPERT) & _eq(d, "fam_databases", "None")
    for tools, comp in TOOL_DOMAIN_RULES:
        out |= _any_in(d, tools, [TOOL_POWER]) & _eq(d, comp, "None")
    return out


_rule("R2-TE-H04", G, "hard", "도구 파워유저인데 그 도구의 핵심 지식이 None (SQL→DB, 클라우드→클라우드 인프라, BI→시각화, 통계패키지→통계, Blender→3D)",
      _tool_domain_scope, _tool_domain_hit)
_rule("R2-TE-H05", G, "hard", "코딩 전용 도구(JetBrains IDE·Copilot·Jupyter)를 상시 쓰는데 코딩 스킬/언어 숙련도가 전혀 없음",
      lambda d: _any_in(d, CODE_TOOLS, TOOL_HEAVY) | _in(d, JUPYTER, TOOL_HEAVY),
      lambda d: (_any_in(d, CODE_TOOLS, TOOL_HEAVY) & _eq(d, SKILL_CODING, SKILL_CODING_NONE))
      | (_in(d, JUPYTER, TOOL_HEAVY) & _all_in(d, JUPYTER_LANGS, [PROG_NONE])))
_rule("R2-TE-H06", G, "hard", "기술 회피형(Avoidant)인데 최신 기술 선도 수용자(Bleeding edge)이거나 전문 개발자",
      lambda d: _eq(d, TECH, TECH_AVOIDANT),
      lambda d: _eq(d, ADOPT, ADOPT_BLEEDING) | _eq(d, DEVSTAT, DEVSTAT_PRO))
_rule("R2-TE-S01", G, "soft", "코딩 스킬 'Master'인데 33개 언어 숙련도가 모두 None(C++/C#만 쓰는 경우 등 드묾)",
      lambda d: _eq(d, SKILL_CODING, SKILL_CODING_MASTER), lambda d: _all_in(d, PROG_COLS, [PROG_NONE]))
_rule("R2-TE-S02", G, "soft", "도구 의존관계 불일치 (Kubernetes 파워유저인데 Docker 미사용, GitHub 파워유저인데 Git 미사용)",
      lambda d: _eq(d, K8S, TOOL_POWER) | _eq(d, GITHUB, TOOL_POWER),
      lambda d: (_eq(d, K8S, TOOL_POWER) & _eq(d, DOCKER, TOOL_NEVER)) | (_eq(d, GITHUB, TOOL_POWER) & _eq(d, GIT, TOOL_NEVER)))
G = "developer_ai"
_rule("R2-DV-H01", G, "hard", "개발에 AI를 쓰지 않는다(미사용·회피·중단)면서 코드 작성·리뷰·디버깅 등 개발 작업을 현재 AI로 하거나 GitHub Copilot 사용",
      lambda d: _in(d, AIUSE, AIUSE_LOW),
      lambda d: _any_in(d, AI_TASK_CODING_COLS, AI_TASK_CURRENT)
      | (_eq(d, AIUSE, AIUSE_NEVER) & _in(d, COPILOT, COPILOT_ANY))
      | (_in(d, AIUSE, [AIUSE_AVOID, AIUSE_TRIED]) & _in(d, COPILOT, TOOL_HEAVY)))
_rule("R2-DV-H02", G, "hard", "코딩 에이전트를 정기 사용하는데 개발용 AI 도구는 미사용/중단(에이전트는 AI 도구의 일부)",
      lambda d: _in(d, AGENT, AGENT_REGULAR),
      lambda d: (_in(d, AGENT, AGENT_REGULAR) & _in(d, AIUSE, AIUSE_NONE))
      | (_in(d, AGENT, AGENT_FREQ) & _eq(d, AIUSE, AIUSE_TRIED)))
_rule("R2-DV-H03", G, "hard", "에이전트를 쓰지 않는다면서 정기 사용 빈도 또는 업무 영향(생산성 변화)을 보고",
      lambda d: _in(d, AGENT, AGENT_NONUSER) | _eq(d, AGFAIL, AGFAIL_NOUSE),
      lambda d: (_eq(d, AGFAIL, AGFAIL_NOUSE) & _in(d, AGENT, AGENT_REGULAR))
      | ((_in(d, AGENT, AGENT_NONUSER) | _eq(d, AGFAIL, AGFAIL_NOUSE)) & _in(d, AGIMPACT, AGIMPACT_CHANGED)))
_rule("R2-DV-H04", G, "hard", "에이전트 자율성 선호와 통제 기대가 정반대 (자율성 없음·제안만인데 파일 수정·광범위 자율 실행 허용, 완전 자율인데 읽기 전용)",
      lambda d: _in(d, AUTON, AUTON_LOW + [AUTON_FULL]),
      lambda d: (_in(d, AUTON, AUTON_LOW) & _in(d, CONTROL, CONTROL_EDITS))
      | (_eq(d, AUTON, AUTON_FULL) & _eq(d, CONTROL, CONTROL_READONLY)))
_rule("R2-DV-H05", G, "hard", "전문·취미 개발자 또는 오픈소스 정기 기여자인데 코딩 스킬 'None'",
      lambda d: _in(d, DEVSTAT, DEVSTAT_CODERS) | _in(d, OSS, OSS_CONTRIB),
      lambda d: _eq(d, SKILL_CODING, SKILL_CODING_NONE))
_rule("R2-DV-H06", G, "hard", "코딩 활동 '없음'인데 전문·취미 개발자이거나 커밋·PR·저장소 관리·릴리스 담당",
      lambda d: _eq(d, OSS, OSS_NOCODE),
      lambda d: _in(d, DEVSTAT, DEVSTAT_CODERS) | _in(d, GHMODE, GHMODE_CODE) | _eq(d, RELEASE, RELEASE_ACTIVE))
_rule("R2-DV-H07", G, "hard", "커밋·PR·저장소 관리 중심인데 Git·GitHub·GitLab 모두 사용 경험 없음",
      lambda d: _in(d, GHMODE, GHMODE_CODE), lambda d: _all_in(d, [GIT, GITHUB, GITLAB], [TOOL_NEVER]))
_rule("R2-DV-S01", G, "soft", "코딩 에이전트 사용 빈도가 개발 AI 도구 전체 사용 빈도보다 높음",
      lambda d: _in(d, AIUSE, AIUSE_MIDDLE) & _in(d, AGENT, AGENT_RANK),
      lambda d: _lvl(d, AGENT, AGENT_RANK) > _lvl(d, AIUSE, AIUSE_RANK))
_rule("R2-DV-S02", G, "soft", "전문 개발자인데 코딩 스킬 초급이거나 33개 언어 숙련도가 모두 None",
      lambda d: _eq(d, DEVSTAT, DEVSTAT_PRO),
      lambda d: _eq(d, SKILL_CODING, SKILL_CODING_BEGINNER) | _all_in(d, PROG_COLS, [PROG_NONE]))
_rule("R2-DV-S03", G, "soft", "AI 코딩 태도·신뢰·사용이 정반대 (강한 신뢰↔매우 비호의, 불신↔매우 호의, 적극 회피↔매우 호의)",
      lambda d: _in(d, AISENT, [AISENT_VF, AISENT_VU]),
      lambda d: (_eq(d, AITRUST, AITRUST_STRONG) & _eq(d, AISENT, AISENT_VU))
      | (_eq(d, AITRUST, AITRUST_NONE) & _eq(d, AISENT, AISENT_VF))
      | (_eq(d, AIUSE, AIUSE_AVOID) & _eq(d, AISENT, AISENT_VF)))
_rule("R2-DV-S04", G, "soft", "AI 전반 태도와 AI 사용·코딩 AI 태도가 정반대 (AI 반대인데 ChatGPT/Claude 파워유저·코딩 AI 매우 호의, AI 열광인데 코딩 AI 매우 비호의)",
      lambda d: _in(d, ATT_AI, [OPPOSED, ENTHUSIAST]),
      lambda d: (_eq(d, ATT_AI, OPPOSED) & (_any_in(d, CHATBOT_TOOLS, [TOOL_POWER]) | _eq(d, AISENT, AISENT_VF)))
      | (_eq(d, ATT_AI, ENTHUSIAST) & _eq(d, AISENT, AISENT_VU)))

G = "personality_values"


def _bfi_domain_hit(d):
    out = _F(d)
    for dom, facets in BFI2_DOMAINS.items():
        ld = _lvl(d, dom, L5_MAP)
        lf = np.vstack([_lvl(d, f, L5_MAP) for f in facets])
        out |= ((ld >= 3) & np.all(lf <= 1, axis=0)) | ((ld <= 1) & np.all(lf >= 3, axis=0))
    return out


def _bfi_domain_scope(d):
    out = _F(d)
    for dom, facets in BFI2_DOMAINS.items():
        ok = ~np.isnan(_lvl(d, dom, L5_MAP))
        for f in facets:
            ok &= ~np.isnan(_lvl(d, f, L5_MAP))
        out |= ok
    return out


_rule("R2-PS-H01", G, "hard", "BFI-2 영역 점수가 하위 3개 facet과 반대쪽 (영역 높음인데 facet 모두 낮음, 또는 그 반대: 영역=facet 평균)",
      _bfi_domain_scope, _bfi_domain_hit)


def _domtrait_lvl(d):
    dt = _s(d, DOMTRAIT)
    out = np.full(len(d), np.nan)
    for v, col in DOMTRAIT_MAP.items():
        out = np.where((dt == v).to_numpy(dtype=bool), _lvl(d, col, L5_MAP), out)
    return out


_rule("R2-PS-H02", G, "hard", "가장 두드러진 Big Five 특성(dominant_trait)이 해당 BFI-2 영역에서 낮음/매우 낮음",
      lambda d: ~np.isnan(_domtrait_lvl(d)), lambda d: _domtrait_lvl(d) <= 1)
_rule("R2-PS-H03", G, "hard", "같은 성격 facet을 두 척도(BFI-2와 big5_*)로 쟀는데 매우 높음↔매우 낮음",
      lambda d: _pairs_scope(d, FACET_DUP_PAIRS), lambda d: _pairs_extreme(d, FACET_DUP_PAIRS))
_rule("R2-PS-H04", G, "hard", "성격강점·인지 스타일과 같은 구성개념의 다른 속성이 정반대 극단 (호기심·개방성·낙관·절제·겸손·창의성·심미성·주장성)",
      lambda d: _pairs_scope(d, TRAIT_DUP_PAIRS), lambda d: _pairs_extreme(d, TRAIT_DUP_PAIRS))
_rule("R2-PS-H05", G, "hard", "생활 자기기술과 성격 척도가 정반대 (엉망인 정리정돈↔질서성 매우 높음, 강한 내향↔외향성 매우 높음 등)",
      lambda d: _in(d, TIDY, [TIDY_CHAOTIC, TIDY_SPOTLESS]) | _in(d, BATTERY, [BATTERY_SI, BATTERY_SE]),
      lambda d: (_eq(d, TIDY, TIDY_CHAOTIC) & _any_in(d, ORDER_COLS, ["Very high"]))
      | (_eq(d, TIDY, TIDY_SPOTLESS) & _any_in(d, ORDER_COLS, ["Very low"]))
      | (_eq(d, BATTERY, BATTERY_SI) & _eq(d, EXTRAV, "Very high"))
      | (_eq(d, BATTERY, BATTERY_SE) & _eq(d, EXTRAV, "Very low")))
_rule("R2-PS-H06", G, "hard", "같은 내용의 가치를 두 척도(val_*와 Schwartz)로 쟀는데 핵심 가치↔매우 낮음 또는 무관↔매우 높음",
      lambda d: _pairs_scope(d, VALUE_DUP_PAIRS), lambda d: _pairs_extreme(d, VALUE_DUP_PAIRS))


def _vprio_lvl(d):
    vp = _s(d, VPRIO)
    out = np.full(len(d), np.nan)
    for v, col in VPRIO_MAP.items():
        out = np.where((vp == v).to_numpy(dtype=bool), _lvl(d, col, _scale_for(col)), out)
    return out


_rule("R2-PS-H07", G, "hard", "최우선 가치(values_priority)가 해당 가치 척도에서 '낮음/사소/무관'",
      lambda d: ~np.isnan(_vprio_lvl(d)), lambda d: _vprio_lvl(d) <= 1)
_rule("R2-PS-H08", G, "hard", "위험 성향 모순 (위험 회피↔위험감수 열광, 위험 추구↔위험감수 반대, 전반 성향과 DOSPERT 5개 영역이 모두 반대쪽)",
      lambda d: _in(d, RISK, [RISK_AVERSE, RISK_SEEKING]),
      lambda d: (_eq(d, RISK, RISK_AVERSE) & _eq(d, ATT_RISK, ENTHUSIAST))
      | (_eq(d, RISK, RISK_SEEKING) & _eq(d, ATT_RISK, OPPOSED))
      | (_eq(d, RISK, RISK_SEEKING) & _all_in(d, DOSPERT_COLS, ["Low", "Very low"]))
      | (_eq(d, RISK, RISK_AVERSE) & _all_in(d, DOSPERT_COLS, ["High", "Very high"])))
_rule("R2-PS-H09", G, "hard", "인지 스타일 모순 (큰 그림만↔세부지향 매우 높음, 세부 집착↔세부지향 낮음/없음, 즉흥 결정↔숙고형 결정)",
      lambda d: _in(d, BIGPIC, [BIGPIC_ONLY, BIGPIC_OBSESSED]) | _eq(d, DSPEED, DSPEED_SNAP),
      lambda d: (_eq(d, BIGPIC, BIGPIC_ONLY) & _eq(d, DETAIL, "Very high"))
      | (_eq(d, BIGPIC, BIGPIC_OBSESSED) & _in(d, DETAIL, ["Low", "None"]))
      | (_eq(d, DSPEED, DSPEED_SNAP) & _eq(d, DSTYLE, DSTYLE_DELIB)))
_rule("R2-PS-S01", G, "soft", "MBTI 외향/내향(E/I)과 사회적 에너지·BFI-2 외향성이 정반대 극단",
      lambda d: _in(d, MBTI, MBTI_ALL),
      lambda d: (_in(d, MBTI, MBTI_E) & (_eq(d, BATTERY, BATTERY_SI) | _eq(d, EXTRAV, "Very low")))
      | (_in(d, MBTI, MBTI_I) & (_eq(d, BATTERY, BATTERY_SE) | _eq(d, EXTRAV, "Very high"))))
G = "interests"


def _pair_scope_vals(d, pairs, va, vb):
    out = _F(d)
    for a, b in pairs:
        out |= _in(d, a, va) | _in(d, b, vb)
    return out


_rule("R2-IN-H01", G, "hard", "같은 주제의 관심도(topic_*와 acad_*, 정의 동일)가 열정↔혐오",
      lambda d: _pair_scope_vals(d, TOPIC_ACAD_PAIRS, [PASSIONATE, AVERSE], [PASSIONATE, AVERSE]),
      lambda d: np.any([(_eq(d, a, PASSIONATE) & _eq(d, b, AVERSE)) | (_eq(d, a, AVERSE) & _eq(d, b, PASSIONATE))
                        for a, b in TOPIC_ACAD_PAIRS], axis=0))
_rule("R2-IN-H02", G, "hard", "주제에 혐오(Averse)인데 같은 주제의 책 장르를 매우 좋아함 (시·철학·실화범죄·역사)",
      lambda d: _pair_scope_vals(d, TOPIC_BOOK_PAIRS, [AVERSE], [BOOK_LOVE]),
      lambda d: np.any([_eq(d, a, AVERSE) & _eq(d, b, BOOK_LOVE) for a, b in TOPIC_BOOK_PAIRS], axis=0))
_rule("R2-IN-H03", G, "hard", "주제에 혐오(Averse)인데 같은 활동을 최고 수준으로 함 (열성 취미·직접 운동·매일 명상/게임/팟캐스트)",
      lambda d: np.any([_eq(d, t, AVERSE) for t, _a, _v in TOPIC_ACTIVITY_TOP], axis=0),
      lambda d: np.any([_eq(d, t, AVERSE) & _eq(d, a, v) for t, a, v in TOPIC_ACTIVITY_TOP], axis=0))
_rule("R2-IN-S01", G, "soft", "주제에 열정(Passionate)인데 같은 활동을 전혀 하지 않음 (팟캐스트·게임·명상·요리·독서·운동·음악·공예 등)",
      lambda d: np.any([_eq(d, t, PASSIONATE) for t, _a, _v in TOPIC_ACTIVITY_ZERO], axis=0),
      lambda d: np.any([_eq(d, t, PASSIONATE) & _eq(d, a, v) for t, a, v in TOPIC_ACTIVITY_ZERO], axis=0))

G = "attitudes_politics"
_rule("R2-AT-H01", G, "hard", "정치 성향이 '비정치적'인데 정치 참여가 '활동가'",
      lambda d: _eq(d, LEAN, LEAN_APOLITICAL), lambda d: _eq(d, ENGAGE, ENGAGE_ACTIVIST))
_rule("R2-AT-H02", G, "hard", "근무 장소 선호와 재택/사무실 근무 태도가 정반대 (강한 재택 선호↔재택 반대·사무실 열광 등)",
      lambda d: _in(d, WORKLOC, [WORKLOC_OFFICE, WORKLOC_REMOTE]),
      lambda d: (_eq(d, WORKLOC, WORKLOC_REMOTE) & (_eq(d, ATT_REMOTE, OPPOSED) | _eq(d, ATT_OFFICE, ENTHUSIAST)))
      | (_eq(d, WORKLOC, WORKLOC_OFFICE) & (_eq(d, ATT_REMOTE, ENTHUSIAST) | _eq(d, ATT_OFFICE, OPPOSED))))
_rule("R2-AT-S01", G, "soft", "정치 활동가·조직가인데 정치/사회운동 주제에 혐오",
      lambda d: _eq(d, ENGAGE, ENGAGE_ACTIVIST) | _eq(d, SERVICE, SERVICE_ACTIVIST),
      lambda d: (_eq(d, ENGAGE, ENGAGE_ACTIVIST) & _eq(d, TOPIC_POL, AVERSE))
      | (_eq(d, SERVICE, SERVICE_ACTIVIST) & _eq(d, TOPIC_ACTIVISM, AVERSE)))
# register value strings used inline above
_VC(ORDER_COLS, "Very high", "Very low")
_V(EXTRAV, "Very high", "Very low")
_V(DETAIL, "Very high", "Low", "None")
_VC(DOSPERT_COLS, "Low", "Very low", "High", "Very high")
_V("fam_databases", "None")
_V(ENGP, "Native")
_V(LANG_EN, "Native")


# =====================================================================================
# engine
# =====================================================================================
def violation_matrix(df, rules=None):
    """Return (hits, scopes, status).  hits/scopes: bool DataFrames rows x rule ids,
    hits already ANDed with scope.  status: rule id -> 'ok' | 'skipped: missing <col>' |
    'error: <msg>'.  Never raises for a single rule."""
    rules = RULES if rules is None else rules
    n = len(df)
    hits, scopes, status = {}, {}, {}
    for r in rules:
        rid = r["id"]
        try:
            sc = r["scope"](df)
            ht = r["hit"](df)
            sc = np.asarray(sc.to_numpy() if isinstance(sc, pd.Series) else sc, dtype=bool)
            ht = np.asarray(ht.to_numpy() if isinstance(ht, pd.Series) else ht, dtype=bool)
            if sc.shape != (n,) or ht.shape != (n,):
                raise ValueError("bad shape")
            scopes[rid] = sc
            hits[rid] = ht & sc
            status[rid] = "ok"
        except MissingColumn as e:
            col = e.args[0] if e.args else "?"
            scopes[rid] = np.zeros(n, dtype=bool)
            hits[rid] = np.zeros(n, dtype=bool)
            status[rid] = "skipped: missing %s" % col
        except Exception as e:  # noqa: BLE001 - must never raise
            scopes[rid] = np.zeros(n, dtype=bool)
            hits[rid] = np.zeros(n, dtype=bool)
            status[rid] = "error: %s: %s" % (type(e).__name__, e)
    ids = [r["id"] for r in rules]
    hits_df = pd.DataFrame({i: hits[i] for i in ids}, index=df.index, columns=ids).astype(bool)
    scopes_df = pd.DataFrame({i: scopes[i] for i in ids}, index=df.index, columns=ids).astype(bool)
    return hits_df, scopes_df, status


def run(df, rules=None):
    """Return (table, any_hard_rate).  table: one row per rule with id, group, severity,
    description, n_scope, n_hit, violation_rate, status.  any_hard_rate: share of rows that
    violate at least one hard rule that ran ('ok')."""
    rules = RULES if rules is None else rules
    hits, scopes, status = violation_matrix(df, rules)
    rows = []
    for r in rules:
        rid = r["id"]
        n_scope = int(scopes[rid].sum())
        n_hit = int(hits[rid].sum())
        rows.append({
            "id": rid, "group": r["group"], "severity": r["severity"],
            "description": r["description"], "n_scope": n_scope, "n_hit": n_hit,
            "violation_rate": (n_hit / n_scope) if n_scope else np.nan,
            "status": status[rid],
        })
    table = pd.DataFrame(rows, columns=["id", "group", "severity", "description", "n_scope",
                                        "n_hit", "violation_rate", "status"])
    hard_ok = [r["id"] for r in rules if r["severity"] == "hard" and status[r["id"]] == "ok"]
    if len(df) == 0 or not hard_ok:
        any_hard_rate = np.nan
    else:
        any_hard_rate = float(hits[hard_ok].any(axis=1).mean())
    return table, any_hard_rate


# =====================================================================================
# catalog check
# =====================================================================================
def _catalog_frame(cat_vals):
    """Frame with every catalog column, cycling through each column's values."""
    n = max(len(v) for v in cat_vals.values())
    n = max(n, 2)
    data = {c: [vals[i % len(vals)] for i in range(n)] for c, vals in cat_vals.items()}
    return pd.DataFrame(data)


def check_values(catalog_path):
    """Verify every attribute id and value string referenced by the rules exists in the
    catalog (exact spelling).  Returns a list of problem strings (empty when all is well)."""
    with open(catalog_path, encoding="utf-8") as f:
        cat = json.load(f)
    cat_vals = {a["id"]: list(a["values"]) for a in cat["attributes"]}
    problems = []

    # 1) registered columns and values
    for col in sorted(_REF):
        if col not in cat_vals:
            problems.append("unknown attribute id: %s" % col)
            continue
        allowed = set(cat_vals[col])
        for v in sorted(_REF[col]):
            if v not in allowed:
                problems.append("unknown value for %s: %r" % (col, v))

    # 2) columns actually read by each rule (wrap _s) and rule health on a full frame
    frame = _catalog_frame(cat_vals)
    g = globals()
    orig = g["_s"]
    seen = {}

    for r in RULES:
        used = set()

        def tracking(df, col, _used=used):
            _used.add(col)
            return orig(df, col)

        g["_s"] = tracking
        try:
            r["scope"](frame)
            r["hit"](frame)
        except Exception as e:  # noqa: BLE001
            problems.append("rule %s fails on a full catalog frame: %s: %s" % (r["id"], type(e).__name__, e))
        finally:
            g["_s"] = orig
        seen[r["id"]] = used
        for col in used:
            if col not in cat_vals:
                problems.append("rule %s reads unknown attribute id: %s" % (r["id"], col))
            if col not in _REF:
                problems.append("rule %s reads unregistered column: %s" % (r["id"], col))

    # 3) rule metadata
    ids = [r["id"] for r in RULES]
    if len(ids) != len(set(ids)):
        problems.append("duplicate rule ids")
    for r in RULES:
        if not r["id"].startswith("R2-"):
            problems.append("bad id prefix: %s" % r["id"])
        sev_letter = r["id"].split("-")[2][0]
        if (sev_letter == "H") != (r["severity"] == "hard"):
            problems.append("id/severity mismatch: %s" % r["id"])
    return problems


if __name__ == "__main__":  # pragma: no cover
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "attribute_catalog.json"
    probs = check_values(path)
    n_h = sum(r["severity"] == "hard" for r in RULES)
    print("%d rules (%d hard, %d soft); %d problems" % (len(RULES), n_h, len(RULES) - n_h, len(probs)))
    for p in probs:
        print(" -", p)
