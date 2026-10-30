"""Self-tests for krpersona.rules.set2 (synthetic frames only; no generated persona data)."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from krpersona.rules import set2 as m  # noqa: E402

CAT = os.path.join(os.path.dirname(m.__file__), "catalogs", "set2.json")
with open(CAT, encoding="utf-8") as f:
    ATTRS = json.load(f)["attributes"]
VALUES = {a["id"]: a["values"] for a in ATTRS}

# ----------------------------------------------------------------------------------
# a consistent base persona (male, 35-44, married office worker with two young kids)
# ----------------------------------------------------------------------------------
NEUTRAL_ORDER = ["Neutral", "Average", "Moderate", "Balanced", "Mixed", "Either", "Situational",
                 "Never used", "None", "Never", "Unfamiliar", "Not applicable"]


def neutral(vals):
    for n in NEUTRAL_ORDER:
        if n in vals:
            return n
    return vals[len(vals) // 2]


BASE = {c: neutral(v) for c, v in VALUES.items()}
BASE.update({
    "age_bracket": "35-44", "gender_identity": "Man", "region": "East Asia",
    "urbanicity": "Dense urban", "demo_marital_status": "Married",
    "demo_citizenship_status": "Citizen by birth", "demo_employment_status": "Full-time",
    "lstyle_household_size": "3-4 people", "demo_disability_status": "No disability",
    "demo_driver_status": "Daily driver", "lifex_geographic_mobility": "Moved nationally",
    "demo_relationship_length": "5-15 years", "demo_generation": "Millennial",
    "lifex_formative_decade": "2000s", "lifex_immigration_generation": "Native multi-generational",
    "demo_ethnicity_broad": "East Asian", "cultural_background": "Collectivist (East Asian)",
    "demo_children_count": "2 children", "demo_parental_status": "Parent of minors",
    "lifex_parenting_journey": "Young children", "life_stage": "Parent of young kids",
    "demo_household_income": "$50k-100k", "socioeconomic_band": "Middle",
    "demo_housing_status": "Mortgage", "kr_depression_level": "Minimal",
    "kr_anxiety_level": "Minimal", "demo_religion_affiliation": "None",
    "lifex_faith_journey": "Always secular", "religiosity": "Secular",
    "demo_political_engagement": "Occasional voter", "lifex_travel_breadth": "A few countries",
    "demo_veteran_status": "Civilian", "lifex_childhood_environment": "Modest but steady",
    "lifex_career_path_shape": "Linear climb", "lifex_education_journey": "Traditional track",
    "lifex_financial_trajectory": "Stable", "lifex_adversity_level": "Minor setbacks",
    "lifex_military_history": "Served briefly", "lifex_entrepreneurship_history": "Considered it",
    "lifex_relationship_history": "Married once", "lifex_health_journey": "No major events",
    "lifex_cultural_exposure": "Some exposure", "lifex_service_history": "Occasional volunteer",
    "lifex_languages_lifetime": "Two", "lifex_hometown_tie": "Occasional return",
    "lifex_loss_experience": "Lost a grandparent", "major_life_events": "None notable",
    "demo_home_language": "Same as primary", "register": "Colloquial",
    "health_vision": "Corrected", "health_hearing": "Normal", "health_mobility": "Full",
    "health_dexterity": "Full", "years_experience": "11-20", "seniority": "Mid",
    "mbti_type": "ISTJ — Logistician", "health_chronic_condition": "None",
    "health_general_health": "Good", "lstyle_alcohol_use": "Socially",
    "lstyle_exercise_freq": "Weekly", "health_mental_health": "Stable",
    "health_insurance_status": "Comprehensive", "health_pain_level": "None",
    "health_medication_use": "None", "health_fitness_level": "Average",
    "accessibility_needs": "None", "health_assistive_tech": "None",
    "device_context": "Mobile, on-the-go", "lstyle_work_schedule": "9-to-5",
    "role_function": "Office / clerical", "company_size": "100-299 employees",
    "highest_education": "Bachelor's", "academic_field": "Business",
    "institution_tier": "Mid-tier university", "research_output": "None",
    "domain": "Business & Management", "subject_specialty": "Operations",
    "domain_characteristics": "Practitioner", "tech_savviness": "Comfortable",
    "pref_early_vs_late": "Mainstream", "political_lean": "Center",
    "primary_language": "Korean", "lang_korean": "Native", "lang_english": "Basic",
    "english_proficiency": "Basic (A1-A2)", "multilingualism": "Monolingual",
    "cult_south_korea": "Native", "cult_japan": "Visited",
    "tool_excel": "Regular", "tool_word": "Regular", "tool_chatgpt": "Occasional",
    "skill_driving": "Advanced", "skill_coding": "None",
    "dev_professional_status": "Non-developer technologist",
    "coding_ai_usage_frequency": "Never used", "coding_agent_usage_frequency": "No plans",
    "coding_agent_failure_tolerance": "Does not use agents",
    "coding_agent_workflow_impact": "No clear change",
    "coding_agent_autonomy_preference": "Suggest-only assistant",
    "coding_agent_control_expectation": "Requires explicit approval before edits",
    "open_source_activity_level": "No coding activity",
    "github_contribution_mode": "Rare GitHub user",
    "release_maintenance_orientation": "Not involved in releases",
    "lstyle_device_ecosystem": "Android/Google", "lstyle_browser": "Chrome",
    "lstyle_primary_messenger": "SMS", "lstyle_primary_social": "YouTube",
    "linkedin_activity": "Inactive", "media_diet": "Video-first", "lstyle_news_freq": "Daily",
    "lstyle_screen_time": "4-8 hrs", "lstyle_streaming_hours": "3-7",
    "lstyle_reading_freq": "Monthly", "lstyle_hobby_intensity": "Casual",
    "lstyle_gaming_freq": "Rarely", "lstyle_caffeine": "Moderate",
    "lstyle_coffee_ritual": "Café regular", "lstyle_banking_style": "Traditional bank",
    "lstyle_payment_pref": "Credit card", "lstyle_investment_style": "Index investor",
    "lstyle_travel_freq": "Occasional", "lstyle_diet_type": "Omnivore",
    "lstyle_social_battery": "Ambivert", "lstyle_tidiness": "Tidy",
    "lstyle_volunteering": "Rarely", "lstyle_planning_horizon": "Monthly",
    "pref_work_location": "Hybrid", "health_caregiver_status": "Child caregiver",
    "values_priority": "Security", "dominant_trait": "Balanced", "risk_tolerance": "Balanced",
    "kr_happiness": "Happy", "kr_life_satisfaction": "Satisfied", "kr_confucian_way_of_life": "Yes",
})
for c in VALUES:
    if c.startswith("cult_") and c not in ("cult_south_korea", "cult_japan"):
        BASE[c] = "Unfamiliar"
    if c.startswith("lang_") and c not in ("lang_korean", "lang_english"):
        BASE[c] = "None"
for c, v in BASE.items():
    assert v in VALUES[c], (c, v)


def frame(*mutations):
    rows = []
    for mu in mutations:
        r = dict(BASE)
        for k, v in mu.items():
            assert k in VALUES, k
            assert v in VALUES[k], (k, v)
            r[k] = v
        rows.append(r)
    return pd.DataFrame(rows, columns=list(VALUES))


def fired(df):
    hits, _, status = m.violation_matrix(df)
    assert all(s == "ok" for s in status.values()), {k: s for k, s in status.items() if s != "ok"}
    return [set(hits.columns[hits.iloc[i].to_numpy()]) for i in range(len(df))]


# ----------------------------------------------------------------------------------
# one contradictory mutation per rule
# ----------------------------------------------------------------------------------
NOLANG = {}
POSITIVE = {
    "R2-PO-H01": {"age_bracket": "13-17"},
    "R2-PO-H02": {"demo_citizenship_status": "Visa holder"},
    "R2-PO-H03": {"lifex_immigration_generation": "First-generation immigrant"},
    "R2-PO-H04": {"region": "Western Europe"},
    "R2-PO-H05": {"cult_south_korea": "Visited"},
    "R2-PO-S01": {"demo_citizenship_status": "Naturalized citizen"},
    "R2-PO-S02": {"demo_ethnicity_broad": "Multiracial"},
    "R2-PO-S03": {"english_proficiency": "Native", "lang_english": "Native",
                  "multilingualism": "Bilingual"},
    "R2-PO-S04": {"cult_south_korea": "Lived there"},
    "R2-AG-H01": {"demo_generation": "Boomer"},
    "R2-AG-H02": {"lifex_formative_decade": "1960s-70s"},
    "R2-AG-H03": {"age_bracket": "18-24", "years_experience": "20+", "demo_generation": "Gen Z",
                  "lifex_formative_decade": "2010s", "demo_marital_status": "Single",
                  "lifex_relationship_history": "A few relationships",
                  "demo_relationship_length": "Not in one", "demo_children_count": "None",
                  "demo_parental_status": "Not a parent", "lifex_parenting_journey": "No children",
                  "life_stage": "Early career"},
    "R2-AG-H04": {"age_bracket": "18-24", "demo_relationship_length": "15+ years",
                  "demo_generation": "Gen Z", "lifex_formative_decade": "2010s",
                  "years_experience": "0-2", "demo_marital_status": "In a relationship",
                  "lifex_relationship_history": "Long-term partnership",
                  "demo_children_count": "None", "demo_parental_status": "Not a parent",
                  "lifex_parenting_journey": "No children", "life_stage": "Early career"},
    "R2-AG-S01": {"age_bracket": "25-34", "years_experience": "20+", "demo_generation": "Millennial",
                  "lifex_formative_decade": "2000s"},
    "R2-AG-S02": {"age_bracket": "25-34", "demo_employment_status": "Retired",
                  "lstyle_work_schedule": "Flexible hours"},
    "R2-AG-S03": {"age_bracket": "65-74", "demo_generation": "Boomer",
                  "lifex_formative_decade": "1960s-70s", "demo_employment_status": "Student",
                  "lifex_parenting_journey": "Grown children", "life_stage": "Mid-life",
                  "demo_children_count": "Adult children", "demo_parental_status": "Parent of adults"},
    "R2-AG-S04": {"age_bracket": "18-24", "demo_generation": "Gen Z",
                  "lifex_formative_decade": "2010s", "years_experience": "0-2",
                  "demo_marital_status": "Single", "lifex_relationship_history": "Limited",
                  "demo_relationship_length": "Not in one", "demo_children_count": "None",
                  "demo_parental_status": "Not a parent", "lifex_parenting_journey": "No children",
                  "life_stage": "Early career", "ind_retail": "Veteran"},
    "R2-FM-H01": {"demo_relationship_length": "Not in one"},
    "R2-FM-H02": {"demo_marital_status": "Single", "lifex_relationship_history": "Widowed",
                  "lifex_loss_experience": "Lost a partner / child"},
    "R2-FM-H03": {"demo_children_count": "None", "lifex_parenting_journey": "No children",
                  "life_stage": "Mid-life"},
    "R2-FM-H04": {"lifex_parenting_journey": "No children", "life_stage": "Mid-life",
                  "demo_parental_status": "Step / foster parent"},
    "R2-FM-H05": {"demo_parental_status": "Not a parent", "demo_children_count": "Expecting"},
    "R2-FM-H06": {"life_stage": "Empty nester"},
    "R2-FM-H07": {"age_bracket": "18-24", "demo_generation": "Gen Z",
                  "lifex_formative_decade": "2010s", "years_experience": "0-2",
                  "lifex_parenting_journey": "Teenagers", "life_stage": "Mid-life"},
    "R2-FM-H08": {"gender_identity": "Woman", "age_bracket": "55-64", "demo_generation": "Boomer",
                  "lifex_formative_decade": "1980s", "demo_children_count": "Expecting",
                  "demo_parental_status": "Step / foster parent",
                  "lifex_parenting_journey": "Young children", "lifex_military_history": "Never served"},
    "R2-FM-H09": {"lstyle_household_size": "Lives alone", "demo_housing_status": "Living with family"},
    "R2-FM-H10": {"demo_marital_status": "Widowed", "lifex_relationship_history": "Widowed",
                  "lifex_loss_experience": "No major loss"},
    "R2-FM-S01": {"age_bracket": "25-34", "lifex_parenting_journey": "Grown children",
                  "life_stage": "Mid-life", "demo_children_count": "Adult children",
                  "demo_parental_status": "Parent of adults", "years_experience": "6-10"},
    "R2-FM-S02": {"age_bracket": "75-84", "demo_generation": "Silent",
                  "lifex_formative_decade": "1960s-70s", "years_experience": "20+"},
    "R2-FM-S03": {"age_bracket": "18-24", "demo_generation": "Gen Z",
                  "lifex_formative_decade": "2010s", "years_experience": "0-2",
                  "demo_marital_status": "Divorced", "demo_relationship_length": "Not in one"},
    "R2-FM-S04": {"demo_marital_status": "Widowed", "lifex_relationship_history": "Widowed",
                  "lifex_loss_experience": "Lost a parent"},
    "R2-MI-H01": {"demo_veteran_status": "Veteran", "lifex_military_history": "Never served"},
    "R2-MI-H02": {"demo_veteran_status": "Active duty", "age_bracket": "65-74",
                  "demo_generation": "Boomer", "lifex_formative_decade": "1980s",
                  "lifex_military_history": "Career veteran"},
    "R2-MI-H03": {"major_life_events": "Military service", "lifex_military_history": "Never served"},
    "R2-MI-S01": {"gender_identity": "Woman"},
    "R2-MI-S02": {"lifex_military_history": "Combat veteran", "demo_veteran_status": "Veteran"},
    "R2-MI-S03": {"lifex_military_history": "Career veteran"},
    "R2-ED-H01": {"lifex_education_journey": "Advanced degrees", "highest_education": "Associate's",
                  "institution_tier": "Community college"},
    "R2-ED-H02": {"institution_tier": "Self-taught"},
    "R2-ED-H03": {"highest_education": "Primary"},
    "R2-ED-H04": {"research_output": "Thesis only", "highest_education": "Secondary",
                  "institution_tier": "Not applicable"},
    "R2-ED-S01": {"age_bracket": "18-24", "demo_generation": "Gen Z",
                  "lifex_formative_decade": "2010s", "years_experience": "0-2",
                  "highest_education": "Doctorate", "lifex_education_journey": "Advanced degrees",
                  "demo_children_count": "1 child", "demo_parental_status": "New parent"},
    "R2-ED-S02": {"highest_education": "Secondary", "institution_tier": "Not applicable",
                  "fam_surgery": "Expert"},
    "R2-EX-H01": {"subject_specialty": "Cardiology"},
    "R2-EM-H01": {"lstyle_work_schedule": "Unemployed"},
    "R2-EM-H02": {"company_size": "Not applicable"},
    "R2-EM-H03": {"seniority": "Retired"},
    "R2-EM-H04": {"demo_employment_status": "Retired", "lstyle_work_schedule": "Flexible hours",
                  "life_stage": "Early career", "lifex_career_path_shape": "Pivoted once"},
    "R2-EM-H05": {"lifex_career_path_shape": "Just starting", "years_experience": "20+"},
    "R2-EM-H06": {"seniority": "Founder", "lifex_entrepreneurship_history": "Never considered"},
    "R2-EM-S01": {"seniority": "Director", "years_experience": "0-2"},
    "R2-EM-S02": {"life_stage": "Retirement", "lifex_parenting_journey": "Grown children",
                  "demo_children_count": "Adult children", "demo_parental_status": "Parent of adults",
                  "age_bracket": "55-64", "demo_generation": "Boomer",
                  "lifex_formative_decade": "1980s", "years_experience": "20+"},
    "R2-EC-H01": {"demo_household_income": "$200k+", "socioeconomic_band": "Low income"},
    "R2-EC-H02": {"lstyle_banking_style": "Unbanked", "demo_housing_status": "Renting"},
    "R2-EC-H03": {"lstyle_investment_style": "Crypto-heavy", "att_cryptocurrency": "Opposed"},
    "R2-EC-H04": {"pref_save_vs_spend": "Hard saver", "lstyle_frugality": "Splurger"},
    "R2-EC-S01": {"demo_household_income": "<$25k", "socioeconomic_band": "High income"},
    "R2-EC-S02": {"lstyle_banking_style": "Mostly cash"},
    "R2-LA-H01": {"lang_korean": "Basic"},
    "R2-LA-H02": {"english_proficiency": "Fluent (C1-C2)", "lang_english": "Basic"},
    "R2-LA-H03": {"multilingualism": "Trilingual+", "lifex_languages_lifetime": "Four+"},
    "R2-LA-H04": {"lifex_languages_lifetime": "One", "lang_english": "Conversational",
                  "english_proficiency": "Intermediate (B1-B2)", "multilingualism": "Bilingual"},
    "R2-LA-H05": {"register": "Technical jargon", "cog_use_of_jargon": "Avoids jargon"},
    "R2-LA-S01": {"lang_korean": "Conversational", "lang_english": "Conversational",
                  "english_proficiency": "Intermediate (B1-B2)", "multilingualism": "Bilingual"},
    "R2-LA-S02": {"english_proficiency": "Native", "lang_english": "Fluent",
                  "multilingualism": "Bilingual"},
    "R2-LA-S03": {"demo_home_language": "Bilingual home"},
    "R2-MO-H01": {"lifex_travel_breadth": "Never left country"},
    "R2-MO-H02": {"lifex_travel_breadth": "Never left country", "cult_japan": "Studied",
                  "lifex_immigration_generation": "Returnee / repatriate"},
    "R2-MO-H03": {"lifex_geographic_mobility": "Never left hometown"},
    "R2-MO-H04": {"lifex_geographic_mobility": "Never left hometown", "lifex_hometown_tie": "Still there",
                  "lifex_travel_breadth": "Lived abroad"},
    "R2-MO-H05": {"lifex_travel_breadth": "Global nomad", "lstyle_travel_freq": "Homebody"},
    "R2-MO-S01": {"lifex_travel_breadth": "Never left country", "cult_japan": "Studied",
                  "lstyle_travel_freq": "Frequent flyer"},
    "R2-MO-S02": {"lifex_cultural_exposure": "Monocultural", "lifex_travel_breadth": "Lived abroad"},
    "R2-RE-H01": {"religiosity": "Devout", "lifex_faith_journey": "Still searching"},
    "R2-RE-H02": {"lifex_faith_journey": "Lifelong faith", "religiosity": "Spiritual"},
    "R2-RE-H03": {"lifex_faith_journey": "Lapsed / left faith", "religiosity": "Observant",
                  "demo_religion_affiliation": "Catholic"},
    "R2-RE-S01": {"demo_religion_affiliation": "Buddhist", "religiosity": "Nominal"},
    "R2-RE-S02": {"religiosity": "Devout", "demo_religion_affiliation": "Protestant",
                  "lifex_faith_journey": "Lifelong faith", "trait_spirituality": "Absent"},
    "R2-RE-S03": {"health_dietary_restriction": "Religious"},
    "R2-HE-H01": {"health_vision": "Blind", "demo_driver_status": "Non-driver",
                  "accessibility_needs": "Visual", "health_assistive_tech": "Screen reader",
                  "skill_driving": "None"},
    "R2-HE-H02": {"demo_disability_status": "Sensory"},
    "R2-HE-H03": {"lifex_health_journey": "Manages chronic condition"},
    "R2-HE-H04": {"health_vision": "Blind", "demo_disability_status": "Sensory",
                  "accessibility_needs": "Visual", "health_assistive_tech": "Screen reader"},
    "R2-HE-H05": {"accessibility_needs": "Hearing"},
    "R2-HE-H06": {"device_context": "Accessibility tool"},
    "R2-HE-H07": {"neurotype": "ADHD"},
    "R2-HE-H08": {"lstyle_exercise_freq": "Never", "sport_soccer": "Play"},
    "R2-HE-H09": {"health_mental_health": "Thriving", "kr_depression_level": "Severe"},
    "R2-HE-S01": {"health_mobility": "Uses mobility aid"},
    "R2-HE-S02": {"health_vision": "Normal", "health_text_size_need": "Requires"},
    "R2-HE-S03": {"health_fitness_level": "Sedentary", "lstyle_exercise_freq": "Daily"},
    "R2-HE-S04": {"health_mental_health": "Thriving", "kr_life_satisfaction": "Very dissatisfied"},
    "R2-HE-S05": {"kr_happiness": "Very happy", "kr_life_satisfaction": "Very dissatisfied"},
    "R2-HE-S06": {"health_pain_level": "Severe"},
    "R2-HE-S07": {"health_medication_use": "Multiple daily"},
    "R2-HE-S08": {"health_insurance_status": "Uninsured"},
    "R2-EV-H01": {"major_life_events": "Started a business"},
    "R2-EV-H02": {"major_life_events": "Bereavement", "lifex_loss_experience": "No major loss"},
    "R2-EV-H03": {"major_life_events": "Health journey"},
    "R2-EV-H04": {"major_life_events": "Major relocation", "lifex_geographic_mobility": "Never left hometown",
                  "lifex_hometown_tie": "Still there"},
    "R2-EV-H05": {"lifex_childhood_environment": "Hardship", "lifex_adversity_level": "Sheltered"},
    "R2-EV-H06": {"lstyle_volunteering": "Weekly", "lifex_service_history": "None"},
    "R2-EV-S01": {"lifex_childhood_environment": "Turbulent", "lifex_adversity_level": "Sheltered"},
    "R2-EV-S02": {"age_bracket": "75-84", "demo_generation": "Silent",
                  "lifex_formative_decade": "1960s-70s", "years_experience": "20+",
                  "lifex_loss_experience": "No major loss", "lifex_parenting_journey": "Grown children",
                  "life_stage": "Retirement", "demo_children_count": "Adult children",
                  "demo_parental_status": "Grandparent", "demo_employment_status": "Retired",
                  "lstyle_work_schedule": "Flexible hours"},
    "R2-LS-H01": {"lstyle_browser": "Safari"},
    "R2-LS-H02": {"lstyle_primary_social": "None", "media_diet": "Social-media-heavy"},
    "R2-LS-H03": {"lstyle_news_freq": "Avoids news", "media_diet": "News"},
    "R2-LS-H04": {"lstyle_screen_time": "<2 hrs", "lstyle_streaming_hours": "16+"},
    "R2-LS-H05": {"lstyle_reading_freq": "Never", "habit_reading_before_bed": "Daily"},
    "R2-LS-H06": {"lstyle_hobby_intensity": "None", "hob_pottery": "Avid"},
    "R2-LS-H07": {"lstyle_diet_type": "Vegan", "att_veganism": "Opposed"},
    "R2-LS-H08": {"skill_driving": "None"},
    "R2-LS-H09": {"lstyle_gaming_freq": "Never", "sport_esports": "Play"},
    "R2-LS-S01": {"tool_keynote": "Regular"},
    "R2-LS-S03": {"lstyle_alcohol_use": "Never", "topic_wine": "Passionate"},
    "R2-TE-H01": {"prog_python": "Expert", "skill_coding": "Advanced", "tool_python": "Never used"},
    "R2-TE-H02": {"prog_java": "Proficient"},
    "R2-TE-H03": {"prog_typescript": "Proficient", "skill_coding": "Intermediate"},
    "R2-TE-H04": {"tool_aws": "Power user", "fam_cloud_infrastructure": "None"},
    "R2-TE-H05": {"tool_jupyter": "Regular"},
    "R2-TE-H06": {"tech_savviness": "Avoidant", "pref_early_vs_late": "Bleeding edge"},
    "R2-TE-S01": {"skill_coding": "Master"},
    "R2-TE-S02": {"tool_kubernetes": "Power user", "tool_docker": "Never used"},
    "R2-DV-H01": {"ai_task_testing": "Currently partially AI-assisted"},
    "R2-DV-H02": {"coding_agent_usage_frequency": "Weekly",
                  "coding_agent_failure_tolerance": "Moderate tolerance for prototypes"},
    "R2-DV-H03": {"coding_agent_workflow_impact": "Major productivity increase"},
    "R2-DV-H04": {"coding_agent_control_expectation": "Agent may edit within scoped files"},
    "R2-DV-H05": {"dev_professional_status": "Hobbyist / side-project coder",
                  "open_source_activity_level": "Private-only developer"},
    "R2-DV-H06": {"github_contribution_mode": "Code pusher", "tool_git": "Regular"},
    "R2-DV-H07": {"github_contribution_mode": "Pull request contributor",
                  "open_source_activity_level": "Occasional contributor"},
    "R2-DV-S01": {"coding_ai_usage_frequency": "Weekly", "coding_agent_usage_frequency": "Daily",
                  "coding_agent_failure_tolerance": "Low tolerance with easy rollback",
                  "coding_agent_workflow_impact": "Moderate productivity increase"},
    "R2-DV-S02": {"dev_professional_status": "Professional developer", "skill_coding": "Beginner",
                  "prog_python": "Familiar", "open_source_activity_level": "Private-only developer"},
    "R2-DV-S03": {"coding_ai_output_trust": "Does not trust", "coding_ai_sentiment": "Very favorable"},
    "R2-DV-S04": {"att_ai": "Opposed", "tool_claude": "Power user"},
    "R2-PS-H01": {"bfi2_domain_extraversion": "Very high", "bfi2_facet_sociability": "Low",
                  "bfi2_facet_assertiveness": "Very low", "bfi2_facet_energy_level": "Low"},
    "R2-PS-H02": {"dominant_trait": "High openness", "bfi2_domain_open_mindedness": "Low"},
    "R2-PS-H03": {"bfi2_facet_anxiety": "Very high", "big5_anxiety": "Very low"},
    "R2-PS-H04": {"trait_curiosity": "Signature", "cog_curiosity": "None"},
    "R2-PS-H05": {"lstyle_tidiness": "Chaotic", "big5_orderliness": "Very high"},
    "R2-PS-H06": {"val_tradition": "Core value", "schwartz_value_tradition": "Very low"},
    "R2-PS-H07": {"values_priority": "Achievement", "val_achievement": "Irrelevant"},
    "R2-PS-H08": {"risk_tolerance": "Risk-seeking", "att_risk_taking": "Opposed"},
    "R2-PS-H09": {"cog_decision_speed": "Snap decisions", "decision_style": "Deliberative"},
    "R2-PS-S01": {"mbti_type": "ENFP — Campaigner", "lstyle_social_battery": "Strong introvert"},
    "R2-IN-H01": {"topic_philosophy": "Passionate", "acad_philosophy": "Averse"},
    "R2-IN-H02": {"topic_history": "Averse", "bookg_history": "Love"},
    "R2-IN-H03": {"topic_gaming": "Averse", "lstyle_gaming_freq": "Daily"},
    "R2-IN-S01": {"topic_podcasts": "Passionate", "lstyle_podcast_listening": "Never"},
    "R2-AT-H01": {"political_lean": "Apolitical", "demo_political_engagement": "Activist"},
    "R2-AT-H02": {"pref_work_location": "Strongly remote", "att_remote_work": "Opposed"},
    "R2-AT-S01": {"demo_political_engagement": "Activist", "topic_politics": "Averse"},
}

# ----------------------------------------------------------------------------------
# consistent near-boundary rows: (label, mutation, rules that must NOT fire)
# ----------------------------------------------------------------------------------
YOUNG = {"age_bracket": "18-24", "demo_generation": "Gen Z", "lifex_formative_decade": "2010s",
         "years_experience": "0-2", "demo_marital_status": "Single",
         "lifex_relationship_history": "A few relationships", "demo_relationship_length": "Not in one",
         "demo_children_count": "None", "demo_parental_status": "Not a parent",
         "lifex_parenting_journey": "No children", "life_stage": "Student",
         "demo_employment_status": "Student", "lstyle_work_schedule": "Flexible hours",
         "highest_education": "Secondary", "institution_tier": "Mid-tier university",
         "health_caregiver_status": "Not a caregiver"}
OLD = {"age_bracket": "85+", "demo_generation": "Silent", "lifex_formative_decade": "1960s-70s",
       "years_experience": "20+", "demo_employment_status": "Retired", "life_stage": "Retirement",
       "seniority": "Retired", "lstyle_work_schedule": "Flexible hours",
       "demo_marital_status": "Widowed", "lifex_relationship_history": "Widowed",
       "lifex_loss_experience": "Multiple bereavements", "demo_children_count": "Adult children",
       "demo_parental_status": "Grandparent", "lifex_parenting_journey": "Grown children",
       "health_caregiver_status": "Not a caregiver", "lifex_military_history": "Combat veteran",
       "demo_veteran_status": "Veteran", "highest_education": "Primary",
       "institution_tier": "Not applicable"}
NEGATIVE = [
    ("base persona", {}, None),   # None = no rule at all may fire
    ("young student (Gen Z, 2010s)", YOUNG, None),
    ("oldest bracket, widowed Korean War veteran", OLD, None),
    ("25-34 Gen Z boundary", {"age_bracket": "25-34", "demo_generation": "Gen Z",
                              "lifex_formative_decade": "2010s", "years_experience": "6-10"}, None),
    ("45-54 Boomer (Korean 2nd boom)", {"age_bracket": "45-54", "demo_generation": "Boomer",
                                        "lifex_formative_decade": "1980s", "years_experience": "20+",
                                        "lifex_parenting_journey": "Teenagers",
                                        "life_stage": "Mid-life"}, None),
    ("divorced, single now", {"demo_marital_status": "Divorced", "demo_relationship_length": "Not in one",
                              "lifex_relationship_history": "Married once"}, None),
    ("expecting woman 35-44", {"gender_identity": "Woman", "demo_children_count": "Expecting",
                               "demo_parental_status": "Not a parent",
                               "lifex_parenting_journey": "No children", "life_stage": "Mid-life",
                               "lifex_military_history": "Never served"}, None),
    ("trilingual, consistent", {"lang_english": "Fluent", "english_proficiency": "Fluent (C1-C2)",
                                "lang_japanese": "Conversational", "multilingualism": "Trilingual+",
                                "lifex_languages_lifetime": "Three"}, None),
    ("bilingual, English 1-step mismatch", {"lang_english": "Conversational",
                                            "english_proficiency": "Fluent (C1-C2)",
                                            "multilingualism": "Bilingual"}, None),
    ("never left country, only studied cultures", {"lifex_travel_breadth": "Never left country",
                                                   "cult_japan": "Studied"}, None),
    ("never exercises, plays screen golf & esports", {"lstyle_exercise_freq": "Never",
                                                      "sport_golf": "Play", "sport_esports": "Play",
                                                      "lstyle_gaming_freq": "Daily"}, None),
    ("professional developer, consistent", {
        "dev_professional_status": "Professional developer", "skill_coding": "Advanced",
        "prog_python": "Expert", "tool_python": "Power user", "prog_javascript": "Proficient",
        "prog_typescript": "Proficient", "prog_sql": "Proficient", "tool_sql": "Regular",
        "fam_databases": "Proficient", "tool_git": "Power user", "tool_github": "Power user",
        "tool_vs_code": "Power user", "tool_github_copilot": "Regular",
        "tech_savviness": "Digital native", "coding_ai_usage_frequency": "Daily",
        "ai_task_code_generation": "Currently mostly AI-assisted",
        "coding_agent_usage_frequency": "Weekly",
        "coding_agent_failure_tolerance": "Moderate tolerance for prototypes",
        "coding_agent_workflow_impact": "Moderate productivity increase",
        "coding_agent_autonomy_preference": "Plan-then-act with checkpoints",
        "coding_agent_control_expectation": "Approval before risky edits only",
        "open_source_activity_level": "Occasional contributor",
        "github_contribution_mode": "Pull request contributor",
        "role_function": "Engineering", "domain": "Software & AI",
        "subject_specialty": "Machine learning"}, None),
    ("agent experimental, AI never", {"coding_agent_usage_frequency": "Experimental only",
                                      "coding_agent_failure_tolerance": "High tolerance for experimental work"}, None),
    ("specialty taught in Education domain", {"domain": "Education", "subject_specialty": "Cardiology"}, None),
    ("culinary arts in hospitality", {"domain": "Hospitality", "subject_specialty": "Culinary arts",
                                      "role_function": "Service"}, None),
    ("BFI-2 domain high, mixed facets", {"bfi2_domain_extraversion": "High",
                                         "bfi2_facet_sociability": "High",
                                         "bfi2_facet_assertiveness": "Average",
                                         "bfi2_facet_energy_level": "Low"}, None),
    ("extravert MBTI, extrovert battery", {"mbti_type": "ENFP — Campaigner",
                                           "lstyle_social_battery": "Extrovert"}, None),
    ("active duty officer 55-64", {"age_bracket": "55-64", "demo_generation": "Boomer",
                                   "lifex_formative_decade": "1980s", "years_experience": "20+",
                                   "demo_veteran_status": "Active duty",
                                   "lifex_military_history": "Career veteran",
                                   "lifex_parenting_journey": "Grown children",
                                   "demo_children_count": "Adult children",
                                   "demo_parental_status": "Parent of adults",
                                   "life_stage": "Mid-life"},
     {"R2-MI-H02"}),   # S-rule for 55-64 was intentionally not added; check hard one only
    ("low-vision driver", {"health_vision": "Low vision"}, None),
    ("non-coder uses AI only for search/learning", {"ai_task_search_answers": "Currently mostly AI-assisted",
                                                    "ai_task_learning_concepts": "Currently partially AI-assisted"}, None),
    ("never left hometown but childhood moves inside it", {"lifex_geographic_mobility": "Never left hometown",
                                                          "lifex_hometown_tie": "Still there",
                                                          "lifex_childhood_environment": "Frequently uprooted"}, None),
    ("university student: highest completed = Secondary", {"highest_education": "Secondary",
                                                           "institution_tier": "Top-tier research",
                                                           "demo_employment_status": "Student",
                                                           "lstyle_work_schedule": "Flexible hours",
                                                           "role_function": "Not applicable"}, None),
    ("devout Catholic, consistent", {"demo_religion_affiliation": "Catholic", "religiosity": "Devout",
                                     "lifex_faith_journey": "Lifelong faith",
                                     "val_spirituality_faith": "Core value"}, None),
]


def test_check_values():
    probs = m.check_values(CAT)
    assert probs == [], probs
    print("check_values: [] (ok)")


def test_full_frame_runs():
    df = frame({})
    _, _, status = m.violation_matrix(df)
    bad = {k: v for k, v in status.items() if v != "ok"}
    assert not bad, bad
    # random catalog-valued frame: every rule must run
    rng = np.random.default_rng(0)
    rnd = pd.DataFrame({c: rng.choice(v, size=3000) for c, v in VALUES.items()})
    table, rate = m.run(rnd)
    assert (table["status"] == "ok").all(), table[table["status"] != "ok"]
    assert 0.0 <= rate <= 1.0
    assert (table["n_hit"] <= table["n_scope"]).all()
    print("full frame: all %d rules ok; random-frame any_hard_rate=%.3f" % (len(m.RULES), rate))


def test_positive():
    ids = {r["id"] for r in m.RULES}
    assert set(POSITIVE) == ids, (sorted(ids - set(POSITIVE)), sorted(set(POSITIVE) - ids))
    rule_ids = list(POSITIVE)
    res = fired(frame(*[POSITIVE[i] for i in rule_ids]))
    missed = [i for i, f in zip(rule_ids, res) if i not in f]
    assert not missed, missed
    extra = {i: sorted(f - {i}) for i, f in zip(rule_ids, res) if f - {i}}
    print("positive: %d/%d rules caught their hand-made contradiction" % (len(rule_ids) - len(missed), len(rule_ids)))
    if extra:
        print("  (rows that also tripped other rules: %d)" % len(extra))
        for k, v in extra.items():
            print("   ", k, "->", v)


def test_negative():
    res = fired(frame(*[mu for _, mu, _ in NEGATIVE]))
    failures = []
    for (label, _mu, must_not), f in zip(NEGATIVE, res):
        bad = f if must_not is None else (f & must_not)
        if bad:
            failures.append((label, sorted(bad)))
    assert not failures, failures
    print("negative: %d consistent rows, no false hits" % len(NEGATIVE))


def test_missing_columns():
    df = frame({}, POSITIVE["R2-FM-H08"])
    dropped = df.drop(columns=["age_bracket", "lang_swahili"])
    hits, scopes, status = m.violation_matrix(dropped)
    skipped = {k for k, v in status.items() if v.startswith("skipped")}
    assert skipped, "nothing skipped"
    for k in skipped:
        assert status[k] in ("skipped: missing age_bracket", "skipped: missing lang_swahili"), status[k]
        assert not hits[k].any() and not scopes[k].any()
    # the rules that do not read those columns still ran
    ok = {k for k, v in status.items() if v == "ok"}
    assert "R2-FM-H01" in ok and "R2-AG-H01" in skipped and "R2-LA-H03" in skipped
    assert "R2-FM-H08" in skipped
    # error handling: a broken rule is reported, never raised
    broken = [{"id": "R2-XX-H99", "group": "t", "severity": "hard", "description": "x",
               "scope": lambda d: pd.Series(True, index=d.index),
               "hit": lambda d: 1 / 0}]
    _, _, st = m.violation_matrix(df, broken)
    assert st["R2-XX-H99"].startswith("error:"), st
    # empty frame and frame with nothing
    t0, r0 = m.run(df.iloc[0:0])
    assert np.isnan(r0) and t0["violation_rate"].isna().all()
    t1, r1 = m.run(pd.DataFrame({"x": ["a", "b"]}))
    assert (t1["status"].str.startswith("skipped")).all() and np.isnan(r1)
    print("missing columns: %d rules skipped cleanly, %d ran; errors reported; empty frames ok"
          % (len(skipped), len(ok)))


def test_external_column_discovery():
    """The caller discovers columns by wrapping _s; make sure that works."""
    df = frame({})
    orig = m._s
    used = {}
    try:
        for r in m.RULES:
            cols = set()

            def wrap(d, c, _cols=cols):
                _cols.add(c)
                return orig(d, c)
            m._s = wrap
            r["scope"](df)
            r["hit"](df)
            used[r["id"]] = cols
    finally:
        m._s = orig
    assert all(used.values()), [k for k, v in used.items() if not v]
    n_cols = len(set().union(*used.values()))
    print("column discovery via _s wrap: %d distinct columns across %d rules" % (n_cols, len(used)))


def test_run_table():
    df = frame({}, POSITIVE["R2-LA-H02"], POSITIVE["R2-HE-H07"], POSITIVE["R2-PO-S02"])
    table, rate = m.run(df)
    assert list(table.columns) == ["id", "group", "severity", "description", "n_scope", "n_hit",
                                   "violation_rate", "status"]
    row = table.set_index("id").loc["R2-LA-H02"]
    assert row["n_hit"] == 1 and row["n_scope"] == 4 and abs(row["violation_rate"] - 0.25) < 1e-12
    assert abs(rate - 0.5) < 1e-12, rate   # rows 1 and 2 violate hard rules, row 3 only a soft one
    print("run(): table shape %s, any_hard_rate=%.2f (expected 0.50)" % (table.shape, rate))


if __name__ == "__main__":
    test_check_values()
    test_full_frame_runs()
    test_positive()
    test_negative()
    test_missing_columns()
    test_external_column_discovery()
    test_run_table()
    print("ALL TESTS PASSED")
