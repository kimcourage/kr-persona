# -*- coding: utf-8 -*-
"""Tests for krpersona.rules.set3.  Run:  python -m pytest tests/rules/test_set3.py   (pytest-compatible too).

Builds synthetic frames from catalog values only (no generated persona data):
  * a fully consistent base persona that triggers no rule,
  * one hand-made contradictory row per rule (base + overrides) that the rule must catch,
  * consistent rows near rule boundaries that must trigger nothing,
  * categorical / NaN / missing-column / error-handling / API checks, and check_values() == [].
"""
import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from krpersona.rules import set3 as m  # noqa: E402

CATALOG = os.path.join(os.path.dirname(m.__file__), "catalogs", "set3.json")
with open(CATALOG, encoding="utf-8") as _fh:
    _CAT = json.load(_fh)
VALUES = {a["id"]: list(a["values"]) for a in _CAT["attributes"]}
IDS = list(VALUES)

# -----------------------------------------------------------------------------
# A consistent base persona: 35-44 married man, office worker, hobby coder.
# -----------------------------------------------------------------------------
FAMILY_DEFAULTS = (
    ("bfi2_", "Average"), ("big5_", "Average"), ("schwartz_value_", "Average"),
    ("dospert_", "Average"), ("mft_", "Average"), ("sdt_", "Average"), ("attachment_", "Average"),
    ("interpersonal_", "Average"), ("trait_", "Moderate"), ("val_", "Moderate"),
    ("topic_", "Neutral"), ("acad_", "Neutral"), ("att_", "Neutral"), ("fam_", "Familiar"),
    ("skill_", "Intermediate"), ("prog_", "Familiar"), ("tool_", "Occasional"),
    ("habit_", "Monthly"), ("hob_", "Occasional"), ("lang_", "None"), ("cult_", "Unfamiliar"),
    ("musg_", "Like"), ("filmg_", "Like"), ("bookg_", "Like"), ("cuis_", "Like"),
    ("sport_", "Casual"), ("ind_", "Some exposure"), ("ai_task_", "Currently partially AI-assisted"),
    ("peeve_", "Annoys"), ("coding_tool_", None), ("kr_trust_", "Moderate"),
)

EXPLICIT = {
    "age_bracket": "35-44", "gender_identity": "Man", "region": "East Asia",
    "urbanicity": "Dense urban", "demo_marital_status": "Married",
    "demo_citizenship_status": "Citizen by birth", "demo_employment_status": "Full-time",
    "demo_disability_status": "No disability", "demo_driver_status": "Occasional driver",
    "demo_relationship_length": "5-15 years", "demo_generation": "Millennial",
    "demo_ethnicity_broad": "East Asian", "demo_children_count": "2 children",
    "demo_parental_status": "Parent of minors", "demo_household_income": "$50k-100k",
    "socioeconomic_band": "Middle", "demo_housing_status": "Mortgage",
    "demo_religion_affiliation": "None", "demo_political_engagement": "Occasional voter",
    "demo_birth_order": "Eldest", "demo_veteran_status": "Veteran",
    "demo_sexual_orientation": "Heterosexual", "demo_home_language": "Same as primary",
    "register": "Colloquial", "lstyle_household_size": "3-4 people",
    "lifex_geographic_mobility": "Moved nationally", "lifex_formative_decade": "1990s",
    "lifex_immigration_generation": "Native multi-generational",
    "lifex_parenting_journey": "Young children", "life_stage": "Parent of young kids",
    "lifex_faith_journey": "Always secular", "lifex_travel_breadth": "A few countries",
    "lifex_childhood_environment": "Modest but steady", "lifex_career_path_shape": "Linear climb",
    "lifex_education_journey": "Traditional track", "lifex_financial_trajectory": "Stable",
    "lifex_adversity_level": "Minor setbacks", "lifex_military_history": "Served briefly",
    "lifex_entrepreneurship_history": "Considered it", "lifex_relationship_history": "Married once",
    "lifex_health_journey": "No major events", "lifex_cultural_exposure": "Some exposure",
    "lifex_public_recognition": "Private life", "lifex_service_history": "Occasional volunteer",
    "lifex_turning_point": "No single one", "lifex_languages_lifetime": "Two",
    "lifex_hometown_tie": "Visits often", "lifex_loss_experience": "Lost a grandparent",
    "major_life_events": "Military service", "cultural_background": "Collectivist (East Asian)",
    "kr_depression_level": "Minimal", "kr_anxiety_level": "Minimal", "kr_subjective_class": "Middle",
    "kr_view_of_north_korea": "Someone to be wary of", "kr_unification_need": "Somewhat necessary",
    "kr_country_felt_closest": "United States", "kr_life_satisfaction": "Satisfied",
    "kr_happiness": "Happy", "kr_belief_in_saju": "Unsure",
    "kr_attitude_same_sex_relations": "Sometimes wrong",
    "kr_family_wealth_for_success": "Fairly important",
    "kr_living_standard_vs_parents": "Somewhat better", "kr_next_generation_outlook": "About the same",
    "kr_confidants": "A few", "kr_confucian_way_of_life": "No",
    "kr_ancestry_for_korean_identity": "Fairly important", "kr_political_efficacy": "Moderate",
    "kr_children_necessity": "Necessary",
    "highest_education": "Bachelor's", "academic_field": "Business",
    "institution_tier": "Mid-tier university", "years_experience": "11-20", "seniority": "Senior",
    "linkedin_activity": "Lurker", "research_output": "None", "role_function": "Office / clerical",
    "company_size": "SMB (50-500)", "learning_style": "Mixed", "lstyle_work_schedule": "9-to-5",
    "pref_work_location": "Hybrid",
    "health_vision": "Corrected", "health_hearing": "Normal", "health_mobility": "Full",
    "health_chronic_condition": "None", "health_general_health": "Good",
    "health_stress_level": "Moderate", "health_mental_health": "Stable",
    "health_insurance_status": "Comprehensive", "health_energy_level": "Moderate",
    "health_health_literacy": "Moderate", "health_pain_level": "None",
    "health_color_vision": "Typical", "health_dexterity": "Full", "health_dietary_restriction": "None",
    "health_neurodivergence": "Neurotypical", "health_attention_condition": "None",
    "health_motion_sensitivity": "None", "health_contrast_need": "No", "health_text_size_need": "No",
    "health_assistive_tech": "None", "health_medication_use": "None",
    "health_caregiver_status": "Child caregiver", "health_fitness_level": "Average",
    "health_cognitive_load_capacity": "Moderate", "health_sleep_quality": "Good",
    "lstyle_alcohol_use": "Socially", "lstyle_diet_type": "Omnivore", "lstyle_exercise_freq": "Weekly",
    "intent": "Get task done", "query_complexity": "Multi-step", "prior_context": "Returning user",
    "device_context": "Desktop, focused", "emotional_state": "Calm", "lstyle_sleep_schedule": "Early bird",
    "time_pressure": "Moderate", "risk_tolerance": "Balanced", "decision_style": "Deliberative",
    "coding_ai_job_threat_perception": "Unsure / mixed",
    "future_developer_skill_belief": "Problem decomposition",
    "human_help_boundary_for_ai_coding": "When learning deeply", "coding_ai_usage_frequency": "Weekly",
    "coding_ai_learning_source": "Documentation / tutorials", "coding_ai_sentiment": "Somewhat favorable",
    "coding_ai_output_trust": "Trusts after verification",
    "coding_ai_complexity_comfort": "Comfortable with bounded complex tasks",
    "dev_professional_status": "Hobbyist / side-project coder", "dev_contribution_context": "Personal projects",
    "dev_ic_manager_orientation": "Individual contributor", "dev_primary_work_surface": "IDE / editor",
    "dev_role_archetype": "Back-end developer", "dev_toolchain_complexity": "Focused small stack",
    "bug_fix_scope_preference": "Patch plus adjacent cleanup", "bug_fix_test_behavior": "Usually adds tests",
    "code_api_design_orientation": "Internal ergonomics first", "code_complexity_tolerance": "Moderate",
    "code_modularity_preference": "Moderately modular",
    "code_observability_habit": "Logs important flows / errors",
    "code_security_review_habit": "Checks common vulnerabilities",
    "code_type_system_orientation": "No strong preference",
    "codebase_onboarding_style": "Traces execution paths", "debugging_strategy": "Reproduce first",
    "code_review_participation_level": "Occasional reviewer",
    "open_source_activity_level": "Occasional contributor", "commit_cadence_style": "Task-sized commits",
    "github_contribution_mode": "Pull request contributor", "issue_engagement_style": "Reproduction-focused",
    "pr_collaboration_style": "Moderate scoped PRs", "release_maintenance_orientation": "Not involved in releases",
    "coding_agent_adoption_barrier": "Accuracy / reliability concerns",
    "coding_agent_usage_frequency": "Weekly",
    "coding_agent_explanation_preference": "Explain decisions and tradeoffs",
    "coding_agent_failure_tolerance": "Moderate tolerance for prototypes",
    "coding_agent_memory_preference": "Wants task-scoped memory",
    "coding_agent_security_expectation": "Expects basic secret / privacy checks",
    "coding_agent_tool_integration_preference": "IDE-integrated",
    "coding_agent_workflow_impact": "Moderate productivity increase",
    "coding_agent_autonomy_preference": "Plan-then-act with checkpoints",
    "coding_agent_context_sharing_comfort": "Shares selected files only",
    "coding_agent_control_expectation": "Approval before risky edits only",
    "stackoverflow_usage_pattern": "Occasional troubleshooting",
    "developer_help_seeking_preference": "Search docs first",
    "developer_problem_profile": "Debugging runtime errors",
    "stackoverflow_participation_style": "Reads / searches only",
    "coding_tool_ai_capability_importance": "Moderate", "coding_tool_alternative_sensitivity": "Moderate concern",
    "coding_tool_api_completeness_importance": "Moderate", "coding_tool_ethics_blocker": "Moderate concern",
    "coding_tool_obsolescence_blocker": "Moderate concern", "coding_tool_open_source_importance": "Moderate",
    "coding_tool_reliability_latency_importance": "Moderate",
    "coding_tool_security_privacy_blocker": "Moderate concern",
    "code_comment_style": "Mixed approach",
    "code_summary_documentation": "Includes TLDR for complex functions only",
    "code_naming_verbosity": "Moderately verbose", "code_indentation_style": "4 spaces",
    "code_structure_preference": "Mixed paradigm", "code_error_handling": "Explicit error returns/validation",
    "code_abstraction_level": "Moderate abstraction",
    "code_dependencies_approach": "Balance of libraries and custom",
    "code_performance_priority": "Balanced consideration",
    "code_testing_approach": "Mix of unit and integration tests",
    "code_refactoring_frequency": "Refactor when necessary",
    "code_function_length": "Small functions (10-30 lines)",
    "lang_korean": "Native", "lang_english": "Conversational", "english_proficiency": "Intermediate (B1-B2)",
    "primary_language": "Korean", "multilingualism": "Bilingual",
    "cult_south_korea": "Native", "cult_japan": "Visited",
    "lstyle_commute_mode": "Public transit", "lstyle_pet_ownership": "None",
    "lstyle_social_battery": "Ambivert", "lstyle_smoking": "Never", "lstyle_giving": "Occasional",
    "lstyle_gaming_freq": "Weekly", "lstyle_music_listening": "Daily", "lstyle_reading_freq": "Weekly",
    "lstyle_planning_horizon": "Monthly", "lstyle_volunteering": "Rarely", "lstyle_cooking_freq": "Weekly",
    "lstyle_caffeine": "Moderate", "lstyle_hobby_intensity": "Casual", "lstyle_primary_messenger": "SMS",
    "lstyle_shopping_style": "Researcher", "lstyle_subscription_count": "3-5", "lstyle_frugality": "Balanced",
    "lstyle_tidiness": "Tidy", "lstyle_device_ecosystem": "Android/Google", "lstyle_browser": "Chrome",
    "lstyle_punctuality": "On time", "lstyle_morning_routine": "Loosely structured",
    "lstyle_podcast_listening": "Monthly", "lstyle_banking_style": "Traditional bank",
    "lstyle_news_freq": "Daily", "lstyle_travel_freq": "Occasional", "lstyle_vacation_style": "Relaxation",
    "lstyle_streaming_hours": "3-7", "lstyle_screen_time": "4-8 hrs", "lstyle_coffee_ritual": "Café regular",
    "lstyle_payment_pref": "Credit card", "lstyle_fashion_sense": "Practical",
    "lstyle_primary_social": "YouTube", "lstyle_investment_style": "Index investor",
    "values_priority": "Security", "religiosity": "Secular", "economic_motivation": "Value-driven",
    "need_for_cognition": "Average", "need_for_closure": "Average", "political_lean": "Center",
    "trust_level": "Verifying", "safety_sensitivity": "Benign", "domain": "Business & Management",
    "subject_specialty": "Operations", "expertise_gap": "Peer-level", "tech_savviness": "Comfortable",
    "mbti_type": "ISTJ — Logistician", "neurotype": "Neurotypical", "dominant_trait": "Balanced",
    "domain_characteristics": "Practitioner", "accessibility_needs": "None", "media_diet": "News",
    "modality_pref": "Text", "tone_expected": "Concise", "peeve_micromanagement": "Annoys",
    "cog_politeness": "Polite", "cog_formality": "Neutral", "cog_big_picture_vs_detail": "Both",
    "cog_decision_speed": "Balanced", "cog_reading_vs_watching": "No preference",
}


def _middle(vals):
    return vals[len(vals) // 2]


def base_row():
    row = {}
    for col in IDS:
        val = None
        for prefix, default in FAMILY_DEFAULTS:
            if col.startswith(prefix) and default is not None and default in VALUES[col]:
                val = default
                break
        row[col] = val if val is not None else _middle(VALUES[col])
    row.update(EXPLICIT)
    for col, v in row.items():
        assert v in VALUES[col], (col, v)
    return row


BASE = base_row()


def frame(overrides_list, as_category=False):
    rows = []
    for ov in overrides_list:
        r = dict(BASE)
        for k, v in ov.items():
            assert k in VALUES, f"unknown column in test: {k}"
            assert v in VALUES[k], f"unknown value in test: {k}={v!r}"
            r[k] = v
        rows.append(r)
    df = pd.DataFrame(rows, columns=IDS)
    if as_category:
        df = df.astype("category")
    return df


def fail_all(col_list, value):
    return {c: value for c in col_list}


# -----------------------------------------------------------------------------
# One contradictory row per rule
# -----------------------------------------------------------------------------
BAD = {
    "R3-POP-H01": {"age_bracket": "13-17"},
    "R3-POP-H02": {"demo_citizenship_status": "Visa holder"},
    "R3-POP-H03": {"lifex_immigration_generation": "First-generation immigrant"},
    "R3-POP-H04": {"region": "Western Europe"},
    "R3-POP-H05": {"cult_south_korea": "Visited"},
    "R3-POP-S01": {"demo_citizenship_status": "Naturalized citizen"},
    "R3-POP-S02": {"demo_ethnicity_broad": "White / European"},
    "R3-AGE-H01": {"age_bracket": "65-74", "demo_generation": "Millennial", "lifex_formative_decade": "1980s"},
    "R3-AGE-H02": {"lifex_formative_decade": "2020s"},
    "R3-AGE-H03": {"age_bracket": "18-24", "demo_generation": "Gen Z", "lifex_formative_decade": "2010s",
                   "demo_children_count": "Adult children", "demo_parental_status": "Parent of adults",
                   "lifex_parenting_journey": "Grown children", "life_stage": "Mid-life"},
    "R3-AGE-H04": {"age_bracket": "18-24", "demo_generation": "Gen Z", "lifex_formative_decade": "2010s",
                   "demo_relationship_length": "15+ years"},
    "R3-AGE-H05": {"gender_identity": "Woman", "age_bracket": "55-64", "demo_generation": "Boomer",
                   "lifex_formative_decade": "1980s", "demo_children_count": "Expecting",
                   "lifex_military_history": "Never served", "demo_veteran_status": "Civilian",
                   "major_life_events": "None notable"},
    "R3-AGE-S01": {"age_bracket": "25-34", "lifex_formative_decade": "2000s",
                   "demo_children_count": "Adult children", "demo_parental_status": "Parent of adults"},
    "R3-AGE-S02": {"age_bracket": "25-34", "lifex_formative_decade": "2000s", "years_experience": "20+"},
    "R3-FAM-H01": {"demo_relationship_length": "Not in one"},
    "R3-FAM-H02": {"demo_children_count": "None"},
    "R3-FAM-H03": {"lifex_parenting_journey": "No children"},
    "R3-FAM-H04": {"demo_children_count": "None", "demo_parental_status": "Not a parent",
                   "lifex_parenting_journey": "No children"},
    "R3-FAM-H05": {"lstyle_household_size": "Lives alone", "demo_housing_status": "Living with family"},
    "R3-FAM-S01": {"lstyle_household_size": "Lives alone", "demo_housing_status": "Shared housing"},
    "R3-MIL-H01": {"lifex_military_history": "Never served"},
    "R3-MIL-H02": {"demo_veteran_status": "Active duty", "age_bracket": "65-74", "demo_generation": "Boomer",
                   "lifex_formative_decade": "1980s"},
    "R3-MIL-H03": {"demo_veteran_status": "Active duty", "demo_employment_status": "Homemaker"},
    "R3-MIL-S01": {"gender_identity": "Woman"},
    "R3-EDU-H01": {"lifex_education_journey": "Advanced degrees"},
    "R3-EDU-H02": {"institution_tier": "Not applicable"},
    "R3-EDU-H03": {"highest_education": "Doctorate", "lifex_education_journey": "Advanced degrees",
                   "research_output": "None"},
    "R3-EDU-H04": dict({"academic_field": "Law"}, **fail_all(m.LAW_FAM, "None")),
    "R3-EDU-S01": {"highest_education": "Primary", "institution_tier": "Not applicable"},
    "R3-EDU-S02": {"lifex_education_journey": "Dropped out", "highest_education": "Master's"},
    "R3-CAR-H01": {"demo_employment_status": "Unemployed"},
    "R3-CAR-H02": {"role_function": "Not applicable"},
    "R3-CAR-H03": {"lifex_career_path_shape": "Just starting"},
    "R3-CAR-H04": {"seniority": "Founder"},
    "R3-CAR-H05": {"role_function": "Clinical", "ind_healthcare": "None"},
    "R3-CAR-H06": {"dev_ic_manager_orientation": "People manager", "skill_people_management": "None"},
    "R3-CAR-S01": {"demo_employment_status": "Retired"},
    "R3-CAR-S02": {"years_experience": "0-2", "seniority": "Director"},
    "R3-INC-H01": {"demo_household_income": "<$25k", "socioeconomic_band": "High income"},
    "R3-INC-H02": {"lstyle_banking_style": "Unbanked"},
    "R3-INC-H03": {"pref_save_vs_spend": "Hard saver", "lstyle_frugality": "Splurger"},
    "R3-INC-S01": {"kr_subjective_class": "Upper class", "socioeconomic_band": "Low income"},
    "R3-INC-S02": {"lstyle_banking_style": "Unbanked", "lstyle_payment_pref": "Mobile wallet"},
    "R3-LAN-H01": {"primary_language": "English"},
    "R3-LAN-H02": {"english_proficiency": "Native"},
    "R3-LAN-H03": {"multilingualism": "Monolingual", "lang_english": "Fluent",
                   "english_proficiency": "Fluent (C1-C2)"},
    "R3-LAN-H04": {"lifex_languages_lifetime": "One"},
    "R3-LAN-S01": {"lang_korean": "Conversational"},
    "R3-LAN-S02": {"lang_japanese": "Native"},
    "R3-MOB-H01": {"lifex_travel_breadth": "Never left country"},
    "R3-MOB-H02": {"lifex_travel_breadth": "Never left country", "cult_japan": "Unfamiliar",
                   "lifex_geographic_mobility": "Moved internationally"},
    "R3-MOB-H03": {"lifex_geographic_mobility": "Never left hometown"},
    "R3-MOB-H04": {"lifex_geographic_mobility": "Never left hometown", "lifex_hometown_tie": "Still there",
                   "lifex_travel_breadth": "Lived abroad"},
    "R3-MOB-S01": {"lifex_cultural_exposure": "Monocultural", "lifex_travel_breadth": "Lived abroad"},
    "R3-REL-H01": {"religiosity": "Devout"},
    "R3-REL-H02": {"lifex_faith_journey": "Lifelong faith"},
    "R3-REL-H03": {"religiosity": "Observant", "demo_religion_affiliation": "Buddhist"},
    "R3-REL-H04": {"val_spirituality_faith": "Core value", "trait_spirituality": "Absent"},
    "R3-REL-H05": {"health_dietary_restriction": "Religious"},
    "R3-REL-S01": {"demo_religion_affiliation": "Protestant"},
    "R3-HLT-H01": {"health_vision": "Blind", "demo_driver_status": "Non-driver"},
    "R3-HLT-H02": {"demo_disability_status": "Sensory"},
    "R3-HLT-H03": {"demo_disability_status": "Chronic illness"},
    "R3-HLT-H04": {"accessibility_needs": "Hearing"},
    "R3-HLT-H05": {"neurotype": "Autistic"},
    "R3-HLT-H06": {"health_attention_condition": "Diagnosed"},
    "R3-HLT-H07": {"health_fitness_level": "Sedentary", "lstyle_exercise_freq": "Daily"},
    "R3-HLT-H08": {"lstyle_exercise_freq": "Never", "sport_running": "Play"},
    "R3-HLT-H09": {"health_vision": "Blind", "demo_disability_status": "Sensory"},
    "R3-HLT-S01": {"health_hearing": "Deaf / hard of hearing"},
    "R3-HLT-S02": {"health_hearing": "Deaf / hard of hearing", "demo_disability_status": "Sensory",
                   "pref_text_vs_call": "Calls only"},
    "R3-HLT-S03": {"lstyle_exercise_freq": "Never", "habit_morning_walk": "Daily"},
    "R3-HLT-S04": {"health_medication_use": "Multiple daily"},
    "R3-LIF-H01": {"lstyle_hobby_intensity": "None", "hob_pottery": "Avid"},
    "R3-LIF-H02": {"lstyle_reading_freq": "Never", "habit_reading_before_bed": "Daily"},
    "R3-LIF-H03": {"lstyle_cooking_freq": "Daily", "skill_cooking": "None"},
    "R3-LIF-H04": {"hob_bouldering": "Avid", "sport_climbing": "None"},
    "R3-LIF-H05": {"lstyle_diet_type": "Vegan", "att_veganism": "Opposed",
                   "health_dietary_restriction": "Ethical"},
    "R3-LIF-H06": {"cuis_sushi": "Love", "cuis_japanese": "Avoid"},
    "R3-LIF-H07": {"lstyle_news_freq": "Avoids news"},
    "R3-LIF-H08": {"lstyle_volunteering": "Weekly", "lifex_service_history": "None"},
    "R3-LIF-S01": {"lstyle_caffeine": "None"},
    "R3-LIF-S02": {"lstyle_screen_time": "<2 hrs", "lstyle_streaming_hours": "16+"},
    "R3-LIF-S03": {"lstyle_reading_freq": "Never", "habit_reading_before_bed": "Never",
                   "cog_reading_vs_watching": "Strongly prefers reading"},
    "R3-LIF-S04": {"cuis_korean": "Avoid"},
    "R3-DEV-H01": {"lstyle_primary_messenger": "iMessage"},
    "R3-DEV-H02": {"lstyle_browser": "Safari"},
    "R3-DEV-H03": {"pref_early_vs_late": "Bleeding edge", "att_new_technology": "Opposed"},
    "R3-DEV-H04": {"tech_savviness": "Avoidant", "dev_professional_status": "Professional developer"},
    "R3-DEV-S01": {"tech_savviness": "Reluctant", "tool_docker": "Power user"},
    "R3-DEV-S02": {"tech_savviness": "Digital native", "age_bracket": "55-64", "demo_generation": "Boomer",
                   "lifex_formative_decade": "1980s"},
    "R3-TEC-H01": {"coding_ai_usage_frequency": "Never used", "tool_github_copilot": "Never used",
                   "coding_agent_usage_frequency": "No plans", "coding_agent_workflow_impact": "No clear change"},
    "R3-TEC-H02": {"coding_ai_usage_frequency": "Never used"},
    "R3-TEC-H03": {"coding_agent_failure_tolerance": "Does not use agents"},
    "R3-TEC-H04": {"developer_help_seeking_preference": "Ask AI first",
                   "coding_ai_usage_frequency": "Actively avoids"},
    "R3-TEC-H05": {"stackoverflow_usage_pattern": "Never used"},
    "R3-TEC-H06": {"open_source_activity_level": "No coding activity"},
    "R3-TEC-H07": {"dev_contribution_context": "Open-source / community work",
                   "open_source_activity_level": "Private-only developer"},
    "R3-TEC-H08": {"skill_coding": "None", "prog_python": "Expert"},
    "R3-TEC-H09": {"dev_professional_status": "Professional developer", "skill_coding": "None"},
    "R3-TEC-H10": {"prog_python": "Expert", "tool_python": "Never used"},
    "R3-TEC-H11": {"prog_typescript": "Expert", "prog_javascript": "None"},
    "R3-TEC-H12": {"code_refactoring_frequency": "Never refactors",
                   "bug_fix_scope_preference": "Broad refactor while fixing"},
    "R3-TEC-H13": {"coding_agent_autonomy_preference": "Autonomous execution",
                   "coding_agent_control_expectation": "Read-only assistance preferred"},
    "R3-TEC-H14": {"coding_agent_adoption_barrier": "Security / privacy / compliance",
                   "coding_tool_security_privacy_blocker": "Not a concern"},
    "R3-TEC-S01": {"att_ai": "Opposed", "coding_ai_sentiment": "Very favorable"},
    "R3-TEC-S02": {"open_source_activity_level": "Maintainer / core contributor"},
    "R3-TEC-S03": {"skill_coding": "Beginner", "prog_go": "Expert"},
    "R3-TEC-S04": {"tool_kubernetes": "Power user", "tool_docker": "Never used"},
    "R3-EXP-H01": {"fam_deep_learning": "Expert", "fam_machine_learning": "Aware"},
    "R3-EXP-H02": {"skill_photography": "Master", "fam_photography": "Aware"},
    "R3-EXP-H03": {"fam_operations_management": "None"},
    "R3-EXP-H04": {"ind_accounting": "Veteran", "fam_accounting": "None"},
    "R3-EXP-H05": {"cog_numeracy_comfort": "None", "skill_mathematics": "Advanced"},
    "R3-EXP-H06": {"tool_excel": "Power user", "skill_spreadsheet_modeling": "None"},
    "R3-EXP-H07": {"tool_aws": "Power user", "fam_cloud_infrastructure": "None"},
    "R3-EXP-S01": {"subject_specialty": "Cardiology", "fam_cardiology": "Familiar"},
    "R3-EXP-S02": {"fam_data_science": "Expert", "fam_statistics": "Aware"},
    "R3-PER-H01": {"bfi2_facet_anxiety": "Very high", "big5_anxiety": "Very low"},
    "R3-PER-H02": {"bfi2_facet_sociability": "Very low", "big5_gregariousness": "Very high"},
    "R3-PER-H03": {"bfi2_facet_trust": "Very high", "big5_trust": "Very low"},
    "R3-PER-H04": {"bfi2_facet_responsibility": "Very low", "big5_dutifulness": "Very high"},
    "R3-PER-H05": {"bfi2_facet_creative_imagination": "Very high", "big5_imagination": "Very low"},
    "R3-PER-H06": {"bfi2_domain_agreeableness": "Very high", "bfi2_facet_compassion": "Low",
                   "bfi2_facet_respectfulness": "Average", "bfi2_facet_trust": "Low"},
    "R3-PER-H07": {"dominant_trait": "High openness", "bfi2_domain_open_mindedness": "Low"},
    "R3-PER-H08": {"mbti_type": "ESTJ — Executive", "lstyle_social_battery": "Strong introvert"},
    "R3-PER-H09": {"pref_plan_vs_spontaneous": "Highly spontaneous"},
    "R3-PER-H10": {"trait_discipline": "Signature", "big5_self_discipline": "Very low"},
    "R3-PER-H11": {"trait_resilience": "Signature", "big5_vulnerability": "Very high"},
    "R3-PER-H12": {"cog_optimism": "None", "trait_hope_optimism": "Signature"},
    "R3-PER-H13": {"cog_politeness": "Rude", "bfi2_facet_respectfulness": "Very high"},
    "R3-PER-H14": {"lstyle_tidiness": "Chaotic", "big5_orderliness": "Very high"},
    "R3-PER-H15": {"cog_ambiguity_tolerance": "Very high", "need_for_closure": "Very high"},
    "R3-PER-H16": {"cog_detail_orientation": "Very high", "cog_big_picture_vs_detail": "Big-picture only"},
    "R3-PER-H17": {"decision_style": "Analytical", "pref_logic_vs_intuition": "Pure intuition"},
    "R3-PER-H18": {"big5_gregariousness": "Very low", "pref_big_group_vs_one_on_one": "Loves big groups"},
    "R3-PER-H19": {"pref_novelty_vs_familiarity": "Always novel", "pref_routine_vs_variety": "Craves routine"},
    "R3-PER-H20": {"big5_liberalism": "Very high", "political_lean": "Right"},
    "R3-PER-H21": {"trait_competitiveness": "Absent", "pref_competition_vs_collab": "Highly competitive"},
    "R3-PER-H22": {"cog_formality": "Very formal"},
    "R3-PER-S01": {"mbti_type": "INFP — Mediator", "bfi2_domain_open_mindedness": "Very low"},
    "R3-PER-S02": {"big5_excitement_seeking": "Very low", "hob_skydiving": "Avid"},
    "R3-PER-S03": {"trait_teamwork": "Signature", "pref_team_vs_solo": "Strongly solo"},
    "R3-PER-S04": {"lstyle_social_battery": "Strong introvert",
                   "pref_big_group_vs_one_on_one": "Loves big groups"},
    "R3-VAL-H01": {"schwartz_value_power": "Very high", "val_power_influence": "Irrelevant"},
    "R3-VAL-H02": {"schwartz_value_tradition": "Very low", "val_tradition": "Core value"},
    "R3-VAL-H03": {"val_security_stability": "Minor"},
    "R3-VAL-H04": {"schwartz_value_security": "Low"},
    "R3-VAL-H05": {"att_authority": "Enthusiast", "mft_authority_subversion": "Very low"},
    "R3-VAL-H06": {"att_data_privacy": "Opposed", "val_privacy": "Core value"},
    "R3-VAL-S01": {"mft_loyalty_betrayal": "Very high", "val_loyalty": "Irrelevant"},
    "R3-VAL-S02": {"demo_sexual_orientation": "Bisexual", "kr_attitude_same_sex_relations": "Always wrong"},
    "R3-INT-H01": {"topic_sports": "Averse", "sport_baseball": "Follow"},
    "R3-INT-H02": {"hob_knitting": "Avid", "topic_knitting": "Averse"},
    "R3-INT-H03": {"topic_film": "Averse", "filmg_thriller": "Love"},
    "R3-INT-H04": {"topic_history": "Averse", "bookg_history": "Love"},
    "R3-INT-H05": {"acad_music": "Passionate", "topic_music": "Averse"},
    "R3-INT-H06": {"habit_meditation": "Daily", "topic_meditation": "Averse"},
    "R3-INT-S01": {"sport_running": "Play", "topic_running": "Averse"},
    "R3-INT-S02": {"lstyle_gaming_freq": "Never", "topic_gaming": "Passionate"},
    "R3-ATT-H01": {"att_risk_taking": "Opposed", "risk_tolerance": "Risk-seeking"},
    "R3-ATT-H02": {"att_remote_work": "Opposed", "pref_work_location": "Strongly remote"},
    "R3-ATT-H03": {"att_rapid_change": "Enthusiast", "pref_stability_vs_change": "Craves stability"},
    "R3-ATT-H04": {"lstyle_shopping_style": "Brand loyal", "att_brand_loyalty": "Opposed"},
    "R3-ATT-H05": {"att_open_source": "Opposed", "coding_tool_open_source_importance": "Critical"},
    "R3-ATT-H06": {"att_space_exploration": "Enthusiast", "topic_space": "Averse"},
    "R3-ATT-H07": {"demo_political_engagement": "Activist", "topic_politics": "Averse"},
    "R3-ATT-H08": {"hob_paragliding": "Avid", "dospert_recreational_risk_tolerance": "Very low"},
    "R3-ATT-S01": {"att_public_transit": "Opposed"},
    "R3-ATT-S02": {"att_climate_action": "Opposed", "val_sustainability": "Core value"},
    "R3-ATT-S03": {"dospert_financial_risk_tolerance": "Very low", "lstyle_investment_style": "Active trader"},
    "R3-ATT-S04": {"demo_political_engagement": "Protest participant", "political_lean": "Apolitical"},
    "R3-ATT-S05": {"kr_country_felt_closest": "North Korea", "kr_view_of_north_korea": "Enemy"},
    "R3-WEL-H01": {"health_mental_health": "Thriving", "kr_anxiety_level": "Severe"},
    "R3-WEL-H02": {"kr_depression_level": "Severe", "kr_happiness": "Very happy"},
    "R3-WEL-S01": {"kr_happiness": "Very unhappy", "kr_life_satisfaction": "Very satisfied"},
    "R3-WEL-S02": {"kr_depression_level": "Severe", "big5_depression": "Very low"},
    "R3-EVT-H01": {"lifex_loss_experience": "No major loss", "demo_marital_status": "Widowed",
                   "lifex_relationship_history": "Widowed"},
    "R3-EVT-H02": {"major_life_events": "Health journey"},
    "R3-EVT-H03": {"lifex_adversity_level": "Sheltered", "lifex_childhood_environment": "Hardship"},
    "R3-EVT-S01": {"lifex_adversity_level": "Sheltered", "lifex_financial_trajectory": "Rebuilt after setback"},
}

# Further contradictory rows that exercise other clauses of multi-clause rules.
BAD_EXTRA = [
    ("R3-AGE-H01", {"age_bracket": "85+", "demo_generation": "Boomer", "lifex_formative_decade": "1960s-70s"}),
    ("R3-AGE-H02", {"age_bracket": "75-84", "demo_generation": "Silent", "lifex_formative_decade": "1980s"}),
    ("R3-AGE-H05", {"gender_identity": "Woman", "age_bracket": "75-84", "demo_generation": "Silent",
                    "lifex_formative_decade": "1960s-70s", "lifex_military_history": "Never served",
                    "demo_veteran_status": "Civilian", "major_life_events": "None notable"}),
    ("R3-FAM-H02", {"demo_parental_status": "Not a parent", "lifex_parenting_journey": "No children",
                    "life_stage": "Mid-life"}),
    ("R3-FAM-H03", {"demo_parental_status": "Not a parent", "demo_children_count": "None"}),
    ("R3-FAM-H04", {"life_stage": "Empty nester", "lifex_parenting_journey": "No children",
                    "demo_children_count": "None", "demo_parental_status": "Not a parent"}),
    ("R3-MIL-H01", {"lifex_military_history": "Never served", "demo_veteran_status": "Civilian"}),
    ("R3-EDU-H01", {"highest_education": "No formal", "institution_tier": "Not applicable"}),
    ("R3-EDU-H02", {"highest_education": "Primary", "institution_tier": "Top-tier research"}),
    ("R3-EDU-H03", {"highest_education": "Secondary", "research_output": "Thesis only",
                    "institution_tier": "Not applicable"}),
    ("R3-EDU-H04", dict({"academic_field": "Medicine"}, **fail_all(m.MED_FAM, "None"))),
    ("R3-CAR-H01", {"lstyle_work_schedule": "Unemployed"}),
    ("R3-CAR-H02", {"company_size": "Not applicable"}),
    ("R3-CAR-H03", {"life_stage": "Early career", "years_experience": "20+"}),
    ("R3-CAR-H04", {"lifex_career_path_shape": "Entrepreneurial", "lifex_entrepreneurship_history": "Never considered"}),
    ("R3-CAR-H04", {"major_life_events": "Started a business"}),
    ("R3-INC-H01", {"demo_household_income": "$200k+", "socioeconomic_band": "Low income"}),
    ("R3-LAN-H01", {"primary_language": "Japanese", "lang_japanese": "Basic"}),
    ("R3-LAN-H03", {"lang_english": "Basic", "english_proficiency": "Basic (A1-A2)",
                    "lifex_languages_lifetime": "One"}),
    ("R3-LAN-H04", {"lifex_languages_lifetime": "Two", "multilingualism": "Trilingual+",
                    "lang_japanese": "Conversational"}),
    ("R3-LAN-H04", {"lifex_languages_lifetime": "Two", "lang_japanese": "Conversational"}),
    ("R3-MOB-H01", {"lifex_travel_breadth": "Never left country", "cult_japan": "Unfamiliar",
                    "cult_united_states": "Lived there"}),
    ("R3-MOB-H02", {"lifex_travel_breadth": "Never left country", "cult_japan": "Unfamiliar",
                    "lifex_immigration_generation": "Returnee / repatriate"}),
    ("R3-MOB-H04", {"lifex_geographic_mobility": "Never left hometown", "lifex_hometown_tie": "Still there",
                    "major_life_events": "Major relocation"}),
    ("R3-REL-H02", {"demo_religion_affiliation": "Atheist / agnostic", "lifex_faith_journey": "Returned to faith"}),
    ("R3-REL-H03", {"religiosity": "Devout", "demo_religion_affiliation": "Catholic",
                    "lifex_faith_journey": "Lapsed / left faith"}),
    ("R3-REL-H04", {"religiosity": "Devout", "demo_religion_affiliation": "Catholic",
                    "lifex_faith_journey": "Lifelong faith", "val_spirituality_faith": "Irrelevant"}),
    ("R3-HLT-H03", {"lifex_health_journey": "Manages chronic condition"}),
    ("R3-HLT-H04", {"accessibility_needs": "Motor"}),
    ("R3-HLT-H04", {"accessibility_needs": "Visual"}),
    ("R3-HLT-H04", {"device_context": "Accessibility tool"}),
    ("R3-HLT-H05", {"health_neurodivergence": "Dyslexic"}),
    ("R3-HLT-H06", {"health_neurodivergence": "ADHD", "neurotype": "ADHD"}),
    ("R3-HLT-H07", {"health_fitness_level": "Athlete", "lstyle_exercise_freq": "Rarely"}),
    ("R3-HLT-H07", {"health_fitness_level": "Sedentary", "sport_triathlon": "Play"}),
    ("R3-HLT-H08", {"lstyle_exercise_freq": "Never", "hob_kayaking": "Active"}),
    ("R3-LIF-H02", {"lstyle_cooking_freq": "Never", "habit_meal_prepping": "Weekly"}),
    ("R3-LIF-H02", {"lstyle_gaming_freq": "Never", "sport_esports": "Play"}),
    ("R3-LIF-H03", {"hob_vegetable_gardening": "Active", "skill_gardening": "None"}),
    ("R3-LIF-H03", {"skill_driving": "None"}),
    ("R3-LIF-H05", {"lstyle_diet_type": "Vegetarian", "cuis_vegetarian": "Avoid",
                    "health_dietary_restriction": "Ethical"}),
    ("R3-LIF-H06", {"cuis_cantonese": "Love", "cuis_chinese": "Avoid"}),
    ("R3-LIF-H07", {"media_diet": "Social-media-heavy", "lstyle_primary_social": "None"}),
    ("R3-DEV-H03", {"pref_early_vs_late": "Bleeding edge", "tech_savviness": "Avoidant"}),
    ("R3-TEC-H01", {"coding_ai_usage_frequency": "Daily", **{c: "Not applicable" for c in m.AI_TASK_COLS}}),
    ("R3-TEC-H02", {"coding_ai_usage_frequency": "Actively avoids", "tool_github_copilot": "Regular"}),
    ("R3-TEC-H03", {"coding_agent_usage_frequency": "No plans"}),
    ("R3-TEC-H04", {"developer_help_seeking_preference": "Search Stack Overflow / Q&A first",
                    "stackoverflow_usage_pattern": "Never used", "stackoverflow_participation_style": "Does not participate"}),
    ("R3-TEC-H04", {"coding_ai_learning_source": "AI-enabled coding tools", "coding_ai_usage_frequency": "Never used",
                    "tool_github_copilot": "Never used", "coding_agent_usage_frequency": "No plans",
                    "coding_agent_workflow_impact": "No clear change", **{c: "Not applicable" for c in m.AI_TASK_COLS}}),
    ("R3-TEC-H06", {"open_source_activity_level": "No coding activity", "github_contribution_mode": "Rare GitHub user",
                    "pr_collaboration_style": "Rare PR usage", "coding_ai_usage_frequency": "Daily"}),
    ("R3-TEC-H07", {"github_contribution_mode": "Star / fork explorer", "tool_github": "Never used",
                    "tool_gitlab": "Never used"}),
    ("R3-TEC-H08", {"skill_coding": "Master", **{c: "None" for c in m.PROG_COLS}}),
    ("R3-TEC-H08", {"code_review_participation_level": "Frequent reviewer", "skill_code_review": "None"}),
    ("R3-TEC-H09", {"dev_professional_status": "Professional developer", "demo_employment_status": "Retired"}),
    ("R3-TEC-H10", {"tool_sql": "Power user", "prog_sql": "None"}),
    ("R3-TEC-H11", {"tool_jupyter": "Regular", "prog_python": "None", "prog_r": "None", "prog_julia": "None"}),
    ("R3-TEC-H11", {"prog_objective_c": "Proficient", "prog_c": "None"}),
    ("R3-TEC-H12", {"bug_fix_test_behavior": "No tests", "code_testing_approach": "Test-driven development (TDD)"}),
    ("R3-TEC-H12", {"code_comment_style": "No comments", "code_summary_documentation": "Always includes file-level TLDR"}),
    ("R3-TEC-H13", {"coding_agent_autonomy_preference": "No agent autonomy",
                    "coding_agent_control_expectation": "Agent may edit within scoped files"}),
    ("R3-TEC-S04", {"tool_github_copilot": "Power user", "skill_coding": "None", "prog_go": "None"}),
    ("R3-EXP-H01", {"fam_auditing": "Proficient", "fam_accounting": "None"}),
    ("R3-EXP-H02", {"skill_audio_production": "Advanced", "fam_sound_engineering": "None",
                    "fam_music_production": "None"}),
    ("R3-EXP-H06", {"tool_blender": "Power user", "fam_3d_modeling": "None"}),
    ("R3-EXP-H07", {"tool_spss": "Power user", "fam_statistics": "None"}),
    ("R3-PER-H01", {"neurotype": "Anxiety-prone", "big5_anxiety": "Very low"}),
    ("R3-PER-H06", {"bfi2_domain_negative_emotionality": "Very low", "bfi2_facet_anxiety": "High",
                    "bfi2_facet_depression": "Average", "bfi2_facet_emotional_volatility": "Average"}),
    ("R3-PER-H07", {"dominant_trait": "High extraversion", "lstyle_social_battery": "Introvert"}),
    ("R3-PER-H08", {"bfi2_domain_extraversion": "Very high", "bfi2_facet_sociability": "Very high",
                    "bfi2_facet_assertiveness": "Very high", "bfi2_facet_energy_level": "Very high"}),
    ("R3-PER-H09", {"mbti_type": "ENFP — Campaigner", "pref_plan_vs_spontaneous": "Highly planned"}),
    ("R3-PER-H11", {"trait_self_regulation": "Signature", "big5_immoderation": "Very high"}),
    ("R3-PER-H13", {"interpersonal_communion_warmth": "Very low", "big5_friendliness": "Very high"}),
    ("R3-PER-H14", {"cog_procrastination": "None", "bfi2_facet_productiveness": "Very low"}),
    ("R3-PER-H15", {"cog_ambiguity_tolerance": "None", "need_for_closure": "Very low"}),
    ("R3-PER-H17", {"big5_cautiousness": "Very high", "cog_decision_speed": "Snap decisions"}),
    ("R3-PER-H19", {"big5_adventurousness": "Very high", "pref_novelty_vs_familiarity": "Always familiar"}),
    ("R3-PER-H21", {"trait_leadership": "Absent", "pref_lead_vs_follow": "Always leads"}),
    ("R3-PER-H22", {"cog_formality": "Slangy", "register": "Formal / standard"}),
    ("R3-VAL-H01", {"schwartz_value_self_direction": "Very low", "val_independence": "Core value"}),
    ("R3-VAL-H02", {"schwartz_value_universalism": "Very high", "val_equality": "Irrelevant"}),
    ("R3-VAL-H03", {"values_priority": "Autonomy", "val_independence": "Minor", "val_personal_freedom": "Irrelevant"}),
    ("R3-VAL-H04", {"values_priority": "Community", "schwartz_value_benevolence": "Low",
                    "schwartz_value_universalism": "Very low"}),
    ("R3-INT-H03", {"topic_music": "Averse", "musg_k_pop": "Love"}),
    ("R3-INT-H05", {"acad_world_history": "Averse", "topic_history": "Passionate"}),
    ("R3-INT-S02", {"topic_anime": "Passionate", "filmg_animation": "Dislike"}),
    ("R3-ATT-H02", {"att_working_from_office": "Opposed", "pref_work_location": "Strongly office"}),
    ("R3-WEL-H01", {"health_mental_health": "Thriving", "kr_depression_level": "Moderately severe"}),
    ("R3-EVT-H01", {"lifex_loss_experience": "No major loss", "major_life_events": "Bereavement"}),
]

# -----------------------------------------------------------------------------
# Consistent rows close to rule boundaries (must trigger no rule at all)
# -----------------------------------------------------------------------------
BOUNDARY = [
    # cohort borders under birth year = 2020 - age
    {"age_bracket": "25-34", "demo_generation": "Gen Z", "lifex_formative_decade": "2020s",
     "years_experience": "6-10"},
    {"age_bracket": "18-24", "demo_generation": "Millennial", "lifex_formative_decade": "2000s",
     "demo_marital_status": "Single", "demo_relationship_length": "1-5 years",
     "lifex_relationship_history": "A few relationships", "demo_children_count": "None",
     "demo_parental_status": "Not a parent", "lifex_parenting_journey": "No children",
     "life_stage": "Student", "demo_employment_status": "Student", "lstyle_work_schedule": "Flexible hours",
     "years_experience": "6-10", "seniority": "Student / intern", "lifex_career_path_shape": "Just starting",
     "highest_education": "Secondary", "institution_tier": "Mid-tier university",
     "lstyle_household_size": "3-4 people", "demo_housing_status": "Living with family",
     "health_caregiver_status": "Not a caregiver", "dev_ic_manager_orientation": "Student / learner",
     "demo_veteran_status": "Veteran", "lifex_military_history": "Served briefly"},
    {"age_bracket": "45-54", "demo_generation": "Boomer", "lifex_formative_decade": "2000s",
     "lifex_parenting_journey": "Teenagers", "life_stage": "Mid-life"},
    {"age_bracket": "55-64", "demo_generation": "Gen X", "lifex_formative_decade": "1960s-70s",
     "demo_veteran_status": "Active duty", "lifex_military_history": "Career veteran",
     "lifex_parenting_journey": "Teenagers", "life_stage": "Mid-life", "tech_savviness": "Comfortable"},
    {"age_bracket": "85+", "demo_generation": "Silent", "lifex_formative_decade": "1960s-70s",
     "demo_children_count": "Adult children", "demo_parental_status": "Grandparent",
     "lifex_parenting_journey": "Empty nester", "life_stage": "Retirement",
     "demo_employment_status": "Retired", "lstyle_work_schedule": "Unemployed", "seniority": "Retired",
     "years_experience": "20+", "demo_veteran_status": "Veteran", "lifex_military_history": "Combat veteran",
     "dev_professional_status": "Former / inactive developer", "health_caregiver_status": "Not a caregiver",
     "company_size": "Not applicable", "role_function": "Not applicable"},
    # women near the fertility-arithmetic limits, no military service
    {"gender_identity": "Woman", "age_bracket": "55-64", "demo_generation": "Boomer",
     "lifex_formative_decade": "1980s", "demo_parental_status": "Parent of minors",
     "lifex_parenting_journey": "Teenagers", "life_stage": "Mid-life",
     "lifex_military_history": "Never served", "demo_veteran_status": "Civilian",
     "major_life_events": "Caregiving"},
    {"gender_identity": "Woman", "age_bracket": "65-74", "demo_generation": "Silent",
     "lifex_formative_decade": "1960s-70s", "demo_parental_status": "Parent of minors",
     "lifex_parenting_journey": "Teenagers", "life_stage": "Mid-life",
     "lifex_military_history": "Military family", "demo_veteran_status": "Military family",
     "major_life_events": "None notable"},
    # languages at the edge
    {"english_proficiency": "Fluent (C1-C2)", "lang_english": "Conversational",
     "multilingualism": "Trilingual+", "lang_japanese": "Conversational", "lifex_languages_lifetime": "Three"},
    {"multilingualism": "Monolingual", "lang_english": "Basic", "english_proficiency": "Basic (A1-A2)",
     "lifex_languages_lifetime": "One"},
    {"primary_language": "Korean", "lang_korean": "Fluent"},
    # travel / mobility edges
    {"lifex_travel_breadth": "Never left country", "cult_japan": "Studied", "lstyle_travel_freq": "Rare"},
    {"lifex_geographic_mobility": "Never left hometown", "lifex_hometown_tie": "Still there"},
    # religion edges
    {"religiosity": "Devout", "demo_religion_affiliation": "Protestant", "lifex_faith_journey": "Lifelong faith",
     "val_spirituality_faith": "Core value", "trait_spirituality": "Strong",
     "health_dietary_restriction": "Religious", "att_organized_religion": "Positive"},
    {"religiosity": "Spiritual", "demo_religion_affiliation": "Spiritual but unaffiliated",
     "lifex_faith_journey": "Still searching"},
    # health edges
    {"health_vision": "Low vision", "demo_disability_status": "Sensory",
     "demo_driver_status": "Licensed, rarely drives", "accessibility_needs": "Visual",
     "health_text_size_need": "Requires", "health_assistive_tech": "Magnifier"},
    {"demo_disability_status": "Sensory", "health_vision": "Normal", "health_hearing": "Mild loss",
     "accessibility_needs": "Hearing"},
    {"accessibility_needs": "Visual", "health_motion_sensitivity": "Strong (reduced motion)"},
    {"role_function": "Clinical", "ind_healthcare": "Experienced", "dev_ic_manager_orientation": "People manager",
     "skill_people_management": "Beginner"},
    {"health_fitness_level": "Athlete", "lstyle_exercise_freq": "Weekly"},
    {"health_fitness_level": "Sedentary", "lstyle_exercise_freq": "Rarely"},
    {"lstyle_exercise_freq": "Never", "sport_golf": "Play", "sport_esports": "Play",
     "lstyle_hobby_intensity": "None", "hob_karaoke": "Avid",
     **{c: "Never" for c in m.HOBBY_COLS_FOR_INTENSITY}},
    {"neurotype": "Anxiety-prone", "bfi2_facet_anxiety": "High", "big5_anxiety": "Average"},
    # dev / AI edges
    {"coding_ai_usage_frequency": "Never used", "tool_github_copilot": "Never used",
     "coding_agent_usage_frequency": "Not yet but interested",
     "coding_agent_failure_tolerance": "Does not use agents", "coding_agent_workflow_impact": "No clear change",
     "coding_ai_learning_source": "Not learning AI tooling",
     **{c: "Plans partial AI use" for c in m.AI_TASK_COLS}},
    {"coding_ai_usage_frequency": "Actively avoids", "tool_github_copilot": "Tried it",
     "coding_agent_usage_frequency": "Monthly or infrequent",
     **{c: "Does not plan AI use" for c in m.AI_TASK_COLS}},
    {"dev_professional_status": "Professional developer", "skill_coding": "Advanced", "prog_python": "Expert",
     "tool_python": "Regular", "prog_typescript": "Proficient", "prog_javascript": "Familiar",
     "tool_jupyter": "Power user"},
    # personality edges: strong but not opposite extremes
    {"bfi2_facet_anxiety": "Very high", "big5_anxiety": "Low", "bfi2_domain_extraversion": "Very high",
     "bfi2_facet_sociability": "High", "bfi2_facet_assertiveness": "High", "bfi2_facet_energy_level": "Average",
     "mbti_type": "ESTJ — Executive", "lstyle_social_battery": "Extrovert",
     "pref_plan_vs_spontaneous": "Highly planned"},
    {"mbti_type": "INFP — Mediator", "lstyle_social_battery": "Introvert", "bfi2_domain_extraversion": "Low",
     "pref_plan_vs_spontaneous": "Spontaneous", "trait_discipline": "Signature", "big5_self_discipline": "Low"},
    # values edges
    {"schwartz_value_power": "Very high", "val_power_influence": "Minor",
     "values_priority": "Autonomy", "val_independence": "Minor", "val_personal_freedom": "Important"},
    # food edges
    {"cuis_sushi": "Love", "cuis_japanese": "Neutral", "lstyle_diet_type": "Vegetarian",
     "cuis_vegan": "Avoid", "health_dietary_restriction": "Ethical"},
    # income / education edges
    {"demo_household_income": "<$25k", "socioeconomic_band": "Middle", "kr_subjective_class": "Upper-middle"},
    {"age_bracket": "25-34", "lifex_formative_decade": "2000s", "highest_education": "Doctorate",
     "research_output": "Thesis only", "lifex_education_journey": "Advanced degrees",
     "institution_tier": "Top-tier research", "years_experience": "3-5"},
]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_check_values_empty():
    assert m.check_values(CATALOG) == []


def test_rule_metadata():
    ids = [r["id"] for r in m.RULES]
    assert len(ids) == len(set(ids))
    assert 120 <= len(ids) <= 190, len(ids)
    n_hard = sum(r["severity"] == "hard" for r in m.RULES)
    assert n_hard >= 70, n_hard
    pat = re.compile(r"^R3-[A-Z]{3}-([HS])\d{2}$")
    for r in m.RULES:
        mt = pat.match(r["id"])
        assert mt, r["id"]
        assert r["severity"] in ("hard", "soft")
        assert (mt.group(1) == "H") == (r["severity"] == "hard"), r["id"]
        assert r["group"] and r["description"] and "\n" not in r["description"]
        assert callable(r["scope"]) and callable(r["hit"])
    pop = [r for r in m.RULES if r["group"] == "population"]
    assert pop and all(r["id"].startswith("R3-POP-") for r in pop)


def test_all_rules_run_on_full_frame():
    df = frame([{}])
    hits, scopes, status = m.violation_matrix(df)
    bad = {k: v for k, v in status.items() if v != "ok"}
    assert not bad, bad
    assert list(hits.columns) == [r["id"] for r in m.RULES]
    assert hits.dtypes.eq(bool).all() and scopes.dtypes.eq(bool).all()


def test_base_row_is_consistent():
    hits, _, _ = m.violation_matrix(frame([{}]))
    fired = [c for c in hits.columns if hits[c].iloc[0]]
    assert not fired, fired


def test_every_rule_has_a_contradictory_example():
    missing = [r["id"] for r in m.RULES if r["id"] not in BAD]
    extra = [k for k in BAD if k not in {r["id"] for r in m.RULES}]
    assert not missing and not extra, (missing, extra)


def test_contradictory_rows_are_caught():
    ids = [r["id"] for r in m.RULES]
    df = frame([BAD[i] for i in ids])
    hits, scopes, status = m.violation_matrix(df)
    assert all(v == "ok" for v in status.values())
    missed = [rid for k, rid in enumerate(ids) if not hits[rid].iloc[k]]
    assert not missed, missed
    for k, rid in enumerate(ids):
        assert scopes[rid].iloc[k], rid


def test_extra_contradictory_clauses_are_caught():
    df = frame([ov for _, ov in BAD_EXTRA])
    hits, _, status = m.violation_matrix(df)
    assert all(v == "ok" for v in status.values())
    missed = [(k, rid) for k, (rid, _) in enumerate(BAD_EXTRA) if not hits[rid].iloc[k]]
    assert not missed, missed


def test_softened_clauses_are_not_hard():
    # Teaching without education-industry experience and a coding-free Copilot power user are soft only.
    df = frame([{"role_function": "Teaching", "ind_education": "None"},
                {"tool_github_copilot": "Power user", "skill_coding": "None", "prog_go": "None"}])
    hits, _, _ = m.violation_matrix(df)
    sev = {r["id"]: r["severity"] for r in m.RULES}
    for i in range(2):
        fired = [c for c in hits.columns if hits[c].iloc[i]]
        assert fired and all(sev[c] == "soft" for c in fired), fired


def test_contradictory_rows_also_caught_as_category():
    ids = [r["id"] for r in m.RULES]
    df_s = frame([BAD[i] for i in ids])
    df_c = frame([BAD[i] for i in ids], as_category=True)
    assert all(str(t) == "category" for t in df_c.dtypes)
    h_s, sc_s, st_s = m.violation_matrix(df_s)
    h_c, sc_c, st_c = m.violation_matrix(df_c)
    assert st_s == st_c
    assert h_s.equals(h_c) and sc_s.equals(sc_c)


def test_boundary_rows_trigger_nothing():
    df = frame(BOUNDARY)
    hits, _, status = m.violation_matrix(df)
    assert all(v == "ok" for v in status.values())
    fired = {i: [c for c in hits.columns if hits[c].iloc[i]] for i in range(len(df))}
    fired = {i: f for i, f in fired.items() if f}
    assert not fired, fired


def test_hits_only_inside_scope():
    df = frame([BAD[r["id"]] for r in m.RULES])
    hits, scopes, _ = m.violation_matrix(df)
    assert not (hits & ~scopes).any().any()
    custom = [{"id": "X-1", "group": "t", "severity": "hard", "description": "t",
               "scope": lambda d: m._in(d, "age_bracket", "85+"),
               "hit": lambda d: m._rows(d, "age_bracket")}]
    h, s, st = m.violation_matrix(frame([{}, {"age_bracket": "85+"}]), custom)
    assert st["X-1"] == "ok" and h["X-1"].tolist() == [False, True] and s["X-1"].tolist() == [False, True]


def test_missing_columns_skip_cleanly():
    df = frame([BAD[r["id"]] for r in m.RULES])
    used = m.columns_by_rule(df)
    drop = ["lifex_travel_breadth", "bfi2_facet_anxiety", "coding_ai_usage_frequency", "age_bracket"]
    hits, scopes, status = m.violation_matrix(df.drop(columns=drop))
    for r in m.RULES:
        rid = r["id"]
        if used[rid] & set(drop):
            assert status[rid].startswith("skipped: missing "), (rid, status[rid])
            assert status[rid].split("skipped: missing ")[1] in drop
            assert not hits[rid].any() and not scopes[rid].any()
        else:
            assert status[rid] == "ok", (rid, status[rid])
    # a frame with only a handful of columns: nothing raises, everything else skipped
    tiny = df[["age_bracket", "region"]]
    hits, scopes, status = m.violation_matrix(tiny)
    assert status["R3-POP-H01"] == "ok" and status["R3-POP-H04"] == "ok"
    assert all(v == "ok" or v.startswith("skipped: missing ") for v in status.values())
    table, rate = m.run(pd.DataFrame(index=range(3)))
    assert (table["status"].str.startswith("skipped")).all() and np.isnan(rate)


def test_errors_are_reported_not_raised():
    def boom(d):
        raise ValueError("kaput")

    custom = [{"id": "X-err", "group": "t", "severity": "hard", "description": "t",
               "scope": lambda d: m._rows(d, "age_bracket"), "hit": boom}]
    h, s, st = m.violation_matrix(frame([{}]), custom)
    assert st["X-err"].startswith("error: ValueError") and not h["X-err"].any() and not s["X-err"].any()


def test_nan_cells_trigger_nothing():
    df = frame([{}, {}])
    df.iloc[:, :] = np.nan
    hits, _, status = m.violation_matrix(df)
    assert all(v == "ok" for v in status.values())
    assert not hits.any().any()


def test_run_table_and_any_hard_rate():
    rows = [{}, BAD["R3-POP-H04"], BAD["R3-LIF-S04"]]
    table, rate = m.run(frame(rows))
    assert list(table.columns) == ["id", "group", "severity", "description", "n_scope", "n_hit",
                                   "violation_rate", "status"]
    assert len(table) == len(m.RULES)
    t = table.set_index("id")
    assert t.loc["R3-POP-H04", "n_scope"] == 3 and t.loc["R3-POP-H04", "n_hit"] == 1
    assert abs(t.loc["R3-POP-H04", "violation_rate"] - 1 / 3) < 1e-12
    assert t.loc["R3-LIF-S04", "n_hit"] == 1
    assert abs(rate - 1 / 3) < 1e-12  # only the region row violates a hard rule
    empty_scope = t[t["n_scope"] == 0]
    assert empty_scope["violation_rate"].isna().all()


def test_columns_read_only_through_accessor():
    used = m.columns_by_rule(frame([{}]))
    for rid, cols in used.items():
        assert cols, rid
        assert cols <= set(IDS), (rid, cols - set(IDS))
    # static check: the persona frame is subscripted only inside _s
    src = open(m.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    offenders = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.Lambda)):
            name = getattr(fn, "name", "<lambda>")
            for node in ast.walk(fn):
                if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                        and node.value.id in ("df", "d", "frame") and name != "_s"):
                    offenders.append((name, node.lineno))
    assert not offenders, offenders


def test_minor_age_and_label_semantics():
    # '18-24' is an adult bracket (19-24), not a population violation
    hits, _, _ = m.violation_matrix(frame([BOUNDARY[1]]))
    assert not hits["R3-POP-H01"].iloc[0]


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {str(e)[:2000]}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    sys.exit(1 if failed else 0)
