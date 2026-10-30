"""Tests for krpersona.rules.set4 (run with `python -m pytest tests/rules/test_set4.py`; pytest-compatible too)."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from krpersona.rules import set4 as m  # noqa: E402

CATALOG = os.path.join(os.path.dirname(m.__file__), 'catalogs', 'set4.json')
with open(CATALOG, encoding='utf-8') as _f:
    CAT = json.load(_f)
ALLOWED = {a['id']: set(a['values']) | set(a['extra_values_not_in_matrAIx']) for a in CAT['attributes']}

# ---------------------------------------------------------------------------
# a consistent baseline persona (every catalog column)
# ---------------------------------------------------------------------------
NEUTRAL = ['Average', 'Moderate', 'Neutral', 'Balanced', 'Mixed', 'Either', 'Sometimes', 'Situational',
           'Occasional', 'Familiar', 'Intermediate', 'Like', 'Casual', 'Some exposure', 'Weekly']


def make_base():
    base = {}
    for a in CAT['attributes']:
        vals = a['values']
        base[a['id']] = next((n for n in NEUTRAL if n in vals), vals[len(vals) // 2])
    for c in m._LANG_COLS:
        base[c] = 'None'
    for c in m._CULT_FOREIGN:
        base[c] = 'Unfamiliar'
    base.update(dict(
        age_bracket='35-44', demo_generation='Millennial', lifex_formative_decade='1990s',
        region='East Asia', demo_citizenship_status='Citizen by birth', demo_ethnicity_broad='East Asian',
        lifex_immigration_generation='Native multi-generational', primary_language='Korean',
        lang_korean='Native', lang_english='Basic', english_proficiency='Basic (A1-A2)',
        multilingualism='Monolingual', lifex_languages_lifetime='Two', cult_south_korea='Native',
        cult_japan='Visited', cult_china='Visited',
        demo_religion_affiliation='None', religiosity='Secular', lifex_faith_journey='Always secular',
        demo_marital_status='Married', demo_relationship_length='5-15 years',
        lifex_relationship_history='Married once', demo_children_count='2 children',
        demo_parental_status='Parent of minors', lifex_parenting_journey='Young children',
        life_stage='Parent of young kids', lstyle_household_size='3-4 people', demo_housing_status='Mortgage',
        demo_employment_status='Full-time', lstyle_work_schedule='9-to-5', years_experience='11-20',
        seniority='Mid', highest_education="Bachelor's", institution_tier='Mid-tier university',
        lifex_education_journey='Traditional track', research_output='None',
        demo_veteran_status='Veteran', lifex_military_history='Served briefly', gender_identity='Man',
        health_vision='Corrected', demo_driver_status='Daily driver', skill_driving='Intermediate',
        lifex_travel_breadth='A few countries', lifex_geographic_mobility='Moved nationally',
        lifex_hometown_tie='Visits often', lifex_loss_experience='Lost a grandparent',
        major_life_events='None notable', lifex_health_journey='No major events',
        health_chronic_condition='None', demo_disability_status='No disability', health_mental_health='Stable',
        kr_depression_level='Minimal', kr_anxiety_level='Minimal', kr_happiness='Happy',
        kr_life_satisfaction='Satisfied', neurotype='Neurotypical', health_neurodivergence='Neurotypical',
        health_attention_condition='None', lstyle_exercise_freq='Weekly', health_fitness_level='Average',
        health_insurance_status='Comprehensive', accessibility_needs='None',
        lstyle_device_ecosystem='Android/Google', lstyle_browser='Chrome', lstyle_primary_messenger='SMS',
        lstyle_banking_style='Traditional bank', lstyle_payment_pref='Credit card', lstyle_diet_type='Omnivore',
        dev_professional_status='Not a developer', dev_role_archetype='Not applicable', cuis_korean='Love'))
    return base


BASE = make_base()


def frame(*rows, categorical=False):
    df = pd.DataFrame([dict(BASE, **r) for r in rows])
    if categorical:
        df = df.astype('category')
    return df


# ---------------------------------------------------------------------------
# one hand-made contradictory row per rule (overrides on top of BASE)
# ---------------------------------------------------------------------------
_ALL_PROG_NONE = {c: 'None' for c in m._PROG_COLS}
_ALL_DOSPERT_LOW = {c: 'Very low' for c in m._DOSPERT}

CASES = {
    'R4-POP-H01': dict(age_bracket='13-17'),
    'R4-POP-H02': dict(demo_citizenship_status='Visa holder'),
    'R4-POP-H03': dict(region='Western Europe'),
    'R4-POP-H04': dict(lifex_immigration_generation='First-generation immigrant'),
    'R4-POP-H05': dict(cult_south_korea='Unfamiliar'),
    'R4-POP-S01': dict(demo_citizenship_status='Naturalized citizen'),
    'R4-POP-S02': dict(demo_ethnicity_broad='White / European'),
    'R4-POP-S03': dict(primary_language='English', lang_english='Fluent', english_proficiency='Fluent (C1-C2)'),
    'R4-POP-S04': dict(lang_korean='Conversational'),
    'R4-POP-S05': dict(lang_japanese='Native'),
    'R4-POP-S06': dict(demo_religion_affiliation='Muslim'),
    'R4-POP-S07': dict(cult_japan='Native'),
    'R4-POP-S08': dict(cuis_korean='Avoid'),
    'R4-AGE-H01': dict(age_bracket='18-24', demo_generation='Gen X'),
    'R4-AGE-H02': dict(age_bracket='65-74', demo_generation='Boomer', lifex_formative_decade='2010s'),
    'R4-AGE-H03': dict(age_bracket='18-24', demo_generation='Gen Z', demo_relationship_length='15+ years'),
    'R4-AGE-H04': dict(age_bracket='18-24', demo_children_count='Adult children'),
    'R4-AGE-S01': dict(age_bracket='25-34', demo_parental_status='Parent of adults'),
    'R4-AGE-S02': dict(demo_parental_status='Grandparent'),
    'R4-AGE-S03': dict(age_bracket='18-24', demo_marital_status='Widowed'),
    'R4-AGE-S04': dict(age_bracket='25-34', demo_employment_status='Retired'),
    'R4-AGE-S06': dict(age_bracket='75-84', demo_employment_status='Student'),
    'R4-FAM-H01': dict(demo_children_count='None'),
    'R4-FAM-H02': dict(demo_parental_status='Not a parent'),
    'R4-FAM-H03': dict(lifex_parenting_journey='No children'),
    'R4-FAM-H04': dict(demo_children_count='None', demo_parental_status='Not a parent'),
    'R4-FAM-H05': dict(demo_parental_status='Not a parent'),
    'R4-FAM-H06': dict(life_stage='Empty nester', demo_children_count='None'),
    'R4-FAM-H07': dict(demo_relationship_length='Not in one'),
    'R4-FAM-H08': dict(demo_marital_status='Single'),
    'R4-FAM-H09': dict(lstyle_household_size='Lives alone', demo_housing_status='Living with family'),
    'R4-FAM-H10': dict(lifex_relationship_history='Widowed', lifex_loss_experience='No major loss'),
    'R4-LIF-H01': dict(lifex_geographic_mobility='Never left hometown'),
    'R4-LIF-H02': dict(lifex_geographic_mobility='Moved internationally', lifex_travel_breadth='Never left country'),
    'R4-LIF-H03': dict(lifex_travel_breadth='Lived abroad', lifex_geographic_mobility='Never left hometown'),
    'R4-LIF-H04': dict(lifex_immigration_generation='Returnee / repatriate', lifex_travel_breadth='Never left country'),
    'R4-LIF-H05': dict(lifex_cultural_exposure='Third-culture kid', lifex_travel_breadth='Never left country'),
    'R4-LIF-H06': dict(major_life_events='Major relocation', lifex_geographic_mobility='Never left hometown'),
    'R4-LIF-H07': dict(major_life_events='Bereavement', lifex_loss_experience='No major loss'),
    'R4-LIF-H08': dict(major_life_events='Health journey'),
    'R4-LIF-H09': dict(major_life_events='Started a business', lifex_entrepreneurship_history='Considered it'),
    'R4-LIF-H10': dict(lifex_career_path_shape='Entrepreneurial', lifex_entrepreneurship_history='Never considered'),
    'R4-LIF-H12': dict(lifex_adversity_level='Sheltered', lifex_childhood_environment='Hardship'),
    'R4-LIF-S02': dict(major_life_events='Military service', lifex_military_history='Never served',
                       demo_veteran_status='Civilian'),
    'R4-MIL-H01': dict(lifex_military_history='Never served'),
    'R4-MIL-H02': dict(lifex_military_history='Career veteran', demo_veteran_status='Civilian'),
    'R4-MIL-H03': dict(age_bracket='65-74', demo_generation='Boomer', demo_veteran_status='Active duty'),
    'R4-MIL-H04': dict(demo_veteran_status='Active duty', demo_employment_status='Retired'),
    'R4-MIL-S01': dict(gender_identity='Woman'),
    'R4-MIL-S02': dict(lifex_military_history='Combat veteran'),
    'R4-MIL-S03': dict(age_bracket='45-54', demo_generation='Gen X', demo_veteran_status='Reserve / guard'),
    'R4-EDU-H01': dict(age_bracket='18-24', highest_education='Doctorate'),
    'R4-EDU-S01': dict(age_bracket='18-24', highest_education="Master's"),
    'R4-EDU-H02': dict(lifex_education_journey='Advanced degrees'),
    'R4-EDU-H03': dict(highest_education='Primary'),
    'R4-EDU-S02': dict(highest_education='Secondary', institution_tier='Top-tier research'),
    'R4-EDU-H04': dict(institution_tier='Self-taught'),
    'R4-EDU-H05': dict(highest_education='Secondary', institution_tier='Not applicable', research_output='Thesis only'),
    'R4-EDU-S05': dict(highest_education='Secondary', role_function='Clinical'),
    'R4-EMP-H01': dict(age_bracket='18-24'),
    'R4-EMP-S01': dict(age_bracket='25-34', years_experience='20+'),
    'R4-EMP-H02': dict(lstyle_work_schedule='Unemployed'),
    'R4-EMP-H03': dict(demo_employment_status='Unemployed'),
    'R4-EMP-S03': dict(age_bracket='18-24', seniority='C-suite'),
    'R4-EMP-S04': dict(age_bracket='25-34', seniority='Retired'),
    'R4-EMP-S05': dict(age_bracket='18-24', ind_retail='Veteran'),
    'R4-INC-H01': dict(demo_household_income='$200k+', socioeconomic_band='Low income'),
    'R4-INC-S01': dict(socioeconomic_band='High income', kr_subjective_class='Lower class'),
    'R4-LAN-H01': dict(lang_korean='Basic'),
    'R4-LAN-H02': dict(english_proficiency='Native'),
    'R4-LAN-H03': dict(lang_english='Fluent', english_proficiency='Fluent (C1-C2)'),
    'R4-LAN-H04': dict(multilingualism='Bilingual'),
    'R4-LAN-H05': dict(lifex_languages_lifetime='One', multilingualism='Bilingual',
                       lang_english='Conversational', english_proficiency='Intermediate (B1-B2)'),
    'R4-PER-H01': dict(bfi2_domain_open_mindedness='Very high', bfi2_facet_creative_imagination='Very low'),
    'R4-PER-H02': dict(bfi2_domain_conscientiousness='Very low', bfi2_facet_organization='Very high'),
    'R4-PER-H03': dict(bfi2_domain_extraversion='Very high', bfi2_facet_energy_level='Very low'),
    'R4-PER-H04': dict(bfi2_domain_agreeableness='Very high', bfi2_facet_trust='Very low'),
    'R4-PER-H05': dict(bfi2_domain_negative_emotionality='Very low', bfi2_facet_anxiety='Very high'),
    'R4-PER-H06': dict(bfi2_facet_intellectual_curiosity='Very high', big5_intellect='Very low'),
    'R4-PER-H07': dict(bfi2_facet_aesthetic_sensitivity='Very low', big5_artistic_interest='Very high'),
    'R4-PER-H08': dict(bfi2_facet_creative_imagination='Very high', big5_imagination='Very low'),
    'R4-PER-H09': dict(bfi2_facet_organization='Very high', big5_orderliness='Very low'),
    'R4-PER-H10': dict(bfi2_facet_responsibility='Very high', big5_dutifulness='Very low'),
    'R4-PER-H11': dict(bfi2_facet_sociability='Very low', big5_gregariousness='Very high'),
    'R4-PER-H12': dict(bfi2_facet_assertiveness='Very high', big5_assertiveness='Very low'),
    'R4-PER-H14': dict(bfi2_facet_energy_level='Very high', big5_activity_level='Very low'),
    'R4-PER-H15': dict(bfi2_facet_compassion='Very high', big5_sympathy='Very low'),
    'R4-PER-H16': dict(bfi2_facet_trust='Very high', big5_trust='Very low'),
    'R4-PER-H17': dict(bfi2_facet_anxiety='Very high', big5_anxiety='Very low'),
    'R4-PER-H18': dict(bfi2_facet_depression='Very low', big5_depression='Very high'),
    'R4-PER-H20': dict(big5_self_discipline='Very high', trait_discipline='Absent'),
    'R4-PER-H21': dict(trait_curiosity='Signature', cog_curiosity='None'),
    'R4-PER-H22': dict(trait_open_mindedness='Absent', cog_open_mindedness='Very high'),
    'R4-PER-H23': dict(trait_hope_optimism='Signature', cog_optimism='None'),
    'R4-PER-H24': dict(trait_humility='Signature', big5_modesty='Very low'),
    'R4-PER-H25': dict(trait_honesty='Signature', big5_morality='Very low'),
    'R4-PER-H26': dict(trait_prudence='Absent', big5_cautiousness='Very high'),
    'R4-PER-H27': dict(trait_creativity='Signature', bfi2_facet_creative_imagination='Very low'),
    'R4-PER-H28': dict(trait_appreciation_of_beauty='Signature', bfi2_facet_aesthetic_sensitivity='Very low'),
    'R4-PER-H29': dict(trait_kindness='Signature', big5_altruism='Very low'),
    'R4-PER-H30': dict(trait_empathy='Absent', big5_sympathy='Very high'),
    'R4-PER-H31': dict(trait_self_regulation='Signature', big5_immoderation='Very high'),
    'R4-PER-H32': dict(trait_resilience='Absent', big5_vulnerability='Very low'),
    'R4-PER-H34': dict(cog_procrastination='Very high', big5_self_discipline='Very high'),
    'R4-PER-H35': dict(need_for_closure='Very low', cog_ambiguity_tolerance='None'),
    'R4-PER-H36': dict(cog_detail_orientation='None', cog_big_picture_vs_detail='Detail-obsessed'),
    'R4-PER-H37': dict(decision_style='Deliberative', cog_decision_speed='Snap decisions'),
    'R4-PER-H38': dict(dominant_trait='High openness', bfi2_domain_open_mindedness='Very low'),
    'R4-PER-H39': dict(neurotype='Anxiety-prone', big5_anxiety='Very low'),
    'R4-PER-S01': dict(mbti_type='INTJ — Architect', bfi2_domain_extraversion='Very high'),
    'R4-PER-S02': dict(mbti_type='ISTJ — Logistician', bfi2_domain_open_mindedness='Very high'),
    'R4-HEA-H01': dict(health_vision='Blind'),
    'R4-HEA-S01': dict(health_vision='Low vision'),
    'R4-HEA-H02': dict(health_vision='Blind', demo_driver_status='Non-driver'),
    'R4-HEA-S02': dict(demo_disability_status='Sensory', health_hearing='Normal'),
    'R4-HEA-H03': dict(lifex_health_journey='Manages chronic condition'),
    'R4-HEA-H04': dict(health_mental_health='Thriving', kr_anxiety_level='Severe'),
    'R4-HEA-H05': dict(kr_depression_level='Severe', kr_happiness='Very happy'),
    'R4-HEA-S05': dict(kr_depression_level='Severe', health_energy_level='Very high'),
    'R4-HEA-H06': dict(health_neurodivergence='Autistic'),
    'R4-HEA-H07': dict(health_attention_condition='Diagnosed'),
    'R4-HEA-H08': dict(lstyle_exercise_freq='Never', health_fitness_level='Athlete'),
    'R4-HEA-S07': dict(lstyle_exercise_freq='Rarely', health_fitness_level='Athlete'),
    'R4-HEA-H09': dict(lstyle_exercise_freq='Daily', health_fitness_level='Sedentary'),
    'R4-HEA-S09': dict(age_bracket='85+', demo_generation='Silent', lifex_formative_decade='1960s-70s',
                       health_fitness_level='Athlete'),
    'R4-HEA-S10': dict(health_insurance_status='Uninsured'),
    'R4-REL-H01': dict(religiosity='Devout'),
    'R4-REL-H02': dict(demo_religion_affiliation='Protestant'),
    'R4-REL-H03': dict(religiosity='Observant', demo_religion_affiliation='Catholic'),
    'R4-REL-H04': dict(lifex_faith_journey='Lifelong faith'),
    'R4-REL-H05': dict(religiosity='Devout', demo_religion_affiliation='Catholic', lifex_faith_journey='Lifelong faith',
                       trait_spirituality='Absent'),
    'R4-REL-S01': dict(lifex_faith_journey='Converted', demo_religion_affiliation='Atheist / agnostic'),
    'R4-REL-S02': dict(demo_religion_affiliation='Buddhist', lifex_faith_journey='Lapsed / left faith'),
    'R4-REL-S04': dict(demo_religion_affiliation='Confucian', kr_confucian_way_of_life='No'),
    'R4-VAL-H01': dict(val_achievement='Core value', schwartz_value_achievement='Very low'),
    'R4-VAL-H02': dict(val_power_influence='Irrelevant', schwartz_value_power='Very high'),
    'R4-VAL-H03': dict(val_security_stability='Core value', schwartz_value_security='Very low'),
    'R4-VAL-H04': dict(val_tradition='Irrelevant', schwartz_value_tradition='Very high'),
    'R4-VAL-H05': dict(val_helping_others='Core value', schwartz_value_benevolence='Very low'),
    'R4-VAL-H06': dict(val_independence='Core value', schwartz_value_self_direction='Very low'),
    'R4-VAL-H07': dict(val_adventure='Irrelevant', schwartz_value_stimulation='Very high'),
    'R4-VAL-H08': dict(val_fun_enjoyment='Core value', schwartz_value_hedonism='Very low'),
    'R4-VAL-H09': dict(val_equality='Irrelevant', schwartz_value_universalism='Very high'),
    'R4-VAL-H10': dict(values_priority='Community', val_community='Irrelevant'),
    'R4-VAL-S01': dict(val_spirituality_faith='Core value', demo_religion_affiliation='Atheist / agnostic'),
    'R4-ATT-H01': dict(att_remote_work='Opposed', pref_work_location='Strongly remote'),
    'R4-ATT-H02': dict(att_working_from_office='Enthusiast', pref_work_location='Strongly remote'),
    'R4-ATT-H03': dict(att_risk_taking='Opposed', risk_tolerance='Risk-seeking'),
    'R4-ATT-H04': dict(risk_tolerance='Risk-seeking', **_ALL_DOSPERT_LOW),
    'R4-ATT-H05': dict(lstyle_diet_type='Vegan', att_veganism='Opposed'),
    'R4-ATT-H06': dict(political_lean='Apolitical', demo_political_engagement='Activist'),
    'R4-ATT-H07': dict(pref_early_vs_late='Bleeding edge', att_new_technology='Opposed'),
    'R4-ATT-S03': dict(media_diet='Social-media-heavy', att_social_media='Opposed'),
    'R4-ATT-S05': dict(demo_sexual_orientation='Gay / lesbian', kr_attitude_same_sex_relations='Always wrong'),
    'R4-PRF-H01': dict(pref_routine_vs_variety='Craves variety', pref_novelty_vs_familiarity='Always familiar'),
    'R4-PRF-H02': dict(pref_logic_vs_intuition='Pure intuition', decision_style='Analytical'),
    'R4-PRF-H03': dict(pref_lead_vs_follow='Prefers to follow', trait_leadership='Signature'),
    'R4-PRF-H04': dict(pref_competition_vs_collab='Highly competitive', trait_competitiveness='Absent'),
    'R4-PRF-H05': dict(pref_big_group_vs_one_on_one='One-on-one only', big5_gregariousness='Very high'),
    'R4-PRF-H06': dict(pref_detail_brief_vs_full='Every detail', cog_big_picture_vs_detail='Big-picture only'),
    'R4-PRF-H07': dict(media_diet='Minimal', habit_doomscrolling='Daily'),
    'R4-PRF-H08': dict(health_vision='Blind', demo_driver_status='Non-driver', demo_disability_status='Sensory'),
    'R4-WEL-S01': dict(kr_happiness='Very unhappy', kr_life_satisfaction='Very satisfied'),
    'R4-DEV-H01': dict(coding_ai_usage_frequency='Never used', ai_task_testing='Currently mostly AI-assisted'),
    'R4-DEV-H02': dict(coding_ai_usage_frequency='Tried but not active', coding_agent_usage_frequency='Weekly'),
    'R4-DEV-H03': dict(coding_agent_failure_tolerance='Does not use agents', coding_agent_usage_frequency='Daily',
                       coding_ai_usage_frequency='Daily'),
    'R4-DEV-H04': dict(coding_ai_usage_frequency='Actively avoids', coding_ai_learning_source='AI-enabled coding tools'),
    'R4-DEV-H05': dict(coding_ai_usage_frequency='Never used', coding_agent_workflow_impact='Major productivity increase'),
    'R4-DEV-H06': dict(coding_ai_usage_frequency='Never used', developer_help_seeking_preference='Ask AI first'),
    'R4-DEV-H07': dict(tool_github_copilot='Regular', coding_ai_usage_frequency='Never used'),
    'R4-DEV-H08': dict(coding_ai_sentiment='Very favorable', coding_ai_output_trust='Does not trust'),
    'R4-DEV-H09': dict(coding_agent_autonomy_preference='No agent autonomy',
                       coding_agent_control_expectation='Agent may run broad tasks autonomously'),
    'R4-DEV-H10': dict(dev_role_archetype='Back-end developer'),
    'R4-DEV-H11': dict(dev_professional_status='Professional developer', open_source_activity_level='No coding activity'),
    'R4-DEV-H12': dict(open_source_activity_level='No coding activity', coding_ai_usage_frequency='Daily',
                       coding_agent_usage_frequency='Daily'),
    'R4-DEV-H13': dict(pr_collaboration_style='Rare PR usage', github_contribution_mode='Pull request contributor'),
    'R4-DEV-H14': dict(stackoverflow_usage_pattern='Never used', stackoverflow_participation_style='Answers questions'),
    'R4-DEV-H15': dict(dev_ic_manager_orientation='Student / learner', dev_role_archetype='Engineering manager / lead'),
    'R4-DEV-H16': dict(code_testing_approach='No tests', bug_fix_test_behavior='Always adds regression tests'),
    'R4-DEV-H17': dict(prog_sql='Expert', tool_sql='Never used'),
    'R4-CUL-H01': dict(lifex_travel_breadth='Never left country'),
    'R4-DVC-H01': dict(lstyle_primary_messenger='iMessage', lstyle_device_ecosystem='Windows'),
    'R4-DVC-H02': dict(lstyle_browser='Safari'),
    'R4-DVC-H03': dict(lstyle_streaming_hours='16+', lstyle_screen_time='<2 hrs'),
    'R4-DVC-H04': dict(lstyle_primary_social='None', media_diet='Social-media-heavy'),
    'R4-TEC-H01': dict(tech_savviness='Avoidant', pref_early_vs_late='Early adopter'),
    'R4-TEC-S01': dict(age_bracket='75-84', demo_generation='Silent', lifex_formative_decade='1960s-70s',
                       tech_savviness='Digital native'),
    'R4-TEC-S02': dict(tool_claude='Power user', att_ai='Opposed'),
    'R4-LST-H01': dict(lstyle_banking_style='Unbanked', lstyle_payment_pref='Debit card'),
    'R4-LST-H02': dict(topic_cryptocurrency='Averse', lstyle_payment_pref='Crypto'),
    'R4-LST-H03': dict(lstyle_social_battery='Strong extrovert', bfi2_domain_extraversion='Very low'),
    'R4-LST-H04': dict(lstyle_volunteering='Daily', lifex_service_history='None'),
    'R4-LST-H05': dict(lstyle_reading_freq='Never', habit_reading_before_bed='Daily'),
    'R4-LST-H06': dict(lstyle_cooking_freq='Never', habit_meal_prepping='Weekly'),
    'R4-LST-H07': dict(lstyle_tidiness='Spotless', big5_orderliness='Very low'),
    'R4-LST-H08': dict(lstyle_frugality='Splurger', pref_save_vs_spend='Hard saver'),
    'R4-LST-H09': dict(sport_esports='Play', lstyle_gaming_freq='Never'),
    'R4-LST-S01': dict(topic_craft_beer='Passionate', lstyle_alcohol_use='Never'),
    'R4-INT-H01': dict(topic_astronomy='Averse', hob_stargazing='Avid'),
    'R4-INT-H02': dict(acad_world_history='Averse', topic_history='Passionate'),
    'R4-INT-H03': dict(topic_meditation='Averse', habit_meditation='Daily'),
    'R4-FOD-H01': dict(lstyle_diet_type='Vegan', cuis_vegan='Avoid'),
    'R4-FOD-S01': dict(lstyle_diet_type='Vegan', cuis_american_bbq='Love'),
    'R4-SKL-H01': dict(demo_driver_status='Cannot drive', skill_driving='Master'),
    'R4-SKL-H02': dict(skill_coding='Advanced', **_ALL_PROG_NONE),
    'R4-SKL-H03': dict(skill_interpretation='Advanced'),
    'R4-SKL-H04': dict(skill_photography='Master', fam_photography='None'),
}

# ---------------------------------------------------------------------------
# consistent rows near rule boundaries: must trigger nothing
# ---------------------------------------------------------------------------
BOUNDARY = [
    # youngest bracket at the edges of every age rule
    dict(age_bracket='18-24', demo_generation='Gen Z', lifex_formative_decade='2000s', years_experience='6-10',
         demo_relationship_length='5-15 years', seniority='Lead / Principal', highest_education="Bachelor's",
         demo_veteran_status='Reserve / guard'),
    dict(age_bracket='18-24', demo_generation='Millennial', lifex_formative_decade='2020s',
         demo_marital_status='Separated', years_experience='0-2'),
    dict(age_bracket='25-34', demo_generation='Gen Z', lifex_formative_decade='1990s', years_experience='11-20',
         highest_education='Doctorate', demo_children_count='3+ children'),
    dict(age_bracket='35-44', demo_generation='Gen X', lifex_formative_decade='2020s', seniority='Retired',
         demo_employment_status='Retired', lstyle_work_schedule='Unemployed', life_stage='Retirement',
         demo_veteran_status='Reserve / guard'),
    dict(age_bracket='45-54', demo_generation='Boomer', lifex_formative_decade='1960s-70s',
         demo_children_count='Adult children', demo_parental_status='Grandparent',
         lifex_parenting_journey='Raising grandchildren', life_stage='Mid-life'),
    dict(age_bracket='55-64', demo_generation='Gen X', lifex_formative_decade='2000s', demo_veteran_status='Active duty',
         lifex_military_history='Career veteran', tech_savviness='Digital native'),
    dict(age_bracket='65-74', demo_generation='Silent', lifex_formative_decade='1990s',
         lifex_military_history='Combat veteran', demo_children_count='Adult children',
         demo_parental_status='Parent of adults', lifex_parenting_journey='Grown children', life_stage='Empty nester',
         demo_employment_status='Retired', lstyle_work_schedule='Unemployed', seniority='Retired',
         lifex_career_path_shape='Just starting'),
    dict(age_bracket='75-84', demo_generation='Boomer', lifex_formative_decade='1980s', health_fitness_level='Fit',
         demo_employment_status='Part-time', lstyle_work_schedule='Flexible hours'),
    dict(age_bracket='85+', demo_generation='Silent', lifex_formative_decade='1960s-70s', health_fitness_level='Fit'),
    # language: distance 2 between the two English scales; Bilingual with one conversational L2
    dict(english_proficiency='Native', lang_english='Conversational', multilingualism='Bilingual'),
    dict(english_proficiency='Basic (A1-A2)', lang_english='Fluent', multilingualism='Bilingual'),
    dict(multilingualism='Monolingual', lang_english='Conversational', english_proficiency='Intermediate (B1-B2)'),
    # personality: one extreme vs the next-to-opposite level is not flagged
    dict(bfi2_domain_open_mindedness='Very high', bfi2_facet_creative_imagination='Low',
         bfi2_facet_anxiety='Very high', big5_anxiety='Low', trait_curiosity='Signature', cog_curiosity='Low',
         trait_self_regulation='Signature', big5_immoderation='Very low', cog_procrastination='None',
         big5_self_discipline='Very high', trait_discipline='Slight', need_for_closure='Very high',
         cog_ambiguity_tolerance='None', dominant_trait='High openness', mbti_type='INFJ — Advocate'),
    dict(mbti_type='INTJ — Architect', bfi2_domain_extraversion='High', bfi2_domain_open_mindedness='Very high', dominant_trait='High extraversion',
         lstyle_social_battery='Strong introvert'),
    dict(decision_style='Deliberative', cog_decision_speed='Quick', pref_logic_vs_intuition='Pure logic'),
    # values: core value vs low (not very low)
    dict(val_achievement='Core value', schwartz_value_achievement='Low', values_priority='Achievement'),
    # developer / AI: equal frequencies; excluded non-coding task; no-agent person
    dict(dev_professional_status='Professional developer', dev_role_archetype='Back-end developer',
         skill_coding='Intermediate', open_source_activity_level='Private-only developer', prog_python='Expert',
         tool_python='Occasional', coding_ai_usage_frequency='Weekly', coding_agent_usage_frequency='Weekly'),
    dict(coding_ai_usage_frequency='Tried but not active', coding_agent_usage_frequency='Experimental only'),
    dict(coding_ai_usage_frequency='Never used', coding_agent_usage_frequency='No plans',
         ai_task_search_answers='Currently mostly AI-assisted', ai_task_learning_concepts='Currently mostly AI-assisted',
         coding_agent_failure_tolerance='Does not use agents', tool_github_copilot='Tried it',
         coding_agent_workflow_impact='Not applicable', developer_help_seeking_preference='Not applicable'),
    dict(dev_professional_status='Not a developer', dev_role_archetype='Product / UX technical role',
         coding_ai_sentiment='Not applicable', coding_ai_output_trust='Not applicable'),
    # health
    dict(health_vision='Low vision', demo_driver_status='Occasional driver', health_mental_health='Thriving',
         kr_depression_level='Moderate', lstyle_exercise_freq='Weekly', health_fitness_level='Athlete'),
    dict(health_vision='Blind', demo_driver_status='Cannot drive', demo_disability_status='Sensory',
         accessibility_needs='Visual', skill_driving='None'),
    # culture / travel
    dict(lifex_travel_breadth='Never left country', cult_japan='Studied', cult_china='Unfamiliar'),
    # religion
    dict(demo_religion_affiliation='Protestant', religiosity='Nominal', lifex_faith_journey='Lapsed / left faith'),
    dict(demo_religion_affiliation='Buddhist', religiosity='Devout', lifex_faith_journey='Lifelong faith',
         val_spirituality_faith='Core value', trait_spirituality='Signature'),
]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_check_values_empty():
    assert m.check_values(CATALOG) == []


def test_rule_metadata():
    ids = [r['id'] for r in m.RULES]
    assert len(ids) == len(set(ids))
    for r in m.RULES:
        assert r['id'].startswith('R4-')
        assert r['severity'] in ('hard', 'soft')
        assert r['group'] and r['description']
        assert callable(r['scope']) and callable(r['hit'])
    n_hard = sum(r['severity'] == 'hard' for r in m.RULES)
    assert n_hard >= 70


def test_every_rule_has_a_case():
    missing = [r['id'] for r in m.RULES if r['id'] not in CASES]
    extra = [k for k in CASES if k not in {r['id'] for r in m.RULES}]
    assert not missing, missing
    assert not extra, extra


def test_case_values_are_catalog_values():
    bad = []
    for rid, ov in CASES.items():
        for col, v in ov.items():
            ok = ALLOWED.get(col, set()) | {'Not applicable'}
            if v not in ok:
                bad.append((rid, col, v))
    for row in BOUNDARY:
        for col, v in row.items():
            if v not in ALLOWED.get(col, set()) | {'Not applicable'}:
                bad.append(('boundary', col, v))
    assert not bad, bad


def test_baseline_triggers_nothing():
    df = frame({})
    hits, scopes, status = m.violation_matrix(df)
    assert all(v == 'ok' for v in status.values()), {k: v for k, v in status.items() if v != 'ok'}
    fired = [c for c in hits.columns if hits[c].any()]
    assert not fired, fired


def test_each_rule_catches_its_case():
    rids = list(CASES)
    df = frame(*[CASES[r] for r in rids])
    for categorical in (False, True):
        d = df.astype('category') if categorical else df
        hits, scopes, status = m.violation_matrix(d)
        assert all(v == 'ok' for v in status.values())
        missed = [r for i, r in enumerate(rids) if not hits[r].iloc[i]]
        assert not missed, (categorical, missed)
        # hits are always inside the scope
        assert not (hits & ~scopes).any().any()


def test_boundary_rows_trigger_nothing():
    df = frame(*BOUNDARY)
    hits, _, status = m.violation_matrix(df)
    assert all(v == 'ok' for v in status.values())
    fired = {i: [c for c in hits.columns if hits[c].iloc[i]] for i in range(len(df))}
    fired = {i: f for i, f in fired.items() if f}
    assert not fired, fired


def test_missing_columns_skip_cleanly():
    df = frame({}, CASES['R4-AGE-H01'])[['age_bracket', 'demo_generation']]
    hits, scopes, status = m.violation_matrix(df)
    assert status['R4-AGE-H01'] == 'ok'
    assert hits['R4-AGE-H01'].tolist() == [False, True]
    for rid, st in status.items():
        assert st == 'ok' or st.startswith('skipped: missing '), (rid, st)
        if st != 'ok':
            assert not hits[rid].any() and not scopes[rid].any()
    assert status['R4-FAM-H09'] == 'skipped: missing lstyle_household_size'
    # a frame without any column
    empty = pd.DataFrame(index=range(3))
    hits, scopes, status = m.violation_matrix(empty)
    assert all(st.startswith('skipped: missing ') for st in status.values())
    assert hits.shape == (3, len(m.RULES)) and not hits.values.any()
    # dropping one column from a full frame only skips the rules that read it
    full = frame(*[CASES[r] for r in CASES])
    part = full.drop(columns=['lang_swahili'])
    _, _, st2 = m.violation_matrix(part)
    skipped = sorted(k for k, v in st2.items() if v != 'ok')
    assert skipped == sorted(['R4-POP-S05', 'R4-LAN-H01', 'R4-LAN-H03', 'R4-LAN-H04', 'R4-LAN-H05']), skipped


def test_missing_values_are_no_hit():
    df = frame(CASES['R4-FAM-H09'], CASES['R4-AGE-H01'], CASES['R4-LAN-H02'])
    df.loc[0, 'demo_housing_status'] = np.nan
    df.loc[1, 'demo_generation'] = None
    df.loc[2, 'lang_english'] = np.nan
    for d in (df, df.astype('category')):
        hits, scopes, status = m.violation_matrix(d)
        assert not hits.loc[0, 'R4-FAM-H09']
        assert not hits.loc[1, 'R4-AGE-H01']
        assert not hits.loc[2, 'R4-LAN-H02']
        assert not scopes.loc[2, 'R4-LAN-H02']
        # population rules with a missing value are out of scope, not hits
        d2 = d.copy()
        d2['region'] = d2['region'].astype(object)
        d2.loc[0, 'region'] = np.nan
        h2, s2, _ = m.violation_matrix(d2)
        assert not h2.loc[0, 'R4-POP-H03'] and not s2.loc[0, 'R4-POP-H03']


def test_errors_are_reported_not_raised():
    bad_rule = dict(id='R4-TST-H99', group='test', severity='hard', description='x',
                    scope=lambda df: m._in(df, 'age_bracket', '35-44'),
                    hit=lambda df: 1 / 0)
    hits, scopes, status = m.violation_matrix(frame({}), rules=[bad_rule])
    assert status['R4-TST-H99'].startswith('error: ZeroDivisionError')
    assert not hits['R4-TST-H99'].any()


def test_run_table_and_rate():
    rows = [{}, CASES['R4-AGE-H01'], CASES['R4-HEA-S10'], CASES['R4-FAM-H09']]
    df = frame(*rows)
    table, rate = m.run(df)
    assert list(table.columns) == ['id', 'group', 'severity', 'description', 'n_scope', 'n_hit',
                                   'violation_rate', 'status']
    assert len(table) == len(m.RULES)
    t = table.set_index('id')
    assert t.loc['R4-AGE-H01', 'n_hit'] == 1
    assert t.loc['R4-AGE-H01', 'n_scope'] == 4
    assert abs(t.loc['R4-AGE-H01', 'violation_rate'] - 0.25) < 1e-12
    assert t.loc['R4-HEA-S10', 'n_hit'] == 1 and t.loc['R4-HEA-S10', 'severity'] == 'soft'
    # rows 1 and 3 violate a hard rule; row 2 only soft rules -> 2/4
    assert abs(rate - 0.5) < 1e-12, rate
    # empty-scope rules report NaN
    t0, rate0 = m.run(frame({}).iloc[:0])
    assert t0['violation_rate'].isna().all() and np.isnan(rate0)


def test_columns_discoverable_by_wrapping_accessor():
    seen = {}
    orig = m._s

    def spy(df, col):
        seen.setdefault(current[0], set()).add(col)
        return orig(df, col)
    current = [None]
    m._s = spy
    try:
        df = frame({})
        for r in m.RULES:
            current[0] = r['id']
            m.violation_matrix(df, rules=[r])
    finally:
        m._s = orig
    for r in m.RULES:
        cols = seen.get(r['id'], set())
        assert len(cols) >= 1, r['id']
        assert cols <= set(ALLOWED), (r['id'], cols - set(ALLOWED))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print('PASS', t.__name__)
        except AssertionError as e:
            failed += 1
            print('FAIL', t.__name__, str(e)[:2000])
    print('%d/%d passed' % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
