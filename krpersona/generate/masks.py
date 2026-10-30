"""Generation-time masks on MatrAIx attributes (attributes the Korean blocks do not make).

conflict_masks       static masks: a MatrAIx value is forbidden given a value of another attribute (written from rule sets 1-3)
dyn_language_masks   masks computed from what is already drawn (number of languages, home language)
nondev_masks         developer-only attributes take their "none" value for people who are not developers
rule_pair_masks      masks compiled from the pairwise parts of rule sets 1-3 (generate/compile_masks.py)
"""
import os, json, pickle, numpy as np
from krpersona import paths
from krpersona.matraix.graph import graph_values  # noqa: F401  (re-exported for callers)

NOT_DEV = 'Not a developer'; NA_DEV = 'Not applicable'
# step 5 round 2: developer-only attributes. A non-developer gets the attribute's "none" value (masked draw) or 'Not applicable' (output override).
DEV_TOOLS = ['tool_git', 'tool_github', 'tool_gitlab', 'tool_vs_code', 'tool_jetbrains_ides', 'tool_vim', 'tool_linux_cli', 'tool_docker', 'tool_kubernetes', 'tool_terraform',
             'tool_aws', 'tool_azure', 'tool_google_cloud', 'tool_postman', 'tool_jupyter', 'tool_github_copilot', 'tool_python', 'tool_sql', 'tool_r', 'tool_matlab',
             'tool_linear', 'tool_unity', 'tool_unreal_engine']
NONDEV_VALUE = {'skill_coding': 'None', 'open_source_activity_level': 'No coding activity', 'coding_ai_usage_frequency': 'Never used', 'coding_agent_usage_frequency': 'No plans',
                'coding_agent_failure_tolerance': 'Does not use agents', 'stackoverflow_usage_pattern': 'Never used', 'stackoverflow_participation_style': 'Does not participate',
                'coding_ai_learning_source': 'Not learning AI tooling', 'dev_ic_manager_orientation': 'Not applicable', 'code_observability_habit': 'Not applicable',
                'release_maintenance_orientation': 'Not involved in releases', 'pr_collaboration_style': 'Rare PR usage', 'issue_engagement_style': 'Rare issue usage',
                'github_contribution_mode': 'Rare GitHub user', 'code_review_participation_level': 'Rare / no review'}
def nondev_plan(V):
    """-> (attribute -> forced value for non-developers, attributes shown as 'Not applicable' for non-developers)"""
    cat = {d['id']: d.get('category', '') for d in json.load(open(paths.dims_path()))['dimensions']}
    dev = [t for t in V if t in cat and (cat[t].startswith('Developer:') or cat[t] == 'Skills: Programming') and t != 'dev_professional_status']
    val = dict(NONDEV_VALUE)
    for t in V:
        if t.startswith('prog_'): val[t] = 'None'
        if t.startswith('ai_task_'): val[t] = 'Not applicable'
    for t in DEV_TOOLS: val[t] = 'Never used'
    for t, v in val.items(): assert v in V[t], (t, v)
    na = [t for t in dev if t not in val]
    return val, na


def nondev_masks(V):
    val, _ = nondev_plan(V); sent = len(V['dev_professional_status']); out = []
    for t, v in val.items():
        k = len(V[t]); oh = np.zeros(k, np.float32); oh[V[t].index(v)] = 1
        def f(codes, avail, oh=oh, k=k):
            rows = codes['dev_professional_status'] == sent
            if not rows.any(): return None
            m = np.ones((k, len(rows)), np.float32); m[:, rows] = oh[:, None]; return m
        out.append((t, f))
    return out


def rule_pair_masks(K):
    """round 3: masks compiled from the pairwise parts of the fixing rule sets (compile_masks.py -> rule_masks.pkl)"""
    path = paths.art('masks', 'rule_masks.pkl')
    if not path.exists() or os.environ.get('KR_NO_RULE_MASKS') == '1': return []   # KR_NO_RULE_MASKS=1: baseline sample for compile_masks
    out = []
    for (t, c), Fm in pickle.load(open(path, 'rb')).items():
        if t in K: continue
        A = (~Fm).astype(np.float32)          # (len(V[c]) + 1, len(V[t])): 1 = allowed
        def f(codes, avail, A=A, c=c):
            if c not in avail: return None
            return A[np.minimum(codes[c].astype(np.int64), A.shape[0] - 1)].T
        out.append((t, f))
    return out


# specialty -> domains where it plausibly occurs (first independent rule set, EX-S02); 'Operations' fits every domain
SPEC_DOM = {'Machine learning': ['Agriculture', 'Business & Management', 'Education', 'Engineering', 'Finance & Economics', 'Healthcare & Medicine', 'Manufacturing', 'Media & Journalism', 'Natural Sciences', 'Public Sector', 'Social Sciences', 'Software & AI'],
            'Cardiology': ['Education', 'Healthcare & Medicine', 'Natural Sciences'], 'Constitutional law': ['Education', 'Law & Policy', 'Media & Journalism', 'Public Sector', 'Social Sciences'],
            'Quant trading': ['Business & Management', 'Engineering', 'Finance & Economics', 'Natural Sciences', 'Software & AI'],
            'Curriculum design': ['Arts & Humanities', 'Business & Management', 'Education', 'Healthcare & Medicine', 'Public Sector', 'Social Sciences', 'Software & AI'],
            'Structural engineering': ['Education', 'Engineering', 'Manufacturing', 'Public Sector', 'Skilled Trades'], 'Molecular biology': ['Agriculture', 'Education', 'Engineering', 'Healthcare & Medicine', 'Natural Sciences'],
            'Behavioral economics': ['Business & Management', 'Education', 'Finance & Economics', 'Law & Policy', 'Public Sector', 'Social Sciences', 'Software & AI'],
            'Comparative literature': ['Arts & Humanities', 'Education', 'Media & Journalism', 'Social Sciences'], 'Agronomy': ['Agriculture', 'Education', 'Manufacturing', 'Natural Sciences', 'Public Sector'],
            'Robotics': ['Education', 'Engineering', 'Healthcare & Medicine', 'Manufacturing', 'Natural Sciences', 'Software & AI'], 'Investigative reporting': ['Law & Policy', 'Media & Journalism', 'Public Sector', 'Social Sciences'],
            'Urban planning': ['Arts & Humanities', 'Business & Management', 'Education', 'Engineering', 'Law & Policy', 'Public Sector', 'Social Sciences'],
            'Culinary arts': ['Agriculture', 'Arts & Humanities', 'Business & Management', 'Education', 'Hospitality', 'Skilled Trades'],
            'Electrical work': ['Agriculture', 'Engineering', 'Hospitality', 'Manufacturing', 'Public Sector', 'Skilled Trades']}
ADV_FAM = ['fam_particle_physics', 'fam_astrophysics', 'fam_molecular_biology', 'fam_biochemistry', 'fam_organic_chemistry', 'fam_physical_chemistry', 'fam_nanotechnology', 'fam_materials_science',
           'fam_nuclear_engineering', 'fam_epidemiology', 'fam_actuarial_science', 'fam_compilers', 'fam_cryptography', 'fam_cardiology', 'fam_neurology', 'fam_oncology', 'fam_radiology', 'fam_surgery',
           'fam_immunology', 'fam_pharmacology', 'fam_constitutional_law', 'fam_tax_law', 'fam_quantitative_trading', 'fam_deep_learning', 'fam_distributed_systems']


def conflict_masks(V, K):
    """step 5 generation-time masks (target, condition, multiplier) so that MatrAIx (5) values do not contradict Korean values or each other.
    excl(a, A, b, B): a in A and b in B cannot hold together; the mask goes on whichever of the two MatrAIx draws later (a Korean node is
    always drawn first). Written from the first independent rule set (hard rules, plus near-certain soft ones such as two attributes that
    describe the same fact); the second, unseen rule set measures the result."""
    def chk(t, vals):
        miss = [v for v in vals if v not in V[t]]; assert not miss, (t, miss)
    def bad(t, vals): chk(t, vals); return np.array([0.0 if v in vals else 1.0 for v in V[t]])
    def only(t, vals): chk(t, vals); return np.array([1.0 if v in vals else 0.0 for v in V[t]])
    def excl(a, A, b, B):
        chk(a, A); chk(b, B); assert not (a in K and b in K), (a, b)
        return ([(b, {a: A}, bad(b, B))] if b not in K else []) + ([(a, {b: B}, bad(a, A))] if a not in K else [])
    NEVER_ABROAD = ['Never left country']; COL = ['Top-tier research', 'Mid-tier university', 'Community college']
    NONSTUDENT = [v for v in V['demo_employment_status'] if v != 'Student']; NEVER_ENT = ['Never considered', 'Considered it']
    ms = []
    # mobility and travel: never abroad -> no foreign residence or visit, no migration; never left hometown -> still there, no relocation
    for t in V:
        if t.startswith('cult_') and t != 'cult_south_korea': ms += excl('lifex_travel_breadth', NEVER_ABROAD, t, ['Lived there', 'Visited'])
    ms += excl('lifex_travel_breadth', NEVER_ABROAD, 'lifex_cultural_exposure', ['Third-culture kid', 'Lived across cultures'])
    ms += excl('lifex_travel_breadth', NEVER_ABROAD, 'major_life_events', ['Migration / immigration'])
    ms += excl('lifex_geographic_mobility', ['Never left hometown'], 'major_life_events', ['Migration / immigration', 'Major relocation'])
    ms += excl('lifex_geographic_mobility', ['Never left hometown'], 'lifex_hometown_tie', ['Visits often', 'Occasional return', 'Rarely returns', 'No connection'])
    ms += excl('lifex_geographic_mobility', ['Moved nationally', 'Moved internationally'], 'lifex_hometown_tie', ['Still there'])
    # career
    ms += excl('years_experience', ['11-20', '20+'], 'lifex_career_path_shape', ['Just starting'])
    ms += excl('demo_employment_status', ['Retired'], 'lifex_career_path_shape', ['Just starting'])
    ms += excl('seniority', ['Founder'], 'lifex_entrepreneurship_history', NEVER_ENT)
    ms += excl('lifex_career_path_shape', ['Entrepreneurial'], 'lifex_entrepreneurship_history', NEVER_ENT)
    ms += excl('lifex_entrepreneurship_history', NEVER_ENT, 'major_life_events', ['Started a business'])
    # education -> institution type (Korean system: 전문대 = community college; 4-year and graduate degrees = universities, the open/cyber universities counted as universities)
    ms += [('institution_tier', {'highest_education': ['No formal', 'Primary']}, bad('institution_tier', COL)),
           ('institution_tier', {'highest_education': ['Secondary'], 'demo_employment_status': NONSTUDENT}, bad('institution_tier', COL)),
           ('institution_tier', {'highest_education': ['Some college']}, bad('institution_tier', ['Not applicable', 'Self-taught'])),
           ('institution_tier', {'highest_education': ["Associate's"]}, only('institution_tier', ['Community college'])),
           ('institution_tier', {'highest_education': ["Bachelor's", "Master's", 'Doctorate', 'Postdoc']}, only('institution_tier', ['Top-tier research', 'Mid-tier university']))]
    for t in ADV_FAM: ms += excl('highest_education', ['No formal', 'Primary'], t, ['Expert'])
    # specialty must fit the field of work
    for d in V['domain']:
        if d == 'Not applicable': continue
        badsp = [sp for sp, ds in SPEC_DOM.items() if d not in ds]
        if badsp: ms.append(('subject_specialty', {'domain': [d]}, bad('subject_specialty', badsp)))
    # military (combat veterans: Korean War, Vietnam 1964-73 -> 65+ in 2020)
    ms += excl('demo_veteran_status', ['Active duty', 'Reserve / guard', 'Veteran'], 'lifex_military_history', ['Never served', 'Military family'])
    ms += excl('lifex_military_history', ['Never served', 'Military family'], 'major_life_events', ['Military service'])
    ms += excl('age_bracket', ['65-74', '75-84', '85+'], 'demo_veteran_status', ['Active duty'])
    ms += excl('demo_employment_status', ['Retired', 'Homemaker', 'Unemployed'], 'demo_veteran_status', ['Active duty'])   # after round 3: also homemaker, unemployed
    ms += excl('age_bracket', ['18-24'], 'lifex_military_history', ['Career veteran', 'Combat veteran'])
    ms += excl('age_bracket', ['25-34', '35-44', '45-54', '55-64'], 'lifex_military_history', ['Combat veteran'])
    # health
    ms += excl('health_chronic_condition', ['None'], 'lifex_health_journey', ['Manages chronic condition'])
    ms += excl('lifex_health_journey', ['No major events'], 'major_life_events', ['Health journey'])
    ms += excl('lstyle_exercise_freq', ['Monthly', 'Rarely', 'Never'], 'health_fitness_level', ['Athlete'])
    ms += excl('lstyle_exercise_freq', ['Daily'], 'health_fitness_level', ['Sedentary'])
    ms += excl('health_chronic_condition', ['Multiple'], 'health_medication_use', ['None'])
    ms += excl('health_attention_condition', ['Diagnosed'], 'health_neurodivergence', ['Neurotypical'])
    ms += excl('health_neurodivergence', ['ADHD'], 'health_attention_condition', ['None'])
    ms += excl('neurotype', ['ADHD'], 'health_attention_condition', ['None'])
    for t in V:
        if t.startswith('sport_') and t not in ('sport_esports', 'sport_darts'): ms += excl('lstyle_exercise_freq', ['Never'], t, ['Play'])
    # loss and relationships
    ms += excl('lifex_relationship_history', ['Widowed'], 'lifex_loss_experience', ['No major loss', 'Lost a grandparent', 'Lost a parent'])
    ms += excl('age_bracket', ['75-84', '85+'], 'lifex_loss_experience', ['No major loss'])
    ms += excl('lifex_loss_experience', ['No major loss'], 'major_life_events', ['Bereavement'])
    ms += excl('demo_marital_status', ['Married'], 'lifex_relationship_history', ['Widowed'])
    # devices, platforms, screen use, driving, food, drink, diet
    ms += excl('lstyle_primary_messenger', ['iMessage'], 'lstyle_device_ecosystem', ['Android/Google', 'Linux'])
    ms += excl('lstyle_device_ecosystem', ['Android/Google', 'Linux'], 'lstyle_browser', ['Safari'])
    ms += excl('linkedin_activity', ['Inactive'], 'lstyle_primary_social', ['LinkedIn'])
    ms += excl('lstyle_screen_time', ['<2 hrs'], 'lstyle_streaming_hours', ['16+'])
    ms += excl('demo_driver_status', ['Daily driver', 'Occasional driver'], 'skill_driving', ['None'])
    ms += excl('demo_driver_status', ['Cannot drive'], 'skill_driving', ['Master', 'Advanced', 'Intermediate'])
    ms += excl('cult_south_korea', ['Native'], 'cuis_korean', ['Avoid'])
    ms += excl('demo_religion_affiliation', ['None', 'Atheist / agnostic'], 'health_dietary_restriction', ['Religious'])
    ms += excl('lstyle_diet_type', ['Vegan', 'Vegetarian'], 'att_veganism', ['Opposed'])
    ms += excl('lstyle_diet_type', ['Vegan'], 'cuis_vegan', ['Avoid'])
    ms += excl('lstyle_diet_type', ['Vegetarian'], 'cuis_vegetarian', ['Avoid'])
    for t, v in [('topic_wine', 'Passionate'), ('topic_craft_beer', 'Passionate'), ('fam_sommelier_knowledge', 'Expert'), ('hob_winemaking', 'Avid'), ('hob_home_brewing', 'Avid')]:
        ms += excl('lstyle_alcohol_use', ['Never'], t, [v])
    # developer and AI tools
    ms += excl('coding_agent_usage_frequency', ['Daily', 'Weekly', 'Monthly or infrequent'], 'coding_agent_failure_tolerance', ['Does not use agents'])
    ms += excl('dev_professional_status', ['Professional developer'], 'skill_coding', ['None', 'Beginner'])
    ms += excl('dev_professional_status', ['Professional developer'], 'tech_savviness', ['Avoidant'])
    ms += excl('dev_professional_status', ['Professional developer'], 'open_source_activity_level', ['No coding activity'])
    ms += excl('open_source_activity_level', ['Maintainer / core contributor', 'Regular contributor'], 'skill_coding', ['None'])
    for t in V:
        if t.startswith('prog_'): ms += excl(t, ['Expert'], 'skill_coding', ['None'])
    for tool, prog in [('tool_python', 'prog_python'), ('tool_sql', 'prog_sql'), ('tool_r', 'prog_r'), ('tool_matlab', 'prog_matlab')]:
        ms += excl(tool, ['Power user', 'Regular'], prog, ['None']); ms += excl(prog, ['Expert', 'Proficient'], tool, ['Never used'])
    # languages: English (Korean value) counts toward the lifetime count; MatrAIx ties the lifetime count to multilingualism
    # (Monolingual -> One), so both are opened; the other languages are capped in dyn_language_masks
    ms += excl('english_proficiency', ['Intermediate (B1-B2)', 'Fluent (C1-C2)'], 'multilingualism', ['Monolingual'])
    ms += excl('english_proficiency', ['Intermediate (B1-B2)', 'Fluent (C1-C2)'], 'lifex_languages_lifetime', ['One'])
    ms += excl('lifex_immigration_generation', ['Native multi-generational'], 'demo_home_language', ['Heritage language'])
    # round 2 (second rule set): Korean value -> MatrAIx value
    ms += excl('values_priority', ['Community'], 'val_community', ['Minor', 'Irrelevant'])
    ms += excl('lstyle_volunteering', ['Daily', 'Weekly'], 'lifex_service_history', ['None'])
    ms += excl('lstyle_gaming_freq', ['Never'], 'sport_esports', ['Play'])
    ms += excl('risk_tolerance', ['Risk-averse'], 'att_risk_taking', ['Enthusiast'])
    ms += excl('risk_tolerance', ['Risk-seeking'], 'att_risk_taking', ['Opposed'])
    ms += excl('lifex_geographic_mobility', ['Never left hometown'], 'lifex_cultural_exposure', ['Lived across cultures', 'Third-culture kid'])
    # round 3 (third rule set): Korean value -> MatrAIx value
    ms += excl('religiosity', ['Devout'], 'trait_spirituality', ['Absent'])
    ms += excl('val_spirituality_faith', ['Core value'], 'trait_spirituality', ['Absent'])
    ms += excl('att_data_privacy', ['Opposed'], 'val_privacy', ['Core value'])
    ms += excl('att_data_privacy', ['Enthusiast'], 'val_privacy', ['Irrelevant'])
    return ms


def dyn_language_masks(V):
    """number of languages at Conversational or better (Korean + English + others drawn so far) <= lifetime count (One/Two/Three)"""
    LIFE = {'One': 1, 'Two': 2, 'Three': 3}
    cap_of = np.array([LIFE.get(v, 99) for v in V['lifex_languages_lifetime']] + [99])
    eng = np.array([v in ('Native', 'Fluent (C1-C2)', 'Intermediate (B1-B2)') for v in V['english_proficiency']] + [False])
    foreign = [t for t in V if t.startswith('lang_') and t not in ('lang_korean', 'lang_english')]
    conv = {t: np.array([v in ('Conversational', 'Fluent', 'Native') for v in V[t]] + [False]) for t in foreign}
    def make(t):
        rows_conv = np.flatnonzero(conv[t][:-1])
        def f(codes, avail):
            if 'lifex_languages_lifetime' not in avail: return None
            cnt = 1 + eng[codes['english_proficiency']].astype(int)
            for x in foreign:
                if x != t and x in avail: cnt = cnt + conv[x][codes[x]]
            full = np.flatnonzero(cnt >= cap_of[codes['lifex_languages_lifetime']])
            m = np.ones((len(V[t]), len(cnt)), np.float32); m[np.ix_(rows_conv, full)] = 0
            return m
        return f
    hl = V['demo_home_language']; not_same = [i for i, v in enumerate(hl) if v != 'Same as primary']
    def home(codes, avail):
        """a home language other than Korean needs a foreign language at Conversational or better (English or one drawn so far)"""
        cnt = eng[codes['english_proficiency']].astype(int)
        for x in foreign:
            if x in avail: cnt = cnt + conv[x][codes[x]]
        m = np.ones((len(hl), len(cnt)), np.float32); m[np.ix_(not_same, np.flatnonzero(cnt == 0))] = 0
        return m
    return [(t, make(t)) for t in foreign] + [('demo_home_language', home)]

