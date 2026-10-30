"""Korean persona profile renderer: same interface and output format as MatrAIx
`matraix.persona_dimension_catalog` (section -> [(id, label, value)], "- label: value" lines), with every attribute kept.

Differences from the MatrAIx renderer
1. no value is dropped: no 'none'-like list, no defaultValue comparison, no length budget
2. values are written as they are (custom Korean categories included)
3. an empty (null) value is a generation error and raises
4. same sections and order; 'Health: Mental' (new attributes) goes to "Lifestyle & health"
5. first line of "Personality & values" says what the 5 levels mean for Korean adults
6. label fix: cog_assertiveness -> "Assertiveness (communication style)" (in dimensions_kr.json)
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PATH = str(Path(__file__).resolve().parent / 'dimensions_kr.json')
PRIMARY_LANGUAGE_OUTPUT_INSTRUCTION = 'Default written language: use your primary language for outputs.'
LEVEL_GUIDE = ('Levels are relative to Korean adults. Personality traits: Very low = bottom 6.5%, Low = next 24%, Average = middle 39%, '
               'High = next 24%, Very high = top 6.5%. Basic values (self-direction ... universalism): Very low = bottom 5.5%, Low = next 23%, '
               'Average = middle 43%, High = next 23%, Very high = top 5.5%.')
_SECTIONS = (
    ('Identity', ('Demographic: Core', 'Demographic: Cultural', 'Demographic: Family', 'Demographic: Life Events')),
    ('Career & education', ('Professional: Career', 'Professional: Industry', 'Learning: Academic', 'Learning: Style')),
    ('Language & communication', ('Linguistic: Language', 'Linguistic: Communication')),
    ('Personality & values', ('Personality: Big Five', 'Personality: Character', 'Personality: MBTI', 'Personality: Relationships', 'Values & Motivation', 'Risk & Decision')),
    ('Current interaction state', ('State: Emotional', 'Behavior: Time', 'Behavior: Work')),
    ('Worldview', ('Worldview: Beliefs',)),
    ('Interests', ('Interests: Topics', 'Interests: Hobbies', 'Interests: Media', 'Interests: Culture', 'Interests: Sports', 'Interests: Food')),
    ('Skills & expertise', ('Expertise: Domains', 'Expertise: Skills', 'Skills: Tools', 'Skills: Programming')),
    ('Lifestyle & health', ('Health: Physical', 'Health: Fitness', 'Health: Lifestyle', 'Health: Mental', 'Behavior: Preferences', 'Behavior: Habits')),
    ('Developer & AI', ('Developer:',)),
)
_STATE_IDS = frozenset({'emotional_state', 'intent', 'query_complexity', 'expertise_gap', 'tone_expected', 'trust_level', 'safety_sensitivity',
                        'time_pressure', 'prior_context', 'device_context', 'modality_pref', 'accessibility_needs'})


class RenderError(ValueError):
    pass


@lru_cache(maxsize=4)
def load_dimension_catalog(catalog_path: str = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(catalog_path).read_text(encoding='utf-8'))
    return {'by_id': {str(r['id']): r for r in payload.get('dimensions') or [] if isinstance(r, dict) and r.get('id')}}


def _matches(category: str, matchers) -> bool:
    for m in matchers:
        if m.endswith(':'):
            if category.startswith(m): return True
        elif category == m or category.startswith(f'{m}:'): return True
    return False


def _section_for(dim_id: str, category: str) -> str:
    if dim_id in _STATE_IDS: return 'Current interaction state'
    for heading, matchers in _SECTIONS:
        if _matches(category, matchers): return heading
    return 'Other attributes'


def _text(raw: Any, dim_id: str) -> str:
    if raw is None: raise RenderError(f'empty value for {dim_id} (the generator must write a value, e.g. "Not applicable")')
    if isinstance(raw, (list, tuple)): text = ', '.join(str(x).strip() for x in raw if str(x).strip())
    else: text = str(raw).strip()
    if not text or text.lower() in ('nan', 'null'): raise RenderError(f'empty value for {dim_id}')
    return text


def collect_dimension_items(dimensions: dict[str, Any], *, catalog_path: str = DEFAULT_CATALOG_PATH,
                            dimension_labels: dict[str, str] | None = None, return_ids: bool = False):
    """section -> [(dim_id, label, value)] for every attribute in `dimensions`; with return_ids also the list of ids used"""
    by_id = load_dimension_catalog(catalog_path)['by_id']
    grouped = {h: [] for h, _ in _SECTIONS}; grouped['Other attributes'] = []
    ordered = [i for i in by_id if i in dimensions] + [i for i in dimensions if i not in by_id]
    used = []
    for dim_id in ordered:
        meta = by_id.get(dim_id) or {}
        text = _text(dimensions[dim_id], dim_id)
        label = (dimension_labels or {}).get(dim_id) or meta.get('label') or dim_id.replace('_', ' ')
        grouped[_section_for(dim_id, str(meta.get('category') or ''))].append((dim_id, str(label).strip(), text)); used.append(dim_id)
    grouped = {k: v for k, v in grouped.items() if v}
    return (grouped, used) if return_ids else grouped


def build_dimension_narrative(dimensions: dict[str, Any], *, catalog_path: str = DEFAULT_CATALOG_PATH, max_chars: int | None = None,
                              dimension_labels: dict[str, str] | None = None) -> list[str]:
    grouped = collect_dimension_items(dimensions, catalog_path=catalog_path, dimension_labels=dimension_labels)
    out = []
    for heading in [h for h, _ in _SECTIONS] + ['Other attributes']:
        items = grouped.get(heading)
        if not items: continue
        lines = [f'### {heading}']
        if heading == 'Personality & values': lines.append(LEVEL_GUIDE)
        lines += [f'- {label}: {value}' for _i, label, value in items]
        if heading == 'Language & communication' and dimensions.get('primary_language') is not None: lines.append(PRIMARY_LANGUAGE_OUTPUT_INSTRUCTION)
        out.append('\n'.join(lines))
    return out


def build_template_context_extras(dimensions: dict[str, Any], *, catalog_path: str = DEFAULT_CATALOG_PATH, max_chars: int | None = None,
                                  dimension_labels: dict[str, str] | None = None) -> dict[str, Any]:
    return {'dimension_profile_narrative': build_dimension_narrative(dimensions, catalog_path=catalog_path, dimension_labels=dimension_labels),
            'dimension_catalog_path': catalog_path}


def overlay_labels_from_persona_path(persona_path):
    """same manifest.json label overlay as MatrAIx (kept so templating.py only changes its import)"""
    from matraix.persona_dimension_catalog import overlay_labels_from_persona_path as f
    return f(persona_path)
