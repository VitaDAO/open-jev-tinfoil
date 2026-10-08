"""A catalog id and a Vita inventory id of one analyte ("hscrp" / "hs_crp") must never make a requested metric vanish from a plan.

Runs the deployed r8 parser pair and plan checker (learned_models/, staged by scripts/stage_learned_models.py, or /opt/learned and
/tinfoil/models/jev-learned in the image; skipped when absent unless REQUIRE_LEARNED_MODELS=1, which CI sets) under the deployed
accept rule (Dockerfile: LEARNED_PARSER_MIN_CONFIDENCE=0.9). Synthetic questions only.
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_HOMES = [ROOT / 'learned_models', Path('/opt/learned'), Path('/tinfoil/models/jev-learned')]
MODELS = next((m for m in MODEL_HOMES if (m / 'verifier_v4').is_dir()), MODEL_HOMES[0])
VITA_METRICS = ['hs_crp', 'fasting_glucose', 'hba1c', 'ldl_cholesterol', 'oxygen_saturation', 'steps']


@pytest.fixture(scope='module')
def agreement():
    if not (MODELS / 'verifier_v4').is_dir():
        msg = 'no learned models in ' + ', '.join(map(str, MODEL_HOMES))
        if os.environ.get('REQUIRE_LEARNED_MODELS') == '1': pytest.fail(msg + ' (REQUIRE_LEARNED_MODELS=1)')
        pytest.skip(msg)
    mp = pytest.MonkeyPatch()
    mp.setenv('LEARNED_PARSER_DIR', f'{MODELS}/jevparse_r8a,{MODELS}/jevparse_r8b')
    mp.setenv('LEARNED_VERIFIER_DIR', f'{MODELS}/verifier_v4')
    mp.setenv('LEARNED_PARSER_MIN_CONFIDENCE', '0.9')
    mp.setenv('LEARNED_VERIFIER_MIN', '0')
    mp.setenv('HF_HUB_OFFLINE', '1')
    import query_selector
    mp.setattr(query_selector, '_LEARNED', None)
    yield query_selector.learned_parser()
    mp.undo()


def ask(agreement, text, metrics=VITA_METRICS):
    return agreement.select({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': [], 'reference_date': '2026-09-26', 'time_zone': 'Europe/Bucharest'},
        'reference_time': '2026-09-25T22:41:32Z', 'available_metrics': metrics, 'available_record_types': ['profile', 'workouts', 'labs', 'calendar'],
        'available_sources': [], 'literature_available': True})


def read(plan):
    return {m for q in plan['queries'] for m in q.get('metrics', [])}


@pytest.mark.parametrize('text,want', [
    ('What are my fasting glucose, HbA1c, and hsCRP?', {'fasting_glucose', 'hba1c', 'hs_crp'}),
    ('What is my hsCRP?', {'hs_crp'}),
    ('What is my hs-CRP?', {'hs_crp'}),
    ('What is my LDL?', {'ldl_cholesterol'}),
    ('What is my SpO2?', {'oxygen_saturation'}),
])
def test_a_requested_analyte_is_read_under_the_inventory_id(agreement, text, want):
    plan = ask(agreement, text)
    assert plan['status'] == 'planned', plan
    assert read(plan) == want


def test_alias_does_not_lower_confidence_of_the_direct_id(agreement):
    # "hs-CRP" links straight to the inventory id; its alias twin is no competing alternative (it stayed tier 1 before the alias existed).
    plan = ask(agreement, 'What is my hs-CRP?')
    assert read(plan) == {'hs_crp'} and plan['diagnostics']['acceptance_tier'] == 'tier1'


@pytest.mark.parametrize('text,metrics,want', [
    ('What is my LDL?', ['ldl', 'steps'], {'ldl'}),
    ('What is my SpO2?', ['spo2', 'steps'], {'spo2'}),
    ('What is my hsCRP?', ['hscrp', 'steps'], {'hscrp'}),
])
def test_either_spelling_of_the_inventory_id_is_read_as_given(agreement, text, metrics, want):
    plan = ask(agreement, text, metrics)
    assert plan['status'] == 'planned' and read(plan) == want


def test_overview_reads_the_inventory_ids_of_its_labs(agreement):
    # The overview bundle lists catalog-side ids (bundles.json: ldl, hscrp); Vita's inventory names them ldl_cholesterol and hs_crp.
    plan = ask(agreement, 'Give me an overview of my health.', [*VITA_METRICS, 'weight', 'apob', 'resting_heart_rate'])
    assert plan['status'] == 'planned', plan
    assert {'hs_crp', 'ldl_cholesterol', 'hba1c', 'fasting_glucose', 'apob', 'weight'} <= read(plan)
    assert not read(plan) & {'hscrp', 'ldl'}


# A metric the question names but the inventory lacks is a handoff, not a plan of the rest (the user would get an answer that silently omits it).
def ask_h(agreement, text, history, metrics=VITA_METRICS):
    return agreement.select({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': history, 'reference_date': '2026-09-26', 'time_zone': 'Europe/Bucharest'},
        'reference_time': '2026-09-25T22:41:32Z', 'available_metrics': metrics, 'available_record_types': ['profile', 'workouts', 'labs', 'calendar'],
        'available_sources': [], 'literature_available': True})


@pytest.mark.parametrize('text', [
    'What are my fasting glucose and ferritin?',
    'Show my HbA1c, ferritin, and steps.',
])
def test_a_named_metric_missing_from_the_inventory_hands_off(agreement, text):
    plan = ask(agreement, text)
    assert plan['status'] == 'handoff' and plan['reason_codes'] == ['requested_metric_unavailable'], plan


def test_the_same_question_plans_when_the_metric_is_available(agreement):
    plan = ask(agreement, 'What are my fasting glucose and ferritin?', [*VITA_METRICS, 'ferritin'])
    assert plan['status'] == 'planned' and read(plan) == {'fasting_glucose', 'ferritin'}


def test_overview_still_plans_when_a_bundle_member_is_absent(agreement):
    # the bundle intersects with the inventory on purpose: no apob, weight or resting_heart_rate here
    plan = ask(agreement, 'Give me an overview of my health.')
    assert plan['status'] == 'planned' and {'hs_crp', 'hba1c', 'fasting_glucose'} <= read(plan), plan


def test_a_metric_named_only_in_history_does_not_hand_off(agreement):
    plan = ask_h(agreement, 'What is my fasting glucose?', ['What is my ferritin?'])
    assert plan['status'] == 'planned' and read(plan) == {'fasting_glucose'}, plan


# A named metric the inventory lacks hands off even when another request item looks like a reason to let it go: the exemptions are scoped to
# what the request actually says.
@pytest.mark.parametrize('text', [
    'What are my TSH and steps?',
    'What are my free T4 and steps?',
    'What was my total testosterone last week and my steps?',
    'What are my ApoB and steps?',
    'What are my HDL and steps?',
])
def test_an_app_analyte_the_request_does_not_name_exempts_nothing(agreement, text):
    # tsh, free_t4, total_testosterone, hdl and apob are listed in some app analyte's instead_of; that exempts them only when that app term is asked for
    plan = ask(agreement, text)
    assert plan['status'] == 'handoff' and not read(plan), plan


def test_an_overview_does_not_swallow_a_named_missing_metric(agreement):
    plan = ask(agreement, 'Give me an overview of my health and my ferritin')
    assert plan['status'] == 'handoff' and plan['reason_codes'] == ['requested_metric_unavailable'], plan
    plan = ask(agreement, 'Give me an overview of my health and my ferritin', [*VITA_METRICS, 'ferritin'])
    assert plan['status'] != 'planned' or 'ferritin' in read(plan), plan      # present: the overview never silently drops it


@pytest.mark.parametrize('text', [
    'What are my absolute neutrophils and neutrophil %?',
    'My ANC and neutrophil %?',
])
def test_a_percent_id_is_not_the_absolute_count_it_was_not_asked_to_stand_in_for(agreement, text):
    # only neutrophils_percent exists: the absolute count is a different measurement (unit variants are not spellings of one analyte)
    plan = ask(agreement, text, ['neutrophils_percent', 'steps'])
    assert plan['status'] == 'handoff' and not read(plan), plan
    both = ask(agreement, 'What are my absolute neutrophils and neutrophil %?', ['neutrophils_percent', 'neutrophils_absolute', 'steps'])
    assert both['status'] == 'planned' and read(both) == {'neutrophils_absolute', 'neutrophils_percent'}, both


def test_the_unit_split_ids_are_not_spellings_of_one_analyte():
    from learned_parser import decode
    assert decode.ANALYTE['neutrophils_percent'] == decode.ANALYTE['neutrophils_absolute']       # the agreement check still treats them as one analyte
    assert decode.SPELLING_ANALYTE.get('neutrophils_percent', 'neutrophils_percent') != decode.SPELLING_ANALYTE.get('neutrophils_absolute', 'neutrophils_absolute')
    assert decode.SPELLING_ANALYTE['hscrp'] == decode.SPELLING_ANALYTE['hs_crp']


# The exemptions that stay: each lets a named-but-absent item through for its own reason.
def test_exempt_profile_twin(agreement):
    # "profile weight" names the weight metric; the inventory has none and the profile holds the field
    plan = ask(agreement, 'my profile weight')
    assert plan['status'] == 'planned' and not read(plan) and plan['queries'][0]['records'] == ['profile'], plan


def test_exempt_narrowed_panel(agreement):
    # a panel narrowed to one member: the other members are no names to be missed
    plan = ask(agreement, 'Lipids: just LDL')
    assert plan['status'] == 'planned' and read(plan) == {'ldl_cholesterol'}, plan


def test_exempt_app_bound_instead_of_with_the_app_term_present(agreement):
    # "bone density" binds to the app's bone_mineral_density; the catalog look-alike bone_mass it replaces (instead_of) is no missing name
    plan = ask(agreement, 'What is my bone density and steps?', [*VITA_METRICS, 'bone_mineral_density'])
    assert plan['status'] == 'planned' and read(plan) == {'bone_mineral_density', 'steps'}, plan


def test_exempt_full_sleep_area_with_absent_members(agreement):
    plan = ask(agreement, 'Give me a full breakdown of my sleep.', ['total_sleep', 'sleep_rem', 'steps'])
    assert plan['status'] == 'planned' and read(plan) == {'total_sleep', 'sleep_rem'}, plan
