"""A catalog id and a Vita inventory id of one analyte ("hscrp" / "hs_crp") must never make a requested metric vanish from a plan.

Runs the deployed r8 parser pair and plan checker (learned_models/, staged by scripts/stage_learned_models.py; skipped when absent) under
the deployed accept rule (Dockerfile: LEARNED_PARSER_MIN_CONFIDENCE=0.9). Synthetic questions only.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'learned_models'
VITA_METRICS = ['hs_crp', 'fasting_glucose', 'hba1c', 'ldl_cholesterol', 'oxygen_saturation', 'steps']


@pytest.fixture(scope='module')
def agreement():
    if not (MODELS / 'verifier_v4').is_dir():
        pytest.skip('no learned models staged in learned_models/')
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
