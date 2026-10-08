"""An unrelated earlier request must not vote in the learned parser's agreement check, and a follow-up that names nothing still reads
its subject from the earlier request. Runs the deployed r8 parsers (learned_models/, staged by scripts/stage_learned_models.py;
skipped when absent). Synthetic questions only."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'learned_models'
METRICS = ['recovery_score', 'sleep_deep', 'steps', 'total_sleep', 'resting_heart_rate']
RECORDS = ['profile', 'workouts', 'labs', 'calendar']


@pytest.fixture(scope='module')
def parsers():
    if not (MODELS / 'jevparse_r8a').is_dir():
        pytest.skip('no learned models staged in learned_models/')
    mp = pytest.MonkeyPatch()
    mp.setenv('HF_HUB_OFFLINE', '1')
    from learned_parser.decode import Parser
    yield [Parser(MODELS / 'jevparse_r8a'), Parser(MODELS / 'jevparse_r8b')]
    mp.undo()


def ask(parser, text, history=()):
    return parser.select({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': list(history), 'reference_date': '2026-09-26', 'time_zone': 'Europe/Bucharest'},
        'reference_time': '2026-09-25T22:41:32Z', 'available_metrics': METRICS, 'available_record_types': RECORDS, 'available_sources': [], 'literature_available': True})


def read(plan):
    return {x for q in plan['queries'] for x in q.get('metrics', []) + q.get('records', [])}


@pytest.mark.parametrize('history', [[], ['Compare my Whoop and Oura recovery this week.'], ['What was my HRV over the last 7 days?', 'Compare my Whoop and Oura recovery this week.']])
def test_a_recovery_question_before_is_no_second_opinion_on_a_lab_question(parsers, history):
    plan = ask(parsers[0], 'When was my last lab test?', history)
    assert plan['status'] == 'planned' and read(plan) == {'labs'}, plan.get('reason_codes')


def test_a_follow_up_naming_nothing_still_reads_the_earlier_subject(parsers):
    for parser in parsers:
        plan = ask(parser, 'and for Wednesday last week?', ['What was my deep sleep for yesterday?'])
        assert plan['status'] == 'planned' and read(plan) == {'sleep_deep'}, plan.get('reason_codes')
