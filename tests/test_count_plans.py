"""Count questions about whole record types are planned as a record summary; every other count stays with the model.

Decoder cases replay recorded neural outputs (tests/fixtures/learned_replay/count_r8a, count_r8b), so the real decoder runs without
encoder weights. Re-record with:  PYTHONPATH=vendor:. python -c "from tests.test_count_plans import record; record()"
"""
import copy
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import query_selector
from query_plan import HealthRead, QueryPlan, QueryRequest, compile_batch, request_identity
from test_p0_rule_fixes import request, selector  # noqa: F401  (fixture)

REF = '2026-09-26'  # a Saturday
FIXTURES = Path(__file__).parent / 'fixtures' / 'learned_replay'
ALL = {'kind': 'all_history'}

# request -> (first day, last day) of the counted window, or ALL
COUNTED = {
    'How many workouts did I do this month?': (date(2026, 9, 1), date(2026, 9, 26)),
    'How many workouts did I do between May 17 and August 17 2026?': (date(2026, 5, 17), date(2026, 8, 17)),
    'How many workouts have I ever recorded?': ALL,
    'How many workouts did I do last week?': (date(2026, 9, 14), date(2026, 9, 20)),
    'How many workouts did I do in August?': (date(2026, 8, 1), date(2026, 8, 31)),
    'How many workouts did I do last month?': (date(2026, 8, 1), date(2026, 8, 31)),
    'How many workout sessions have I logged in total?': ALL,
    'How many workouts did I do in the last 30 days?': (date(2026, 8, 28), date(2026, 9, 26)),
}
# Counts the single summary cannot answer: a filter, a different unit, another record type, a metric or no stated window.
KEPT = [
    'Number of workouts in the last 30 days',     # the count head does not call this a count: the parsers' plain read (low confidence) stands
    'Count my runs in August.',
    'How many days did I work out this month?',
    'How many strength workouts did I do this month?',
    'How many workouts over 30 minutes did I do this month?',
    'How many workouts did I do with my Garmin this month?',
    'In how many workouts did I reach zone 5?',
    'How many workouts did I do?',
    'How many workouts did I do this month compared to last month?',
    'How many days this month did I hit 10,000 steps?',
    'How many HRV readings do I have?',
    'How many lab reports do I have?',
    'How many screenings have I completed this year?',
    'How many nights did I sleep under 6 hours this month?',
]
CASES = [*COUNTED, *KEPT]
PRIOR = 'How many workouts did I do last month?'
FOLLOW_UP = 'How many workouts did I do this month?'


def parser_request(text, history=()):
    from json import load
    inventory = load(open('/Users/alexdobrin/Documents/open-jev-experiments/exp/heldout_inventory.json'))
    return {'state': {'current_request': text, 'recent_user_requests': list(history), 'reference_date': REF, 'time_zone': 'Europe/Bucharest'},
            'available_metrics': inventory['available_metrics'], 'available_record_types': inventory['available_record_types'],
            'literature_available': True}


def record():
    """Record both parsers' neural outputs for CASES (needs the local model directories)."""
    import os
    os.environ.setdefault('MODEL_DIR', '/Users/alexdobrin/Documents/open-jev/model')
    from learned_parser.decode import Parser
    from scripts.learned_replay import record as record_replay
    root = Path(__file__).resolve().parents[1] / 'learned_models'
    for name in ('r8a', 'r8b'):
        requests = [parser_request(text) for text in CASES] + [parser_request(FOLLOW_UP, [PRIOR])]
        record_replay(Parser(root / ('jevparse_' + name)), requests, FIXTURES / ('count_' + name))


def window(query, request_):
    return query_selector.read_window(SimpleNamespace(period=query['period']), request_)


def replayed(name):
    from scripts.learned_replay import load
    parser, cases = load(FIXTURES / ('count_' + name))
    return parser, {(req['state']['current_request'], tuple(req['state']['recent_user_requests'])): req for req, _ in cases}


# ---- the contract ----

def summary(**changes):
    return HealthRead(records=['workouts'], operation='summary', period={'kind': 'calendar', 'period': 'month'}, date_basis='started_at', **changes)


def test_a_record_summary_is_a_plan_the_contract_takes():
    read = summary()
    assert read.operation == 'summary' and read.records == ['workouts'] and read.metrics == []
    for period in ({'kind': 'all_history'}, {'kind': 'relative', 'amount': 30, 'unit': 'days'},
                   {'kind': 'between', 'start_at': '2026-05-17', 'end_at': '2026-08-17'}):
        assert HealthRead(records=['workouts'], operation='summary', period=period, date_basis='started_at').period == period


@pytest.mark.parametrize('changes', [
    {'limit': 5},                                  # a summary has no page
    {'source': 'oura'},
    {'metrics': ['steps']},                        # metric summaries are trends
    {'records': ['profile'], 'profile_fields': ['all'], 'date_basis': 'current_snapshot', 'period': {'kind': 'all_history'}},
    {'records': ['workouts', 'labs']},
])
def test_a_summary_is_refused_outside_one_record_type(changes):
    base = dict(records=['workouts'], operation='summary', period={'kind': 'calendar', 'period': 'month'}, date_basis='started_at')
    with pytest.raises(ValidationError):
        HealthRead(**{**base, **changes})


def test_the_plan_wire_shape_round_trips_and_stays_closed():
    req = request('How many workouts did I do this month?')
    plan = QueryPlan(status='planned', queries=[summary()], time_zone=req.state.time_zone, selector_sha256='1' * 64,
                     adapter_sha256='2' * 64, model_revision='3' * 40, request_sha256=request_identity(req))
    assert QueryPlan.model_validate(plan.model_dump(mode='json')).queries[0].operation == 'summary'
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({**plan.model_dump(mode='json'), 'queries': [{**plan.model_dump(mode='json')['queries'][0], 'aggregation': 'count'}]})


def test_compiled_summary_asks_for_counts_not_record_pages():
    req = request('How many workouts did I do this month?').model_copy(update={'available_record_types': ['workouts']})
    plan = QueryPlan(status='planned', queries=[summary()], time_zone=req.state.time_zone, selector_sha256='1' * 64,
                     adapter_sha256='2' * 64, model_revision='3' * 40, request_sha256=request_identity(req))
    [operation] = compile_batch(plan, req)['health_reads']
    assert operation['record_types'] == ['workouts'] and operation['purpose'] == 'trend' and operation['result_view'] == 'summary'
    assert 'record_detail_limit' not in operation and 'record_detail_dataset' not in operation


# ---- the decoder ----

@pytest.mark.parametrize('name', ['r8a', 'r8b'])
@pytest.mark.parametrize('text,expected', COUNTED.items())
def test_whole_record_counts_are_planned_as_summaries(name, text, expected):
    parser, requests = replayed(name)
    result = parser.select(copy.deepcopy(requests[(text, ())]))
    assert result['status'] == 'planned', result
    [read] = result['queries']
    assert (read['records'], read['metrics'], read['operation'], read.get('source')) == (['workouts'], [], 'summary', None)
    if expected is ALL:
        assert read['period'] == ALL
    else:
        assert window(read, request(text)) == expected


@pytest.mark.parametrize('name', ['r8a', 'r8b'])
@pytest.mark.parametrize('text', KEPT)
def test_other_counts_stay_with_the_model(name, text):
    parser, requests = replayed(name)
    result = parser.select(copy.deepcopy(requests[(text, ())]))
    # Counted wording never becomes a summary; a plain read of the record is the parsers' existing (low confidence, disagreeing) view.
    assert not any(q.get('operation') == 'summary' for q in result['queries'])
    assert result['status'] == 'handoff' or text in ('Count my runs in August.', 'Number of workouts in the last 30 days')


@pytest.mark.parametrize('name', ['r8a', 'r8b'])
def test_a_complete_count_after_another_count_is_planned(name):
    parser, requests = replayed(name)
    # A complete count stands alone, whatever came before it (the decoder re-reads it without history).
    result = parser.select(copy.deepcopy(requests[(FOLLOW_UP, (PRIOR,))]))
    assert result['status'] == 'planned' and result['queries'][0]['operation'] == 'summary'


# ---- the selector ----

class Stub:
    def __init__(self, result): self.result = result
    def select(self, req): return self.result


def plan_with(monkeypatch, selector, queries, text):
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: Stub({'status': 'planned', 'queries': queries}))
    return selector.select_query(request(text, records=['workouts']))


def test_the_selector_serves_a_summary_plan(monkeypatch, selector):
    result = plan_with(monkeypatch, selector, [{'kind': 'health', 'metrics': [], 'records': ['workouts'], 'operation': 'summary',
        'period': {'kind': 'calendar', 'period': 'month'}, 'source': None, 'profile_fields': []}], 'How many workouts did I do this month?')
    assert result['status'] == 'planned'
    [read] = result['queries']
    assert (read['operation'], read['records'], read['date_basis'], read['limit']) == ('summary', ['workouts'], 'started_at', None)


def test_the_selector_hands_off_a_summary_the_inventory_lacks(monkeypatch, selector):
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: Stub({'status': 'planned', 'queries': [{'kind': 'health', 'metrics': [],
        'records': ['workouts'], 'operation': 'summary', 'period': ALL, 'source': None, 'profile_fields': []}]}))
    assert selector.select_query(request('How many workouts have I ever recorded?', records=['labs']))['status'] == 'handoff'


def test_agreement_requires_both_parsers_to_return_the_identical_summary():
    read = {'kind': 'health', 'metrics': [], 'records': ['workouts'], 'operation': 'summary', 'period': ALL, 'source': None, 'profile_fields': []}
    plan = {'status': 'planned', 'reason_codes': [], 'confidence': 0.95, 'queries': [read]}
    latest = {**plan, 'queries': [{**read, 'operation': 'latest'}]}
    assert query_selector._Agreement([Stub(plan), Stub(plan)], 0.9).select({})['status'] == 'planned'
    assert query_selector._Agreement([Stub(plan), Stub(latest)], 0.9).select({})['reason_codes'] == ['learned_models_disagree']
    assert query_selector._Agreement([Stub({**plan, 'confidence': 0.89}), Stub(plan)], 0.9).select({})['reason_codes'] == ['low_joint_confidence']


# ---- the wording a summary can carry (model-free) ----

def spans(text, *parts):
    return [(text.index(part), text.index(part) + len(part)) for part in parts]


@pytest.mark.parametrize('text,mention,window,expected', [
    ('How many workouts did I do this month?', 'workouts', 'this month', ('workouts', 'workouts', 'this month')),
    ('how many workouts have I done in August', 'workouts', 'in August', ('workouts', 'workouts', 'in August')),
    ('How many of my workout sessions did I log between May 17 and August 17 2026?', 'workout sessions', 'between May 17 and August 17 2026',
     ('workouts', 'workout sessions', 'between May 17 and August 17 2026')),
    ('How many workouts have I ever recorded?', 'workouts', None, ('workouts', 'workouts', None)),
    ('What is the total number of workouts I have in total?', 'workouts', None, ('workouts', 'workouts', None)),
    ('Count the workouts I did last week', 'workouts', 'last week', ('workouts', 'workouts', 'last week')),
])
def test_a_pure_count_has_a_template(text, mention, window, expected):
    from learned_parser.decode import count_template
    assert count_template(text, spans(text, mention), spans(text, window) if window else []) == expected


@pytest.mark.parametrize('text,mention,window', [
    ('How many strength workouts did I do this month?', 'workouts', 'this month'),        # a filter word
    ('How many workouts over 45 minutes did I do this month?', 'workouts', 'this month'),
    ('How many workouts did I do with my Garmin this month?', 'workouts', 'this month'),
    ('How many workouts did I do this month compared to last month?', 'workouts', 'this month'),
    ('How many workouts did I not do this month?', 'workouts', 'this month'),
    ('How many workouts did I do? Also my sleep.', 'workouts', None),
    ('How many workouts did I do this month ever?', 'workouts', 'this month'),            # two windows
    ('How many workouts did I do?', 'workouts', None),                                    # no window
    ('How many days did I work out this month?', 'work out', 'this month'),
    ('How many runs did I do in August?', 'runs', 'in August'),                           # a kind of workout, not all of them
    ('Show my workouts this month', 'workouts', 'this month'),                            # not a count
    ('What was the number of steps this month?', 'steps', 'this month'),                  # another record or metric
])
def test_a_count_with_anything_more_has_no_template(text, mention, window):
    from learned_parser.decode import count_template
    assert count_template(text, spans(text, mention), spans(text, window) if window else []) is None
