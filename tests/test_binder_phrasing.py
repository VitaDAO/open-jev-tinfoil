"""Everyday phrasings of supported reads bind instead of handing off.

Semantics are held fixed (as in test_selector_sources) so these cases isolate
the binder: entity, source, date and follow-up handling plus the compiler.
"""
import pytest

from proposal_selector import ProposalSelector
from query_plan import QueryRequest, compile_batch
from query_selector import QuerySelector, enhance
from trained_proposal_selector import TrainedProposalSelector

METRICS = ['steps', 'weight', 'vitamin_d', 'ldl', 'hdl', 'cholesterol', 'triglycerides',
           'heart_rate_variability']


def request(text, history=(), metrics=METRICS):
    return QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': list(history),
                  'reference_date': '2026-09-23', 'time_zone': 'UTC'},
        'reference_time': '2026-09-23T12:00:00Z', 'available_metrics': metrics,
        'available_record_types': ['labs'], 'available_sources': ['oura', 'whoop'],
        'literature_available': False})


@pytest.fixture
def selector(monkeypatch):
    def select(self, req):
        predicted = dict(task='health', coverage='targeted', purpose='trend', research='none')
        return ProposalSelector.select(self, req,
            prediction=(predicted, {key: .8 for key in predicted}),
            coverage_decision=({'choice': 'the proposed read covers the request', 'confidence': .9}, None))
    monkeypatch.setattr(TrainedProposalSelector, 'select', select)
    monkeypatch.setattr(TrainedProposalSelector, 'intent_check',
        lambda self, text: ({'choice': 'recorded_health_read'}, None))
    return QuerySelector.__new__(QuerySelector)


def read_for(selector, text, history=(), metrics=METRICS):
    req = request(text, history, metrics)
    result = selector.select_query(req)
    assert result['status'] == 'planned', result['reason_codes']
    batch = compile_batch(result, req)
    assert len(batch['health_reads']) == 1
    return batch['health_reads'][0]


@pytest.mark.parametrize('text,concept,purpose', [
    ('Show my weight', 'weight', 'trend'),                       # weight is a metric, not the profile field
    ("What's my current weight?", 'weight', 'latest'),           # a current value is the latest one
    ('What was my vitamin D level last time I tested?', 'vitamin_d', 'latest'),
    ("What's my cholestrol?", 'cholesterol', 'trend'),           # one-edit typo of a catalogue word
    ('Show my triglicerides', 'triglycerides', 'trend'),
])
def test_everyday_metric_phrasings_bind(selector, text, concept, purpose):
    read = read_for(selector, text)
    assert read['concepts'] == [concept] and read['purpose'] == purpose


def test_according_to_binds_the_source(selector):
    read = read_for(selector, "What's my HRV according to Oura?")
    assert read['concepts'] == ['heart_rate_variability'] and read['source'] == 'oura'


def test_lab_uploads_are_the_lab_record(selector):
    read = read_for(selector, 'List my lab uploads')
    assert read['record_types'] == ['labs'] and read['concepts'] == []


def test_since_reads_through_the_reference_day(selector):
    read = read_for(selector, 'Plot my weight since January')
    assert read['concepts'] == ['weight']
    assert read['range'] == {'kind': 'between', 'start_at': '2026-01-01', 'end_at': '2026-09-23'}


@pytest.mark.parametrize('followup', ['And my HDL?', 'How about my HDL?'])
def test_subject_followup_keeps_the_prior_operation(selector, followup):
    read = read_for(selector, followup, ['Show my latest LDL cholesterol'])
    assert read['concepts'] == ['hdl'] and read['purpose'] == 'latest'


def test_unbound_weight_still_names_the_profile_field(selector):
    req = request('Show my weight', metrics=['steps'])
    result = selector.select_query(req)
    assert result['status'] == 'handoff'


def test_ordinary_words_and_calendar_periods_are_not_rewritten():
    assert 'testing' in enhance('How was I testing lately?')
    assert 'this week' in enhance('Show my steps for the current week')
    assert 'latest' not in enhance('Show my steps for the current week')


def test_appointment_request_still_hands_off(selector):
    assert selector.select_query(request('Book a dentist appointment next Tuesday'))['status'] == 'handoff'


def test_a_bare_labs_mention_does_not_add_a_lab_read(selector):
    req = request('Sleep only, no labs', metrics=['total_sleep', 'sleep_efficiency'])
    result = selector.select_query(req)
    assert all('labs' not in (query.get('records') or []) for query in result.get('queries') or [])
