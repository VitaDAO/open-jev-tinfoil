"""Research outside the fixed vocabulary plans a minimised public question.

The outgoing question keeps subject, intervention, outcome and catalogue metric
names, and never carries the user's sentence, values, dates or first person.
"""
import pytest

from query_plan import QueryRequest, compile_batch
from tests.test_binder_phrasing import selector  # noqa: F401  (fixture: fixed semantics)

METRICS = ['steps', 'triglycerides', 'ldl', 'apob', 'heart_rate_variability']


def request(text, literature=True):
    return QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': [],
                  'reference_date': '2026-09-23', 'time_zone': 'UTC'},
        'reference_time': '2026-09-23T12:00:00Z', 'available_metrics': METRICS,
        'available_record_types': [], 'available_sources': [], 'literature_available': literature})


def plan(selector, text, literature=True):
    req = request(text, literature)
    return req, selector.select_query(req)


@pytest.mark.parametrize('text,topic', [
    ('What does the research say about creatine and cognitive performance?', 'creatine and cognitive performance'),
    ('What does recent research say about rapamycin and mTOR in aging?', 'rapamycin and mtor in aging'),
    ('Is there evidence that intermittent fasting improves insulin sensitivity?', 'intermittent fasting improves insulin sensitivity'),
    ('What do studies say about omega-3 supplementation and triglycerides?', 'omega-3 supplementation and triglycerides'),
])
def test_general_research_plans_a_public_topic(selector, text, topic):
    req, result = plan(selector, text)
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'] == [{'kind': 'research', 'targets': [], 'interventions': [], 'goal': 'improve', 'topic': topic}]
    read = compile_batch(result, req)['literature_reads'][0]
    assert read['question'].startswith(f'What does published research say about {topic}?')
    assert read['subject_basis'] == 'explicit_subjects_in_current_user_message'
    assert text.lower().rstrip('?') not in read['question']


def test_personal_context_keeps_only_the_metric_name_and_reads_the_value(selector):
    req, result = plan(selector, 'Given my triglycerides of 220, what does research say about omega-3?')
    assert result['status'] == 'planned', result['reason_codes']
    research, health = result['queries']
    assert research['topic'] == 'omega-3 and triglycerides'
    assert health['metrics'] == ['triglycerides'] and health['operation'] == 'latest'
    question = compile_batch(result, req)['literature_reads'][0]['question']
    assert '220' not in question and ' my ' not in f' {question.lower()} '


@pytest.mark.parametrize('text', [
    'What does research say about my creatine dose?',          # first person
    'What does research say about magnesium 400 mg?',          # value and unit
    'What does the research say about creatine since March?',  # date
    'What does research say about it?',                        # unresolved reference
    'What does research say about ' + ' '.join(['creatine'] * 13) + '?',  # too long
])
def test_personal_or_unclear_research_hands_off(selector, text):
    assert plan(selector, text)[1]['status'] == 'handoff'


def test_no_research_without_literature_consent(selector):
    _, result = plan(selector, 'What does the research say about creatine and cognitive performance?', literature=False)
    assert result['status'] == 'handoff' and result['reason_codes'] == ['research_not_available']


def test_fixed_vocabulary_research_is_unchanged(selector):
    _, result = plan(selector, 'What do randomized trials say about exercise and diet for bringing ApoB down?')
    assert result['status'] == 'planned'
    assert result['queries'][0]['targets'] == ['apob'] and 'topic' not in result['queries'][0]
