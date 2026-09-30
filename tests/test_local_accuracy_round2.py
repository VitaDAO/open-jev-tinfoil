"""Readable research requests must retain personal restrictions and qualifiers."""
import pytest
import query_selector as qs
from test_p0_rule_fixes import request, selector

@pytest.mark.parametrize('text,topic', [
    ('Find research on sleep for me', 'sleep'),
    ('Find papers about our sleep', 'sleep'),
    ('What does the science say about the ideal bedroom temperature for sleep?', 'bedroom temperature for sleep'),
    ('Find research on benefits and risks of sauna use', 'benefits of sauna use'),
])
def test_research_model_cannot_omit_request_constraints(monkeypatch, selector, text, topic):
    class Stub:
        def select(self, req):
            return {'status': 'planned', 'queries': [{'kind': 'research', 'topic': topic}]}
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(qs, 'learned_parser', lambda: Stub())
    result = selector.select_query(request(text))
    assert result['status'] == 'handoff'
    assert result['reason_codes'] == ['unbound_research_topic']

@pytest.mark.parametrize('text,topic', [
    ('Find research on the ideal bedroom temperature for sleep', 'ideal bedroom temperature for sleep'),
    ('Find studies on zone 2 training and sleep', 'zone-2 training and sleep'),
    ('Find papers on vitamin A and vision', 'vitamin a and vision'),
    ('literature on thyroid function and heart rate variability pls', 'thyroid function and heart rate variability'),
    ('published research on meditation and blood pressure ty', 'meditation and blood pressure'),
    ('Studies on the effect of heat acclimation on plasma volume ⌚', 'heat acclimation on plasma volume'),
])
def test_complete_public_subject_still_plans(monkeypatch, selector, text, topic):
    class Stub:
        def select(self, req):
            return {'status': 'planned', 'queries': [{'kind': 'research', 'topic': topic}]}
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(qs, 'learned_parser', lambda: Stub())
    assert selector.select_query(request(text))['status'] == 'planned'


def test_reading_about_research_is_plain_context_for_a_health_read(monkeypatch, selector):
    class Stub:
        def select(self, req):
            return {'status': 'planned', 'queries': [{'kind': 'health', 'records': [], 'metrics': ['steps'],
                'operation': 'trend', 'period': {'kind': 'calendar', 'period': 'day'}, 'source': None}]}
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(qs, 'learned_parser', lambda: Stub())
    result = selector.select_query(request("I read studies on running cadence over lunch. Show my steps today."))
    assert result['status'] == 'planned'
