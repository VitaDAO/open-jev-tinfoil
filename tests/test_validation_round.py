"""Regression boundaries for the local correctness round; no model download."""
import pytest
import query_selector as qs
from learned_parser.decode import Parser
from learned_parser.verify import fits
from test_p0_rule_fixes import request, selector  # noqa: F401


class Tokens:
    def __call__(self, *args, **kwargs):
        return {'input_ids': list(range(300))}


def test_old_checker_cannot_score_a_truncated_request():
    assert not fits(Tokens(), ('a long request', [], [], '2026-09-26'), 2)


def test_long_request_hands_off_before_model_or_prefetch():
    parser = Parser.__new__(Parser)
    parser.tok = Tokens()
    parser._prefetch = lambda req: pytest.fail('must reject before inference')
    result = parser.select({'state': {'current_request': 'long', 'recent_user_requests': [], 'reference_date': '2026-09-26'}})
    assert result['reason_codes'] == ['input_token_budget_exceeded']


@pytest.mark.parametrize('topic', ['sleep in adults over 65', 'sleep since 2024', ' '.join(['sleep'] * 13)])
def test_public_topic_never_silently_drops_constraints(topic):
    assert qs.public_topic(topic) == ''


@pytest.mark.parametrize('topic', ['5-htp and sleep', '5:2 fasting', '4-7-8 breathing'])
def test_unsupported_numbered_names_hand_off_instead_of_changing_subject(topic):
    # The current public contract cannot represent these names. Preserve safety
    # by handing off the entire request, not by searching a different subject.
    assert qs.public_topic(topic) == ''


def test_acute_reason_survives_pair_disagreement():
    class Stub:
        def __init__(self, result): self.result = result
        def select(self, req): return self.result
    plan = {'status': 'planned', 'queries': [], 'confidence': .99}
    acute = {'status': 'handoff', 'queries': [], 'reason_codes': ['acute_or_crisis_requires_model']}
    for a, b in [(plan, acute), (acute, plan)]:
        assert qs._Agreement([Stub(a), Stub(b)], .9).select({})['reason_codes'] == acute['reason_codes']


@pytest.mark.parametrize('text,reason', [
    ('Show latest 201 lab reports', 'record_limit_out_of_range'),
    ('Show lab reports by upload date', 'upload_order_not_available'),
    ('Show my full profile but hide allergies', 'unbound_request_constraint'),
    ('How many steps Monday morning between 8 and 10?', 'clock_time_not_supported'),
    ('Find studies on sleep in adults over 65', 'unbound_research_topic'),
])
def test_constraints_survive_even_a_confident_neural_plan(monkeypatch, selector, text, reason):
    class Stub:
        def select(self, req):
            return {'status': 'planned', 'queries': [{'kind': 'health', 'records': ['labs'], 'metrics': [],
                'operation': 'latest', 'period': {'kind': 'all_history'}, 'source': None}]}
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(qs, 'learned_parser', lambda: Stub())
    result = selector.select_query(request(text))
    assert result['status'] == 'handoff'
    assert result['reason_codes'] == [reason]


def test_explicit_record_count_is_preserved(monkeypatch, selector):
    class Stub:
        def select(self, req):
            return {'status': 'planned', 'queries': [{'kind': 'health', 'records': ['labs'], 'metrics': [],
                'operation': 'latest', 'period': {'kind': 'all_history'}, 'source': None}]}
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(qs, 'learned_parser', lambda: Stub())
    result = selector.select_query(request('Show newest 5 lab reports'))
    assert result['queries'][0]['limit'] == 5


@pytest.mark.parametrize('text', ['Between March and May 2025, what was my SpO2?',
                                 'Between March 2025 and May 2025, what was my SpO2?'])
def test_month_range_normalization_preserves_explicit_year(text):
    req = {'state': {'current_request': text, 'recent_user_requests': [], 'reference_date': '2026-09-26'}}
    _, normalized, _ = Parser._normalized(req)
    assert '2026-03' not in normalized and '2026-05' not in normalized


@pytest.mark.parametrize('text', ['Steps between May and June 15', 'Show my appointments between October and November'])
def test_range_dates_are_resolved_after_record_type_is_known(text):
    req = {'state': {'current_request': text, 'recent_user_requests': [], 'reference_date': '2026-09-26'}}
    assert Parser._normalized(req)[1] == text


def test_actual_encoder_input_cannot_be_truncated():
    from transformers import AutoTokenizer
    from pathlib import Path
    from learned_parser.decode import InputTokenBudgetExceeded
    parser = Parser.__new__(Parser)
    parser.tok = AutoTokenizer.from_pretrained(Path(__file__).parent / 'fixtures/learned_replay/jevparse_r13a/tokenizer', local_files_only=True)
    with pytest.raises(InputTokenBudgetExceeded):
        parser._batch('walking ' * 200)

@pytest.mark.parametrize('text', ['Show lab reports uploaded in September 2025', 'Show my very first LDL result', 'Show latest 3 ApoB readings'])
def test_unsupported_order_and_quantity_are_not_silently_lost(monkeypatch, selector, text):
    class Stub:
        def select(self, req):
            return {'status': 'planned', 'queries': [{'kind':'health','records':[],'metrics':['ldl'],'operation':'latest','period':{'kind':'all_history'},'source':None}]}
    monkeypatch.setenv('OPEN_JEV_PARSER','learned')
    monkeypatch.setattr(qs,'learned_parser',lambda:Stub())
    assert selector.select_query(request(text))['status']=='handoff'


def test_record_quantity_survives_window_only_followup(monkeypatch, selector):
    class Stub:
        def select(self, req):
            return {'status':'planned','queries':[{'kind':'health','records':['labs'],'metrics':[],'operation':'latest','period':{'kind':'all_history'},'source':None}]}
    monkeypatch.setenv('OPEN_JEV_PARSER','learned')
    monkeypatch.setattr(qs,'learned_parser',lambda:Stub())
    result=selector.select_query(request('What about last week?', ['Show latest 4 lab reports in 2025']))
    assert result['status']=='planned'
    assert result['queries'][0]['limit']==4


def test_open_ended_iso_date_is_not_a_single_day():
    from learned_parser.decode import open_start
    assert open_start('Show LDL from ', '2026-02-01', ' onward.')=='2026-02-01'
    assert open_start('Show LDL from ', '2026-02-01', '.') is None
