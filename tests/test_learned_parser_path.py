"""The learned parser path (OPEN_JEV_PARSER=learned) keeps every contract guard of the rule path."""
import pytest

import query_selector
from test_p0_rule_fixes import request, selector  # noqa: F401  (fixture)


class Stub:
    def __init__(self, result): self.result = result
    def select(self, req): return self.result


def plan_with(monkeypatch, selector, result, text='whats my ldl'):
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: Stub(result))
    return selector.select_query(request(text))


def test_learned_plan_passes_the_contract(monkeypatch, selector):
    result = plan_with(monkeypatch, selector, {'status': 'planned', 'queries': [
        {'kind': 'health', 'metrics': ['ldl'], 'records': [], 'operation': 'latest', 'period': {'kind': 'all_history'}, 'source': None, 'date_basis': 'observed_at'}]})
    assert result['status'] == 'planned' and result['queries'][0]['metrics'] == ['ldl'] and result['diagnostics']['method'] == 'learned_parser'


@pytest.mark.parametrize('suffix,eligible', [('', True), (' in a table', False)])
def test_two_period_plan_keeps_its_presentation_requirement(monkeypatch, selector, suffix, eligible):
    reads = [{'kind': 'health', 'metrics': ['steps'], 'records': [], 'operation': 'trend',
              'period': period, 'source': None, 'date_basis': 'observed_at'}
             for period in [{'kind': 'calendar', 'period': 'week'},
                            {'kind': 'between', 'start_at': '2026-09-14', 'end_at': '2026-09-20'}]]
    result = plan_with(monkeypatch, selector, {'status': 'planned', 'queries': reads},
                       text='Compare my steps this week with last week' + suffix)
    assert result['status'] == 'planned'
    assert result['diagnostics']['direct_answer_eligible'] is eligible


def test_learned_handoff_codes_are_published(monkeypatch, selector):
    result = plan_with(monkeypatch, selector, {'status': 'handoff', 'reason_codes': ['model_judgement'], 'queries': []})
    assert result['status'] == 'handoff' and result['reason_codes'] == ['model_judgement']
    result = plan_with(monkeypatch, selector, {'status': 'handoff', 'reason_codes': ['not_a_published_code'], 'queries': []})
    assert result['reason_codes'] == ['query_contract_unrepresentable']


def test_contract_rejects_an_invalid_learned_plan(monkeypatch, selector):
    # A night basis on a non-night metric is refused by HealthRead, so the request hands off.
    result = plan_with(monkeypatch, selector, {'status': 'planned', 'queries': [
        {'kind': 'health', 'metrics': ['steps'], 'records': [], 'operation': 'trend', 'period': {'kind': 'between', 'start_at': '2026-09-26', 'end_at': '2026-09-26'},
         'source': None, 'date_basis': 'sleep_end_day'}]})
    assert result['status'] == 'handoff'


def test_acute_guard_runs_before_the_learned_parser(monkeypatch, selector):
    result = plan_with(monkeypatch, selector, {'status': 'planned', 'queries': [
        {'kind': 'health', 'metrics': ['apob'], 'records': [], 'operation': 'trend', 'period': {'kind': 'all_history'}, 'source': None}]},
        text='I want to end my life. show my apob trend')
    assert result['status'] == 'handoff' and result['reason_codes'] == ['acute_or_crisis_requires_model']


def test_research_topic_is_minimised(monkeypatch, selector):
    result = plan_with(monkeypatch, selector, {'status': 'planned', 'queries': [{'kind': 'research', 'topic': 'zone 2 and insulin sensitivity since last year'}]},
                       text='any research on zone 2 and insulin sensitivity')
    assert result['status'] == 'handoff'  # never erase an unrepresentable date restriction


def test_agreement_pair_hands_off_on_disagreement_and_low_confidence():
    plan = {'status': 'planned', 'reason_codes': [], 'confidence': 0.9, 'queries': [
        {'kind': 'health', 'metrics': ['ldl'], 'records': [], 'operation': 'latest', 'period': {'kind': 'all_history'}, 'source': None}]}
    other = {**plan, 'queries': [{**plan['queries'][0], 'metrics': ['hdl']}]}
    assert query_selector._Agreement([Stub(plan), Stub(plan)], 0.5).select({})['status'] == 'planned'
    assert query_selector._Agreement([Stub(plan), Stub(other)], 0.5).select({})['reason_codes'] == ['learned_models_disagree']
    assert query_selector._Agreement([Stub(plan), Stub(plan)], 0.95).select({})['reason_codes'] == ['low_joint_confidence']


def test_calendar_reads_keep_the_decoders_date_basis(monkeypatch, selector):
    # Completion wording reads the done basis (spec v1.8 10g.19); the repo must not reset it to the default due basis.
    window = {'kind': 'between', 'start_at': '2026-08-01', 'end_at': '2026-08-31'}
    result = plan_with(monkeypatch, selector, {'status': 'planned', 'queries': [
        {'kind': 'health', 'metrics': [], 'records': ['calendar'], 'operation': 'latest', 'period': window, 'source': None, 'profile_fields': [], 'date_basis': 'last_done_date'},
        {'kind': 'health', 'metrics': [], 'records': ['calendar'], 'operation': 'latest', 'period': window, 'source': None, 'profile_fields': [], 'date_basis': 'next_due_date'}]},
        text='what was on my calendar in august')
    assert result['status'] == 'planned'
    assert [q['date_basis'] for q in result['queries']] == ['last_done_date', 'next_due_date']


class StubChecker:
    def __init__(self, score): self.value = score; self.seen = None
    def score(self, req, queries): self.seen = queries; return self.value


def test_plan_checker_gates_the_agreed_plan():
    plan = {'status': 'planned', 'reason_codes': [], 'confidence': 0.99, 'queries': [
        {'kind': 'health', 'metrics': ['ldl'], 'records': [], 'operation': 'latest', 'period': {'kind': 'all_history'}, 'source': None}]}
    low, high = StubChecker(0.2), StubChecker(0.97)
    assert query_selector._Agreement([Stub(plan), Stub(plan)], 0.82, low, 0.9).select({})['reason_codes'] == ['learned_plan_unverified']
    assert query_selector._Agreement([Stub(plan), Stub(plan)], 0.82, high, 0.9).select({})['queries'] == plan['queries']
    assert high.seen == plan['queries']


def test_tier2_accepts_below_the_cut_off_when_decisions_and_the_checker_clear_it():
    # Accept rule tier 2 (exp/fixes/calibration/ACCEPT.md): below the cut-off a plan is accepted when both models' decision confidence
    # (the confidence of the decisions that reach the plan) and the plan checker clear the same bar.
    plan = {'status': 'planned', 'reason_codes': [], 'confidence': 0.7, 'decision_confidence': 0.95, 'queries': [
        {'kind': 'health', 'metrics': ['heart_rate_variability'], 'records': [], 'operation': 'trend',
         'period': {'kind': 'relative', 'amount': 30, 'unit': 'days'}, 'source': None}]}
    assert query_selector._Agreement([Stub(plan), Stub(plan)], 0.9, StubChecker(0.95)).select({})['queries'] == plan['queries']
    assert query_selector._Agreement([Stub(plan), Stub(plan)], 0.9, StubChecker(0.85)).select({})['reason_codes'] == ['learned_plan_unverified']
    weak, older = {**plan, 'decision_confidence': 0.85}, {k: v for k, v in plan.items() if k != 'decision_confidence'}
    for other in (weak, older):     # one model's decisions below the bar, or a decoder without decision confidence: no tier 2
        checker = StubChecker(0.99)
        assert query_selector._Agreement([Stub(plan), Stub(other)], 0.9, checker).select({})['reason_codes'] == ['low_joint_confidence']
        assert checker.seen is None
    assert query_selector._Agreement([Stub(plan), Stub(plan)], 0.9).select({})['reason_codes'] == ['low_joint_confidence']   # no checker


def test_plan_clearing_the_cut_off_consults_the_checker_only_with_a_veto_bar():
    plan = {'status': 'planned', 'reason_codes': [], 'confidence': 0.95, 'decision_confidence': 0.95, 'queries': [
        {'kind': 'health', 'metrics': ['ldl'], 'records': [], 'operation': 'latest', 'period': {'kind': 'all_history'}, 'source': None}]}
    checker = StubChecker(0.0)
    assert query_selector._Agreement([Stub(plan), Stub(plan)], 0.9, checker).select({})['queries'] == plan['queries'] and checker.seen is None


def test_identity_attests_the_tier2_rule(monkeypatch, tmp_path):
    # With a checker the tier-2 rule is part of learned_parser_sha256 (/health); without one tier 2 never accepts.
    for root, names in ((tmp_path / 'm', ('config.json', 'model.safetensors', 'heads.pt', 'items.json', 'crisis.json', 'calibration.json', 'tokenizer.json')),
                        (tmp_path / 'v', ('config.json', 'model.safetensors', 'verifier_head.pt', 'verifier.json', 'tokenizer.json'))):
        root.mkdir()
        for n in names: (root / n).write_bytes(n.encode())
    fed = []

    class Digest:
        def update(self, b): fed.append(bytes(b))
        def hexdigest(self): return 'digest'
    monkeypatch.setattr(query_selector.hashlib, 'sha256', Digest)
    monkeypatch.setenv('LEARNED_PARSER_DIR', str(tmp_path / 'm'))
    monkeypatch.setenv('LEARNED_PARSER_MIN_CONFIDENCE', '0.9')
    monkeypatch.delenv('LEARNED_VERIFIER_DIR', raising=False)
    query_selector.learned_parser_identity()
    assert not any(b.startswith(b'tier2') for b in fed)
    fed.clear()
    monkeypatch.setenv('LEARNED_VERIFIER_DIR', str(tmp_path / 'v'))
    query_selector.learned_parser_identity()
    assert b'tier2:decision_confidence>=0.9,checker>=0.9' in fed


def test_plan_checker_skips_research_only_plans():
    plan = {'status': 'planned', 'reason_codes': [], 'queries': [{'kind': 'research', 'topic': 'zone-2 training'}]}
    assert query_selector._Agreement([Stub(plan)], 0.0, StubChecker(0.0), 0.9).select({})['queries'] == plan['queries']


def test_food_intake_wording_hands_off_and_burn_wording_reads():
    # Spec v1.8 10g.30: the word lists ship as data (metadata/learned/food_intake.json); a sync that drops them fails here.
    from learned_parser.decode import Parser
    assert Parser._intake('calories consumed yesterday') and Parser._intake('How much protein did I eat this week?')
    assert not any(Parser._intake(t) for t in ('calories burned yesterday', 'active calories today', 'how have my calories been lately'))
