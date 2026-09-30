import copy

import pytest

from query_selector import _Agreement


PLAN = {'status': 'planned', 'confidence': .99, 'decision_confidence': .99,
        'queries': [{'kind': 'health', 'metrics': ['steps'], 'records': [],
                     'operation': 'trend', 'period': {'kind': 'calendar', 'period': 'week'}}]}


class Parser:
    def __init__(self, original=None, normalized=None):
        self.original = original or {**PLAN, 'confidence': .65}
        self.normalized = normalized or PLAN
        self.seen = []

    def select(self, req):
        text = req['state']['current_request']
        self.seen.append(text)
        return self.original if text.endswith('.') else self.normalized


class Checker:
    def score(self, req, queries):
        return .01


def request(text='Compare the periods.'):
    return {'state': {'current_request': text, 'recent_user_requests': ['An earlier request.']}}


def test_normalized_plan_requires_both_original_plans_and_keeps_request():
    req = request(); before = copy.deepcopy(req)
    parsers = [Parser(), Parser()]
    got = _Agreement(parsers, .9, Checker(), 0).select(req)
    assert got['status'] == 'planned'
    assert got['queries'] == PLAN['queries']
    assert got['diagnostics']['sentence_terminator_normalized'] is True
    assert req == before
    assert all(p.seen == ['Compare the periods.', 'Compare the periods'] for p in parsers)


@pytest.mark.parametrize('changed', [
    {'metrics': ['heart_rate']}, {'operation': 'latest'},
    {'period': {'kind': 'all_history'}}, {'source': 'device'},
])
def test_changed_semantic_plan_cannot_be_accepted(changed):
    altered = {**PLAN, 'queries': [{**PLAN['queries'][0], **changed}]}
    got = _Agreement([Parser(normalized=altered), Parser(normalized=altered)], .9, Checker()).select(request())
    assert got['status'] == 'handoff'


def test_low_original_decision_cannot_be_recovered():
    low = {**PLAN, 'confidence': .65, 'decision_confidence': .8}
    parsers = [Parser(original=low), Parser()]
    assert _Agreement(parsers, .9, Checker()).select(request())['status'] == 'handoff'
    assert all(len(p.seen) == 1 for p in parsers)


def test_original_disagreement_is_never_normalized():
    other = {**PLAN, 'queries': [{**PLAN['queries'][0], 'metrics': ['heart_rate']}]}
    parsers = [Parser(), Parser(original=other)]
    assert _Agreement(parsers, .9, Checker()).select(request())['reason_codes'] == ['learned_models_disagree']
    assert all(len(p.seen) == 1 for p in parsers)


@pytest.mark.parametrize('text', ['through 1.10.', 'from 2026-10-01.', 'Show it...', 'from Dr.', 'including e.g.', 'Show it?', 'Show it!'])
def test_dates_abbreviations_and_other_punctuation_are_unchanged(text):
    # Common one-token abbreviations remain a model decision: a full stop can
    # denote an abbreviation. The normalization below deliberately excludes it.
    parsers = [Parser(), Parser()]
    got = _Agreement(parsers, .9, Checker()).select(request(text))
    assert not got.get('diagnostics', {}).get('sentence_terminator_normalized')
    assert all(len(p.seen) == 1 for p in parsers)


def test_explicit_checker_veto_still_applies():
    assert _Agreement([Parser(), Parser()], .9, Checker(), .9).select(request())['status'] == 'handoff'


def test_no_checker_cannot_promote_a_low_confidence_original():
    assert _Agreement([Parser(), Parser()], .9).select(request())['status'] == 'handoff'


def test_one_parser_cannot_promote_a_low_confidence_original():
    assert _Agreement([Parser()], .9, Checker()).select(request())['status'] == 'handoff'


def test_normalized_models_must_still_agree():
    other = {**PLAN, 'queries': [{**PLAN['queries'][0], 'metrics': ['heart_rate']}]}
    got = _Agreement([Parser(), Parser(normalized=other)], .9, Checker()).select(request())
    assert got['status'] == 'handoff'
