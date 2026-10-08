"""The plan checker must not veto a plan because of unrelated earlier requests: when every parser plans the same reads with no history,
the history decided nothing and the checker scores the request alone (tier 2 only; synthetic stubs, no models)."""
from query_selector import _Agreement

PLAN = {'status': 'planned', 'confidence': .5, 'decision_confidence': .95,
        'queries': [{'kind': 'health', 'metrics': ['sleep_score'], 'records': [], 'operation': 'trend', 'period': {'kind': 'calendar', 'period': 'day'}, 'source': None}]}
OTHER = {**PLAN, 'queries': [{**PLAN['queries'][0], 'metrics': ['total_sleep']}]}
HISTORY = ['How is my HRV today?', 'Show my steps this month.']


class Parser:
    def __init__(self, with_history=PLAN, without_history=PLAN):
        self.with_history, self.without_history, self.calls = with_history, without_history, []
    def select(self, req):
        self.calls.append(list(req['state']['recent_user_requests']))
        return self.with_history if req['state']['recent_user_requests'] else self.without_history


class Checker:
    """Sunk by any earlier turn, as the deployed checker is by unrelated ones."""
    def __init__(self): self.seen = []
    def score(self, req, queries):
        self.seen.append(list(req['state']['recent_user_requests']))
        return .02 if req['state']['recent_user_requests'] else .97


def request(history):
    return {'state': {'current_request': 'What was my sleep score last night?', 'recent_user_requests': list(history)}}


def test_unrelated_history_does_not_sink_the_checker():
    parsers, checker = [Parser(), Parser()], Checker()
    got = _Agreement(parsers, .9, checker, 0).select(request(HISTORY))
    assert got['status'] == 'planned' and got['diagnostics']['acceptance_tier'] == 'tier2'
    assert checker.seen == [HISTORY, []]               # first the request as asked, then alone


def test_a_plan_that_needs_the_history_stays_vetoed():
    parsers, checker = [Parser(), Parser(without_history=OTHER)], Checker()
    got = _Agreement(parsers, .9, checker, 0).select(request(HISTORY))
    assert got['status'] == 'handoff' and got['reason_codes'] == ['learned_plan_unverified']
    assert checker.seen == [HISTORY]


def test_a_failing_history_free_score_still_vetoes():
    class Never(Checker):
        def score(self, req, queries): return .1
    got = _Agreement([Parser(), Parser()], .9, Never(), 0).select(request(HISTORY))
    assert got['status'] == 'handoff' and got['reason_codes'] == ['learned_plan_unverified']


def test_no_history_means_no_second_look():
    parsers, checker = [Parser(), Parser()], Checker()
    got = _Agreement(parsers, .9, checker, 0).select(request([]))
    assert got['status'] == 'planned' and checker.seen == [[]]
    assert all(c == [] for p in parsers for c in p.calls)


def test_tier1_never_consults_the_history_free_path():
    hi = {**PLAN, 'confidence': .99}
    parsers, checker = [Parser(hi, hi), Parser(hi, hi)], Checker()
    got = _Agreement(parsers, .9, checker, 0).select(request(HISTORY))
    assert got['diagnostics']['acceptance_tier'] == 'tier1' and checker.seen == []
