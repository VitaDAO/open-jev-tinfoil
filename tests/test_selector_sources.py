"""Source qualifiers must survive binding and admitted-inventory checks."""
import pytest

from proposal_selector import ProposalSelector
from query_plan import compile_batch
from query_selector import QuerySelector
from scripts.evaluate_query_plan import request
from trained_proposal_selector import TrainedProposalSelector


@pytest.fixture
def selector(monkeypatch):
    # Hold semantics fixed to isolate source parsing, date binding and the
    # compiler. Real model acceptance runs separately against the frozen cases.
    def select(self, req):
        predicted = dict(task='health', coverage='targeted', purpose='trend', research='none')
        return ProposalSelector.select(self, req,
            prediction=(predicted, {key: .8 for key in predicted}),
            coverage_decision=({'choice': 'the proposed read covers the request', 'confidence': .9}, None))
    monkeypatch.setattr(TrainedProposalSelector, 'select', select)
    return QuerySelector.__new__(QuerySelector)


@pytest.mark.parametrize('text,source', [
    ('Retrieve Garmin steps for February 2026.', 'garmin'),
    ('Show my Oura steps for February 2026.', 'oura'),
    ('Show steps from Garmin for February 2026.', 'garmin'),
])
def test_provider_qualifies_read_subject_without_losing_month(selector, text, source):
    req = request(text, [])
    result = selector.select_query(req)
    assert result['status'] == 'planned'
    read = compile_batch(result, req)['health_reads'][0]
    assert read['source'] == source and read['concepts'] == ['steps']
    assert read['range'] == {'kind': 'between', 'start_at': '2026-02-01', 'end_at': '2026-02-28'}


@pytest.mark.parametrize('text', [
    'Show Garmin steps from Oura in February 2026.',
    'Show Garmin steps and Oura sleep in February 2026.',
    'Show steps in February 2026 excluding Garmin.',
    'Show steps in February 2026 and send them to Garmin.',
    'Show non-Garmin steps in February 2026.',
    'Show all but Garmin steps in February 2026.',
    'Show steps in February 2026, but not Garmin steps.',
    'Show steps not from Garmin in February 2026.',
    'Show steps recorded by sources other than Garmin in February 2026.',
    'Show Garmin steps and unfiltered total sleep in February 2026.',
    'Show Garmin steps and total sleep in February 2026.',
])
def test_conflicting_or_non_subject_provider_mentions_are_not_erased(selector, text):
    assert selector.select_query(request(text, []))['status'] == 'handoff'


def test_provider_adjective_cannot_enable_unavailable_source(selector):
    req = request('Show Garmin steps for February 2026.', [])
    req = req.model_copy(update={'available_sources': ['oura']})
    result = selector.select_query(req)
    assert result['status'] == 'handoff' and result['queries'] == []
