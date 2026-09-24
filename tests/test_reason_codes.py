"""The published reason-code list is complete, and only listed codes leave the selector (spec §3.15)."""
import json
from pathlib import Path

import pytest

from proposal_selector import ProposalSelector
from query_plan import QueryRequest
from query_selector import QuerySelector, REASON_CODES
from trained_proposal_selector import TrainedProposalSelector

ROOT = Path(__file__).resolve().parents[1]


def test_published_list_matches_the_source():
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    from reason_codes import scan, document
    assert (ROOT / 'metadata/reason-codes.v1.json').read_text() == document(scan()), \
        'Run scripts/reason_codes.py and review the new or removed codes'


def test_acute_code_is_published():
    assert 'acute_or_crisis_requires_model' in REASON_CODES


def test_unlisted_codes_are_replaced(monkeypatch):
    monkeypatch.setattr(QuerySelector, 'health', lambda self, *a: (_ for _ in ()).throw(ValueError('some_new_internal_code')))
    selector = QuerySelector.__new__(QuerySelector)
    req = QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
        'state': {'current_request': 'Show my steps', 'recent_user_requests': [],
                  'reference_date': '2026-09-23', 'time_zone': 'UTC'},
        'reference_time': '2026-09-23T12:00:00Z', 'available_metrics': ['steps'],
        'available_record_types': [], 'available_sources': [], 'literature_available': False})
    result = selector.select_query(req)
    assert result['status'] == 'handoff' and result['reason_codes'] == ['query_contract_unrepresentable']
