from copy import deepcopy
import pytest
from scripts.learned_replay import same_output

BASE = {'status': 'planned', 'reason_codes': [], 'confidence': 0.8363801836967468,
    'decision_confidence': 0.8363801836967468,
    'queries': [{'metrics': ['steps'], 'source': 'oura', 'period': {'start_at': '2026-09-16', 'end_at': '2026-09-16'}}]}

def test_only_small_cpu_confidence_rounding_is_permitted():
    other = deepcopy(BASE)
    other['confidence'] = other['decision_confidence'] = 0.8363802433013916
    assert same_output(other, BASE)
    other['confidence'] += 0.001
    assert not same_output(other, BASE)

@pytest.mark.parametrize('field,value', [('status', 'handoff'), ('reason_codes', ['unbound_source']), ('queries', []), ('confidence', None)])
def test_replay_contract_changes_still_fail(field, value):
    other = deepcopy(BASE)
    if value is None:
        other.pop(field)
    else:
        other[field] = value
    assert not same_output(other, BASE)

@pytest.mark.parametrize('field,value', [('metrics', ['ldl']), ('source', 'whoop'), ('period', {'start_at': '2026-09-15', 'end_at': '2026-09-16'})])
def test_replay_read_constraints_stay_exact(field, value):
    other = deepcopy(BASE)
    other['queries'][0][field] = value
    assert not same_output(other, BASE)
