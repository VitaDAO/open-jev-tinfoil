"""Night-contract boundaries; execution and evidence remain Vita's responsibility."""
import pytest

from query_plan import HealthRead, QueryPlan, compile_batch, request_identity
from scripts.evaluate_query_plan import request
from routing import MODEL_REVISION


def night(**changes):
    return HealthRead.model_validate({'metrics': ['sleep_deep'],
        'period': {'kind': 'between', 'start_at': '2026-09-23', 'end_at': '2026-09-23'},
        'date_basis': 'sleep_end_day', **changes})


@pytest.mark.parametrize('metrics', [['sleep_deep'], ['total_sleep', 'sleep_deep'],
    ['total_sleep', 'sleep_efficiency', 'sleep_deep']])
def test_deep_sleep_preserves_night_and_inventory(metrics):
    req = request('Show my deep sleep last night')
    req.available_metrics = metrics
    query = night(metrics=metrics)
    plan = QueryPlan(status='planned', queries=[query], time_zone=req.state.time_zone,
        selector_sha256='1'*64, adapter_sha256='2'*64, model_revision=MODEL_REVISION,
        request_sha256=request_identity(req))
    batch = compile_batch(plan, req)
    assert batch['health_reads'][0]['concepts'] == metrics
    assert query.date_basis == 'sleep_end_day'
    assert query.period['start_at'] == query.period['end_at'] == '2026-09-23'
    # Compilation acquires detail, never claims that unprojected rows answer a night.
    assert batch['health_reads'][0]['result_view'] == 'detail'
    restricted = req.model_copy(update={'available_metrics': ['total_sleep']})
    with pytest.raises(ValueError):
        compile_batch(plan, restricted)


@pytest.mark.parametrize('changes', [
    {'metrics': ['steps']}, {'metrics': ['sleep_deep', 'steps']},
    {'metrics': ['sleep_rem']}, {'metrics': []},
    {'period': {'kind': 'all_history'}},
    {'period': {'kind': 'between', 'start_at': '2026-09-22', 'end_at': '2026-09-23'}},
    {'period': {'kind': 'between', 'start_at': '2026-09-23T00:00:00Z', 'end_at': '2026-09-23T00:00:00Z'}},
])
def test_night_contract_does_not_expand_unrelated_scopes(changes):
    with pytest.raises(ValueError):
        night(**changes)
