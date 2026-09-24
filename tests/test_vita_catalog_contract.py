"""Full catalog compatibility against the explicitly selected actual Vita checkout."""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('VITA_SOURCE_ROOT'),
                                reason='requires explicit native Vita checkout')


def test_full_catalog_read_matches_current_native_schema_and_keeps_live_limits():
    import copy
    import vita_agent
    from vita_agent.health.generated_metrics import HEALTH_METRICS
    from vita_agent.health.metric_registry import canonical_authorized_metric_ids
    from vita_agent.kernel.source_batch_contracts import SourceOperationKind
    from vita_agent.kernel.source_batch_tool import _tool_schema
    from examples.vita_native import native_batch
    from query_plan import QueryRequest, QueryPlan, HealthRead, request_identity
    from routing import MODEL_REVISION

    assert Path(vita_agent.__file__).resolve().is_relative_to(
        Path(os.environ['VITA_SOURCE_ROOT']).resolve())
    metrics = list(HEALTH_METRICS)
    assert len(metrics) > 48  # The old six-per-read splitter exhausted eight reads.
    request = QueryRequest.model_validate({
        'schema_version': 'vita-selector/v2',
        'state': {'current_request': 'Summarize my health trends this week',
                  'reference_date': '2026-09-24', 'time_zone': 'Europe/Bucharest'},
        'reference_time': '2026-09-24T10:00:00Z', 'available_metrics': metrics,
        'available_record_types': [], 'available_sources': [], 'literature_available': False})
    plan = QueryPlan(status='planned', queries=[HealthRead(metrics=metrics,
        period={'kind': 'calendar', 'period': 'week'})], time_zone=request.state.time_zone,
        selector_sha256='1' * 64, adapter_sha256='2' * 64, model_revision=MODEL_REVISION,
        request_sha256=request_identity(request))
    schema = _tool_schema(enabled_kinds=frozenset({SourceOperationKind.HEALTH_READ}),
                          operation_budget=8, authorized_metrics=metrics)
    batch = native_batch(plan, request, schema)
    assert len(batch['health_reads']) == 1
    assert batch['required_operation_ids'] == [1]
    read = batch['health_reads'][0]
    assert read['concepts'] == list(canonical_authorized_metric_ids(metrics))
    assert read['range'] == {'kind': 'between', 'start_at': '2026-09-21T00:00:00+03:00',
                             'end_at': '2026-09-24T13:00:00+03:00'}
    # The adapter must honor a stricter live tool schema rather than bypass it.
    narrower = copy.deepcopy(schema)
    narrower['properties']['health_reads']['items']['properties']['concepts']['maxItems'] = 6
    with pytest.raises(ValueError, match='native_contract_mismatch'):
        native_batch(plan, request, narrower)
    restricted = _tool_schema(enabled_kinds=frozenset({SourceOperationKind.HEALTH_READ}),
                              operation_budget=8, authorized_metrics=metrics[:10])
    with pytest.raises(ValueError, match='native_contract_mismatch'):
        native_batch(plan, request, restricted)
