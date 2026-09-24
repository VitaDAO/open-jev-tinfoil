import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from query_plan import HealthRead, QueryPlan, request_identity
from scripts import evaluate_http_selector as evaluator


@pytest.fixture
def oracle(tmp_path):
    case = {'id': 'synthetic-source', 'request': 'Show my Garmin steps yesterday',
            'history': [], 'expected_queries': None}
    fixture_path = tmp_path / 'synthetic.json'
    fixture_path.write_text(json.dumps([case]))
    fixture = evaluator.load_fixture(fixture_path)
    req = evaluator.case_request(case)
    queries = [HealthRead(metrics=['steps'], source='garmin', period={
        'kind': 'between', 'start_at': '2026-09-22', 'end_at': '2026-09-22'}).model_dump(mode='json')]
    entry = {'fixture': fixture_path.name, 'fixture_sha256': fixture['sha256'],
        'case_id': case['id'], 'original_expected_queries': None,
        'request_sha256': request_identity(req), 'acceptable_queries': queries,
        'reason': 'Synthetic explicit source, metric and interval review.'}
    manifest = {'schema_version': 'reviewed-selector-oracles/v1', 'entries': [entry]}
    path = tmp_path / 'reviewed.json'
    path.write_text(json.dumps(manifest))
    return SimpleNamespace(case=case, fixture=fixture, request=req, queries=queries,
                           entry=entry, manifest=manifest, path=path)


def evaluate(oracle, *, queries=None, override=None, reviewed=True, case=None):
    req = evaluator.case_request(case or oracle.case)
    raw = QueryPlan(status='planned', queries=queries or oracle.queries,
        time_zone=req.state.time_zone, selector_sha256='1'*64,
        adapter_sha256=evaluator.INTENT_SHA256, model_revision=evaluator.MODEL_REVISION,
        request_sha256=request_identity(req)).model_dump(mode='json')
    raw.update(override or {})
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw))) as http:
        return evaluator.evaluate_case(http, 'http://127.0.0.1:1234', {},
            case or oracle.case, '1'*64, oracle.entry if reviewed else None)


def test_reviewed_plan_keeps_strict_failure_and_original_gold(oracle):
    loaded = evaluator.load_reviewed_oracles(oracle.path, [oracle.fixture])
    assert evaluator.reviewed_entry(loaded, oracle.fixture, oracle.case) == oracle.entry
    row = evaluate(oracle)
    assert row['grade'] == 'wrong_plan' and row['expected_queries'] is None
    assert row['acceptance'] == 'reviewed_plan' and row['batch']['health_reads']
    report = {'outcomes': evaluator.counts([row])}
    evaluator.set_acceptance(report, [row], True)
    assert report['outcomes']['wrong_plan'] == 1 and report['strict_passed'] is False
    assert report['reviewed_plan_count'] == 1 and report['passed'] is True
    evaluator.set_acceptance(report, [row], False)
    assert report['passed'] is False


def test_without_flag_strict_result_stays_failed(oracle):
    row = evaluate(oracle, reviewed=False)
    report = {'outcomes': evaluator.counts([row])}
    evaluator.set_acceptance(report, [row], True)
    assert row['grade'] == 'wrong_plan' and 'acceptance' not in row
    assert not report['passed'] and not report['strict_passed'] and report['reviewed_plan_count'] == 0


@pytest.mark.parametrize('change', [
    {'source': 'oura'}, {'metrics': ['total_sleep']},
    {'period': {'kind': 'between', 'start_at': '2026-09-21', 'end_at': '2026-09-21'}},
])
def test_alternate_never_waives_wrong_source_metric_or_window(oracle, change):
    row = evaluate(oracle, queries=[{**oracle.queries[0], **change}])
    assert row['grade'] == 'wrong_plan' and 'acceptance' not in row


@pytest.mark.parametrize('override', [
    {'advisory': False}, {'schema_version': 'wrong'}, {'selector_sha256': '2'*64},
    {'request_sha256': '2'*64}, {'queries': []}, {'unexpected': True},
])
def test_invalid_response_or_binding_never_gets_reviewed_acceptance(oracle, override):
    row = evaluate(oracle, override=override)
    assert row['grade'] == 'invalid' and 'acceptance' not in row


@pytest.mark.parametrize('mode', ['missing', 'exception'])
def test_missing_or_failed_compilation_never_gets_reviewed_acceptance(oracle, monkeypatch, mode):
    def compile_result(*_):
        if mode == 'exception': raise ValueError('synthetic compile failure')
        return None
    monkeypatch.setattr(evaluator, 'compile_batch', compile_result)
    row = evaluate(oracle)
    assert 'acceptance' not in row
    if mode == 'exception': assert row['grade'] == 'invalid' and row['compile_failed']


@pytest.mark.parametrize('change', [
    {'fixture_sha256': '2'*64}, {'fixture': 'unknown.json'}, {'case_id': 'unknown'},
    {'request_sha256': '2'*64}, {'original_expected_queries': []},
    {'acceptable_queries': []}, {'acceptable_queries': [{'kind': 'health'}]},
    {'reason': ''}, {'extra': True},
])
def test_manifest_rejects_stale_unknown_changed_gold_or_malformed_entry(oracle, change):
    oracle.manifest['entries'][0].update(change)
    oracle.path.write_text(json.dumps(oracle.manifest))
    with pytest.raises(ValueError): evaluator.load_reviewed_oracles(oracle.path, [oracle.fixture])


def test_manifest_rejects_duplicate_entries_and_json_keys(oracle):
    oracle.manifest['entries'].append(copy.deepcopy(oracle.entry))
    oracle.path.write_text(json.dumps(oracle.manifest))
    with pytest.raises(ValueError, match='duplicate'): evaluator.load_reviewed_oracles(oracle.path, [oracle.fixture])
    oracle.path.write_text('{"schema_version":"reviewed-selector-oracles/v1","entries":[],"entries":[]}')
    with pytest.raises(ValueError, match='duplicate'): evaluator.load_reviewed_oracles(oracle.path, [oracle.fixture])


def test_full_request_binding_includes_inventory_and_history(oracle):
    changed = {**oracle.case, 'history': ['Different synthetic context'],
               'request_overrides': {'available_sources': ['garmin']}}
    fixture = {**oracle.fixture, 'cases': [changed]}
    with pytest.raises(ValueError, match='request'): evaluator.load_reviewed_oracles(oracle.path, [fixture])
    row = evaluate(oracle, case=changed)
    assert row['grade'] == 'wrong_plan' and 'acceptance' not in row


def test_correct_strict_plan_is_not_relabelled(oracle):
    case = {**oracle.case, 'expected_queries': oracle.queries}
    oracle.entry['original_expected_queries'] = oracle.queries
    row = evaluate(oracle, case=case)
    assert row['grade'] == 'correct_plan' and 'acceptance' not in row


def test_manifest_hash_is_frozen_in_report_and_source_snapshot(oracle):
    report = {'source_sha256': evaluator.source_hashes()}
    loaded = evaluator.freeze_reviewed_oracles(oracle.path, [oracle.fixture], report)
    assert report['reviewed_oracles']['sha256'] == evaluator.digest(oracle.path)
    assert evaluator.source_hashes(loaded) == report['source_sha256']
    oracle.path.write_text(oracle.path.read_text() + '\n')
    assert evaluator.source_hashes(loaded) != report['source_sha256']
