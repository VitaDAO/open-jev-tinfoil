"""Exact synthetic selector acceptance over loopback HTTP, never attestation.

Uses the seven frozen v5 suites, plus an optional independent fixture. Each
question is sent once; no retries or oracle text enter the model request.
Existing output files are refused, and completed rows survive a failed run.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]

from proposal_selector import SOURCES as PROPOSAL_SOURCES
from query_plan import QueryRequest, QueryPlan, compile_batch
from query_selector import identity, INTENT_SHA256
from routing import MODEL_REVISION, ADAPTER_SHA256
from scripts.evaluate_query_plan import request

FIXTURES = (
    'expanded-48.json', 'regression-59-v2.json', 'remaining-19-oracle.json',
    'holdout-24.json', 'additional-160-v2.json', 'future-boundaries-40.json',
    'review-profile-16.json',
)
GRADES = ('correct_plan', 'correct_handoff', 'missed_plan', 'wrong_plan', 'invalid')
SOURCES = tuple(sorted(set(PROPOSAL_SOURCES) | {
    'server.py', 'routing.py', 'trained_proposal_selector.py', 'query_plan.py',
    'query_selector.py', 'query_execution.py', 'entity_candidates.py',
    'adapters/vita-intent-v1.json', 'adapters/vita-selector-experimental-v1.json',
    'adapters/vita-read-intent-v2.json', 'adapters/vita-read-intent-v3.json',
    'scripts/evaluate_query_plan.py', 'scripts/evaluate_http_selector.py',
} | {str(p.relative_to(ROOT)) for p in (ROOT / 'compat').glob('*.py')}
  | {str(p.relative_to(ROOT)) for p in (ROOT / 'vendor').rglob('*.py')}))


def loopback_url(value):
    try:
        parsed = urlparse(value)
        valid = (parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'
                 and parsed.port is not None and 1 <= parsed.port <= 65535
                 and parsed.username is None and parsed.password is None
                 and parsed.path in ('', '/') and not parsed.query and not parsed.fragment
                 and parsed.netloc == f'127.0.0.1:{parsed.port}')
    except ValueError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError('Requires http://127.0.0.1:<explicit-port>')
    return value.rstrip('/')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def case_request(case):
    body = request(case['request'], case.get('history', [])).model_dump(mode='json')
    overrides = case.get('request_overrides', {})
    if not isinstance(overrides, dict):
        raise ValueError('invalid_fixture_overrides')
    for key, value in overrides.items():
        if key == 'state':
            if not isinstance(value, dict):
                raise ValueError('invalid_fixture_state')
            body['state'] = {**body['state'], **value}
        else:
            body[key] = value
    return QueryRequest.model_validate(body)


def load_fixture(path):
    raw = path.read_bytes()
    cases = json.loads(raw)
    if not isinstance(cases, list) or not cases:
        raise ValueError('fixture_must_be_nonempty_list')
    ids = set()
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get('id'), str)
                or not case['id'] or case['id'] in ids or 'expected_queries' not in case):
            raise ValueError('invalid_fixture_case')
        expected = case['expected_queries']
        if expected is not None and (not isinstance(expected, list) or not expected):
            raise ValueError('invalid_fixture_oracle')
        ids.add(case['id'])
        case_request(case)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'case_count': len(cases), 'cases': cases}


def check_health(http, base_url, expected):
    response = http.get(base_url + '/health')
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('health_identity_mismatch')
    # Retain only reviewed public identity fields, not arbitrary server output.
    return {key: data[key] for key in expected}


def evaluate_case(http, base_url, headers, case, selector_sha256):
    req = case_request(case)
    row = {'id': case['id'], 'request': req.model_dump(mode='json'),
           'expected_queries': case['expected_queries'], 'grade': 'invalid'}
    started = time.perf_counter()
    try:
        response = http.post(base_url + '/v1/select', headers=headers,
                             json=req.model_dump(mode='json'))
        row['http_status'] = response.status_code
        response.raise_for_status()
        raw = response.json()
        json.dumps(raw, allow_nan=False)
        if (not isinstance(raw, dict) or raw.get('schema_version') != 'vita-query-plan/v2'
                or raw.get('advisory') is not True):
            raise ValueError('invalid_wire_contract')
        result = QueryPlan.model_validate(raw)
        if (result.selector_sha256 != selector_sha256 or result.adapter_sha256 != INTENT_SHA256
                or result.model_revision != MODEL_REVISION):
            raise ValueError('response_identity_mismatch')
        row['result'] = result.model_dump(mode='json')
        row['batch'] = compile_batch(result, req)
        actual = [q.model_dump(mode='json') for q in result.queries] if result.status == 'planned' else None
        expected = case['expected_queries']
        row['grade'] = ('correct_handoff' if actual is None and expected is None else
                        'missed_plan' if actual is None else
                        'correct_plan' if actual == expected else 'wrong_plan')
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        # HTTP exception strings may include request details. Never save them,
        # response text, headers or the environment's service key.
        row['error_type'] = type(exc).__name__
        row['transport_failed'] = isinstance(exc, httpx.TransportError)
        row['compile_failed'] = 'result' in row and 'batch' not in row
    finally:
        row['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
    return row


def counts(rows):
    return {grade: sum(row['grade'] == grade for row in rows) for grade in GRADES}


def run(args, report):
    paths = [ROOT / 'evidence/selector-v5' / name for name in FIXTURES]
    paths.extend(path.resolve() for path in args.additional_fixture)
    if len(set(paths)) != len(paths):
        raise ValueError('duplicate_fixture_path')
    frozen = [load_fixture(path) for path in paths]
    report['source_sha256'] = source_hashes()
    report['selector_sha256'] = identity()
    report['adapter_sha256'] = INTENT_SHA256
    report['model_revision'] = MODEL_REVISION
    report['expected_case_entries'] = sum(f['case_count'] for f in frozen)
    report['fixtures'] = [{k: v for k, v in f.items() if k != 'cases'} for f in frozen]
    token = os.environ.get('OPEN_JEV_API_KEY', '')
    if len(token) < 32:
        raise ValueError('service_key_required')
    expected_health = {'status': 'ready', 'device': 'cpu', 'model_revision': MODEL_REVISION,
        'adapter_sha256': ADAPTER_SHA256, 'selector_enabled': True,
        'selector_schema': 'vita-selector/v2', 'selector_sha256': report['selector_sha256'],
        'selector_adapter_sha256': INTENT_SHA256}
    try:
        # Disallow proxy environment variables and redirects to keep the key
        # and every synthetic request on this explicit loopback endpoint.
        with httpx.Client(timeout=30, trust_env=False, follow_redirects=False) as http:
            report['health_before'] = check_health(http, args.base_url, expected_health)
            for fixture in frozen:
                suite = {'fixture': fixture['path'], 'rows': []}
                report['suites'].append(suite)
                for case in fixture['cases']:
                    row = evaluate_case(http, args.base_url, {'Authorization': 'Bearer ' + token},
                                        case, report['selector_sha256'])
                    suite['rows'].append(row)
                    if row.get('transport_failed'):
                        raise RuntimeError('transport_failed_no_retry')
                suite['outcomes'] = counts(suite['rows'])
                print(json.dumps({'fixture': Path(fixture['path']).name,
                                  'outcomes': suite['outcomes']}), flush=True)
            report['health_after'] = check_health(http, args.base_url, expected_health)
    finally:
        report['source_unchanged'] = source_hashes() == report['source_sha256']
        report['selector_identity_unchanged'] = identity() == report['selector_sha256']
        report['fixtures_unchanged'] = all(digest(Path(f['path'])) == f['sha256'] for f in frozen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, type=loopback_url)
    parser.add_argument('--additional-fixture', type=Path, action='append', default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = {'scope': 'Synthetic real-model loopback HTTP; NOT attested, deployed or final Vita answers.',
              'independence': 'Case entries overlap: remaining-19 is a subset of additional-160; reused suites are regression evidence.',
              'base_url': args.base_url, 'suites': [], 'passed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve this attempt before contacting the model. Never replace a previous
    # report, including one recording failures.
    with args.output.open('x') as output:
        try:
            run(args, report)
        except (Exception, KeyboardInterrupt) as exc:
            report['fatal_error_type'] = type(exc).__name__
        finally:
            rows = [row for suite in report['suites'] for row in suite['rows']]
            for suite in report['suites']:
                suite['outcomes'] = counts(suite['rows'])
            report['outcomes'] = counts(rows)
            report['completed_case_entries'] = len(rows)
            report['compile_failures'] = sum(row.get('compile_failed', False) for row in rows)
            times = sorted(row['elapsed_ms'] for row in rows)
            if times:
                report['http_latency_ms'] = {'median': statistics.median(times),
                    'p95': times[math.ceil(len(times) * .95) - 1]}
            report['passed'] = (not report.get('fatal_error_type')
                and len(rows) == report.get('expected_case_entries')
                and all(report.get(key) is True for key in ('source_unchanged',
                    'selector_identity_unchanged', 'fixtures_unchanged'))
                and not any(report['outcomes'][grade] for grade in ('missed_plan', 'wrong_plan', 'invalid')))
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
    print(json.dumps({key: report[key] for key in ('passed', 'completed_case_entries', 'outcomes')}), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
