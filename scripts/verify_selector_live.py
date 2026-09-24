"""Synthetic acceptance through the pinned, attested Vita client.

Only the client's fixed enclave/repository are used. No plaintext fallback,
application retries, real user records or secret-bearing logs are permitted.
Run only after the intended release is deployed; an existing report is refused.
"""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]

from examples.vita_client import VitaClient, HOST, REPO
from query_plan import HealthRead
from query_selector import identity, INTENT_SHA256
from routing import MODEL_REVISION, ADAPTER_SHA256
from scripts.evaluate_http_selector import FIXTURES, counts, digest, evaluate_case, load_fixture, source_hashes
from scripts.evaluate_query_plan import request

URL = 'https://' + HOST


def sha256_pin(value):
    if not re.fullmatch(r'[0-9a-f]{64}', value):
        raise argparse.ArgumentTypeError('A reviewed lowercase SHA256 pin is required')
    return value


def snapshot_sources():
    sources = source_hashes()
    for name in ('examples/vita_client.py', 'scripts/verify_selector_live.py'):
        sources[name] = digest(ROOT / name)
    return sources


def verification(client, release_digest):
    document = client.verifier.get_verification_document()
    if (document is None or document.security_verified is not True
            or document.release_digest != release_digest
            or document.config_repo != REPO or document.enclave_host != HOST):
        raise ValueError('attestation_pin_mismatch')
    # These are public attestation identities. Omit arbitrary error strings and
    # transport state, especially anything containing application credentials.
    return {field: getattr(document, field, None) for field in (
        'security_verified', 'config_repo', 'enclave_host', 'release_digest',
        'release_tag', 'verified_at', 'tls_public_key', 'code_fingerprint')}


def check_health(client, expected):
    response = client.http.get(URL + '/health', timeout=30)
    response.raise_for_status()
    data = response.json()
    if (not isinstance(data, dict) or data.get('selector_enabled') is not True
            or any(data.get(key) != value for key, value in expected.items())):
        raise ValueError('health_identity_mismatch')
    return {key: data[key] for key in expected}


def negative_checks(client, rows):
    body = request('Show my steps yesterday').model_dump(mode='json')
    wrong_key = 'synthetic-invalid-selector-acceptance-key'
    if wrong_key == client.token:
        wrong_key += '-different'
    for name, headers in (
        ('missing_auth', {}),
        ('wrong_auth', {'Authorization': 'Bearer ' + wrong_key}),
    ):
        response = client.http.post(URL + '/v1/select', json=body, headers=headers, timeout=30)
        rows.append({'check': name, 'http_status': response.status_code,
                     'passed': response.status_code == 401})
    headers = {'Authorization': 'Bearer ' + client.token}
    sentinel = 'FICTIONAL_SELECTOR_VALIDATION_SENTINEL'
    for name, change in (
        ('old_schema', {'schema_version': 'vita-selector/v1'}),
        ('extra_field_redaction', {'private_records': sentinel}),
    ):
        response = client.http.post(URL + '/v1/select', json={**body, **change},
                                    headers=headers, timeout=30)
        rows.append({'check': name, 'http_status': response.status_code,
                     'passed': response.status_code == 422 and sentinel not in response.text})
    return rows


def warm_cases():
    def health(**kwargs):
        return HealthRead(**kwargs).model_dump(mode='json')
    return {
        'simple': ('Show my steps yesterday', [health(metrics=['steps'],
            period={'kind': 'between', 'start_at': '2026-09-22', 'end_at': '2026-09-22'})]),
        'semantic': ('What is the most recent ApoB value I have?', [health(metrics=['apob'],
            operation='latest', period={'kind': 'all_history'})]),
        'multiclause': ('Show my ApoB in 2023; show my steps in 2024', [
            health(metrics=['apob'], period={'kind': 'between', 'start_at': '2023-01-01', 'end_at': '2023-12-31'}),
            health(metrics=['steps'], period={'kind': 'between', 'start_at': '2024-01-01', 'end_at': '2024-12-31'})]),
        'profile': ('Show my complete profile', [health(records=['profile'], operation='latest',
            profile_fields=['all'], date_basis='current_snapshot', period={'kind': 'all_history'})]),
    }


def warm_latency(client, report):
    for name, (question, expected) in warm_cases().items():
        timing = {'warmup_count': 1, 'measured_count': 10, 'attempts': []}
        report['warm_latency'][name] = timing
        req = request(question)
        for number in range(11):
            started = time.perf_counter()
            row = {'warmup': number == 0, 'correct_plan': False}
            try:
                result = client.select(req)
                actual = [q.model_dump(mode='json') for q in result.queries] if result.status == 'planned' else None
                row.update(correct_plan=actual == expected, status=result.status)
            except Exception as exc:
                row['error_type'] = type(exc).__name__
                raise
            finally:
                row['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
                timing['attempts'].append(row)
            if not row['correct_plan']:
                raise ValueError('warm_plan_mismatch')
        values = sorted(row['elapsed_ms'] for row in timing['attempts'] if not row['warmup'])
        timing.update(median_ms=statistics.median(values), p95_ms=values[math.ceil(len(values) * .95) - 1])


def run(args, report):
    if args.selector_sha256 != identity() or args.adapter_sha256 != INTENT_SHA256:
        raise ValueError('reviewed_source_pin_mismatch')
    paths = [ROOT / 'evidence/selector-v5' / name for name in FIXTURES]
    paths.extend(path.resolve() for path in args.additional_fixture)
    if len(set(paths)) != len(paths):
        raise ValueError('duplicate_fixture_path')
    frozen = [load_fixture(path) for path in paths]
    report.update(source_sha256=snapshot_sources(), selector_sha256=args.selector_sha256,
        adapter_sha256=args.adapter_sha256, model_revision=MODEL_REVISION,
        expected_case_entries=sum(f['case_count'] for f in frozen),
        fixtures=[{key: value for key, value in fixture.items() if key != 'cases'} for fixture in frozen])
    client = None
    try:
        # The rejected client never receives a decide/route/select invocation.
        # Its release pin must fail during attestation, before private inference.
        wrong_digest = '0' * 64 if args.release_digest != '0' * 64 else '1' * 64
        unexpected = None
        try:
            unexpected = VitaClient(release_digest=wrong_digest,
                selector_sha256=args.selector_sha256, adapter_sha256=args.adapter_sha256)
        except RuntimeError as exc:
            if str(exc) != 'Unapproved enclave release':
                raise
            report['wrong_release_rejected_before_inference'] = True
        else:
            unexpected.close()
            raise ValueError('wrong_release_was_accepted')
        started = time.perf_counter()
        client = VitaClient(release_digest=args.release_digest,
            selector_sha256=args.selector_sha256, adapter_sha256=args.adapter_sha256)
        report['attestation_setup_ms'] = round((time.perf_counter() - started) * 1000, 3)
        report['attestation_before'] = verification(client, args.release_digest)
        client.http.timeout = httpx.Timeout(30)
        expected_health = {'status': 'ready', 'device': 'cpu', 'model_revision': MODEL_REVISION,
            'adapter_sha256': ADAPTER_SHA256, 'selector_enabled': True,
            'selector_schema': 'vita-selector/v2', 'selector_sha256': args.selector_sha256,
            'selector_adapter_sha256': args.adapter_sha256}
        report['health_before'] = check_health(client, expected_health)
        negative_checks(client, report['negative_checks'])
        if not all(row['passed'] for row in report['negative_checks']):
            raise ValueError('negative_contract_check_failed')
        for fixture in frozen:
            suite = {'fixture': fixture['path'], 'rows': []}
            report['suites'].append(suite)
            for case in fixture['cases']:
                row = evaluate_case(client.http, URL, {'Authorization': 'Bearer ' + client.token},
                                    case, args.selector_sha256)
                suite['rows'].append(row)
                if row.get('transport_failed'):
                    raise RuntimeError('transport_failed_no_application_retry')
            suite['outcomes'] = counts(suite['rows'])
            print(json.dumps({'fixture': Path(fixture['path']).name,
                              'outcomes': suite['outcomes']}), flush=True)
        warm_latency(client, report)
        report['health_after'] = check_health(client, expected_health)
        report['attestation_after'] = verification(client, args.release_digest)
    finally:
        if client is not None:
            client.close()
        report['source_unchanged'] = snapshot_sources() == report['source_sha256']
        report['selector_identity_unchanged'] = identity() == args.selector_sha256
        report['fixtures_unchanged'] = all(digest(Path(f['path'])) == f['sha256'] for f in frozen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-digest', required=True, type=sha256_pin)
    parser.add_argument('--selector-sha256', required=True, type=sha256_pin)
    parser.add_argument('--adapter-sha256', required=True, type=sha256_pin)
    parser.add_argument('--additional-fixture', action='append', type=Path, default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'scope': 'Synthetic attested selector acceptance; not final Vita answer or browser acceptance.',
        'independence': 'Case entries overlap: remaining-19 is a subset of additional-160; reused suites are regression evidence.',
        'enclave': HOST, 'repo': REPO, 'release_digest': args.release_digest,
        'sdk_version': version('tinfoil'), 'transport': 'attestation-pinned TLS',
        'retry_policy': 'No application retries; SDK may retry once after certificate rotation only after fresh attestation and the same release pin.',
        'suites': [], 'negative_checks': [], 'warm_latency': {}, 'passed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        try:
            run(args, report)
        except (Exception, KeyboardInterrupt) as exc:
            # Do not record exception text: upstream errors may contain inputs.
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
                and report.get('wrong_release_rejected_before_inference') is True
                and len(rows) == report.get('expected_case_entries')
                and 'attestation_after' in report and 'health_after' in report
                and len(report['negative_checks']) == 4
                and all(row['passed'] for row in report['negative_checks'])
                and len(report['warm_latency']) == 4
                and all(len(t['attempts']) == 11 and all(a['correct_plan'] for a in t['attempts'])
                        for t in report['warm_latency'].values())
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
