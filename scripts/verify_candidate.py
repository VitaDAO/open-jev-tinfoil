"""Synthetic serving parity over loopback, or attested TLS to a pinned candidate."""
import argparse, json, os, statistics, time
from pathlib import Path
from urllib.parse import urlparse
import httpx

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--base-url')
    target.add_argument('--enclave')
    parser.add_argument('--release-digest')
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads((ROOT/'tests/fixtures/candidate-release-parity.json').read_text())
    verification = None
    if args.enclave:
        if not args.release_digest:
            raise ValueError('A reviewed candidate release digest is required')
        from tinfoil import SecureClient
        class ReleasePinnedClient(SecureClient):
            def verify(self):
                result = super().verify()
                document = self.get_verification_document()
                if not document or document.security_verified is not True or document.release_digest != args.release_digest:
                    raise RuntimeError('Candidate attestation or release pin mismatch')
                return result
        verifier = ReleasePinnedClient(enclave=args.enclave, repo='VitaDAO/open-jev-candidate-tinfoil-config', transport='tls')
        verifier.verify()
        document = verifier.get_verification_document()
        if not document or document.security_verified is not True or document.release_digest != args.release_digest:
            raise RuntimeError('Candidate attestation or release pin mismatch')
        verification = {'security_verified': True, 'release_digest': document.release_digest}
        http = verifier.make_secure_http_client()
        base = 'https://'+args.enclave
    else:
        parsed = urlparse(args.base_url)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not parsed.port:
            raise ValueError('Plain HTTP is restricted to explicit loopback testing')
        http = httpx.Client(timeout=30)
        base = args.base_url.rstrip('/')
    # Credentials are accessed only after remote attestation succeeds.
    key = args.key_file.read_text().strip() if args.key_file else os.environ['OPEN_JEV_CANDIDATE_API_KEY']
    headers = {'Authorization': 'Bearer '+key}
    health = http.get(base+'/health', timeout=30); health.raise_for_status(); health = health.json()
    for name in ('selector_sha256', 'learned_parser_sha256'):
        if health.get(name) != fixture[name]:
            raise RuntimeError('Candidate identity mismatch: '+name)
    assert health['status'] == 'ready' and health['selector_parser'] == 'learned'
    assert http.post(base+'/v1/select', json=fixture['cases'][0]['request'], timeout=30).status_code == 401
    from query_plan import QueryRequest, QueryPlan, compile_batch
    rows = []
    try:
        for case in fixture['cases']:
            started = time.perf_counter()
            response = http.post(base+'/v1/select', json=case['request'], headers=headers, timeout=30)
            response.raise_for_status(); result = response.json()
            request = QueryRequest.model_validate(case['request']); plan = QueryPlan.model_validate(result)
            compile_batch(plan, request)
            passed = result['status'] == case['status'] and result['queries'] == case['queries']
            rows.append({'id': case['id'], 'passed': passed, 'status': result['status'],
                'elapsed_ms': (time.perf_counter()-started)*1000})
            if len(rows)%50 == 0:
                print('Parity', len(rows), '/', len(fixture['cases']), 'failures', sum(not r['passed'] for r in rows), flush=True)
        after = http.get(base+'/health', timeout=30); after.raise_for_status(); after = after.json()
        assert after['selector_sha256'] == health['selector_sha256']
        assert after['learned_parser_sha256'] == health['learned_parser_sha256']
        if args.enclave:
            document = verifier.get_verification_document()
            assert document.security_verified is True and document.release_digest == args.release_digest
        values = sorted(r['elapsed_ms'] for r in rows)
        report = {'scope': fixture['scope'], 'attestation': verification, 'health': health,
            'passed': sum(r['passed'] for r in rows), 'failed': sum(not r['passed'] for r in rows),
            'authentication_rejection_passed': True, 'p50_http_ms': statistics.median(values),
            'p95_http_ms': values[int(.95*(len(values)-1))], 'rows': rows}
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('rows','health')}, indent=2))
        if report['failed']:
            raise SystemExit('Candidate serving parity failed')
    finally:
        http.close()

if __name__ == '__main__':
    main()
