"""Matched public-only diagnostic. No API key, selector request or user records."""
import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

from examples.trust_cache import PublicTrustCache
from examples.vita_client import HOST, REPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-digest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to replace prior evidence')
    assert version('tinfoil') == '0.14.0+vita2'
    cache = PublicTrustCache()
    start = time.perf_counter()
    snapshot = cache.refresh()
    refresh_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    from tinfoil import SecureClient
    sdk_import_ms = (time.perf_counter() - start) * 1000
    rows = []
    report = {
        'scope': 'Six fresh client objects in one diagnostic process; SDK imports '
                 'excluded and reported separately; disk caches retained; same '
                 'five-minute trust snapshot in both arms. No private request, '
                 'credential, selector call or TLS application request. These '
                 'measure client setup, not total Vita request latency.',
        'public_refresh_ms': refresh_ms, 'sdk_import_ms': sdk_import_ms,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'release_digest': args.release_digest, 'rows': rows,
    }
    for parallel in [False, True, True, False, False, True]:
        snapshot.assert_valid()
        start = time.perf_counter()
        client = SecureClient(enclave=HOST, repo=REPO, transport='tls',
                              sigstore_verifier_factory=snapshot.make_verifier,
                              parallel_public_fetches=parallel)
        http = client.make_secure_http_client()
        try:
            elapsed = (time.perf_counter() - start) * 1000
            snapshot.assert_valid()
            document = client.get_verification_document()
            assert document.security_verified is True
            assert document.release_digest == args.release_digest
            assert all(s.status == 'success' for s in document.steps.values())
            row = {'parallel': parallel, 'bootstrap_ms': elapsed,
                   'verified': document.security_verified, 'release_tag': document.release_tag}
            rows.append(row)
            output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
        finally:
            http.close()


if __name__ == '__main__':
    main()
