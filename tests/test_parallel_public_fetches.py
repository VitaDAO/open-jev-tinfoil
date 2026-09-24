"""Exercise the opt-in wheel's real verification flow without network/private data."""
from importlib.metadata import version
from threading import Barrier, enumerate as threads, get_ident
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(version('tinfoil') != '0.14.0+vita2',
                                reason='requires opt-in vita2 wheel')


@pytest.fixture
def flow(monkeypatch):
    import tinfoil.client as sdk
    calls = []
    errors = {}
    rendezvous = [None]
    owner = get_ident()

    def stage(name):
        calls.append((name, get_ident()))
        if name in errors:
            raise ValueError(name)

    measurement = SimpleNamespace(type='synthetic-snp', registers=['measurement'],
                                  fingerprint=lambda: 'measurement-fingerprint')
    verification = SimpleNamespace(measurement=measurement, public_key_fp='public-key',
                                   hpke_public_key='')

    def enclave_verify():
        stage('verify_enclave')
        return verification

    def fetch_attestation(host):
        assert host == 'synthetic.example'
        if rendezvous[0]:
            rendezvous[0].wait(2)
        stage('fetch_enclave')
        return SimpleNamespace(verify=enclave_verify)

    def fetch_release(repo):
        assert repo == 'synthetic/repo'
        if rendezvous[0]:
            rendezvous[0].wait(2)
        stage('fetch_release')
        return SimpleNamespace(tag='v1', digest='1' * 64)

    def verify_code(bundle, digest, repo, tag, *, verifier):
        assert (bundle, digest, repo, tag, verifier) == (
            b'public-bundle', '1' * 64, 'synthetic/repo', 'v1', 'fresh-verifier')
        stage('verify_code')
        return SimpleNamespace(fingerprint=lambda: 'code-fingerprint',
                               assert_equal=lambda actual: stage('compare_measurements'))

    monkeypatch.setattr(sdk, 'resolve_user_cache_secret', lambda *_: 'synthetic')
    monkeypatch.setattr(sdk, 'fetch_attestation', fetch_attestation)
    monkeypatch.setattr(sdk, 'fetch_latest_release', fetch_release)
    monkeypatch.setattr(sdk, 'fetch_attestation_bundle', lambda *_: b'public-bundle')
    monkeypatch.setattr(sdk, 'verify_attestation', verify_code)
    monkeypatch.setattr(sdk, 'fetch_latest_hardware_measurements', lambda: 'hardware')
    monkeypatch.setattr(sdk, 'verify_tdx_hardware', lambda *_: stage('verify_hardware'))

    def factory():
        stage('verifier_factory')
        return 'fresh-verifier'

    def client(**kwargs):
        return sdk.SecureClient(enclave='synthetic.example', repo='synthetic/repo',
                                transport='tls', sigstore_verifier_factory=factory, **kwargs)

    yield SimpleNamespace(sdk=sdk, client=client, calls=calls, errors=errors,
                          rendezvous=rendezvous, owner=owner, measurement=measurement)
    assert not any(t.name.startswith('tinfoil-public') for t in threads())


def test_fetches_overlap_and_join_before_verification_in_request_thread(flow):
    flow.rendezvous[0] = Barrier(2)
    client = flow.client(parallel_public_fetches=True)
    assert flow.calls == []  # Construction starts no thread or network operation.
    for _ in range(2):
        flow.calls.clear()
        assert client.verify().digest == '1' * 64
        assert {name for name, _ in flow.calls[:2]} == {'fetch_enclave', 'fetch_release'}
        assert all(tid != flow.owner for _, tid in flow.calls[:2])
        assert len({tid for _, tid in flow.calls[:2]}) == 2
        assert all(tid == flow.owner for _, tid in flow.calls[2:])
        assert not any(t.name.startswith('tinfoil-public') for t in threads())
        doc = client.get_verification_document()
        assert doc.security_verified is True and doc.release_tag == 'v1'
        assert all(s.status == 'success' for s in doc.steps.values())


@pytest.mark.parametrize('failure,step', [
    ('fetch_enclave', 'verify_enclave'), ('verify_enclave', 'verify_enclave'),
    ('fetch_release', 'fetch_digest'), ('verify_code', 'verify_code'),
    ('compare_measurements', 'compare_measurements'), ('verify_hardware', 'verify_enclave'),
])
def test_failures_keep_original_stage_and_never_finalize(flow, failure, step):
    flow.rendezvous[0] = Barrier(2)
    flow.errors[failure] = True
    if failure == 'verify_hardware':
        flow.measurement.type = next(iter(flow.sdk.TDX_TYPES))
    client = flow.client(parallel_public_fetches=True)
    with pytest.raises(ValueError, match=failure) as caught:
        client.verify()
    doc = client.get_verification_document()
    assert doc.steps[step].status == 'failed'
    assert doc.security_verified is not True and client.ground_truth is None
    assert getattr(caught.value, 'verification_document', None) is not None


def test_two_fetch_failures_keep_enclave_failure_precedence_and_join_workers(flow):
    flow.rendezvous[0] = Barrier(2)
    flow.errors.update(fetch_enclave=True, fetch_release=True)
    with pytest.raises(ValueError, match='fetch_enclave'):
        flow.client(parallel_public_fetches=True).verify()
    assert {name for name, _ in flow.calls} == {'fetch_enclave', 'fetch_release'}


def test_default_retains_serial_order_and_no_worker_threads(flow):
    flow.client().verify()
    assert [name for name, _ in flow.calls] == [
        'fetch_enclave', 'verify_enclave', 'fetch_release', 'verifier_factory',
        'verify_code', 'compare_measurements']
    assert all(tid == flow.owner for _, tid in flow.calls)


def test_measurement_mode_never_fetches_release(flow):
    client = flow.client(parallel_public_fetches=True,
                         measurement={'snp_measurement': 'measurement'})
    assert client.verify().digest == 'pinned_no_digest'
    assert [name for name, _ in flow.calls] == ['fetch_enclave', 'verify_enclave']
    assert all(tid == flow.owner for _, tid in flow.calls)


def test_bundle_mode_never_prefetches_direct_inputs(flow, monkeypatch):
    bundle = object()
    monkeypatch.setattr(flow.sdk, 'fetch_bundle_from', lambda *a, **kw: bundle)
    monkeypatch.setattr(flow.sdk.SecureClient, 'verify_from_bundle', lambda self, b: b)
    client = flow.client(parallel_public_fetches=True,
                         attestation_bundle_url='https://synthetic.example/bundle')
    assert client.verify() is bundle
    assert flow.calls == []


def test_wrapper_requires_vita2_before_any_network(monkeypatch):
    import importlib.metadata
    from examples.vita_client import VitaClient
    monkeypatch.setattr(importlib.metadata, 'version', lambda _: '0.14.0+vita1')
    with pytest.raises(RuntimeError, match='vita2'):
        VitaClient(release_digest='1' * 64, parallel_public_fetches=True)


def test_parallel_wrapper_checks_release_before_reading_credentials(flow, monkeypatch):
    from examples import vita_client
    monkeypatch.setattr(vita_client, 'HOST', 'synthetic.example')
    monkeypatch.setattr(vita_client, 'REPO', 'synthetic/repo')
    # This test exercises the real patched SDK; trust factory is injected only
    # to avoid online TUF, while all verification stages above still execute.
    monkeypatch.setattr(flow.sdk, 'verify_attestation', lambda *a, **kw:
        SimpleNamespace(fingerprint=lambda: 'code', assert_equal=lambda _: None))
    monkeypatch.delenv('OPEN_JEV_API_KEY', raising=False)
    flow.rendezvous[0] = Barrier(2)
    with pytest.raises(RuntimeError, match='Unapproved enclave release'):
        vita_client.VitaClient(release_digest='2' * 64, parallel_public_fetches=True)
