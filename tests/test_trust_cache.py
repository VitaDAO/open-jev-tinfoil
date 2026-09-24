import hashlib
import os
from dataclasses import replace
import pytest
from examples import trust_cache as tc


@pytest.fixture
def clock(monkeypatch):
    values=[1000.,100.]
    monkeypatch.setattr(tc.time,'time',lambda:values[0])
    monkeypatch.setattr(tc.time,'monotonic',lambda:values[1])
    return values


def snapshot(clock, lifetime=300):
    raw=b'synthetic-public-root'
    return tc.TrustSnapshot(raw,hashlib.sha256(raw).hexdigest(),*clock,clock[0]+lifetime)


def test_maximum_age_and_earlier_signed_expiry(clock):
    value=snapshot(clock,45)
    clock[:]=[1044,144];value.assert_valid()
    clock[:]=[1045,145]
    with pytest.raises(RuntimeError,match='expired'):value.assert_valid()
    clock[:]=[1000,100];value=snapshot(clock)
    clock[:]=[1299,399];value.assert_valid()
    clock[:]=[1300,400]
    with pytest.raises(RuntimeError,match='expired'):value.assert_valid()


@pytest.mark.parametrize('wall,mono',[(999,101),(1001,99),(1001,401),(1400,101)])
def test_clock_changes_cannot_extend_cache(clock,wall,mono):
    value=snapshot(clock);clock[:]=[wall,mono]
    with pytest.raises(RuntimeError,match='expired'):value.assert_valid()


def test_modified_bytes_and_extended_validity_rejected(clock):
    value=snapshot(clock)
    with pytest.raises(RuntimeError,match='integrity'):replace(value,root_bytes=b'changed').assert_valid()
    with pytest.raises(RuntimeError,match='expired'):replace(value,expires_at=1400).assert_valid()


def test_refresh_failure_retains_only_original_validity(clock,monkeypatch):
    monkeypatch.setattr(tc,'_validated_root',lambda:(b'root',1200))
    monkeypatch.setattr(tc.TrustSnapshot,'make_verifier',lambda self:object())
    monkeypatch.setattr(tc,'_fetch_snapshot',tc._refresh_snapshot)
    cache=tc.PublicTrustCache();first=cache.refresh()
    assert first.expires_at==1200 and not cache.refresh_due
    clock[:]=[1141,241];assert cache.refresh_due
    def fail():raise RuntimeError('synthetic_refresh_failure')
    monkeypatch.setattr(tc,'_validated_root',fail)
    with pytest.raises(RuntimeError):cache.refresh()
    assert cache.snapshot() is first
    clock[:]=[1200,300]
    with pytest.raises(RuntimeError,match='expired'):cache.snapshot()
    assert cache.refresh_due


def test_slow_refresh_does_not_restart_age(clock,monkeypatch):
    def load():
        clock[:]=[1301,401]
        return b'root',5000
    monkeypatch.setattr(tc,'_validated_root',load)
    monkeypatch.setattr(tc,'_fetch_snapshot',tc._refresh_snapshot)
    cache=tc.PublicTrustCache()
    with pytest.raises(RuntimeError,match='expired'):cache.refresh()
    with pytest.raises(RuntimeError,match='not_ready'):cache.snapshot()


def test_child_never_touches_parent_refresh_lock(clock):
    cache=tc.PublicTrustCache();cache._owner_pid=os.getpid()+1
    cache._refresh_lock=None
    with pytest.raises(RuntimeError,match='parent_only'):cache.refresh()
    with pytest.raises(RuntimeError,match='parent_only'):cache.snapshot()


def test_expired_snapshot_stops_request_but_allows_owner_cleanup(clock):
    from types import SimpleNamespace
    from examples.vita_client import VitaClient
    client=VitaClient.__new__(VitaClient);client._owner_pid=os.getpid()
    client._trust_snapshot=snapshot(clock);closed=[]
    client.http=SimpleNamespace(close=lambda:closed.append(True))
    clock[:]=[1300,400]
    for method,args in [('select',(None,)),('route',('synthetic',)),('decide',('synthetic',[]))]:
        with pytest.raises(RuntimeError,match='expired'):getattr(client,method)(*args)
    client.close();assert closed==[True]


def test_injected_verifier_retains_signature_and_identity_checks():
    import json
    from pathlib import Path
    from importlib.metadata import version
    if version('tinfoil')!='0.14.0+vita1':pytest.skip('opt-in patched SDK test')
    from sigstore.models import TrustedRoot
    from sigstore.verify import Verifier
    from tinfoil.sigstore import verify_attestation
    root=Path(__file__).parent/'fixtures/trust-cache'
    verifier=Verifier(trusted_root=TrustedRoot.from_file(str(root/'public-root.json')))
    bundle=(root/'release-bundle.json').read_bytes()
    digest='e9145d5c5610cc9096ef5ed1b253a22f13503898bd1319e39dc24c54bcdf88b5'
    assert verify_attestation(bundle,digest,'VitaDAO/open-jev-tinfoil','v0.3.2',verifier=verifier)
    for d,r,t in [('0'*64,'VitaDAO/open-jev-tinfoil','v0.3.2'),
                  (digest,'wrong/repo','v0.3.2'),(digest,'VitaDAO/open-jev-tinfoil','v0.3.1')]:
        with pytest.raises(ValueError):verify_attestation(bundle,d,r,t,verifier=verifier)
    tampered=json.loads(bundle)
    tampered['dsseEnvelope']['signatures'][0]['sig']='AAAA'
    with pytest.raises(ValueError):
        verify_attestation(json.dumps(tampered).encode(),digest,'VitaDAO/open-jev-tinfoil','v0.3.2',verifier=verifier)


def test_loader_uses_authenticated_target_and_earliest_metadata_expiry(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from datetime import datetime, timezone
    import sigstore._internal.tuf as tuf_module
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata,'version',lambda n:{'sigstore':'4.5.0','tuf':'6.0.0'}[n])
    raw=b'authenticated-public-target';path=tmp_path/'trusted_root.json';path.write_bytes(raw)
    checked=[]
    target=SimpleNamespace(verify_length_and_hashes=lambda b:checked.append(b))
    # Actual TUF6 TrustedMetadataSet iteration yields Signed objects, not role keys.
    signed=[SimpleNamespace(expires=datetime.fromtimestamp(t,timezone.utc)) for t in (1400,1200,1500)]
    inner=SimpleNamespace(get_targetinfo=lambda name:target,_trusted_set=signed)
    monkeypatch.setattr(tuf_module,'TrustUpdater',lambda url:SimpleNamespace(_updater=inner,get_trusted_root_path=lambda:str(path)))
    assert tc._validated_root()==(raw,1200)
    assert checked==[raw]
    def reject(_):raise ValueError('target hash mismatch')
    target.verify_length_and_hashes=reject
    with pytest.raises(ValueError):tc._validated_root()


def test_snapshot_constructs_new_crypto_objects_without_tuf_network(monkeypatch):
    import time
    from pathlib import Path
    from sigstore.verify import Verifier
    raw=(Path(__file__).parent/'fixtures/trust-cache/public-root.json').read_bytes()
    value=tc.TrustSnapshot(raw,hashlib.sha256(raw).hexdigest(),time.time(),time.monotonic(),time.time()+299)
    def fail(**kw):raise AssertionError('Unexpected online refresh in child')
    monkeypatch.setattr(Verifier,'production',fail)
    first=value.make_verifier();second=value.make_verifier()
    assert first is not second


def test_public_refresh_worker_receives_no_credentials(clock,monkeypatch):
    import base64,json
    from types import SimpleNamespace
    value=snapshot(clock)
    monkeypatch.setenv('OPEN_JEV_API_KEY','synthetic-secret-never-forwarded')
    monkeypatch.setenv('UNRELATED_PRIVATE_TOKEN','another-synthetic-secret')
    seen=[]
    def run(args,**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(stdout=json.dumps({'root_base64':base64.b64encode(value.root_bytes).decode(),
            'sha256':value.sha256,'issued_at':value.issued_at,
            'issued_monotonic':value.issued_monotonic,'expires_at':value.expires_at}))
    monkeypatch.setattr(tc.subprocess,'run',run)
    assert tc._fetch_snapshot(*clock)==value
    assert 'OPEN_JEV_API_KEY' not in seen[0]['env']
    assert 'UNRELATED_PRIVATE_TOKEN' not in seen[0]['env']
    assert seen[0]['timeout']==30 and seen[0]['check'] is True
