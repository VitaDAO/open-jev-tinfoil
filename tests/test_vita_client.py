import httpx
import pytest
from examples.vita_client import VitaClient, MODEL_REVISION, ADAPTER_SHA256

GOOD={'model_revision':MODEL_REVISION,'adapter_sha256':ADAPTER_SHA256,'advisory':True,'action':'explain a concept','record_access':False}

@pytest.mark.parametrize('override', [{'model_revision':'wrong'},{'adapter_sha256':'wrong'},{'advisory':False},{'action':'execute'},{'record_access':'false'}])
def test_route_client_rejects_unexpected_contract(override):
    client=VitaClient.__new__(VitaClient);client.token='synthetic'
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={**GOOD,**override}))) as http:
        client.http=http
        with pytest.raises(RuntimeError):client.route('synthetic input')


def test_route_client_accepts_expected_adapter():
    client=VitaClient.__new__(VitaClient);client.token='synthetic'
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json=GOOD))) as http:
        client.http=http
        assert client.route('synthetic input')==GOOD

@pytest.mark.parametrize('verified,digest', [(False,'wrong'),(True,'wrong')])
def test_attestation_pin_failure_precedes_key_access(monkeypatch,verified,digest):
    import sys
    from types import SimpleNamespace
    closed=[]
    http=SimpleNamespace(close=lambda:closed.append(True))
    class Verifier:
        def __init__(self,**kwargs):pass
        def make_secure_http_client(self):return http
        def get_verification_document(self):return SimpleNamespace(security_verified=verified,release_digest=digest)
    monkeypatch.setitem(sys.modules,'tinfoil',SimpleNamespace(SecureClient=Verifier))
    monkeypatch.delenv('OPEN_JEV_API_KEY',raising=False)
    with pytest.raises(RuntimeError,match='Unapproved'):
        VitaClient(release_digest="1"*64)
    assert closed==[True]
