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


@pytest.mark.parametrize('rotated_digest,allowed', [('1'*64,True),('2'*64,False)])
def test_real_sdk_rotation_cannot_retry_into_an_unapproved_release(monkeypatch,rotated_digest,allowed):
    import ssl
    from types import SimpleNamespace
    sdk=pytest.importorskip('tinfoil.client')
    digest=['1'*64];verified=[];transports=[];delivered=[]
    def verify(self):
        verified.append(digest[0])
        self._verification_document=SimpleNamespace(security_verified=True,release_digest=digest[0])
        return SimpleNamespace(public_key='synthetic-attested-key')
    monkeypatch.setattr(sdk.SecureClient,'verify',verify)
    monkeypatch.setattr(sdk.SecureClient,'_build_sync_ssl_context',lambda *_:object())
    class SimulatedTLS(sdk.httpx.BaseTransport):
        def __init__(self,**kwargs):self.number=len(transports);transports.append(self)
        def handle_request(self,request):
            if self.number==0:
                # TLS fails before application bytes are sent. The real SDK
                # then re-verifies before constructing its retry transport.
                raise sdk.httpx.ConnectError('certificate verify failed') from ssl.SSLCertVerificationError('certificate verify failed')
            delivered.append(request)
            return sdk.httpx.Response(200,json=GOOD)
    monkeypatch.setattr(sdk.httpx,'HTTPTransport',SimulatedTLS)
    monkeypatch.setenv('OPEN_JEV_API_KEY','synthetic-key-for-rotation-test-only')
    client=VitaClient(release_digest='1'*64)
    digest[0]=rotated_digest
    try:
        if allowed:
            assert client.route('synthetic decision')==GOOD
            assert len(transports)==2 and len(delivered)==1
        else:
            with pytest.raises(sdk.httpx.ConnectError):client.route('synthetic decision')
            assert len(transports)==1 and delivered==[]
        assert verified==['1'*64,rotated_digest]
    finally:client.close()
