import httpx
import pytest
from examples.vita_client import VitaDecisionClient, MODEL_REVISION, ADAPTER_SHA256

GOOD={'model_revision':MODEL_REVISION,'adapter_sha256':ADAPTER_SHA256,'advisory':True,'action':'explain a concept','record_access':False}

@pytest.mark.parametrize('override', [{'model_revision':'wrong'},{'adapter_sha256':'wrong'},{'advisory':False},{'action':'execute'},{'record_access':'false'}])
def test_route_client_rejects_unexpected_contract(override):
    client=VitaDecisionClient.__new__(VitaDecisionClient);client.token='synthetic'
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={**GOOD,**override}))) as http:
        client.http=http
        with pytest.raises(RuntimeError):client.route('synthetic input')


def test_route_client_accepts_expected_adapter():
    client=VitaDecisionClient.__new__(VitaDecisionClient);client.token='synthetic'
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json=GOOD))) as http:
        client.http=http
        assert client.route('synthetic input')==GOOD
