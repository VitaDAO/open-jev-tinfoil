import pytest
from fastapi.testclient import TestClient
from server import create_app

BODY = {'state': 'Public synthetic request.', 'questions': [
    {'type': 'noul', 'instructions': 'The request is public.'}]}


class FakeEngine:
    def decide(self, request):
        return {'answers': [{'noul': 0.9}]}


def test_tinfoil_mode_delegates_authentication_without_jev_secret(monkeypatch):
    monkeypatch.delenv('OPEN_JEV_API_KEY', raising=False)
    monkeypatch.delenv('OPEN_JEV_CANDIDATE_API_KEY', raising=False)
    app = create_app(FakeEngine, auth_mode='tinfoil')
    with TestClient(app) as client:
        assert client.post('/decide', json=BODY).json() == {'answers': [{'noul': 0.9}]}
        assert client.post('/decide', content=b'x' * 65537).status_code == 413


@pytest.mark.parametrize('peer', ['127.0.0.1', '198.51.100.1'])
def test_application_does_not_apply_an_ip_policy(peer):
    app = create_app(FakeEngine, auth_mode='tinfoil')
    with TestClient(app, client=(peer, 12345)) as client:
        assert client.get('/health').status_code == 200
        response = client.post('/decide', json=BODY, headers={
            'X-Forwarded-For': '172.31.255.1', 'X-Real-IP': '172.31.255.1',
            'Authorization': 'Bearer synthetic-token'})
        assert response.status_code == 200


def test_unknown_authentication_mode_fails_closed():
    with pytest.raises(RuntimeError, match='authentication mode'):
        create_app(FakeEngine, auth_mode='none')
