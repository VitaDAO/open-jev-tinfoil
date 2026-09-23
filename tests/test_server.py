import asyncio
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from server import create_app

TOKEN = 'synthetic-test-token-' * 3
HEADERS = {'Authorization': 'Bearer ' + TOKEN}
BODY = {'state': 'I want a refund.', 'questions': [{'type': 'noul', 'instructions': 'The customer wants a refund.'}]}

class FakeEngine:
    def decide(self, request):
        return {'answers': [{'noul': 0.9}]}

@pytest.fixture
def client():
    with TestClient(create_app(FakeEngine, TOKEN)) as client:
        yield client

def test_auth_health_and_success(client):
    assert client.get('/health').status_code == 200
    assert client.post('/decide', json=BODY).status_code == 401
    assert client.post('/decide', json=BODY, headers={'Authorization': 'Bearer wrong'}).status_code == 401
    assert client.post('/decide', json=BODY, headers=HEADERS).json()['answers'] == [{'noul': 0.9}]

@pytest.mark.parametrize('question', [
    {'type': 'unknown', 'instructions': 'test'},
    {'type': 'choice', 'instructions': 'test', 'options': ['same', 'same']},
    {'type': 'score', 'instructions': 'test', 'options': ['one']},
    {'type': 'noul', 'instructions': 'test', 'options': ['yes', 'no']},
    {'type': 'choice', 'instructions': '', 'options': ['a','b']},
])
def test_invalid_question_is_rejected_without_echo(client, question):
    response = client.post('/decide', json={'state': 'private-sentinel', 'questions': [question]}, headers=HEADERS)
    assert response.status_code == 422
    assert 'private-sentinel' not in response.text

def test_oversize_and_empty(client):
    assert client.post('/decide', content=b'x' * 65537, headers=HEADERS).status_code == 413
    assert client.post('/decide', json={'state': 'test', 'questions': []}, headers=HEADERS).status_code == 422

def test_busy_rejects_and_recovers():
    started, release = threading.Event(), threading.Event()
    class SlowEngine:
        def decide(self, request):
            started.set()
            release.wait(5)
            return {'answers': []}
    async def check():
        app = create_app(SlowEngine, TOKEN)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                task = asyncio.create_task(client.post('/decide', json=BODY, headers=HEADERS))
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(.01)
                assert started.is_set()
                try:
                    assert (await client.post('/decide', json=BODY, headers=HEADERS)).status_code == 429
                    assert (await client.get('/health')).status_code == 200
                finally:
                    release.set()
                assert (await task).status_code == 200
                assert (await client.post('/decide', json=BODY, headers=HEADERS)).status_code == 200
    asyncio.run(check())

def test_missing_key_fails_closed(monkeypatch):
    monkeypatch.delenv('OPEN_JEV_API_KEY', raising=False)
    with pytest.raises(RuntimeError):
        create_app(FakeEngine)

class RouteEngine(FakeEngine):
    def route(self, request):
        return {'action':'explain a concept','record_access':False,'advisory':True}


def test_route_auth_contract_and_invalid_inputs():
    with TestClient(create_app(RouteEngine,TOKEN)) as client:
        body={'state':'Explain sleep latency.'}
        assert client.post('/route',json=body).status_code==401
        assert client.post('/route',json=body,headers=HEADERS).json()['record_access'] is False
        for payload in [{'state':' '},{'state':123},{'state':'private-sentinel','questions':[]}]:
            response=client.post('/route',json=payload,headers=HEADERS)
            assert response.status_code==422
            assert 'private-sentinel' not in response.text
        assert client.post('/route',content=b'x'*65537,headers=HEADERS).status_code==413


def test_route_and_decide_share_inference_lock():
    started,release=threading.Event(),threading.Event()
    class SlowRoute(RouteEngine):
        def route(self,request):
            started.set();release.wait(5)
            return super().route(request)
    async def check():
        app=create_app(SlowRoute,TOKEN)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                task=asyncio.create_task(client.post('/route',json={'state':'test'},headers=HEADERS))
                for _ in range(100):
                    if started.is_set():break
                    await asyncio.sleep(.01)
                assert started.is_set()
                try:
                    assert (await client.post('/decide',json=BODY,headers=HEADERS)).status_code==429
                    assert (await client.post('/route',json={'state':'test'},headers=HEADERS)).status_code==429
                finally:release.set()
                assert (await task).status_code==200
                assert (await client.post('/route',json={'state':'test'},headers=HEADERS)).status_code==200
    asyncio.run(check())
