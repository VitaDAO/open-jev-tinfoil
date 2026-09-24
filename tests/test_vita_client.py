import os
import httpx
import pytest
from examples.vita_client import VitaClient, MODEL_REVISION, ADAPTER_SHA256

GOOD={'model_revision':MODEL_REVISION,'adapter_sha256':ADAPTER_SHA256,'advisory':True,'action':'explain a concept','record_access':False}

@pytest.mark.parametrize('override', [{'model_revision':'wrong'},{'adapter_sha256':'wrong'},{'advisory':False},{'action':'execute'},{'record_access':'false'}])
def test_route_client_rejects_unexpected_contract(override):
    client=VitaClient.__new__(VitaClient);client.token='synthetic';client._owner_pid=os.getpid()
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={**GOOD,**override}))) as http:
        client.http=http
        with pytest.raises(RuntimeError):client.route('synthetic input')


def test_route_client_accepts_expected_adapter():
    client=VitaClient.__new__(VitaClient);client.token='synthetic';client._owner_pid=os.getpid()
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


@pytest.fixture
def selector_client(monkeypatch):
    import sys
    from types import SimpleNamespace
    from query_plan import QueryRequest, QueryPlan, HealthRead, request_identity
    req=QueryRequest.model_validate({'schema_version':'vita-selector/v2',
        'state':{'current_request':'Show my steps','reference_date':'2026-09-23','time_zone':'UTC'},
        'reference_time':'2026-09-23T12:00:00Z','available_metrics':['steps'],
        'available_record_types':[],'available_sources':[],'literature_available':False})
    good=QueryPlan(status='planned',queries=[HealthRead(metrics=['steps'],period={'kind':'all_history'})],
        time_zone='UTC',selector_sha256='2'*64,adapter_sha256='3'*64,
        model_revision=MODEL_REVISION,request_sha256=request_identity(req)).model_dump(mode='json')
    http=httpx.Client(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=good)))
    class Verifier:
        def __init__(self,**kwargs):pass
        def make_secure_http_client(self):return http
        def get_verification_document(self):
            return SimpleNamespace(security_verified=True,release_digest='1'*64)
    monkeypatch.setitem(sys.modules,'tinfoil',SimpleNamespace(SecureClient=Verifier))
    monkeypatch.setenv('OPEN_JEV_API_KEY','synthetic-selector-test-key')
    client=VitaClient(release_digest='1'*64,selector_sha256='2'*64,adapter_sha256='3'*64)
    try:yield client,req
    finally:client.close()


def test_cancelled_waiter_keeps_admission_until_worker_finishes(selector_client,monkeypatch):
    import asyncio
    from threading import Event
    client,req=selector_client
    entered=Event();release=Event();finished=Event();calls=[]
    post=client.http.post
    def blocked_post(*args,**kwargs):
        calls.append(True)
        if len(calls)==1:
            entered.set()
            assert release.wait(5), 'Test did not release the blocked HTTP worker'
        return post(*args,**kwargs)
    monkeypatch.setattr(client.http,'post',blocked_post)
    def first_select():
        try:return client.select(req)
        finally:finished.set()
    async def scenario():
        waiting=asyncio.create_task(asyncio.to_thread(first_select))
        try:
            assert await asyncio.to_thread(entered.wait,2)
            with pytest.raises(TimeoutError):await asyncio.wait_for(waiting,.01)
            assert waiting.cancelled() and not finished.is_set()
            for _ in range(10):
                with pytest.raises(RuntimeError,match='already in progress'):
                    await asyncio.to_thread(client.select,req)
            assert len(calls)==1
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait,2)
        assert (await asyncio.to_thread(client.select,req)).status=='planned'
        assert len(calls)==2
    asyncio.run(scenario())


@pytest.mark.parametrize('failure',['transport','response'])
def test_selector_exception_releases_admission(selector_client,monkeypatch,failure):
    client,req=selector_client
    post=client.http.post;calls=[]
    def fail_once(*args,**kwargs):
        calls.append(True)
        if len(calls)==1:
            if failure=='transport':raise httpx.ReadTimeout('synthetic timeout')
            return httpx.Response(200,json={},request=httpx.Request('POST',args[0]))
        return post(*args,**kwargs)
    monkeypatch.setattr(client.http,'post',fail_once)
    with pytest.raises((httpx.ReadTimeout,RuntimeError)):client.select(req)
    assert client.select(req).status=='planned' and len(calls)==2


@pytest.mark.parametrize('method,args', [('select', (None,)), ('route', ('synthetic',)),
                                         ('decide', ('synthetic', [])), ('close', ())])
def test_forked_client_rejected_before_lock_or_transport_access(method,args):
    # Intentionally no lock, token or transport: a guard placed after access fails.
    client=VitaClient.__new__(VitaClient)
    client._owner_pid=os.getpid()+1
    with pytest.raises(RuntimeError,match='another process'):
        getattr(client,method)(*args)


@pytest.mark.skipif(not hasattr(os,'fork'),reason='requires POSIX fork')
def test_real_fork_rejects_inherited_locked_client_and_parent_stays_usable(selector_client):
    import select
    client,request=selector_client
    reader,writer=os.pipe()
    client._select_lock.acquire()
    pid=os.fork()
    if pid==0:
        os.close(reader)
        try:
            for method,args in [('select',(request,)),('route',('synthetic',)),
                                ('decide',('synthetic',[])),('close',())]:
                try:getattr(client,method)(*args)
                except RuntimeError as exc:
                    if 'another process' not in str(exc):raise
                else:raise AssertionError('Inherited operation succeeded')
            os.write(writer,b'guarded')
            os._exit(0)
        except BaseException:
            os._exit(1)
    os.close(writer)
    try:
        ready,_,_=select.select([reader],[],[],3)
        assert ready, 'Child blocked on inherited lock'
        assert os.read(reader,64)==b'guarded'
        assert os.waitpid(pid,0)[1]==0
        pid=None
    finally:
        os.close(reader)
        if pid is not None:
            import signal
            os.kill(pid,signal.SIGKILL)
            os.waitpid(pid,0)
        client._select_lock.release()
    assert client.select(request).status=='planned'
