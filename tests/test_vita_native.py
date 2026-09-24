"""Run with current Vita's Python/PYTHONPATH; no provider keys or patient data."""
import asyncio
import copy
import json
import os
from pathlib import Path
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

vita = pytest.importorskip('vita_agent')
pytest.importorskip('jsonschema')
from backbone.synthetic import fixture, QUESTION, ANSWER, READ, final_args
from backbone.test_manager import action
from examples.vita_native import NativeSelectorProvider, native_batch
from query_plan import QueryRequest, QueryPlan, HealthRead, compile_batch, request_identity
from routing import MODEL_REVISION


def request(question=QUESTION):
    return QueryRequest.model_validate({'schema_version':'vita-selector/v2',
        'state':{'current_request':question,'recent_user_requests':[],
                 'time_zone':'Europe/Bucharest','reference_date':'2026-09-24'},
        'reference_time':'2026-09-24T12:00:00Z',
        'available_metrics':['heart_rate_variability'],
        'available_record_types':[], 'available_sources':['whoop']})


def plan(req, *, handoff=False):
    return QueryPlan(status='handoff' if handoff else 'planned',
        queries=[] if handoff else [HealthRead(metrics=['heart_rate_variability'],
            operation='latest', period={'kind':'all_history'})],
        reason_codes=['unsupported'] if handoff else [], time_zone=req.state.time_zone,
        selector_sha256='1'*64,adapter_sha256='2'*64,model_revision=MODEL_REVISION,
        request_sha256=request_identity(req))


class Client:
    def __init__(self, result):self.result=result;self.calls=0
    def select(self, req):self.calls+=1;return self.result


def envelope(manager, req):
    state=manager.state()
    return {'request':{'messages':[{'role':'system','content':[{'type':'text',
        'text':state['instructions']}]},{'role':'user','content':[{'type':'text',
        'text':req.state.current_request}]}], 'tools':state['tools']},
        'controls':state['controls']}


def test_environment_is_the_requested_current_vita_checkout():
    root=Path(os.environ['VITA_SOURCE_ROOT']).resolve()
    assert Path(vita.__file__).resolve().is_relative_to(root)


@pytest.mark.parametrize('handoff',[False,True])
def test_real_native_manager_brokers_reads_registers_sources_and_accepts_answer(handoff):
    async def run():
        manager,resolver,_=fixture()
        req=request();client=Client(plan(req,handoff=handoff));seen=[]
        async def provider(env):
            seen.append(copy.deepcopy(env))
            name,args=('acquire_sources',READ) if handoff and len(seen)==1 else ('submit_final_answer',final_args())
            async for event in action(name,args,call_id='native-'+str(len(seen))):yield event
        bridge=NativeSelectorProvider(client,req,provider,authorize=manager.check_disclosure)
        result=await manager.run(QUESTION,model=bridge)
        assert result['answer']==ANSWER
        assert resolver.reads==1 and client.calls==1
        assert len(seen)==(2 if handoff else 1)
        assert manager.context.source_batch_state.results_by_source_id[1].status.value=='ok'
        assert any(a.capability_id=='health.query' and a.outcome=='authorized' for a in manager.broker.attempts)
        assert bridge.receipt['status']==('native_planner' if handoff else 'native_acquisition_proposed')
        assert '57' in json.dumps(seen[-1]['request']['messages'])
    asyncio.run(run())


@pytest.mark.parametrize('mode',['handoff','schema','question','history','forced','long','network'])
def test_fallback_preserves_whole_native_context_without_source_io(mode):
    async def run():
        manager,resolver,_=fixture();req=request();p=plan(req,handoff=mode=='handoff')
        original=envelope(manager,req)
        original['request']['messages'][0]['content'][0]['text']+='\nInventory: labs partial; profile withheld; calendar unknown.'
        if mode=='schema':
            tool=next(t for t in original['request']['tools'] if t['name']=='acquire_sources')
            tool['parameters']['properties']['health_reads']['items']['properties']['concepts']['items']['enum']=[]
        if mode=='question':original['request']['messages'][-1]['content'][0]['text']='Compare sources and count workouts.'
        if mode=='history':req=req.model_copy(update={'state':req.state.model_copy(update={'recent_user_requests':['Wrong history']})})
        if mode=='forced':original['controls']['toolChoice']='submit_final_answer'
        if mode=='long':
            long='Analyze every aspect. '*100
            original['request']['messages'][-1]['content'][0]['text']=long
            req=req.model_dump();req['state']['current_request']=long
        client=Client(p)
        if mode=='network':
            def fail(_):raise RuntimeError('SYNTHETIC_PRIVATE_EXCEPTION')
            client.select=fail
        seen=[]
        async def provider(env):
            seen.append(env)
            yield {'type':'finish','reason':{'kind':'stop'}}
        bridge=NativeSelectorProvider(client,req,provider,authorize=manager.check_disclosure)
        frozen=copy.deepcopy(original)
        events=[e async for e in bridge(original)]
        assert seen==[frozen] and seen[0] is original
        assert original==frozen and resolver.reads==0
        assert bridge.receipt['status']=='native_planner'
        assert 'SYNTHETIC_PRIVATE_EXCEPTION' not in json.dumps(bridge.receipt)
        if mode == 'network': assert bridge.receipt['reason_code'] == 'selector_failed'
        assert events==[{'type':'finish','reason':{'kind':'stop'}}]
    asyncio.run(run())


def test_revocation_after_selection_does_not_emit_proposal_or_call_provider():
    async def run():
        manager,resolver,_=fixture();req=request();checks=[];provider_calls=[]
        async def authorize():
            checks.append(1)
            if len(checks)>1:raise PermissionError('revoked')
        async def provider(env):
            provider_calls.append(env)
            yield {}
        bridge=NativeSelectorProvider(Client(plan(req)),req,provider,authorize=authorize)
        with pytest.raises(PermissionError):[e async for e in bridge(envelope(manager,req))]
        assert resolver.reads==0 and not provider_calls
    asyncio.run(run())


def test_relative_window_uses_native_arithmetic_and_exact_trusted_clock():
    manager,_,_=fixture();req=request();p=plan(req)
    p=p.model_copy(update={'queries':[HealthRead(metrics=['heart_rate_variability'],
        period={'kind':'relative','amount':7,'unit':'days'})]})
    schema=next(t['parameters'] for t in manager.state()['tools'] if t['name']=='acquire_sources')
    assert native_batch(p,req,schema)['health_reads'][0]['range']=={
        'kind':'between','start_at':'2026-09-17T12:00:00+00:00','end_at':'2026-09-24T12:00:00+00:00'}


@pytest.mark.parametrize('date_basis',['next_due_date','last_done_date'])
@pytest.mark.parametrize('period,first_day,metric_bounds',[
    ({'kind':'relative','amount':7,'unit':'days'},'2026-03-23',
     ('2026-03-22T23:30:00+00:00','2026-03-29T23:30:00+00:00')),
    ({'kind':'calendar','period':'month'},'2026-03-01',
     ('2026-03-01T00:00:00+02:00','2026-03-30T02:30:00+03:00')),
    ({'kind':'calendar','period':'day'},'2026-03-30',
     ('2026-03-30T00:00:00+03:00','2026-03-30T02:30:00+03:00')),
])
def test_frozen_calendar_window_matches_native_executor_whole_local_days(period,first_day,metric_bounds,date_basis):
    from vita_agent.health.calendar_dates import select_calendar_dates
    from vita_agent.health.query_tool import build_health_query_executor, HealthOperationPublication
    from vita_agent.kernel.health_range_input import normalize_health_ranges
    from vita_agent.kernel.source_batch_contracts import SourceBatchRequest

    async def run():
        # This UTC instant is the next local day, after the Bucharest DST change.
        payload=request().model_dump()
        payload.update(reference_time=datetime.fromisoformat('2026-03-29T23:30:00+00:00'),
                       available_record_types=['calendar'])
        payload['state']['reference_date']='2026-03-30'
        req=QueryRequest.model_validate(payload)
        p=plan(req).model_copy(update={'queries':[
            HealthRead(records=['calendar'],operation='latest',period=period,date_basis=date_basis),
            HealthRead(metrics=['heart_rate_variability'],period=period)]})
        manager,_,_=fixture()
        schema=next(t['parameters'] for t in manager.state()['tools'] if t['name']=='acquire_sources')
        original=compile_batch(p,req)['health_reads'][0]
        frozen=native_batch(p,req,schema)['health_reads']
        last_day='2026-03-30'
        days=[(date.fromisoformat(first_day)-timedelta(days=1)).isoformat(),first_day,
              last_day,(date.fromisoformat(last_day)+timedelta(days=1)).isoformat()]
        other_basis='last_done_date' if date_basis=='next_due_date' else 'next_due_date'
        records=[{'id':key,date_basis:day,other_basis:'1900-01-01'}
                 for key,day in zip(('before','first','last','after'),days,strict=True)]

        class CalendarResolver:
            def __init__(self):self.windows=[];self.selected=[]
            async def get_snapshot_records(self,record_type,*,start,end,calendar_date_basis,calendar_time_zone,**kwargs):
                assert record_type=='calendar' and calendar_date_basis==date_basis
                self.windows.append((start,end))
                selection=select_calendar_dates(records,date_basis=calendar_date_basis,
                    start_at=start,end_at=end,time_zone=calendar_time_zone)
                self.selected.append({row['id'] for row in selection.records})
                return {'status':'ok','records':list(selection.records),
                        'scope_complete':True,'detail_complete':True}

        resolver=CalendarResolver()
        executor=build_health_query_executor(resolver,authorized_metrics=(),coverage=None,
                                            clock=lambda:req.reference_time)
        for operation in (original,frozen[0]):
            batch=normalize_health_ranges({'health_reads':[operation],'required_operation_ids':[1]})
            native=SourceBatchRequest.model_validate(batch).health_reads[0]
            result=await executor.execute(request=native,publication=HealthOperationPublication())
            assert result['records']['calendar']['status']=='ok',(operation['range'],result['records']['calendar'])
        assert resolver.selected==[{'last','first'},{'last','first'}]
        assert resolver.windows[0]==resolver.windows[1]
        zone=ZoneInfo(req.state.time_zone)
        start,end=(value.astimezone(zone) for value in resolver.windows[1])
        assert start.time()==time.min and end.time()==time.max
        assert (start.date().isoformat(),end.date().isoformat())==(first_day,last_day)
        # Even a current-day query includes dates for events later today; the
        # calendar end cannot be truncated to the trusted clock's 02:30 time.
        assert end>req.reference_time
        if first_day!=last_day:
            assert start.utcoffset()==timedelta(hours=2)
            assert end.utcoffset()==timedelta(hours=3)
        assert frozen[0]['range']=={'kind':'between','start_at':first_day,'end_at':last_day}
        assert frozen[1]['range']=={'kind':'between','start_at':metric_bounds[0],'end_at':metric_bounds[1]}
    asyncio.run(run())


def test_tampered_request_binding_is_rejected_before_native_dispatch():
    manager,_,_=fixture();req=request();p=plan(req).model_copy(update={'request_sha256':'3'*64})
    schema=next(t['parameters'] for t in manager.state()['tools'] if t['name']=='acquire_sources')
    with pytest.raises(ValueError):native_batch(p,req,schema)


def bootstrap_messages(manager):
    from types import SimpleNamespace
    from backbone.session_bridge import items_to_messages
    from vita_agent.kernel.context_assembler import _context_pair
    turn = SimpleNamespace(turn_id='synthetic-turn')
    items = []
    for name, kind, schema in [('vita_profile_planning', 'profile_planning', 'vita-profile-planning/v1'),
                               ('vita_skc', 'skc', 'vita-skc/v1')]:
        items.extend(_context_pair(turn=turn, kind=kind, index=0, name=name,
                                   payload={'schema': schema}))
    return items_to_messages(items, model_id=manager.model_id)


def test_real_session_policy_bootstrap_round_reaches_selector_and_native_broker():
    from agents import RunConfig
    from backbone.session_bridge import SessionInputPolicy
    async def run():
        manager, resolver, _ = fixture()
        req = request(); client = Client(plan(req)); seen = []
        policy = SessionInputPolicy(manager, RunConfig(tracing_disabled=True))
        bootstrap = bootstrap_messages(manager)
        async def provider(env):
            seen.append(env)
            async for event in action('submit_final_answer', final_args(), call_id='bootstrap-final'):
                yield event
        bridge = NativeSelectorProvider(client, req, provider, authorize=manager.check_disclosure)
        first_round = []
        async def record(env):
            if not first_round:
                first_round.extend(copy.deepcopy(env['request']['messages']))
            async for event in bridge(env):
                yield event
        result = await manager.run(QUESTION, model=record, initial_session=bootstrap, input_policy=policy)
        question_index = next(i for i, m in enumerate(first_round)
                              if m['content'] == [{'type':'text','text':QUESTION}])
        assert question_index > 0
        assert sum(b.get('name') in {'vita_skc','vita_profile_planning'}
                   for m in first_round for b in m['content']) == 2
        assert result['answer'] == ANSWER and resolver.reads == 1
        assert client.calls == 1 and len(seen) == 1
        assert bridge.receipt['status'] == 'native_acquisition_proposed'
        assert bridge.receipt['reason_code'] is None
    asyncio.run(run())
