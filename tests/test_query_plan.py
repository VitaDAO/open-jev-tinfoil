import asyncio,copy
from datetime import UTC,datetime
from threading import Lock
import httpx,pytest
from query_plan import QueryRequest,QueryPlan,HealthRead,ResearchRead,compile_batch,request_identity
from query_execution import execute_plan,project_sleep
from routing import MODEL_REVISION
from examples.vita_client import VitaClient

NOW=datetime(2026,9,23,12,tzinfo=UTC)
def request(**overrides):
 return QueryRequest.model_validate({'schema_version':'vita-selector/v2','state':{'current_request':'Show my steps yesterday','reference_date':'2026-09-23','time_zone':'UTC'},'reference_time':NOW,'available_metrics':['steps','total_sleep','sleep_efficiency','ldl_cholesterol','oxygen_saturation'],'available_record_types':['profile','labs','workouts','calendar'],'available_sources':['oura','garmin'],'literature_available':True,**overrides})
def plan(queries=None,req=None,**overrides):
 return QueryPlan(status='planned',queries=queries or [HealthRead(metrics=['steps'],period={'kind':'all_history'})],time_zone='UTC',selector_sha256='1'*64,adapter_sha256='2'*64,model_revision=MODEL_REVISION,request_sha256=request_identity(req or request()),**overrides)
def run(payloads,p=None,req=None,**kwargs):
 calls=[]
 async def callback(kind,op):
  calls.append((kind,op));return copy.deepcopy(payloads[min(len(calls)-1,len(payloads)-1)])
 result=asyncio.run(execute_plan(p or plan(),req or request(),callback,**kwargs))
 return result,calls

def metric_payload(**series):
 return {'status':'ok','coverage':{'scope_complete':True},'claim_scope':{'purpose':'trend'},'series':[{'concept':'steps','source':'oura','scope_complete':True,'points':[['2026-09-22T12:00:00Z',1234]],'grouping':'none',**series}]}

def test_request_and_inventory_binding_before_any_io():
 for req in [request(available_sources=[]),request(reference_time=NOW.replace(hour=11))]:
  with pytest.raises(ValueError,match='binding'):compile_batch(plan(),req)
 bad=plan().model_copy(update={'time_zone':'Europe/Bucharest'})
 with pytest.raises(ValueError,match='binding'):compile_batch(bad,request())

@pytest.mark.parametrize('change',[
 {'reference_time':'2026-09-23T12:00:00'},
 {'reference_time':'2026-09-22T12:00:00Z'},
 {'available_metrics':['steps','steps']},
 {'available_sources':['oura','oura']},
 {'available_sources':['arbitrary secret text']},
 {'literature_available':'true'},
 {'schema_version':'vita-selector/v1'},
 {'health_records':'private-sentinel'},
])
def test_closed_request(change):
 with pytest.raises(ValueError):request(**change)

@pytest.mark.parametrize('change',[
 {'period':{'kind':'auto'}},
 {'period':{'kind':'between','start_at':'2025-02-29','end_at':'2025-02-29'}},
 {'period':{'kind':'between','start_at':'2026-09-23T11:00:00','end_at':'2026-09-23T12:00:00'}},
 {'period':{'kind':'between','start_at':'2026-10-01','end_at':'2026-09-01'}},
 {'limit':5}, {'operation':'count'}, {'date_basis':'exam_date'},
 {'records':['labs']}, {'date_basis':'sleep_end_day'}, {'metrics':['steps','steps']},
])
def test_unrepresentable_query_rejected(change):
 with pytest.raises(ValueError):HealthRead.model_validate({'metrics':['steps'],'period':{'kind':'all_history'},**change})


def test_canonical_concepts_and_source_inventory():
 p=plan([HealthRead(metrics=['oxygen_saturation','ldl_cholesterol'],period={'kind':'all_history'})])
 assert compile_batch(p,request())['health_reads'][0]['concepts']==['spo2','ldl']
 for q in [HealthRead(metrics=['ldl_cholesterol'],period={'kind':'all_history'},source='oura'),HealthRead(metrics=['unknown'],period={'kind':'all_history'})]:
  with pytest.raises(ValueError):compile_batch(plan([q]),request())


def test_atomic_operation_budget_and_disabled_research():
 q=HealthRead(metrics=['steps'],period={'kind':'all_history'})
 with pytest.raises(ValueError):plan([q]*9)
 r=request(literature_available=False)
 with pytest.raises(ValueError):compile_batch(plan([ResearchRead(targets=['sleep'])],req=r),r)
 assert 'private' not in ResearchRead(targets=['sleep']).question()


@pytest.mark.parametrize('size',[7,12,13,48])
def test_large_metric_selection_preserves_every_metric_with_six_per_native_read(size):
 metrics=[f'synthetic_metric_{i}' for i in range(size)]
 req=request(available_metrics=metrics)
 p=plan([HealthRead(metrics=metrics,period={'kind':'relative','amount':6,'unit':'months'})],req=req)
 batch=compile_batch(p,req);reads=batch['health_reads']
 assert [metric for op in reads for metric in op['concepts']]==metrics
 assert all(1<=len(op['concepts'])<=6 for op in reads)
 assert len(reads)==(size+5)//6
 assert batch['required_operation_ids']==list(range(1,len(reads)+1))
 assert all(op['range']=={'kind':'relative','amount':6,'unit':'months'} for op in reads)


@pytest.mark.parametrize('size',[7,13])
def test_execution_keeps_chunked_metrics_and_following_research_bound_to_their_queries(size):
 metrics=[f'synthetic_metric_{i}' for i in range(size)]
 req=request(available_metrics=metrics)
 p=plan([HealthRead(metrics=metrics,period={'kind':'all_history'}),ResearchRead(targets=['sleep'])],req=req)
 calls=[]
 async def callback(kind,op):
  calls.append((kind,op))
  if kind=='literature_reads':return {'status':'ok','sources':[]}
  return {'status':'ok','coverage':{'scope_complete':True},'claim_scope':{'purpose':'trend'},
   'series':[{'concept':metric,'source':'manual','scope_complete':True} for metric in op['concepts']]}
 result=asyncio.run(execute_plan(p,req,callback));chunks=(size+5)//6
 assert [metric for kind,op in calls if kind=='health_reads' for metric in op['concepts']]==metrics
 assert [o['query_index'] for o in result['operations']]==[0]*chunks+[1]
 assert [o['status'] for o in result['operations']]==['complete']*chunks+['acquired']
 assert calls[-1][0]=='literature_reads' and len(calls)==chunks+1
 assert result['status']=='needs_evidence_review' and not result['all_requested_delivered']


def test_ninth_compiled_operation_rejects_the_whole_plan_before_any_io():
 metrics=[f'synthetic_metric_{i}' for i in range(49)]
 req=request(available_metrics=metrics)
 p=plan([HealthRead(metrics=metrics,period={'kind':'all_history'})],req=req)
 calls=[]
 async def callback(*args):
  calls.append(args)
  raise AssertionError('Over-budget plan must not execute even its first chunk')
 with pytest.raises(ValueError,match='Operation budget exceeded'):compile_batch(p,req)
 with pytest.raises(ValueError,match='Operation budget exceeded'):
  asyncio.run(execute_plan(p,req,callback))
 assert calls==[]


def test_metric_evidence_and_missing_source_are_not_silently_delivered():
 result,calls=run([metric_payload()]);assert result['all_requested_delivered'] and len(calls)==1
 q=HealthRead(metrics=['steps'],period={'kind':'all_history'},source='oura')
 bad=[metric_payload(concept='apob'),metric_payload(source='garmin'),metric_payload(scope_complete=False),{'status':'empty','coverage':{'scope_complete':True},'claim_scope':{'purpose':'trend'},'series':[]}]
 for payload in bad:
  result,_=run([payload],plan([q]));assert result['status']=='incomplete'
 result,_=run([{**metric_payload(),'gaps':[{'code':'missing'}]}]);assert not result['all_requested_delivered']


def test_latest_requires_raw_single_point_per_source():
 q=HealthRead(metrics=['steps'],operation='latest',period={'kind':'all_history'})
 for payload in [metric_payload(grouping='week'),metric_payload(points=[['a',1],['b',2]])]:
  payload['claim_scope']['purpose']='latest'
  assert run([payload],plan([q]))[0]['status']=='incomplete'


def sleep_fixture():
 q=HealthRead(metrics=['total_sleep'],period={'kind':'between','start_at':'2026-09-23','end_at':'2026-09-23'},date_basis='sleep_end_day')
 p={'status':'ok','coverage':{'scope_complete':True,'detail_complete':True},'series':[{'concept':'total_sleep','source':'oura','unit':'hours','scope_complete':True,'detail_complete':True,'grouping':'none','points':[['2026-09-22T20:00:00Z',7],['2026-09-23T20:00:00Z',8]],'sleep_episodes':[{'schema_version':1,'start_at':'2026-09-22T23:00:00Z','end_at':'2026-09-23T06:00:00Z'},{'schema_version':1,'start_at':'2026-09-23T23:00:00Z','end_at':'2026-09-24T07:00:00Z'}],'statistics':{'mean':7.5}}]}
 return q,p


def test_sleep_uses_episode_time_not_recorded_day_and_drops_old_statistics():
 q,p=sleep_fixture();result=project_sleep(q,p,'UTC')
 assert result['series'][0]['points']==[['2026-09-22T20:00:00Z',7]]
 assert 'statistics' not in result['series'][0]
 assert run([p],plan([q]))[0]['status']=='complete'

@pytest.mark.parametrize('bad',['missing','naive','reversed','boolean_version','nap','multiple','incomplete'])
def test_sleep_requires_unambiguous_explicit_episode_metadata(bad):
 q,p=sleep_fixture();s=p['series'][0]
 if bad=='missing':s['sleep_episodes']=[]
 if bad=='naive':s['sleep_episodes'][0]['start_at']='2026-09-22T23:00:00'
 if bad=='reversed':s['sleep_episodes'][0]['start_at']='2026-09-24T23:00:00Z'
 if bad=='boolean_version':s['sleep_episodes'][0]['schema_version']=True
 if bad=='nap':s['sleep_episodes'][0]['start_at']='2026-09-23T05:00:00Z'
 if bad=='multiple':s['sleep_episodes'][1]=copy.deepcopy(s['sleep_episodes'][0])
 if bad=='incomplete':p['coverage']['detail_complete']=False
 assert run([p],plan([q]))[0]['status']=='incomplete'


def test_partial_multi_clause_never_reports_all_delivered():
 q=HealthRead(metrics=['total_sleep'],period={'kind':'all_history'})
 result,calls=run([metric_payload(),{'status':'unavailable'}],plan([HealthRead(metrics=['steps'],period={'kind':'all_history'}),q]))
 assert result['status']=='incomplete' and len(calls)==2 and result['operations'][0]['status']=='complete'


def test_client_validates_exact_request_closed_schema_and_pins():
 client=VitaClient.__new__(VitaClient);client.token='synthetic';client.selector_sha256='1'*64;client.adapter_sha256='2'*64
 client._select_lock=Lock()
 good=plan().model_dump(mode='json')
 variants=[{**good,'selector_sha256':'3'*64},{**good,'adapter_sha256':'3'*64},{**good,'request_sha256':'3'*64},{**good,'time_zone':'Europe/Bucharest'},{**good,'advisory':False},{**good,'advisory':1},{**good,'status':'handoff'},{**good,'answers':{}},{**good,'queries':[]},
           {k:v for k,v in good.items() if k!='schema_version'},{k:v for k,v in good.items() if k!='advisory'}]
 for value in variants:
  with httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=value))) as http:
   client.http=http
   with pytest.raises((ValueError,RuntimeError)):client.select(request())
 with httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=good))) as http:
  client.http=http;assert client.select(request()).model_dump(mode='json')==good


def test_client_handoff_is_inert():
 p=plan().model_copy(update={'status':'handoff','queries':[],'reason_codes':['unsupported']})
 result,calls=run([],p);assert result['status']=='handoff' and calls==[]


def test_one_endpoint_auth_bounds_identity_and_unsupported_old_format(monkeypatch):
 from fastapi.testclient import TestClient
 from server import create_app
 class Engine:
  def select(self,req):
   if req.state.current_request=='overflow':raise ValueError('private-sentinel')
   return plan(req=req).model_dump(mode='json')
 token='synthetic-test-token-'*3;headers={'Authorization':'Bearer '+token};body=request().model_dump(mode='json')
 with TestClient(create_app(Engine,token,selector_enabled=True)) as client:
  assert client.post('/v1/select',json=body).status_code==401
  assert client.post('/v1/select',json=body,headers=headers).status_code==200
  assert client.post('/v1/select-baseline',json=body,headers=headers).status_code==404
  assert client.post('/v1/select/v2',json=body,headers=headers).status_code==404
  assert client.get('/health').json()['selector_schema']=='vita-selector/v2'
  for change in [{'health_records':'private-sentinel'},{'schema_version':'vita-selector/v1'},
                 {'state':{**body['state'],'current_request':'x'*1201}},
                 {'state':{**body['state'],'recent_user_requests':['a']*5}},
                 {'state':{**body['state'],'current_request':'overflow'}}]:
   response=client.post('/v1/select',json={**body,**change},headers=headers)
   assert response.status_code==422 and 'private-sentinel' not in response.text
  assert client.post('/v1/select',content=b'x'*65537,headers=headers).status_code==413
 monkeypatch.delenv('ENABLE_EXPERIMENTAL_SELECTOR',raising=False)
 with TestClient(create_app(Engine,token)) as client:
  assert client.post('/v1/select',json=body,headers=headers).status_code==503


def test_untrusted_callback_exceptions_never_echo_values():
 async def callback(*_):raise ValueError('SyntheticPrivateValue')
 result=asyncio.run(execute_plan(plan(),request(),callback))
 assert result['operations'][0]['reason']=='invalid_source_payload'
 assert 'SyntheticPrivateValue' not in str(result)


def test_research_acquisition_requires_existing_evidence_gate():
 result,_=run([{'status':'ok','sources':[]}],plan([ResearchRead(targets=['sleep'])]))
 assert result['status']=='needs_evidence_review' and not result['all_requested_delivered']


def test_validated_latest_fact_projects_out_history_and_statistics():
 q=HealthRead(metrics=['steps'],operation='latest',period={'kind':'all_history'})
 payload=metric_payload(points=[['2026-09-21T12:00:00Z',2000],['2026-09-22T12:00:00Z',1000]],unit='steps',statistics={'mean':1500})
 payload['claim_scope']['purpose']='latest'
 payload['facts']=[{'concept':'steps','source':'oura','unit':'steps','value':1000,'observed_at':'2026-09-22T12:00:00Z','evidence_ref':'synthetic-latest'}]
 result,_=run([payload],plan([q]));value=result['operations'][0]['result']
 assert value['series'][0]['points']==[['2026-09-22T12:00:00Z',1000]]
 assert 'statistics' not in value['series'][0]
 payload['facts'][0]['value']=2000
 assert run([payload],plan([q]))[0]['status']=='incomplete'

@pytest.mark.parametrize('record,basis',[('labs','exam_date'),('calendar','next_due_date'),('calendar','last_done_date')])
def test_record_date_contract_rejects_subday_bounds(record,basis):
 with pytest.raises(ValueError,match='whole days'):
  HealthRead(records=[record],operation='latest',date_basis=basis,period={'kind':'between','start_at':'2026-01-01','end_at':'2026-09-23T12:00:00Z'})

@pytest.mark.parametrize('gap',['gaps','denied','scan_continuation_ref'])
def test_later_record_page_cannot_hide_a_source_gap(gap):
 q=HealthRead(records=['labs'],operation='latest',date_basis='exam_date',period={'kind':'all_history'})
 first={'status':'ok','records':{'labs':{'status':'ok','scope_complete':True,'record_details':[{'lab_provider':'synthetic-first'}],'record_continuation_token':'page-2','detail_page_offset':0}}}
 final={'status':'ok',gap:['synthetic-gap'],'records':{'labs':{'status':'ok','scope_complete':True,'record_details':[{'lab_provider':'synthetic-second'}],'detail_page_offset':1,'more_detail_available':False}}}
 result,calls=run([first,final],plan([q]))
 assert len(calls)==2 and result['status']=='incomplete'
 assert result['operations'][0]['reason']=='source_scope_incomplete'
 assert 'result' not in result['operations'][0]

@pytest.mark.parametrize('qualifier',['in London time, not my default timezone','from 9am to 5pm','at 09:30',
 'at 9 a.m.','before noon','after midnight','in the morning','hourly','using UTC','in Europe/Bucharest'])
def test_unrepresented_clocks_are_rejected_before_inference(qualifier):
 from query_selector import QuerySelector
 selector=QuerySelector.__new__(QuerySelector)
 with pytest.raises(ValueError,match='clock_or_timezone'):
  selector.health('show my steps yesterday '+qualifier.lower(),request(),{})

def test_features_are_reused_only_inside_one_request_and_cleared_on_error(monkeypatch):
 from query_selector import QuerySelector,_FEATURES
 from learned_selector import LearnedSelector
 calls=[];scopes=[]
 def encode(self,text):calls.append(text);return object()
 monkeypatch.setattr(LearnedSelector,'encode',encode)
 selector=QuerySelector.__new__(QuerySelector)
 def inner(req):
  scopes.append(_FEATURES.get()[1])
  first=selector.encode('synthetic text')
  assert selector.encode('synthetic text') is first
  assert selector.encode('different text') is not first
  raise RuntimeError('synthetic failure')
 monkeypatch.setattr(selector,'_select_query',inner)
 for _ in range(2):
  with pytest.raises(RuntimeError):selector.select_query(request())
  assert _FEATURES.get() is None and scopes[-1]=={}
 assert calls==['synthetic text','different text']*2

def test_request_features_are_isolated_between_concurrent_calls(monkeypatch):
 from concurrent.futures import ThreadPoolExecutor
 from threading import Barrier
 from query_selector import QuerySelector,_FEATURES
 from learned_selector import LearnedSelector
 barrier=Barrier(2)
 monkeypatch.setattr(LearnedSelector,'encode',lambda *_:object())
 selector=QuerySelector.__new__(QuerySelector)
 def inner(req):
  feature=selector.encode('same synthetic text');barrier.wait(timeout=5)
  assert selector.encode('same synthetic text') is feature
  return feature
 monkeypatch.setattr(selector,'_select_query',inner)
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(selector.select_query,request()) for _ in range(2)]
  assert futures[0].result() is not futures[1].result()
 assert _FEATURES.get() is None

def test_profile_review_regressions_exact_scope_and_read_intent():
 import json
 from pathlib import Path
 from query_selector import QuerySelector
 selector=QuerySelector.__new__(QuerySelector)
 for case in json.loads((Path(__file__).parents[1]/'evidence/selector-v5/review-profile-16.json').read_text()):
  req=request(state={'current_request':case['request'],'reference_date':'2026-09-23','time_zone':'UTC'})
  result=selector.select_query(req)
  assert (result['queries'] if result['status']=='planned' else None)==case['expected_queries'],case['id']

@pytest.mark.parametrize('text',['show unknownmarker, steps and total sleep','show steps, unknownmarker and total sleep',
 'show steps, total sleep and unknownmarker','show steps and an unknown biomarker'])
def test_unknown_metric_list_items_never_reach_model_as_partial_reads(text):
 from query_selector import QuerySelector
 selector=QuerySelector.__new__(QuerySelector)
 with pytest.raises(ValueError,match='unbound_list_item'):selector.health(text,request(),{})

def test_common_hscrp_alias_preserves_all_three_requested_markers():
 from query_selector import enhance
 from proposal_binding import entities
 text=enhance('What are my fasting glucose, HbA1c, and hsCRP?')
 assert {v for _,_,values in entities(text,['fasting_glucose','hba1c','hs_crp']) for kind,v in values if kind=='metric'}=={'fasting_glucose','hba1c','hs_crp'}

@pytest.mark.parametrize('text',['Show my profile and medications','Show my profile and allergies','Show my allergies and my profile'])
def test_additive_profile_scope_is_not_reduced_to_a_named_field(text):
 from query_selector import QuerySelector
 selector=QuerySelector.__new__(QuerySelector)
 req=request(state={'current_request':text,'reference_date':'2026-09-23','time_zone':'UTC'})
 result=selector.select_query(req)
 assert result['status']=='planned' and result['queries'][0]['profile_fields']==['all']

@pytest.mark.parametrize('exception',[TimeoutError,RuntimeError,OSError,httpx.ReadTimeout])
def test_operational_failures_are_incomplete_without_losing_earlier_reads(exception):
 calls=[]
 async def callback(*_):
  calls.append(1)
  if len(calls)==2:raise exception('synthetic-private-sentinel')
  return metric_payload()
 q=HealthRead(metrics=['steps'],period={'kind':'all_history'})
 result=asyncio.run(execute_plan(plan([q,q]),request(),callback))
 assert result['status']=='incomplete' and not result['all_requested_delivered']
 assert result['operations'][0]['status']=='complete'
 assert result['operations'][1]['reason']=='source_operation_failed'
 assert 'synthetic-private-sentinel' not in str(result)

def test_cancellation_is_never_converted_into_an_incomplete_read():
 async def callback(*_):raise asyncio.CancelledError()
 with pytest.raises(asyncio.CancelledError):asyncio.run(execute_plan(plan(),request(),callback))



def test_digit_leading_native_inventory_is_preserved_and_still_authorized():
 metric='25_hydroxy_vitamin_d'
 req=request(available_metrics=[metric,'steps'])
 assert req.available_metrics==[metric,'steps']
 proposal=plan([HealthRead(metrics=[metric],period={'kind':'all_history'})],req=req)
 assert compile_batch(proposal,req)['health_reads'][0]['concepts']==[metric]
 narrowed=request(available_metrics=['steps'])
 with pytest.raises(ValueError):
  compile_batch(plan([HealthRead(metrics=[metric],period={'kind':'all_history'})],req=narrowed),narrowed)


@pytest.mark.parametrize('metric',['_leading','has-dash','has space','../path','a'*97,'','µmol'])
def test_metric_identifier_boundaries_remain_closed(metric):
 with pytest.raises(ValueError):request(available_metrics=[metric])
