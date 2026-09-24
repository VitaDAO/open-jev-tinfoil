"""Actual Vita query executor with synthetic source rows; no DB or provider I/O."""
import asyncio,hashlib,inspect,json,os,sys
from datetime import UTC,datetime,timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID
ROOT=Path(__file__).resolve().parents[1]
VITA=Path(os.environ['VITA_SOURCE_ROOT']).resolve()
sys.path[:0]=[str(ROOT),str(VITA/'vita/py/src')]
from query_plan import QueryRequest,QueryPlan,HealthRead,request_identity,compile_batch
from query_execution import execute_plan
from routing import MODEL_REVISION
from vita_agent.health.query_tool import HealthOperationPublication,build_health_query_executor
from vita_agent.health.resolver import HealthResolver
from vita_agent.health.turn_materialization import LiveHealthTurnMaterialization,MaterializationRecordsAdapter,DOMAIN_DATASETS
from vita_agent.kernel.source_batch_contracts import SourceBatchRequest
from vita_agent.kernel.health_range_input import normalize_health_ranges
assert str(VITA) in inspect.getfile(SourceBatchRequest)
NOW=datetime(2026,9,23,12,tzinfo=UTC)
class NoDirectStore:
 async def query_wearable_readings(self,**kwargs):raise AssertionError('Direct store access forbidden')
 async def query_biomarkers(self,**kwargs):raise AssertionError('Direct store access forbidden')
REQUEST=QueryRequest.model_validate({'schema_version':'vita-selector/v2','state':{'current_request':'Synthetic executor check','reference_date':'2026-09-23','time_zone':'UTC'},'reference_time':NOW,'available_metrics':['steps','total_sleep','sleep_efficiency'],'available_sources':['oura','garmin'],'available_record_types':['profile','labs','workouts','calendar']})

def plan(query):
 return QueryPlan(status='planned',queries=[query],time_zone='UTC',selector_sha256='1'*64,adapter_sha256='2'*64,model_revision=MODEL_REVISION,request_sha256=request_identity(REQUEST))

async def run(query,records_by_domain,*,exhausted=True,max_pages=8):
 calls=[]
 async def backend(**kwargs):
  rows=next((v for k,v in records_by_domain.items() if DOMAIN_DATASETS[k]==kwargs['dataset_id']),[])
  if kwargs.get('record_types'):rows=[r for r in rows if r.get('record_type') in kwargs['record_types']]
  if kwargs.get('source'):rows=[r for r in rows if r.get('source')==kwargs['source']]
  return {'status':'ok','records':rows,'rows_scanned':len(rows),'rejected_rows':0,'exhausted':exhausted}
 async def fence():return {'source_revision':1,'as_of':NOW.isoformat()}
 materialization=await LiveHealthTurnMaterialization.open_live(owner_user_id=UUID('11111111-1111-4111-8111-111111111111'),authority_scope_digest='synthetic',granted_domains=frozenset(records_by_domain),slice_reader=backend,fence_reader=fence)
 reader=MaterializationRecordsAdapter(materialization,SimpleNamespace())
 executor=build_health_query_executor(HealthResolver(NoDirectStore(),records_reader=reader),authorized_metrics=REQUEST.available_metrics,coverage=None,clock=lambda:NOW)
 async def callback(kind,op):
  calls.append(op)
  batch=SourceBatchRequest.model_validate(normalize_health_ranges({'health_reads':[op],'required_operation_ids':[op['operation_id']]}))
  return await executor.execute(request=batch.health_reads[0],publication=HealthOperationPublication())
 try:return await execute_plan(plan(query),REQUEST,callback,max_pages=max_pages),calls
 finally:await materialization.aclose()

async def main():
 rows=[]
 async def check(name,query,data,expected,**kw):
  result,calls=await run(query,data,**kw)
  expected(result)
  rows.append({'id':name,'status':result['status'],'callback_calls':len(calls),'passed':True})
  return result,calls
 def complete(result):assert result['all_requested_delivered'],result
 def incomplete(result):assert not result['all_requested_delivered'],result
 labs=[{'record_id':str(i),'payload':{'exam_date':(NOW-timedelta(days=i)).date().isoformat(),'lab_provider':f'Synthetic-{i}'}} for i in range(405)]
 latest=HealthRead(records=['labs'],operation='latest',period={'kind':'all_history'},date_basis='exam_date',limit=5)
 result,_=await check('latest-five-of-405',latest,{'lab_uploads':labs},complete)
 assert [r['lab_provider'] for r in result['operations'][0]['result']['record_details']]==[f'Synthetic-{i}' for i in range(5)]
 whole=latest.model_copy(update={'limit':None})
 result,calls=await check('all-405-across-three-pages',whole,{'lab_uploads':labs},complete)
 assert len(calls)==3
 actual=[r['lab_provider'] for r in result['operations'][0]['result']['record_details']]
 assert actual==[f'Synthetic-{i}' for i in range(405)]
 assert all({k:v for k,v in c.items() if k!='record_continuation_token'}==calls[0] for c in calls)
 await check('page-budget-exhausted',whole,{'lab_uploads':labs},incomplete,max_pages=1)
 await check('source-scan-incomplete',latest,{'lab_uploads':labs},incomplete,exhausted=False)
 await check('missing-exam-date',latest,{'lab_uploads':[{'record_id':'undated','payload':{'lab_provider':'Synthetic'}}]},incomplete)
 result,_=await check('empty-is-not-widened',latest.model_copy(update={'period':{'kind':'between','start_at':'2020-01-01','end_at':'2020-12-31'}}),{'lab_uploads':labs},complete)
 assert result['operations'][0]['result']['record_details']==[]
 profile=HealthRead(records=['profile'],operation='latest',period={'kind':'all_history'},date_basis='current_snapshot',profile_fields=['all'])
 profiles={'profile_health':[{'record_id':'profile','payload':{'goals':['synthetic-goal'],'allergies':[],'smoking_status':'never','unknown_private_field':'must not appear'}}],'profile_preferences':[],'biomarkers':[]}
 result,_=await check('whole-profile-explicit-missing-fields',profile,profiles,complete)
 value=result['operations'][0]['result'];assert len(value['requested_fields'])==23
 assert 'smoking_status' in value['field_scope']['returned'] and 'medications' in value['field_scope']['requested_but_not_returned']
 assert 'unknown_private_field' not in json.dumps(value)
 result,_=await check('narrow-profile-no-overread',profile.model_copy(update={'profile_fields':['allergies']}),profiles,complete)
 assert all('goals' not in r for r in result['operations'][0]['result']['records'])
 calendar=HealthRead(records=['calendar'],operation='latest',period={'kind':'between','start_at':'2026-09-24','end_at':'2026-09-24'},date_basis='next_due_date')
 result,_=await check('calendar-due-not-completion-date',calendar,{'health_calendar_events':[
  {'record_id':'a','payload':{'name':'Synthetic due','next_due_date':'2026-09-24','last_done_date':'2026-01-01'}},
  {'record_id':'b','payload':{'name':'Synthetic done','next_due_date':'2026-10-01','last_done_date':'2026-09-24'}}]},complete)
 assert [r['name'] for r in result['operations'][0]['result']['record_details']]==['Synthetic due']
 # Source-separated raw metric records exercise the real resolver, not a mock result.
 metrics=[{'record_id':f'{source}-{i}','record_type':'steps','source':source,
   'recorded_at':f'2026-09-{21+i}T10:00:00Z','value':1000+i+(9000 if source=='garmin' else 0),'unit':'steps'}
   for source in ('oura','garmin') for i in range(2)]
 query=HealthRead(metrics=['steps'],source='oura',period={'kind':'between','start_at':'2026-09-21','end_at':'2026-09-22'})
 result,_=await check('metric-source-and-date-filter',query,{'wearable_records':metrics},complete)
 series=result['operations'][0]['result']['series']
 assert all(s['source']=='oura' for s in series)
 assert sorted(float(v) for s in series for _,v in s['points'])==[1000,1001]
 result,_=await check('metric-latest-raw',query.model_copy(update={'operation':'latest'}),{'wearable_records':metrics},complete)
 assert [float(v) for s in result['operations'][0]['result']['series'] for _,v in s['points']]==[1001]
 await check('profile-withheld-domain-is-incomplete',profile,{'profile_health':profiles['profile_health']},incomplete)
 sleep=HealthRead(metrics=['total_sleep'],source='oura',period={'kind':'between','start_at':'2026-09-23','end_at':'2026-09-23'},date_basis='sleep_end_day')
 sleep_rows=[{'record_id':'sleep','record_type':'total_sleep','source':'oura','recorded_at':'2026-09-22T20:00:00Z','value':420,'unit':'min','sleep_episode':{'schema_version':1,'start_at':'2026-09-22T23:00:00Z','end_at':'2026-09-23T06:00:00Z'}}]
 result,_=await check('sleep-explicit-episode-not-recorded-date',sleep,{'wearable_records':sleep_rows},complete)
 assert result['operations'][0]['result']['series'][0]['points']==[['2026-09-22T20:00:00Z',420.0]]
 missing=[{k:v for k,v in sleep_rows[0].items() if k!='sleep_episode'}]
 await check('sleep-missing-timing-remains-incomplete',sleep,{'wearable_records':missing},incomplete)
 workout=HealthRead(records=['workouts'],operation='latest',period={'kind':'all_history'},date_basis='started_at',limit=2)
 sessions=[{'record_id':str(i),'source':'oura','payload':{'started_at':f'2026-09-{20+i}T12:00:00Z','ended_at':f'2026-09-{20+i}T13:00:00Z','workout_type':'synthetic','duration_minutes':60}} for i in range(3)]
 result,_=await check('latest-two-workout-sessions',workout,{'workout_sessions':sessions},complete)
 assert [r['started_at'][:10] for r in result['operations'][0]['result']['record_details']]==['2026-09-22','2026-09-21']
 calendar_ytd=HealthRead(records=['calendar'],operation='latest',period={'kind':'between','start_at':'2026-01-01','end_at':'2026-09-23'},date_basis='last_done_date')
 result,_=await check('calendar-year-to-date-whole-days',calendar_ytd,{'health_calendar_events':[
  {'record_id':'a','payload':{'name':'Previous year','last_done_date':'2025-12-31'}},
  {'record_id':'b','payload':{'name':'Completed this year','last_done_date':'2026-09-23'}},
  {'record_id':'c','payload':{'name':'Future date','last_done_date':'2026-09-24'}}]},complete)
 assert [r['name'] for r in result['operations'][0]['result']['record_details']]==['Completed this year']
 # Every complete planned fixture also passes the actual closed batch contract.
 contracts=0
 for name in ('evaluation-48.json','remaining-19.json','regression-59.json','holdout-evaluation-24.json','additional-160.json'):
  for row in json.loads((ROOT/'evidence/selector-v5'/name).read_text())['rows']:
   if row['result']['status']!='planned':continue
   # Already bound/compiled by evaluation; this verifies backend contract shape.
   if 'batch' in row:batch=row['batch']
   else:
    from scripts.evaluate_query_plan import request
    req=request(row['request'],row['history']);batch=compile_batch(row['result'],req)
   SourceBatchRequest.model_validate(normalize_health_ranges(batch));contracts+=1
 files=['vita/py/src/vita_agent/health/'+f+'.py' for f in ['query_tool','resolver','turn_materialization','record_detail_pages','lab_dates','calendar_dates']]+['vita/py/src/vita_agent/kernel/source_batch_contracts.py']
 report={'scope':'Actual Vita contracts/query executor; synthetic rows only. No database, private records or integration installed.','actual_vita_source':str(VITA),'source_sha256':{f:hashlib.sha256((VITA/f).read_bytes()).hexdigest() for f in files},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'consumer_sha256':hashlib.sha256((ROOT/'query_execution.py').read_bytes()).hexdigest(),'contract_cases':contracts,'passed':len(rows),'rows':rows}
 (ROOT/'evidence/selector-v5/actual-executor.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':asyncio.run(main())
