"""Exact complete-query evaluation for the experimental single-plan contract."""
import hashlib,json,sys,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
from query_selector import QuerySelector,identity
from query_plan import QueryRequest,QueryPlan,compile_batch
INVENTORY=['total_sleep','sleep_efficiency','steps','apob','ldl_cholesterol','oxygen_saturation','respiratory_rate','custom_metric']
def request(text,history=()):
 return QueryRequest.model_validate({'schema_version':'vita-selector/v2','available_metrics':INVENTORY,'available_record_types':['profile','labs','workouts','calendar'],'available_sources':['oura','garmin'],'literature_available':True,'reference_time':'2026-09-23T12:00:00Z','state':{'current_request':text,'recent_user_requests':history,'reference_date':'2026-09-23','time_zone':'UTC'}})
def main():
 import torch
 from typed_decisions.open_jev import OpenJev
 torch.set_num_threads(4)
 model=OpenJev.from_pretrained(str(ROOT/'model-fp16'),device='cpu')
 model.collator._ids=lambda text:model.tok(text,add_special_tokens=False)['input_ids'];model.collator._cache.clear()
 selector=QuerySelector(model);start_id=identity()
 files=[('expanded-48.json','evaluation-48.json'),('regression-59-v2.json','regression-59.json'),
        ('remaining-19-oracle.json','remaining-19.json'),('holdout-24.json','holdout-evaluation-24.json'),
        ('additional-160-v2.json','additional-160.json'),
        ('future-boundaries-40.json','future-boundaries-40-results.json'),
        ('review-profile-16.json','review-profile-16-results.json')]
 failures=0
 for fixture,output in files:
  path=ROOT/'evidence/selector-v5'/fixture;fixture_hash=hashlib.sha256(path.read_bytes()).hexdigest()
  cases=json.loads(path.read_text());rows=[]
  for c in cases:
   started=time.perf_counter();req=request(c['request'],c['history']);result=selector.select_query(req)
   actual=result['queries'] if result['status']=='planned' else None
   grade='correct_plan' if actual==c['expected_queries'] and actual else 'correct_handoff' if actual is None and c['expected_queries'] is None else 'missed_plan' if actual is None else 'wrong_plan'
   batch=None;compile_error=None
   try:batch=compile_batch(result,req)
   except ValueError as exc:compile_error=str(exc)
   rows.append({**c,'result':result,'grade':grade,'batch':batch,'compile_error':compile_error,'elapsed_ms':(time.perf_counter()-started)*1000})
  assert identity()==start_id and hashlib.sha256(path.read_bytes()).hexdigest()==fixture_hash
  counts={k:sum(r['grade']==k for r in rows) for k in ['correct_plan','correct_handoff','missed_plan','wrong_plan']}
  report={'selector_sha256':start_id,'fixture_sha256':fixture_hash,'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'outcomes':counts,'compile_failures':sum(bool(r['compile_error']) for r in rows),'rows':rows}
  (ROOT/'evidence/selector-v5'/output).write_text(json.dumps(report,indent=2)+'\n');print(output,json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
  failures+=counts['missed_plan']+counts['wrong_plan']+report['compile_failures']
 if failures:raise SystemExit(f'{failures} evaluation failures; inspect saved evidence')
if __name__=='__main__':main()
