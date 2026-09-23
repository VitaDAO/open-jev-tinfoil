"""Confirmation for the hybrid selected after first holdout; no provider calls."""
import hashlib, json, sys, time, statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.local_vita_router import local_client,route,QUESTIONS
root=Path(__file__).resolve().parents[1]
fixture=root/'experiments/confirmation.json'
cases=json.loads(fixture.read_text())
report={'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'selection':'After first validation, preserve joint choice/score and replace only noul with isolated call. No thresholds or weights changed.','runs':{'baseline':[],'hybrid':[]}}
with local_client() as client:
 for i,(name,state,gold_choice,gold_noul) in enumerate(cases):
  for mode in (['baseline','hybrid'] if i%2==0 else ['hybrid','baseline']):
   start=time.perf_counter()
   if mode=='hybrid': result=route(client,state)
   else:
    response=client.post('/decide',json={'state':state,'questions':list(QUESTIONS)})
    response.raise_for_status();a=response.json()['answers']
    result={'action':a[0]['choice'],'record_access_probability':a[1]['noul'],'urgency_score':a[2]['score'],'answers':a}
   report['runs'][mode].append({'case':name,'elapsed_ms':round((time.perf_counter()-start)*1000,2),'choice_correct':result['action']==gold_choice,'noul_correct':(result['record_access_probability']>=.5)==gold_noul,'result':result})
report['summary']={mode:{'cases':len(rows),'choice_correct':sum(r['choice_correct'] for r in rows),'noul_correct':sum(r['noul_correct'] for r in rows),'median_ms':statistics.median(r['elapsed_ms'] for r in rows)} for mode,rows in report['runs'].items()}
assert all(a['result']['action']==b['result']['action'] and a['result']['urgency_score']==b['result']['urgency_score'] for a,b in zip(report['runs']['baseline'],report['runs']['hybrid']))
(root/'evidence/local-tuning-confirmation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['summary'],indent=2))
