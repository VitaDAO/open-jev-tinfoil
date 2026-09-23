"""Bounded synthetic comparison; never logs credentials; no retries."""
import json, math, statistics, time
from datetime import datetime, timezone
from pathlib import Path
import httpx

questions = [
 {'type':'choice','instructions':'Classify the requested action.','options':['read existing data','explain a concept','change settings']},
 {'type':'noul','instructions':'Does the user ask to access their existing personal records?'},
 {'type':'score','instructions':'How urgent is this request?','options':['not urgent','moderately urgent','very urgent']},
]
cases = [
 ('sleep-read','Show my sleep duration from last night.','read existing data',True),
 ('sleep-explain','Explain what REM sleep means in general.','explain a concept',False),
 ('notifications','Turn off all push notifications.','change settings',False),
 ('steps-read','What was my total step count yesterday?','read existing data',True),
 ('hrv-explain','What does heart rate variability mean?','explain a concept',False),
 ('units','Switch the distance unit to kilometers.','change settings',False),
 ('negated-read','Do not look up my records; just explain how sleep scores are calculated.','explain a concept',False),
 ('read-only','Show my notification settings, but do not change them.','read existing data',True),
 ('urgent-setting','Urgent: stop sending push notifications immediately!','change settings',False),
 ('comparison','Compare my step totals from yesterday and the day before.','read existing data',True),
]
keys=Path.home()/'.config/open-jev-tinfoil'
report={'started_at':datetime.now(timezone.utc).isoformat(),'method':'10 frozen synthetic cases x 3 repetitions, 3 questions each. Persistent client per arm; alternating order; no retries. Local arm excludes WAN and enclave overhead. Not a held-out accuracy benchmark.','cases':cases,'questions':questions,'arms':{}}
clients={
 'open_jev_local':(httpx.Client(timeout=45),'http://127.0.0.1:18080/decide',(keys/'api.key').read_text().strip()),
 'typesafe_jev':(httpx.Client(timeout=45),'https://api.typesafe.ai/v1/systemone',(keys/'typesafe.key').read_text().strip()),
}
for name in clients: report['arms'][name]={'runs':[]}
for repeat in range(3):
 for index,(case,state,expected_choice,expected_noul) in enumerate(cases):
  for name in (list(clients) if (repeat*10+index)%2==0 else list(reversed(clients))):
   client,url,key=clients[name]
   payload={'state':state,'questions':questions}
   if name=='typesafe_jev':
    qs={}
    for i,q in enumerate(questions):
     item={k:v for k,v in q.items() if k!='options'}
     if q['type']=='choice': item['criteria']={x:None for x in q['options']}
     elif q['type']=='score': item['criteria']=q['options']
     qs[f'q{i}']=item
    payload={'state':state,'model':'jev-1.13.0','questions':qs}
   start=time.perf_counter()
   try:
    r=client.post(url,json=payload,headers={'Authorization':'Bearer '+key})
    run={'case':case,'repeat':repeat,'elapsed_ms':round((time.perf_counter()-start)*1000,2),'status':r.status_code,'response':r.json()}
   except Exception as exc: run={'case':case,'repeat':repeat,'elapsed_ms':round((time.perf_counter()-start)*1000,2),'error':type(exc).__name__}
   report['arms'][name]['runs'].append(run)
   if run.get('status')!=200:
    Path('evidence/typesafe-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(f'{name}: failed with status {run.get("status")}, error {run.get("error")}')
 for name,arm in report['arms'].items():
  runs=arm['runs']; warm=[r['elapsed_ms'] for r in runs[1:]]
  arm['summary']={'requests':len(runs),'first_call_ms':runs[0]['elapsed_ms'],'warm_n':len(warm),'warm_p50_ms':statistics.median(warm),'warm_p95_ms':sorted(warm)[math.ceil(.95*len(warm))-1]}
  first=runs[:10]; choices=[]; nouls=[]
  for r in first:
   a=r['response']['answers']; a=list(a.values()) if isinstance(a,dict) else a
   expected=next(c for c in cases if c[0]==r['case'])
   choices.append(a[0]['choice']==expected[2]); nouls.append((a[1]['noul']>=.5)==expected[3])
  arm['summary'].update(choice_correct=sum(choices),noul_correct=sum(nouls),unique_cases=len(first))
  if name=='typesafe_jev':
   tokens=sum(r['response'].get('usage',{}).get('input_tokens',0) for r in runs)
   arm['summary'].update(input_tokens=tokens,estimated_input_cost_usd=tokens*.042/1e6,models=sorted(set(r['response'].get('model','unknown') for r in runs)))
 Path('evidence/typesafe-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v['summary'] for k,v in report['arms'].items()}),flush=True)
for c,_,_ in clients.values(): c.close()
