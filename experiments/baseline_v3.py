import json,statistics,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from examples.local_vita_router import local_client,route,QUESTIONS
rows={'baseline':[],'previous_hybrid':[]}
with local_client() as client:
 for case,state,choice,read in json.loads((root/'experiments/fresh-v3.json').read_text()):
  for name in rows:
   start=time.perf_counter()
   if name=='baseline':
    r=client.post('/decide',json={'state':state,'questions':list(QUESTIONS)});r.raise_for_status();a=r.json()['answers']
    result={'action':a[0]['choice'],'record_access_probability':a[1]['noul']}
   else:result=route(client,state)
   rows[name].append({'case':case,'correct_choice':result['action']==choice,'correct_record':(result['record_access_probability']>=.5)==read,'elapsed_ms':(time.perf_counter()-start)*1000,'result':result})
summary={name:{'choice_correct':sum(r['correct_choice'] for r in rs),'record_correct':sum(r['correct_record'] for r in rs),'cases':len(rs),'median_ms':statistics.median(r['elapsed_ms'] for r in rs)} for name,rs in rows.items()}
(root/'evidence/local-v3-baselines.json').write_text(json.dumps({'rows':rows,'summary':summary},indent=2)+'\n')
print(json.dumps(summary,indent=2))
