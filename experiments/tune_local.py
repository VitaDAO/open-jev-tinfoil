"""Development-only selection, followed by one frozen synthetic holdout evaluation."""
import argparse, hashlib, json, statistics, time
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
original=json.loads((ROOT/'evidence/typesafe-comparison.json').read_text())
baseline=original['questions']
statement=[baseline[0],{'type':'noul','instructions':'The user wants to retrieve or view information already stored in their own account.'},baseline[2]]
described=[{'type':'choice','instructions':'What does the user want the assistant to do?', 'options':['view or retrieve saved personal data or current account settings without changing them','explain a general concept without accessing personal data','modify, enable, disable or update an account setting']},statement[1],baseline[2]]
VARIANTS={'baseline':(baseline,False),'statement':(statement,False),'described':(described,False),'isolated_baseline':(baseline,True),'isolated_described':(described,True)}
LABELS=baseline[0]['options']

def evaluate(client,cases,name):
 questions,isolated=VARIANTS[name]; runs=[]
 for case,state,choice,noul in cases:
  start=time.perf_counter();answers=[]
  for group in ([[q] for q in questions] if isolated else [questions]):
   response=client.post('/decide',json={'state':state,'questions':group})
   response.raise_for_status();answers.extend(response.json()['answers'])
  predicted=LABELS[questions[0]['options'].index(answers[0]['choice'])]
  runs.append({'case':case,'choice':predicted,'noul':answers[1]['noul'],'choice_correct':predicted==choice,'noul_correct':(answers[1]['noul']>=.5)==noul,'elapsed_ms':round((time.perf_counter()-start)*1000,2),'answers':answers})
 return {'runs':runs,'summary':{'cases':len(runs),'choice_correct':sum(r['choice_correct'] for r in runs),'noul_correct':sum(r['noul_correct'] for r in runs),'median_ms':statistics.median(r['elapsed_ms'] for r in runs)}}

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['dev','holdout']);args=parser.parse_args()
 key=(Path.home()/'.config/open-jev-tinfoil/api.key').read_text().strip()
 with httpx.Client(base_url='http://127.0.0.1:18080',headers={'Authorization':'Bearer '+key},timeout=30) as client:
  if args.stage=='dev':
   result={'holdout_sha256':hashlib.sha256((ROOT/'experiments/heldout.json').read_bytes()).hexdigest(),'variants':{k:{'questions':v[0],'isolated':v[1]} for k,v in VARIANTS.items()},'results':{}}
   for name in VARIANTS:
    result['results'][name]=evaluate(client,original['cases'],name)
    print(name,result['results'][name]['summary'],flush=True)
   result['selected']=max(result['results'],key=lambda n:(sum(result['results'][n]['summary'][k] for k in ['choice_correct','noul_correct']),-result['results'][n]['summary']['median_ms']))
   (ROOT/'evidence/local-tuning-dev.json').write_text(json.dumps(result,indent=2)+'\n')
   print('SELECTED',result['selected'])
  else:
   dev=json.loads((ROOT/'evidence/local-tuning-dev.json').read_text())
   assert hashlib.sha256((ROOT/'experiments/heldout.json').read_bytes()).hexdigest()==dev['holdout_sha256']
   cases=json.loads((ROOT/'experiments/heldout.json').read_text())
   result={'selected_on_dev':dev['selected'],'holdout_sha256':dev['holdout_sha256'],'results':{}}
   for name in dict.fromkeys(['baseline',dev['selected']]):
    result['results'][name]=evaluate(client,cases,name)
    print(name,result['results'][name]['summary'],flush=True)
   (ROOT/'evidence/local-tuning-holdout.json').write_text(json.dumps(result,indent=2)+'\n')
