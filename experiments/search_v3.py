import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.local_vita_router import local_client
root=Path(__file__).resolve().parents[1]
cases=json.loads((root/'experiments/confirmation.json').read_text())
choice_prompts=[
 ('What does the user want?', ['view personal information','learn general information','modify a setting']),
 ('Classify the requested action.', ['read existing data','explain a concept','change settings']),
 ('What action is requested, rather than prohibited?', ['read existing data','explain a concept','change settings']),
 ('What is the intent of this message?', ['view saved information','ask for an explanation','change a preference']),
 ('What does the user ask the assistant to do?', ['look up current data','explain a general concept','update account settings']),
 ('Choose the action that the user actually requests. Respect negations.', ['retrieve information','explain meaning','modify settings']),
 ('Which category best describes the request?', ['question about personal data','general knowledge question','instruction to change a setting']),
 ('Is the user asking to read, understand, or change something?', ['read','understand','change']),
 ('What is the request?', ['show existing records or settings','explain a topic','edit settings']),
 ('Which action should be performed?', ['retrieve','explain','update']),
]
noul_prompts=[
 'Does the user ask to access their existing personal records?',
 'The user asks to see information already stored in their account.',
 'The user wants to retrieve their own saved data or current settings.',
 'Does answering the request require looking up information in the user account?',
 'Is the user asking to view or retrieve existing personal data or current account settings?',
 'The user asks to read existing information, not change settings or explain a general concept.',
 'Is the user requesting a lookup of their own data?',
 'Does the user want to know a value already saved in their account?',
 'The user wants the assistant to show their stored information.',
 'Does the request ask for existing records, measurements or currently configured settings?',
]
report={'fresh_sha256':hashlib.sha256((root/'experiments/fresh-v3.json').read_bytes()).hexdigest(),'choice':[],'noul':[]}
with local_client() as c:
 for kind,variants in [('choice',choice_prompts),('noul',noul_prompts)]:
  for v in variants:
   q={'type':kind,'instructions':v[0] if kind=='choice' else v}
   if kind=='choice': q['options']=v[1]
   rows=[]
   for case,state,choice,noul in cases:
    start=time.perf_counter();r=c.post('/decide',json={'state':state,'questions':[q]});r.raise_for_status();a=r.json()['answers'][0]
    pred=q['options'].index(a['choice']) if kind=='choice' else a['noul']>=.5
    gold=['read existing data','explain a concept','change settings'].index(choice) if kind=='choice' else noul
    rows.append({'case':case,'correct':pred==gold,'answer':a,'ms':(time.perf_counter()-start)*1000})
   report[kind].append({'question':q,'correct':sum(r['correct'] for r in rows),'rows':rows})
   print(kind,len(report[kind]),report[kind][-1]['correct'],flush=True)
   (root/'evidence/local-v3-search.json').write_text(json.dumps(report,indent=2)+'\n')
