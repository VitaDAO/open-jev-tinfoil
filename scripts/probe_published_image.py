"""Synthetic acceptance/profiling of one immutable image; no live service IO."""
import copy,json,os,secrets,statistics,subprocess,sys,time,http.client,urllib.request,urllib.error
from pathlib import Path
IMAGE='ghcr.io/vitadao/open-jev-tinfoil@sha256:b6f570567c35ef0222dfb8563a35057f95843d22b9564f20e5bd59a53e2a3174'
SELECTOR='6beaf54f171b47a6f068c785e63f85a2e2aac64e63a73e0b4eb36777a55b5cc5'
ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'evidence/selector-v7/fresh-selector-acceptance-v1.json'
def core_cases(packet):
 if os.environ.get('PROBE_SUITE')=='vita':return json.loads((ROOT/'evidence/selector-v7/new-image-core4-v1.json').read_text())['cases']
 base=packet['cases'][0]['request'];result=[]
 for name,text in [('profile','Show my profile.'),('simple','Show my steps yesterday.'),('metadata','When were my lab reports done, and which lab issued them?'),('weekly','Give me a descriptive health summary for last week.')]:
  req=copy.deepcopy(base);req['state']['current_request']=text;result.append({'id':name,'request':req})
 return result

def grade(case,result):
 e=case['expected'];qs=result['queries'];issues=[]
 if result['status']=='handoff':
  return ['handoff_not_inert'] if qs else ['missed_read'] if e['mode']=='required_read' else []
 if e['mode']=='must_handoff':issues.append('unexpected_acquisition')
 if e['mode']=='broad_without_research' and any(q['kind']=='research' for q in qs):issues.append('research_despite_opt_out')
 if len({json.dumps(q,sort_keys=True) for q in qs})!=len(qs):issues.append('duplicate_acquisition')
 if e['queries'] is not None:
  left=list(qs)
  for target in e['queries']:
   match=next((q for q in left if all(q.get(k)==v for k,v in target.items())),None)
   if match is None:issues.append('missing_or_changed_clause')
   else:left.remove(match)
  if left:issues.append('extra_or_changed_clause')
 return issues

def instrument():
 sys.path.insert(0,'/app')
 from server import Engine
 from query_plan import QueryRequest
 engine=Engine()
 import torch
 startup_threads=torch.get_num_threads();requested_threads=int(os.environ['OMP_NUM_THREADS'])
 torch.set_num_threads(requested_threads)
 assert torch.get_num_threads()==requested_threads
 events=[];backbone=engine.model.model.backbone;original=backbone.forward
 def forward(*args,**kwargs):
  start=time.perf_counter();value=original(*args,**kwargs)
  events.append({'ms':(time.perf_counter()-start)*1000,'shape':list(kwargs['input_ids'].shape)})
  return value
 backbone.forward=forward;rows=[]
 for case in core_cases(json.loads(FIXTURE.read_text())):
  for repetition in range(3):
   events.clear();start=time.perf_counter();result=engine.select(QueryRequest.model_validate(case['request']))
   rows.append({'id':case['id'],'repetition':repetition,'elapsed_ms':(time.perf_counter()-start)*1000,'forwards':list(events),'result':result})
 print('PROFILE_JSON='+json.dumps({'requested_threads':requested_threads,'startup_threads':startup_threads,'actual_threads':torch.get_num_threads(),'rows':rows}))

def main():
 out=ROOT/'image-probe-output';out.mkdir(exist_ok=False)
 os.environ['OPEN_JEV_API_KEY']=secrets.token_hex(32)
 def command(*args):return subprocess.check_output(args,text=True)
 def docker(*args):return command('docker',*args)
 name='openjev-exact-image-probe';packet=json.loads(FIXTURE.read_text());rows=[];health=[];resources=[]
 def save():
  (out/'http-results.json').write_text(json.dumps({'image':IMAGE,'suite':os.environ.get('PROBE_SUITE','diagnostic'),'scope':'Synthetic loopback HTTP, not attested; first selector call follows built-in Engine warmup. Sequential requests, no concurrent load.','rows':rows,'health':health,'resources':resources},indent=2)+'\n')
 def call(req=None,authorized=True):
  url='http://127.0.0.1:18080/'+('health' if req is None else 'v1/select')
  headers={'Content-Type':'application/json'}
  if authorized:headers['Authorization']='Bearer '+os.environ['OPEN_JEV_API_KEY']
  request=urllib.request.Request(url,data=None if req is None else json.dumps(req).encode(),headers=headers)
  started=time.perf_counter()
  try:
   with urllib.request.urlopen(request,timeout=30) as response:return response.status,json.load(response),(time.perf_counter()-started)*1000
  except urllib.error.HTTPError as exc:return exc.code,json.load(exc),(time.perf_counter()-started)*1000
 def stop():
  docker('rm','-f',name)
 def start():
  docker('run','-d','--name',name,'--memory','8g','--cpus','4','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777','-e','OPEN_JEV_API_KEY','-e','ENABLE_EXPERIMENTAL_SELECTOR=1','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4','-p','127.0.0.1:18080:8080',IMAGE)
  started=time.perf_counter()
  for _ in range(120):
   try:
    code,data,_=call()
    if code==200:
     assert data['selector_sha256']==SELECTOR;health.append({'startup_ms':(time.perf_counter()-started)*1000,'response':data});return
   except (urllib.error.URLError,ConnectionError,TimeoutError,http.client.RemoteDisconnected):pass
   time.sleep(1)
  raise RuntimeError('Candidate did not become healthy')
 def record(case,phase):
  code,result,ms=call(case['request']);assert code==200 and result['selector_sha256']==SELECTOR
  row={**case,'phase':phase,'http_status':code,'elapsed_ms':ms,'result':result}
  if 'expected' in case:row['issues']=grade(case,result)
  rows.append(row);save()
 docker('pull',IMAGE)
 (out/'image-inspect.json').write_text(docker('image','inspect',IMAGE))
 (out/'runner-cpu.txt').write_text(command('lscpu'))
 for case in ([] if os.environ.get('PROFILE_ONLY')=='1' else core_cases(packet)):
  start()
  try:
   code,_,_=call(case['request'],False);assert code==401
   record(case,'first_selector_after_start')
   for _ in range(5):record(case,'warm')
   resources.append({'case':case['id'],'memory_peak_bytes':int(docker('exec',name,'cat','/sys/fs/cgroup/memory.peak'))});save()
  finally:stop()
 if os.environ.get('PROFILE_ONLY')!='1' and os.environ.get('PROBE_SUITE')!='vita':
  start()
  try:
   for case in packet['cases']:record(case,'regression42')
   resources.append({'case':'regression42','memory_peak_bytes':int(docker('exec',name,'cat','/sys/fs/cgroup/memory.peak'))});save()
  finally:stop()
 profiles=[]
 for threads in [1,2,4]:
  text=docker('run','--rm','--memory','8g','--cpus','4','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777','-e','OPEN_JEV_API_KEY','-e','PROBE_SUITE','-e',f'OMP_NUM_THREADS={threads}','-e',f'MKL_NUM_THREADS={threads}','-v',str(ROOT)+':/probe:ro','--entrypoint','python',IMAGE,'/probe/scripts/probe_published_image.py','--in-container')
  profiles.append(json.loads(next(s.split('=',1)[1] for s in text.splitlines() if s.startswith('PROFILE_JSON='))))
  (out/'inference-profile.json').write_text(json.dumps(profiles,indent=2)+'\n')
 failures=[r['id'] for r in rows if r.get('issues')];print(json.dumps({'http_cases':len(rows),'regression_failures':failures,'peak_bytes':max((r['memory_peak_bytes'] for r in resources),default=None)}));assert not failures
if __name__=='__main__':
 instrument() if '--in-container' in sys.argv else main()
