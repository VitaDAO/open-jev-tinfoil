"""Single-client synthetic loopback smoke. Never provides enclave attestation.

Owns and stops its subprocess; always writes selector-v5 evidence. There is no
plaintext fallback in VitaClient. This test deliberately injects a loopback-only
transport after bypassing its attestation constructor, using a synthetic key.
"""
import argparse,hashlib,json,os,secrets,socket,statistics,subprocess,sys,tempfile,time
from pathlib import Path
from urllib.parse import urlparse
import httpx
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
from examples.vita_client import VitaClient,HOST
from query_selector import identity,INTENT_SHA256
from scripts.evaluate_query_plan import request

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--base-url')
 parser.add_argument('--output',type=Path,default=ROOT/'evidence/selector-v5/http-smoke.json')
 args=parser.parse_args()
 if args.base_url:
  parsed=urlparse(args.base_url)
  if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or not parsed.port or parsed.path not in ('','/'):
   raise ValueError('Synthetic smoke requires an explicit loopback URL')
 start_identity=identity()
 with socket.socket() as sock:
  sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 url=args.base_url.rstrip('/') if args.base_url else f'http://127.0.0.1:{port}'
 token=os.environ['OPEN_JEV_API_KEY'] if args.base_url else secrets.token_hex(32)
 env={**os.environ,'OPEN_JEV_API_KEY':token,'ENABLE_EXPERIMENTAL_SELECTOR':'1',
      'MODEL_DIR':str(ROOT/'model-fp16'),'PYTHONPATH':str(ROOT/'vendor'),'OMP_NUM_THREADS':'4',
      'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
 with tempfile.TemporaryFile() as log:
  process=None if args.base_url else subprocess.Popen([sys.executable,'-m','uvicorn','server:create_app','--factory','--host','127.0.0.1','--port',str(port),'--no-access-log'],cwd=ROOT,env=env,stdout=log,stderr=log)
  try:
   with httpx.Client(timeout=30) as http:
    deadline=time.monotonic()+90
    while True:
     if process is not None and process.poll() is not None:raise RuntimeError('Owned local server failed during startup')
     try:
      health=http.get(url+'/health');health.raise_for_status();break
     except httpx.TransportError:
      if time.monotonic()>deadline:raise RuntimeError('Owned local server startup timed out')
      time.sleep(.25)
    data=health.json()
    assert data['selector_enabled'] and data['selector_sha256']==start_identity and data['selector_adapter_sha256']==INTENT_SHA256
    body=request('Show my steps yesterday').model_dump(mode='json');headers={'Authorization':'Bearer '+token}
    assert http.post(url+'/v1/select',json=body).status_code==401
    assert http.post(url+'/v1/select-baseline',json=body,headers=headers).status_code==404
    for change in [{'schema_version':'vita-selector/v1'},{'private_records':'synthetic-sentinel'}]:
     response=http.post(url+'/v1/select',json={**body,**change},headers=headers)
     assert response.status_code==422 and 'synthetic-sentinel' not in response.text
    class LoopbackTransport:
     def post(self,target,**kwargs):
      parsed=urlparse(target)
      assert parsed.hostname==HOST and parsed.path=='/v1/select'
      return http.post(url+parsed.path,**kwargs)
    client=VitaClient.__new__(VitaClient);client.http=LoopbackTransport();client.token=token
    client.selector_sha256=start_identity;client.adapter_sha256=INTENT_SHA256
    # Every fresh composition crosses actual HTTP and the single validated client.
    cases=json.loads((ROOT/'evidence/selector-v5/holdout-24.json').read_text());rows=[]
    for case in cases:
     started=time.perf_counter();result=client.select(request(case['request'],case['history']))
     actual=[q.model_dump(mode='json') for q in result.queries] if result.status=='planned' else None
     assert actual==case['expected_queries'],case['id']
     rows.append({'id':case['id'],'passed':True,'status':result.status,'elapsed_ms':(time.perf_counter()-started)*1000})
    paths={'simple':'Show my steps yesterday','semantic':'What is the most recent ApoB value I have?',
           'two_clauses':'Show my ApoB in 2023; show my steps in 2024','profile':'Show my complete profile'}
    timings={}
    for name,text in paths.items():
     req=request(text);client.select(req);values=[]
     for _ in range(10):
      started=time.perf_counter();result=client.select(req);values.append((time.perf_counter()-started)*1000)
      assert result.status=='planned'
     values.sort();timings[name]={'requests':10,'median_ms':statistics.median(values),'p95_ms':values[9]}
    assert identity()==start_identity and http.get(url+'/health').json()['selector_sha256']==start_identity
    paths=['server.py','query_plan.py','query_selector.py','query_execution.py','examples/vita_client.py','scripts/smoke_proposal.py']
    report={'scope':'Local 4-thread CPU, synthetic HTTP and injected loopback transport; NOT attested or a deployed service. No database or external inference.',
      'selector_sha256':start_identity,'adapter_sha256':INTENT_SHA256,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
      'passed':len(rows),'total':len(rows),'rows':rows,'warm_http_latency':timings,'auth_and_invalid_contract_checks':True}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_sha256')},indent=2))
  finally:
   if process is not None:
    process.terminate()
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
if __name__=='__main__':main()
