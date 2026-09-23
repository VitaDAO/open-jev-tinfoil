"""Verify exact-release TLS before synthetic /route calls; never plain fallback."""
import json,sys,time,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from examples.vita_client import VitaDecisionClient, HOST, ADAPTER_SHA256
start=time.perf_counter();client=VitaDecisionClient();attestation_ms=(time.perf_counter()-start)*1000
report={'attestation_ms':attestation_ms,'verification':client.verifier.get_verification_document().to_dict(),'adapter_sha256':ADAPTER_SHA256,'results':{}}
try:
 health=client.http.get(f'https://{HOST}/health',timeout=30);health.raise_for_status()
 assert health.json()['adapter_sha256']==ADAPTER_SHA256
 assert client.http.post(f'https://{HOST}/route',json={'state':'test'},timeout=15).status_code==401
 for body in [{'state':' '},{'state':'word '*300},{'state':'private-sentinel','extra':True}]:
  r=client.http.post(f'https://{HOST}/route',headers={'Authorization':'Bearer '+client.token},json=body,timeout=15)
  assert r.status_code==422 and 'private-sentinel' not in r.text
 recorded=json.loads((root/'evidence/local-v3-verified.json').read_text())
 for name,file in [('target24','confirmation.json'),('fresh30','fresh-v3.json')]:
  ref={r['case']:r['result'] for r in recorded['results'][name]['rows']};times=[]
  for case,state,_,_ in json.loads((root/'experiments'/file).read_text()):
   start=time.perf_counter();data=client.route(state);times.append((time.perf_counter()-start)*1000)
   assert (data['action'],data['record_access'])==(ref[case]['action'],ref[case]['record_access']),case
  report['results'][name]={'requests':len(times),'median_ms':statistics.median(times),'max_ms':max(times),'prediction_parity':True}
 (root/'evidence/route-live.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='verification'},indent=2))
finally:client.close()
