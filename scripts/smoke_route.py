"""Actual model /route contract, recorded prediction parity, and input rejection."""
import hashlib,json,os,statistics,time,urllib.request,urllib.error
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=os.environ.get('BASE_URL','http://127.0.0.1:18080')
key=os.environ['OPEN_JEV_API_KEY']
expected_sha=hashlib.sha256((root/'adapters/vita-intent-v1.json').read_bytes()).hexdigest()
recorded=json.loads((root/'evidence/local-v3-verified.json').read_text())
def post(body,auth=True):
    req=urllib.request.Request(base+'/route',data=json.dumps(body).encode(),headers={'Content-Type':'application/json',**({'Authorization':'Bearer '+key} if auth else {})})
    try:
        with urllib.request.urlopen(req,timeout=30) as r:return r.status,json.load(r)
    except urllib.error.HTTPError as e:return e.code,json.load(e)
assert post({'state':'test'},False)[0]==401
for b in [{'state':' '},{'state':'word '*300},{'state':123},{'state':'private-sentinel','extra':'no'}]:
    status,data=post(b);assert status==422,(status,data)
    assert 'private-sentinel' not in json.dumps(data)
results={}
for name,file in [('target24','confirmation.json'),('fresh30','fresh-v3.json')]:
    ref={r['case']:r['result'] for r in recorded['results'][name]['rows']}
    times=[]
    for case,state,_,_ in json.loads((root/'experiments'/file).read_text()):
        t=time.perf_counter();status,data=post({'state':state});times.append((time.perf_counter()-t)*1000)
        assert status==200,(status,data)
        assert data['model_revision']=='19bf9a64815add579fbf6c907bef584d9277a8e4'
        assert data['adapter_sha256']==expected_sha and data['advisory'] is True
        assert data['weight_storage_dtype']=='float16' and data['backbone_compute_dtype']=='float32'
        assert (data['action'],data['record_access'])==(ref[case]['action'],ref[case]['record_access']),case
    results[name]={'n':len(times),'median_ms':statistics.median(times),'max_ms':max(times)}
print(json.dumps({'adapter_sha256':expected_sha,'prediction_parity':'all54 match recorded results','timings':results},indent=2))
