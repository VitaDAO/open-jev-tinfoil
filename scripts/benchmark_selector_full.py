"""Synthetic local full-selector + pure plan latency; no WAN/attestation claim."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, UTC

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from plan_adapter import CATALOG,build_plan
from evaluate_selector import evaluate

if __name__=='__main__':
    base='http://127.0.0.1:18086';env={**os.environ,'OPEN_JEV_API_KEY':'synthetic-model-evaluation-only-not-a-real-secret',
      'ENABLE_EXPERIMENTAL_SELECTOR':'1','MODEL_DIR':str(ROOT/'model-fp16'),'PYTHONPATH':str(ROOT/'vendor'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    started=time.perf_counter()
    with tempfile.TemporaryFile() as logs:
        process=subprocess.Popen([sys.executable,'-m','uvicorn','server:create_app','--factory','--host','127.0.0.1','--port','18086','--no-access-log'],cwd=ROOT,env=env,stdout=logs,stderr=logs)
        try:
            while True:
                if process.poll() is not None:
                    logs.seek(0);sys.stderr.write(logs.read().decode())
                    raise RuntimeError('Synthetic server exited before readiness')
                try:
                    with urllib.request.urlopen(base+'/health',timeout=1) as r:
                        if r.status==200:break
                except OSError:pass
                if time.perf_counter()-started>90:raise RuntimeError('Synthetic server startup exceeded bound')
                time.sleep(.1)
            cold=time.perf_counter()-started
            def call(body):
                req=urllib.request.Request(base+'/v1/select',data=json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+env['OPEN_JEV_API_KEY'],'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=15) as response:return json.load(response)
            inventory=list(CATALOG['metrics'])
            body={'schema_version':'vita-selector/v1','available_metrics':inventory,'literature_available':True,
                  'state':{'current_request':'Analyze me','reference_date':'2026-09-23','time_zone':'UTC','recent_user_requests':[]}}
            times=[];plans=[];last=None
            for _ in range(101):
                t=time.perf_counter();last=call(body)
                plan=build_plan(last,inventory,now=datetime(2026,9,23,12,tzinfo=UTC),time_zone='UTC',
                                record_types=('profile','workouts','labs','calendar'),literature_available=True)
                times.append((time.perf_counter()-t)*1000);plans.append(plan['status'])
            os.environ['SELECTOR_CASES']=str(ROOT/'evidence/selector-v1/model-heldout.json')
            os.environ['EVALUATION_LABEL']='Frozen new synthetic paraphrases/categories after learned adapter freeze. Assistant authored, not independent human labels. No retuning after this evaluation. Prior 36 cases and 449 training examples are development only.'
            report=evaluate(call)
            report['full_selector_and_plan']={'cold_process_setup_ms':cold*1000,'first_request_ms':times[0],
                'warm_n':100,'warm_p50_ms':statistics.median(times[1:]),'warm_p95_ms':sorted(times[1:])[94],
                'inventory_size':len(inventory),'plan_status_counts':{s:plans.count(s) for s in set(plans)},
                'operation_count':len(plan['batch']['required_operation_ids']) if plan.get('batch') else 0,
                'target_warm_ms':1000,'scope':'local HTTP one Open-JEV encoder pass + deterministic plan; no database reads, WAN, TLS attestation, Fable or hosted CPU timing. Cold process may use warm OS file cache.'}
            report['analyze_me_answer']=last
            print(json.dumps(report,indent=2))
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
