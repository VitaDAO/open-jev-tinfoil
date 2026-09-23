"""Synthetic selector evaluation; the 59-case suite uses frozen exact contract plans.

Other legacy fixtures retain their explicit metric/range and coarse-label checks
and are labelled as such. They are not a production quality measurement.
"""
import json
import hashlib
import os
import statistics
import sys
import time
from pathlib import Path
from datetime import datetime, UTC

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from selector import SelectorRequest, select, SELECTOR_SHA256
from contracts.vita_read_contract import read_arguments
from scripts.evaluate_selector_exact import grade

INVENTORY=['total_sleep','sleep_efficiency','steps','apob','ldl_cholesterol',
           'oxygen_saturation','respiratory_rate','custom_metric']

def payload(case, inventory=INVENTORY):
    return {'schema_version':'vita-selector/v1','available_metrics':inventory,'literature_available':True,
            'state':{'current_request':case['request'],'recent_user_requests':case['history'],
                     'reference_date':'2026-09-23','time_zone':'UTC'}}

def evaluate(call):
    path=Path(os.environ.get('SELECTOR_CASES',str(ROOT/'evidence/selector-v1/heldout.json')));cases=json.loads(path.read_text())
    fixture_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    gold=None
    if fixture_sha=='576e34e4cd0b5c78ced289a25d2fb1db67be22eade9a21d5a30409977eca830d':
        exact=json.loads((ROOT/'evidence/selector-v2/regression-59.json').read_text())
        gold={c['id']:c['gold'] for c in exact}
        if {c['id'] for c in cases} != set(gold):raise ValueError('Incomplete exact oracle')
    rows=[]
    for case in cases:
        start=time.perf_counter();result=call(payload(case));elapsed=(time.perf_counter()-start)*1000
        plan=read_arguments(result,INVENTORY,literature_available=True,
                            record_types=('profile','workouts','labs','calendar'),
                            now=datetime(2026,9,23,12,tzinfo=UTC),time_zone='UTC')
        outcome=None
        if gold is not None:
            outcome=grade(result,plan,gold[case['id']])
            correct=outcome in ('correct_plan','correct_handoff')
        elif case['expected'] is None:
            correct=result['status']=='unsupported' and plan is None
        else:
            correct=result['status']=='selected' and plan is not None and all(
                result['answers'][k]['choice']==v for k,v in case['expected'].items())
            if correct and 'concepts' in case:
                correct=sorted(m for read in plan['health_reads'] for m in read.get('concepts',[]))==case['concepts']
            if correct and 'range' in case:
                correct=all(read['range']==case['range'] for read in plan['health_reads'] if 'concepts' in read)
        rows.append({'id':case['id'],'correct':correct,'outcome':outcome,'status':result['status'],'elapsed_ms':round(elapsed,3),
                     'plan':plan,'expected_selection':not gold[case['id']]['handoff_ok'] if gold is not None else case['expected'] is not None,'category':case.get('category','legacy_development'),
                     'vita_supported':case.get('vita_supported',True),'reason':result.get('reason')})
    return {'scoring':'exact_legacy_contract_plan' if gold is not None else 'legacy_labels_with_optional_scope_assertions',
            'outcomes':{name:sum(r['outcome']==name for r in rows) for name in sorted({r['outcome'] for r in rows if r['outcome']})},
            'selector_sha256':result['selector_sha256'],'model_revision':result.get('model_revision'),
            'adapter_sha256':result.get('adapter_sha256'),'implementation':result['implementation'],'fixtures_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'cases':len(rows),'correct':sum(r['correct'] for r in rows),
            'valid_requests':sum(r['expected_selection'] for r in rows),
            'correct_selected':sum(r['correct'] and r['status']=='selected' and r['plan'] is not None for r in rows),
            'wrong_executable_plans':sum(r['outcome']=='wrong_plan' for r in rows) if gold is not None else None,
            'selected_valid_definition':'Selected expected-plan requests, not necessarily correct; use correct_selected for correctness',
            'selected_valid':sum(r['expected_selection'] and r['status']=='selected' for r in rows),
            'incorrect_selected':sum(not r['correct'] and r['status']=='selected' for r in rows),
            'notes':os.environ.get('EVALUATION_LABEL','Development fixture; not independent validation'),
            'supported_vita_requests':sum(r['vita_supported'] for r in rows),
            'fallback_count':sum(r['status']=='unsupported' for r in rows),
            'category_results':{c:{'n':sum(r['category']==c for r in rows),'correct':sum(r['category']==c and r['correct'] for r in rows),
                'selected':sum(r['category']==c and r['status']=='selected' for r in rows)} for c in sorted({r['category'] for r in rows})},
            'results':rows}

if __name__=='__main__':
    base=os.environ.get('BASE_URL')
    if base:
        import urllib.request
        from contextlib import nullcontext
        with nullcontext():
            def call(body):
                req=urllib.request.Request(base+os.environ.get('SELECTOR_ENDPOINT','/v1/select-baseline'),data=json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+os.environ['OPEN_JEV_API_KEY'],'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=15) as response:
                    return json.load(response)
            report=evaluate(call)
            times=[]
            # Full 512-identifier request: first request separately, then warm.
            b=payload({'request':'Analyze me','history':[]},['metric_'+str(n) for n in range(512)])
            for _ in range(101):
                t=time.perf_counter();call(b);times.append((time.perf_counter()-t)*1000)
            warm=sorted(times[1:])
            report['full_512_http']={'first_ms':times[0],'warm_n':100,'p50_ms':statistics.median(warm),'p95_ms':warm[94],
                                     'scope':'loopback HTTP; excludes attestation, WAN and process startup; '+report['implementation']}
    else:
        report=evaluate(lambda body:select(SelectorRequest.model_validate(body)))
    print(json.dumps(report,indent=2))
