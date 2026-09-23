"""Synthetic loopback API/adapter smoke; does not perform enclave attestation."""
import hashlib
import argparse
import json
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from contracts.vita_read_contract import read_arguments
from examples.selector_client import VitaSelectorClient
from learned_selector import ADAPTER_SHA256
from trained_proposal_selector import identity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/selector-v3/http-smoke.json')
    args=parser.parse_args()
    url=os.environ.get('BASE_URL','http://127.0.0.1:18126')
    if not url.startswith('http://127.0.0.1:'):
        raise ValueError('This smoke is limited to a local synthetic service')
    token=os.environ['OPEN_JEV_API_KEY']
    headers={'Authorization':'Bearer '+token}
    def body(text,history=(),records=('profile','workouts','labs','calendar')):
        return {'schema_version':'vita-selector/v1','available_metrics':['steps','respiratory_rate','sleep_efficiency','apob'],
            'available_record_types':list(records),'literature_available':False,
            'state':{'current_request':text,'recent_user_requests':list(history),'reference_date':'2026-09-23','time_zone':'UTC'}}
    rows=[]
    with httpx.Client(timeout=30) as http:
        assert http.get(url+'/health').status_code==200
        assert http.post(url+'/v1/select',json=body('Show steps')).status_code==401
        for text,history,records,expected in [
            ('Show my latest respiratory rate yesterday',(),('profile','workouts','labs','calendar'),'selected'),
            ('During 2024, how was my sleep efficiency doing?',(),('profile','workouts','labs','calendar'),'selected'),
            ('What is the most recent ApoB value I have?',(),('profile','workouts','labs','calendar'),'selected'),
            ('Now do last month',['Show my steps from Oura this week'],('profile','workouts','labs','calendar'),'unsupported'),
            ('Show my lab reports',(),(),'unsupported'),
        ]:
            response=http.post(url+'/v1/select',json=body(text,history,records),headers=headers)
            assert response.status_code==200
            value=response.json();assert value['status']==expected
            plan=read_arguments(value,body(text)['available_metrics'],record_types=records,
                                now=datetime(2026,9,23,12,tzinfo=UTC))
            assert bool(plan)==(expected=='selected')
            if text.startswith('Show my latest'):
                assert plan['health_reads'][0]['concepts']==['respiratory_rate']
                assert plan['health_reads'][0]['range']=={'kind':'between','start_at':'2026-09-22','end_at':'2026-09-22'}
            if text.startswith('During'):
                assert plan['health_reads'][0]['range']=={'kind':'between','start_at':'2024-01-01','end_at':'2024-12-31'}
            if text.startswith('What is'):
                assert plan['health_reads'][0]['concepts']==['apob']
                assert plan['health_reads'][0]['purpose']=='latest'
            rows.append({'request':text,'status':value['status'],'reason_codes':value['reason_codes'],'plan':plan})
        # Exercise the shipped response validation against actual local responses;
        # construction deliberately bypasses attestation for this loopback test.
        class LocalTransport:
            def post(self,remote,**kwargs):return http.post(url+'/v1/select',**kwargs)
        client=VitaSelectorClient.__new__(VitaSelectorClient)
        client.token=token;client.selector_sha256=identity();client.adapter_sha256=ADAPTER_SHA256
        client.http=LocalTransport()
        assert client.select(current_request='Show my steps yesterday',available_metrics=['steps'],
            available_record_types=[],reference_date='2026-09-23',time_zone='UTC')['status']=='selected'
        timings={}
        for label,text in [('grammar','Show my latest respiratory rate yesterday'),
                           ('semantic','During 2024, how was my sleep efficiency doing?')]:
            values=[]
            for _ in range(10):
                start=time.perf_counter()
                r=http.post(url+'/v1/select',json=body(text),headers=headers)
                assert r.status_code==200 and r.json()['status']=='selected'
                values.append((time.perf_counter()-start)*1000)
            timings[label]={'requests':10,'p50_ms':round(statistics.median(values),3),
                            'p95_ms':round(sorted(values)[int(.95*(len(values)-1))],3)}
    from fastapi.testclient import TestClient
    from server import create_app
    with TestClient(create_app(engine_factory=lambda:object(),token=token,selector_enabled=False)) as disabled:
        assert disabled.post('/v1/select',headers=headers,json=body('Show steps')).status_code==503
    result={'scope':'actual local HTTP and response adapter; synthetic data; attestation not exercised',
        'selector_sha256':identity(),'api_sha256':hashlib.sha256((ROOT/'server.py').read_bytes()).hexdigest(),
        'client_sha256':hashlib.sha256((ROOT/'examples/selector_client.py').read_bytes()).hexdigest(),
        'unauthenticated':401,'disabled':503,'response_adapter':'passed','rows':rows,'latency':timings}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checks':'passed','latency':timings},indent=2))


if __name__=='__main__':main()
