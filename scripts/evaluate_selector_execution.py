"""Independent synthetic record-ID oracle for a bounded read subset.

This fixture executor is NOT Vita's vault/database runtime. It implements only
date-bounded raw metric selection and latest-per-metric, with deterministic ties.
Expected record IDs are hand-specified, not rendered by the plan compiler.
"""
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor'),str(ROOT/'tests')]
from selector import SelectorRequest
from trained_proposal_selector import TrainedProposalSelector, identity
from contracts.vita_read_contract import read_arguments

RECORDS=[
    {'id':'leap','metric':'steps','at':'2024-02-29T10:00:00+00:00','value':1200},
    {'id':'monday','metric':'steps','at':'2026-09-21T20:00:00+00:00','value':2000},
    {'id':'tuesday','metric':'steps','at':'2026-09-22T10:00:00+00:00','value':3000},
    {'id':'late-utc','metric':'steps','at':'2026-09-22T21:30:00+00:00','value':4000},
    {'id':'wednesday','metric':'steps','at':'2026-09-23T00:15:00+00:00','value':5000},
    {'id':'resp-tuesday','metric':'respiratory_rate','at':'2026-09-22T12:00:00+00:00','value':16},
]
# IDs reflect local calendar membership; not the output of a second parser.
CASES=[
    ('latest-utc','Show my latest steps yesterday','UTC',[],['late-utc']),
    ('latest-bucharest','Show my latest steps yesterday','Europe/Bucharest',[],['tuesday']),
    ('trend-utc','Show steps yesterday','UTC',[],['late-utc','tuesday']),
    ('trend-bucharest','Show steps yesterday','Europe/Bucharest',[],['tuesday']),
    ('leap','Show steps on February 29, 2024','UTC',[],['leap']),
    ('empty','Show steps in 2028','UTC',[],[]),
    ('sparse','Show respiratory rate in 2024','UTC',[],[]),
    ('two-metrics','Show steps and respiratory rate yesterday','UTC',[],['late-utc','resp-tuesday','tuesday']),
    ('new-subject','Show respiratory rate yesterday','UTC',['Show steps from Oura in 2025'],['resp-tuesday']),
    ('source-inherited','Now do yesterday','UTC',['Show steps from Oura in 2025'],None),
    ('count-inherited','Now do yesterday','UTC',['Show the latest 3 steps readings in 2025'],None),
    ('invalid-leap','Show steps on February 29, 2025','UTC',[],None),
    ('two-windows','Show steps in 2024 and respiratory rate in 2025','UTC',[],None),
    ('absent-metric','Show steps and resting heart rate yesterday','UTC',[],None),
]


def execute(plan):
    """No paging, widening, fallback reads, aggregates, or fabricated values."""
    ids=[]
    for operation in plan['health_reads']:
        if operation.get('record_types') or operation['purpose'] not in ('latest','trend'):
            raise ValueError('Outside fixture executor coverage')
        bounds=operation['range']
        if bounds['kind']!='between':raise ValueError('Fixture requires explicit date bounds')
        zone=ZoneInfo(operation['time_zone'])
        rows=[r for r in RECORDS if r['metric'] in operation['concepts'] and
              bounds['start_at']<=datetime.fromisoformat(r['at']).astimezone(zone).date().isoformat()<=bounds['end_at']]
        if operation['purpose']=='latest':
            rows=[max([r for r in rows if r['metric']==metric],key=lambda r:(r['at'],r['id']))
                  for metric in sorted({r['metric'] for r in rows})]
        ids.extend(r['id'] for r in rows)
    return sorted(ids)


def main():
    import torch
    from typed_decisions.open_jev import OpenJev
    torch.set_num_threads(4)
    model=OpenJev.from_pretrained(str(ROOT/'model-fp16'),device='cpu')
    model.collator._ids=lambda t:model.tok(t,add_special_tokens=False)['input_ids']
    model.collator._cache.clear()
    selector=TrainedProposalSelector(model)
    start_identity=identity();rows=[]
    for name,text,zone,history,expected in CASES:
        request=SelectorRequest.model_validate({'schema_version':'vita-selector/v1',
            'available_metrics':['steps','respiratory_rate'],'available_record_types':[],
            'literature_available':False,'state':{'current_request':text,'recent_user_requests':history,
            'reference_date':'2026-09-23','time_zone':zone}})
        result=selector.select(request)
        plan=read_arguments(result,request.available_metrics,record_types=(),
            now=datetime(2026,9,23,12,tzinfo=UTC),time_zone=zone)
        actual=execute(plan) if plan else None
        rows.append({'id':name,'request':text,'history':history,'time_zone':zone,
            'expected_record_ids':expected,'actual_record_ids':actual,'passed':actual==expected,
            'reason_codes':result['reason_codes'],'plan':plan})
    assert start_identity==identity(),'Source changed during execution evaluation'
    report={'scope':'Synthetic fixture executor only; NOT the Vita database/vault runtime',
        'selector_sha256':start_identity,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'records':RECORDS,'passed':sum(r['passed'] for r in rows),'total':len(rows),'rows':rows}
    (ROOT/'evidence/selector-v4/execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('scope','passed','total')}))
    if report['passed']!=report['total']:raise SystemExit(1)


if __name__=='__main__':main()
