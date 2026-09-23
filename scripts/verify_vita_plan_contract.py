"""Synthetic plan/projection integration against the actual read-only Vita modules.

Run with the Vita environment, PYTHONDONTWRITEBYTECODE=1 and VITA_SOURCE_ROOT.
Does not instantiate a database, provider, app runner, or real-data client.
"""
import copy
import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
VITA=Path(os.environ['VITA_SOURCE_ROOT']).resolve()
sys.path[:0]=[str(ROOT),str(VITA/'vita/py/src')]
from plan_adapter import build_plan,audit_delivery,CATALOG,canonical_metrics,project_required_summaries
from selector import SelectorRequest,select
from vita_agent.kernel.source_batch_contracts import SourceBatchRequest,SourceBatchResult,SourceOperationResult
from vita_agent.kernel.health_range_input import normalize_health_ranges
from vita_agent.kernel.health_result_pages import result_page,tokens
from vita_agent.kernel.source_batch_render import serialize_model_batch

assert str(VITA) in inspect.getfile(SourceBatchRequest)
NOW=datetime(2026,9,23,12,tzinfo=UTC)
INVENTORY=list(CATALOG['metrics'])

def selection(text,inventory=INVENTORY):
    return select(SelectorRequest.model_validate({'schema_version':'vita-selector/v1','available_metrics':inventory,
       'literature_available':True,'state':{'current_request':text,'reference_date':'2026-09-23','time_zone':'UTC'}}))

def plan(text,inventory=INVENTORY,**kw):
    return build_plan(selection(text,inventory),inventory,now=NOW,time_zone='UTC',
                      record_types=('profile','workouts','labs','calendar'),literature_available=True,**kw)


def retained(operation, *, devices=2):
    payload={'schema':'vita-manager-health-source/v1','result_view':'summary',
             'claim_scope':{'purpose':operation.purpose,'requested_range':{key:getattr(operation,key) for key in ('range_mode','start_at','end_at','history_days','history_months') if getattr(operation,key) is not None}},'coverage':{'scope_complete':True},'series':[]}
    for concept in operation.concepts:
        unit=CATALOG['metrics'][concept]['unit']
        for source in ['withings','oura'][:devices]:
            latest=operation.purpose=='latest'
            # Values and sources deliberately synthetic; this is projection input,
            # not proof that either real provider emits all these concepts.
            dates=['2026-07-01T12:00:00Z'] if latest else ['2026-09-01T00:00:00Z','2026-09-08T00:00:00Z']
            values=[70.0] if latest else [68.0,71.0]
            payload['series'].append({'concept':concept,'source':source,'unit':unit,
                'grouping':'none' if latest else 'week','scope_complete':True,
                'points':list(map(list,zip(dates,values))),'sample_counts':[1] if latest else [7,7],
                'statistics':{'mean':70,'minimum':68,'maximum':71,'sample_count':1 if latest else 14,'complete':True}})
    if 'profile' in operation.record_types:
        payload['records']={'profile':{'status':'ok','scope_complete':True,'records':[{'goals':['sleep'],'observed_at':'2026-09-01'}]}}
    return payload


def deliver(planned,*,allowance=None):
    request=SourceBatchRequest.model_validate(normalize_health_ranges(planned['batch']))
    state=SimpleNamespace(results_by_source_id={})
    operations=[]
    for operation in request.health_reads:
        full=retained(operation)
        required=next(x for x in planned['coverage']['required'] if x['operation_id']==operation.operation_id)
        page=result_page(full,budget=allowance or required['summary_token_allowance'],canonical=full)
        state.results_by_source_id[operation.operation_id]=SimpleNamespace(result=full)
        operations.append(SourceOperationResult(operation_id=operation.operation_id,source_id=operation.operation_id,
                                                kind='health_read',status='ok',result=page))
    wire=serialize_model_batch(SourceBatchResult(batch_number=1,outcome='settled',operations=tuple(operations)),state)
    output=json.loads(wire)['operations']
    return request,output,tokens(json.loads(wire)),state


def main():
    started=time.perf_counter();broad=plan('Analyze me')
    assert len(INVENTORY)>70 and len(broad['batch']['required_operation_ids'])<=8
    request,visible,used,state=deliver(broad)
    audit=audit_delivery(broad,visible)
    domains={item['domain']:item for item in audit['operations']}
    essential={'labs','body','sleep','activity','recovery','vitals'}
    assert essential<=domains.keys()
    for domain in essential:
        assert any(v['state']=='present' for v in domains[domain]['concept_states'].values()),domain
    for operation in request.health_reads:
        if operation.concepts:
            assert len(operation.concepts)<=8
            domain=next(x['domain'] for x in broad['coverage']['required'] if x['operation_id']==operation.operation_id)
            assert operation.range_mode=='all' if domain in ('labs','body') else operation.history_days==30
            assert operation.purpose=='latest' if domain in ('labs','body') else operation.purpose=='trend'
    assert used<=broad['coverage']['summary_token_budget']
    from vita_agent.kernel.health_summary_evidence import summary_evidence
    retained_ops=[]
    for op in visible:
        retained_ops.append({**op,'result':state.results_by_source_id[op['source_id']].result,
            'admitted_record_evidence':op['result'].get('evidence',[])})
    proposed=project_required_summaries(broad,retained_ops,summarize_series=summary_evidence,token_count=tokens)
    assert proposed['status']=='projected'
    assert proposed['tokens']<=14000
    assert proposed['audit']['planned_summaries_delivered']
    # A >70-concept alphabetic bulk page cannot establish overall coverage.
    bulk=copy.deepcopy(broad);bulk['batch']['health_reads']=[{'operation_id':1,'concepts':sorted(INVENTORY),'purpose':'trend','range':{'kind':'all_history'},'time_zone':'UTC'}]
    bulk['batch']['literature_reads']=[];bulk['batch']['required_operation_ids']=[1]
    bulk['coverage']['required']=[{**broad['coverage']['required'][0],'operation_id':1,'domain':'bulk','concepts':sorted(INVENTORY),'purpose':'trend','summary_token_allowance':2000}]
    _,bulk_visible,_,_=deliver(bulk)
    bulk_audit=audit_delivery(bulk,bulk_visible)
    assert not bulk_audit['planned_summaries_delivered']
    # Retained-but-not-admitted and partial pages stay explicit.
    assert not audit_delivery(broad,visible[:1])['planned_summaries_delivered']
    _,partial,_,_=deliver(broad,allowance=2000)
    assert not audit_delivery(broad,partial)['planned_summaries_delivered']
    # Same explicit window on every measurement, including latest sparse body data.
    for text in ['Show my health summary last month','Show my health summary this week']:
        explicit=plan(text);validated=SourceBatchRequest.model_validate(normalize_health_ranges(explicit['batch']))
        for op in validated.health_reads:
            if not op.concepts:continue
            assert op.range_mode=='between' if 'last month' in text else op.range_mode=='current_week'
            if 'last month' in text:assert (op.start_at,op.end_at)==('2026-08-01','2026-08-31')
            assert op.purpose=='trend'
    narrow=plan('Show my latest weight reading', ['weight'])
    assert len(narrow['batch']['health_reads'])==1
    assert narrow['batch']['health_reads'][0]['grouping']=='none'
    # Wrong grouped latest cannot count as delivered raw observation.
    wrong=copy.deepcopy(visible)
    labs=next(op for op in wrong if op['operation_id']==domains['labs']['operation_id'])
    for e in labs['result']['evidence']:
        if e['kind']=='series':e['grouping']='week'
    assert any(v['state']=='wrong_aggregation' for r in audit_delivery(broad,wrong)['operations'] for v in r['concept_states'].values())
    # No merging of source evidence. Both source rows reach a delivered concept.
    assert any(len(v['sources'])==2 for r in audit['operations'] for v in r['concept_states'].values() if v['state']=='present')
    # Failed source and complete empty scope differ from pagination absence.
    failure=[{'operation_id':1,'source_id':1,'status':'unavailable','result':{}}]
    assert audit_delivery(broad,failure)['operations'][0]['outcome']=='unavailable'
    empty=[{'operation_id':1,'source_id':1,'status':'empty','result':{'claim_scope':{'purpose':'latest','requested_range':{'range_mode':'all'}},'coverage':{'scope_complete':True},'evidence':[]}}]
    assert all(v['state']=='authorized_empty' for v in audit_delivery(broad,empty)['operations'][0]['concept_states'].values())
    assert canonical_metrics(['LDL Cholesterol','hdl_cholesterol'])=={'LDL Cholesterol':'ldl','hdl_cholesterol':'hdl'}
    research=plan('Find research on sleep and physical activity')
    assert research['batch']['health_reads']==[]
    SourceBatchRequest.model_validate(normalize_health_ranges(research['batch']))
    files=['vita/py/src/vita_agent/kernel/source_batch_contracts.py','vita/py/src/vita_agent/kernel/health_result_pages.py','vita/py/src/vita_agent/kernel/health_summary_evidence.py','vita/py/src/vita_agent/kernel/source_batch_render.py']
    return {'actual_vita_source':str(VITA),'source_sha256':{f:hashlib.sha256((VITA/f).read_bytes()).hexdigest() for f in files},
            'inventory_metrics':len(INVENTORY),'operation_count':len(broad['batch']['required_operation_ids']),
            'selected_metric_count':sum(len(op.concepts) for op in request.health_reads),
            'synthetic_retained_series':sum(len(x.result.get('series',[])) for x in state.results_by_source_id.values()),
            'model_visible_tokens':used,'domains_with_initial_evidence':sorted(essential),
            'all_planned_summaries_delivered':audit['planned_summaries_delivered'],
            'proposed_projection_tokens':proposed['tokens'],
            'proposed_planned_summaries_delivered':proposed['audit']['planned_summaries_delivered'],
            'proposed_projection_installed_in_vita':False,
            'all_requested_delivered':audit['all_requested_delivered'],'deferred_count':len(broad['coverage']['deferred']),
            'elapsed_ms':round((time.perf_counter()-started)*1000,2),
            'tests':'typed validation, canonical aliases, broad bounded parallel reads, exact windows, latest vs averages, multiple sources, actual summary pagination and final serializer, omitted pages, unavailable and empty outcomes',
            'limitations':'Synthetic post-query payloads exercise actual projection; no database/provider query or resolver execution, dedup/import-date oracle, clinical correctness, or end-to-end Fable acceptance is established.',
            'plan':broad,'audit':audit}

if __name__=='__main__':print(json.dumps(main(),indent=2))
