from datetime import datetime, UTC
import copy
import pytest
from plan_adapter import build_plan,audit_delivery,CATALOG,canonical_metrics,project_required_summaries
from selector import SelectorRequest,select

NOW=datetime(2026,9,23,12,tzinfo=UTC)

def make(text,inventory=None,**kw):
    inventory=inventory or list(CATALOG['metrics'])
    response=select(SelectorRequest.model_validate({'schema_version':'vita-selector/v1','available_metrics':inventory,
        'literature_available':True,'state':{'current_request':text,'reference_date':'2026-09-23','time_zone':'UTC'}}))
    return build_plan(response,inventory,now=NOW,time_zone='UTC',literature_available=True,
                      record_types=('profile','workouts','labs','calendar'),**kw)


def test_domain_budget_is_explicit_not_alphabetic_all_inventory():
    plan=make('Analyze me');reads=plan['batch']['health_reads']
    assert len(plan['batch']['required_operation_ids'])==8
    assert {r['domain'] for r in plan['coverage']['required']}=={'labs','body','sleep','activity','recovery','vitals','record:profile'}
    assert sum(len(r.get('concepts',[])) for r in reads)==32
    assert len(plan['coverage']['deferred'])==91
    assert not plan['coverage']['all_requested_delivered']


def test_explicit_window_does_not_fetch_old_body_measurements():
    plan=make('Show my weight last month',['weight'])
    read=plan['batch']['health_reads'][0]
    assert read['range']=={'kind':'between','start_at':'2026-08-01','end_at':'2026-08-31'}
    assert read['purpose']=='trend'
    assert len(plan['batch']['health_reads'])==1


def test_narrow_latest_is_raw_and_general_research_has_no_health():
    plan=make('Show my latest weight',['weight'])
    read=plan['batch']['health_reads'][0]
    assert read['purpose']=='latest' and read['grouping']=='none' and read['range']=={'kind':'all_history'}
    assert make('Find research on sleep and physical activity')['batch']['health_reads']==[]


def test_unknown_catalog_and_insufficient_budget_fail_explicitly():
    with pytest.raises(ValueError):canonical_metrics(['invented_metric'])
    assert make('Analyze me',operation_budget=3)['status']=='unsupported'
    assert make('Analyze me',summary_token_budget=8000)['reason']=='insufficient_summary_budget'


def test_retained_but_not_admitted_is_not_delivered():
    plan=make('Show my latest weight',['weight'])
    assert not audit_delivery(plan,[])['planned_summaries_delivered']
    empty=[{'operation_id':1,'source_id':1,'status':'empty','result':{'claim_scope':{'purpose':'latest','requested_range':{'range_mode':'all'}},'coverage':{'scope_complete':False},'evidence':[]}}]
    assert audit_delivery(plan,empty)['operations'][0]['concept_states']['weight']['state']=='not_delivered'
    empty[0]['result']['coverage']['scope_complete']=True
    assert audit_delivery(plan,empty)['operations'][0]['concept_states']['weight']['state']=='authorized_empty'


def test_no_permission_inferred_from_selected_record():
    inventory=['weight'];response=select(SelectorRequest.model_validate({'schema_version':'vita-selector/v1','available_metrics':inventory,
        'literature_available':True,'state':{'current_request':'Analyze me','reference_date':'2026-09-23','time_zone':'UTC'}}))
    plan=build_plan(response,inventory,now=NOW,time_zone='UTC',record_types=(),literature_available=False)
    assert not any(r.get('record_types') for r in plan['batch']['health_reads'])
    assert {r['record_type'] for r in plan['coverage']['deferred']}=={'profile','workouts','labs','calendar'}
