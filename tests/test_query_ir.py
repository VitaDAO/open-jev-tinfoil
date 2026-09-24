import pytest
from pydantic import ValidationError
from query_ir import QueryIR,compile_ir


def compile(*clauses,available=('steps','total_sleep','weight'),capabilities=('health',)):
    return compile_ir({'schema_version':'vita-query-ir/v1','inventory_version':'synthetic120',
                       'time_zone':'UTC','clauses':list(clauses)},available_metrics=available,
                      authorized_record_types=('profile','workouts','labs','calendar'),enabled_read_capabilities=capabilities)

def health(**kw):
    return {'kind':'health','operation':'trend','metrics':['steps'],'time_range':{'kind':'calendar','period':'week'},**kw}


def test_source_filter_and_exclusion_are_supported_not_intrinsically_unsafe():
    result=compile(health(metrics=[],domains=['sleep'],exclude_metrics=['sleep_efficiency'],sources=['oura']),
                   available=('total_sleep','sleep_efficiency'))
    read=result['batch']['health_reads'][0]
    assert read['concepts']==['total_sleep'] and read['source']=='oura'
    assert read['grouping']=='auto' and read['range']=={'kind':'calendar','period':'week'}


def test_comparison_periods_are_separate_parallel_scopes():
    result=compile(health(time_range={'kind':'between','start_at':'2026-08-01','end_at':'2026-08-31'}),
                   health(time_range={'kind':'between','start_at':'2026-09-01','end_at':'2026-09-23'}))
    reads=result['batch']['health_reads']
    assert len(reads)==2 and reads[0]['range']!=reads[1]['range']
    assert result['batch']['required_operation_ids']==[1,2]


@pytest.mark.parametrize('override,code',[
    ({'operation':'latest','limit':5},'latest_n_not_representable_by_health_operation'),
    ({'aggregation':'sum'},'aggregate_not_explicitly_supported'),
    ({'date_basis':'uploaded_at'},'upload_timestamp_filter_not_supported'),
    ({'operation':'latest','grouping':'week'},'grouped_latest_is_not_raw'),
    ({'operation':'latest','aggregation':'mean'},'aggregate_conflicts_with_latest_raw'),
])
def test_never_downgrade_unsupported_semantics(override,code):
    result=compile(health(**override))
    assert result['status']=='handoff' and result['batch'] is None
    assert result['reason_codes'][0]['code']==code


def test_explicit_new_inventory_id_is_usable_without_invented_metadata():
    result=compile(health(metrics=['new_registered_metric']),available=('new_registered_metric',))
    assert result['batch']['health_reads'][0]['concepts']==['new_registered_metric']
    result=compile(health(metrics=[],domains=['sleep']),available=('new_registered_metric',))
    assert result['reason_codes'][0]['code']=='domain_metadata_required_for_new_inventory'


def test_correlation_preserves_grouping_and_basis():
    result=compile(health(operation='correlate',metrics=['steps','total_sleep'],grouping='day',correlation_basis='sleep_start_day'))
    read=result['batch']['health_reads'][0]
    assert (read['purpose'],read['grouping'],read['correlation_basis'])==('correlate','day','sleep_start_day')


def test_profile_fields_calendar_basis_and_workout_count():
    result=compile(health(metrics=[],record_types=['profile'],profile_fields=['goals'],operation='latest',time_range={'kind':'all_history'}))
    assert result['batch']['health_reads'][0]['profile_fields']==['goals']
    result=compile(health(metrics=[],record_types=['calendar'],calendar_date_basis='last_done_date'))
    assert result['batch']['health_reads'][0]['calendar_date_basis']=='last_done_date'
    result=compile(health(metrics=[],record_types=['workouts'],aggregation='count'))
    assert result['status']=='compiled'


def test_memory_conversation_and_actions_keep_authorization_path():
    result=compile({'kind':'memory_recall','query':'sleep goal'},capabilities=('memory',))
    assert result['batch']['memory_recalls'][0]['query']=='sleep goal'
    result=compile({'kind':'conversation_search','query':'fasting','match_mode':'phrase'},capabilities=('conversation',))
    assert result['batch']['conversation_searches'][0]['match_mode']=='phrase'
    result=compile(health(),{'kind':'action_proposal','capability':'change_goal','reason':'existing_authorization_path'})
    assert result['status']=='handoff' and result['batch'] is None


def test_bounds_and_closed_schema():
    with pytest.raises(ValidationError):
        QueryIR.model_validate({'inventory_version':'v1','time_zone':'UTC','clauses':[health(sql='select *')]})
    result=compile(*[health() for _ in range(8)],capabilities=())
    assert result['status']=='handoff'
