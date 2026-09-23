"""Typed query IR and compiler for existing Vita tool semantics.

This schema is broader than the experimental learned decoder. It does not claim
that the current model can produce it. Unknown clauses hand off as a whole;
never downgrade N to latest, average to raw, or comparison to trend.
"""
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator
from selector import Metric
from compat.health_range_input import _RANGE
from plan_adapter import CATALOG, POLICY, PROFILE_FIELDS, RECORDS, RESEARCH

class Closed(BaseModel):
    model_config=ConfigDict(extra='forbid')

class HealthClause(Closed):
    kind:Literal['health']='health'
    operation:Literal['latest','trend','baseline','compare','correlate']
    metrics:list[Metric]=Field(default_factory=list,max_length=120)
    domains:list[Literal['labs','body','sleep','activity','recovery','vitals']]=Field(default_factory=list,max_length=6)
    exclude_metrics:list[Metric]=Field(default_factory=list,max_length=120)
    sources:list[Annotated[str,Field(min_length=1,max_length=64)]]=Field(default_factory=list,max_length=4)
    time_range:dict
    aggregation:Literal['source_default','raw','mean','minimum','maximum','sum','count','median']='source_default'
    grouping:Literal['auto','none','day','week','month']='auto'
    order:Literal['source_default','observed_at_desc','uploaded_at_desc']='source_default'
    limit:int|None=Field(default=None,ge=1,le=200)
    record_types:list[Literal['profile','workouts','labs','calendar']]=Field(default_factory=list,max_length=4)
    profile_fields:list[Annotated[str,Field(min_length=1,max_length=64)]]=Field(default_factory=list,max_length=32)
    record_detail_limit:int=Field(default=0,ge=0,le=200)
    calendar_date_basis:Literal['next_due_date','last_done_date']|None=None
    date_basis:Literal['source_default','observed_at','exam_date','uploaded_at']='source_default'
    correlation_basis:Literal['recorded_day','sleep_start_day']='recorded_day'
    analyses:list[Literal['weekday_profile']]=Field(default_factory=list,max_length=1)
    context_reference:Literal['current','inherited','corrected']='current'

    @model_validator(mode='after')
    def closed_range(self):
        _RANGE.validate_python(self.time_range)
        if set(self.metrics)&set(self.exclude_metrics):raise ValueError('Contradictory metric selection')
        return self

class LiteratureClause(Closed):
    kind:Literal['literature']='literature'
    topic:Literal['broad_overview','sleep_activity','cardiometabolic','custom']
    custom_question:str|None=Field(default=None,max_length=2000)

class MemoryClause(Closed):
    kind:Literal['memory_recall']='memory_recall'
    query:str=Field(min_length=1,max_length=2000)
    limit:int=Field(default=5,ge=1,le=20)
    tags:list[str]=Field(default_factory=list,max_length=20)

class ConversationClause(Closed):
    kind:Literal['conversation_search']='conversation_search'
    query:str=Field(max_length=2000)
    match_mode:Literal['any','all','phrase']='any'
    limit:int=Field(default=5,ge=1,le=8)
    start_at:str|None=None
    end_at:str|None=None
    conversation_id:str|None=None
    sort_order:Literal['newest_first','oldest_first']='newest_first'
    speaker:Literal['all','user','assistant']='all'

class HandoffClause(Closed):
    kind:Literal['action_proposal','no_read','unsupported']
    capability:str=Field(min_length=1,max_length=64)
    reason:str=Field(min_length=1,max_length=128)

Clause=Annotated[Union[HealthClause,LiteratureClause,MemoryClause,ConversationClause,HandoffClause],Field(discriminator='kind')]

class QueryIR(Closed):
    schema_version:Literal['vita-query-ir/v1']='vita-query-ir/v1'
    clauses:list[Clause]=Field(min_length=1,max_length=8)
    time_zone:str=Field(min_length=1,max_length=64)
    inventory_version:str=Field(min_length=1,max_length=128)


def compile_ir(ir, *, available_metrics, authorized_record_types=(), enabled_read_capabilities=('health',),
               operation_budget=8):
    """Structural compilation only; actual broker/tool authorization remains mandatory."""
    if not isinstance(ir,QueryIR):ir=QueryIR.model_validate(ir)
    if not 1<=operation_budget<=8:raise ValueError('Invalid operation budget')
    from zoneinfo import ZoneInfo
    ZoneInfo(ir.time_zone)
    inventory=set(available_metrics);reasons=[]
    batch={'health_reads':[],'literature_reads':[],'memory_recalls':[],'conversation_searches':[], 'required_operation_ids':[]}
    semantic=[]
    for index,clause in enumerate(ir.clauses):
        def reject(code):reasons.append({'clause':index,'code':code})
        if isinstance(clause,HandoffClause):
            reject('existing_'+clause.kind+'_path');continue
        capability={'health':'health','literature':'literature','memory_recall':'memory','conversation_search':'conversation'}[clause.kind]
        if capability not in enabled_read_capabilities:
            reject('capability_not_enabled');continue
        operations=[];field=None
        if isinstance(clause,HealthClause):
            if not set(clause.metrics)<=inventory or not set(clause.exclude_metrics)<=inventory:
                reject('metric_not_in_available_inventory');continue
            chosen=set(clause.metrics)
            if clause.domains:
                unknown=inventory-set(POLICY['domains'])
                if unknown:
                    reject('domain_metadata_required_for_new_inventory');continue
                chosen|={m for m in inventory if POLICY['domains'][m] in clause.domains}
            chosen-=set(clause.exclude_metrics)
            if not set(clause.record_types)<=set(authorized_record_types):
                reject('record_authorization_or_availability_missing');continue
            if not chosen and not clause.record_types:
                reject('empty_explicit_selection');continue
            if clause.limit not in (None,1):
                reject('latest_n_not_representable_by_health_operation');continue
            if clause.limit==1 and clause.operation!='latest':
                reject('limit_not_representable_for_operation');continue
            if clause.order=='uploaded_at_desc' or clause.date_basis=='uploaded_at':
                reject('upload_timestamp_filter_not_supported');continue
            if clause.order=='observed_at_desc' and clause.operation!='latest':
                reject('sort_not_representable_for_operation');continue
            if clause.aggregation in ('sum','median') or (clause.aggregation=='count' and (chosen or not clause.record_types)):
                reject('aggregate_not_explicitly_supported');continue
            if clause.aggregation in ('mean','minimum','maximum') and clause.operation=='latest':
                reject('aggregate_conflicts_with_latest_raw');continue
            if clause.operation=='latest' and clause.grouping not in ('auto','none'):
                reject('grouped_latest_is_not_raw');continue
            if clause.aggregation=='raw' and clause.grouping not in ('auto','none'):
                reject('raw_conflicts_with_grouping');continue
            if clause.profile_fields and ('profile' not in clause.record_types or not set(clause.profile_fields)<=set(PROFILE_FIELDS)):
                reject('profile_field_not_in_supported_projection');continue
            if chosen and clause.record_types and len(chosen)>8:
                reject('mixed_record_scope_requires_separate_clauses');continue
            if clause.record_detail_limit and (len(clause.record_types)!=1 or clause.record_types[0]=='profile'):
                reject('record_detail_requires_one_dataset');continue
            if clause.sources and clause.record_types:
                reject('record_source_filter_not_verified');continue
            if clause.date_basis=='exam_date' and clause.record_types!=['labs']:
                reject('exam_date_requires_lab_report_dataset');continue
            if clause.date_basis=='observed_at' and clause.record_types:
                reject('record_timestamp_basis_not_verified');continue
            if 'calendar' in clause.record_types and clause.time_range['kind']!='all_history' and clause.calendar_date_basis is None:
                reject('calendar_date_basis_required');continue
            for source in clause.sources:
                if any(m not in CATALOG['metrics'] or source not in CATALOG['metrics'][m]['valid_sources'] for m in chosen):
                    reject('source_not_in_metric_schema')
            if any(r['clause']==index for r in reasons):continue
            # Grouping remains exactly requested; do not force weekly buckets on
            # narrow windows. Stats are computed by existing source tools.
            if clause.operation=='correlate' and len(chosen)>8:
                reject('joint_scope_exceeds_bounded_operation');continue
            metric_chunks=[sorted(chosen)[i:i+8] for i in range(0,len(chosen),8)] or [[]]
            for source,concept_chunk in ((source,chunk) for source in clause.sources or [None] for chunk in metric_chunks):
                operation={'purpose':clause.operation,'concepts':concept_chunk,'record_types':clause.record_types,
                    'profile_fields':clause.profile_fields,'range':clause.time_range,'time_zone':ir.time_zone,
                    'grouping':'none' if clause.operation=='latest' else clause.grouping,
                    'result_view':'detail' if clause.aggregation=='raw' else 'summary',
                    'correlation_basis':clause.correlation_basis,'analyses':clause.analyses}
                if source is not None:operation['source']=source
                if clause.calendar_date_basis:operation['calendar_date_basis']=clause.calendar_date_basis
                if clause.record_detail_limit:
                    operation.update(record_detail_limit=clause.record_detail_limit,record_detail_dataset=RECORDS[clause.record_types[0]])
                operations.append(operation)
            field='health_reads'
        elif isinstance(clause,LiteratureClause):
            if clause.topic=='custom':reject('existing_depersonalized_research_path_required');continue
            operations=[{'question':RESEARCH[clause.topic],'task':'evidence_summary','depth':'fast',
                         'subject_basis':'general_overview_research','basis_source_ids':[]}];field='literature_reads'
        elif isinstance(clause,MemoryClause):
            operations=[clause.model_dump(exclude={'kind'})];field='memory_recalls'
        elif isinstance(clause,ConversationClause):
            operations=[clause.model_dump(exclude={'kind'},exclude_none=True)];field='conversation_searches'
        for operation in operations:
            identity=len(batch['required_operation_ids'])+1
            operation['operation_id']=identity;batch[field].append(operation);batch['required_operation_ids'].append(identity)
            semantic.append({'operation_id':identity,'clause':index,'context_reference':getattr(clause,'context_reference','current'),
                             'requested_aggregation':getattr(clause,'aggregation',None),'stage':'selected_not_executed'})
    if len(batch['required_operation_ids'])>operation_budget:reasons.append({'clause':None,'code':'operation_budget_exceeded'})
    if reasons:
        return {'schema_version':'vita-compiled-query/v1','status':'handoff','reason_codes':reasons,'batch':None,
                'advisory':True,'recovery':'Pass original request/context to existing model once; no downgraded partial execution.'}
    return {'schema_version':'vita-compiled-query/v1','status':'compiled','batch':batch,'semantics':semantic,
            'inventory_version':ir.inventory_version,'advisory':True}
