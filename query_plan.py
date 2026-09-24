"""One closed advisory read-plan contract. Never grants access or executes SQL."""
import hashlib
from datetime import datetime
from typing import Annotated, Literal, Union
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from selector import Metric, SelectorState
from schema_index import INDEX, metric_definition
from compat.health_range_input import _RANGE


class Closed(BaseModel):
    model_config = ConfigDict(extra='forbid',revalidate_instances='always')


Digest = Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{64}$')]
Source = Annotated[str, StringConstraints(strict=True, pattern=r'^[a-z][a-z0-9_]{0,63}$')]
# Vita accepts the complete admitted metric inventory in one health read.
# Bound it by our request inventory limit, independently of the operation budget.
MAX_METRICS_PER_READ = 512


class QueryRequest(Closed):
    schema_version: Literal['vita-selector/v2']
    state: SelectorState
    reference_time: datetime
    available_metrics: list[Metric] = Field(max_length=MAX_METRICS_PER_READ)
    available_record_types: list[Literal['profile','workouts','labs','calendar']] = Field(max_length=4)
    available_sources: list[Source] = Field(default_factory=list,max_length=32)
    literature_available: bool = Field(default=False,strict=True)

    @model_validator(mode='after')
    def check_clock_inventory(self):
        if self.reference_time.tzinfo is None:
            raise ValueError('Aware reference time required')
        if self.reference_time.astimezone(ZoneInfo(self.state.time_zone)).date().isoformat()!=self.state.reference_date:
            raise ValueError('Reference clock/date mismatch')
        for values in (self.available_metrics,self.available_record_types,self.available_sources):
            if len(values)!=len(set(values)):raise ValueError('Duplicate inventory')
        return self


class HealthRead(Closed):
    kind: Literal['health']='health'
    metrics: list[Metric]=Field(default_factory=list,max_length=MAX_METRICS_PER_READ)
    records: list[Literal['profile','workouts','labs','calendar']]=Field(default_factory=list,max_length=1)
    operation: Literal['latest','trend']='trend'
    period: dict
    source: Source|None=None
    profile_fields: list[str]=Field(default_factory=list,max_length=32)
    limit: int|None=Field(default=None,ge=1,le=200,strict=True)
    date_basis: Literal['observed_at','exam_date','started_at','next_due_date','last_done_date','current_snapshot','sleep_end_day']='observed_at'

    @model_validator(mode='after')
    def validate_semantics(self):
        parsed=_RANGE.validate_python(self.period)
        self.period=parsed.model_dump()
        if parsed.kind=='auto':raise ValueError('Unbound automatic range')
        if parsed.kind=='between':
            start=datetime.fromisoformat(parsed.start_at);end=datetime.fromisoformat(parsed.end_at)
            if len(parsed.start_at)==10 and end.tzinfo:start=start.replace(tzinfo=end.tzinfo)
            if len(parsed.end_at)==10 and start.tzinfo:end=end.replace(tzinfo=start.tzinfo)
            if bool(start.tzinfo)!=bool(end.tzinfo):raise ValueError('Mixed date bases')
            if start>end:raise ValueError('Inverted range')
            if not 1900<=start.year<=end.year<=2100:raise ValueError('Unsupported year')
            if (len(parsed.start_at)>10 and start.tzinfo is None) or (len(parsed.end_at)>10 and end.tzinfo is None):
                raise ValueError('Naive timestamp')
        if not self.metrics and not self.records:raise ValueError('Empty selection')
        if self.metrics and self.records:raise ValueError('Separate metric and record clauses required')
        if len(self.metrics)!=len(set(self.metrics)):raise ValueError('Duplicate metric')
        if self.limit is not None and (not self.records or self.records==['profile'] or self.operation!='latest'):
            raise ValueError('Limit requires latest records')
        if self.records and self.operation!='latest':raise ValueError('Record reads require latest ordering')
        if self.source and self.records:raise ValueError('Record source contract unavailable')
        if self.records==['profile']:
            if self.operation!='latest':raise ValueError('Profile snapshot only')
            if self.period!={'kind':'all_history'} or self.date_basis!='current_snapshot':raise ValueError('Profile snapshot only')
            if not self.profile_fields:raise ValueError('Explicit profile projection required')
            allowed=set(INDEX['records']['profile']['backend_fields'])
            if self.profile_fields!=['all'] and not set(self.profile_fields)<=allowed:raise ValueError('Unknown profile field')
        elif self.profile_fields:raise ValueError('Profile fields without profile')
        basis={'labs':{'exam_date'},'workouts':{'started_at'},'calendar':{'next_due_date','last_done_date'}}
        if self.records and self.records[0] in basis and self.date_basis not in basis[self.records[0]]:
            raise ValueError('Record date basis mismatch')
        if self.records and self.records[0] in ('labs','calendar') and parsed.kind=='between':
            if len(parsed.start_at)!=10 or len(parsed.end_at)!=10:raise ValueError('Record dates require whole days')
        if self.metrics and self.date_basis not in ('observed_at','sleep_end_day'):raise ValueError('Metric date basis mismatch')
        if self.date_basis=='sleep_end_day':
            if not self.metrics or any(m not in ('total_sleep','sleep_efficiency') for m in self.metrics):raise ValueError('Night basis requires sleep')
            if parsed.kind!='between' or len(parsed.start_at)!=10 or parsed.start_at!=parsed.end_at:
                raise ValueError('Sleep episode projection requires one explicit day')
        if len(self.profile_fields)!=len(set(self.profile_fields)):raise ValueError('Duplicate profile field')
        return self


class ResearchRead(Closed):
    kind: Literal['research']='research'
    targets: list[Literal['sleep','physical_activity','apob','ldl_cholesterol','glucose','cardiometabolic_health']]=Field(min_length=1,max_length=6)
    interventions: list[Literal['diet','exercise']]=Field(default_factory=list,max_length=2)
    goal: Literal['improve','lower']='improve'

    def question(self):
        names={'sleep':'sleep duration and quality','physical_activity':'physical activity',
               'apob':'ApoB','ldl_cholesterol':'LDL cholesterol','glucose':'glucose control',
               'cardiometabolic_health':'cardiometabolic health'}
        targets=' and '.join(names[x] for x in self.targets)
        intervention=' and '.join({'diet':'dietary changes','exercise':'exercise'}[x] for x in self.interventions) or 'interventions'
        return f'For adults, what systematic reviews and randomized trials evaluate {intervention} to {self.goal} {targets}? Include outcomes, studied populations and applicability limits.'


Query=Annotated[Union[HealthRead,ResearchRead],Field(discriminator='kind')]


def query_operation_count(query):
    return max(1,(len(query.metrics)+MAX_METRICS_PER_READ-1)//MAX_METRICS_PER_READ) if isinstance(query,HealthRead) else 1


class QueryPlan(Closed):
    schema_version: Literal['vita-query-plan/v2']='vita-query-plan/v2'
    status: Literal['planned','handoff']
    queries: list[Query]=Field(default_factory=list,max_length=8)
    reason_codes: list[str]=Field(default_factory=list)
    advisory: Literal[True]=True
    time_zone: str
    selector_sha256: Digest
    adapter_sha256: Digest
    model_revision: Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{40}$')]
    request_sha256: Digest
    diagnostics: dict=Field(default_factory=dict)

    @model_validator(mode='after')
    def complete_or_empty(self):
        ZoneInfo(self.time_zone)
        if self.status=='planned' and (not self.queries or self.reason_codes):raise ValueError('Incomplete planned response')
        if self.status=='handoff' and (self.queries or not self.reason_codes):raise ValueError('Handoff must be inert')
        return self


def request_identity(request):
    return hashlib.sha256(request.model_dump_json().encode()).hexdigest()


def canonical_metric(metric):
    return INDEX['inventory_to_catalog'].get(metric,metric)


def validate_inventory(plan, request):
    if plan.request_sha256!=request_identity(request) or plan.time_zone!=request.state.time_zone:
        raise ValueError('Plan request binding mismatch')
    for query in plan.queries:
        if isinstance(query,ResearchRead):
            if not request.literature_available:raise ValueError('Research not enabled')
            continue
        if not set(query.metrics)<=set(request.available_metrics):raise ValueError('Metric unavailable')
        if not set(query.records)<=set(request.available_record_types):raise ValueError('Record category unavailable')
        if query.source:
            if query.source not in request.available_sources:raise ValueError('Source unavailable')
            for metric in query.metrics:
                definition=metric_definition(metric)
                if not definition or query.source not in definition['valid_sources']:raise ValueError('Source invalid for metric')


def compile_batch(plan, request):
    """Translate only represented Vita capabilities; never discard a qualifier."""
    plan=QueryPlan.model_validate(plan)
    validate_inventory(plan,request)
    if plan.status!='planned':return None
    batch={'health_reads':[],'literature_reads':[],'required_operation_ids':[]}
    for query in plan.queries:
        if isinstance(query,ResearchRead):
            operations=[{'question':query.question(),'task':'evidence_summary','depth':'fast',
                         'subject_basis':'general_overview_research','basis_source_ids':[]}]
            field='literature_reads'
        else:
            operations=[];field='health_reads'
            for i in range(0,max(1,len(query.metrics)),MAX_METRICS_PER_READ):
                op={'concepts':list(dict.fromkeys(canonical_metric(m) for m in query.metrics[i:i+MAX_METRICS_PER_READ])),'record_types':query.records,
                    'profile_fields':query.profile_fields,'purpose':query.operation,
                    'range':query.period,'time_zone':plan.time_zone,'grouping':'none' if query.operation=='latest' else 'auto',
                    'result_view':'summary'}
                if query.source:op['source']=query.source
                if query.date_basis=='sleep_end_day':
                    # No backend index exists for episode end time. The consumer
                    # must prove raw all-history coverage and project episodes;
                    # this batch alone is NOT a completed user query.
                    op.update(range={'kind':'all_history'},purpose='trend',grouping='none',result_view='detail')
                if query.records and query.records!=['profile']:
                    op.update(record_detail_dataset=INDEX['records'][query.records[0]]['dataset'],record_detail_limit=query.limit or 200)
                if query.records==['calendar']:op['calendar_date_basis']=query.date_basis
                operations.append(op)
        for operation in operations:
            number=len(batch['required_operation_ids'])+1
            batch['required_operation_ids'].append(number);batch[field].append({'operation_id':number,**operation})
    if len(batch['required_operation_ids'])>8:raise ValueError('Operation budget exceeded')
    return batch
