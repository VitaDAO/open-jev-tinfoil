"""Bounded consumer for the single query plan; caller supplies authorized I/O.

No credentials, database connection or provider selection lives here. The caller
must use its capability broker for EVERY operation and continuation. A compiled
batch is only an acquisition request; this module owns completion checks.
"""
import copy
from datetime import datetime
from zoneinfo import ZoneInfo

from query_plan import QueryPlan, HealthRead, ResearchRead, compile_batch, validate_inventory
from schema_index import INDEX


class IncompleteRead(ValueError):
    """Internal, non-sensitive completion failure code."""



def project_sleep(query,payload,time_zone):
    coverage=payload.get('coverage') or {}
    if coverage.get('scope_complete') is not True or coverage.get('detail_complete') is not True:
        raise IncompleteRead('sleep_episode_scan_incomplete')
    if payload.get('gaps') or payload.get('denied') or payload.get('scan_continuation_ref'):
        raise IncompleteRead('sleep_episode_source_gap')
    zone=ZoneInfo(time_zone);projected=[];represented=set()
    for series in payload.get('series',[]):
        if series.get('concept') not in query.metrics:raise IncompleteRead('unexpected_metric')
        if query.source and series.get('source')!=query.source:raise IncompleteRead('unexpected_source')
        points=series.get('points',[]);episodes=series.get('sleep_episodes',[])
        if (series.get('scope_complete') is not True or series.get('detail_complete') is not True
                or series.get('grouping')!='none' or series.get('sample_counts')
                or len(points)!=len(episodes)):
            raise IncompleteRead('sleep_episode_timing_incomplete')
        selected=[];selected_episodes=[]
        for point,episode in zip(points,episodes):
            if not isinstance(episode,dict) or type(episode.get('schema_version')) is not int or episode['schema_version']!=1:raise IncompleteRead('sleep_episode_timing_incomplete')
            if not all(isinstance(episode.get(k),str) and 'T' in episode[k] for k in ('start_at','end_at')):
                raise IncompleteRead('invalid_sleep_episode')
            start=datetime.fromisoformat(episode['start_at'].replace('Z','+00:00'))
            end=datetime.fromisoformat(episode['end_at'].replace('Z','+00:00'))
            if start.tzinfo is None or end.tzinfo is None or start>=end:raise IncompleteRead('invalid_sleep_episode')
            day=end.astimezone(zone).date().isoformat()
            if query.period['start_at']<=day<=query.period['end_at']:
                # Without a main-sleep flag a same-day episode may be a nap.
                # Never label those episodes, or multiple overnight episodes,
                # as the single preceding night's sleep.
                if start.astimezone(zone).date()==end.astimezone(zone).date():
                    raise IncompleteRead('ambiguous_sleep_episode')
                selected.append(point);selected_episodes.append(episode)
        if len(selected)>1:raise IncompleteRead('ambiguous_sleep_episode')
        represented.add(series['concept'])
        # Never carry all-history statistics, facts, dates or stale references
        # into a response claiming the requested episode-day window.
        projected.append({k:copy.deepcopy(series[k]) for k in ('concept','source','unit') if k in series} |
            {'points':selected,'sleep_episodes':selected_episodes,'scope_complete':True,'detail_complete':True,'grouping':'none'})
    if set(query.metrics)-represented:
        raise IncompleteRead('sleep_metric_coverage_missing')
    return {'series':projected,'period':query.period,'date_basis':'sleep_end_day',
            'scope_complete':True,'detail_complete':True}


def profile_projection(query,payload):
    record=(payload.get('records') or {}).get('profile',{})
    if record.get('status')!='ok' or record.get('scope_complete') is not True:
        raise IncompleteRead('profile_scope_incomplete')
    expected=set(INDEX['records']['profile']['backend_fields'] if query.profile_fields==['all'] else query.profile_fields)
    if set(record.get('requested_fields',[]))!=expected:raise IncompleteRead('profile_projection_mismatch')
    scope=record.get('field_scope') or {}
    returned=set(scope.get('returned',[]));missing=set(scope.get('requested_but_not_returned',[]))
    if returned|missing!=expected or returned&missing:raise IncompleteRead('profile_field_coverage_missing')
    if any(set(row)-expected-{'source','record_type','observed_at'} for row in record.get('records',[])):raise IncompleteRead('unrequested_profile_fields')
    # Missing is data availability, never converted into empty/null values.
    return copy.deepcopy(record)


def project_latest(payload):
    """Vita returns latest facts beside historical raw series; project the facts.

    Use the executor's computed latest fact, check its series/source identity,
    and drop whole-window statistics. Never pick the largest numeric value.
    """
    facts=payload.get('facts',[]);series=payload.get('series',[]);projected=[]
    keys=[(f['concept'],f['source']) for f in facts]
    if len(keys)!=len(set(keys)) or set(keys)!={(s['concept'],s['source']) for s in series}:
        raise IncompleteRead('latest_fact_coverage_mismatch')
    for s in series:
        if s.get('grouping')!='none' or s.get('sample_counts'):
            raise IncompleteRead('latest_metric_aggregation_mismatch')
        fact=next(f for f in facts if (f['concept'],f['source'])==(s['concept'],s['source']))
        moment=datetime.fromisoformat(fact['observed_at'].replace('Z','+00:00'))
        if moment.tzinfo is None or fact.get('unit')!=s.get('unit'):raise IncompleteRead('latest_fact_mismatch')
        matched=False
        for at,value in s.get('points',[]):
            point_at=datetime.fromisoformat(at.replace('Z','+00:00'))
            if point_at.tzinfo is None or point_at>moment:raise IncompleteRead('latest_fact_mismatch')
            if point_at==moment and value==fact['value']:matched=True
        if not matched:raise IncompleteRead('latest_fact_mismatch')
        projected.append({k:copy.deepcopy(s[k]) for k in ('concept','source','unit')} |
            {'points':[[fact['observed_at'],fact['value']]],'grouping':'none',
             'projection':'latest_only','scope_complete':True,'evidence_ref':fact.get('evidence_ref')})
    return {'facts':copy.deepcopy(facts),'series':projected,'claim_scope':copy.deepcopy(payload['claim_scope']),
            'coverage':{'scope_complete':True,'projection':'latest_only',
                        'requested_range':copy.deepcopy(payload['coverage'].get('requested_range'))}}


async def execute_plan(plan,request,run_operation,*,max_pages=8):
    """run_operation(kind, operation) is the existing authorized Vita executor.

    No retries on errors and no widening on empty windows. Record continuation
    preserves every query field. Research execution remains an explicit callback.
    A partial operation never becomes a completed all-clause result.
    """
    plan=QueryPlan.model_validate(plan)
    validate_inventory(plan,request)
    if plan.status!='planned':return {'status':'handoff','operations':[]}
    if type(max_pages) is not int or not 1<=max_pages<=8:raise IncompleteRead('Invalid page budget')
    batch=compile_batch(plan,request);outputs=[];index=0
    ordered=sorted([(kind,op) for kind in ('health_reads','literature_reads') for op in batch[kind]],key=lambda x:x[1]['operation_id'])
    for query_index,query in enumerate(plan.queries):
        count=max(1,(len(query.metrics)+7)//8) if isinstance(query,HealthRead) else 1
        for _ in range(count):
            kind,operation=ordered[index];index+=1
            try:
                payload=await run_operation(kind,copy.deepcopy(operation))
                if not isinstance(payload,dict) or payload.get('status') not in ('ok','empty'):
                    raise IncompleteRead('source_operation_failed')
                if isinstance(query,ResearchRead):
                    outputs.append({'query_index':query_index,'status':'acquired','reason':'research_evidence_review_required','result':payload});continue
                if payload.get('gaps') or payload.get('denied') or payload.get('scan_continuation_ref'):
                    raise IncompleteRead('source_scope_incomplete')
                if query.date_basis=='sleep_end_day':
                    value=project_sleep(query.model_copy(update={'metrics':operation['concepts']}),payload,plan.time_zone)
                elif query.records==['profile']:
                    value=profile_projection(query,payload)
                elif query.records:
                    record=query.records[0];page=(payload.get('records') or {}).get(record,{})
                    details=[];seen=set();page_count=0;offset=0
                    while True:
                        page_count+=1
                        if page.get('status')!='ok' or page.get('scope_complete') is not True:
                            raise IncompleteRead('record_scope_incomplete')
                        if page.get('detail_page_offset',offset)!=offset:raise IncompleteRead('record_page_offset_mismatch')
                        items=page.get('record_details')
                        if not isinstance(items,list):raise IncompleteRead('record_details_missing')
                        if len(items)>operation['record_detail_limit']:raise IncompleteRead('record_page_limit_mismatch')
                        details.extend(items);offset+=len(items)
                        token=page.get('record_continuation_token')
                        if query.limit and len(details)>=query.limit:
                            value={'record_details':details[:query.limit],'requested_limit':query.limit,'scope_complete':True,'selection_complete':True};break
                        if not token:
                            # Vita marks detail_complete for a whole single page,
                            # not for the final page of a continued result. A
                            # contiguous sequence ending with no-more-detail is
                            # also complete; never confuse this with scan paging.
                            exhausted=page.get('detail_complete') is True or (
                                'detail_page_offset' in page and page.get('more_detail_available') is False)
                            if not exhausted:raise IncompleteRead('record_details_incomplete')
                            value={'record_details':details,'requested_limit':query.limit,'scope_complete':True,'selection_complete':True};break
                        if token in seen or not items:raise IncompleteRead('record_continuation_stalled')
                        if page_count>=max_pages:raise IncompleteRead('record_page_budget_exhausted')
                        seen.add(token)
                        # Keep the logical operation identity and frozen range.
                        following={**operation,'record_continuation_token':token}
                        payload=await run_operation(kind,copy.deepcopy(following))
                        if not isinstance(payload,dict) or payload.get('status')!='ok':raise IncompleteRead('record_continuation_failed')
                        if payload.get('gaps') or payload.get('denied') or payload.get('scan_continuation_ref'):
                            raise IncompleteRead('source_scope_incomplete')
                        page=(payload.get('records') or {}).get(record,{})
                else:
                    if (payload.get('coverage') or {}).get('scope_complete') is not True or payload.get('gaps') or payload.get('denied'):
                        raise IncompleteRead('metric_scope_incomplete')
                    expected=set(operation['concepts']);represented=set()
                    if (payload.get('claim_scope') or {}).get('purpose')!=query.operation:
                        raise IncompleteRead('metric_purpose_mismatch')
                    for series in payload.get('series',[]):
                        if series.get('concept') not in expected:raise IncompleteRead('unexpected_metric')
                        if query.source and series.get('source')!=query.source:raise IncompleteRead('unexpected_source')
                        if series.get('scope_complete') is not True:raise IncompleteRead('metric_scope_incomplete')
                        represented.add(series['concept'])
                    # Missing series are never evidence that a requested metric
                    # is absent. Let the caller handle incomplete evidence.
                    if represented!=expected:raise IncompleteRead('metric_coverage_missing')
                    value=project_latest(payload) if query.operation=='latest' else copy.deepcopy(payload)
                outputs.append({'query_index':query_index,'status':'complete','result':value})
            except IncompleteRead as exc:
                outputs.append({'query_index':query_index,'status':'incomplete','reason':str(exc)})
            except (ValueError,KeyError,TypeError,AttributeError):
                # Third-party exceptions can contain record values.
                outputs.append({'query_index':query_index,'status':'incomplete','reason':'invalid_source_payload'})
            except Exception:
                # Operational callback failures must preserve earlier evidence
                # and reach the answering model as incomplete acquisition.
                # Cancellation/SystemExit are BaseExceptions and still propagate.
                outputs.append({'query_index':query_index,'status':'incomplete','reason':'source_operation_failed'})
    complete=all(o['status']=='complete' for o in outputs)
    status='complete' if complete else 'incomplete' if any(o['status']=='incomplete' for o in outputs) else 'needs_evidence_review'
    return {'status':status,'operations':outputs,
            'all_requested_delivered':all(o['status']=='complete' for o in outputs)}
