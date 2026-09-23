"""Pure typed plan construction and coverage accounting. Never reads a database.

Selection is advisory. Authorization, source queries, arithmetic, deduplication,
observed timestamps and paging remain in the existing Vita tools.
"""
import copy
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from compat.jev_dates import resolve_range

ROOT=Path(__file__).resolve().parent
CATALOG=json.loads((ROOT/'metadata/health_metrics.v1.json').read_text())
POLICY=json.loads((ROOT/'metadata/overview-policy.v1.json').read_text())
CATALOG_SHA256=hashlib.sha256((ROOT/'metadata/health_metrics.v1.json').read_bytes()).hexdigest()
if CATALOG_SHA256!=POLICY['canonical_catalog_sha256'] or set(CATALOG['metrics'])!=set(POLICY['domains']):
    raise RuntimeError('Catalog/policy identity mismatch')
RECORDS={'profile':'profile.health.v1','workouts':'workouts.sessions.v1','labs':'labs.upload_metadata.v1','calendar':'calendar.events.v1'}
PROFILE_FIELDS=['goals','chronic_conditions','medications','no_regular_medications','allergies','sex','birth_year','height_cm','weight_kg']
RESEARCH={
 'broad_overview':'For adults, summarize systematic review and randomized trial evidence on sleep duration and regularity, physical activity, and diet for cardiometabolic health. Identify practical interventions, studied populations, and applicability limits; distinguish associations from intervention effects.',
 'sleep_activity':'For adults, what systematic review and randomized trial evidence supports improving sleep duration and regularity and physical activity? Describe practical interventions, outcomes, and applicability limits.',
 'cardiometabolic':'For adults, what systematic review and randomized trial evidence supports physical activity and dietary changes for lipid and glucose control? Describe practical interventions, outcomes, and applicability limits.'}


def canonical_metrics(available):
    aliases={}
    for name,definition in CATALOG['metrics'].items():
        for alias in [name,*definition['aliases']]:
            normalized=alias.casefold()
            if normalized in aliases and aliases[normalized]!=name:
                raise ValueError('Ambiguous canonical catalog alias')
            aliases[normalized]=name
    mapped={}
    for metric in available:
        if not isinstance(metric,str) or metric.casefold() not in aliases:
            raise ValueError('Unsupported metric catalog version or identifier')
        mapped[metric]=aliases[metric.casefold()]
    return mapped


def build_plan(selection, available_metrics, *, now, time_zone, record_types=(),
               literature_available=False, operation_budget=8, summary_token_budget=16000):
    """Construct at most one parallel batch. No implicit second request/retry.

    Broad undated overview policy: <=8 concepts/domain; latest-known labs/body,
    30-day trends elsewhere. Every omitted concept/record is declared deferred.
    Explicit and narrow requests never lose selected concepts to overview caps.
    """
    if operation_budget<1 or operation_budget>8 or summary_token_budget<2000:
        raise ValueError('Unsupported execution budget')
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Trusted aware reference clock required')
    mapped=canonical_metrics(available_metrics)
    if selection.get('status')!='selected' or selection.get('advisory') is not True:
        return {'schema_version':'vita-read-plan/v1','status':'unsupported','reason':'selector_handoff_required','batch':None}
    answers=selection['answers']
    choice=lambda key:answers[key]['choice']
    task,coverage,purpose=choice('task'),choice('coverage'),choice('purpose')
    if task not in ('health','research') or purpose not in ('latest','trend','unsupported'):
        raise ValueError('Invalid selector plan semantics')
    if task=='health' and purpose=='unsupported':
        return {'schema_version':'vita-read-plan/v1','status':'unsupported','reason':'read_form_requires_existing_model','batch':None}
    requested=resolve_range(answers,now=now,time_zone=time_zone) if task=='health' else None
    broad=coverage=='broad';undated_broad=broad and requested is None
    health=[];literature=[];manifest=[];deferred=[]
    selected=set(mapped.values()) if broad else {mapped[m] for m in available_metrics if answers['metric__'+m]['choice']=='include'}
    if task!='health':selected=set()
    requested_records=[name for name in RECORDS if (broad or answers[name]['noul']>.5)] if task=='health' else []
    for name in requested_records:
        if name not in record_types:deferred.append({'record_type':name,'reason':'not_authorized_or_available'})
    research=choice('research') if literature_available else 'none'
    if research!='none' and research not in RESEARCH:raise ValueError('Unsupported research topic')
    # Reserve literature before metric partitioning; no request is issued here.
    health_budget=operation_budget-int(research!='none')
    grouped={}
    for metric in selected:grouped.setdefault(POLICY['domains'][metric],[]).append(metric)
    for domain in ('labs','body','sleep','activity','recovery','vitals'):
        values=grouped.get(domain,[])
        if not values:continue
        priority=POLICY['priority'][domain]
        values=sorted(values,key=lambda m:(priority.index(m) if m in priority else len(priority),m))
        if broad:
            cap=POLICY['overview_metrics_per_domain'][domain]
            for metric in values[cap:]:deferred.append({'concept':metric,'domain':domain,'reason':'overview_detail_budget'})
            values=values[:cap]
        for offset in range(0,len(values),8):
            concepts=values[offset:offset+8]
            read_purpose='latest' if undated_broad and domain in ('labs','body') else purpose
            selected_range=requested or ({'kind':'all_history'} if read_purpose=='latest' else {'kind':'relative','amount':30,'unit':'days'})
            operation={'operation_id':len(health)+1,'purpose':read_purpose,'concepts':concepts,
                       'range':selected_range,'time_zone':time_zone,'result_view':'summary',
                       'grouping':'none' if read_purpose=='latest' else 'week'}
            health.append(operation)
            manifest.append({'operation_id':operation['operation_id'],'domain':domain,'concepts':concepts,
                             'purpose':read_purpose,'range':copy.deepcopy(selected_range),
                             'scope_origin':'explicit_request' if requested else 'undated_overview_policy' if broad else 'default_recent_trend' if read_purpose=='trend' else 'latest_known',
                             'required_summary':'latest_individual_observation_per_returned_source' if read_purpose=='latest' else 'whole_window_statistics_and_coverage_per_returned_source',
                             'stage':'selected','delivered':False})
    if len(health)>health_budget:
        return {'schema_version':'vita-read-plan/v1','status':'unsupported','reason':'requested_scope_exceeds_operation_budget','batch':None}
    for name in requested_records:
        if name not in record_types:continue
        if len(health)>=health_budget:
            if not broad:
                return {'schema_version':'vita-read-plan/v1','status':'unsupported','reason':'requested_scope_exceeds_operation_budget','batch':None}
            deferred.append({'record_type':name,'reason':'operation_budget'});continue
        operation={'operation_id':len(health)+1,'purpose':'latest','record_types':[name],
                   'range':{'kind':'all_history'},'time_zone':time_zone,'result_view':'summary'}
        if name=='profile':operation['profile_fields']=PROFILE_FIELDS
        elif name=='workouts':operation['range']=requested or {'kind':'relative','unit':'days','amount':30}
        else:
            operation.update(record_detail_dataset=RECORDS[name],record_detail_limit=3 if name=='labs' else 8)
            if name=='labs':operation['range']=requested or {'kind':'all_history'}
        if name=='calendar' and choice('calendar_basis')!='current_plans':
            operation['calendar_date_basis']=choice('calendar_basis')
            operation['range']=requested or {'kind':'all_history'}
        elif name=='calendar' and requested and not broad and requested['kind']!='all_history':
            return {'schema_version':'vita-read-plan/v1','status':'unsupported','reason':'calendar_basis_required','batch':None}
        health.append(operation)
        manifest.append({'operation_id':operation['operation_id'],'domain':'record:'+name,'record_types':[name],
                         'purpose':'latest','range':copy.deepcopy(operation['range']),
                         'scope_origin':'current_profile_context' if name=='profile' else 'record_selection',
                         'required_summary':'bounded_record_summary_with_source_scope','stage':'selected','delivered':False})
    if research!='none':
        literature=[{'operation_id':len(health)+1,'question':RESEARCH[research],'task':'evidence_summary',
                     'depth':'fast','subject_basis':'general_overview_research','basis_source_ids':[]}]
    batch={'health_reads':health,'literature_reads':literature,
           'required_operation_ids':list(range(1,len(health)+len(literature)+1))}
    if not batch['required_operation_ids']:
        return {'schema_version':'vita-read-plan/v1','status':'unsupported','reason':'no_authorized_work','batch':None}
    # Reserve one bounded summary per domain/record. This is a request to Vita's
    # serializer, not proof that its existing admission algorithm honours it.
    each=(summary_token_budget-2000*len(literature))//max(1,len(manifest))
    if each<2000:
        return {'schema_version':'vita-read-plan/v1','status':'unsupported','reason':'insufficient_summary_budget','batch':None}
    for item in manifest:item['summary_token_allowance']=each
    return {'schema_version':'vita-read-plan/v1','status':'planned','batch':batch,
            'catalog_version':CATALOG['version'],'catalog_sha256':CATALOG_SHA256,
            'coverage':{'required':manifest,'deferred':deferred,'all_requested_delivered':False,
                        'summary_token_budget':summary_token_budget,'literature_token_reserve':2000*len(literature)},
            'advisory':True,'time_zone':time_zone,'reference_date':now.astimezone(ZoneInfo(time_zone)).date().isoformat(),
            'recovery':'Use existing authorized tools for explicit remaining gaps; never widen scope automatically.'}


def audit_delivery(plan, delivered_operations):
    """Audit only final MODEL-VISIBLE payloads, not retained full source results.

    Input operation wrapper: operation_id, status, source_id, result (final
    serialized Vita summary page). Missing concepts remain missing, not empty.
    Per-source evidence stays separate; no recomputation or deduplication here.
    """
    by_id={item['operation_id']:item for item in delivered_operations}
    result=[]
    for required in plan['coverage']['required']:
        item=by_id.get(required['operation_id'])
        row={**required,'stage':'not_admitted','delivered':False,'concept_states':{}}
        if item is None:result.append(row);continue
        status=item.get('status')
        if status not in ('ok','empty'):
            row.update(stage='admitted_error',outcome=status,source_id=item.get('source_id'))
            result.append(row);continue
        payload=item.get('result') or {};coverage=payload.get('coverage') or {}
        evidence=payload.get('evidence') or []
        requested=required['range']
        expected_filter=({'range_mode':'all'} if requested['kind']=='all_history' else
            {'range_mode':'current_'+requested['period']} if requested['kind']=='calendar' else
            {'range_mode':'auto','history_'+requested['unit']:requested['amount']} if requested['kind']=='relative' else
            {'range_mode':'between','start_at':requested['start_at'],'end_at':requested['end_at']})
        claim=payload.get('claim_scope') or {}
        reported=claim.get('requested_range') or {}
        scope_matches=claim.get('purpose')==required['purpose'] and all(reported.get(k)==v for k,v in expected_filter.items())
        page=payload.get('result_page') or {}
        row.update(stage='admitted',source_id=item.get('source_id'),page_complete=page.get('complete') is True,
                   scan_incomplete=bool(payload.get('scan_continuation_ref')) or coverage.get('scope_complete') is False)
        for concept in required.get('concepts',[]):
            matches=[e for e in evidence if e.get('kind')=='series' and e.get('concept')==concept]
            if matches:
                state='present'
                if not matches or any(not e.get('unit') or not e.get('source') for e in matches):
                    state='missing_unit_or_source'
                elif required['purpose']=='latest' and any(e.get('grouping','none')!='none' or any(o.get('sample_count',1)!=1 for o in e.get('observations',[])) for e in matches):
                    state='wrong_aggregation'
                elif row['scan_incomplete'] or any(e.get('scope_complete') is not True for e in matches):state='scan_incomplete'
                stale=[]
                if required['purpose']=='latest':
                    from datetime import date,datetime
                    for e in matches:
                        for observed in e.get('observations',[]):
                            try:
                                instant=datetime.fromisoformat(observed['date'].replace('Z','+00:00'))
                                observed_day=instant.astimezone(ZoneInfo(plan['time_zone'])).date() if instant.tzinfo else instant.date()
                                age=(date.fromisoformat(plan['reference_date'])-observed_day).days
                                if age>CATALOG['metrics'][concept]['fresh_days']:stale.append({'source':e.get('source'),'age_days':age})
                            except (ValueError,KeyError,TypeError):state='invalid_observation_date'
                row['concept_states'][concept]={'state':state,'sources':[e.get('source') for e in matches],'stale_last_known':stale}
            elif status=='empty' and coverage.get('scope_complete') is True:
                row['concept_states'][concept]={'state':'authorized_empty'}
            else:row['concept_states'][concept]={'state':'not_delivered'}
        if required.get('concepts'):
            row['delivered']=all(v['state'] in ('present','authorized_empty') for v in row['concept_states'].values())
        else:
            row['delivered']=bool(evidence) or (status=='empty' and coverage.get('scope_complete') is True)
        if not scope_matches:
            row['delivered']=False
            row['outcome']='query_scope_unverified'
        else:
            row['outcome']='delivered' if row['delivered'] else 'partial_or_unavailable'
        result.append(row)
    return {'schema_version':'vita-delivery-audit/v1','operations':result,
            'planned_summaries_delivered':all(row['delivered'] for row in result),
            'all_requested_delivered':not plan['coverage']['deferred'] and all(row['delivered'] for row in result)}


def project_required_summaries(plan, retained_operations, *, summarize_series, token_count):
    """Proposed Vita-side admission adapter; not installed into Vita by this repo.

    summarize_series must be Vita's existing summary_evidence function. Inputs
    must be settled authorized operation results. Existing source arithmetic and
    provenance are preserved; values are neither combined nor recomputed.
    The return value is the exact candidate model-visible bundle and its audit.
    """
    by_id={item['operation_id']:item for item in retained_operations}
    required=plan['coverage']['required'];operations=[];queues=[]
    reserve=plan['coverage']['literature_token_reserve']
    budget=plan['coverage']['summary_token_budget']-reserve
    keep={'kind','concept','source','unit','grouping','scope_complete','statistics',
          'observations','omitted_observations','detail_available','measurement_basis',
          'latest_report_label','labelled_test_observation_count','coverage','incomplete_reason',
          'expected_cadence','period','date_basis'}
    for item in required:
        incoming=by_id.get(item['operation_id'])
        if incoming is None:
            operations.append({'operation_id':item['operation_id'],'status':'unavailable','result':{'reason':'not_queried_or_settled'}})
            queues.append([]);continue
        payload=incoming.get('result') or {}
        result={key:copy.deepcopy(payload[key]) for key in ('claim_scope','coverage','scan_continuation_ref','gaps','denied') if key in payload}
        result.update(schema='vita-manager-health-summary/v1',evidence=[],result_page={'complete':False,'next_offset':None})
        operation={'operation_id':item['operation_id'],'source_id':incoming.get('source_id'),'status':incoming['status'],'result':result}
        operations.append(operation)
        if item.get('concepts'):
            summaries=[]
            for entry in summarize_series(payload):
                evidence=entry['evidence']
                if evidence.get('concept') not in item['concepts']:continue
                compact={key:copy.deepcopy(value) for key,value in evidence.items() if key in keep}
                if item['purpose']=='latest':
                    compact.pop('statistics',None)
                compact['source_finding_id']=entry['finding_id']
                summaries.append(compact)
            queues.append(summaries)
        else:
            # Records require Vita's already authorized/profile-field projection;
            # do not expose an arbitrary retained record dictionary.
            queues.append(copy.deepcopy(incoming.get('admitted_record_evidence',[])))
    # Reserve every operation's scope/error envelope before any evidence.
    if token_count({'operations':operations})>budget:
        return {'status':'budget_refused','operations':[],'audit':audit_delivery(plan,[])}
    positions=[0]*len(queues);blocked=set()
    while True:
        added=False
        for index,queue in enumerate(queues):
            if index in blocked or positions[index]>=len(queue):continue
            evidence=operations[index]['result']['evidence'];evidence.append(queue[positions[index]])
            if token_count({'operations':operations})>budget-512:
                evidence.pop();blocked.add(index);continue
            positions[index]+=1;added=True
        if not added:break
    for index,queue in enumerate(queues):
        operation=operations[index]
        operation['result']['result_page']={'complete':positions[index]==len(queue),
            'next_offset':positions[index] if positions[index]<len(queue) else None,
            'total_items':len(queue),'query_reexecuted':False,'projection':'required_summary_v1'}
    # Metadata itself counts. Refuse rather than exceed the declared input budget.
    if token_count({'operations':operations})>budget:
        return {'status':'budget_refused','operations':[],'audit':audit_delivery(plan,[])}
    return {'status':'projected','operations':operations,'tokens':token_count({'operations':operations}),
            'audit':audit_delivery(plan,operations)}
