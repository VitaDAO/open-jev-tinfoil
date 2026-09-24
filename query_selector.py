"""Experimental structured decoder; evaluate before changing the serving path."""
import hashlib
import json
import math
import re
from datetime import datetime
from contextvars import ContextVar
from pathlib import Path
from zoneinfo import ZoneInfo

from compat.jev_dates import resolve_range
from proposal_binding import canonicalize, entities, erase, temporal, resolve_context
from query_plan import QueryRequest, QueryPlan, HealthRead, ResearchRead, validate_inventory, request_identity, query_operation_count
from selector import SelectorRequest, AREAS
from schema_index import INDEX
from trained_proposal_selector import TrainedProposalSelector
from routing import MODEL_REVISION

ROOT=Path(__file__).resolve().parent
INTENT_SHA256='79057d0e2673aa2813a8ea64193d74182a33392b1221c7c06d4afc7afe18dc36'
SPELLINGS={'stpes':'steps','slep':'sleep','wk':'week','hscrp':'hs crp'}
SOURCES={'oura':'oura','garmin':'garmin','whoop':'whoop','fitbit':'fitbit',
         'withings':'withings','apple health':'apple_health','polar':'polar'}
FIELDS={field.replace('_',' '):field for field in INDEX['records']['profile']['backend_fields']}
FIELDS.update({'conditions':'chronic_conditions','meds':'medications','bio':'bio'})
_FEATURES=ContextVar('query_request_features',default=None)


def read_restricted(text):
    # Restrictions are not permissions a classifier may override. Ambiguous
    # scope or later revocation is left intact for the native conversation.
    text=canonicalize(text)
    return bool(re.search(r"\b(?:do not|don't|never|must not|stop|avoid)\s+(?:show|read|access|fetch|retrieve|open|use|look (?:at|up))\b",text)
        or re.search(r'\b(?:keep|leave) (?:my |the )?(?:health |personal )?(?:records|data|reports|profile) (?:closed|unopened|private|off[- ]limits)\b',text))


def lab_metadata_request(text):
    """Complete personal date/issuer questions over the catalog's lab records.

    Unlike a metric mention, these constructions explicitly request recorded
    facts. Every clause must bind; definitions, filters and other people do not.
    """
    aliases='(?:'+'|'.join(re.escape(a) for a in INDEX['records']['labs']['aliases'])+')'
    subject=r'my '+aliases
    date=r'(?:when were '+subject+r' (?:done|issued|performed)|what (?:are|were) (?:the )?(?:exam |examination )?dates (?:on|of|for) '+subject+r')'
    issuer=lambda target:r'(?:which (?:lab|laboratory|provider) (?:issued|produced) '+target+r'|who (?:issued|produced) '+target+r')'
    text=canonicalize(text).strip(' ?!.')
    field=r'(?:(?:test |exam |examination )?dates|(?:issuing )?(?:labs|laboratories|providers))'
    listing=r'(?:list|show(?: me)?) (?:the )?'+field+r'(?: and '+field+r')? (?:on|of|for) '+subject
    if re.fullmatch(date+'|'+issuer(subject)+'|'+listing,text):return True
    # An anaphor is permitted only after its own explicit personal subject.
    return bool(re.fullmatch(date+r',? and '+issuer(r'them'),text))


def identity():
    from trained_proposal_selector import identity as base_identity
    return hashlib.sha256(base_identity().encode()+b''.join((ROOT/p).read_bytes() for p in
        ('query_plan.py','query_selector.py','query_execution.py'))+INTENT_SHA256.encode()).hexdigest()


def enhance(text):
    text=canonicalize(text)
    for word,replacement in SPELLINGS.items():text=re.sub(r'\b'+word+r'\b',replacement,text)
    text=re.sub(r'^different question\s*[-:,]\s*','',text)
    # A discourse marker before an explicit new read is not a list item.
    # Leave projection/period corrections ("actually, just...", "make that...")
    # intact for the history resolver.
    text=re.sub(r'^actually\s*,\s*(?=(?:please\s+)?(?:show|fetch|retrieve|view|inspect)\b)','',text)
    text=re.sub(r'\blipid (?:picture|profile)\b','lipids',text)
    text=re.sub(r'\bhow much (?:have i been|am i) sleeping\b','show my total sleep',text)
    text=re.sub(r'\b(?:all (?:of )?(?:my )?)?step history since i started tracking\b','steps all history',text)
    # Resolve a singular measurement anaphor while leaving numeric limits and
    # every source/date/action qualifier in place for the later checks.
    if re.search(r'\b(?:newest|latest|most recent)\b',text):
        text=re.sub(r'\bone(?= (?:i have|can you)|[,?!.]|$)','value',text)
        text=re.sub(r'\bnewest\b','latest',text)
    return text


def legacy_request(request,text,history=()):
    return SelectorRequest.model_validate({'schema_version':'vita-selector/v1',
        'available_metrics':request.available_metrics,'available_record_types':request.available_record_types,
        'literature_available':request.literature_available,'state':{**request.state.model_dump(),
        'current_request':text,'recent_user_requests':list(history)}})


class QuerySelector(TrainedProposalSelector):
    def __init__(self,model):
        super().__init__(model)
        raw=(ROOT/'adapters/vita-read-intent-v4.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=INTENT_SHA256:raise ValueError('Unapproved query intent weights')
        data=json.loads(raw)
        if data['format_version']!=1 or type(data['threshold']) not in (int,float) or not math.isfinite(data['threshold']):
            raise ValueError('Invalid intent artifact')
        if data['model_revision']!=MODEL_REVISION or data['question']!=self.question.instructions:raise ValueError('Encoder mismatch')
        self.intent_weights=self.torch.tensor(data['weights'],dtype=self.torch.float64)
        self.intent_threshold=data['threshold']
        if self.intent_weights.shape!=(1024,) or not self.torch.isfinite(self.intent_weights).all():raise ValueError('Invalid weights')

    def encode(self,current):
        # Only identical encoder inputs within this call share frozen features.
        # Context isolation prevents concurrent callers sharing private text;
        # select_query clears the dictionary even when inference raises.
        scope=_FEATURES.get()
        if scope is None or scope[0] is not self:return super().encode(current)
        features=scope[1]
        if current not in features:features[current]=super().encode(current)
        return features[current]

    def select_query(self,request):
        request=QueryRequest.model_validate(request)
        features={};token=_FEATURES.set((self,features))
        try:return self._select_query(request)
        finally:
            features.clear()
            _FEATURES.reset(token)

    def _select_query(self,request):
        queries=[];diagnostics={};reasons=[]
        try:
            if any(read_restricted(s) for s in [request.state.current_request,*request.state.recent_user_requests]):
                raise ValueError('read_restriction_requires_native_context')
            if re.search(r'\b(?:translate|translation|rephrase|rewrite|paraphrase)\b',canonicalize(request.state.current_request)):
                raise ValueError('text_transformation_requires_native_context')
            text=enhance(request.state.current_request)
            req=legacy_request(request,text,[enhance(s) for s in request.state.recent_user_requests])
            text=resolve_context(req)
            text=re.sub(r'^skip (?:the )?trend(?: line)?\s*[;,]\s*','',text)
            # Complete clauses have their own subjects and windows. Never split
            # dates such as "between June 3 and June 17" or a plain metric list.
            pieces=re.split(r'\s*(?:;|,?\s+(?:and|plus|then)\s+(?=(?:find|what|tell me|show|look at)\b))\s*',text)
            if len(pieces)>1 and ';' not in text and all(
                    not entities(p,request.available_metrics) and not re.search(r'\b(?:research|studies|trials|evidence)\b',p)
                    for p in pieces[1:]):
                # An unbound tail may qualify the preceding overview. Keep the
                # whole utterance for the existing intent/coverage checks;
                # never silently delete the tail or invent another read.
                pieces=[text]
            if len(pieces)==1:
                candidate=re.split(r'\s+and\s+',text)
                if len(candidate)>1 and all(entities(s,request.available_metrics) and temporal(s,request.state.reference_date)[1] for s in candidate):
                    pieces=candidate
            for piece in pieces:
                overview=bool(re.search(r'\bhealth\b',piece) and re.search(r'\b(?:analysis|analyse|analyze|assessment|summary|rundown)\b',piece))
                if re.search(r'\b(?:research|studies|trials|evidence)\b',piece) and not overview and not re.search(r'\bno (?:studies|research)\b',piece):
                    queries.append(self.research(piece,queries,request));continue
                queries.extend(self.health(piece,request,diagnostics))
            # Identical reads have the same subject, source, operation, window
            # and projection. Repeating a clause must not repeat acquisition.
            unique=[]
            for query in queries:
                if query not in unique:unique.append(query)
            queries=unique
            result=QueryPlan(status='planned',queries=queries,time_zone=request.state.time_zone,
                selector_sha256=identity(),adapter_sha256=INTENT_SHA256,model_revision=MODEL_REVISION,
                request_sha256=request_identity(request),diagnostics=diagnostics)
            validate_inventory(result,request)
            # Budget counts actual metric chunks, not just input clauses.
            if sum(query_operation_count(q) for q in queries)>8:
                raise ValueError('operation_budget_exceeded')
            return result.model_dump(mode='json')
        except (ValueError,KeyError) as exc:
            # Closed, non-sensitive codes only. Never echo failed raw user input.
            code=str(exc)
            if not re.fullmatch(r'[a-z][a-z0-9_]{0,90}',code):code='query_contract_unrepresentable'
            reasons=[code]
        return QueryPlan(status='handoff',reason_codes=reasons,time_zone=request.state.time_zone,
            selector_sha256=identity(),adapter_sha256=INTENT_SHA256,model_revision=MODEL_REVISION,
            request_sha256=request_identity(request),diagnostics=diagnostics).model_dump(mode='json')

    def health(self,text,request,diagnostics):
        original=text;source=None;limit=None;fields=[];night=False
        if lab_metadata_request(text):
            if 'labs' not in request.available_record_types:
                raise ValueError('requested_record_category_unavailable')
            diagnostics.setdefault('clause_decisions',[]).append({
                'status':'selected','reason_codes':[],'method':'complete_lab_metadata_binding'})
            return [HealthRead(records=['labs'],operation='latest',
                               period={'kind':'all_history'},date_basis='exam_date')]
        # The date binder represents whole days and a trusted request timezone.
        # Never let a semantic confidence score erase an explicit clock/window
        # or a requested timezone override that this contract cannot represent.
        if re.search(r'\b(?:\d{1,2}:\d{2}|\d{1,2}\s*[ap]\.?m\.?(?!\w)|time\s*zone|utc|gmt|noon|midnight|morning|afternoon|evening|hourly|hours?)\b|\b[a-z_]+/[a-z_]+\b|\b(?:in|on|using)\b[^,;]*\btime\b',text):
            raise ValueError('clock_or_timezone_qualifier_unavailable')
        # Exact provider names are arguments; unknown provider text survives and
        # is rejected by the inherited constraint/semantic coverage checks.
        for alias,value in SOURCES.items():
            pattern=r'\b(?:from|using|recorded by) (?:my |the )?'+re.escape(alias)+r'\b'
            spans=[(m.start(),m.end()) for m in re.finditer(pattern,text)]
            # A provider may directly qualify a recognized subject, as in
            # "Garmin steps". Bind only an adjacent entity; do not erase a
            # provider mention elsewhere in a comparison or an instruction.
            subjects=entities(text,request.available_metrics)
            for match in re.finditer(r'\b'+re.escape(alias)+r'\b',text):
                if re.search(r'\b(?:non|not|no|without|except|excluding|exclude|other than|all but)(?:[\s-]+(?:from|using|recorded|by|for|my|the|just|only|solely|exclusively))*[\s-]+$',text[:match.start()]):
                    raise ValueError('negated_source_filter_unavailable')
                if any(start>match.end() and not text[match.end():start].strip()
                       for start,_,_ in subjects):
                    if len(subjects)!=1:raise ValueError('source_subject_scope_ambiguous')
                    spans.append((match.start(),match.end()))
            if spans:
                if source and source!=value:raise ValueError('multiple_sources_require_separate_clauses')
                source=value;text=erase(text,spans)
        count=re.search(r'\b(?:latest|newest|most recent|last)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?=(?:lab |blood |health )?(?:reports|workouts|appointments|plans)\b)',text)
        if count:
            from proposal_binding import NUMBER_WORDS
            limit=int(count[1]) if count[1].isdigit() else {'zero':0,**NUMBER_WORDS}[count[1]]
            if not 1<=limit<=200:raise ValueError('record_limit_out_of_range')
            text=text[:count.start()]+'latest '+text[count.end():]
        if re.search(r'\b(?:uploaded|upload time|upload date|imported)\b',text):raise ValueError('upload_order_not_available')
        entity_spans=entities(text,request.available_metrics)
        record_kinds={v for _,_,values in entity_spans for kind,v in values if kind=='record'}
        if record_kinds-{'profile'}:
            # Counts must be bound explicitly above. Never let the model erase
            # an alternate quantity construction and silently use the page cap.
            # Calendar quantities belong to their date span, not the row limit.
            from proposal_binding import NUMBER_WORDS
            _,date_spans,_=temporal(text,request.state.reference_date)
            remaining=erase(text,date_spans)
            if re.search(r'\b(?:\d+|zero|'+'|'.join(NUMBER_WORDS)+r')\b',remaining):
                raise ValueError('unbound_record_quantity')
        # Field-level profile requests are bound against the actual public schema.
        field_matches=[]
        for alias,field in sorted(FIELDS.items(),key=lambda item:-len(item[0])):
            for m in re.finditer(r'\b'+re.escape(alias)+r'\b',text):
                if not any(m.start()<b and a<m.end() for a,b,_ in field_matches):field_matches.append((m.start(),m.end(),field))
        if 'profile' in record_kinds or field_matches:
            if source or limit:raise ValueError('profile_source_or_limit_unavailable')
            if record_kinds-{'profile'}:raise ValueError('mixed_profile_scope_requires_clause')
            prefix=re.match(r'^(?:please )?(?:(?:can|could|would|may) you )?(?:please )?(?:show(?: me)?|fetch|retrieve|view|inspect|(?:pull|bring) up|(?:take|have) (?:a )?look at|what (?:are|is) my)\b',text)
            _,date_spans,date_error=temporal(text[prefix.end():] if prefix else text,request.state.reference_date)
            if date_spans or date_error:raise ValueError('profile_history_unavailable')
            complete=bool(re.search(r'\b(?:full|whole|complete) (?:health )?profile\b',text))
            fields=sorted({f for _,_,f in field_matches})
            restricted=bool(fields and (re.search(r'\b(?:only|just)\b',text) or
                re.search(r'\bfrom (?:my |the )?(?:full|whole|complete) (?:health )?profile\b',text)))
            additive=bool(re.search(r'\bprofile (?:and|plus)\b|\b(?:and|plus) (?:my |the )?(?:health )?profile\b',text))
            if additive and restricted:raise ValueError('ambiguous_profile_scope')
            if additive or (complete and not restricted) or not fields:fields=['all']
            rest=erase(text,field_matches)
            medication_read=bool(re.fullmatch(r'what\s+(?:am i taking|do i take)\s*[?!.]*',rest.strip()) and fields==['medications'])
            if medication_read:rest=''
            elif prefix:rest=rest[prefix.end():]
            else:raise ValueError('profile_read_intent_unconfirmed')
            # Parse read constructions before removing argument filler. Modal
            # treatment questions must not become reads by erasing "can I take".
            rest=re.sub(r'\b(?:please|my|me|only|just|and|from|the|profile|health|full|whole|complete|current|of|demographics)\b',' ',rest)
            if re.search(r'[a-z0-9]',rest):raise ValueError('unbound_profile_qualifier')
            return [HealthRead(records=['profile'],operation='latest',profile_fields=fields,
                               date_basis='current_snapshot',period={'kind':'all_history'})]
        # Exclusions operate on a named metric group, not arbitrary records or values.
        if re.search(r'\bsleep\b',text):
            only=re.search(r'\b(duration|efficiency) only\b',text)
            exclude=re.search(r'\b(?:except|excluding|without|leave) (?:sleep )?(duration|efficiency)(?: out(?: of it)?)?\b',text)
            if only or exclude:
                chosen=('total_sleep' if only[1]=='duration' else 'sleep_efficiency') if only else ('sleep_efficiency' if exclude[1]=='duration' else 'total_sleep')
                if only and exclude and only[1]==exclude[1]:raise ValueError('contradictory_metric_projection')
                spans=[(m.start(),m.end()) for m in (only,exclude) if m]
                text=erase(text,spans);text=re.sub(r'\bsleep\b',chosen.replace('_',' '),text)
                text=re.sub(r'[—,]+',' ',text)
        if re.search(r'\blast night\b',text):
            if any('sleep' not in v for _,_,vs in entities(text,request.available_metrics) for kind,v in vs if kind=='metric'):
                raise ValueError('night_requires_sleep')
            night=True;text=re.sub(r'\blast night\b','today',text)
        # A model confidence score cannot certify an unbound list item. Check
        # each additional target independently so a new/unknown biomarker cannot
        # disappear from a supported multi-marker request.
        parts=re.split(r'\s*(?:,|\band\b|\bplus\b)\s*',text)
        if len(parts)>1 and any(k=='metric' for _,_,vs in entities(text,request.available_metrics) for k,_ in vs):
            for tail in parts:
                if not tail or entities(tail,request.available_metrics):continue
                if re.fullmatch(r'(?:please )?no (?:labs|studies|research|recommendations)(?: please)?',tail):continue
                _,spans,error=temporal(tail,request.state.reference_date)
                remainder=erase(tail,spans)
                remainder=re.sub(r'\b(?:and|only|please|thanks|thank|you|values|measurements|results|levels|in|for|the|during|over|past|last|show|me|my|what|are|is|can|could|would|if|possible)\b',' ',remainder)
                if error or re.search(r'[a-z0-9]',remainder):raise ValueError('unbound_list_item')
        req=legacy_request(request,text)
        result=super().select(req)
        diagnostics.setdefault('clause_decisions',[]).append({'status':result['status'],'reason_codes':result['reason_codes']})
        if result['status']!='selected':raise ValueError((result['reason_codes'] or ['unsupported_read'])[0])
        proposal=result['diagnostics']['proposal']
        if proposal['task']!='health':raise ValueError('health_clause_required')
        answers=result['answers'];period=resolve_range(answers,now=request.reference_time,time_zone=request.state.time_zone)
        period=period or {'kind':'all_history'}
        if night:period={'kind':'between','start_at':request.state.reference_date,'end_at':request.state.reference_date}
        output=[];metrics=proposal['metrics']
        if proposal['coverage']=='broad':metrics=list(request.available_metrics)
        if metrics:
            if limit:raise ValueError('metric_latest_n_not_supported')
            # The legacy fixture's default recent trend remains explicit.
            if proposal['period']['period_kind']=='unstated' and proposal['operation']=='trend' and all(m in ('steps','total_sleep','sleep_efficiency') for m in metrics):
                period={'kind':'relative','amount':30,'unit':'days'}
            if proposal['coverage']=='broad' and proposal['period']['period_kind']=='unstated':
                frequent={'steps','total_sleep','sleep_efficiency'}
                for group,window in ((set(metrics)-frequent,{'kind':'all_history'}),
                                     (set(metrics)&frequent,{'kind':'relative','amount':30,'unit':'days'})):
                    if group:output.append(HealthRead(metrics=sorted(group),operation=proposal['operation'],period=window,source=source))
            else:
                output.append(HealthRead(metrics=sorted(metrics),operation=proposal['operation'],period=period,
                    source=source,date_basis='sleep_end_day' if night else 'observed_at'))
        for record in sorted(proposal['records'],key=('profile','workouts','labs','calendar').index):
            basis={'profile':'current_snapshot','labs':'exam_date','workouts':'started_at',
                   'calendar':proposal['calendar_basis'] if proposal['calendar_basis']!='current_plans' else 'next_due_date'}[record]
            record_period=({'kind':'relative','amount':30,'unit':'days'} if record=='workouts' and proposal['period']['period_kind']=='unstated' else period)
            if record in ('calendar','labs') and record_period['kind']=='between':
                # These datasets store dates, not instants. Year-to-date metric
                # bounds end at now; the equivalent record bound is today in
                # the trusted timezone. Explicit subday requests are rejected
                # upstream, never rounded here.
                record_period={**record_period,**{key:(datetime.fromisoformat(record_period[key]).astimezone(ZoneInfo(request.state.time_zone)).date().isoformat()
                    if len(record_period[key])>10 else record_period[key]) for key in ('start_at','end_at')}}
            output.append(HealthRead(records=[record],operation='latest',period={'kind':'all_history'} if record=='profile' else record_period,
                profile_fields=['all'] if record=='profile' else [],limit=limit,date_basis=basis,source=source))
        if proposal['research']!='none':
            output.append(ResearchRead(targets=['sleep','physical_activity','cardiometabolic_health'],interventions=['diet','exercise']))
        return output

    def research(self,text,previous,request):
        if not request.literature_available:raise ValueError('research_not_available')
        # Construct the external research question entirely from public enum slots;
        # never forward private request text, values or personal history.
        targets=[];interventions=[]
        for term,target in [('apob','apob'),('ldl','ldl_cholesterol'),('glucose','glucose'),('sleep','sleep'),
                            ('physical activity','physical_activity'),('getting more active','physical_activity')]:
            if re.search(r'\b'+term+r'\b',text):targets.append(target)
        if re.search(r'\bdiet(?:ary)?\b',text):interventions.append('diet')
        if re.search(r'\b(?:exercise|physical activity)\b',text):interventions.append('exercise')
        if re.search(r'\b(?:sleep and exercise|exercise and sleep)\b',text):targets.append('physical_activity')
        if re.search(r'\b(?:it|both|them)\b',text):
            if not previous:raise ValueError('unresolved_research_reference')
            for q in previous:
                if isinstance(q,HealthRead):
                    for metric in q.metrics:
                        target='sleep' if 'sleep' in metric else 'physical_activity' if metric=='steps' else metric
                        if target not in ('sleep','physical_activity','apob','ldl_cholesterol','glucose'):
                            raise ValueError('unbound_research_target')
                        targets.append(target)
        if not targets:raise ValueError('unbound_research_target')
        remaining=re.sub(r"don't need my own numbers",'',text)
        allowed=set('please what do does the studies study trials trial randomized randomised research evidence say says about on diet dietary exercise physical activity for lowering bringing down improving improve using sleep getting more active any are there good find published tell me and both it them then how supports support lipid glucose ldl cholesterol apob my to'.split())
        words=re.findall(r'[a-z]+|\d+',remaining)
        if any(w not in allowed for w in words):raise ValueError('unbound_research_qualifier')
        return ResearchRead(targets=sorted(set(targets)),interventions=sorted(set(interventions)),
                            goal='lower' if re.search(r'\b(?:lowering|bringing|down)\b',text) else 'improve')
