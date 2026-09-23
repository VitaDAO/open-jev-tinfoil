"""Experimental one-pass Open-JEV selector; synthetic-trained, not release approved."""
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path
from selector import _base, _norm, _period, _split_period, _interpret, ALIASES, AREAS, RECORDS, SCHEMA
from routing import MODEL_REVISION

ROOT=Path(__file__).resolve().parent
# Filled from reproducible training artifact; changing this needs a new measured release.
ADAPTER_SHA256='52e885aefec1230c71b9f68f65cca7c94b67341cce962e03674430ef828ba69c'


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'selector.py').read_bytes()+ADAPTER_SHA256.encode()).hexdigest()


def dynamic_metrics(text,available):
    """Resolve exact identifiers/known aliases; longer names win overlapping spans."""
    labels={metric.replace('_',' '):[metric] for metric in available}
    labels.update({metric:[metric] for metric in available})
    labels.update({alias:[metric] for alias,metric in ALIASES.items() if metric in available})
    for area,members in AREAS.items():
        labels[area]=[metric for metric in available if metric in members or
                      (area=='sleep' and 'sleep' in metric.split('_'))]
    candidates=[]
    for label,metrics in labels.items():
        if not metrics:continue
        for match in re.finditer(r'(?<!\w)'+re.escape(label)+r'(?!\w)',text):
            candidates.append((match.start(),match.end(),metrics))
    occupied=[];selected=set()
    for start,end,metrics in sorted(candidates,key=lambda c:c[1]-c[0],reverse=True):
        if any(start<other_end and other_start<end for other_start,other_end in occupied):continue
        occupied.append((start,end));selected.update(metrics)
    return sorted(selected)


class UnrepresentableRequest(ValueError):
    """A recognized request constraint that this adapter cannot preserve."""


def normalize_request(text):
    text = _norm(text)
    text = re.sub(r'^please\s+', '', text)
    return re.sub(r',?\s+please$', '', text)


def resolve_request(request):
    """Resolve period-only follow-ups before any model or constraint checks.

    Preserve the previous subject verbatim, including unsupported source/count
    constraints. A new complete request does not inherit unrelated history.
    """
    current = normalize_request(request.state.current_request)
    reference = date.fromisoformat(request.state.reference_date)
    history = list(request.state.recent_user_requests)

    def resolve(text, previous):
        follow = re.fullmatch(r'(?:what about|how about|and|(?:actually,? )?make (?:that|it)) (.+)', text)
        if not follow:
            return text
        if not previous or _period(follow[1], reference) is None:
            raise UnrepresentableRequest('unresolved_followup')
        parent = resolve(normalize_request(previous[-1]), previous[:-1])
        subject, period = _split_period(parent, reference)
        if period['period_kind'] == 'unsupported':
            raise UnrepresentableRequest('unresolved_inherited_period')
        return subject + ' ' + follow[1]

    current = resolve(current, history)
    return request.model_copy(update={'state': request.state.model_copy(update={
        'current_request': current, 'recent_user_requests': []})})


# These words do not encode a source, exclusion, count, action or extra operation.
# Every other part of a health request must be bound to a metric, record or date.
FILLER = set('show give me my the a an what is are was were i have has how can could '
             'you would like to please most recent latest newest individual value values '
             'reading readings measurement measurements result results trend trends summary '
             'snapshot summarize summarise review analyze analyse assess assessment evaluate '
             'evolved changed over time of for in on from during overall comprehensive '
             'health healthwise wellbeing check checkin check-in and been doing about '
             'improve improvements suggest want just descriptive this past completed due planned'.split())


def bound_scope(subject, available):
    metrics = dynamic_metrics(subject, available)
    labels = {m.replace('_', ' ') for m in available} | set(available)
    labels.update(alias for alias, metric in ALIASES.items() if metric in available)
    labels.update(area for area in AREAS if dynamic_metrics(area, available))
    record_labels = {**RECORDS, 'health events': 'calendar'}
    records = set()
    remaining = subject
    for label in sorted(labels | set(record_labels), key=len, reverse=True):
        pattern = r'(?<!\w)' + re.escape(label) + r'(?!\w)'
        if re.search(pattern, remaining):
            if label in record_labels:
                records.add(record_labels[label])
            remaining = re.sub(pattern, ' ', remaining)
    tokens = re.findall(r'[a-z]+(?:-[a-z]+)?|[^\s]', remaining)
    if any(token not in FILLER for token in tokens):
        raise UnrepresentableRequest('unbound_request_constraint')
    if (metrics or records) and re.search(r'\b(?:health|healthwise|wellbeing|overall)\b', remaining):
        raise UnrepresentableRequest('mixed_broad_and_targeted_scope')
    if records & {'profile'} and re.search(r'\b(?:goals|medications|allergies)\b', subject):
        raise UnrepresentableRequest('narrow_profile_projection_unavailable')
    return metrics, sorted(records)


class LearnedSelector:
    def __init__(self,model):
        import torch
        from typed_decisions.schema import Question
        raw=(ROOT/'adapters/vita-selector-experimental-v1.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ADAPTER_SHA256:
            raise ValueError('Unapproved selector adapter')
        data=json.loads(raw)
        if data['model_revision']!=MODEL_REVISION or data['format_version']!=1:
            raise ValueError('Wrong selector model')
        self.torch=torch;self.model=model;self.heads=data['heads']
        self.weights=torch.tensor(data['weights'],dtype=torch.float64)
        if self.weights.shape!=(1024,sum(len(v) for v in self.heads.values())) or not torch.isfinite(self.weights).all():
            raise ValueError('Invalid selector weights')
        self.question=Question('selector','choice',data['question'],['health','research','other'],0)

    def select(self,request,*,confidence_policy="bound_arguments_v2",prediction=None):
        started=time.perf_counter();torch=self.torch
        if confidence_policy != 'bound_arguments_v2':
            raise ValueError('Unknown experimental confidence policy')
        answers=_base(request); reasons=[]; predicted={}; margins={}; required=[]
        try:
            resolved=resolve_request(request)
            current=resolved.state.current_request
            subject,period=_split_period(current,date.fromisoformat(request.state.reference_date))
            if period['period_kind']=='unsupported':
                raise UnrepresentableRequest('unresolved_period')
            # The finite research topics have a complete parser, including no-date checks.
            research_decision=_interpret(current,[],resolved)
            research_decision=research_decision if research_decision and research_decision['task']=='research' else None
            if research_decision:
                metrics,records=[],[]
            else:
                metrics,records=bound_scope(subject,request.available_metrics)
            latest=bool(re.search(r'\b(?:latest|newest|most recent)\b',subject))
            if latest and re.search(r'\b(?:trends?|evolved|changed|over time)\b',subject):
                raise UnrepresentableRequest('conflicting_latest_and_trend')
            if not (metrics or records or research_decision) and not re.search(
                    r'\b(?:health|healthwise|wellbeing)\b|\b(?:analyze|analyse) me\b',subject):
                raise UnrepresentableRequest('unresolved_metric_or_record')
            if prediction is None:
                text='Current request: '+current+'\nRecent user requests:\n'
                tokens=self.model.tok(text,add_special_tokens=False)['input_ids']
                if not 1<=len(tokens)<=256:
                    raise ValueError('Selector state exceeds 256 tokens; no truncation')
                with torch.inference_mode():
                    batch=self.model.collator([(text,[self.question])],self.model.device)
                    hidden=self.model.model.backbone(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).last_hidden_state
                    pooled=torch.nn.functional.normalize(hidden[0,2:2+len(tokens)].mean(0).double(),dim=0)
                    scores=pooled@self.weights
                offset=0
                for key,options in self.heads.items():
                    values=scores[offset:offset+len(options)];offset+=len(options)
                    top=values.topk(2)
                    predicted[key]=options[int(top.indices[0])]
                    margins[key]=float(top.values[0]-top.values[1])
            else:
                predicted,margins=prediction
            required=['task']
            task=predicted['task']
            if research_decision:
                if task!='research':
                    reasons.append('inconsistent_research_task')
                coverage='none';purpose='unsupported';research=research_decision['research']
            elif task=='health':
                coverage='targeted' if metrics or records else predicted['coverage']
                purpose='latest' if latest else 'trend'
                research='none'
                if not (metrics or records):
                    required+=['coverage']
                    if coverage!='broad':
                        reasons.append('unresolved_health_scope')
                    if request.literature_available:
                        descriptive=bool(re.search(r'\b(?:summary|summarize|summarise|snapshot|descriptive)\b',subject))
                        actionable=bool(re.search(r'\b(?:analyze|analyse|assess|evaluate|improve|improvements)\b',subject))
                        if descriptive and actionable:
                            reasons.append('mixed_summary_and_analysis')
                        elif not descriptive:
                            required+=['research'];research=predicted['research']
                        if research not in ('none','broad_overview'):
                            reasons.append('unsupported_broad_research')
                    records=['profile','workouts','labs','calendar']
            else:
                coverage='none';purpose='unsupported';research='none'
                reasons.append('no_supported_read')
            rejected=[name for name in required if margins[name]<.10]
            if rejected:reasons.append('uncertain_model_decision')
            basis='current_plans'
            if 'calendar' in records:
                if re.search(r'\bcompleted\b',subject):basis='last_done_date'
                elif re.search(r'\b(?:due|planned)\b',subject):basis='next_due_date'
                elif coverage!='broad' and period['period_kind'] not in ('unstated','all_history'):
                    reasons.append('unresolved_calendar_basis')
            elif re.search(r'\b(?:completed|due|planned)\b',subject):
                reasons.append('unbound_record_date_basis')
            if not reasons:
                for key,value in dict(task=task,coverage=coverage,purpose=purpose,research=research,calendar_basis=basis).items():
                    answers[key]={'choice':value}
                answers.update({key:{'choice':value} for key,value in period.items()})
                for metric in metrics:answers['metric__'+metric]={'choice':'include'}
                for record in records:answers[record]={'noul':1.0}
        except UnrepresentableRequest as exc:
            reasons.append(str(exc))
        return {'schema_version':SCHEMA,'status':'unsupported' if reasons else 'selected','reason':reasons[0] if reasons else None,
                'reason_codes':list(dict.fromkeys(reasons)),
                'diagnostics':{'confidence_policy':'bound_arguments_v2','required_heads':required,
                               'predicted_decisions':predicted},
                'answers':answers,'advisory':True,'selector_sha256':identity(),
                'model_revision':MODEL_REVISION,'adapter_sha256':ADAPTER_SHA256,
                'implementation':'open_jev_frozen_encoder_multihead_experimental',
                'decision_margins':margins,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}


def required_heads(predicted,request):
    """Experimental dependency mask; thresholds unchanged, not calibrated."""
    needed=['task']
    if request.state.recent_user_requests:needed.append('context')
    task=predicted['task']
    current=_norm(request.state.current_request)
    if task=='health':
        needed+=['coverage','purpose','period_kind']
        if predicted['coverage']!='broad':
            record_words={'profile':r'profile|goals|medications|conditions|allergies',
                          'workouts':r'workouts|exercise sessions',
                          'labs':r'lab reports|blood tests|lab results',
                          'calendar':r'calendar|plans|screening|events'}
            needed += [name for name,pattern in record_words.items()
                       if predicted[name]=='yes' or re.search(r'\b(?:'+pattern+r')\b',current)]
            if 'calendar' in needed:needed.append('calendar_basis')
    if request.literature_available and (task=='research' or predicted['coverage']=='broad' or
            predicted['research']!='none' or re.search(r'\b(?:research|evidence|studies|literature)\b',current)):
        needed.append('research')
    return list(dict.fromkeys(needed))
