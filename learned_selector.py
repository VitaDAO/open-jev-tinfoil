"""Experimental one-pass Open-JEV selector; synthetic-trained, not release approved."""
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path
from selector import _base, _norm, _period, _split_period, ALIASES, AREAS, SCHEMA
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

    def select(self,request):
        started=time.perf_counter();torch=self.torch
        text='Current request: '+request.state.current_request+'\nRecent user requests:\n'+'\n'.join(request.state.recent_user_requests)
        tokens=self.model.tok(text,add_special_tokens=False)['input_ids']
        if not 1<=len(tokens)<=256:
            raise ValueError('Selector state exceeds 256 tokens; no truncation')
        with torch.inference_mode():
            batch=self.model.collator([(text,[self.question])],self.model.device)
            hidden=self.model.model.backbone(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).last_hidden_state
            pooled=torch.nn.functional.normalize(hidden[0,2:2+len(tokens)].mean(0).double(),dim=0)
            scores=pooled@self.weights
        predicted={};margins={};offset=0
        for key,options in self.heads.items():
            values=scores[offset:offset+len(options)];offset+=len(options)
            top=values.topk(2)
            predicted[key]=options[int(top.indices[0])]
            margins[key]=float(top.values[0]-top.values[1])
        answers=_base(request);reason=None
        current=_norm(request.state.current_request)
        period_subject,period=_split_period(current,date.fromisoformat(request.state.reference_date))
        scope=current
        if predicted['context']=='inherit':
            follow=re.fullmatch(r'(?:what about|how about|and) (?:the )?(.+)',current)
            if not follow or not request.state.recent_user_requests:
                reason='unresolved_followup'
            else:
                period=_period(follow[1],date.fromisoformat(request.state.reference_date))
                scope=_norm(request.state.recent_user_requests[-1])
                if not period:reason='unsupported_period'
                # Multi-hop elliptical history requires explicit resolution, not a guess.
                if re.match(r'^(what about|how about|and)\b',scope):reason='unresolved_followup'
        required=['task','coverage','purpose','research','period_kind','context','calendar_basis',
                  'profile','workouts','labs','calendar']
        if any(margins[name]<.10 for name in required):reason='uncertain_model_decision'
        if predicted['task']=='other':reason='no_supported_read'
        if predicted['task']=='health' and predicted['purpose']=='unsupported':reason='unsupported_read_form'
        if predicted['research']=='unsupported':reason='unsupported_research'
        if re.search(r'\b(?:not|without|except|only|ignore|delete|change|set|compare|correlate)\b',current):
            reason='unsupported_qualifier_or_action'
        if not period or predicted['period_kind']!=period['period_kind'] or predicted['period_kind']=='unsupported':
            reason='unresolved_period'
        if re.search(r'\b(?:garmin|oura|whoop|withings|fitbit|polar|apple health|labcorp|quest)\b',current):
            reason='source_qualified_query_requires_existing_tools'
        if re.search(r'\b(?:last|latest|top) \d+ (?:readings|reports|records|measurements)\b',current):
            reason='latest_n_requires_existing_tools'
        if re.search(r'\b(?:in pounds|in kilograms|in mmol|in milligrams|median|sum|count|how many|minimum|maximum)\b',current):
            reason='aggregate_or_unit_request_requires_existing_tools'
        metrics=dynamic_metrics(scope,request.available_metrics)
        records=[key for key in ('profile','workouts','labs','calendar') if predicted[key]=='yes']
        if predicted['task']=='health' and predicted['coverage']=='targeted' and not metrics and not records:
            reason='unresolved_metric_or_record'
        if predicted['task']=='health' and predicted['coverage']=='none':reason='inconsistent_scope'
        if predicted['task']=='research' and (predicted['research']=='none' or not request.literature_available):
            reason='no_available_research'
        if predicted['task']=='research' and (not period or period['period_kind']!='unstated' or predicted['coverage']!='none' or records):
            reason='unsupported_research_scope'
        if 'profile' in records and re.search(r'\b(?:medications|goals|allergies)\b',scope) and predicted['coverage']!='broad':
            reason='narrow_profile_projection_unavailable'
        if 'calendar' in records and predicted['calendar_basis']=='current_plans' and predicted['coverage']!='broad' and period and period['period_kind'] not in ('unstated','all_history'):
            reason='unresolved_calendar_basis'
        if not reason:
            for key in ('task','coverage','purpose','calendar_basis','research'):
                answers[key]={'choice':predicted[key]}
            if not request.literature_available:answers['research']={'choice':'none'}
            for key,value in period.items():answers[key]={'choice':value}
            for metric in metrics:answers[f'metric__{metric}']={'choice':'include'}
            for record in records:answers[record]={'noul':1.0}
        return {'schema_version':SCHEMA,'status':'unsupported' if reason else 'selected','reason':reason,
                'answers':answers,'advisory':True,'selector_sha256':identity(),
                'model_revision':MODEL_REVISION,'adapter_sha256':ADAPTER_SHA256,
                'implementation':'open_jev_frozen_encoder_multihead_experimental',
                'decision_margins':margins,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
