"""Bounded canonical lexical spans for a future learned query-IR decoder.

This module proposes entities; it does not decide intent, fetch data, or claim
semantic resolution of unseen paraphrases. Unknown exact inventory IDs work.
"""
import re
from plan_adapter import CATALOG,POLICY

DOMAIN_LABELS={'sleep':'sleep','body composition':'body','activity':'activity',
               'recovery':'recovery','vitals':'vitals','blood tests':'labs'}
RECORD_LABELS={'lab reports':'labs','workouts':'workouts','exercise sessions':'workouts',
               'profile':'profile','health plans':'calendar','screening events':'calendar'}
PROFILE_LABELS={'medications':'medications','goals':'goals','allergies':'allergies',
                'conditions':'chronic_conditions','birth year':'birth_year','height':'height_cm'}


def catalog_candidates(text,available_metrics):
    labels={}
    for metric in available_metrics:
        if not isinstance(metric,str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,95}',metric):raise ValueError('Invalid inventory identifier')
        definition=CATALOG['metrics'].get(metric,{})
        for label in [metric,metric.replace('_',' '),definition.get('display_name',''),*definition.get('aliases',[])]:
            if label:labels.setdefault(label.casefold(),set()).add(('metric',metric))
    for label,domain in DOMAIN_LABELS.items():
        if any(POLICY['domains'].get(m)==domain for m in available_metrics):labels.setdefault(label,set()).add(('domain',domain))
    for label,record in RECORD_LABELS.items():labels.setdefault(label,set()).add(('record',record))
    for label,field in PROFILE_LABELS.items():labels.setdefault(label,set()).add(('profile_field',field))
    sources={s for definition in CATALOG['metrics'].values() for s in definition['valid_sources']}
    for source in sources:labels.setdefault(source.replace('_',' '),set()).add(('source',source))
    matches=[]
    for label,values in labels.items():
        for match in re.finditer(r'(?<!\w)'+re.escape(label)+r'(?!\w)',text,re.I):
            matches.append((match.start(),match.end(),values))
    result=[];occupied=[];reasons=[]
    for start,end,values in sorted(matches,key=lambda x:(-(x[1]-x[0]),x[0])):
        if any(start<b and a<end for a,b in occupied):continue
        occupied.append((start,end))
        left=text[max(0,start-24):start]
        polarity='exclude' if re.search(r'\b(?:no|not|without|excluding|except)\s*$',left,re.I) else 'include'
        if len(values)>1:reasons.append('ambiguous_catalog_entity')
        for kind,value in sorted(values):
            result.append({'kind':kind,'value':value,'start':start,'end':end,'polarity':polarity})
    if re.search(r'\bnot only\b',text,re.I):reasons.append('negation_binding_requires_semantic_decoder')
    return {'status':'ambiguous' if reasons else 'candidates','reason_codes':sorted(set(reasons)),
            'entities':sorted(result,key=lambda x:(x['start'],x['end'],x['kind'])),
            'advisory':True}
