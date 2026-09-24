"""Bounded date phrase extraction, separate from learned intent classification.

Returns explicit spans and legacy date fields. It does not decide which health
operation owns a span or silently choose one of multiple periods.
"""
import calendar
import re
from datetime import date
from selector import _period,_dates,DATE_KEYS

MONTHS='|'.join(name.lower() for name in calendar.month_name if name)
DAY=r'\d{1,2}(?:st|nd|rd|th)?'
ISO=r'\d{4}-\d{2}-\d{2}'
PATTERN=re.compile(r'\b(?:'+ISO+r'(?:\s+(?:to|through|and)\s+'+ISO+r')?'
  +r'|(?:today|yesterday|tomorrow)'
  +r'|(?:this|last|next|previous|current)\s+(?:calendar\s+)?(?:week|month|year)'
  +r'|(?:past|last)\s+(?:(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+)?(?:days?|weeks?|months?|years?)'
  +r'|(?:'+MONTHS+r')(?:\s+(?:'+DAY+r'(?:,\s*|\s+)\d{4}|\d{4}|'+DAY+r'))?'
  +r'|all (?:available )?history|all time|\d{4})\b',re.I)
NUMBERS=dict(zip(('one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'),range(1,13)))


def _since(text,reference):
    """'since <month [year]|ISO date>' reads from that day through the reference day."""
    if len(re.findall(r'\bsince\b',text,re.I))!=1 or re.search(r'\b(?:before|after|until)\b',text,re.I):
        return None
    m=re.search(r'\bsince (?:(\d{4}-\d{2}-\d{2})|('+MONTHS+r')(?: (\d{1,2})(?:st|nd|rd|th)?)?(?:,? (\d{4}))?)\b',text,re.I)
    if not m:
        return None
    try:
        if m[1]:
            start=date.fromisoformat(m[1])
        else:
            month=[n.lower() for n in calendar.month_name].index(m[2].lower())
            day=int(m[3]) if m[3] else 1
            year=int(m[4]) if m[4] else (reference.year if (month,day)<=(reference.month,reference.day) else reference.year-1)
            start=date(year,month,day)
    except ValueError:
        return False
    if not date(1900,1,1)<=start<=reference:
        return False
    fields={'period_kind':'absolute',**dict.fromkeys(DATE_KEYS,'none')}
    fields.update(start_year=str(start.year),start_month=str(start.month),start_day=str(start.day),
                  end_year=str(reference.year),end_month=str(reference.month),end_day=str(reference.day))
    return {'start':m.start(),'end':m.end(),'phrase':m.group(),'date_fields':fields}


def extract_temporal(text,reference_date):
    reference=date.fromisoformat(reference_date)
    since=_since(text,reference)
    if since is False:
        return {'status':'unsupported','reason':'invalid_or_out_of_range_date','spans':[],'date_fields':None}
    if since:
        rest=text[:since['start']]+text[since['end']:]
        if PATTERN.search(rest) or re.search(r'\d|\b(?:last|past|next|this|previous|between|through)\b',rest,re.I):
            return {'status':'unsupported','reason':'multiple_periods_require_clause_binding','spans':[since],'date_fields':None}
        return {'status':'resolved','reason':None,'spans':[since],'date_fields':since['date_fields']}
    if re.search(r'\b(?:since|before|after|until)\b',text,re.I):
        return {'status':'unsupported','reason':'open_or_unresolved_bounds','spans':[],'date_fields':None}
    if re.search(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',text):
        return {'status':'unsupported','reason':'ambiguous_numeric_date','spans':[],'date_fields':None}
    spans=[]
    for match in PATTERN.finditer(text):
        phrase=match.group().lower()
        phrase=re.sub(r'(\d)(?:st|nd|rd|th)\b',r'\1',phrase)
        phrase=re.sub(r'\b(previous|current)\b',lambda m:'last' if m[1]=='previous' else 'this',phrase)
        phrase=phrase.replace('calendar ','')
        phrase=re.sub(r'\b('+'|'.join(NUMBERS)+r')\b',lambda m:str(NUMBERS[m[1]]),phrase)
        if re.fullmatch(ISO+r'\s+and\s+'+ISO,phrase):phrase=phrase.replace(' and ',' to ')
        fields=_period(phrase,reference)
        spans.append({'start':match.start(),'end':match.end(),'phrase':match.group(),'date_fields':fields})
    if any(s['date_fields'] is None for s in spans):
        return {'status':'unsupported','reason':'invalid_or_out_of_range_date','spans':spans,'date_fields':None}
    if len(spans)>1:
        return {'status':'unsupported','reason':'multiple_periods_require_clause_binding','spans':spans,'date_fields':None}
    if spans:
        # Unconsumed temporal qualifiers cannot narrow/broaden a valid subspan.
        rest=text[:spans[0]['start']]+text[spans[0]['end']:]
        if re.search(r'\b(?:last|past|next|this|previous|between|to|through)\b',rest,re.I) and not re.search(r'\bbetween\s*$',text[:spans[0]['start']],re.I):
            return {'status':'unsupported','reason':'unbound_temporal_qualifier','spans':spans,'date_fields':None}
        return {'status':'resolved','reason':None,'spans':spans,'date_fields':spans[0]['date_fields']}
    if re.search(r'\b(?:last|past|next|this|previous|spring|summer|autumn|winter|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',text,re.I):
        return {'status':'unsupported','reason':'unresolved_temporal_phrase','spans':[],'date_fields':None}
    return {'status':'unstated','reason':None,'spans':[],'date_fields':_dates()}


def extract_count_constraint(text):
    """A row-count phrase is never a rolling duration; preserve the whole span."""
    number=r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)'
    match=re.search(r'\b(?:last|latest|most recent|top)\s+'+number+r'\s+((?:(?!\b(?:days?|weeks?|months?|years?)\b).){0,100}?)\b(readings|reports|records|measurements|observations)\b',text,re.I)
    if not match:return None
    raw=match[1].lower();count=int(raw) if raw.isdigit() else NUMBERS[raw]
    return {'limit':count,'start':match.start(),'end':match.end(),'entity_text':match[2].strip(),'kind':match[3].lower()}
