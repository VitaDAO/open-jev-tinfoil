"""Conservative, versioned read selector. No model, network, records or permissions.

This is a bounded grammar, not an Open-JEV accuracy improvement. Unrecognised
wording returns unsupported so the caller can use its existing selector/model.
Calendar arithmetic remains in Vita's read_arguments consumer.
"""
import calendar
import hashlib
import re
import time
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from typing import Annotated, Literal
from pydantic import StringConstraints

SCHEMA = 'vita-selector/v1'
SELECTOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
Metric = Annotated[str, StringConstraints(strict=True, pattern=r'^[a-z0-9][a-z0-9_]{0,95}$')]
RequestText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=1200)]

class SelectorState(BaseModel):
    model_config = ConfigDict(extra='forbid')
    current_request: RequestText
    recent_user_requests: list[RequestText] = Field(default_factory=list, max_length=4)
    reference_date: Annotated[str, StringConstraints(strict=True, pattern=r'^\d{4}-\d{2}-\d{2}$')]
    time_zone: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=64)]

    @model_validator(mode='after')
    def check_calendar(self):
        date.fromisoformat(self.reference_date)
        try:
            ZoneInfo(self.time_zone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError('Invalid time zone') from None
        return self

class SelectorRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: str = Field(pattern=r'^vita-selector/v1$')
    state: SelectorState
    available_metrics: list[Metric] = Field(max_length=512)
    literature_available: StrictBool = False
    # Compatibility default only; clients should supply their actual authorized inventory.
    available_record_types: list[Literal['profile','workouts','labs','calendar']] = Field(
        default_factory=lambda: ['profile','workouts','labs','calendar'], max_length=4)

    @model_validator(mode='after')
    def distinct_metrics(self):
        if len(set(self.available_record_types)) != len(self.available_record_types):
            raise ValueError('Duplicate record categories')
        if len(set(self.available_metrics)) != len(self.available_metrics):
            raise ValueError('Duplicate metrics')
        return self

DATE_KEYS = ('calendar_unit', 'calendar_offset', 'rolling_amount', 'rolling_unit',
             'start_year', 'start_month', 'start_day', 'end_year', 'end_month', 'end_day')
ALIASES = {
    'hrv': 'heart_rate_variability', 'resting heart rate': 'resting_heart_rate',
    'blood oxygen': 'oxygen_saturation', 'ldl': 'ldl_cholesterol', 'hdl': 'hdl_cholesterol',
    'a1c': 'hba1c', 'apo b': 'apob', 'vo2 max': 'vo2_max',
}
AREAS = {
    'sleep': ('sleep',), 'lipids': ('apob', 'ldl_cholesterol', 'hdl_cholesterol', 'triglycerides'),
    'blood pressure': ('systolic_blood_pressure', 'diastolic_blood_pressure'),
}
RECORDS = {'profile': 'profile', 'goals': 'profile', 'medications': 'profile',
           'workouts': 'workouts', 'exercise sessions': 'workouts',
           'lab reports': 'labs', 'health plans': 'calendar', 'screening events': 'calendar'}


def _norm(text):
    return re.sub(r'\s+', ' ', text.lower().strip().rstrip('?.!'))


def _dates():
    return {'period_kind': 'unstated', **dict.fromkeys(DATE_KEYS, 'none')}


def _period(text, reference):
    """Parse a whole period phrase, never discard unconsumed qualifiers."""
    out = _dates()
    if text in ('all time', 'all history', 'all available history'):
        return {**out, 'period_kind': 'all_history'}
    if text in ('today', 'yesterday', 'tomorrow'):
        return {**out, 'period_kind': 'calendar', 'calendar_unit': 'day',
                'calendar_offset': {'today': 'current', 'yesterday': 'previous', 'tomorrow': 'next'}[text]}
    m = re.fullmatch(r'(this|last|next) (week|month|year)', text)
    if m:
        return {**out, 'period_kind': 'calendar', 'calendar_unit': m[2],
                'calendar_offset': {'this': 'current', 'last': 'previous', 'next': 'next'}[m[1]]}
    m = re.fullmatch(r'(?:past|last) (?:(\d+) )?(day|week|month|year)s?', text)
    if m:
        n = int(m[1] or 1)
        if n not in {*range(1, 121), 180, 365, 730}:
            return None
        return {**out, 'period_kind': 'rolling', 'rolling_amount': str(n), 'rolling_unit': m[2] + 's'}
    m = re.fullmatch(r'(\d{4}-\d{2}-\d{2})(?: (?:to|through) (\d{4}-\d{2}-\d{2}))?', text)
    if m:
        try:
            start = date.fromisoformat(m[1]); end = date.fromisoformat(m[2]) if m[2] else start
            if not (1900 <= start.year <= end.year <= 2100) or end < start:
                return None
        except ValueError:
            return None
        out.update(period_kind='absolute', start_year=str(start.year), start_month=str(start.month), start_day=str(start.day))
        if m[2]:
            out.update(end_year=str(end.year), end_month=str(end.month), end_day=str(end.day))
        return out
    months = {name.lower(): str(n) for n, name in enumerate(calendar.month_name) if name}
    m = re.fullmatch(r'(' + '|'.join(months) + r')(?: (\d{1,2})(?:,)?(?= |$))?(?: (\d{4}))?', text)
    # Month alone or month + year (the optional day requires a trailing space).
    if m:
        year = int(m[3]) if m[3] else reference.year
        day = int(m[2]) if m[2] else None
        try:
            if not 1900 <= year <= 2100:
                return None
            date(year, int(months[m[1]]), day or 1)
        except ValueError:
            return None
        return {**out, 'period_kind': 'absolute', 'start_year': str(year) if m[3] else 'reference_year',
                'start_month': months[m[1]], 'start_day': str(day) if day else 'none'}
    if re.fullmatch(r'\d{4}', text) and 1900 <= int(text) <= 2100:
        return {**out, 'period_kind': 'absolute', 'start_year': text}
    return None


def _split_period(text, reference):
    # Try suffixes; the remaining subject must subsequently match the full grammar.
    temporal = re.search(r'\b(?:today|yesterday|tomorrow|this|last|next|past|since|before|after|until|all time|all history|all available history|'
                         + '|'.join(name.lower() for name in calendar.month_name if name)
                         + r'|\d{4}(?:-\d{2}-\d{2})?)\b', text)
    if temporal is None:
        return text, _dates()
    value = _period(text[temporal.start():], reference)
    subject = text[:temporal.start()].strip()
    subject = re.sub(r'\b(?:in|for|during|over|from|on|the)$', '', subject).strip()
    subject = re.sub(r'\b(?:in|for|during|over|from|on)$', '', subject).strip()
    return subject, value or {**_dates(), 'period_kind': 'unsupported'}



def _base(request):
    values = {'task': 'other', 'coverage': 'none', 'purpose': 'unsupported',
              'calendar_basis': 'current_plans', **_dates(), 'research': 'none'}
    values.update({f'metric__{metric}': 'skip' for metric in sorted(request.available_metrics)})
    answers = {key: {'choice': value} for key, value in values.items()}
    answers.update({name: {'noul': 0.0} for name in ('profile', 'workouts', 'labs', 'calendar')})
    return answers


def _interpret(text, history, request):
    reference = date.fromisoformat(request.state.reference_date)
    text = _norm(text)
    followup = re.fullmatch(r'(?:what about|how about|and) (.+)', text)
    if followup:
        period = _period(followup[1], reference)
        if not period or not history:
            return None
        previous = _interpret(history[-1], history[:-1], request)
        if previous is None or previous['task'] != 'health':
            return None
        return {**previous, **period}
    subject, period = _split_period(text, reference)
    if period['period_kind']=='unsupported':
        return None
    subject = re.sub(r'^please ', '', subject)
    result = {'task': 'health', 'coverage': 'targeted', 'purpose': 'trend',
              'calendar_basis': 'current_plans', 'research': 'none', 'metrics': [], 'records': [], **period}
    if re.fullmatch(r'(?:analy[sz]e me|analy[sz]e my (?:overall )?health|give me (?:an? )?(?:overall |comprehensive )?health analysis)', subject):
        return {**result, 'coverage': 'broad', 'records': list(RECORDS.values()),
                'research': 'broad_overview' if request.literature_available else 'none'}
    if re.fullmatch(r'(?:(?:show|give) me |show |give )?(?:a |my )?(?:health (?:summary|overview)|summary of my health)|summari[sz]e my (?:overall )?health', subject):
        return {**result, 'coverage': 'broad', 'records': list(RECORDS.values())}
    # General research has no personal reads. Only the consumer's finite topics fit.
    research = re.fullmatch(r'(?:find|show|summari[sz]e)(?: me)? (?:research|published evidence|studies) (?:on|about) (.+)', subject)
    if research:
        topics = {'sleep and physical activity': 'sleep_activity', 'sleep and exercise': 'sleep_activity',
                  'diet and exercise for lipid and glucose control': 'cardiometabolic',
                  'sleep, physical activity, diet and cardiometabolic health': 'broad_overview'}
        topic = topics.get(research[1])
        if not topic or not request.literature_available or period['period_kind'] != 'unstated':
            return None
        return {**result, 'task': 'research', 'coverage': 'none', 'purpose': 'unsupported', 'research': topic}
    # A closed set of whole-request templates prevents an unmatched qualifier,
    # negation, provider filter, or second action from broadening a read.
    m = re.fullmatch(r'(?:show(?: me)? |what (?:is|are) |give me )?(?:my )?(?:latest|most recent) (.+?)(?: readings?| measurements?)?', subject)
    if m:
        result['purpose'] = 'latest'; target = m[1]
    else:
        m = re.fullmatch(r'(?:show(?: me)? |summari[sz]e |analy[sz]e |review |what (?:is|are) )?(?:my )?(.+?)(?: trends?| summary| measurements?| readings?)?', subject)
        if not m:
            return None
        target = m[1]
    calendar_target = re.fullmatch(r'(?:health plans|health events|screening events) (due|planned|completed)', target)
    if calendar_target:
        return {**result, 'records': ['calendar'], 'calendar_basis':
                'last_done_date' if calendar_target[1] == 'completed' else 'next_due_date'}
    available = set(request.available_metrics)
    labels = {metric.replace('_', ' '): [metric] for metric in available}
    labels.update({metric: [metric] for metric in available})
    labels.update({alias: [metric] for alias, metric in ALIASES.items() if metric in available})
    for area, members in AREAS.items():
        selected = [metric for metric in available if metric in members or
                    (area == 'sleep' and 'sleep' in metric.split('_'))]
        if selected:
            labels[area] = selected
    selected_metrics, selected_records = set(), set()
    for part in re.split(r',\s*| and ', target):
        if part in labels:
            selected_metrics.update(labels[part])
        elif part in RECORDS:
            selected_records.add(RECORDS[part])
        else:
            return None
    if not selected_metrics and not selected_records:
        return None
    # The consumer always projects broad profile fields; do not pretend it can
    # honour narrower goals/medication-only record projections.
    if selected_records and any(part in ('goals', 'medications') for part in re.split(r',\s*| and ', target)):
        return None
    if 'calendar' in selected_records and period['period_kind'] not in ('unstated', 'all_history'):
        return None
    return {**result, 'metrics': sorted(selected_metrics), 'records': sorted(selected_records)}


def select(request):
    started = time.perf_counter()
    answers = _base(request)
    decision = _interpret(request.state.current_request, request.state.recent_user_requests, request)
    if decision is not None:
        for key, value in decision.items():
            if key not in ('metrics', 'records'):
                answers[key] = {'choice': value}
        for metric in decision['metrics']:
            answers[f'metric__{metric}'] = {'choice': 'include'}
        for record in decision['records']:
            answers[record] = {'noul': 1.0}
    return {'schema_version': SCHEMA, 'status': 'selected' if decision else 'unsupported',
            'reason': None if decision else 'outside_supported_grammar',
            'answers': answers, 'advisory': True, 'selector_sha256': SELECTOR_SHA256,
            'implementation': 'bounded_grammar_no_model',
            'elapsed_ms': round((time.perf_counter() - started) * 1000, 3)}
