"""Typed argument proposals for the experimental selector, never permissions.

Entity/date matches supply arguments, not intent. Unsupported operations remain
explicit constraints; arbitrary wording is judged by the model, not a word list.
"""
import calendar
import re
from datetime import date
from schema_index import INDEX, CATALOG, metric_labels
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from learned_selector import dynamic_metrics, normalize_request
from selector import ALIASES, AREAS, RECORDS, _dates, _period
from temporal_spans import extract_temporal


class Constraint(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: str
    evidence: str
    supported: bool = False


class ReadProposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task: Literal['health', 'research', 'other']
    coverage: Literal['broad', 'targeted', 'none']
    operation: Literal['latest', 'trend', 'unsupported']
    metrics: list[str] = Field(default_factory=list)
    records: list[Literal['profile', 'workouts', 'labs', 'calendar']] = Field(default_factory=list)
    period: dict[str, str]
    calendar_basis: Literal['current_plans', 'next_due_date', 'last_done_date'] = 'current_plans'
    research: Literal['none', 'broad_overview', 'sleep_activity', 'cardiometabolic', 'unsupported'] = 'none'
    constraints: list[Constraint] = Field(default_factory=list)


# Catalog synonyms, shared across wording forms; no whole-request templates.
METRIC_LABELS = INDEX['metric_aliases']
RECORD_LABELS = {alias: kind for kind, definition in INDEX['records'].items() for alias in definition['aliases']}
NUMBER_WORDS = dict(zip('one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(1, 20)))
NUMBER_WORDS.update(twenty=20, thirty=30, forty=40, fifty=50, sixty=60, seventy=70, eighty=80, ninety=90)
MONTHS = '|'.join(n.lower() for n in calendar.month_name if n)


def canonicalize(text):
    text = normalize_request(text).replace('’', "'")
    text = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', r'\1', text)
    for short, full in zip(calendar.month_abbr[1:], calendar.month_name[1:]):
        text = re.sub(r'\b' + short.lower() + r'\.?\b', full.lower(), text)
    text = re.sub(r'\bsept\b', 'september', text)
    for word, value in NUMBER_WORDS.items():
        # Only duration/count number words, not arbitrary prose such as "one".
        text = re.sub(r'\b' + word + r'(?= (?:days?|weeks?|months?|years?|readings?|results?|reports?|measurements?)\b)', str(value), text)
    text = re.sub(r'\b(?:past|last) fortnight\b', 'past 14 days', text)
    text = re.sub(r'\b(?:year-to-date|year to date)\b', 'this year', text)
    text = re.sub(r'\b(?:previous|current) (?:calendar )?(week|month|year)\b',
                  lambda m: ('last' if m[0].startswith('previous') else 'this') + ' ' + m[1], text)
    return text


def entities(text, available):
    labels = {m: [m] for m in available}
    for label, values in metric_labels(available).items():
        labels.setdefault(label, values)
    for label, value in {**ALIASES, **METRIC_LABELS}.items():
        if value in available:
            labels.setdefault(label, [value])
    ambiguous = {label for label, values in labels.items() if len(values) > 1}
    labels.update({k: dynamic_metrics(k, available) for k in AREAS})
    # Only explicit domain names intentionally expand to several metrics.
    labels = {k: [('ambiguous' if k in ambiguous and k not in AREAS else 'metric', v)
                  for v in vals] for k, vals in labels.items() if vals}
    labels.update({k: [('record', v)] for k, v in RECORD_LABELS.items()})
    # Known but unavailable measurements must not disappear next to an available
    # one. Only fill absent labels so exact inventory IDs keep their identity.
    for metric, definition in CATALOG.items():
        for label in [metric, metric.replace('_', ' '), definition['display_name'], *definition['aliases']]:
            labels.setdefault(label.casefold(), [('unavailable', metric)])
    matches = []
    for label, values in labels.items():
        for match in re.finditer(r'(?<!\w)' + re.escape(label) + r'(?!\w)', text, re.I):
            matches.append((match.start(), match.end(), values))
    spans = []
    for start, end, values in sorted(matches, key=lambda x: -(x[1] - x[0])):
        if any(start < b and a < end for a, b, _ in spans):
            continue
        spans.append((start, end, values))
    return sorted(spans)


def erase(text, spans):
    # Keep offsets stable for independently extracted entity/date spans.
    chars = list(text)
    for start, end, *_ in spans:
        chars[start:end] = ' ' * (end - start)
    return ''.join(chars)


def date_context_spans(text, spans):
    """Consume only prepositions/articles immediately attached to bound dates."""
    expanded = []
    for start, end in spans:
        prefix = re.search(r'\b(?:(?:from|for|over|during|in|on|throughout)\s+)?(?:the\s+)?$', text[:start])
        expanded.append((prefix.start() if prefix else start, end))
    return expanded


def operation_signals(text):
    # Negation applies to this operator only, never to arbitrary following text.
    negative = [(m.start(), m.end()) for m in re.finditer(
        r'\b(?:skip|not after) (?:the |a )?(?:trend(?: line)?|one number)(?=[ ,;.!?]|$)', text)]
    positive = erase(text, negative)
    positive = re.sub(r'\blatest (?=trend\b)', '       ', positive)
    latest = bool(re.search(r'\b(?:latest|newest|most recent)\b', positive))
    trend = bool(re.search(r'\b(?:trends?|trended|trending|over time)\b', positive))
    return latest, trend, negative


def research_requested(text):
    # A bounded opt-out clause cannot suppress another positive research clause.
    negative = [(m.start(), m.end()) for m in re.finditer(
        r'\bno (?:studies|research|recommendations)(?: please)?(?=\s*[,;.!?]|$)', text)]
    return bool(re.search(r'\b(?:research|evidence|studies|trials)\b', erase(text, negative)))


def temporal(text, reference):
    """Extract one complete period wherever it occurs; reject partial dates."""
    # Modal permission questions are not the month May. Preserve offsets and
    # leave independently stated dates, including a later "in May", untouched.
    text = re.sub(r'\bmay(?=\s+(?:i|we)\b)', '   ', text)
    # "Not after one number" negates a result shape, not a date boundary.
    _, _, negated_operators = operation_signals(text)
    text = erase(text, negated_operators)
    # Explicit named spans with inherited end month/year, including month ranges.
    named = re.compile(r'\b(' + MONTHS + r')\s+(\d{1,2}(?!\d)(?:,?\s+\d{4})?|\d{4})\s+(?:to|through|and)\s+(?:(' + MONTHS + r')\s+)?(?:the\s+)?(\d{1,2}(?!\d)(?:,?\s+\d{4})?|\d{4})\b')
    match = named.search(text)
    if match:
        def boundary(month, value, default_year):
            nums = [int(x) for x in re.findall(r'\d+', value)]
            day = nums[0] if nums[0] < 100 else None
            year = nums[-1] if nums[-1] >= 100 else default_year
            month_num = [n.lower() for n in calendar.month_name].index(month)
            date(year, month_num, day or 1)
            return year, month_num, day
        try:
            start = boundary(match[1], match[2], date.fromisoformat(reference).year)
            end = boundary(match[3] or match[1], match[4], start[0])
            # An omitted end year can cross New Year, never reverse same-month days.
            if match[3] and end[1] < start[1] and not re.search(r'\d{4}', match[4]):
                end = (end[0] + 1, end[1], end[2])
            if date(*start[:2], start[2] or 1) > date(*end[:2], end[2] or calendar.monthrange(*end[:2])[1]):
                raise ValueError('Inverted dates')
            if not 1900 <= start[0] <= end[0] <= 2100:
                raise ValueError('Date outside contract')
        except ValueError:
            return None, [], 'invalid_or_out_of_range_date'
        rest = erase(text, [(match.start(), match.end())])
        extra = extract_temporal(rest, reference)
        # "between" belongs to this explicit span.
        if extra['spans'] or extra['status'] == 'unsupported' or re.search(r'\b(?:since|before|after|until|onward)\b', rest):
            return None, [], 'multiple_or_open_periods'
        fields = _dates()
        fields.update(period_kind='absolute', **dict(zip(('start_year','start_month','start_day'), map(lambda x: 'none' if x is None else str(x), start))),
                      **dict(zip(('end_year','end_month','end_day'), map(lambda x: 'none' if x is None else str(x), end))))
        return fields, [(match.start(), match.end())], None
    iso_month = re.search(r'\b(\d{4})-(\d{2})(?![-\d])\b', text)
    if iso_month:
        try:
            date(int(iso_month[1]), int(iso_month[2]), 1)
            if not 1900 <= int(iso_month[1]) <= 2100: raise ValueError('year')
        except ValueError:
            return None, [], 'invalid_or_out_of_range_date'
        rest = erase(text, [(iso_month.start(), iso_month.end())])
        extra = extract_temporal(rest, reference)
        if extra['spans'] or extra['status'] == 'unsupported':
            return None, [], 'multiple_or_open_periods'
        return {**_dates(), 'period_kind':'absolute', 'start_year':iso_month[1],
                'start_month':str(int(iso_month[2]))}, [(iso_month.start(), iso_month.end())], None
    # Prefix "last" in a count is not a date, and "to" in polite prose is not a bound.
    masked = re.sub(r'\b(?:last|latest|most recent) \d+ (?!days?\b|weeks?\b|months?\b|years?\b)',
                    lambda m: ' ' * len(m[0]), text)
    result = extract_temporal(masked, reference)
    if result['status'] == 'unsupported' and result['reason'] == 'unbound_temporal_qualifier':
        rest = erase(masked, [(s['start'], s['end']) for s in result['spans']])
        if not re.search(r'\b(?:last|past|next|this|previous|between|through)\b', rest):
            result['status'] = 'resolved'; result['date_fields'] = result['spans'][0]['date_fields']
    spans = [(s['start'], s['end']) for s in result['spans']]
    if result['status'] == 'unsupported':
        return None, spans, result['reason']
    # Detect time-of-day and vague bounds even when a supported subspan exists.
    if re.search(r'\b(?:onward|hours?|night|weekend|summer|winter|spring|autumn|ago)\b', erase(text, spans)):
        return None, spans, 'unresolved_temporal_phrase'
    return result['date_fields'], spans, None


def resolve_context(request):
    """Resolve unambiguous period corrections while retaining the whole subject."""
    reference = request.state.reference_date
    def resolve(text, history):
        text = canonicalize(text)
        # A projection correction may select only an already-bound parent metric.
        correction = re.fullmatch(r'actually[,]? just (?:the )?(.+)', text)
        if correction:
            if not history: raise ValueError('unresolved_followup')
            parent = resolve(history[-1], history[:-1])
            old = entities(parent, request.available_metrics)
            candidates = {value for _,_,values in old for kind,value in values
                          if kind == 'metric' and correction[1] in value.replace('_', ' ').split()}
            if len(old) != 1 or len(candidates) != 1:
                raise ValueError('ambiguous_followup_subject')
            start,end,_ = old[0]
            return parent[:start] + next(iter(candidates)).replace('_', ' ') + parent[end:]
        trend_follow = re.fullmatch(r'and how has it (?:trended|been trending) (.+)', text)
        if trend_follow:
            if not history: raise ValueError('unresolved_followup')
            parent = resolve(history[-1], history[:-1])
            if len(entities(parent, request.available_metrics)) != 1:
                raise ValueError('ambiguous_followup_subject')
            _,spans,error = temporal(trend_follow[1],reference)
            if error or not spans or erase(trend_follow[1],date_context_spans(trend_follow[1],spans)).strip(' ,?!.'):
                raise ValueError('unresolved_followup')
            _,old_spans,error = temporal(parent,reference)
            if error: raise ValueError('unresolved_inherited_period')
            parent = erase(parent,date_context_spans(parent,old_spans))
            parent = re.sub(r'\b(?:latest|newest|most recent)\b','',parent)
            return parent.strip() + ' trend ' + trend_follow[1]
        replacement = re.fullmatch(r'(?:what about (?:my )?|same window for |no,? i meant )(.+?)(?: then)?', text)
        if replacement and entities(replacement[1], request.available_metrics):
            if not history: raise ValueError('unresolved_followup')
            parent = resolve(history[-1], history[:-1])
            old_entities = entities(parent, request.available_metrics)
            new_entities = entities(replacement[1], request.available_metrics)
            if len(old_entities) != 1 or len(new_entities) != 1:
                raise ValueError('ambiguous_followup_subject')
            # Replace only the old subject, preserving its dates and constraints.
            start, end, _ = old_entities[0]
            return parent[:start] + replacement[1] + parent[end:]
        follow = re.fullmatch(r'(?:what about|how about|and|(?:actually,? )?make (?:that|it)|now do|sorry[,]? i meant|(?:can you )?redo that but only looking at) (.+)', text)
        if not follow:
            return text
        period, spans, error = temporal(follow[1], reference)
        # Only period-only fragments qualify. Entity-bearing followups stay explicit.
        residue = erase(follow[1], date_context_spans(follow[1], spans)).strip(' ,?!.')
        if error or not spans or residue:
            if entities(follow[1], request.available_metrics):
                raise ValueError('unresolved_followup')
            raise ValueError('unresolved_followup')
        if not history:
            raise ValueError('unresolved_followup')
        parent = resolve(history[-1], history[:-1])
        _, parent_spans, parent_error = temporal(parent, reference)
        if parent_error:
            raise ValueError('unresolved_inherited_period')
        return erase(parent, date_context_spans(parent, parent_spans)).strip() + ' ' + follow[1]
    return resolve(request.state.current_request, request.state.recent_user_requests)


# Capability constraints are checked after model inference. They are not a prose
# vocabulary: unmatched wording reaches a semantic coverage decision.
CONSTRAINT_PATTERNS = {
    'source_filter': r'\b(?:oura|garmin|whoop|fitbit|withings|polar|apple watch|apple health|quest|labcorp|manually entered)\b',
    'exclusion_or_filter': r'\b(?:excluding|except|without|apart from|omit\w*|leave\b.*\bout|weekdays?|weekends?|mornings?|evenings?|awake|asleep|fasting|above|below|exceeded|under|over \d|only from)\b',
    'relationship_question': r'\b(?:does|do|can)\b.*\b(?:help|affect|cause|lead to)\b',
    'unsupported_operation': r'\b(?:compare|comparison|versus|vs|correlat\w*|affect|average|median|sum|how many|count the|in total|difference|higher than|lower than|better than|worse than|first|earliest|oldest|top|highest|lowest|worst|best)\b',
    'write_or_external_action': r'\b(?:create|delete|remove|log|add|set|update|change|email|send|upload|share|schedule|remind)\b',
    'other_person': r"\b(?:partner|spouse|wife|husband|daughter|son|mother|father|patient|someone else)(?:'s)?\b",
    'unit_conversion': r'\b(?:in hours|in minutes|in seconds|pounds|kilograms|convert\w*|instead of|rather than)\b',
    'instruction_override': r'\b(?:system|override|ignore|questionnaire|selector|coverage|task:)\b',
    'unresolved_context': r'\b(?:that one|other one|last time|last chatted|we discussed|what did i tell|what did you say)\b',
}


def constraints(text, entity_spans, date_spans):
    latest, trend, operator_negations = operation_signals(text)
    rest = erase(text, entity_spans + date_context_spans(text, date_spans) + operator_negations)
    if latest and not trend:
        # A singular anaphor denotes the latest value; arbitrary counts remain.
        rest = re.sub(r'\b((?:latest|newest|most recent)\s+)one\b', r'\1   ', rest)
    found = []
    for kind, pattern in CONSTRAINT_PATTERNS.items():
        for match in re.finditer(pattern, rest):
            found.append(Constraint(kind=kind, evidence=match[0]))
    for match in re.finditer(r'\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\b', rest):
        found.append(Constraint(kind='unbound_quantity', evidence=match[0]))
    for match in re.finditer(r'\b(?:from|by) (?:my |the )?([a-z][a-z -]*?)(?=[,.?!]|$)', rest):
        if match[1].strip(): found.append(Constraint(kind='unbound_source', evidence=match[0]))
    return found
