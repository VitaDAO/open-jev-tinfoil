"""Experimental Vita selector using Open-JEV's native typed decisions.

The model judges only bounded semantic choices. Dates, entities, cardinality,
capability limits and the final Vita plan are checked in code. This is not a
deployed replacement for Vita's current selector.
"""
import hashlib
import re
import time
from pathlib import Path

from entity_candidates import catalog_candidates
from routing import MODEL_REVISION
from selector import SCHEMA, _base, _norm
from temporal_spans import extract_count_constraint, extract_temporal


IDENTITY = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

QUESTIONS = [
    {'type': 'choice', 'instructions': 'Which action does the current user request?',
     'options': ['read my health information', 'find published health research',
                 'change or delete information', 'no information read']},
    {'type': 'choice', 'instructions': 'How much of the user health information is requested?',
     'options': ['broad health overview', 'specific measurement or record', 'no personal health information']},
    {'type': 'choice', 'instructions': 'What kind of health data read does the user request?',
     'options': ['latest individual measurement or record', 'summary or trend over time',
                 'comparison or calculation', 'no health data read']},
    {'type': 'choice', 'instructions': 'Which kind of answer does the user request?',
     'options': ['health analysis with research and suggestions', 'descriptive personal data summary only',
                 'published research without personal data', 'none of these']},
]


class DirectSelector:
    def __init__(self, model):
        self.model = model

    def select(self, request):
        started = time.perf_counter()
        current = _norm(request.state.current_request)
        state = request.state.current_request
        reasons = []
        if request.state.recent_user_requests:
            # An elliptical reference must bind to a specific prior request;
            # the current one-pass contract does not establish that binding.
            reasons.append('context_binding_requires_existing_model')
        if len(self.model.tok(state, add_special_tokens=False)['input_ids']) > 256:
            reasons.append('state_token_limit')
        temporal = extract_temporal(current, request.state.reference_date)
        if temporal['status'] == 'unsupported':
            reasons.append(temporal['reason'])
        if extract_count_constraint(current):
            reasons.append('exact_record_count_requires_existing_tools')
        if re.search(r'\b(?:not|without|except|only|ignore|delete|change|set|compare|correlate|average|median|sum|count|how many|minimum|maximum)\b', current):
            reasons.append('compound_or_unrepresentable_read')
        if re.search(r'\b(?:garmin|oura|whoop|withings|fitbit|polar|apple health|labcorp|quest)\b', current):
            reasons.append('source_filter_requires_existing_tools')
        candidates = catalog_candidates(current, request.available_metrics)
        if candidates['status'] != 'candidates':
            reasons.extend(candidates['reason_codes'])
        # Keep an actual model call even on a handoff for diagnostic parity;
        # callers must not execute a plan when any deterministic gate rejects.
        try:
            raw = self.model.decide(state, QUESTIONS)
        except ValueError:
            raw = None
            reasons.append('native_decision_exceeds_context')
        answers = _base(request)
        if raw is not None:
            valid = (isinstance(raw, list) and len(raw) == len(QUESTIONS) and
                     all(isinstance(item, dict) and item.get('choice') in question['options'] and
                         type(item.get('confidence')) in (int, float) and
                         0 <= item['confidence'] <= 1
                         for item, question in zip(raw, QUESTIONS)))
            if not valid:
                reasons.append('invalid_native_decision')
                raw = None
        if raw is not None:
            choices = [item['choice'] for item in raw]
            confidence = [item['confidence'] for item in raw]
            if any(c < .45 for c in confidence):
                reasons.append('uncertain_native_decision')
            task, coverage, purpose, answer_style = choices
            if task == 'change or delete information':
                reasons.append('write_requires_existing_path')
            if task == 'no information read':
                reasons.append('no_supported_read')
            if task == 'find published health research':
                reasons.append('research_topic_requires_existing_path')
            if task == 'read my health information' and coverage == 'no personal health information':
                reasons.append('inconsistent_scope')
            if task == 'read my health information' and purpose in ('comparison or calculation', 'no health data read'):
                reasons.append('unsupported_read_form')
            if task == 'read my health information' and answer_style == 'published research without personal data':
                reasons.append('inconsistent_answer_style')
            explicit_latest = bool(re.search(r'\b(?:latest|newest|most recent|current value)\b', current))
            if explicit_latest and purpose != 'latest individual measurement or record':
                reasons.append('latest_purpose_conflict')
            if not explicit_latest and re.search(r'\b(?:trend|changed|summary|summarize|summarise|over time)\b', current) and purpose == 'latest individual measurement or record':
                reasons.append('trend_purpose_conflict')
            broad = coverage == 'broad health overview'
            metrics = sorted({e['value'] for e in candidates['entities'] if e['kind'] == 'metric' and e['polarity'] == 'include'})
            records = sorted({e['value'] for e in candidates['entities'] if e['kind'] == 'record' and e['polarity'] == 'include'})
            if not broad and not metrics and not records:
                reasons.append('unresolved_metric_or_record')
            if broad and metrics:
                # A mixed broad/narrow request needs clause binding.
                reasons.append('mixed_scope_requires_existing_model')
            if not reasons:
                answers['task'] = {'choice': 'health'}
                answers['coverage'] = {'choice': 'broad' if broad else 'targeted'}
                answers['purpose'] = {'choice': 'latest' if purpose == 'latest individual measurement or record' else 'trend'}
                answers['research'] = {'choice': 'broad_overview' if broad and answer_style == 'health analysis with research and suggestions' and request.literature_available else 'none'}
                for key, value in temporal['date_fields'].items():
                    answers[key] = {'choice': value}
                for metric in metrics:
                    answers['metric__' + metric] = {'choice': 'include'}
                for record in (['profile', 'workouts', 'labs', 'calendar'] if broad else records):
                    answers[record] = {'noul': 1.0}
        return {'schema_version': SCHEMA, 'status': 'unsupported' if reasons else 'selected',
                'reason': reasons[0] if reasons else None, 'reason_codes': list(dict.fromkeys(reasons)),
                'answers': answers, 'advisory': True, 'selector_sha256': IDENTITY,
                'model_revision': MODEL_REVISION, 'implementation': 'open_jev_native_typed_questions_experimental',
                'native_answers': raw, 'elapsed_ms': round((time.perf_counter() - started) * 1000, 3)}
