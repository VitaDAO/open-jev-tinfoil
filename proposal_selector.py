"""Model-first proposal selector; experimental and disabled by default.

The frozen ridge heads propose intent. The native Open-JEV head checks semantic
coverage against the actual bound proposal. Neither model can grant permission
or execute tools. Unrepresentable constraints return to the caller's model.
"""
import hashlib
import math
import re
import time
from pathlib import Path

from learned_selector import LearnedSelector, ADAPTER_SHA256
from proposal_binding import ReadProposal, canonicalize, entities, erase, temporal, resolve_context, constraints, operation_signals, research_requested
from routing import MODEL_REVISION
from schema_index import describe_targets
from selector import SCHEMA, _base, _interpret

ROOT = Path(__file__).resolve().parent
SOURCES = ('proposal_selector.py', 'proposal_binding.py', 'learned_selector.py',
           'selector.py', 'temporal_spans.py', 'schema_index.py',
           'metadata/health_metrics.v1.json', 'metadata/selector-index.v1.json')


def identity():
    return hashlib.sha256(b''.join((ROOT / p).read_bytes() for p in SOURCES) + ADAPTER_SHA256.encode()).hexdigest()


class ProposalSelector(LearnedSelector):
    def intent_check(self, current):
        semantic = {'type':'choice', 'instructions':'What is the user asking to do?',
            'options':['View their recorded health data', 'Get a general explanation',
                       'Calculate, compare or find a relationship', 'Change or share data',
                       'Filter data by source, condition or count', 'The request is not in English']}
        intent = self.model.decide(current, [semantic])
        if not isinstance(intent, list) or len(intent) != 1 or not isinstance(intent[0], dict):
            return None, 'invalid_semantic_decision'
        score = intent[0].get('confidence')
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            return None, 'invalid_semantic_decision'
        if intent[0].get('choice') not in semantic['options']:
            return None, 'invalid_semantic_decision'
        if intent[0]['choice'] != semantic['options'][0] or score < .40:
            return intent[0], 'semantic_read_intent_unconfirmed'
        return intent[0], None

    def coverage_check(self, current, proposal):
        intent = None
        if proposal.task == 'health':
            intent, error = self.intent_check(current)
            if error:
                return {'intent':intent}, error
        target = 'all health metrics and profile, workouts, labs, calendar' if proposal.coverage == 'broad' else describe_targets(proposal.metrics, proposal.records)
        period = '; '.join(f'{k}={v}' for k, v in proposal.period.items() if v not in ('none', 'unstated')) or 'default history'
        instruction = (
            'Does the proposed data read cover the user request accurately? '
            'Reject if it omits a requested metric, source, filter, exclusion, count, calculation, '
            'different person or action. General explanations without a personal data read are not reads. '
            f'Proposal: {proposal.operation}; targets: {target}; dates: {period}; '
            f'research: {proposal.research}. No other filters or actions are supported.')
        question = {'type':'choice', 'instructions':instruction,
                    'options':['the proposed read covers the request', 'the request needs something else']}
        # The upstream collator truncates. Explicitly check both state and question
        # budgets so neither the user request nor the proposal is silently lost.
        tok = self.model.tok
        state_n = len(tok(current, add_special_tokens=False)['input_ids'])
        q_n = len(tok(instruction, add_special_tokens=False)['input_ids'])
        option_n = sum(len(tok(x, add_special_tokens=False)['input_ids']) + 1 for x in question['options'])
        if state_n > 256 or state_n + q_n + option_n + 16 > 512:
            return None, 'semantic_check_context_limit'
        result = self.model.decide(current, [question])
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
            return None, 'invalid_semantic_decision'
        answer = result[0]
        confidence = answer.get('confidence')
        if (answer.get('choice') not in question['options'] or type(confidence) not in (int, float)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            return None, 'invalid_semantic_decision'
        # A fixed experimental threshold, not a calibrated safety guarantee.
        if answer['choice'] != question['options'][0] or confidence < .55:
            return {'intent':intent, 'coverage':answer}, 'semantic_coverage_unconfirmed'
        return {'intent':intent, 'coverage':answer}, None

    def select(self, request, *, prediction=None, coverage_decision=None):
        started = time.perf_counter()
        answers = _base(request)
        reasons = []
        proposal = None
        native = None
        current = canonicalize(request.state.current_request)
        context_error = None
        try:
            current = resolve_context(request)
        except ValueError as exc:
            context_error = str(exc)
        # Every in-budget request reaches the semantic model, including requests
        # subsequently rejected for dates or unsupported capabilities.
        if prediction is None:
            state = 'Current request: ' + current + '\nRecent user requests:\n'
            if len(self.model.tok(state, add_special_tokens=False)['input_ids']) > 256:
                predicted, margins = {}, {}
                reasons.append('state_token_limit')
            else:
                predicted, margins = self.predict(current)
        else:
            predicted, margins = prediction
        if context_error:
            reasons.append(context_error)
        if any(c.isalpha() and not c.isascii() for c in current):
            reasons.append('language_outside_evaluated_contract')
        period, date_spans, date_error = temporal(current, request.state.reference_date)
        if date_error:
            reasons.append(date_error)
        entity_spans = entities(current, request.available_metrics)
        if any(k == 'unavailable' for _, _, values in entity_spans for k, _ in values):
            reasons.append('requested_metric_unavailable')
        if any(k == 'ambiguous' for _, _, values in entity_spans for k, _ in values):
            reasons.append('ambiguous_metric_alias')
        metrics = sorted({v for _, _, values in entity_spans for k, v in values if k == 'metric'})
        records = sorted({v for _, _, values in entity_spans for k, v in values if k == 'record'})
        bound_constraints = constraints(current, entity_spans, date_spans)
        reasons.extend(c.kind for c in bound_constraints if not c.supported)
        subject = erase(current, date_spans)
        if 'profile' in records and re.search(r'\b(?:whole|complete|full) (?:health )?profile\b', subject):
            reasons.append('legacy_profile_projection_incomplete')
        # Profile projections are not expressible by this legacy consumer.
        # 'weight' is also a catalogue metric; only an unbound weight names the profile field.
        if any(word != 'weight' or 'weight' not in metrics for word in
               re.findall(r'\b(?:goals|medications|meds|allergies|birth year|weight|conditions)\b', subject)):
            if not re.search(r'\b(?:whole|complete|full) (?:health )?profile\b', subject):
                reasons.append('narrow_or_unavailable_profile_field')
        if predicted and period is not None:
            task = predicted['task']
            coverage = 'targeted' if metrics or records else predicted['coverage']
            # The dedicated read-intent head and full-proposal check arbitrate
            # bound reads; the older coarse task head is not an additional veto.
            if (metrics or records) and not research_requested(subject):
                task = 'health'
            broad = bool(re.search(r'\b(?:all (?:of )?(?:my )?health data|overall health|health (?:summary|overview))\b', subject))
            if broad and not metrics and not records:
                task = 'health'; coverage = 'broad'
            # Exact operator wording owns explicit latest/trend intent. Learned
            # coverage check must confirm a default trend when no operator is explicit.
            latest, trend, _ = operation_signals(subject)
            if latest and trend:
                reasons.append('conflicting_latest_and_trend')
            operation = 'latest' if latest else 'trend'
            research = 'none'
            # Retain the finite topic binder, never substitute a guessed topic.
            research_decision = _interpret(current, [], request)
            if research_decision and research_decision['task'] == 'research':
                task = 'research'; coverage = 'none'; operation = 'unsupported'
                research = research_decision['research']; metrics = []; records = []
                bound_constraints = []; reasons = [r for r in reasons if r not in ('unsupported_operation',)]
            elif task == 'research':
                reasons.append('unbound_research_topic')
            elif task != 'health':
                reasons.append('no_supported_read')
            elif not metrics and not records:
                if coverage != 'broad':
                    reasons.append('unresolved_metric_or_record')
                else:
                    records = list(request.available_record_types)
                    descriptive = bool(re.search(r'\b(?:summary|summarize|summarise|snapshot|descriptive|rundown)\b', subject))
                    actionable = bool(re.search(r'\b(?:analyze|analyse|assess|evaluate|improve|improvements)\b', subject))
                    if descriptive and actionable:
                        reasons.append('mixed_summary_and_analysis')
                    research = 'none' if descriptive or not request.literature_available else predicted['research']
                    if research not in ('none', 'broad_overview'):
                        reasons.append('unbound_research_topic')
            # A named target cannot absorb a simultaneous broad health request.
            elif re.search(r'\b(?:overall health|all (?:my )?health|health summary|analyze my health|analyse my health)\b', subject):
                reasons.append('mixed_broad_and_targeted_scope')
            if task == 'health' and coverage == 'targeted' and research_requested(subject):
                reasons.append('targeted_research_binding_unavailable')
            basis = 'current_plans'
            if 'calendar' in records and coverage != 'broad':
                if re.search(r'\b(?:completed|done|did .* complete)\b', subject): basis = 'last_done_date'
                elif (re.search(r'\b(?:due|planned|coming up)\b', subject) or period.get('calendar_offset') == 'next'): basis = 'next_due_date'
                elif coverage != 'broad' and period['period_kind'] not in ('unstated', 'all_history'):
                    reasons.append('unresolved_calendar_basis')
            if set(records) - set(request.available_record_types):
                reasons.append('requested_record_category_unavailable')
            if coverage == 'broad' and set(request.available_record_types) != {'profile','workouts','labs','calendar'}:
                # The legacy broad flag would re-enable absent categories in an old consumer.
                reasons.append('broad_record_inventory_requires_new_contract')
            if margins['task'] < .10 and not (metrics or records or broad):
                reasons.append('uncertain_model_task')
            proposal = ReadProposal(task=task, coverage=coverage, operation=operation,
                                    metrics=metrics, records=records, period=period,
                                    calendar_basis=basis, research=research, constraints=bound_constraints)
            if records == ['profile'] and period['period_kind'] not in ('unstated','all_history'):
                reasons.append('profile_history_unavailable')
            # A complete grammar parse is a deterministic certificate, not a
            # vocabulary gate. Unmatched prose still takes the learned path.
            certified = False
            if research_decision:
                expected = ReadProposal(task=research_decision['task'], coverage=research_decision['coverage'],
                    operation=research_decision['purpose'], metrics=sorted(set(research_decision['metrics'])),
                    records=sorted(set(research_decision['records'])),
                    period={key:research_decision[key] for key in period},
                    calendar_basis=research_decision['calendar_basis'], research=research_decision['research'])
                actual = proposal.model_copy(update={'records':sorted(set(proposal.records))})
                # A grammar that accepts "what is ApoB" cannot independently
                # certify personal-read intent when the coarse head disagrees.
                certified = expected == actual and predicted['task'] == task
            if not reasons:
                if certified:
                    # A grammar certificate binds targets and dates; it does
                    # not establish that a metric mention requests personal IO.
                    intent, error = self.intent_check(current) if task == 'health' else (None, None)
                    native = {'method':'complete_grammar_binding', 'intent':intent}
                elif coverage_decision is None:
                    native, error = self.coverage_check(current, proposal)
                else:
                    native, error = coverage_decision
                if error:
                    reasons.append(error)
            if not reasons:
                for key, value in dict(task=task, coverage=coverage, purpose=operation,
                                       calendar_basis=basis, research=research, **period).items():
                    answers[key] = {'choice':value}
                for metric in metrics:
                    answers['metric__' + metric] = {'choice':'include'}
                for record in records:
                    answers[record] = {'noul':1.0}
        return {'schema_version':SCHEMA, 'status':'unsupported' if reasons else 'selected',
                'reason':reasons[0] if reasons else None, 'reason_codes':list(dict.fromkeys(reasons)),
                'answers':answers, 'advisory':True, 'selector_sha256':identity(),
                'model_revision':MODEL_REVISION, 'adapter_sha256':ADAPTER_SHA256,
                'implementation':'open_jev_structured_proposal_experimental',
                'diagnostics':{'confidence_policy':'proposal_v4', 'predicted_decisions':predicted,
                               'decision_margins':margins, 'semantic_coverage':native,
                               'proposal':proposal.model_dump() if proposal else None},
                'elapsed_ms':round((time.perf_counter() - started) * 1000, 3)}
