"""Canonical20 through the current native Vita manager, using fictional data.

Default --dry-run validates the installed schema and frozen source oracles with
zero inference/network calls. --live explicitly enables bounded paid DeepSeek
inference via Vita's existing attested provider, and requires a loopback selector.
Every attempt gets a new output directory; accepted answers are not correctness.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
VITA_DEFAULT = Path('/Users/alexdobrin/Documents/vita-agent-deepseek')
EXPECTED_VITA_HEAD = 'e98298039153a24f9a1649e28a9fa93341ca0592'


def plain(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if is_dataclass(value):
        return {k: plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'value'):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return type(value).__name__


def save(path, value):
    # Refuse reuse instead of replacing a failed attempt or frozen oracle.
    with path.open('x') as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from walk(child)


def source_checks(catalog, expected):
    """Check observable acquisition only; never infer answer correctness."""
    values = list(walk(catalog))
    metrics = {v.get('metric') for v in values if isinstance(v.get('metric'), str)}
    metrics |= {v.get('concept') for v in values if isinstance(v.get('concept'), str)}
    fields = {key for value in values for key in value}
    source_names = {v.get('source') for v in values if isinstance(v.get('source'), str)}
    record_types = {v.get('record_type') for v in values if isinstance(v.get('record_type'), str)}
    checks = {}
    for metric in expected.get('metrics', []):
        checks['acquired_metric_' + metric] = metric in metrics
    for field in expected.get('profile_fields', []):
        checks['acquired_profile_' + field] = field in fields
    for source in expected.get('sources', []):
        checks['acquired_source_' + source] = source in source_names
    for record in expected.get('records', []):
        checks['acquired_records_' + record] = record in record_types or any(
            record in v.get('record_types', []) for v in values if isinstance(v.get('record_types'), list))
    if 'count' in expected:
        complete = [v['session_count'] for v in values if v.get('session_count_complete') is True and 'session_count' in v]
        checks['complete_count_equals_oracle'] = expected['count'] in complete
    if expected.get('research'):
        checks['acquired_research'] = any(v.get('kind') == 'literature_read' for v in values)
    statuses = [v.get('status') for v in values if isinstance(v.get('status'), str)]
    gaps = [plain(v['gaps']) for v in values if v.get('gaps')]
    incomplete = any(v.get('scope_complete') is False for v in values)
    blocked = bool(set(statuses) & {'unavailable', 'denied', 'invalid_arguments', 'blocked_dependency', 'budget_refused'})
    return {'checks': checks, 'all_observable_checks_pass': bool(checks) and all(checks.values()) and not gaps and not incomplete and not blocked,
            'observed_source_statuses': sorted(set(statuses)), 'observed_gaps': gaps,
            'incomplete_scope_observed': incomplete, 'blocked_source_observed': blocked,
            'answer_correctness': 'not_independently_reviewed'}


async def dry_run(prompts, dest, args):
    from scripts.vita_native_battery_fixture import NOW, EXPECTED, make_manager, open_health, METRICS
    from vita_agent.health.query_tool import build_health_query_executor, HealthOperationPublication
    from vita_agent.kernel.source_batch_contracts import SourceBatchRequest
    import jsonschema
    rows = []
    for case in prompts:
        manager, _, turn, _, _ = await make_manager(case['prompt'], max_output_tokens=args.max_output_tokens)
        try:
            definitions = manager.state()['tools']
            schema = next(t['parameters'] for t in definitions if t['name'] == 'acquire_sources')
            jsonschema.Draft202012Validator.check_schema(schema)
            rows.append({'id': case['id'], 'native_acquire_sources_schema_valid': True,
                'schema_sha256': hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                'final_tool_present': any(t['name'] == 'submit_final_answer' for t in definitions)})
        finally:
            await turn.aclose()
    resolver, turn, _ = await open_health()
    oracles = []
    try:
        executor = build_health_query_executor(resolver, authorized_metrics=METRICS, coverage=None, clock=lambda: NOW)
        cases = [
            ('workouts_current_month', {'record_types': ['workouts'], 'range_mode': 'current_month', 'purpose': 'trend'}, 3),
            ('workouts_explicit_window', {'record_types': ['workouts'], 'range_mode': 'between', 'start_at': '2026-05-17', 'end_at': '2026-08-17', 'purpose': 'trend'}, 3),
            ('workouts_all_history', {'record_types': ['workouts'], 'range_mode': 'all', 'purpose': 'trend'}, 8),
        ]
        for name, query, expected in cases:
            batch = SourceBatchRequest.model_validate({'health_reads': [{'operation_id': 1, **query}], 'required_operation_ids': [1]})
            result = await executor.execute(request=batch.health_reads[0], publication=HealthOperationPublication())
            counts = [v['session_count'] for v in walk(result) if v.get('session_count_complete') is True and 'session_count' in v]
            passed = expected in counts
            oracles.append({'id': name, 'expected_complete_count': expected, 'observed_complete_counts': counts, 'passed': passed})
            save(dest / (name + '.json'), result)
    finally:
        await turn.aclose()
    for name, options, query, expectation in (
        ('empty_is_not_withheld', {}, {'range_mode': 'between', 'start_at': '2020-01-01', 'end_at': '2020-12-31'}, 'zero_complete'),
        ('partial_scan_has_no_exact_count', {'exhausted': False}, {'range_mode': 'all'}, 'incomplete'),
        ('withheld_domain_never_reads_backend', {'withheld': ('workout_sessions',)}, {'range_mode': 'all'}, 'withheld'),
    ):
        resolver, turn, calls = await open_health(**options)
        try:
            executor = build_health_query_executor(resolver, authorized_metrics=METRICS, coverage=None, clock=lambda: NOW)
            operation = SourceBatchRequest.model_validate({'health_reads': [{'operation_id': 1,
                'record_types': ['workouts'], 'purpose': 'trend', **query}], 'required_operation_ids': [1]}).health_reads[0]
            result = await executor.execute(request=operation, publication=HealthOperationPublication())
            complete = [v['session_count'] for v in walk(result) if v.get('session_count_complete') is True and 'session_count' in v]
            passed = (complete == [0] if expectation == 'zero_complete' else
                      not complete and (not calls if expectation == 'withheld' else True))
            oracles.append({'id': name, 'passed': passed, 'observed_complete_counts': complete, 'backend_reads': len(calls)})
            save(dest / (name + '.json'), result)
        finally:
            await turn.aclose()
    # Independent fixed-value checks exercise the exact production resolver;
    # the oracle is not derived from the selector or from its returned plan.
    values = {'total_sleep': 450, 'sleep_score': 76, 'heart_rate_variability': 65,
              'recovery_score': 74, 'readiness_score': 82, 'apob': 78,
              'fasting_glucose': 92, 'hba1c': 5.4, 'hs_crp': 1.2}
    value_cases = [(metric, {'concepts': [metric]}, {metric: value}) for metric, value in values.items()]
    value_cases.append(('three_lab_targets', {'concepts': ['fasting_glucose', 'hba1c', 'hs_crp']},
                        {'fasting_glucose': 92, 'hba1c': 5.4, 'hs_crp': 1.2}))
    for name, query, expected in value_cases:
        resolver, turn, _ = await open_health()
        try:
            executor = build_health_query_executor(resolver, authorized_metrics=METRICS, coverage=None, clock=lambda: NOW)
            operation = SourceBatchRequest.model_validate({'health_reads': [{'operation_id': 1,
                'range_mode': 'all', 'purpose': 'latest', **query}], 'required_operation_ids': [1]}).health_reads[0]
            result = await executor.execute(request=operation, publication=HealthOperationPublication())
            found = {f['concept']: f['value'] for f in result.get('facts', [])}
            oracles.append({'id': 'latest_' + name, 'passed': all(found.get(k) == v for k, v in expected.items()),
                            'expected': expected, 'observed': found})
            save(dest / ('latest_' + name + '.json'), result)
        finally:
            await turn.aclose()
    result = {'mode': 'dry_run', 'schema_cases': rows, 'source_oracles': oracles,
              'passed': all(r['final_tool_present'] for r in rows) and all(r['passed'] for r in oracles),
              'inference_calls': 0, 'selector_calls': 0, 'answer_correctness': 'not_evaluated'}
    save(dest / 'dry-run.json', result)
    return result


class LoopbackSelector:
    """Test-only local transport, not an attested enclave client."""
    def __init__(self, base_url, token):
        from query_selector import identity, INTENT_SHA256
        self.base_url, self.token = base_url.rstrip('/'), token
        self.selector_sha256, self.adapter_sha256 = identity(), INTENT_SHA256

    def select(self, request):
        import httpx
        from query_plan import QueryPlan, compile_batch
        from routing import MODEL_REVISION
        response = httpx.post(self.base_url + '/v1/select', json=request.model_dump(mode='json'),
            headers={'Authorization': 'Bearer ' + self.token} if self.token else {}, timeout=15)
        response.raise_for_status()
        raw = response.json()
        if raw.get('schema_version') != 'vita-query-plan/v2' or raw.get('advisory') is not True:
            raise ValueError('selector_contract_mismatch')
        result = QueryPlan.model_validate(raw)
        if (result.selector_sha256 != self.selector_sha256 or result.adapter_sha256 != self.adapter_sha256
                or result.model_revision != MODEL_REVISION):
            raise ValueError('selector_identity_mismatch')
        compile_batch(result, request)
        return result


class InferenceBudget:
    """Fail before the provider call; no silent retry or limit extension."""
    def __init__(self):
        self.calls = 0
        self.input_bytes = 0
        self.exhausted = False

    def admit(self, envelope, maximum_output):
        size = len(json.dumps(envelope, ensure_ascii=False, separators=(',', ':')).encode())
        controls = envelope.get('controls', {}).get('settings', {})
        output = controls.get('max_tokens', envelope.get('request', {}).get('maxTokens'))
        if self.calls >= 80 or self.input_bytes + size > 3_000_000 or type(output) is not int or not 1 <= output <= maximum_output <= 2048:
            self.exhausted = True
            raise RuntimeError('synthetic_inference_budget_exceeded')
        self.calls += 1
        self.input_bytes += size


async def live_run(prompts, dest, args):
    from backbone.local_config import load_inference_config
    from backbone.provider import VitaProvider
    from backbone.admitted_context import AdmittedHistory, HistoryRecord
    from vita_agent.models.provider import resolve_effective_model_target
    from examples.vita_native import NativeSelectorProvider
    from query_plan import QueryRequest
    from scripts.vita_native_battery_fixture import NOW, USER_ID, METRICS, SOURCES, EXPECTED, make_manager
    load_inference_config(args.inference_config)
    target = await resolve_effective_model_target()
    if target.model != 'deepseek-v4-1-flash' or target.config.provider != 'tinfoil':
        raise ValueError('unexpected_inference_target')
    client = LoopbackSelector(args.selector_base_url, os.environ.get(args.selector_token_env, ''))
    conversation = str(uuid4())
    history_records, user_history, rows = [], [], []
    budget = InferenceBudget()
    for case in prompts:
        started = time.monotonic()
        manager, policy, turn, calls, boundary = await make_manager(case['prompt'],
            model_id=target.model, max_rounds=args.max_rounds, max_output_tokens=args.max_output_tokens)
        request = QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
            'state': {'current_request': case['prompt'], 'recent_user_requests': user_history[-4:],
                      'reference_date': NOW.date().isoformat(), 'time_zone': 'UTC'},
            'reference_time': NOW, 'available_metrics': list(METRICS), 'available_sources': list(SOURCES),
            'available_record_types': ['profile', 'workouts', 'labs', 'calendar'], 'literature_available': True})
        async def resolve():
            return target
        async def authorize(_target, _body):
            await manager.check_disclosure()
        native_provider = VitaProvider(resolve=resolve, authorize=authorize)
        async def provider(envelope):
            budget.admit(envelope, args.max_output_tokens)
            stream = native_provider(envelope)
            try:
                async for event in stream:
                    yield event
            finally:
                await stream.aclose()
        wrapped = NativeSelectorProvider(client, request, provider, authorize=manager.check_disclosure)
        history = AdmittedHistory(USER_ID, conversation, policy[0].consent_epoch,
            policy[0].key_epoch, tuple(history_records)) if history_records else None
        row = {'id': case['id'], 'question': case['prompt'], 'terminal_status': 'failed',
               'expected': EXPECTED[case['id']], 'answer_correctness': 'not_independently_reviewed'}
        try:
            result = await asyncio.wait_for(manager.run(case['prompt'], model=wrapped, history=history,
                conversation_id=conversation, timeout=args.timeout), timeout=args.timeout + 10)
            row.update(terminal_status='accepted', answer=result['answer'])
            history_records.extend((HistoryRecord(conversation, str(uuid4()), 'user', case['prompt']),
                HistoryRecord(conversation, str(uuid4()), 'assistant', result['answer'])))
        except Exception as exc:
            # Provider exception strings can contain request bodies/credentials.
            row['error_type'] = type(exc).__name__
            from backbone.host import SdkRunError
            if isinstance(exc, SdkRunError) and len(str(exc)) < 100 and all(c.isalnum() or c == '_' for c in str(exc)):
                row['safe_error_code'] = str(exc)
            # A failed turn adds the actual user request only, no invented answer.
            history_records.append(HistoryRecord(conversation, str(uuid4()), 'user', case['prompt']))
        finally:
            state = getattr(manager.context, 'source_batch_state', None)
            catalog = plain(getattr(state, 'results_by_source_id', {}))
            row.update(seconds=round(time.monotonic() - started, 3), rounds=manager.rounds,
                storage_reads=len(calls), research_calls=len(boundary.calls),
                requested_tools=manager.requested_tools, executed_tools=manager.executed_tools,
                total_tokens=manager.wrapper.usage.total_tokens, source_catalog=catalog,
                acquisition_audit=source_checks(catalog, EXPECTED[case['id']]))
            if hasattr(wrapped, 'receipt'):
                row['selector_receipt'] = plain(wrapped.receipt)
            row['cumulative_inference_calls'] = budget.calls
            row['cumulative_input_bytes'] = budget.input_bytes
            await turn.aclose()
        save(dest / f"turn-{case['id']:02d}.json", row)
        rows.append(row)
        user_history.append(case['prompt'])
        print(json.dumps({'id': row['id'], 'terminal_status': row['terminal_status'],
                          'seconds': row['seconds'], 'rounds': row['rounds']}), flush=True)
        if budget.exhausted:
            break
    summary = {'mode': 'live_synthetic_native_manager', 'question_count': len(rows),
        'accepted_terminal_count': sum(r['terminal_status'] == 'accepted' for r in rows),
        'observable_acquisition_checks_passed': sum(r['acquisition_audit']['all_observable_checks_pass'] for r in rows),
        'answer_correctness': 'not_independently_reviewed', 'browser_acceptance': False,
        'selector_transport': 'loopback_test_only_not_attested',
        'research_provider': 'fictional_local_no_external_research', 'rows': rows,
        'provider_calls': budget.calls, 'cumulative_input_bytes': budget.input_bytes,
        'budget_exhausted': budget.exhausted}
    save(dest / 'live-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--vita-source', type=Path, default=VITA_DEFAULT)
    parser.add_argument('--prompts', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'evidence/selector-native-battery')
    parser.add_argument('--max-questions', '--max-cases', type=int, default=20, choices=range(1, 21))
    parser.add_argument('--max-rounds', type=int, default=4, choices=range(1, 5))
    parser.add_argument('--max-output-tokens', type=int, default=1400, choices=range(128, 2049))
    parser.add_argument('--timeout', type=int, default=90, choices=range(10, 181))
    parser.add_argument('--inference-config', type=Path)
    parser.add_argument('--selector-base-url')
    parser.add_argument('--selector-token-env', default='OPEN_JEV_API_KEY')
    args = parser.parse_args()
    vita = args.vita_source.resolve()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=vita, text=True).strip()
    if head != EXPECTED_VITA_HEAD:
        parser.error('Vita source revision differs from the reviewed battery target')
    if args.live:
        url = urlparse(args.selector_base_url or '')
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1') or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment:
            parser.error('--live requires an http loopback --selector-base-url')
        if args.inference_config is None:
            parser.error('--live requires explicit --inference-config; no config is read during dry-run')
    sys.path[:0] = [str(ROOT), str(ROOT / 'vendor'), str(vita), str(vita / 'vita/py/src')]
    from vita_agent.kernel.source_batch_contracts import SourceBatchRequest
    from scripts.vita_native_battery_fixture import NOW, EXPECTED, records
    if not Path(inspect.getfile(SourceBatchRequest)).resolve().is_relative_to(vita):
        raise RuntimeError('wrong_vita_import_root')
    prompt_path = args.prompts or vita / 'artifacts/backbone/browser-battery20/canonical-prompts.json'
    raw = prompt_path.read_bytes()
    all_prompts = json.loads(raw)['prompts']
    if len(all_prompts) != 20 or [r['id'] for r in all_prompts] != list(range(1, 21)):
        raise ValueError('canonical_twenty_prompts_required')
    prompts = all_prompts[:args.max_questions]
    dest = args.output_root / (time.strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8])
    dest.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / 'scripts/vita_native_battery_fixture.py',
               vita / 'backbone/manager.py', vita / 'backbone/domain.py',
               vita / 'backbone/provider.py', vita / 'backbone/synthetic.py',
               vita / 'backbone/test_research.py', ROOT / 'examples/vita_native.py',
               vita / 'vita/py/src/vita_agent/kernel/source_batch_contracts.py',
               vita / 'vita/py/src/vita_agent/health/resolver.py',
               vita / 'vita/py/src/vita_agent/health/query_tool.py',
               vita / 'vita/py/src/vita_agent/health/turn_materialization.py']
    manifest = {'mode': 'live' if args.live else 'dry_run', 'vita_source': str(vita),
        'vita_commit': head, 'synthetic_only': True, 'reference_time': NOW.isoformat(),
        'prompts_sha256': hashlib.sha256(raw).hexdigest(), 'max_questions': args.max_questions,
        'max_rounds': args.max_rounds, 'max_output_tokens_per_round': args.max_output_tokens,
        'maximum_generated_token_allowance': args.max_questions * args.max_rounds * args.max_output_tokens,
        'max_provider_calls': 80, 'max_cumulative_input_bytes': 3_000_000,
        'timeout_per_turn_seconds': args.timeout, 'paid_inference_enabled': args.live,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    save(dest / 'manifest.json', manifest)
    save(dest / 'synthetic-fixture.json', {'records': records(), 'expected': EXPECTED})
    save(dest / 'prompts.json', {'prompts': prompts})
    print(json.dumps({'attempt_directory': str(dest), 'mode': manifest['mode']}), flush=True)
    result = asyncio.run(live_run(prompts, dest, args) if args.live else dry_run(prompts, dest, args))
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'schema_cases', 'source_oracles')}), flush=True)
    if result.get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
