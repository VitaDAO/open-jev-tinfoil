"""Exact compiled-plan evaluation. Does not execute records or call a provider."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor'), str(ROOT / 'tests')]
from contracts.vita_read_contract import read_arguments
from selector import SelectorRequest, select

INVENTORY = ['total_sleep', 'sleep_efficiency', 'steps', 'apob', 'ldl_cholesterol',
             'oxygen_saturation', 'respiratory_rate', 'custom_metric']
RECORDS = ('profile', 'workouts', 'labs', 'calendar')
NOW = datetime(2026, 9, 23, 12, tzinfo=UTC)


def canonical(plan):
    return json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False)


def grade(result, plan, gold, error=None):
    if error:
        return 'harness_error'
    if result['status'] == 'unsupported':
        if plan is not None:
            return 'invalid_response'
        return 'correct_handoff' if gold['handoff_ok'] else 'missed_plan'
    if result['status'] != 'selected' or plan is None:
        return 'invalid_response'
    if canonical(plan) in {canonical(p) for p in gold['acceptable_plans']}:
        return 'correct_plan'
    return 'wrong_plan'


def summary(rows, gold_key='gold', consensus_only=False):
    outcomes = {}
    considered = [r for r in rows if not consensus_only or r['label_agreement']]
    for row in considered:
        outcome = row['grades'][gold_key]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {'cases': len(considered), 'correct': outcomes.get('correct_plan', 0)
            + outcomes.get('correct_handoff', 0), 'outcomes': outcomes,
            'required_plan_cases': sum(not r['expected'][gold_key]['handoff_ok'] for r in considered),
            'executable_plans': sum(r['plan'] is not None for r in considered),
            'excluded_label_disagreements': len(rows) - len(considered)}


def evaluate(cases, call):
    rows = []
    for case in cases:
        request = SelectorRequest.model_validate({
            'schema_version': 'vita-selector/v1', 'available_metrics': INVENTORY,
            'literature_available': True, 'state': {'current_request': case['request'],
                'recent_user_requests': case['history'], 'reference_date': '2026-09-23', 'time_zone': 'UTC'}})
        started = time.perf_counter()
        result, plan, error = {}, None, None
        try:
            result = call(request)
            plan = read_arguments(result, INVENTORY, literature_available=True,
                                  record_types=RECORDS, now=NOW, time_zone='UTC')
        except Exception as exc:
            # An exception is never a successful abstention. Synthetic fixture only.
            error = {'type': type(exc).__name__, 'message': str(exc)}
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        expected = {k: case[k] for k in ('gold', 'reviewer_gold') if k in case}
        rows.append({'id': case['id'], 'label_agreement': case.get('label_agreement', True),
                     'status': result.get('status'), 'reason_codes': result.get('reason_codes'),
                     'diagnostics': result.get('diagnostics'),
                     'model_evaluated': bool(result.get('diagnostics', {}).get('predicted_decisions')),
                     'plan': plan, 'error': error, 'elapsed_ms': elapsed, 'expected': expected,
                     'grades': {k: grade(result, plan, g, error) for k, g in expected.items()}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('trained', 'proposal', 'ridge', 'grammar'), default='trained')
    parser.add_argument('--quiet', action='store_true', help='Freeze predictions without revealing scores')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    source_paths = ('trained_proposal_selector.py', 'adapters/vita-read-intent-v1.json', 'proposal_selector.py', 'proposal_binding.py', 'temporal_spans.py',
                    'metadata/health_metrics.v1.json', 'metadata/selector-index.v1.json',
                    'schema_index.py', 'learned_selector.py', 'selector.py',
                    'scripts/evaluate_selector_exact.py', 'tests/contracts/vita_read_contract.py',
                    'tests/contracts/jev_dates.py')
    hashes = lambda: {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    source_hashes = hashes()
    started = time.perf_counter()
    call = select
    if args.model in ('ridge', 'proposal', 'trained'):
        import torch
        from typed_decisions.open_jev import OpenJev
        from learned_selector import LearnedSelector
        from proposal_selector import ProposalSelector
        from trained_proposal_selector import TrainedProposalSelector
        torch.set_num_threads(4)
        model = OpenJev.from_pretrained(str(ROOT / 'model-fp16'), device='cpu')
        model.collator._ids = lambda t: model.tok(t, add_special_tokens=False)['input_ids']
        model.collator._cache.clear()
        call = {'trained':TrainedProposalSelector, 'proposal':ProposalSelector, 'ridge':LearnedSelector}[args.model](model).select
    load_ms = round((time.perf_counter() - started) * 1000, 3)
    rows = evaluate(cases, call)
    if hashes() != source_hashes:
        raise RuntimeError('Source changed during evaluation; discard this run')
    times = sorted(r['elapsed_ms'] for r in rows[1:])
    model_times = sorted(r['elapsed_ms'] for r in rows[1:] if r['model_evaluated'])
    result = {'candidate': args.model, 'scope': 'Local 4-thread CPU; compiled legacy plans; no DB, WAN or attestation',
              'fixture_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              'source_sha256': source_hashes,
              'load_ms': load_ms, 'warm_p50_ms': statistics.median(times),
              'warm_p95_ms': times[int(.95 * (len(times) - 1))],
              'warm_model_requests': len(model_times),
              'warm_model_p50_ms': statistics.median(model_times) if model_times else None,
              'warm_model_p95_ms': model_times[int(.95 * (len(model_times) - 1))] if model_times else None,
              'all_author_labels': summary(rows), 'consensus': summary(rows, consensus_only=True), 'rows': rows}
    if all('reviewer_gold' in c for c in cases):
        result['all_reviewer_labels'] = summary(rows, 'reviewer_gold')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Saved predictions:', len(rows)) if args.quiet else print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
