"""Local-only diagnostic of Laya MLX on Vita's synthetic typed questionnaire.

Run with a separate macOS laya-mlx environment. Does not touch Vita, Tinfoil,
personal records, or any live service. The questionnaire fixture was generated
from read-only Vita local_jev.request_body for synthetic 'Analyze me'.
"""
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from contracts.vita_read_contract import read_arguments

import laya_mlx

INVENTORY = ['total_sleep', 'sleep_efficiency', 'steps', 'apob', 'ldl_cholesterol',
             'oxygen_saturation', 'respiratory_rate', 'custom_metric']
CASES = ROOT / 'evidence/selector-v1/model-heldout.json'
TEMPLATE = ROOT / 'evidence/selector-v1/vita-questionnaire-synthetic.json'
MODEL_COMMIT = '20aed815fc6acde75733882e7ec0e3f28aeb9717'
NOW = datetime(2026, 9, 23, 12, tzinfo=UTC)


def questions(template):
    allowed = {k for k in template if not k.startswith('metric__')}
    allowed.update('metric__' + metric for metric in INVENTORY)
    filtered = {k: v for k, v in template.items() if k in allowed}
    if 'metric__custom_metric' not in filtered:
        filtered['metric__custom_metric'] = {'type': 'choice',
            'instructions': 'Is custom metric among the measurements specifically requested?',
            'criteria': {'include': 'The user asks for custom metric.',
                         'skip': 'The request is about something else.'}}
    return filtered


def adapt_questions(template):
    outgoing = {}
    reverse = {}
    for key, spec in template.items():
        if spec['type'] == 'noul':
            outgoing[key] = {'type': 'noul', 'instructions': spec['instructions']}
            continue
        criteria = spec['criteria']
        if len(set(criteria.values())) != len(criteria):
            raise ValueError('Ambiguous criterion descriptions')
        outgoing[key] = {'type': spec['type'], 'instructions': spec['instructions'],
                         'criteria': list(criteria.values())}
        reverse[key] = {v: k for k, v in criteria.items()}
    return outgoing, reverse


def adapt_answers(raw, reverse):
    answers = {}
    for key, item in raw['answers'].items():
        if key in reverse:
            answers[key] = {'choice': reverse[key][item['choice']]}
        else:
            answers[key] = {'noul': item['noul']}
    return {'answers': answers}


def main():
    template = json.loads(TEMPLATE.read_text())
    q, reverse = adapt_questions(questions(template['questions']))
    cases = json.loads(CASES.read_text())
    model_dir = snapshot_download('aac6fef/laya-mlx', revision=MODEL_COMMIT)
    started = time.perf_counter()
    agent = laya_mlx.load(model_dir, device='gpu')
    load_ms = (time.perf_counter() - started) * 1000
    rows = []
    for case in cases:
        state = {**template['state'], 'current_request': case['request'],
                 'recent_user_requests': case['history']}
        start = time.perf_counter()
        raw = agent.predict(state, q)
        prediction_ms = (time.perf_counter() - start) * 1000
        response = adapt_answers(raw, reverse)
        try:
            plan = read_arguments(response, INVENTORY, literature_available=True,
                                  record_types=('profile', 'workouts', 'labs', 'calendar'),
                                  now=NOW, time_zone='UTC')
        except (ValueError, KeyError):
            plan = None
        if case['expected'] is None:
            correct = plan is None
        else:
            correct = plan is not None and all(
                response['answers'][key]['choice'] == value
                for key, value in case['expected'].items())
            if correct and 'concepts' in case:
                correct = sorted(m for read in plan['health_reads']
                                 for m in read.get('concepts', [])) == case['concepts']
            if correct and 'range' in case:
                correct = all(read['range'] == case['range']
                              for read in plan['health_reads'] if 'concepts' in read)
        rows.append({'id': case['id'], 'correct': correct, 'expected_selection': case['expected'] is not None,
                     'selected': plan is not None, 'prediction_ms': round(prediction_ms, 2),
                     'input_tokens': raw['usage']['input_tokens'],
                     'choices': {key: value['choice'] for key, value in response['answers'].items()
                                 if key in ('task', 'coverage', 'purpose', 'research', 'period_kind', 'calendar_basis')},
                     'plan': plan})
    times = sorted(row['prediction_ms'] for row in rows[1:])
    result = {'model': 'aac6fef/laya-mlx', 'model_commit': MODEL_COMMIT,
              'package': 'laya-mlx==0.1.0', 'hardware': 'Apple M5 Max 128GB',
              'questionnaire_sha256': hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
              'fixture_sha256': hashlib.sha256(CASES.read_bytes()).hexdigest(),
              'question_count': len(q), 'model_load_ms': round(load_ms, 2),
              'cases': len(rows), 'correct': sum(r['correct'] for r in rows),
              'accepted_valid': sum(r['selected'] and r['expected_selection'] for r in rows),
              'incorrect_selected': sum(r['selected'] and not r['correct'] for r in rows),
              'fallbacks': sum(not r['selected'] for r in rows),
              'prediction_warm_p50_ms': round(statistics.median(times), 2),
              'prediction_warm_p95_ms': times[int(.95 * (len(times) - 1))],
              'rows': rows}
    out = ROOT / 'evidence/selector-v1/laya-local-diagnostic.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)
    for row in rows:
        if not row['correct']:
            print(row['id'], 'selected' if row['selected'] else 'handoff', row['choices'], flush=True)


if __name__ == '__main__':
    main()
