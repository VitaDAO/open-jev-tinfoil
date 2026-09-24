"""Compare compact Laya semantic questions and option-order sensitivity locally."""
import json
import os
import statistics
import time
from pathlib import Path

import laya_mlx
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / 'evidence/selector-v1/model-heldout.json').read_text())
MODEL_COMMIT = '20aed815fc6acde75733882e7ec0e3f28aeb9717'
MODEL_ID = os.environ.get('LAYA_MODEL_ID', 'aac6fef/laya-mlx')
REVISION = os.environ.get('LAYA_REVISION', MODEL_COMMIT)
OPTIONS = {
    'task': {'health': 'read personal health information',
             'research': 'find published research without personal data',
             'other': 'no personal health read or published research'},
    'coverage': {'broad': 'overall health overview',
                 'targeted': 'specific measurement or record',
                 'none': 'no personal health information'},
    'purpose': {'latest': 'latest individual observation',
                'trend': 'trend or summary across time',
                'unsupported': 'comparison, edit, or another complex operation'},
    'research': {'broad_overview': 'general research for broad health analysis',
                 'sleep_activity': 'sleep and physical activity research',
                 'cardiometabolic': 'lipid and glucose research',
                 'none': 'no published research',
                 'unsupported': 'research on another topic'},
}
INSTRUCTIONS = {
    'task': 'Which task is requested?',
    'coverage': 'How broad is the requested personal health read?',
    'purpose': 'Which kind of personal measurement read is requested?',
    'research': 'What published research is requested, if any?',
}


def questionnaire(reverse=False):
    q = {}
    maps = {}
    for key, named in OPTIONS.items():
        choices = list(named.values())
        if reverse:
            choices.reverse()
        q[key] = {'type': 'choice', 'instructions': INSTRUCTIONS[key], 'criteria': choices}
        maps[key] = {v: k for k, v in named.items()}
    return q, maps


def main():
    model_dir = snapshot_download(MODEL_ID, revision=REVISION)
    agent = laya_mlx.load(model_dir, device='gpu')
    rows = []
    for case in CASES:
        if case['expected'] is None:
            continue
        for state_form in ('text', 'structured'):
            for reversed_options in (False, True):
                questions, maps = questionnaire(reversed_options)
                state = (case['request'] if state_form == 'text' else
                         {'current_request': case['request'],
                          'recent_user_requests': case['history']})
                started = time.perf_counter()
                raw = agent.predict(state, questions)
                elapsed = (time.perf_counter() - started) * 1000
                selected = {key: maps[key][item['choice']]
                            for key, item in raw['answers'].items()}
                expected = case['expected']
                comparable = {key: expected[key] for key in OPTIONS if key in expected}
                rows.append({'id': case['id'], 'state_form': state_form,
                             'reversed_options': reversed_options,
                             'selected': selected, 'expected': comparable,
                             'joint_correct': all(selected[key] == value for key, value in comparable.items()),
                             'elapsed_ms': round(elapsed, 2),
                             'input_tokens': raw['usage']['input_tokens']})
    reports = []
    for state_form in ('text', 'structured'):
        for reversed_options in (False, True):
            subset = [row for row in rows if row['state_form'] == state_form and
                      row['reversed_options'] == reversed_options]
            reports.append({'state_form': state_form, 'reversed_options': reversed_options,
                            'n': len(subset), 'joint_correct': sum(r['joint_correct'] for r in subset),
                            'by_head': {key: sum(r['selected'][key] == r['expected'][key]
                                                 for r in subset if key in r['expected'])
                                        for key in OPTIONS},
                            'median_ms': round(statistics.median(r['elapsed_ms'] for r in subset), 2)})
    changed = {}
    for state_form in ('text', 'structured'):
        ids = {r['id'] for r in rows}
        changed[state_form] = [id for id in ids if any(
            a['selected'][key] != b['selected'][key]
            for key in OPTIONS
            for a in rows for b in rows
            if a['id'] == b['id'] == id and a['state_form'] == b['state_form'] == state_form
            and a['reversed_options'] is False and b['reversed_options'] is True)]
    out = {'model': MODEL_ID, 'model_commit': REVISION,
           'questionnaire': OPTIONS, 'reports': reports,
           'option_order_changed_case_ids': changed, 'rows': rows}
    name = ('laya-typed-question-ablation.json' if MODEL_ID.endswith('typed-decisions-mlx')
            else 'laya-question-ablation.json')
    (ROOT / 'evidence/selector-v1' / name).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'reports': reports, 'option_order_changes': {k: len(v) for k, v in changed.items()}}, indent=2))


if __name__ == '__main__':
    main()
