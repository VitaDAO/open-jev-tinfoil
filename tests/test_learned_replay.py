"""Actual _select and link execution from recorded neural tensors."""
from pathlib import Path
import os
import pytest
from scripts.learned_replay import verify, load


@pytest.mark.parametrize('directory', sorted((Path(__file__).parent / 'fixtures' / 'learned_replay').glob('*')))
def test_real_decoder_replay(directory):
    assert verify(directory) > 0


@pytest.mark.parametrize('model', ['r8a', 'r8b', 'r13a', 'r13b'])
def test_replayed_followups_match_the_spec(model):
    directory = Path(__file__).parent / 'fixtures' / 'learned_replay' / ('jevparse_' + model)
    parser, cases = load(directory)
    outcomes = {req['state']['current_request']: parser.select(req) for req, _ in cases}
    for text, start, end in [('and lastt year', '2025-08-01', '2025-08-31'),
                             ('Between March and May, what was my SpO2?', '2026-03-01', '2026-05-31')]:
        result = outcomes[text]
        assert result['status'] == 'planned'
        assert len(result['queries']) == 1
        assert result['queries'][0]['period'] == {'kind': 'between', 'start_at': start, 'end_at': end}
    assert outcomes['and my ApoB?']['status'] == 'handoff'
    assert outcomes['show MPV too']['queries'][0]['operation'] == 'latest'
    for text, source in [('switch to August', 'withings'), ('how about Wednesday last week?', 'fitbit')]:
        result = outcomes[text]
        assert result['status'] == 'handoff' or all(q['source'] == source for q in result['queries'])


@pytest.mark.skipif(not os.environ.get('JEV_REPLAY_DIR'), reason='extended local replay corpus not selected')
def test_extended_decoder_replay():
    for directory in sorted(Path(os.environ['JEV_REPLAY_DIR']).iterdir()):
        if (directory / 'manifest.json').exists():
            assert verify(directory) >= 1000

@pytest.mark.parametrize('model', ['r8a', 'r8b', 'r13a', 'r13b'])
def test_replayed_constraints_do_not_silently_change_the_read(model):
    parser, cases = load(Path(__file__).parent / 'fixtures' / 'learned_replay' / ('jevparse_' + model))
    outcomes = {req['state']['current_request']: parser.select(req) for req, _ in cases}
    assert outcomes['Please summarize my health for this week']['status'] == 'handoff'
    assert outcomes['just that one for last month']['status'] == 'handoff'
    result = outcomes['Show LDL from 2026-02-01 onward.']
    if result['status'] == 'planned':
        assert all(q['period'] == {'kind': 'between', 'start_at': '2026-02-01', 'end_at': '2026-09-26'} for q in result['queries'])
    result = outcomes['Plot my LDL too']
    if result['status'] == 'planned':
        assert all(q['operation'] == 'trend' for q in result['queries'])
