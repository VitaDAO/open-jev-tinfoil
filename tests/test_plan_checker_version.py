"""The plan checker is scored in the input format it was trained on (review C2, S7/G16).

The checker input is versioned (learned_parser/verify.py TEXT_VERSION; 3 = request and plan as a text pair). From version 3 the
head's file name carries the version (head_file), so code that predates a checker's format finds no head and fails loudly. Here: a
staged checker must be one this code can score, and /health attests the head whatever its version.
"""
import json
from pathlib import Path

import pytest

import query_selector
from learned_parser.verify import TEXT_VERSION, Verifier, head_file

STAGED = Path(__file__).resolve().parents[1] / 'learned_models'     # scripts/stage_learned_models.py; absent in CI


def test_staged_checkers_are_ones_this_code_can_score():
    checkers = sorted({f.parent for f in STAGED.glob('*/verifier_head*.pt')})
    if not checkers:
        pytest.skip('no plan checker staged in learned_models/')
    for d in checkers:
        v = json.loads((d / 'verifier.json').read_text())['text_version'] if (d / 'verifier.json').exists() else 1
        assert v <= TEXT_VERSION, f'{d.name} needs checker text version {v}; learned_parser/verify.py has {TEXT_VERSION}'
        assert (d / head_file(v)).is_file(), f'{d.name}: text version {v} loads {head_file(v)}, which is missing'


def test_a_checker_newer_than_this_code_is_refused(tmp_path):
    (tmp_path / 'verifier.json').write_text(json.dumps({'text_version': TEXT_VERSION + 1}))
    with pytest.raises(ValueError, match='newer than this code'):
        Verifier(tmp_path)


@pytest.mark.parametrize('version', [2, 3])
def test_identity_attests_the_checker_head_of_either_text_version(monkeypatch, tmp_path, version):
    parser, checker = tmp_path / 'm', tmp_path / 'v'
    for root, names in ((parser, ('config.json', 'model.safetensors', 'heads.pt', 'items.json', 'crisis.json', 'calibration.json', 'tokenizer.json')),
                        (checker, ('config.json', 'model.safetensors', head_file(version), 'verifier.json', 'tokenizer.json'))):
        root.mkdir()
        for n in names: (root / n).write_bytes(f'{root.name}:{n}'.encode())
    fed = []

    class Digest:
        def update(self, b): fed.append(bytes(b))
        def hexdigest(self): return 'digest'
    monkeypatch.setattr(query_selector.hashlib, 'sha256', Digest)
    monkeypatch.setenv('LEARNED_PARSER_DIR', str(parser))
    monkeypatch.setenv('LEARNED_VERIFIER_DIR', str(checker))
    query_selector.learned_parser_identity()
    # every checker file, in the order the identity has always used (a text-version-2 checker keeps its digest)
    assert [b for b in fed if b.startswith(b'v:')] == [f'v:{n}'.encode() for n in
                                                       ('config.json', 'model.safetensors', head_file(version), 'verifier.json', 'tokenizer.json')]
