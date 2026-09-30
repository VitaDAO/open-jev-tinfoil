"""scripts/stage_learned_models.py: exact copies the serving user can read, and a manifest sha256sum --check reads."""
import hashlib
import os
import re

import pytest

from scripts.stage_learned_models import stage


def model(source, name, files):
    for rel, data in files.items():
        (source / name / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / name / rel).write_bytes(data)


def test_stage_copies_the_models_with_a_sorted_manifest_readable_by_the_serving_user(tmp_path):
    source, out = tmp_path / 'exp', tmp_path / 'learned_models'
    model(source, 'parser_a', {'model.safetensors': b'weights', 'config.json': b'{}'})
    model(source, 'checker', {'verifier.json': b'{"text_version": 2}', 'sub/tokenizer.json': b'tok'})
    model(source, 'unchosen', {'config.json': b'{}'})
    (source / 'parser_a/model.safetensors').chmod(0o600)     # as training saves the weights
    umask = os.umask(0o077)     # nor may a strict umask reach the image
    try:
        pinned = stage(['parser_a', 'checker'], source, out)
    finally:
        os.umask(umask)
    lines = (out / 'MANIFEST.sha256').read_text().splitlines()
    assert [line[66:] for line in lines] == ['checker/sub/tokenizer.json', 'checker/verifier.json',
                                            'parser_a/config.json', 'parser_a/model.safetensors']
    for line in lines:
        assert re.fullmatch(r'[0-9a-f]{64}  \S+', line)
        digest, path = line.split('  ')
        assert (out / path).read_bytes() == (source / path).read_bytes()
        assert digest == hashlib.sha256((out / path).read_bytes()).hexdigest()
    assert pinned == hashlib.sha256((out / 'MANIFEST.sha256').read_bytes()).hexdigest()
    assert not (out / 'unchosen').exists()
    for path in [out, *out.rglob('*')]:
        assert path.stat().st_mode & 0o777 == (0o755 if path.is_dir() else 0o644), path
    assert (source / 'parser_a/model.safetensors').stat().st_mode & 0o777 == 0o600     # the source is left as it was


def test_restaging_drops_models_no_longer_chosen(tmp_path):
    source, out = tmp_path / 'exp', tmp_path / 'learned_models'
    model(source, 'old', {'config.json': b'old'})
    model(source, 'new', {'config.json': b'new'})
    stage(['old'], source, out)
    stage(['new'], source, out)
    assert sorted(path.name for path in out.iterdir()) == ['MANIFEST.sha256', 'new']


def test_a_missing_model_fails_before_the_staged_models_are_touched(tmp_path):
    source, out = tmp_path / 'exp', tmp_path / 'learned_models'
    model(source, 'kept', {'config.json': b'{}'})
    stage(['kept'], source, out)
    before = (out / 'MANIFEST.sha256').read_bytes()
    with pytest.raises(SystemExit):
        stage(['kept', 'typo'], source, out)
    assert (out / 'MANIFEST.sha256').read_bytes() == before
