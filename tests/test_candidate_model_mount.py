import hashlib
import pytest
from scripts.run_candidate import verify_models

def bundle(tmp_path):
    (tmp_path/'weight.bin').write_bytes(b'exact-pinned-weight')
    content = hashlib.sha256(b'exact-pinned-weight').hexdigest()+'  weight.bin\n'
    (tmp_path/'MANIFEST.sha256').write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()

def test_exact_mounted_model_bundle(tmp_path):
    expected=bundle(tmp_path)
    verify_models(tmp_path,expected)

def test_manifest_replacement_rejected(tmp_path):
    expected=bundle(tmp_path)
    (tmp_path/'MANIFEST.sha256').write_text('replacement')
    with pytest.raises(RuntimeError,match='manifest'):
        verify_models(tmp_path,expected)

def test_model_corruption_rejected(tmp_path):
    expected=bundle(tmp_path)
    (tmp_path/'weight.bin').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='file mismatch'):
        verify_models(tmp_path,expected)

def test_missing_model_rejected(tmp_path):
    expected=bundle(tmp_path)
    (tmp_path/'weight.bin').unlink()
    with pytest.raises(FileNotFoundError):
        verify_models(tmp_path,expected)
