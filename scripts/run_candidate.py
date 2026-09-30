"""Candidate-only verified model mount and secret mapping; inference is unchanged."""
import hashlib
import os
from pathlib import Path

MANIFEST_SHA256 = 'da75040b34439f604e4c812690ae91f1ae0453ef902c54f51a98c73d70ecb97d'

def verify_models(root=Path('/opt/learned'), expected_manifest=MANIFEST_SHA256):
    manifest = (root / 'MANIFEST.sha256').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != expected_manifest:
        raise RuntimeError('Unapproved candidate model manifest')
    for line in manifest.decode().splitlines():
        expected, name = line.split('  ', 1)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Invalid model manifest path')
        digest = hashlib.sha256()
        with (root / relative).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError('Candidate model file mismatch: ' + name)

if __name__ == '__main__':
    verify_models()
    os.environ['OPEN_JEV_API_KEY'] = os.environ.pop('OPEN_JEV_CANDIDATE_API_KEY')
    os.execvp('uvicorn', ['uvicorn', 'server:create_app', '--factory', '--host',
        '0.0.0.0', '--port', '8080', '--workers', '1', '--no-access-log',
        '--limit-concurrency', '16', '--timeout-keep-alive', '5'])
