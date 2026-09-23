import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = '19bf9a64815add579fbf6c907bef584d9277a8e4'
root = Path(os.environ.get('MODEL_DIR', '/opt/model'))
snapshot_download('com-kotobalabs/open-jev-deberta-v3-large', revision=REVISION,
                  local_dir=root, ignore_patterns=['typed_decisions/*', '.gitattributes', 'README.md'])
manifest = {}
for path in sorted(root.iterdir()):
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        manifest[path.name] = digest.hexdigest()
(root / 'manifest.json').write_text(json.dumps({'revision': REVISION, 'files': manifest}, indent=2) + '\n')
