"""Stage the learned request parser's models into the Docker build context: learned_models/<name>/ plus MANIFEST.sha256.

The models exist only in the experiments folder, so the serving image bakes them in: the Dockerfile copies learned_models/
to /opt/learned and checks every file against MANIFEST.sha256, whose own sha256 it pins (printed below; update the
Dockerfile when the models change).

  python scripts/stage_learned_models.py [--source DIR] [NAME ...]
"""
import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'learned_models'
SOURCE = Path.home() / 'Documents/open-jev-experiments/exp'
MODELS = ('jevparse_r8a', 'jevparse_r8b', 'verifier_v4')     # exp/final_choice_v12.json: the parser pair and the plan checker


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stage(names, source=SOURCE, out=OUT):
    """Copy each model dir (a name under source, or a path) to out/<dir name>/ and write out/MANIFEST.sha256: one
    'sha256  path' line per file, sorted by path (the format sha256sum --check reads). Returns the manifest's sha256."""
    dirs = [Path(source) / name for name in names]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing or not dirs or len({d.name for d in dirs}) < len(dirs):
        raise SystemExit(f'need distinct model dirs; not found: {", ".join(missing) or "-"}')
    if out.exists():
        shutil.rmtree(out)      # restage from scratch: a model no longer chosen never reaches the image
    out.mkdir(parents=True)
    for model in dirs:
        for src in sorted(model.rglob('*')):
            if src.is_file():
                dst = out / model.name / src.relative_to(model)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                dst.chmod(0o644)    # the serving container runs as non-root UID 10001; trained weights are saved 0600
    for path in [out, *out.rglob('*')]:
        if path.is_dir():
            path.chmod(0o755)
    files = sorted(path.relative_to(out).as_posix() for path in out.rglob('*') if path.is_file())
    manifest = out / 'MANIFEST.sha256'
    manifest.write_text(''.join(f'{digest(out / f)}  {f}\n' for f in files))
    manifest.chmod(0o644)
    return digest(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('names', nargs='*', default=list(MODELS), help=f'model dirs under --source (default: {" ".join(MODELS)})')
    parser.add_argument('--source', type=Path, default=SOURCE, help=f'experiments folder (default: {SOURCE})')
    args = parser.parse_args()
    pinned = stage(args.names, args.source)
    size = sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file())
    print(f'staged {" ".join(args.names)} in {OUT} ({size:,} bytes)')
    print(f'{pinned}  MANIFEST.sha256   <- the digest the Dockerfile pins')
