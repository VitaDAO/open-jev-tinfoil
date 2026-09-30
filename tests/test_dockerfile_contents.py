"""The serving image holds every module server.py imports and serves the learned parser from a verified model mount.

Model-free: reads the Dockerfile and walks the imports with ast, function-level ones included (server.py loads the learned
parser lazily). The CI image runs this suite with only tests/, scripts/ and evidence/ mounted: no Dockerfile to read there.
"""
import ast
import re
import shlex
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / 'Dockerfile'
LEARNED = PurePosixPath('/opt/learned')
pytestmark = pytest.mark.skipif(not DOCKERFILE.exists(), reason='no Dockerfile here (suite running inside the image)')


def instructions():
    """(INSTRUCTION, arguments) in order: comment lines dropped, backslash continuations joined."""
    steps, pending = [], ''
    for line in DOCKERFILE.read_text().splitlines():
        if line.lstrip().startswith('#'):
            continue
        if line.rstrip().endswith('\\'):
            pending += line.rstrip()[:-1] + ' '
            continue
        line, pending = (pending + line).strip(), ''
        if line:
            name, args = (line.split(None, 1) + [''])[:2]
            steps.append((name.upper(), args.strip()))
    return steps


def copied():
    """In-image paths of the build-context files the COPY steps place, and the final WORKDIR (the server's import root)."""
    workdir, placed = PurePosixPath('/'), set()
    for name, args in instructions():
        if name == 'WORKDIR':
            workdir = workdir / args
        if name != 'COPY' or '--from' in args:
            continue
        *sources, dest = [word for word in args.split() if not word.startswith('--')]
        found = [path for source in sources for path in sorted(ROOT.glob(source))]
        for src in found:
            if src.is_dir():    # a directory's contents land in the destination, not the directory itself
                placed |= {str(workdir / dest / f.relative_to(src)) for f in src.rglob('*') if f.is_file()}
            else:
                into = dest.endswith('/') or dest == '.' or len(found) > 1
                placed.add(str(workdir / dest / src.name if into else workdir / dest))
    return placed, workdir


def local(dotted):
    """The repo files a dotted import loads when it resolves in the repo root: package __init__ files, then the module."""
    files, parts = [], dotted.split('.')
    for i in range(1, len(parts) + 1):
        path = ROOT.joinpath(*parts[:i])
        if path.with_suffix('.py').is_file():
            return files + [path.with_suffix('.py')]
        if not path.is_dir():
            break
        if (path / '__init__.py').is_file():
            files.append(path / '__init__.py')
    return files


def server_modules():
    """Repo-relative paths of the repo-local modules server.py imports, transitively."""
    seen, todo = set(), [ROOT / 'server.py']
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        package = path.relative_to(ROOT).parent.parts
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module.split('.') if node.module else []
                if node.level:      # relative to the importing module's package
                    base = [*package[:len(package) + 1 - node.level], *base]
                names = ['.'.join(base), *('.'.join([*base, alias.name]) for alias in node.names if alias.name != '*')]
            else:
                continue
            for name in names:
                todo.extend(local(name))
    return {path.relative_to(ROOT).as_posix() for path in seen}


def image_env():
    """Variables the ENV steps set (KEY=value pairs, or the legacy 'ENV KEY value')."""
    env = {}
    for name, args in instructions():
        if name != 'ENV':
            continue
        key, _, rest = args.partition(' ')
        if '=' in key:
            env.update(word.split('=', 1) for word in shlex.split(args))
        else:
            env[key] = rest.strip()
    return env


def test_every_module_the_server_imports_is_copied_into_the_image():
    modules = server_modules()
    assert 'learned_parser/decode.py' in modules     # the walk sees imports inside functions
    placed, root = copied()
    missing = sorted(m for m in modules if str(root / m) not in placed)
    assert not missing, f'server.py imports these, but the image has no {root}/<path>: {missing}'


def test_learned_parser_env_points_at_the_mounted_models():
    from scripts.stage_learned_models import MODELS
    env = image_env()
    assert env.get('OPEN_JEV_PARSER') == 'learned'
    parsers = [d.strip() for d in env.get('LEARNED_PARSER_DIR', '').split(',') if d.strip()]
    assert parsers and env.get('LEARNED_VERIFIER_DIR'), 'the parsers and the plan checker'
    dirs = [PurePosixPath(d) for d in [*parsers, env['LEARNED_VERIFIER_DIR']]]
    assert all(d.parent == LEARNED for d in dirs), dirs
    assert sorted(d.name for d in dirs) == sorted(MODELS), 'the models scripts/stage_learned_models.py stages'
    for key in ('LEARNED_PARSER_MIN_CONFIDENCE', 'LEARNED_VERIFIER_MIN'):
        assert 0 <= float(env.get(key, 'nan')) <= 1, key


def test_model_mount_uses_verified_candidate_entrypoint():
    steps = instructions()
    assert ('RUN', 'ln -s /tinfoil/models/jev-learned /opt/learned') in steps
    assert not any(name == 'COPY' and 'learned_models' in args for name, args in steps)
    assert ('CMD', '["python", "/app/scripts/run_candidate.py"]') in steps
    # Integrity checks must execute before starting the serving process.
    tree = ast.parse((ROOT / 'scripts/run_candidate.py').read_text())
    entrypoint = next(node for node in tree.body if isinstance(node, ast.If))
    calls = [node.value for node in entrypoint.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call)]
    assert isinstance(calls[0].func, ast.Name) and calls[0].func.id == 'verify_models'
    assert isinstance(calls[-1].func, ast.Attribute) and calls[-1].func.attr == 'execvp'
