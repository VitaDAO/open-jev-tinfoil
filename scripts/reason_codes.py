"""Generate the published list of selector reason codes (spec §3.15).

Scans the serving modules for every literal code a handoff can carry. The
serving path maps any unlisted code to query_contract_unrepresentable, and
tests fail if this file and the source drift apart.
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'metadata/reason-codes.v1.json'
SOURCES = ['query_selector.py', 'proposal_selector.py', 'proposal_binding.py', 'trained_proposal_selector.py',
           'learned_selector.py', 'selector.py', 'temporal_spans.py', 'query_plan.py', 'schema_index.py']
PATTERNS = [r"(?:ValueError|UnrepresentableRequest)\(\s*'([a-z][a-z0-9_]+)'",
            r"reasons\.append\(\s*'([a-z][a-z0-9_]+)'",
            r"return None, '([a-z][a-z0-9_]+)'",
            r"return None, [^,\n]+, '([a-z][a-z0-9_]+)'",
            r"None if accepted else '([a-z][a-z0-9_]+)'",
            r"'reason'\s*:\s*'([a-z][a-z0-9_]+)'",
            r"Constraint\(kind='([a-z][a-z0-9_]+)'"]


def scan():
    sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
    from proposal_binding import CONSTRAINT_PATTERNS
    codes = set(CONSTRAINT_PATTERNS) | {'query_contract_unrepresentable'}
    for name in SOURCES + sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / 'compat/*.py'))):
        text = (ROOT / name).read_text()
        for pattern in PATTERNS:
            codes.update(match for match in re.findall(pattern, text) if match)
    # Raised and caught locally (proposal_binding.py:180-182); never reaches a response.
    return sorted(codes - {'year'})


def document(codes):
    return json.dumps({'schema_version': 'vita-reason-codes/v1', 'codes': codes}, indent=1) + '\n'


if __name__ == '__main__':
    OUT.write_text(document(scan()))
    print(OUT, len(scan()))
