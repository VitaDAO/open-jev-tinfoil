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
           'learned_selector.py', 'selector.py', 'temporal_spans.py', 'query_plan.py', 'schema_index.py', 'learned_parser/decode.py']
PATTERNS = [r"(?:ValueError|UnrepresentableRequest)\(\s*'([a-z][a-z0-9_]+)'",
            r"reasons\.append\(\s*'([a-z][a-z0-9_]+)'",
            r"return None, '([a-z][a-z0-9_]+)'",
            r"return None, [^,\n]+, '([a-z][a-z0-9_]+)'",
            r"None if accepted else '([a-z][a-z0-9_]+)'",
            r"'reason'\s*:\s*'([a-z][a-z0-9_]+)'",
            r"Constraint\(kind='([a-z][a-z0-9_]+)'",
            r"\bH\(\s*'([a-z][a-z0-9_]+)'",          # learned parser handoffs
            r"'reason_codes'\s*:\s*\[\s*'([a-z][a-z0-9_]+)'"]   # literal handoff results (learned-parser agreement)
LEARNED_MODEL_CODES = r"'(model_[a-z_]+)'"            # learned parser model classes (MODEL_CODES), learned_parser/decode.py only


def scan():
    sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
    from proposal_binding import CONSTRAINT_PATTERNS
    codes = set(CONSTRAINT_PATTERNS) | {'query_contract_unrepresentable'}
    for name in SOURCES + sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / 'compat/*.py'))):
        text = (ROOT / name).read_text()
        for pattern in PATTERNS:
            codes.update(match for match in re.findall(pattern, text) if match)
        if name == 'learned_parser/decode.py':      # the model classes as declared in MODEL_CODES and used in H(...), not any 'model_*' literal ('model_type')
            table = text[text.index('MODEL_CODES = '):text.index(')}', text.index('MODEL_CODES = '))]
            codes.update(re.findall(LEARNED_MODEL_CODES, table) + re.findall(r"H\(\s*'(model_[a-z_]+)'\s*\)", text))
    # Raised and caught locally (proposal_binding.py:180-182); never reaches a response.
    return sorted(codes - {'year'})


def document(codes):
    return json.dumps({'schema_version': 'vita-reason-codes/v1', 'codes': codes}, indent=1) + '\n'


if __name__ == '__main__':
    OUT.write_text(document(scan()))
    print(OUT, len(scan()))
