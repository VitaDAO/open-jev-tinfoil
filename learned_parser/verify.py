"""Plan checker (verifier; behind LEARNED_VERIFIER_DIR, experiments in ~/Documents/open-jev-experiments): scores whether a candidate plan matches the request. Cross-encoder over request + verbalized plan.

train:  jevverify.py train INIT_MODEL_DIR OUT_DIR [epochs]
score:  Verifier(OUT_DIR).score(req, queries) -> probability the plan matches
"""
import hashlib, json, math, random, re, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
V = json.load(open(Path(__file__).resolve().parents[1] / 'metadata/learned/vocabulary.json'))['metrics']
NAME = lambda m: (V.get(m) or {}).get('display_name') or m.replace('_', ' ')


import datetime as _dt
def FMT(d):
    try: return _dt.date.fromisoformat(str(d)[:10]).strftime('%a %d %b %Y')
    except ValueError: return str(d)[:10]                  # an impossible date (a corrupted negative) is shown as written
# Text versions: 1 plain dates; 2 today's date + weekday-named dates; 3 = the v2 text tokenized as a (request, plan) pair, so the
# plan is never cut (review S7/G16: v2 right-truncated one string, which cut the plan first). A v3 pair longer than MAX_LEN is not
# scored: it would need the request or history cut, so it fails closed (score 0.0) and is left out of training.
TEXT_VERSION = 3
MAX_LEN = 256


def period_text(p, v=TEXT_VERSION):
    p = p or {}; k = p.get('kind')
    if v >= 2 and k == 'between':
        s, e = str(p.get('start_at'))[:10], str(p.get('end_at'))[:10]
        return f"on {FMT(s)}" if s == e else f"{FMT(s)} to {FMT(e)}"
    if k == 'all_history' or not k: return 'all history'
    if k == 'relative': return f"last {p.get('amount')} {p.get('unit')}"
    if k == 'calendar': return f"this {p.get('period')}"
    if k == 'between': return f"{str(p.get('start_at'))[:10]} to {str(p.get('end_at'))[:10]}"
    return json.dumps(p, sort_keys=True)


def plan_text(queries, v=TEXT_VERSION):
    parts = []
    for q in queries:
        what = ', '.join(NAME(m) for m in q.get('metrics') or []) or ', '.join(q.get('records') or [])
        if q.get('profile_fields'): what += ' [' + ', '.join(q['profile_fields']) + ']'
        parts.append(f"{what}; {q.get('operation')}; {period_text(q.get('period'), v)}" + (f"; from {q['source']}" if q.get('source') else '')
                     + ('; night basis' if q.get('date_basis') == 'sleep_end_day' else ''))
    return ' | '.join(parts)


def pair_parts(current, history, queries, ref=None, v=TEXT_VERSION):
    head = f"today: {FMT(ref)} || " if v >= 2 and ref else ''
    return head + f"request: {current} || earlier: {' | '.join((history or [])[-3:])}", f"plan: {plan_text(queries, v)}"


def pair_text(current, history, queries, ref=None, v=TEXT_VERSION):
    return ' || '.join(pair_parts(current, history, queries, ref, v))


def encode(tok, rows, v=TEXT_VERSION, **kw):
    """Tokenize (current, history, queries, ref) tuples. v3: as text pairs, never cutting the plan (callers drop pairs that do not
    fit first, see fits); v1/v2: one right-truncated string, as those checkers were trained."""
    if v >= 3: return tok(*zip(*[pair_parts(*r, v=v) for r in rows]), truncation='only_first', max_length=MAX_LEN, **kw)
    return tok([pair_text(*r, v=v) for r in rows], truncation=True, max_length=MAX_LEN, **kw)


def fits(tok, row, v=TEXT_VERSION):
    args = pair_parts(*row, v=v) if v >= 3 else (pair_text(*row, v=v),)
    return len(tok(*args, truncation=False)['input_ids']) <= MAX_LEN


def head_file(v=TEXT_VERSION):
    """From v3 the head's file name carries the text version: code that predates a checker's input format finds no head to load and
    fails, instead of scoring the checker in the wrong format (pre-v3 code loads only verifier_head.pt)."""
    return 'verifier_head.pt' if v < 3 else f'verifier_head.v{v}.pt'


class Model(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc = enc; self.head = nn.Linear(enc.config.hidden_size, 1)
    def forward(self, batch):
        h = self.enc(**batch).last_hidden_state; m = batch['attention_mask'].unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


class Verifier:
    def __init__(self, path):
        path = Path(path); self.v = json.load(open(path / 'verifier.json'))['text_version'] if (path / 'verifier.json').exists() else 1
        if self.v > TEXT_VERSION: raise ValueError(f'checker text version {self.v} is newer than this code ({TEXT_VERSION})')
        self.tok = AutoTokenizer.from_pretrained(path); enc = AutoModel.from_pretrained(path)
        self.m = Model(enc); self.m.head.load_state_dict(torch.load(path / head_file(self.v), map_location='cpu', weights_only=True)); self.m.eval()
    @torch.inference_mode()
    def score(self, req, queries):
        st = req['state']; row = (st['current_request'], st.get('recent_user_requests'), queries, st.get('reference_date'))
        if not fits(self.tok, row, self.v): return 0.0             # fail closed: never scored with the request or history cut (S7)
        return float(torch.sigmoid(self.m(encode(self.tok, [row], self.v, return_tensors='pt')))[0])
