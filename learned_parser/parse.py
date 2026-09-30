"""Learned, catalog-driven parser for Vita health requests (behind OPEN_JEV_PARSER=learned; experiments in ~/Documents/open-jev-experiments).

One ModernBERT pass over the request + history. Catalog items (metrics, records,
profile fields) are linked by late interaction against their embedded descriptions,
so adding an item to the catalog needs data, not code. Only calendar arithmetic and
schema/privacy validation are deterministic.
"""
import functools, json, re
from datetime import date, timedelta
import calendar as cal
import torch, torch.nn as nn, torch.nn.functional as F

STATUS = ['plan', 'handoff', 'research']
HANDOFF = [None, 'definition', 'advice', 'judgement', 'comparison', 'count', 'extreme', 'negation', 'filter', 'other_person',
           'action', 'prediction', 'doc_metadata', 'clock_time', 'unsupported_period', 'mixed_windows', 'ambiguous_name',
           'unknown_metric', 'vague_followup', 'non_english', 'injection', 'crisis', 'multi_topic', 'chitchat']
OPERATION = ['latest', 'trend']
KIND = ['none', 'all_history', 'rolling', 'calendar', 'to_date', 'since', 'between', 'day', 'night']
UNIT = [None, 'day', 'week', 'month', 'year']
OFFSET = [None, 'current', 'previous']
AMOUNT = list(range(0, 37))                      # 0 = not applicable
SOURCE = [None, 'oura', 'whoop', 'garmin', 'withings', 'apple_health', 'fitbit', 'polar', 'quest', 'labcorp']   # labs: registry valid_sources
INHERIT = [None, 'subject', 'period', 'both']
ROLES = ['O', 'metric', 'record', 'profile_field', 'time', 'source', 'operation', 'research_topic']


def request_text(current, history):
    return 'request: ' + current + ' || earlier: ' + ' | '.join(history[-3:])


try:     # catalog-declared ambiguous names (e.g. bare "crp", "calories") never identify a single item
    from pathlib import Path as _P
    AMBIGUOUS = set(json.load(open(_P(__file__).resolve().parents[1] / 'metadata/learned/vocabulary_ambiguous.json'))['terms'])
except FileNotFoundError:
    AMBIGUOUS = set()


# Names each catalog item explicitly claims (synonyms/abbreviations). An id-derived name ("strain score" from strain_score)
# never overrides another item's explicit synonym (spec v1.6: bare "strain score" is day_strain, strain_score is "Workout strain").
CLAIMED = {}
try:
    for _sec in ('metrics', 'records', 'profile_fields'):
        for _k, _e in (json.load(open(_P(__file__).resolve().parents[1] / 'metadata/learned/vocabulary_enriched.json')).get(_sec) or {}).items():
            for _n in (_e.get('synonyms') or []) + (_e.get('abbreviations') or []): CLAIMED.setdefault(_n.lower().strip(), set()).add(_k)
except FileNotFoundError:
    pass


def item_text(kind, key, entry):
    keep = lambda xs: [x for x in xs if x.lower().strip() not in AMBIGUOUS]
    syn = ', '.join(keep((entry.get('synonyms') or [])[:12] + (entry.get('abbreviations') or [])[:6] + (entry.get('lay_terms') or [])[:6]))
    # The item's own id name is part of its text unless another item explicitly claims that name.
    name = key.replace('_', ' '); shown = entry.get('display_name') or name
    if CLAIMED.get(name, {key}) - {key}:
        return f"{kind}: {shown}. {entry.get('description') or ''} also called: {syn}"
    return f"{kind}: {shown} ({name}). {entry.get('description') or ''} also called: {name}, {syn}"


class JevParser(nn.Module):
    def __init__(self, encoder, hidden, dim=256):
        super().__init__()
        self.enc = encoder
        self.tok_proj = nn.Linear(hidden, dim); self.item_proj = nn.Linear(hidden, dim)
        self.link_scale = nn.Parameter(torch.tensor(20.0)); self.link_bias = nn.Parameter(torch.tensor(-8.0))
        self.heads = nn.ModuleDict({n: nn.Linear(hidden, len(v)) for n, v in
            [('status', STATUS), ('handoff', HANDOFF), ('operation', OPERATION), ('kind', KIND), ('unit', UNIT),
             ('offset', OFFSET), ('amount', AMOUNT), ('source', SOURCE), ('inherit', INHERIT), ('crisis', [0, 1])]})
        self.roles = nn.Linear(hidden, len(ROLES))

    def encode(self, batch):
        h = self.enc(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
        m = batch['attention_mask'].unsqueeze(-1).float()
        return h, (h * m).sum(1) / m.sum(1).clamp(min=1)

    def item_embeddings(self, item_batch):
        _, pooled = self.encode(item_batch)
        return F.normalize(self.item_proj(pooled), dim=-1)

    def forward(self, batch, items):
        h, pooled = self.encode(batch)
        out = {n: head(pooled) for n, head in self.heads.items()}
        out['roles'] = self.roles(h); out['hidden'] = h
        t = F.normalize(self.tok_proj(h), dim=-1)                              # B x L x d
        sim = torch.einsum('bld,kd->blk', t, items)                            # B x L x K
        sim = sim.masked_fill(batch['attention_mask'].unsqueeze(-1) == 0, -1e4)
        out['link'] = sim.max(1).values * self.link_scale + self.link_bias      # B x K logits
        return out


def load_parser_state(model, state, new_bias=-30.0):
    """Load parser weights; a head saved with fewer classes (an older label set) keeps its rows, the added classes start with
    zero weight and new_bias (-30 = never predicted by an old model; trainers pass a learnable value)."""
    own = model.state_dict()
    for k, v in list(state.items()):
        if k.startswith('heads.') and k in own and own[k].shape != v.shape and own[k].shape[1:] == v.shape[1:] and v.shape[0] < own[k].shape[0]:
            pad = torch.zeros_like(own[k]); pad[:v.shape[0]] = v
            if k.endswith('.bias'): pad[v.shape[0]:] = new_bias if new_bias is not None else float(v.min())
            state[k] = pad
    return model.load_state_dict(state, strict=False)


# ---------------- deterministic period resolution (calendar arithmetic only) ----------------
def _months_back(d, n):
    m = d.month - n; y = d.year + (m - 1) // 12; m = (m - 1) % 12 + 1
    return date(y, m, min(d.day, cal.monthrange(y, m)[1]))


def resolve_period(kind, amount, unit, offset, ref, time_text=None, date_parser=None):
    """Structured period → {'kind': 'all_history'} | relative | calendar | between, as the QueryPlan contract expects."""
    if kind in ('none', 'all_history'):
        return {'kind': 'all_history'}
    if kind == 'rolling':
        stated = parse_rolling(time_text or '')                      # the stated number wins over the head (heads cap at 36)
        if stated: amount, unit = stated
        if not amount or unit not in ('day', 'week', 'month', 'year'): return None
        return {'kind': 'relative', 'amount': int(amount) * (12 if unit == 'year' else 1), 'unit': 'months' if unit == 'year' else unit + 's'}
    if kind in ('calendar', 'to_date'):
        if unit not in ('day', 'week', 'month', 'year'): return None
        if offset == 'previous':
            if unit == 'day': d = ref - timedelta(days=1); return {'kind': 'between', 'start_at': d.isoformat(), 'end_at': d.isoformat()}
            if unit == 'week':
                s = ref - timedelta(days=ref.weekday() + 7); return {'kind': 'between', 'start_at': s.isoformat(), 'end_at': (s + timedelta(days=6)).isoformat()}
            if unit == 'month':
                s = _months_back(ref.replace(day=1), 1); return {'kind': 'between', 'start_at': s.isoformat(), 'end_at': s.replace(day=cal.monthrange(s.year, s.month)[1]).isoformat()}
            return {'kind': 'between', 'start_at': f'{ref.year - 1}-01-01', 'end_at': f'{ref.year - 1}-12-31'}
        return {'kind': 'calendar', 'period': unit}
    if kind == 'night':
        return {'kind': 'between', 'start_at': ref.isoformat(), 'end_at': ref.isoformat(), 'night': True}
    if kind in ('day', 'between', 'since'):
        if date_parser is None or not time_text: return None
        span = date_parser(time_text, ref)                                  # (start, end) dates or None
        if span is None: return None
        start, end = span
        if kind == 'since': end = ref
        if kind == 'day': end = start
        if start > end or start > ref: return None                          # future or inverted → handoff
        return {'kind': 'between', 'start_at': start.isoformat(), 'end_at': end.isoformat()}
    return None


# ---------------- generic date-syntax parser for tagged time spans ----------------
_MONTHS = {m.lower(): i for i, m in enumerate(cal.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(cal.month_abbr) if m}); _MONTHS['sept'] = 9
_WEEKDAYS = {d.lower(): i for i, d in enumerate(cal.day_name)}
_WEEKDAYS.update({d.lower(): i for i, d in enumerate(cal.day_abbr)})
_WEEKDAYS.update({'tues': 1, 'weds': 2, 'thur': 3, 'thurs': 3})     # common abbreviations that are no English word ("steps tues")
# A dotted date may end a sentence ("on 04.05.", "on 13.08.2026."): a stop after it is punctuation unless a digit follows it.
_TOKEN = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})|(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?|(?<![\d.])(\d{4})\.(\d{1,2})\.(\d{1,2})(?!\d|\.\d)|(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})(?!\d|\.\d)|(?<![\d.])(\d{2})\.(\d{2})(?!\d|\.\d)|(\d{1,2})(?:st|nd|rd|th)\b|[a-z]+|\d{1,4}")
_RANGE_WORDS = {'to', 'through', 'thru', 'till', 'til', 'until'}
_SUBPERIOD = {('first', 'half'): (1, 15), ('second', 'half'): (16, 31), ('first', 'week'): (1, 7), ('last', 'week'): (-7, 31)}


class AmbiguousDate(ValueError):
    pass


class InvalidDate(ValueError):
    """A dotted or ISO date that cannot exist ("31.02", "2026-13-01"): spec v1.8 10g.21 -> invalid_or_out_of_range_date."""


def _undated_year(mo, dy, ref, mode):
    """The year of a date stated without one: the most recent occurrence ('past'), the next ('next'), the current year ('year'). An undated
    "29.02", "29/02" or "February 29" whose resolved year has no such day is invalid on every date syntax (10g.21 -> InvalidDate)."""
    return ref.year - (mode == 'past' and (mo, dy) > (ref.month, ref.day)) + (mode == 'next' and (mo, dy) < (ref.month, ref.day))


class _DayRange(tuple):
    """A (start, end) span read from one explicit range of days ("Sept 7-13", "1-14 September"); parse_range tells it from a period."""


# Misspelled day words ("yestrday", "yesteday", "wendesday") are read as the word they misspell: at most this many edits (a
# transposition is one), and only when no dictionary word is that close (so "sunday", "today" and month names are exact-only:
# "sundae", "toady", "remember"). Plural weekdays/weekends ("mondays") are recurring filters, never corrected into one day.
_TYPO_EDITS = {'yesterday': 2, 'tomorrow': 2, 'wednesday': 2, 'monday': 1, 'tuesday': 1, 'thursday': 1, 'friday': 1, 'saturday': 1, 'weekend': 1}


def _edits(a, b):
    """Optimal-string-alignment distance (insert, delete, substitute, swap adjacent)."""
    d = [[i + j if not i * j else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]: d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


# Clipped period words ("rhr lst wk", "past 2 wks", "last 3 mos"): read as the full words, so the date syntax can check the model's
# window (a "last week" the offset head reads as this week disagrees; unread, it passed). "mo" only after a number or a determiner;
# never "hr" (heart rate).
_ABBR = {'lst': 'last', 'nxt': 'next', 'prev': 'previous', 'wk': 'week', 'wks': 'weeks', 'wek': 'week', 'weks': 'weeks', 'mth': 'month', 'mths': 'months',
         'mnth': 'month', 'mnths': 'months', 'yr': 'year', 'yrs': 'years', 'wknd': 'weekend', 'wknds': 'weekends',   # "wek(s)": the usual typo, no English word
         'lasr': 'last', 'lsat': 'last', 'laast': 'last'}   # typos of "last" that are no English word ("lasr week" was read as this week)
_ABBR_RX = re.compile(r'\b(?:' + '|'.join(_ABBR) + r')\b', re.I)
_MO_RX = re.compile(r'\b(\d{1,3}|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|last|this|next|past|previous)(\s*)mos?\b', re.I)


_ABBR_WORD = lambda m: _ABBR[m.group(0).lower()]
_MO_WORD = lambda m: m.group(1) + (m.group(2) or ' ') + ('months' if m.group(0).lower().endswith('mos') or m.group(1).isdigit() and m.group(1) != '1' else 'month')


def _expand(text):
    return _MO_RX.sub(_MO_WORD, _ABBR_RX.sub(_ABBR_WORD, text))


# Short forms the date syntax reads that the wording guards would miss ("weds", "thur", "yday", "nite"): one spelling for both.
_SHORT_DAY = {'lastt': 'last', 'lasr': 'last', 'tues': 'tuesday', 'weds': 'wednesday', 'thur': 'thursday', 'thurs': 'thursday', 'yday': 'yesterday', 'nite': 'night', 'nites': 'nights'}
_SHORT_RX, _WORD5 = re.compile(r'\b(?:' + '|'.join(_SHORT_DAY) + r')\b'), re.compile(r'[a-z]{5,}')
_ODD_SPACE = re.compile(r'[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]|[\u200b-\u200d\u2060\ufeff]')


def normalize_time_text(text, offsets=False):
    """The text as the date syntax reads it: Unicode spaces as spaces (zero-width ones dropped), clipped period words expanded (_expand),
    lowercase, misspelt day words corrected (_fix_typo) and short day forms written out. The wording guards run on this reading as well as
    on the raw text, so a spelling the parser accepts can never slip past them ("before yestrday", "through lasr month", "until weds").
    offsets=True also returns, for each character of the result (and its end), the index of the raw character it came from."""
    idx = list(range(len(text) + 1)) if offsets else None
    def sub(rx, f, s, idx):
        if idx is None: return rx.sub(f, s), None
        out, m2, k = [], [], 0
        for m in rx.finditer(s):
            r = f(m); out.append(s[k:m.start()]); m2 += idx[k:m.start()]
            out.append(r); m2 += [idx[m.start() + min(j, max(0, m.end() - m.start() - 1))] for j in range(len(r))]; k = m.end()
        out.append(s[k:]); m2 += idx[k:]
        return ''.join(out), m2
    s, idx = sub(_ODD_SPACE, lambda m: '' if m.group(0) in '\u200b\u200c\u200d\u2060\ufeff' else ' ', text, idx)
    s, idx = sub(_ABBR_RX, _ABBR_WORD, s, idx)
    s, idx = sub(_MO_RX, _MO_WORD, s, idx)
    s = ''.join(c.lower() if len(c.lower()) == 1 else c for c in s)      # offsets kept ("İ".lower() is two characters)
    s, idx = sub(_WORD5, lambda m: _fix_typo(m.group(0)), s, idx)
    s, idx = sub(_SHORT_RX, lambda m: _SHORT_DAY[m.group(0)], s, idx)
    return (s, idx) if offsets else s


@functools.lru_cache(maxsize=4096)
def _fix_typo(w):
    if w in _WEEKDAYS or w in _MONTHS or w in _TYPO_EDITS: return w
    near = {c for c, k in _TYPO_EDITS.items() if w[0] == c[0] and abs(len(w) - len(c)) <= k and _edits(w, c) <= k
            and not (c not in ('yesterday', 'tomorrow') and w == c + 's')}
    return near.pop() if len(near) == 1 else w


_ORD_WORD = {w: f'{n}{"th" if 10 < n % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")}' for n, w in enumerate(
    ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth',
     'seventeenth', 'eighteenth', 'nineteenth', 'twentieth'] + [f'twenty-{o}' for o in ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth']] + ['thirtieth', 'thirty-first'], 1)}
_ORD_OF_MONTH = re.compile(r'\b(' + '|'.join(w.replace('-', '[- ]') for w in sorted(_ORD_WORD, key=len, reverse=True)) + r')\s+of\s+(?=(?:' + '|'.join(list(_MONTHS)) + r')\b)')


_MONTH_ORD = re.compile(r'\b(' + '|'.join(list(_MONTHS)) + r')\s+(' + '|'.join(w.replace('-', '[- ]') for w in sorted(_ORD_WORD, key=len, reverse=True)) + r')\b(?!\s+(?:half|week|weekend|weekday|day|of|quarter|mon|tue|wed|thu|fri|sat|sun))')


def _dates_in(text, ref, mode='past'):
    """Every date expression in the text, as (start, end) spans at the stated granularity.
    mode 'past' (observations): the most recent past occurrence, current periods to date. Calendar records (spec v1.8 10g.19):
    'next' (due wording) = the next occurrence, whole periods; 'year' (neutral wording) = within the current calendar year."""
    text = normalize_time_text(text); out = []
    text = _ORD_OF_MONTH.sub(lambda m: f'{_ORD_WORD[re.sub(r"[- ]+", "-", m.group(1))]} of ', text)      # "the first of September" is "the 1st of September"
    text = _MONTH_ORD.sub(lambda m: f'{m.group(1)} {_ORD_WORD[re.sub(r"[- ]+", "-", m.group(2))]}', text)   # "September first" is "September 1st"
    toks = [m for m in _TOKEN.finditer(text)]
    w_ = lambda k: toks[k].group(0) if 0 <= k < len(toks) else ''
    def day_of(k):              # a day-of-month token: "7", "07", "7th"
        if not 0 <= k < len(toks): return None
        t = toks[k]; v = t.group(15) or (t.group(0) if t.group(0).isdigit() and len(t.group(0)) <= 2 else None)
        return int(v) if v and 1 <= int(v) <= 31 else None
    dash = lambda a, b: bool(re.fullmatch(r'\s*[-\u2013\u2014]\s*', text[toks[a].end():toks[b].start()]))   # "1-14", "7 – 13"
    month_near = lambda k: any(w_(x) in _MONTHS for x in range(k + 1, k + 5))
    # "the night of D" (before token k0) / "D night" (at token k1, after D): the night that started on D, so it ends on D+1 (spec v1.8
    # 10g.4), as for weekdays below. A night that starts today ends tomorrow: the caller rejects the future day.
    night = lambda k0, k1: w_(k1) == 'night' or (w_(k0 - 1 - (w_(k0 - 1) == 'the')) == 'of' and w_(k0 - 2 - (w_(k0 - 1) == 'the')) == 'night')
    def count_before(k):        # the number just before token k: "3", "five", "twenty-five", "a hundred and ten"
        if w_(k - 1).isdigit(): return int(w_(k - 1))
        if w_(k - 1) in ('few', 'several'): return None      # "a few days ago" is no count (10g.24 names none): left to the caller's handoff
        return next((n for j in range(max(0, k - 5), k) for n in [num_words(' '.join(w_(x) for x in range(j, k)))] if n), None)
    i = 0
    while i < len(toks):
        m = toks[i]; w = m.group(0)
        if m.group(1):
            try: d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError: raise InvalidDate(m.group(0)) from None                # "2026-13-01", "2026-02-30" (10g.21)
            if night(i, i + 1): d += timedelta(days=1); i += w_(i + 1) == 'night'
            out.append((d, d)); i += 1; continue
        if m.group(4):
            mo, dy = int(m.group(4)), int(m.group(5)); y = int(m.group(6)) if m.group(6) else ref.year
            if mo <= 12 and dy <= 12 and mo != dy: raise AmbiguousDate(m.group(0))   # "03/04": day/month order unknown (spec §2.2)
            if mo > 12 and dy <= 12: mo, dy = dy, mo                                 # "13/04" can only be day/month
            if not m.group(6) and (mo, dy) == (2, 29) and not cal.isleap(_undated_year(mo, dy, ref, 'past')):
                raise InvalidDate(m.group(0))                                        # "29/02" with no year: no such day in that year
            if not (1 <= mo <= 12 and 1 <= dy <= cal.monthrange(y if y >= 100 else y + 2000, mo)[1]): i += 1; continue   # not a date
            y = y + 2000 if y < 100 else y
            d = date(y, mo, dy)
            if not m.group(6) and d > ref: d = date(y - 1, mo, dy)
            if night(i, i + 1): d += timedelta(days=1); i += w_(i + 1) == 'night'
            out.append((d, d)); i += 1; continue
        if m.group(7) or m.group(10) or m.group(13):
            # Dotted numeric dates are day-first (spec v1.8 10g.21): "01.08.2026" = 1 Aug; "YYYY.MM.DD" = year first; a two-digit
            # year is 20YY; no year = the most recent occurrence (calendar records: by wording mode). A clock time ("at 04.05",
            # "04.05 pm", "kl. 14.30") or a value with a unit ("10.05 mmol/L") is not a date.
            before = text[max(0, m.start() - 4):m.start()]; after = text[m.end():m.end() + 6]
            if re.search(r'(?:\bat|\bkl\.?|@)\s*$', before) or re.match(r'\s*(?:am\b|pm\b|a\.m|p\.m|h\b|hrs?\b|o.?clock)|\s*(?:%|mmol|mg|mcg|ng|pg|nmol|pmol|iu\b|u/l|g/dl|g\b|kg|lbs?\b|bpm|ms\b|mm\b|cm\b|meq|mmhg)', after, re.I):
                i += 1; continue
            if m.group(7): y, mo, dy = int(m.group(7)), int(m.group(8)), int(m.group(9))
            elif m.group(10): dy, mo, y = int(m.group(10)), int(m.group(11)), int(m.group(12)); y = y + 2000 if y < 100 else y
            else: dy, mo, y = int(m.group(13)), int(m.group(14)), None
            if not (1 <= mo <= 12 and 1 <= dy <= cal.monthrange(y or 2024, mo)[1]): raise InvalidDate(m.group(0))
            if y is None:
                y = _undated_year(mo, dy, ref, mode)
                if dy > cal.monthrange(y, mo)[1]: raise InvalidDate(m.group(0))      # "29.02" with no year: no such day in that year
            d = date(y, mo, dy)
            if night(i, i + 1): d += timedelta(days=1); i += w_(i + 1) == 'night'
            out.append((d, d)); i += 1; continue
        if m.group(15) and not month_near(i) and w_(i - 1) not in _MONTHS:
            # A day of month with no month ("on the 3rd", "Monday the 28th"): the most recent such day; a weekday named with it
            # picks the adjacent month whose date falls on that weekday (a future date is rejected by the caller).
            n = int(m.group(15)); wd = _WEEKDAYS.get(w_(i - 2)) if w_(i - 1) == 'the' else None
            cands = []
            for k in (0, -1, 1):
                yy, mm = (ref.year, ref.month + k) if 1 <= ref.month + k <= 12 else ((ref.year - 1, 12) if ref.month + k < 1 else (ref.year + 1, 1))
                if n <= cal.monthrange(yy, mm)[1]: cands.append(date(yy, mm, n))
            pick = ([d for d in cands if d.weekday() == wd] if wd is not None else sorted(d for d in cands if d >= ref)[:1] if mode == 'next'
                    else [d for d in cands if (d.year, d.month) == (ref.year, ref.month)][:1] if mode == 'year' else [d for d in cands if d <= ref][:1])
            if pick:
                d = pick[0]
                if night(i, i + 1): d += timedelta(days=1); i += w_(i + 1) == 'night'
                out.append((d, d))
            i += 1; continue
        if w in _WEEKDAYS:          # most recent such day strictly before today (spec v1.8 10g.3); "<day> night" started that day
            prev = toks[i - 1].group(0) if i > 0 else ''; nxt = [t.group(0) for t in toks[i + 1:i + 3]]
            if nxt[:1] == ['the'] and i + 2 < len(toks) and toks[i + 2].group(15): i += 1; continue   # "Monday the 28th": the date decides
            # "Monday to Wednesday last week", "Mon-Wed this week", "Monday and Wednesday last week": the week words after the second
            # weekday qualify the first one too
            k = i + 1 + (w_(i + 1) in _RANGE_WORDS | {'and'})
            share = [w_(k + 1), w_(k + 2)] if w_(k) in _WEEKDAYS and (k > i + 1 or dash(i, i + 1)) else []
            if prev == 'this' or nxt[:2] == ['this', 'week'] or share == ['this', 'week']:     # "this <weekday>", "<weekday> this week": that day of the current week
                d = ref - timedelta(days=ref.weekday() - _WEEKDAYS[w]); i += 2 * (nxt[:2] == ['this', 'week'])
                if d > ref: out.append((d, d)); i += 1; continue          # future -> caller rejects (start > ref)
            elif nxt[:2] == ['last', 'week'] or share == ['last', 'week']:
                d = ref - timedelta(days=ref.weekday() + 7) + timedelta(days=_WEEKDAYS[w]); i += 2 * (nxt[:2] == ['last', 'week'])
            elif mode == 'next':    # due wording: the next such day
                d = ref + timedelta(days=(_WEEKDAYS[w] - ref.weekday()) % 7 or 7)
            else:
                d = ref - timedelta(days=(ref.weekday() - _WEEKDAYS[w]) % 7 or 7)
            if nxt[:1] == ['night'] or (prev == 'of' and w_(i - 2) == 'night'):   # "<day> night" / "the night of <day>" started that day
                d = min(d + timedelta(days=1), ref); i += nxt[:1] == ['night']
            out.append((d, d)); i += 1; continue
        if w in ('today', 'tonight'): out.append((ref, ref)); i += 1; continue
        if w in ('morning', 'afternoon', 'evening'):     # "this morning" is today; a bare "morning!" is a greeting, not a date
            if w_(i - 1) == 'this': out.append((ref, ref))
            i += 1; continue
        if w == 'now':              # only as the end of a range ("from June 1 to now"); a bare "now" is a follow-up marker
            if w_(i - 1) in _RANGE_WORDS: out.append((ref, ref))
            i += 1; continue
        if w == 'tomorrow': d = ref + timedelta(days=1); out.append((d, d)); i += 1; continue
        n = count_before(i) if w in ('day', 'days') and (w_(i + 1) == 'ago' or (w_(i + 1) == 'back' and w_(i + 2) != 'to' and (w_(i + 2), w_(i + 3)) != ('and', 'forth'))) else None
        if n:                       # "3 days ago", "3 days back" (10g.36 item 2), "twenty-five days ago" = that one day
            if 0 < n <= 366: d = ref - timedelta(days=n); out.append((d, d))
            i += 2; continue
        if w == 'before' and w_(i - 1) == 'day' and w_(i + 1) == 'yesterday':      # "the day before yesterday"
            d = ref - timedelta(days=2); out.append((d, d)); i += 2; continue
        if w in ('night', 'nights', 'nite', 'nites') and w_(i - 1) == 'last':     # "last night('s)" = the night ending on the reference date (spec v1.8 10g.4); "nite" is its informal spelling
            out.append((ref, ref)); i += 1; continue
        pre = 1 if w_(i + 1) in ('before', 'preceding') else 2 if (w_(i + 1), w_(i + 2)) == ('prior', 'to') else 0
        if w in ('night', 'week', 'month', 'year') and pre and w_(i + pre + 1) == 'last':
            # "the <unit> before last" is absolute: the whole period two back ("the week before last" on Sat 09-26 = 09-07..09-13;
            # "the night before last" ended yesterday); so are "the <unit> preceding/prior to last (<unit>)", "the week before last week"
            if w == 'night': s = e = ref - timedelta(days=1)
            elif w == 'week': s = ref - timedelta(days=ref.weekday() + 14); e = s + timedelta(days=6)
            elif w == 'month': s = _months_back(ref.replace(day=1), 2); e = s.replace(day=cal.monthrange(s.year, s.month)[1])
            else: s, e = date(ref.year - 2, 1, 1), date(ref.year - 2, 12, 31)
            out.append((s, e)); i += pre + 2 + (w_(i + pre + 2) == w); continue
        lead = {w_(i - 2), w_(i - 3) if w_(i - 2) == 'the' else ''}          # the words before "last/previous (the)"
        if w in ('week', 'month', 'year') and w_(i + 1) != 'of' and not lead & {'over', 'in', 'during', 'within', 'since'} and (
                w_(i - 1) == 'previous' or (w_(i - 1) == 'last' and w_(i - 2) != 'the')):
            # bare "last week/month/year" is the previous calendar period (spec item 10: "last week" = the previous Mon-Sun,
            # "last month" = the previous calendar month, "last year" = 2025); standalone "(the) previous week/month" is the same
            # (10g.25). "(over/in/during/within) the last X" is rolling (parse_rolling) and "the last week of <month>" a part of it;
            # "since last X" (from its start? a rolling X?) is not defined by the spec and stays unparsed.
            if w == 'week': s = ref - timedelta(days=ref.weekday() + 7); e = s + timedelta(days=6)
            elif w == 'month': s = _months_back(ref.replace(day=1), 1); e = s.replace(day=cal.monthrange(s.year, s.month)[1])
            else: s, e = date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31)
            out.append((s, e)); i += 1; continue
        if w in ('mtd', 'ytd'):     # month/year to date: "this month" = MTD, "this year" = YTD (spec item 10)
            out.append((ref.replace(day=1) if w == 'mtd' else ref.replace(month=1, day=1), ref)); i += 1; continue
        if w in ('week', 'month', 'year') and w_(i - 1) in ('this', 'current') and w_(i - 2) != 'of':
            # the current period (spec item 10): to date for observations; whole for calendar records (10g.19). A part of it
            # ("the end of this month", "the first week of this month") is not the whole period and stays unparsed.
            s = ref - timedelta(days=ref.weekday()) if w == 'week' else ref.replace(day=1) if w == 'month' else ref.replace(month=1, day=1)
            e = s + timedelta(days=6) if w == 'week' else s.replace(day=cal.monthrange(s.year, s.month)[1]) if w == 'month' else s.replace(month=12, day=31)
            out.append((s, ref if mode == 'past' else e)); i += 1; continue
        if w in ('next', 'upcoming', 'coming') and i + 1 < len(toks):     # future windows: start after today -> the caller hands off
            nx = toks[i + 1].group(0)
            if mode != 'past':      # calendar records may read the future: whole periods (spec v1.8 10g.19)
                if nx == 'week': a = ref + timedelta(days=7 - ref.weekday()); out.append((a, a + timedelta(days=6))); i += 2; continue
                if nx == 'weekend': a = ref + timedelta(days=(5 - ref.weekday()) % 7 or 7); out.append((a, a + timedelta(days=1))); i += 2; continue
                if nx == 'month':
                    a = (ref.replace(day=1) + timedelta(days=32)).replace(day=1); out.append((a, a.replace(day=cal.monthrange(a.year, a.month)[1]))); i += 2; continue
                if nx == 'year': out.append((date(ref.year + 1, 1, 1), date(ref.year + 1, 12, 31))); i += 2; continue
                if nx.isdigit() and w_(i + 2).rstrip('s') in ('day', 'week', 'month'):
                    n = int(nx); u = w_(i + 2).rstrip('s')
                    e = ref + timedelta(days=n) if u == 'day' else ref + timedelta(weeks=n) if u == 'week' else _months_back(ref, -n)
                    out.append((ref + timedelta(days=1), e)); i += 3; continue
                if nx in _MONTHS:
                    mo = _MONTHS[nx]; yy = ref.year if mo > ref.month else ref.year + 1
                    out.append((date(yy, mo, 1), date(yy, mo, cal.monthrange(yy, mo)[1]))); i += 2; continue
            if nx == 'weekend': d = ref + timedelta(days=(5 - ref.weekday()) % 7 or 7); out.append((d, d)); i += 2; continue   # future marker
            if nx in ('week', 'month', 'year', 'days', 'weeks', 'months'):
                d = ref + timedelta(days=7 - ref.weekday()) if nx == 'week' else (ref.replace(day=1) + timedelta(days=32)).replace(day=1) if nx == 'month' else date(ref.year + 1, 1, 1) if nx == 'year' else ref + timedelta(days=1)
                out.append((d, d)); i += 2; continue
            if nx in _WEEKDAYS:
                d = ref + timedelta(days=(_WEEKDAYS[nx] - ref.weekday()) % 7 or 7); out.append((d, d)); i += 2; continue
            if nx in _MONTHS:
                mo = _MONTHS[nx]; yy = ref.year if mo > ref.month else ref.year + 1; d = date(yy, mo, 1); out.append((d, d)); i += 2; continue
        if w in ('start', 'beginning') and i + 3 < len(toks) and toks[i + 1].group(0) == 'of' and toks[i + 3].group(0) in ('year', 'month', 'week'):
            unit = toks[i + 3].group(0)       # "start/beginning of the year|month|week" -> that period's first day
            d = date(ref.year, 1, 1) if unit == 'year' else ref.replace(day=1) if unit == 'month' else ref - timedelta(days=ref.weekday())
            out.append((d, d)); i += 4; continue
        if w == 'weekend':   # "this weekend" = Saturday of this week -> now; otherwise the most recent complete Sat-Sun (v1.8 10g.3)
            if i > 0 and toks[i - 1].group(0) == 'this':
                sat = ref - timedelta(days=ref.weekday() - 5); out.append((sat, ref if mode == 'past' else sat + timedelta(days=1))); i += 1; continue
            sun = ref - timedelta(days=(ref.weekday() - 6) % 7 or 7)
            if w_(i + 1) == 'before' and w_(i + 2) == 'last': sun -= timedelta(days=7); i += 2      # "the weekend before last"
            out.append((sun - timedelta(days=1), sun)); i += 1; continue
        if w in ('yesterday', 'yday'):   # "yesterday night" is the night that started yesterday, i.e. last night; "yday" is its short form (10d)
            if i + 1 < len(toks) and toks[i + 1].group(0) == 'night': out.append((ref, ref)); i += 2; continue
            d = ref - timedelta(days=1); out.append((d, d)); i += 1; continue
        if w in _MONTHS:
            mo = _MONTHS[w]; day = end = year = sub = None; j = i + 1; k0 = i        # k0: the first token of the day's date
            if day_of(j): day = day_of(j); j += 1
            if day and w_(j) in _RANGE_WORDS and day_of(j + 1) and w_(j + 2) not in _MONTHS: end = day_of(j + 1); j += 2   # "Sept 7 to 13"
            elif day and j < len(toks) and dash(j - 1, j) and day_of(j) and w_(j + 1) not in _MONTHS: end = day_of(j); j += 1  # "Sep 7-13"
            if w_(j).isdigit() and len(w_(j)) == 4: year = int(w_(j)); j += 1
            elif w_(j) == 'of' and w_(j + 1).isdigit() and len(w_(j + 1)) == 4: year = int(w_(j + 1)); j += 2               # "January of 2025"
            elif w_(j + (w_(j) == 'of')) in ('last', 'this') and w_(j + (w_(j) == 'of') + 1) == 'year':                     # "September (of) last year" = 2025-09:
                year = ref.year - (w_(j + (w_(j) == 'of')) == 'last'); j += 2 + (w_(j) == 'of')                              # the year as stated (10g.3; bare "last year" = 2025)
            if day is None and i > 0:
                k = i - 2 if w_(i - 1) == 'of' and day_of(i - 2) else i - 1                                                # "the 17th of August"
                if day_of(k) and not (w_(k - 1) in ('last', 'past', 'previous') ):
                    day = day_of(k); k0 = k
                    if w_(k - 1) in _RANGE_WORDS and day_of(k - 2): end, day = day, day_of(k - 2)                      # "from 1 to 10 September"
                    elif w_(k - 1) == 'and' and w_(k - 3) == 'between' and day_of(k - 2): end, day = day, day_of(k - 2)  # "between 1 and 15 July"
                    elif k > 0 and dash(k - 1, k) and day_of(k - 1): end, day = day, day_of(k - 1)                       # "1-14 September"
            if day is None and w_(i - 1) == 'of' and (w_(i - 3), w_(i - 2)) in _SUBPERIOD: sub = _SUBPERIOD[(w_(i - 3), w_(i - 2))]
            y = year or ref.year
            if day:
                end = end or day
                if end < day: i = j; continue                                                                             # not a range
                ml = lambda yy: cal.monthrange(yy, mo)[1]
                if max(day, end) > cal.monthrange(2024, mo)[1]: raise InvalidDate(w)     # no such day in any year: "February 30th", "April 31"
                if not year: y = _undated_year(mo, end if mode == 'next' else day, ref, mode)   # "Feb 29" with no year: that year's, as "29.02"
                if max(day, end) > ml(y): raise InvalidDate(w)                           # "Feb 29 2025", "Feb 29" in 2026
                d = date(y, mo, min(day, ml(y)))
                if end > day: out.append(_DayRange((d, date(y, mo, min(end, ml(y))))))     # "Sept 7-13", "1-14 September": one explicit range
                else:
                    if night(k0, j): d += timedelta(days=1); j += w_(j) == 'night'
                    out.append((d, d))
            else:   # current month = month to date; "last <this month>" = last year's (v1.8 10g.3)
                s = date(y, mo, 1); last_word = i > 0 and toks[i - 1].group(0) == 'last'
                if not year and (mode == 'past' or last_word) and (s > ref or (mo == ref.month and last_word)): s = date(y - 1, mo, 1)
                elif not year and mode == 'next' and mo < ref.month and not last_word: s = date(y + 1, mo, 1)
                last = date(s.year, mo, cal.monthrange(s.year, mo)[1])
                cap = (lambda d: min(d, ref)) if mode == 'past' or last_word else (lambda d: d)
                if sub:     # "first/second half of", "first/last week of" <month>
                    a = last + timedelta(days=sub[0] + 1) if sub[0] < 0 else s.replace(day=sub[0])
                    b = s.replace(day=min(sub[1], last.day))
                    out.append((a, cap(b) if a <= ref else b))
                else: out.append((s, cap(last)))
            i = j; continue
        if w.isdigit() and len(w) == 4 and 1900 <= int(w) <= 2100 and not (i > 0 and toks[i - 1].group(0) in _MONTHS):
            y = int(w); out.append((date(y, 1, 1), date(y, 12, 31)))
        i += 1
    return out


_NUM = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
        'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'a': 1, 'an': 1, 'couple': 2, 'few': 3}
# A number in words (also for the decoder's "N units ago" / "next N" patterns): 1-19, a tens word with an optional unit ("forty-five",
# "twenty one"), or (a/one) hundred (and ...).
_ONES = 'one|two|three|four|five|six|seven|eight|nine'
_BELOW_100 = rf'(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-]+(?:{_ONES}))?|{_ONES}|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)'
NUM_WORDS = rf'(?:(?:(?:a|one)[\s-]+)?hundred(?:[\s-]+(?:and[\s-]+)?{_BELOW_100})?|{_BELOW_100})'


def num_words(s):
    """The value of a number in words ("sixty", "forty-five", "twenty one", "a hundred and twenty") or of a count word ("a", "couple",
    "few"); None for anything else."""
    s = s.lower().strip()
    if re.fullmatch(r'(?:a\s+)?couple(?:\s+of)?', s): return 2       # "a couple of months ago" (two; "a few" is no count, 10g.24 names none)
    if s in _NUM: return _NUM[s]
    if not re.fullmatch(NUM_WORDS, s): return None
    n = 0
    for w in re.split(r'[\s-]+', s):
        if w == 'hundred': n = (n or 1) * 100
        elif w != 'and': n += _NUM[w]
    return n


def parse_rolling(text):
    """(amount, unit) from "past/last/previous N days|weeks|months|years" (and "past week"), else None. Calendar arithmetic only."""
    text = _expand(text)
    m = re.search(r'\b(?:past|last|previous|prior|trailing)\s+(?:(\d{1,3}|' + NUM_WORDS + r'|[a-z]+)\s+)?(?:of\s+)?(day|week|month|year)s?\b', text.lower())
    if not m: return None
    n = m.group(1); unit = m.group(2)
    if n is None:   # "last week" is calendar; "past week" and "over/in/during/within the last week" are rolling (spec item 10, 10g.3)
        return (1, unit) if re.search(r'\bpast\s+' + unit + r'|\b(?:over|in|during|within)\s+the\s+last\s+' + unit, text.lower()) else None
    amount = int(n) if n.isdigit() else num_words(n)
    return (amount, unit) if amount else None


def parse_time_span(text, ref, mode='past'):
    spans = _dates_in(text, ref, mode)
    if not spans: return None
    return spans[0][0], spans[-1][1] if len(spans) > 1 else spans[0][1]


_RANGE_JOIN = re.compile(r'\s+(?:to|through|thru|till|til|until|and)\s+|\s*[-\u2013\u2014]\s*', re.I)


def parse_range(text, ref, mode='past'):
    """(start, end) when the date syntax of text is one explicit range of days, else None: two single days joined by to/through/till/
    until or a dash, or by "and" after "between" ("Monday to Wednesday", "Mon-Wed", "the 3rd to the 9th", "between the 3rd and the
    9th", "01.09 - 10.09"), or a month with a day range ("Sept 3-9", "3-9 September"). Each end is read as that single day (10g.3:
    weekdays 1-7 days back; a day of month the most recent such day), so an end before the start is returned as is and the caller
    hands off (the spec does not settle it). A list ("Monday and Wednesday") or a period ("last week") is no range."""
    spans = _dates_in(text, ref, mode)
    if len(spans) == 1 and isinstance(spans[0], _DayRange): return tuple(spans[0])
    if len(spans) != 2 or any(s != e for s, e in spans): return None
    def one_day(t):
        try: x = _dates_in(t, ref, mode)
        except ValueError: return False
        return len(x) == 1 and x[0][0] == x[0][1]
    for j in _RANGE_JOIN.finditer(text):       # the join must separate the two days ("2026-09-03 and 2026-09-09" is no range)
        left, right = text[:j.start()], text[j.end():]
        if j.group(0).strip().lower() == 'and' and not re.search(r'\bbetween\b', left, re.I): continue
        if one_day(left) and one_day(right): return spans[0][0], spans[1][1]
    return None
