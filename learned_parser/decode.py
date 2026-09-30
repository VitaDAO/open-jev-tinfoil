"""Decode learned-parser outputs into the v2 plan shape (experimental serving path)."""
import json, re, sys, time
import calendar as _cal
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.models.modernbert.modeling_modernbert import MODERNBERT_ATTENTION_FUNCTION
from learned_parser.parse import *
from learned_parser.parse import _months_back, _MONTHS, _WEEKDAYS, _fix_typo, _edits, _dates_in

DATA = Path(__file__).resolve().parents[1] / 'metadata/learned'
MODEL_CODES = {c: c for c in ('model_handoff', 'model_definition', 'model_advice', 'model_judgement', 'model_comparison', 'model_count',
               'model_extreme', 'model_negation', 'model_filter', 'model_other_person', 'model_action', 'model_prediction', 'model_doc_metadata',
               'model_clock_time', 'model_unsupported_period', 'model_mixed_windows', 'model_ambiguous_name', 'model_unknown_metric',
               'model_vague_followup', 'model_non_english', 'model_injection', 'model_crisis', 'model_multi_topic', 'model_chitchat')}
from proposal_binding import canonicalize
from query_plan import NIGHT_METRICS, TOPIC_EXCLUDED
from schema_index import metric_definition, INDEX, CATALOG
from schema_index import ROOT as REPO_ROOT      # the repo's metadata (10g.28 context classes)
INV2CAT = INDEX['inventory_to_catalog']
# TEST ONLY: route the model's "comparison" handoff to the per-clause path (see _select). A plain constant, never read from the environment or
# changed by serving code: only evaluation harnesses set it on the imported module (regress_vs38.py, work/switch_on.py).
TEST_COMPARISON_AS_MIXED = False
# Spec-defined bundles whose members are read together (spec §3 item 10: qualitative sleep -> core 5); never split by confusable pruning.
BUNDLES = [{'total_sleep', 'sleep_efficiency', 'sleep_deep', 'sleep_rem', 'sleep_score'}]
# One analyte under several inventory ids (spec v1.8 10g.1), generated from the app's alias + dedupe tables.
_AG = json.load(open(DATA / 'analyte_groups.json'))
SPELLING = {m: set(g) for g in _AG['spelling'].values() for m in g}
_SPLITS = list(_AG['unit_split'].values()) + list(_AG['sampling_split'].values())   # 10g.1 unit variants, 10g.27 glucose
UNIT_SPLIT = {g[0]: set(g) for g in _SPLITS}          # legacy/bare id -> its variants (read every present one; a variant stays specific)
ANALYTE = {m: min(g) for g in list(_AG['spelling'].values()) + _SPLITS for m in g}   # id -> one name per analyte
# A measured metric and its self-reported profile twin (spec v1.8 10g.2): bare names read the metric.
TWINS = {'weight': 'weight_kg', 'height': 'height_cm'}
BUNDLE_DEFS = {k: v for k, v in json.load(open(DATA / 'bundles.json')).items() if not k.startswith('_')}
_SRC = json.load(open(DATA / 'source_aliases.json'))
SOURCE_ALIASES = _SRC['aliases']
COMMON_ALIASES = set(_SRC['common_words'])     # bind only when tagged as a source
_words = lambda ws: re.compile(r'(?<![\w-])(?:' + '|'.join(re.escape(w) for w in sorted(ws, key=len, reverse=True)) + r')(?![\w-])', re.I)
NOT_INGESTED = _words(_SRC['not_ingested'])     # brands Vita does not ingest (10g.5)
# A source named after a negation or exclusion cue asks to leave that source out ("all but Garmin steps", "HRV except Whoop", "not from
# Oura", "non-Garmin", "Oura-free"): a restriction on the data selection, never a source filter (10g.28 class 4). The cue must end at most
# 3 words before the name, inside its sentence; a bare "not" only right before it. "since I switched to X" has no cue (10g.31).
SOURCE_EXCLUSION = re.compile(r"(?:\b(?:all|anything|everything) but|\bexcept|\bexclu(?:de[sd]?|ding|sive of)|\bnot from|\bwithout|\bw/o|\bignor(?:e[sd]?|ing)"
                              r"|\bskip(?:s|ped|ping)?|\bother than|\bbesides|\b(?:apart|aside) from|\bbut not|\binstead of|\brather than|\bminus|\bleav(?:e|ing) out"
                              r"|(?:n't|\bnot) (?:include|use|count|want))(?:[^\w.?!;]+[\w'/-]+){0,3}?[^\w.?!;]+$"
                              r"|\bnot(?:[^\w.?!;]+(?:from|by|on|via|using|with|through|the|my|any))*[^\w.?!;]+$", re.I)
# Food-intake wording (spec v1.8 10g.30): the word lists are data (food_intake.json).
_FOOD = json.load(open(DATA / 'food_intake.json'))
FOOD_EXEMPT, FOOD_ALWAYS, FOOD_CAL = _words(_FOOD['not_intake']), _words(_FOOD['always']), _words(_FOOD['calorie_words'])
FOOD_WITH_CAL, FOOD_BURN = _words(_FOOD['with_calories']), _words(_FOOD['burn_words'])
FOOD_PATTERNS = [re.compile(p.replace('{nutrients}', '|'.join(map(re.escape, _FOOD['nutrients']))), re.I) for p in _FOOD['patterns']]
INTAKE_READS = set(_FOOD['intake_reads'])
# 'since <event>' anchors (spec v1.8 10g.29, 10g.31): the word lists are data (since_anchors.json).
_SA = json.load(open(DATA / 'since_anchors.json'))
SA_MEDICAL, SA_ACCOUNT, SA_RESTART, SA_DEVICE_VERB = (_words(_SA[k]) for k in ('medical', 'account_start', 'restart', 'device_start_verbs'))
SA_BACKREF, SA_DATA = re.compile(_SA['back_reference'], re.I), [re.compile(p, re.I) for p in _SA['data_start']]
SA_REPLACE, SA_VERSION = _words(_SA['replacement']), re.compile(_SA['device_version'], re.I)   # same-brand replacement (10g.32)
SA_INTEGRATION = _words(_SA['integration'])     # connected/synced/paired/linked: not a device start (10g.32 item 4)
SA_DEVICES = {a: v for a, v in SOURCE_ALIASES.items() if v not in set(_SA['lab_sources'])}
SINCE_PHRASE = re.compile(r'\bsince\b((?:[^,;:.?!\n]|(?<=\d)\.(?=\d))*)', re.I)     # a dot between digits is part of a date ("since 01.08")
# Open-ended starts other than "since" (spec v1.8 10g.35 item 3 reads "since <period>" from its start to now): a cue before the date
# ("starting in", "going back to", "from the start of") or an open end after it ("onwards", "to date", "until now"). A bare "from X"
# is X's own window ("my sleep from last night"); "(go) back to X" and a fronted "Going back to X, ..." return to X ("go back to June").
OPEN_CUE = r"(?:since|starting(?:\s+(?:from|in|on|at|with))?|beginning(?:\s+(?:from|in|on|at|with))?|going back to|dating back to|as far back as|from(?:\s+the\s+(?:start|beginning)\s+of)?)"
OPEN_END = r"(?:onwards?|forward|to date|(?:until|till|til|to|up to|up until|through|thru)\s+(?:now|today|the present|present|date))"
OPEN_LEAD, OPEN_BEFORE = re.compile(rf"^\s*{OPEN_CUE}\s+", re.I), re.compile(rf"\b{OPEN_CUE}\s*$", re.I)
OPEN_TAIL, OPEN_AFTER = re.compile(rf"\s+(?:{OPEN_END}|on)\s*$", re.I), re.compile(rf"^\s*(?:{OPEN_END}\b|on\s*(?=[,;:.?!\n]|$))", re.I)


def open_start(before, t, after):
    """The date an open-ended start reads from, or None (10g.35 item 3): "starting from January", "March onwards", "June to date",
    "going back to March". t is the tagged time span; before/after, the rest of its sentence. A bare "from X" is X's own window."""
    lead = OPEN_LEAD.match(t); t = t[lead.end():] if lead else t
    tail = OPEN_TAIL.search(t); x = (t[:tail.start()] if tail else t).strip()
    head = before + (lead.group(0) if lead else ''); cue, end = OPEN_BEFORE.search(head), tail or OPEN_AFTER.match(after)
    cue, end = (cue.group(0).strip().lower() if cue and not (cue.group(0).lower().startswith('going') and not re.search(r'\w', head[:cue.start()])) else '',
                end.group(0).strip().lower() if end else '')
    if not ((cue and cue != 'from') or (end and (end != 'on' or cue == 'from'))): return None
    check = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', 'DATE', x)
    if not x or re.search(r"\b(?:to|until|till|til|through|thru|and|this|current|so far)\b|[-–]", check, re.I) or parse_rolling(x) or UNITS_AGO_ANY.search(x): return None
    return x


WEEKDAY_END = re.compile(r'\b(?:' + '|'.join(sorted(_WEEKDAYS, key=len, reverse=True)) + r')(\s+(?:last|this|lst))?$', re.I)   # a tagged span ending in a weekday
SINCE_END = re.compile(r'\s+(?:and|but|or|vs\.?|versus|compared|while|whereas|plus|so)\s.*$|\s+[-–—]\s.*$', re.I)
# Named groups (spec v1.8 10g.16 workout heart rate, 10g.18 panels): the group item expands to its members, exempt from the breadth cap.
PANELS = {k: v for k, v in json.load(open(DATA / 'panels.json')).items() if not k.startswith('_')}
CAL_BASIS = json.load(open(DATA / 'calendar_basis.json'))
APP_ANALYTES = json.load(open(DATA / 'app_analytes.json'))['terms']
AREAS = {k: v for k, v in json.load(open(DATA / 'areas.json')).items() if not k.startswith('_')}   # full areas: Vita's presentation categories (10g.34 item 5)
# Metrics read by the night they end on (10g.4; 10f item 5: the sleep area, HRV, RHR, SpO2): "tonight" is scored tomorrow, a future day (10b).
NIGHT_READ = set(AREAS['sleep']) | {'heart_rate_variability', 'resting_heart_rate', 'spo2'}
# List separators inside one tagged mention run ("HDL, LDL and triglycerides"), split in Parser.split_run.
SEPARATORS = {',', ';', '/', '&', '+', 'and', 'or', 'plus'}
# A clause boundary strong enough to carry its own operation/window: a comma/semicolon, or a connector that restarts a noun phrase.
# A comma alone separates list items ("DHEA sulfate, free T4 since June"), so a comma needs a connector or a new "my ..." after it.
# "and" before a new question ("What's my TSH and what was my RHR last week?") restarts a clause too.
CLAUSE_BOUNDARY = re.compile(r'\s*[,;]\s*(?:(?:and|plus|also|then)\s+)+|\s*[,;]\s*(?=my\b)|\s*[.?!]+\s+(?:(?:and|plus|also|then)\s+)*'
                             r'|\s+(?:and|plus|also|\+)\s+(?:(?:then|also)\s+)?(?=my\b)|\s+and\s+(?:also|then)\s+|\s+and\s+(?=(?:what|how)\b)', re.I)
# The present value asked for by name ("what's my TSH"), item 9: latest; a past "what was my X" may share the other clause's window.
WHAT_NOW = re.compile(r"\bwhat(?:'s|s|\s+is|\s+are)\b|\bwhats\b", re.I)
LATEST_WORDS = re.compile(r'\b(?:latest|most recent|newest|current|currently|right now|at the moment)\b', re.I)
LATEST_OP = re.compile(r'\b(?:latest|most recent(?:ly)?|newest|right now|currently|at the moment|current(?!\s+(?:week|month|year|period|day|quarter|weekend)))\b|\S\s+now\W*$', re.I)   # the operation rule's latest words ("most recently" too)
# Qualitative sleep with no window (spec v1.8 10g.13): the tense picks the window. A present perfect ("how has my sleep been", "how
# have I slept") is a state; "lately/recently/these days" are state words, not windows.
SLEEP_PERFECT = re.compile(r"\b(?:have|has)\s+(?:\w+\s+){0,2}?(?:had|slept|been)\b|'ve\s+(?:\w+\s+){0,2}?(?:had|slept|been)\b", re.I)
SLEEP_PAST = re.compile(r"\b(?:did|was|were|slept|had)\b|\bhow'?d\b", re.I)
SLEEP_STATE = re.compile(r"\b(?:is|are|am|been|do|does|doing|sleeping)\b|\b(?:how|what|where)'?s\b|\b(?:it|that|sleep|there)'s\b|'(?:re|m)\b", re.I)
# Recency words on the qualitative bundle (spec v1.8 10g.32): latest words mean last night, state words the 30-day trend.
BUNDLE_LATEST = re.compile(r'\b(?:latest|most recent|newest)\b|\blast(?=\s+sleep\b)', re.I)   # a bare "last" before the sleep noun ("my last sleep"; "last night/week" are windows)
BUNDLE_STATE = re.compile(r'\b(?:right now|now|currently|at the moment|these days|nowadays)\b', re.I)
# Period words ask for the current state (10g.32 item 9): the latest value of a sparse metric, the 30-day trend of a daily one
# ("how has ... been" wording stays a trend).
STATE_DAYS = re.compile(r'\b(?:these days|nowadays|lately|recently|of late)\b', re.I)
# Change or history of a profile value (spec v1.8 10g.2, 10g.20): the profile is a snapshot, so it cannot be read.
PROFILE_SERIES = re.compile(r'\b(?:trends?|trending|history|historical|over time|graph|chart|plot|progress|changed?|changing|evolution)\b', re.I)
# Plural result nouns ask for a series (spec v1.8 10g.15): "haemoglobin levels", "my HRV numbers" -> trend unless latest is explicit.
TREND_NOUNS = re.compile(r'\b(?:levels|results|numbers|measurements|readings|values|trend(?:s|ed|ing)?|history|graph|chart|progress)\b', re.I)
# "show/pull up/see my X" with no window keeps trend with the default window (spec v1.8 §3 item 9 notes).
# "the sleep recovery contributor to my resilience": the metric named after a contributor/component word is the context of the read,
# not a second read (as a panel name before one member, 10g.34 item 4).
CONTRIB = re.compile(r"\b(?:contributors?|components?|parts?|drivers?|factors?|sub-?scores?)\s+(?:to|of|in|for|behind)\s+(?:my\s+|the\s+|your\s+)?([a-z][\w\s'-]{1,40}?)\s*(?:[?.!,;]|$)", re.I)
# A read that refers back to an event in a context sentence ("I did a charity swim in July. What was my HRV that day?").
CTX_BACKREF = re.compile(r"\b(?:that (?:day|night|week|weekend|month|time|morning|evening|date)|then|since then|the (?:day|week) (?:after|before)|the (?:next|following|previous) day|during (?:the|that|my) \w+|afterwards?|after that|before that|on that (?:day|date))\b", re.I)
# A side sentence states a dated event only when it is first-person, asks nothing (no question, request or correction wording) and
# says something beyond its time phrase, pronouns and hedges: "My half marathon was on the 14th." is an event; "Only for this week.",
# "I want to see it for the past month." and "It was last Sunday I think." are the read's window.
CTX_SUBJECT = re.compile(r"\b(?:i|i'm|im|i've|ive|i'd|we|we're|we've|my|our)\b", re.I)
CTX_REQUEST = re.compile(r"\?|\b(?:can|could|would|will)\s+(?:you|u|i|we)\b|\blet'?s\b|\b(?:want|wanna|need|like|prefer|interested|curious|ask|asking|asked|talking|mean|meant|thinking|wondering|looking|care|question|actually|instead|rather|sorry|correction|show|give|make|do|pull|check|see|view|look|tell|display|plot|graph|chart|compare|include|exclude|use|focus|limit|restrict|filter|narrow|change|switch|zoom|only|just)\b", re.I)
CTX_HEDGE = re.compile(r"\b(?:i|i'm|im|it|it's|its|was|is|were|be|been|think|guess|believe|maybe|probably|perhaps|um|uh|yes|yeah|no|a|an|on|at|from|to|until|since|or|thanks|thank|you|my|our)\b", re.I)


def side_event(s, times=()):
    """A side sentence states a dated event (spec v1.8 10g.28): first-person, asks nothing, and says more than its time phrases
    (s-relative spans), pronouns and hedges."""
    rest = ''.join(' ' if any(a <= i < z for a, z in times) else c for i, c in enumerate(s))
    return bool(CTX_SUBJECT.search(s) and not CTX_REQUEST.search(s) and CTX_HEDGE.sub(' ', FOLLOW_FILLER.sub(' ', re.sub(r"[^\w\s']", ' ', rest))).split())


# A read with no window of its own is re-read without the event only when it asks for a present value ("What's my weight?"); a past
# or open read ("How did I sleep?", "Show my glucose.") may mean the event's date.
PRESENT_READ = re.compile(r"\b(?:what's|whats|what is|what are|what're|how's|hows|how is|how are|where's|where is)\b|\b(?:current(?:ly)?|right now|at the moment|now|latest|most recent|newest)\b", re.I)
# A date bound inside the read restricts its readings ("steps in September after the 15th", "HRV from last month before the 10th"): a
# filter on the data selection (spec v1.8 10g.28 class 4, "before/after <date>") that no plan window states. Hands off.
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DAY_WORD = r"(?:" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")\b"     # every weekday spelling the date parser reads (one lexicon: "weds", "thur")
DATE_BOUND = re.compile(rf"\b(?:before|after|prior to)\s+(?:the\s+)?(?:\d{{1,2}}(?:st|nd|rd|th)\b|{_MON}\s+\d{{1,2}}\b|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON}(?!\w)"
                        r"|\d{4}-\d{1,2}-\d{1,2}|" + _DAY_WORD + ")", re.I)
# A bound stated with a period is no window (spec v1.8 10g.28 class 4, "before/after <date>" applied to the readings; an open start is 10g.29):
# "steps before June", "after March", "before last week", "until Tuesday", "up to last week", "till June", "through March". "until/through X" that closes a
# range ("from Monday until Wednesday", "Monday through Wednesday") has a date before it and is no bound; "the day before yesterday" and "the week before
# last" are absolute dates the parser reads.
_BOUND_MON = r"(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b\.?"
_BOUND_DAY = _DAY_WORD
_BOUND_TIME = rf"(?:the\s+)?(?:(?:last|this|previous|past|next)\s+(?:week|month|year|weekend|night|{_BOUND_MON}|{_BOUND_DAY})|{_BOUND_MON}|{_BOUND_DAY}|(?:19|20)\d{{2}}\b)"
BOUND_FILTER = re.compile(rf"(?<!day\s)(?<!night\s)(?<!week\s)(?<!month\s)(?<!year\s)(?<!weekend\s)\b(?:before|after|prior\s+to)\s+(?:(?:the\s+)?(?:yesterday|today)\b|{_BOUND_TIME})", re.I)
# A vague qualifier on a named period is no window (10g.3 defines whole months and single days; the spec is silent -> handoff). "the last week of August" and
# "the first half of March" are parts the parser reads (singular week, half), not these.
VAGUE_PERIOD = re.compile(rf"\b(?:mid|early|late|around|approximately|roughly|towards?|(?:the\s+)?(?:middle|end)\s+of|(?:the\s+)?(?:first|last)\s+(?:few\s+)?(?:weeks|days)\s+of)[-\s]+"
                          rf"(?:the\s+)?(?:(?:this|last|next)\s+)?(?:{_BOUND_MON}|{_BOUND_DAY}|\d{{1,2}}(?:st|nd|rd|th)\b|week\b|month\b|year\b)", re.I)
BOUND_OPEN = re.compile(rf"\b(?:until|till|til|up\s+to|up\s+until|through|thru)\s+{_BOUND_TIME}", re.I)
# The rest of a current period is future ("active calories later this week", "steps for the rest of the day"): observations cannot be read
# ahead (spec v1.8 §3 item 10b); calendar records can (10g.19).
LATER = re.compile(r"\blater\s+(?:this|today|tonight|on\s+today)\b|\b(?:rest|remainder)\s+of\s+(?:the|this|today)\b", re.I)
# An explicit count in a rolling window ("the last 45 days", "the last sixty days"): the window must come from the date parser, never from
# the amount head's closest class (v13-0404: "strain over the last sixty days" read as 30 days).
SPOKEN_N = re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
                      r"|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|dozen)\b", re.I)
CHANGE_CUE = re.compile(r"\bhow (?:has|have|did|is|are|'s) (?:\w+ ){0,3}?(?:chang\w*|evolv\w*|progress\w*|improv\w*|develop\w*|mov(?:ed|ing))\b", re.I)   # change over time: a series (10g.15)
SHOW_VERBS = re.compile(r"\b(?:show(?:ing)?(?!\s*(?:[?.!]|$))|pull (?:up|out)|bring up|display|see|view|plot|graph|chart|visuali[sz]e)\b", re.I)   # not a result verb ("what did my last LDL test show?")
# A panel or area name narrowed to one member: a dash, colon, "specifically", "just", "only" or "namely" between them (10g.34 item 4).
NARROW_TEXT = re.compile(r"^\W*[\w'. -]{2,30}?\s*[:\u2013\u2014-]\s*(?:just|only|specifically|namely)\s+(?:my\s+|the\s+)?\w", re.I)   # a label (possibly misspelt) narrowed to a member
# 10g.36 item 5: "what did my latest lipid panel say about LDL" / "show for" / "tell me about" after a panel narrows too.
NARROW_SAY = r'(?:say|says|said|show|shows|showed|tell\s+me|tells\s+me|told\s+me)\s+(?:about|for|on|regarding)'
NARROW_SAY_RX = re.compile(NARROW_SAY, re.I)
NARROW_GAP = re.compile(r'(?:\s+(?:results?|history|levels|numbers|values|readings|trends?|data))?\s*(?:(?:[-\u2013\u2014]+|:)\s*(?:(?:specifically|just|only|namely)\s+)?|,?\s*(?:specifically|just|only|namely)\s+|,?\s*' + NARROW_SAY + r'\s+)(?:(?:my|the)\s+)?', re.I)
# A panel/area name and one of its members joined as list items ("body composition and fat percentage") are two items, not a narrowing;
# a new clause between them ("... and what was my REM?") is read clause by clause. Any other joint (a bare comma, "focusing on",
# "in particular", a bracket, "the LDL part of my lipids") is not a settled form of 10g.34 item 4: hand off.
SRC_BEFORE = re.compile(r'(?<![\w-])(?:' + '|'.join(re.escape(a) for a in sorted(SOURCE_ALIASES, key=len, reverse=True)) + r')\s+$', re.I)      # a device name right before the group word
# Words that leave a request a bare named panel ("CBC without differential", "yo fam my cbc?"): the panel is its latest read (10g.18, "my lipid panel").
BARE_WORDS = {'my', 'the', 'a', 'what', "what's", 'whats', 'is', 'are', 'yo', 'hey', 'hi', 'please', 'pls', 'thanks', 'thank', 'you', 'ok', 'okay', 'so', 'um', 'uh', 'fam', 'bro'}
LIST_GAP = re.compile(r'\s*,?\s*(?:and|plus|&|\+|as well as|along with|also)\s+(?:(?:my|the|your|our)\s+)*', re.I)
CHAIN_GAP = re.compile(r'\s*[,;/]?\s*(?:(?:and|plus|&|\+|as well as|along with|also|or)\s+)?(?:(?:my|the|your|our)\s+)*', re.I)
CLAUSE_WORDS = re.compile(r"\b(?:what(?!\s+about)|what's|whats|how(?!\s+about)|which|when|where|why|who|show|tell|give|pull|check|see|list|display|plot|can|could|is|are|was|were|do|does|did|have|has)\b", re.I)
# A full-area ask ("everything about my sleep", "full breakdown of my sleep") reads the whole area (10d; 10g.34 item 5).
FULL_AREA = re.compile(r"\b(?:everything|all)\s+(?:about|on|regarding)\s+(?:my\s+)?sleep\b|\b(?:full|complete|whole|entire)\s+(?:breakdown|picture|overview|report|rundown)\s+"
                       r"(?:of|on|for)\s+(?:my\s+)?sleep\b|\b(?:full|complete)\s+sleep\s+(?:breakdown|picture|overview|report|rundown)\b|\ball\s+(?:of\s+)?my\s+sleep\s+(?:data|metrics|stats|numbers)\b", re.I)
# Extremes over the whole history ("my record for daily steps", "personal best", "all-time high", "highest HRV ever") are the extreme
# handoff family; windowed extremes ("highest HRV this week") stay with the model (the spec has no ruling on them). "record" is the
# noun only after a determiner ("my record for steps"), not the verb ("what did my Garmin record for HRV").
EXTREME = re.compile(r"\b(?:my|the|a|your|new|all[-\s]time)\s+record\s+(?:for|of)\b|\bpersonal\s+(?:best|record)s?\b|\bPBs?\b|\ball[-\s]time\s+(?:high|low|best|worst|max(?:imum)?|min(?:imum)?|record)s?\b"
                     r"|\b(?:highest|lowest|best|worst|longest|shortest|fastest|slowest|most|least)\b(?:\s+[\w'-]+){0,5}?\s+(?:ever|of\s+all\s+time)\b|\bever\s+recorded\b", re.I)
# A discourse marker that starts a new topic: the request stands alone (no follow-up inheritance).
TOPIC_RESET = re.compile(r'\b(?:different|new|another|unrelated|separate)\s+(?:topic|question|subject|thing|matter)\b|\bon (?:another|a different) note\b'
                         r'|\bswitching (?:gears|topics?)\b|\bchanging (?:the )?(?:subject|topic)\b|^\W*(?:separately|unrelated|moving on|next question)\b', re.I)   # 10g.26 reset markers
WHAT_IS = re.compile(r"\b(?:what(?:'s|s| is| was| are| were)|whats)\b", re.I)
# A value question: latest words, a trailing "now", or "my last ... reading/result" (spec item 9, 10g.15).
LATEST_CUE = re.compile(r'\b(?:latest|most recent|newest|current|currently|right now|at the moment)\b|\bnow\W*$|\blast\s+(?:\S+\s+){0,3}(?:reading|result|value|measurement|test|entry)\b', re.I)
SERIES_CUE = re.compile(r'\b(?:average|avg|mean|trend|trends|history|over time|levels|results|numbers|measurements|readings|values|changed|change)\b', re.I)
# "the week before that": the previous turn's window moved back one unit (a window-only follow-up).
BEFORE_THAT = re.compile(r'\b(?:the\s+)?(day|week|month|year)\s+before\s+(?:that|this|then)\b', re.I)
# Follow-ups relative to the prior read (spec v1.8 10g.25): "the previous/prior/preceding <unit>", "the <unit> before (that)",
# "the one before", "the previous N days"; year re-anchoring ("same but last year", "and last year?", "the year before?", "in 2025?").
# "the day before yesterday" and "the <unit> before last" are absolute dates (the date parser), not shifts of the prior read.
# A night is the day unit of a one-night read (spec v1.8 10g.4: a night is named by the day it ended): "the previous night",
# "the night before" shift a one-day window back one day. "the <unit> prior/preceding (that)" says "the <unit> before (that)".
# A weekend is the unit of a Sat-Sun window: "the weekend before" after "last weekend" is the Sat-Sun a week earlier (10g.25, 10g.38 item 1).
PREV_UNIT = re.compile(r'\b(?:the\s+)?(?:previous|prior|preceding)\s+(day|night|week|weekend|month|year)\b|\b(?:the\s+)?(day|night|week|weekend|month|year)\s+(?:before|prior|preceding)\b(?!\s+(?:to\s+)?(?:yesterday|last)\b)|\bthe\s+one\s+before\b', re.I)
PREV_N = re.compile(r'\b(?:the\s+)?(?:previous|prior|preceding)\s+(\d+)\s+(day|week|month)s?\b|\b(?:the\s+)?(\d+)\s+(day|week|month)s?\s+before\b', re.I)
YEAR_REANCHOR = re.compile(r"\bsame\b.{0,30}\b(?:last|previous|prior)\s+year\b|\b(?:a|one)\s+year\s+(?:earlier|before)\b|\bthe\s+year\s+(?:before|prior)\b"
                           r"|^\W*(?:and\s+|what\s+about\s+|how\s+about\s+|now\s+)?(?:for\s+|in\s+)?(?:last\s+year|((?:19|20)\d{2}))\W*$", re.I)
# "the same week in August": which week of August is not defined -> unresolved (10g.25 covers shifts and year re-anchoring only).
SAME_UNIT_IN = re.compile(r'\bsame\s+(?:day|week|weekend|night|days|weeks)\s+(?:in|of|during|for)\b', re.I)
FOLLOW_FILLER = re.compile(r'\b(?:and|what|about|how|show|me|the|then|for|please|pls|same|it|that|this|again|ok|okay|so|now|also|too|but|in|of|period|window|dates|time|one|compared|vs)\b', re.I)
# Same metric, several windows (spec v1.8 10g.38): one read per explicitly stated, resolvable, non-overlapping window, whatever the phrasing.
# The windows of such a comparison besides the tagged ones: a point-in-time word is the value now, one latest read (item 7 "current/now"; the
# point-in-time words of 10g.32 item 9; not the conjunction "now that"); a window stated relative to the one before it ("the week before (that)",
# "the previous month", "the 7 days before", "the one before", item 1; 10g.25); "the same period/days/week last month|year" shifts the window
# before it by that unit and keeps its to-date end (item c). "latest", "most recent" and "newest" are no window: they pick the latest value of
# the read they qualify ("my most recent VO2 max in August" is one read; 10g.15: the latest words win).
CMP_NOW = re.compile(r"\b(?:right\s+now|currently|at\s+the\s+moment|now(?!\s+that\b)|current(?!\s+(?:week|month|year|period|day|quarter|weekend)))\b", re.I)
# A point-in-time word is a window of a comparison only when one of these joins it to the window beside it ("my current RHR vs last month", "HRV
# now and last week", "What's my RHR right now? What about last month?"; item 7); next to a window without one it is discourse or qualifies that
# read ("Show my steps for this week now").
CMP_LINK = re.compile(r"\b(?:and|or|vs|versus|v|compar(?:e|ed|es|ing)|relative\s+to|against|next\s+to|than|plus|what\s+about|how\s+about)\b|[&+/]", re.I)
CMP_REL = re.compile(r"\b(?:the\s+)?(?:previous|prior|preceding)\s+(?:(\d+|" + NUM_WORDS + r")\s+)?(day|night|weekend|week|month|year)s?\b"
                     r"|\bthe\s+(?:(\d+|" + NUM_WORDS + r")\s+)?(day|night|weekend|week|month|year)s?\s+(?:before|prior|preceding)\b(?!\s+(?:to\s+)?(?:yesterday|last)\b)(?:\s+(?:that|this|then|it))?"
                     r"|\bthe\s+one\s+before(?:\s+that)?\b", re.I)
CMP_SAME = re.compile(r"\bthe\s+same\s+(period|days|dates|time|stretch|week|month|weekend)\s+(?:(?:of\s+)?(?:last|the\s+previous|the\s+prior)\s+(week|month|year)"
                      r"|(?:a|one)\s+(week|month|year)\s+(?:ago|earlier|before))\b", re.I)
# Deictic windows the tagger may leave out of a list ("Show my steps last week. Also show this week." with only "week" tagged); not a part of an
# absolute "the week before last week".
CMP_STD = re.compile(r"(?<!before\s)(?<!after\s)\b(?:(?:this|last|previous|current)\s+(?:weekend|week|month|year)|today|yesterday|last\s+night)\b", re.I)
# List words at either end of a tagged window are not part of it ("last month up" of "last month up or down vs the month before").
CMP_EDGE = re.compile(r"^(?:(?:up|down|or|and|vs\.?|versus|compared|to|with|than|then|also|please)\b\W*)+|(?:\W*\b(?:up|down|or|and|vs\.?|versus|compared|to|with|than|then|also|please))+\W*$", re.I)
# What may stand between two windows of one list (item 2: phrasing does not matter), and the words of an elliptical second clause or sentence about
# the same subject that names only a window ("..., and what was it last week?", "? What about ...", ". Also show ...", "and then", item 5).
CMP_JOIN = re.compile(r"(?:[\s,;:.?!&+/\u2013\u2014-]|\b(?:and|or|vs|versus|v|compared?|comparing|compares|to|with|against|relative|next|than|then|also|plus|as|well|opposed|stack(?:s|ed)?"
                      r"|up|down|over|in|on|for|during|from|of|the|my|what|what's|whats|how|how's|about|was|is|were|are|it|its|it's|that|those|show|me|give|pull|see|please|pls|too|did|does|do|looks?)\b)*", re.I)
# The scaffolding of a comparison around its reads, removed from each window's clause (the reads compose; Fable states the comparison): the
# comparison verb, a layout (class 3 refined: a multi-read plan keeps it), "which was higher,", "what's the difference between", "how much did ... change",
# additive words; "from" before a "from <period> to <period>" list. CMP_MORE: the comparative words of a factual comparative (item 6, a), dropped
# (with a yes/no auxiliary before them, CMP_YESNO) only when a clause does not plan with them.
CMP_FRAME = [re.compile(p, re.I) for p in (
    r"\b(?:can|could|would)\s+(?:you|u)\s+(?:please\s+)?(?=compare\b)", r"\b(?:compare|comparing|put)\b",
    r"\bside[-\s]by[-\s]side\b|\bnext\s+to\s+each\s+other\b|\b(?:in|as|into)\s+(?:a|one)\s+(?:table|chart|graph)\b",
    r"\bwhich\s+(?:one\s+)?(?:was|is|were)\s+(?:higher|lower|better|worse|more|less|greater|bigger|smaller)\b\s*,?",
    r"\bwhat(?:'s|s|\s+is|\s+was)\s+the\s+difference\s+(?:between|in)\b", r"\bhow\s+much\s+(?:did|has|have)\b(?=.*\bchang)", r"\bchang(?:e|ed)\b",
    r"\b(?:too|as\s+well|also)\b", r"\bfrom\s*$")]
# A follow-up that compares the prior read with the windows it names (10g.38 item b): "Compare that to 2024", "how does that compare with the past
# 90 days?", "vs last week?", "and compare it with last month", "which was higher, June or July?". The match ends where the windows begin.
# A bare "to/with/against" is no comparison cue: after "Show my steps from Monday", "to Wednesday" ends the range (RANGE_END); it compares only
# after "compare(d)", "how does that compare" or "that/this/it" ("compare that to 2024", "that vs last week").
FUP_CMP = re.compile(r"^\W*(?:(?:and|so|ok|okay|but|now|hmm|also)\W+)*(?:(?:how\s+(?:does|did|do|is|was|would)\s+(?:that|this|it|those|they)\s+(?:compare|stack\s+up|look)"
                     r"|(?:(?:can|could)\s+you\s+)?compare\s+(?:that|this|it|them|those)|(?:and\s+)?(?:that|this|it))\s*(?:(?:compared\s+)?(?:to|with|against)|vs\.?|versus|relative\s+to|next\s+to)"
                     r"|(?:compared\s+(?:to|with|against)|vs\.?|versus|relative\s+to|next\s+to)"
                     r"|which\s+(?:one\s+)?(?:was|is|were)\s+(?:higher|lower|better|worse|more|less)\s*,?)\s+", re.I)
# A follow-up that only says where the prior request's window ends ("Show my steps from Monday" -> "to Wednesday" / "to today"; "steps on Monday"
# -> "to Wednesday?"): it completes that request (10g.26, an elliptical continuation), whose day ends make one span (10g.38 item 3, 10g.36.6).
RANGE_END = re.compile(r"^\W*(?:(?:and|so|ok|okay|but|now|hmm|also)\W+)*(?:up\s+)?(?:to|till|til|until|through|thru)\s+(?P<t>.+?)[\s?.!]*$", re.I)
# "did I sleep more/less/longer" asks how much: total sleep, as "how much did I sleep" reads (the comparison is between the windows).
CMP_SLEEP_MORE = re.compile(r"\b(?:did|do|have|has)\s+i\s+(?:been\s+)?(?:get(?:ting)?\s+)?(?:sleep(?:ing)?|slept)\s+(?:more|less|longer|shorter)\b", re.I)
# The yes/no form of a factual comparative ("is my HRV ...", "did my RHR ...") once its comparative word is dropped: the read itself.
CMP_YESNO = re.compile(r"^\W*(?:(?:so|ok|okay|and|but|hey|hi|um)\s+)*(?:is|was|were|are|has|have|had|did|does|do)\s+(?=(?:my|i)\b)", re.I)
CMP_MORE = re.compile(r"\b(?:higher|lower|better|worse|more|less|greater|up\s+or\s+down|improv(?:e|ed)|gone\s+(?:up|down)|go\s+(?:up|down)|went\s+(?:up|down)|increas(?:e|ed)|decreas(?:e|ed)|dropp?(?:ed)?)\b", re.I)
# Comparisons that stay handoffs (item "Hand off"; 10g.28 classes 5, 6, 9): a relationship or cause; a reference that is no stated window ("than usual",
# "vs my baseline", "compared to before"); another person or a norm; a normative judgement.
CMP_RELATION = re.compile(r"\b(?:affect(?:s|ed|ing)?|effects?\s+(?:of|on)|impact(?:s|ed|ing)?|influenc\w*|because|caus(?:e|es|ed|ing)|due\s+to|correlat\w*|linked\s+(?:to|with)|link\s+between"
                          r"|relat(?:ed|ion|ionship)\s+(?:to|with|between)|relationship|connection\s+between|associated\s+with|driven\s+by|result\s+of)\b", re.I)
CMP_IMPLICIT = re.compile(r"\bthan\s+(?:usual|normal|average|typical|before|ever|expected|it\s+(?:used\s+to|should)|i\s+(?:used\s+to|usually|normally|expected)|my\s+(?:usual|normal|average|typical|baseline|norm))\b"
                          r"|\b(?:vs\.?|versus|compared\s+(?:to|with)|compare[sd]?\s+(?:(?:it|that|this)\s+)?(?:to|with)|relative\s+to|against|next\s+to)\s+(?:my\s+|the\s+|an?\s+)?"
                          r"(?:usual|normal|norm|typical|baseline|before|previously|history|all[-\s]time)\b", re.I)    # not "deviation from baseline" or "my usual RHR" (a metric, 10g.33 item 4)
CMP_AVERAGE_REF = re.compile(r"\b(?:vs\.?|versus|compared\s+(?:to|with)|compare[sd]?\s+(?:(?:it|that|this)\s+)?(?:to|with)|relative\s+to|against|next\s+to|than)\s+(?:my\s+|the\s+|an?\s+)?average\b", re.I)
CMP_NORM = re.compile(r"\b(?:people|others|other\s+(?:users|men|women|people)|everyone|someone|men|women|adults|athletes|population|peers?)\b|\bfor\s+my\s+age\b|\bmy\s+age\b|\bthe\s+average\s+(?:person|man|woman|adult|user|\d+[-\s]year[-\s]old)\b"
                      r"|\b(?:recommend\w*|guidelines?|target|goal|optimal|ideal|healthy\s+range|normal\s+range|reference\s+range)\b"
                      r"|\b(?:wife|husband|partner|brother|sister|son|daughter|mom|mum|mother|dad|father|friend|boyfriend|girlfriend|colleague|coworker|kids?|child)(?:'s|s')?\b", re.I)
CMP_JUDGE = re.compile(r"\b(?:is|are|was|were)\s+(?:that|this|it|these|those)\s+(?:good|bad|ok|okay|fine|normal|healthy|unhealthy|concerning|worrying|alarming|a\s+(?:problem|concern|worry)|too\s+(?:high|low))\b"
                       r"|\bshould\s+i\s+(?:be\s+)?(?:worr\w*|concern\w*)\b|\bworr(?:y|ied|ying)\b|\bconcern(?:ed|ing)\b|\bgood\s+or\s+bad\b|\bhealthy\b|\bnormal\b"
                       r"|\bon\s+track\b|\bred\s+flags?\b|\b(?:good|bad|positive|negative|warning)\s+signs?\b|\b(?:good|safe|optimal|right)\s+range\b|\benough\b"
                       r"|\bdoing\s+(?:well|ok|okay|fine|good|great|alright|badly|poorly)\b|\bthoughts\b|\byour\s+(?:opinion|take|view|verdict)\b|\bwhat\s+do\s+you\s+(?:think|make\s+of)\b", re.I)
# The spec v1.8 10g.28 context classes as the repo publishes them (metadata/context-classes.v1.json, also read by query_selector.unexplained_sentence):
# regex alternatives over canonicalized lower-case text.
CONTEXT = json.loads((REPO_ROOT / 'metadata/context-classes.v1.json').read_text())
_ctx = lambda *names: r'\b(?:' + '|'.join(e for n in names for e in CONTEXT[n]) + r')\b'
_CLINICIAN = '(?:' + '|'.join(CONTEXT['clinician']) + ')'
# Beside a comparison of windows (cmp_guard), in the lists the repo checks the sentence that carries the read with: class 2, medical context (a
# medication, treatment or symptom, or a clinician's involvement: "my doctor asked me to", "for my appointment"; not "doctor appointments", a calendar
# record; 10g.28 class 2 [clarified]); classes 9 and 3, an advice, judgement or explanation ask ("why", "any advice", "should I", "explain"). In any
# other sentence of the request (CMP_CARE_SIDE, CMP_ASK_SIDE) also the lists it checks an extra sentence with: generic medical words, any clinician
# but the user's own job, symptoms that may name a metric, judgement.
CMP_CARE = re.compile(_ctx('medical', 'symptom') + r"|\b(?:my|the|our)\s+(?:[\w-]+\s+)?" + _CLINICIAN + r"\b(?!'?s?\s+(?:appointments?|visits?)\b)"
                      r"|\b(?:for|before|at|after)\s+(?:my|the|our)\s+(?:[\w'-]+\s+)?(?:appointment|consultation|check-?up|visit|follow-?up|physical)s?\b")
CMP_CARE_SIDE = re.compile(_ctx('medical', 'medical_generic', 'care', 'symptom', 'symptom_or_metric')
                           + r"|(?<!i'm a )(?<!i am a )(?<!as a )(?<!i'm an )(?<!i am an )(?<!as an )\b" + _CLINICIAN + r"\b")
CMP_ASK = re.compile(_ctx('judgement_read', 'instruction_read'))
CMP_ASK_SIDE = re.compile(_ctx('judgement', 'instruction_read'))
CMP_PLAIN = re.compile(_ctx('pleasantry', 'not_ask'))       # removed before the classes are matched, as the repo does
CMP_SUPPLEMENT = re.compile(_ctx('not_medication') + r"(?: [a-e]\d*)?(?: (?:pills?|tablets?|capsules?|gummies|powder|shake|drink|supplements?))?")   # plain context (10g.37)
CMP_STYLE = re.compile(_ctx('style'))
# A verb that only introduces a style or language instruction ("answer in Romanian", "reply in a table", "write it briefly") is part of it: beside a
# multi-read plan the instruction keeps the plan (10g.28 class 3 [refined 10g.38]), removed with it from the residue and from each window's clause
# (CMP_STYLE_CLAUSE, raw text). Not "explain" (an explanation ask, class 9).
_STYLE_VERB = r"\b(?:answer|reply|respond|write(?:\s+(?:it|that|this|them))?(?:\s+back)?|say\s+(?:it|that)|give\s+(?:it|that|this|them)(?:\s+to\s+me)?|show\s+(?:it|that|this|them)(?:\s+to\s+me)?)\s+(?:(?:me|please)\s+)?"
CMP_STYLE_ASK = re.compile(_STYLE_VERB + r"(?=" + _ctx('style') + r")")
CMP_STYLE_CLAUSE = re.compile(r"(?:" + _STYLE_VERB + r")?" + _ctx('style'), re.I)
# Discourse fillers and slang that ask for nothing ("... asap", "... ngl", "... lol", "Ta.", "Appreciate it.").
_FILLER = r"asap|lol|lmao|haha|ngl|tbh|fam|rn|ta|thx|ty|pls|plz|please|thanks|cheers|appreciated?(?:\s+it)?|quickly|real\s+quick|again"
# What may stand beside a comparison's reads without asking for anything else (_cmp_residue, with CMP_WHICH): the comparison itself ("by how much",
# "any difference", superlatives over 3+ windows), read verbs and read nouns, and closed-class words. Wh-words, "if", "whether" and the modals of
# advice ("should", "must") are not in it: a clause that keeps any other word asks for something besides the reads.
CMP_REST = re.compile(r"\b(?:by\s+)?how\s+much\b|\b(?:highest|lowest|best|worst|most|least|bigger|smaller|longer|shorter|differen(?:ce|ces|t))\b"
                      r"|\b(?:is|are|was|were|am|be|been|being|do|does|did|has|have|had|can|could|would|will|there|any|some|a|an|the|my|me|i|it|its|they|them|their|this|that|these|those"
                      r"|you|u|your|one|ones|both|each|every|per|and|or|than|vs|versus|v|compared?|compares|comparing|with|to|in|on|at|of|for|from|by|over|during|against|between|up|down|next|relative|then|also|too|so|just|again"
                      r"|show|display|list|see|view|put|lay|give|pull|plot|chart|graph|numbers?|values?|data|readings?|stats|figures|averages?|avg|mean|totals?|sum|results?|levels?"
                      r"|thank|quick\s+question|curious|curiosity|wondering|wonder|" + _FILLER + r")\b|'(?:s|m|re|ve|d|ll)\b|n't\b|[^\w\s]|_", re.I)
# The comparison questions, removed before CMP_FRAME and CMP_MORE take their words: "which (week) was higher", "how do they compare".
CMP_WHICH = re.compile(r"\bwhich(?:\s+(?:one|ones|of\s+(?:them|the\s+two|these|those)|\w+))?\s+(?:was|is|were|are|had|has|did|got)\b"
                       r"(?=[^.?!,;]*\b(?:higher|lower|better|worse|more|less|greater|bigger|smaller|longer|shorter|best|worst|highest|lowest|most|least)\b)"
                       r"|\bhow\s+(?:do|does|did|would)\s+(?:they|these|those|the\s+two|both|it|that|this|them)\s+(?:compare|stack\s+up)\b", re.I)
CMP_LEAD = re.compile(r"^\W*(?:(?:ok|okay|so|well|alright|aight|hey|hi|hello|yo|um|uh|hmm|oh|btw|anyway|good\s+(?:day|morning|afternoon|evening)|quick\s+(?:one|q|question)|real\s+quick"
                      r"|random\s+(?:one|q|question)|one\s+more\s+thing)\b\W*)+", re.I)     # discourse openers of a clause
# A yes/no question about the metric set (_verdict; 10g.38 item 6): an auxiliary before the metric set as subject ("did my HRV", "are my steps"), or
# "how much/far has|did my X" (VERDICT_OPEN); a noun phrase of a change before it, as a question or after "is/has there (been)" ("Any change in my HRV
# this week?", "Has there been any change in my RHR since last week?"; VERDICT_NP; not an existence read: "Any checkups scheduled for next week?",
# "Is there any HRV data for last week?"); or an embedded "if/whether". PLAIN_READ: the words a plain read may add to such a question (existence,
# sync and logging words, read nouns, generic device nouns, closed-class words, fillers: "Is my HRV data from last week available?", "Has my weight
# been logged this month?", "Have my lab results come back yet?", "Did my weight update today?", "Are my steps from Garmin showing up this week?", "Has
# my ferritin been tested this year?", "Have my labs from June been uploaded?"). "on track" is no plain read ("Is my weight on track?", class 9).
_VLEAD = r"^\W*(?:(?:so|ok|okay|hey|hi|yo|um|and|but|also|quick\s+(?:question|one|q))\W+)*"
VERDICT_OPEN = re.compile(_VLEAD + r"(?:(?:is|are|was|were|am|has|have|had|did|does|do)(?:n'?t)?|(?:by\s+)?how\s+(?:much|far)\s+(?:has|have|had|did|does|do))\b", re.I)
VERDICT_NP = re.compile(_VLEAD + r"(?:(?:is|are|was|were|has|have|had)\s+there\s+(?:been\s+)?)?(?:any|some|a|an)\s+(?P<n>[\w-]+)\s+(?:in|to|with|of|on|for)\s+(?:my|the|our)\s+$", re.I)
VERDICT_IF = re.compile(r"\b(?:if|whether)\b", re.I)
PLAIN_READ = re.compile(r"\b(?:there|been|being|be|any|some|all|my|the|our|a|an|it|its|this|that|and|or|in|on|for|from|of|during|over|at|since|between|by|with|within|across|throughout|to"
                        r"|show(?:s|n|ed|ing)?\s+up|updat(?:e|es|ed|ing)|test(?:s|ed|ing)?|check(?:ed)?|import(?:ed|ing)?|receiv(?:ed|ing)|labs?|reports?"
                        r"|yet|still|already|data|available|availability|sync(?:ed|ing)?|record(?:ed|ing)?|log(?:ged|ging)?|(?<!\bon\s)track(?:ed|ing)?|measur(?:ed|ing)|upload(?:ed|ing)?|captur(?:ed|ing)"
                        r"|sav(?:ed|ing)|stor(?:ed|ing)|enter(?:ed)?|come|came|back|arrived|ready|get|got|gotten|show(?:n|ing)?|up\s+to\s+date|readings?|values?|numbers?|results?|entries|entry"
                        r"|levels?|scores?|stats|figures|watch|ring|phone|scale|band|tracker|device|app|" + _FILLER + r")\b|'(?:s|m|re|ve|d|ll)\b|n't\b|[^\w\s]|_", re.I)
# A direction or a change stated of the metric set is a verdict whatever the sentence's opening (_verdict, _direction; 10g.38 item 6): elliptical
# ("HRV up this week?", "Steps down this week?", "HRV lower this week?", "HRV this week: up or down?"), declarative ("My HRV is up this week?", "My sleep
# score improved this week?", "My HRV went up this week, right?"), a fronted alternative ("Up or down: my HRV this week"), a statement to confirm ("Tell
# me my HRV went up this week"), and yes/no questions whose words a plain read may also use ("Did my HRV come back this week?", "Has my weight gotten
# back on track this month?"). Not a direction (VERDICT_PHRASAL and the lookaheads): "up/down" in a phrasal verb or before its object ("pull up my
# HRV", "showing up", "wake-up time", "pull it up"), "up to/until" (a range end); "back" alone ("my HRV 4 weeks back", "go back to June"); "came back
# from/as/at" ("what did it come back at"); a result coming back (VERDICT_RECORD: "Have my lab results come back yet?"); "fall asleep"; climbing as the
# activity; "am I recovered" (the recovery score's own wording, a read). A wh-question asks for the reads ("How recovered am I?", "How is my HRV
# trending?"; CHANGE_CUE, 10g.15) unless it offers the directions ("What's my HRV this week, up or down?", VERDICT_ALT).
VERDICT_CHANGE = re.compile(r"(?<![\w-])(?:up|down)(?![\w-])(?!\s+(?:to|until|till|til|through|thru|my|the|your|a|an|some|all|his|her|their|our)\b)"
                            r"|\b(?:higher|lower|better|worse|improv\w*|worsen\w*|declin\w*|increas\w*|decreas\w*|drop(?:s|ped|ping)?|dip|dipp(?:ed|ing)|ris(?:e|es|en|ing)|rose"
                            r"|fall(?:s|en|ing)?(?!\s+asleep)|fell(?!\s+asleep)|spik(?:e|es|ed|ing)|tank(?:s|ed|ing)?|jump(?:s|ed|ing)?|plummet\w*|soar\w*|surg(?:e|es|ed|ing)"
                            r"|slipp(?:ed|ing)|creep(?:s|ing)?|crept|chang(?:ed|ing)|moved|stabili[sz]\w*|normali[sz]\w*|plateau\w*|rebound\w*"
                            r"|(?<!\bam\si\s)(?<!\bi\sam\s)(?<!\bi'm\s)(?<!\bim\s)recover(?:s|ed|ing)?|on\s+track|back\s+to\s+(?:normal|baseline)"
                            r"|(?:bounc\w*|come|comes|came|coming|got|gotten|getting|get)\s+back(?!\s+(?:from|as|at|with|in)\b)"
                            r"|(?<!go\s)(?<!went\s)(?<!going\s)climb(?:s|ed|ing)?(?!\s+(?:workouts?|sessions?|gym|wall|routes?|stairs|floors)))\b", re.I)
VERDICT_PHRASAL = re.compile(r"\b(?:pull|pul|pulled|bring|brought|look|looked|set|sign(?:ed)?|clean(?:ed)?|wrap(?:ped)?|take|took|keep(?:ing)?|kept|wake|woke|waking|warm"
                             r"|get|gets|got|getting|show(?:s|ed|n|ing)?|sync(?:s|ed|ing)?|add(?:s|ed)?|sum|break|broke|write|wrote|jot|narrow|shut|turn(?:ed)?|scroll|catch"
                             r"|follow|fill(?:ed)?|coming|open|call|dig|load|print|type|run|slow|calm|cool|settle|wind|count|heads)\s+(?:(?:it|them|that|this|those|these)\s+)?$", re.I)
VERDICT_RECORD = re.compile(r"\b(?:results?|labs?|lab\s+\w+|tests?|reports?|panels?|bloodwork|blood\s+work|uploads?|scans?)\b", re.I)
VERDICT_ALT = re.compile(r"\b(?:up|higher|better|increas\w*|improv\w*)\s+or\s+(?:down|lower|worse|decreas\w*|declin\w*|worsen\w*)\b"
                         r"|\b(?:down|lower|worse|decreas\w*|declin\w*)\s+or\s+(?:up|higher|better|increas\w*|improv\w*)\b", re.I)
VERDICT_WH = re.compile(_VLEAD + r"(?:what|what's|whats|how|how's|hows|which|when|where|who|whose|why)\b", re.I)
# The subject of a clause is someone else than the metric set: a clause that opens with a person ("I've been sleeping better, is my HRV ...") or an
# adverbial clause with a person as subject ("HRV since I came back from Lisbon?"); its words say nothing of the metric set.
VERDICT_PERSON = re.compile(r"^\W*(?:(?:so|and|but|also|well|ok|okay|oh|btw|anyway)\W+)*(?:i|i'm|im|i've|ive|i'd|i'll|we|we're|we've|you|he|she|they)\b", re.I)
VERDICT_ADVERBIAL = re.compile(r"\b(?:since|after|when|whenever|because|cause|as|while|before|until|till|once|now\s+that|though|although|if)\s+"
                               r"(?:i|i'm|im|i've|ive|i'd|we|we're|we've|you|he|she|they)\b[^,;:.?!\u2013\u2014]*", re.I)
VERDICT_PRONOUN = re.compile(_VLEAD + r"(?:(?:is|are|was|were|has|have|had|did|does|do)(?:n'?t)?\s+)?(?:it|that|this|they|those|these)\b", re.I)   # "Has it gone up?" (item 6)
# A sentence beside the comparison that states something is plain context (10g.28: lifestyle, training, travel, app use, pleasantries, curiosity): it
# opens with a subject and neither asks nor addresses the assistant. Any other sentence asks for something (a question, an imperative, "I want ...").
CMP_STATEMENT = re.compile(r"^\W*(?:(?:so|ok|okay|well|also|and|but|btw|anyway|fyi|oh)\W+)*(?:i|i'm|im|i've|ive|i'd|i'll|my|we|we're|we've|our|it|it's|its|this|that|there|the|a|an"
                           r"|he|she|they|his|her|their|last|yesterday|today)\b", re.I)
CMP_ADDRESS = re.compile(r"\?|\b(?:you|your|u|ur|please|pls)\b|\blet\s+me\s+know\b|\btell\s+me\b|\b(?:i|we)\s+(?:want|need|would\s+like|wanna|wish)\b|\bi'd\s+like\b", re.I)
CMP_CLAUSE = re.compile(r"(?:(?!\s-\s)[^,;:.?!–—])+")     # a clause: text between stops, commas, colons, semicolons or dashes
# A named device as an adjunct of a read ("... on Oura", "from my Garmin watch"): not a complement of a yes/no comparison.
SRC_NAME = re.compile(r"(?<![\w-])(?:(?:on|in|from|with|via|using|per|by|according\s+to)\s+)?(?:(?:my|the)\s+)?(?:" + '|'.join(re.escape(a) for a in sorted(SOURCE_ALIASES, key=len, reverse=True))
                      + r")(?:\s+(?:app|data|ring|watch|band|device|strap|scale))?(?![\w-])", re.I)
# A factual comparative or verdict, which needs two stated windows to plan (item 6: "is my HRV higher this week than last week?" plans; "has it gone up?",
# "did my RHR improve?", "is my HRV higher than last month?", "is my RHR trending down lately?", "has my deep sleep been trending up since June?" hand
# off in v2); a yes/no question about a change or a direction ("climbing" as a direction, not the activity: "did I go climbing"), or a comparative
# with "than".
CMP_VERDICT = re.compile(r"^\W*(?:(?:so|ok|okay|hey|hi|um|and|but|quick question:?)\s+)*(?:has|have|had|did|does|do|is|are|was|were|am)\b[^.?!]*?\b(?:improv\w*|gone\s+(?:up|down)|go(?:ing)?\s+(?:up|down)|went\s+(?:up|down)|increas\w*|decreas\w*|drop\w*|ris(?:en|ing)|rose|fall(?:en|ing)|fell|gotten\s+(?:better|worse)|got\s+(?:better|worse)|getting\s+(?:better|worse)|higher|lower|better|worse|more|less"
                         r"|trend(?:ing|ed)?\s+(?:up|down|upwards?|downwards?|higher|lower)|declin\w*|on\s+the\s+(?:rise|decline)|creep\w*\s+(?:up|down)|slipping"
                         r"|(?<!go\s)(?<!went\s)(?<!going\s)climbing(?!\s+(?:workouts?|sessions?|gym|wall|routes?|stairs|floors)))\b"
                         r"|\b(?:higher|lower|better|worse|more|less|greater|fewer)\b[^.?!]{0,40}?\bthan\b", re.I)
# Self-contained vs elliptical follow-ups (spec v1.8 10g.26): after stripping politeness, a request with its own predicate that
# names its own metric stands alone, unless it is additive ("too", "as well"), refers back ("then", "that week") or uses a pronoun.
# "what did/do/does ..." and "what <item> am/is/do I ..." are such predicates too ("What did my hs-CRP come back as?", "What did I
# weigh?", "What VO2 max am I at now?", "What readiness score do I have?").
FILLERS = re.compile(r"^\W*(?:(?:(?:can|could|would|will)\s+you|please|pls|hey|hi|ok(?:ay)?|so|well|um+|uh+|thanks|right|alright|cool|great)\b[\s,!.:-]*)*", re.I)
LEAD_CONNECTOR = re.compile(r"^(?:(?:and|now|then|also|plus)\b[\s,]*)+", re.I)
PREDICATE = re.compile(r"^(?:what(?:'s|s|\s+is|\s+was|\s+are|\s+were|\s+did|\s+do|\s+does|\s+(?:[\w()-]+\s+){1,4}?(?:am|is|are|was|were|did|do|does|have|has)\s+(?:i|my))|how(?:'s|\s+is|\s+was|\s+are|\s+were|\s+did|\s+do|\s+much|\s+many|\s+long|\s+often)|show|tell|check|give|pull|list|display|get|find|graph|plot|chart|see|look|view|fetch|report|summari[sz]e|i\s+(?:want|need|would like))\b", re.I)
ADDITIVE = re.compile(r"\b(?:too|as well|also|likewise|the same way|same for)\b", re.I)
FOLLOW_MARKER = re.compile(r"^(?:and|now|then|also|plus|what about|how about|same for|likewise)\b", re.I)   # 10g.26 (a) continuation connectors
BACKREF = re.compile(r"\b(?:then|that (?:week|month|day|year|period|night|time|window)|the same (?:period|days|time|dates|window|week|month|day)|those (?:days|dates)|during that time)\b|\b(?:it|that|those|them|these)\b", re.I)
# A tagged window that only refers back to the prior one (10g.26 (c): "that day", "the same days", "that week", "during that time",
# "then"); group 1 is the unit it names, if any.
BACKREF_WINDOW = re.compile(r"\b(?:(?:on|over|for|during|in|from|at|across|throughout)\s+)?(?:(?:that|those|the)\s+(?:same|very|exact)\s+|that\s+|those\s+)"
                            r"(day|date|night|weekend|week|month|year|days|dates|nights|weeks|months|period|time|stretch|span|window|range|timeframe)\b|\b(?:back\s+)?then\b", re.I)
def window_fits(word, p, ref):
    """The unit a back-reference names fits the prior read's window (decoder period shapes): a day, date or night is one day, a weekend
    Sat-Sun, a week a Mon-Sun week (or its days to date) or 7 rolling days, a month one calendar month (or to date) or a rolling month,
    a year one calendar year (or to date) or 12 rolling months. Plural and generic words ("the same days", "that period") fit any."""
    k, word = p.get('kind'), (word or '').lower()
    if word not in ('day', 'date', 'night', 'weekend', 'week', 'month', 'year'): return True
    if k == 'calendar': return p.get('period') == {'date': 'day', 'night': 'day'}.get(word, word)
    if k == 'relative':
        n = int(p['amount']) * (7 if p['unit'] == 'weeks' else 1)
        return (word == 'week' and p['unit'] in ('days', 'weeks') and n == 7) or (word == 'month' and p['unit'] == 'months' and n == 1) or (word == 'year' and p['unit'] == 'months' and n == 12)
    if k != 'between': return False
    s, e = date.fromisoformat(str(p['start_at'])[:10]), date.fromisoformat(str(p['end_at'])[:10])
    if word in ('day', 'date', 'night'): return s == e
    if word == 'weekend': return s.weekday() == 5 and (e - s).days == 1
    if word == 'week': return (e - s).days == 6 or (s.weekday() == 0 and (e - s).days < 6 and e == ref)
    if word == 'month': return s.day == 1 and (s.year, s.month) == (e.year, e.month) and (e == ref or e.day == _cal.monthrange(e.year, e.month)[1])
    return (s.month, s.day) == (1, 1) and s.year == e.year and (e == ref or (e.month, e.day) == (12, 31))
def self_contained(cur):
    t = LEAD_CONNECTOR.sub('', FILLERS.sub('', cur.strip()))
    if re.match(r"(?:what|how)\s+about\b|same\b", t, re.I) or not PREDICATE.match(t): return False     # a fragment or connector form
    return not (ADDITIVE.search(cur) or BACKREF.search(cur))
# "N units ago" names a point in time (spec v1.8 10g.24); the usual unit abbreviations are those units ("3 wks ago").
# "N units back" is the same point (spec v1.8 10g.36 item 2: "Where was my LDL 2 months back?", incl. the sparse lookback), as are "N units
# before/prior to now|today" and "a couple of" (two). "back N units", "going back N months" and "looking back" are rolling windows, not this.
_AGO_N = r'(?:\d+|' + NUM_WORDS + r'|a|an|(?:a\s+)?couple(?:\s+of)?)\s+(?:day|week|wk|month|mo|mth|year|yr)s?'
_AGO_END = r'(?:ago|back(?!\s+to\b|\s+and\s+forth\b)|(?:before|prior\s+to|earlier\s+than)\s+(?:now|today))\b'
UNITS_AGO_ANY = re.compile(r'\b(since\s+)?(\d+|' + NUM_WORDS + r'|a|an|(?:a\s+)?couple(?:\s+of)?)\s+(day|week|wk|month|mo|mth|year|yr)s?\s+' + _AGO_END, re.I)   # "twenty-five days ago" is 25
UNIT_ABBR = {'wk': 'week', 'mo': 'month', 'mth': 'month', 'yr': 'year'}
AGO_MAX = {'day': 3660, 'week': 520, 'month': 120, 'year': 120}     # the furthest "N units ago" anchor read (about ten years); beyond it -> unsupported_period (10g.24)
def ago_out_of_range(text):
    """True when an "N units ago" / "since N units ago" / "N units back" in the text counts past AGO_MAX (never read: no absurd window, no date overflow)."""
    for m in UNITS_AGO_ANY.finditer(text or ''):
        n, u = m.group(2).lower(), UNIT_ABBR.get(m.group(3).lower(), m.group(3).lower())
        if (int(n) if n.isdigit() else num_words(n) or 1) > AGO_MAX[u]: return True
    return False
# Point-in-time phrases that name no point (spec v1.8 10g.24, 10g.36 item 2; the spec is silent -> handoff): a vague amount ("a few days ago",
# "the past several weeks", "a while ago"), a count with no reference ("2 months earlier", "3 weeks prior", "2 days before my run") and an
# "N units ago" used as an end of a range ("from 3 weeks ago to today", "between 2 months ago and last week"), "N nights ago" and a named period with a
# vague qualifier ("mid-August", "early March", "around June", "the last weeks of August"). Only "since N units ago" is defined.
_VAGUE = r'(?:few|several|many|numerous|handful\s+of|bunch\s+of)'
_UNIT_S = r'(?:day|week|wk|month|mo|mth|year|yr|night)s?'
UNRESOLVED_POINT = re.compile(
    rf'\b(?:a\s+)?{_VAGUE}\s+{_UNIT_S}\s+(?:ago|back|earlier|prior|before)\b|\b(?:past|last|previous|prior|trailing)\s+{_VAGUE}\s+{_UNIT_S}\b'
    r'|\b(?:a\s+)?(?:while|long\s+time|little\s+while)\s+(?:ago|back)\b|\b(?:some\s+time|ages|forever)\s+ago\b'
    rf'|\b(?:\d+|{NUM_WORDS}|a|an|(?:a\s+)?couple(?:\s+of)?)\s+nights?\s+(?:ago|back)\b'      # "3 nights ago": the night that ended or that began then? (10g.4 names neither)
    rf'|\b{_AGO_N}\s+(?:earlier|prior|before|previously)\b(?!\s+(?:(?:than|to)\s+)?(?:now|today)\b|\s+(?:that|then|this|it)\b)'
    rf'|\b(?:between|starting(?:\s+(?:from|at))?|beginning(?:\s+(?:from|at))?)\s+(?:the\s+)?{_AGO_N}\s+{_AGO_END}'
    rf'|\b{_AGO_N}\s+{_AGO_END}\s+(?:to|until|till|til|through|thru|up\s+to|up\s+until)\b'
    rf'|\b(?:to|until|till|til|through|thru)\s+{_AGO_N}\s+{_AGO_END}', re.I)
# Time phrases the period grammar has no form for. The spec (v1.8 10g.3) defines whole named periods, single days, weekdays and counted windows and is silent
# on these, so they hand off (unsupported_period), read from the text of the phrase so that no plan rests on what the heads guessed: (1) an ordinal part of a
# period ("the first Monday of September", "the last day of August", "the third week of March"), (2) a week number ("week 38", "wk 38", "W38", "ISO week"),
# (3) complete periods ("the last 2 full weeks", "the last full month"), (4) repetition and grouping ("month by month", "every Monday", "Mondays", "monthly
# breakdown", "by week"), (5) an inequality on an anchor ("at least 3 days ago", "over 2 months ago"), (6) hours and minutes ("the last 24 hours", "past 48 hours",
# "2 hours ago"). Settled forms stay: "for each day this week", "daily", "per day" (10g.33 item 4), "the last 2 weeks", "last week", "this year", "the first of
# September", "the whole week so far" and the hedged anchors ("about 2 weeks ago", "roughly a month back", "exactly 2 months ago") are plain windows and anchors.
_ORD = r'(?:first|second|third|fourth|fifth|last|\d{1,2}(?:st|nd|rd|th))'
_IN_PERIOD = rf'\s+(?:of|in)\s+(?:the\s+|this\s+|last\s+|next\s+|that\s+)?(?:{_BOUND_MON}|month\b|year\b|week\b|quarter\b)'
_TIME_N = rf'(?:\d+|{NUM_WORDS}|a|an|one|(?:a\s+)?couple(?:\s+of)?|few)'
UNSUPPORTED_PERIOD = re.compile(
    rf'\b{_ORD}\s+(?:{_BOUND_DAY}|day|weekend|weekday)s?{_IN_PERIOD}|\b(?:first|second|third|fourth|fifth|\d{{1,2}}(?:st|nd|rd|th))\s+week{_IN_PERIOD}'   # "the last week of August" is the parser's last 7 days (regress_coverage)
    rf'|\b(?:iso\s+|calendar\s+)?week\s*(?:no\.?|nr\.?|num(?:ber)?\.?|#)?\s*\d{{1,2}}\b(?!\s*(?:days?\b|of\s+(?:my|our)\b))|\b(?:wk|cw|kw)\.?\s*\d{{1,2}}\b|\bw\d{{2}}\b|\biso\s+weeks?\b|\b\d{{1,2}}(?:st|nd|rd|th)\s+week\b'
    rf'|\b(?:last|past|previous|prior|trailing)\s+(?:{_TIME_N}\s+)?(?:full|complete|whole|entire)\s+(?:{_TIME_N}\s+)?(?:day|week|month|year)s?\b'
    rf'|\b(?:\d+|{NUM_WORDS})\s+(?:full|complete|whole|entire)\s+(?:day|week|month|year)s\b|\b(?:full|complete|whole|entire)\s+(?:weeks|months|years)\b'
    rf'|\b(?:day|week|month|year)s?[- ]by[- ](?:day|week|month|year)\b|\b(?:every|each)\s+(?:single\s+|other\s+)?{_BOUND_DAY}|\b(?:mon|tues|wednes|thurs|fri|satur|sun)days\b'
    r'|\b(?:weekly|monthly|yearly|quarterly)\s+(?:breakdown|summary|summaries|totals?|averages?|numbers|stats|statistics|reports?|values|trends?|series|figures|data|aggregates?)\b'
    r'|\b(?:group(?:ed)?|broken\s+down|split|aggregated|summed|averaged|bucketed|binned)\s+(?:it\s+)?(?:by|per)\s+(?:the\s+)?(?:day|week|month|year)\b|\bby\s+(?:the\s+)?(?:week|month)\b'
    rf'|\b(?:at\s+(?:least|most)|(?:no|not)\s+(?:more|less|fewer|greater)\s+than|more\s+than|less\s+than|fewer\s+than|greater\s+than|over|under|just\s+(?:over|under)|nearly|almost|beyond)\s+{_AGO_N}\s+{_AGO_END}'
    rf'|\b(?:last|past|previous|trailing|next|coming)\s+(?:{_TIME_N}[-\s]*)?(?:hours?|hrs?|minutes?|mins?)\b|\b(?:last|past|previous|trailing)\s+\d+\s*h\b|\b{_TIME_N}[-\s]*(?:hours?|hrs?|minutes?|mins?)\s+(?:ago|back)\b', re.I)
def temporal_guard(cur, said_sents, time_at, ref, prev_ok=False):
    """The handoff code of the time-wording guards on the read's sentences (said_sents) and tagged time spans (time_at), else None. Each
    guard reads the raw text and the text as the date parser reads it (normalize_time_text), so a spelling the parser accepts is guarded
    as its plain form is ("HRV before yestrday", "steps through lasr month", "until weds"). prev_ok: a follow-up's "the previous N days"."""
    ncur, nmap = normalize_time_text(cur, offsets=True)
    hits = lambda rx: [(m.start(), m.end()) for m in rx.finditer(cur)] + [(nmap[m.start()], nmap[m.end()]) for m in rx.finditer(ncur)]
    said = lambda a, z: any(x <= a and z <= y for x, y in said_sents)
    # These phrases count only as the read's own window: in a sentence of the read and overlapping a tagged time span (a "while back" in a context clause is not).
    own_window = lambda rx: [(a, z) for a, z in hits(rx) if any(x <= a < y for x, y in said_sents) and any(p < z and q > a for p, q in time_at)]
    if any(said(a, z) for a, z in hits(DATE_BOUND)): return 'exclusion_or_filter'   # "... after the 15th" (10g.28 class 4)
    if own_window(UNRESOLVED_POINT) and not prev_ok:
        return 'unresolved_temporal_phrase'      # no point to read (10g.24, 10g.36 item 2): "a few days ago", "2 months earlier", "from 3 weeks ago to today"
    if own_window(VAGUE_PERIOD): return 'unresolved_temporal_phrase'      # "mid-August", "early March", "around June" (10g.3)
    if own_window(BOUND_FILTER): return 'exclusion_or_filter'      # "steps before June", "after March": a restriction on the readings, no window (10g.28 class 4)
    if own_window(BOUND_OPEN):
        try: dates = len(_dates_in(' '.join(cur[x:y] for x, y in said_sents), ref))
        except (AmbiguousDate, InvalidDate): dates = 0
        if dates < 2: return 'unresolved_temporal_phrase'      # "steps until Tuesday", "through March": an end with no start (10g.29); "Monday through Wednesday" is a range
    # These are read from the text of the request's own sentences whether or not a model tagged them as time ("resting heart rate W36", "my weight monthly summary")
    if any(any(x <= a < y for x, y in said_sents) for a, z in hits(UNSUPPORTED_PERIOD)): return 'unsupported_period'      # ordinal parts, week numbers, complete periods, repetition, inequality anchors, hours (10g.3: silent -> handoff)
    return None


# "the next/coming N days" is a future window (spec item 10b); the date parser marks only "next week/month/year" as future.
NEXT_N = re.compile(r'\b(?:next|upcoming|coming)\s+(?:\d+|' + NUM_WORDS + r'|a|an|few|couple(?:\s+of)?)\s+(?:day|night|week|month|year)s?\b', re.I)
# Month/year/week to date abbreviations: date syntax the date parser does not read (spec item 10: "this month" = MTD).
TO_DATE_ABBR = re.compile(r'\b(?:mtd|ytd|wtd)\b', re.I)
# A time span that only states a granularity ("each", "every day", "daily") is not a window of its own (10g.6 counts windows).
FREQUENCY = re.compile(r'(?:(?:each|every|per|a)\s+)?(?:day|night|week|month)|each|every|per|daily|nightly|weekly|monthly', re.I)
# Nor does one that only states the scope of the named day ("Yesterday's average heart rate across the whole day").
WHOLE_DAY = re.compile(r'(?:(?:across|over|for|during|throughout|through|in)\s+)?(?:the\s+)?(?:whole|entire|full)\s+(?:day|night)', re.I)
# A bare "what is X?" with no first person, value word, window or record is a definition question (v2: handoff).
DEFINITION = re.compile(r"^\W*(?:what(?:'s| is| are)|whats|what does)\s+(?:an?\s+|the\s+)?[\w\s()'/+-]{1,40}?(?:\s+(?:mean|measure|stand for|refer to|do|used for|used to check for))?\s*\??\s*$", re.I)
PERSONAL = re.compile(r"\b(?:my|mine|me|i|i'm|i've|our|ym)\b|\b(?:latest|newest|recent|current|reading|result|results|number|numbers|level|levels|value|values|schedule|calendar|appointment|workout|lab|profile)\b", re.I)
# The assistant's name used as an address ("Hi Vita, ...") is not the Vita sleep score.
ADDRESS = re.compile(r'^\W*(?:(?:hey|hi|hello|ok|okay|yo|dear|good (?:morning|evening|afternoon)|morning)\W+vita\b\W*|vita\s*[,:!]\s*)'
                     r'|\W+(?:thanks|thank you|thx|ty),?\s+vita\W*$', re.I)          # "Vita sleep score" (no comma) stays a metric name
# Pleasantries around a request and runs of spaces carry no meaning but move the model (stress set s2: "Hey, respiratory rate this
# month", "What's my RBC count? Thanks!", "rhr  last  wk" hand off; the plain requests plan). The model reads the request without them,
# as it reads "Hi Vita, ..." without the address. Not "so" ("so far today"), not a leading emoji ("😴 REM last night").
POLITE = re.compile(r"^\W*(?:(?:(?:hey|hi|hello|hiya|yo)(?: there)?|ok|okay|um+|uh+|quick question|question|one more thing|btw)(?:[\s,:;.!?-]+|$))+"
                    r"|(?:[\s,.;:!-]*\b(?:thanks|thank you(?: so much| very much)?|thx|ty|tysm|cheers|please|pls|plz)(?: in advance)?\b[\s.!]*"
                    r"|\s*[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]+)+\s*$", re.I)
SPACES = re.compile(r"[ \t\u00a0]{2,}")
# Vague recency states no window: "how's my sleep lately/these days" reads the default window (spec v1.8 10g.13, 10f item 4).
VAGUE_RECENCY = re.compile(r"\b(?:these days|lately|recently|of late|nowadays)\b", re.I)
# Point-in-time words ask for the value now, not for a window (spec v1.8 10g.32 item 9, clarified: "currently", "right now", "at the
# moment", "now" -> latest for any single metric).
POINT_NOW = re.compile(r"\b(?:right now|currently|at the moment|now)\b", re.I)
# "most recently" asks for the latest item, as a latest word does (10g.15): no window either ("Which workout did I do most recently?").
MOST_RECENTLY = re.compile(r"\bmost\s+recently\b", re.I)
# Current periods have one canonical shape (10g.35 item 1): calendar day/week/month, as Vita freezes them to [start, reference time].
WEEK_NOW, MONTH_NOW = (re.compile(p, re.I) for p in (r'\b(?:this|current)\s+week\b|\bweek\s+to\s+date\b|\bwtd\b', r'\b(?:this|current)\s+month\b|\bmonth\s+to\s+date\b|\bmtd\b'))
# Typical-day wording asks for the usual level: with no other window, a trend over the default window, whose summary carries the mean
# (spec v1.8 10g.33 item 4). "daily X" is a metric adjective, not this; "a day ago" is a point in time (10g.24).
TYPICAL_DAY = re.compile(r"\b(?:(?:in|per)\s+(?:an?\s+)?day|an?\s+day(?!\s+(?:ago|back|or\s+two)\b)|on\s+(?:an?\s+)?(?:typical|normal|average)\s+day|usually|typically|on\s+average|usual|typical)\b", re.I)   # "my usual RHR", "my typical daily distance" too
# Any other time word (or a digit) outside the metric/record/source mentions may carry a window the parsers do not know, so a
# request containing one never takes the no-window default.
TIME_WORD = re.compile(r"\d|\b(?:days?|weeks?|months?|years?|nights?|fortnights?|weekends?|quarters?|today|tonight|yesterday|tomorrow|"
                       r"mornings?|evenings?|afternoons?|since|ago|during|between|until|till|before|after|last|past|previous|prior|"
                       r"ever|all|entire|whole|full|complete|beginning|start|started|began|recent|current|currently|now|then|when|"
                       r"earlier|later|previously|while|daily|weekly|monthly|yearly|annual|nightly|hourly|ytd|mtd|holidays?|vacation|trip|"
                       r"christmas|xmas|easter|thanksgiving|halloween|birthday|anniversary|spring|summer|autumn|fall|winter|seasons?)\b", re.I)
TIME_PART = re.compile(r"day|night|week|month|year|hour|morning|evening|noon|time|dawn|dusk|\b(?:this|next|coming|upcoming|following|"
                       r"first|second|third|fourth|half|end|mid|early|late|final|am|pm|clock|o'?clock|min|mins|hrs?|bed\w*|lunch\w*|dinner\w*|breakfast)\b", re.I)


def names_time(t):
    """A tagged time span names a time if any word is (part of) a time, clock, month or weekday word, a deictic ("this", "next",
    "coming") or a misspelt day word. A stray letter or function word the tagger marked ("n" of "mnay", "ven" of "venu", "I",
    "in", "of the", "each") names none, so it is not a second window to split clauses on (spec v1.8 10g.6 splits clauses that
    carry different windows)."""
    return bool(TIME_WORD.search(t) or TIME_PART.search(t)) or any(
        w in _MONTHS or w in _WEEKDAYS or _fix_typo(w) != w for w in re.findall(r'[a-z]+', t.lower()))


def time_only(t):
    """t names a time and every word of it is a time word or a function word of one ("the week before", "June", "2024")."""
    return names_time(t) and all(names_time(w) or w.isdigit() or w in ('the', 'in', 'on', 'for', 'during', 'over', 'of', 'a', 'an', 'same', 'that', 'one', 'it')
                                 for w in re.findall(r"[a-z']+|\d+", t.lower()))


def plain(t):
    """t without pleasantries around it or runs of spaces (POLITE, SPACES); t itself when nothing else is left."""
    u = POLITE.sub('', SPACES.sub(' ', t)).strip()
    return u if u.strip(' ,.!?') else t


def calm(t):
    """A shouted message (mostly capitals) is read in lowercase: case carries no meaning there, and the model saw little all-caps text
    (crisis layers on UPPER rewrites of the calibration split's acute rows: r7a 126/176, r7b 141/176; read in lowercase 176/176 each).
    Every character offset is kept."""
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 8 or sum(c.isupper() for c in letters) < 0.7 * len(letters): return t
    return ''.join(c.lower() if len(c.lower()) == 1 else c for c in t)


def one_night_split(qs):
    """The two reads of one night (spec v1.8 10g.4): NIGHT_METRICS on the sleep_end_day basis plus a one-day observed_at window of
    the same day, with one operation and source."""
    p = qs[0].get('period') or {} if qs else {}
    return (len(qs) == 2 and {q.get('date_basis') for q in qs} == {'sleep_end_day', 'observed_at'} and all(q.get('metrics') and not q.get('records') for q in qs)
            and len({json.dumps([q.get('period'), q.get('operation'), q.get('source')], sort_keys=True) for q in qs}) == 1
            and p.get('kind') == 'between' and p.get('start_at') == p.get('end_at'))


def contract_period(p, ref, records=()):
    """Period shapes the repo contract takes (compat/health_range_input.py: relative days/months; calendar day/week/month):
    a rolling N weeks is 7N days; the current calendar year is Jan 1 .. the reference date (year to date, spec item 10).
    Record dates are date-typed: a rolling window on labs or calendar records is whole days, the anchor date .. the reference
    date (spec v1.8 10g.33 item 2); relative is for timestamped metrics."""
    if p.get('kind') == 'relative' and {'labs', 'calendar'} & set(records):
        n = int(p['amount']); s = ref - timedelta(days=n * (7 if p['unit'] == 'weeks' else 1)) if p['unit'] in ('days', 'weeks') else _months_back(ref, n)
        return {'kind': 'between', 'start_at': s.isoformat(), 'end_at': ref.isoformat()}
    if p.get('kind') == 'relative' and p.get('unit') == 'weeks': return {'kind': 'relative', 'amount': 7 * int(p['amount']), 'unit': 'days'}
    if p.get('kind') == 'calendar' and p.get('period') == 'year': return {'kind': 'between', 'start_at': f'{ref.year}-01-01', 'end_at': ref.isoformat()}
    return p


# Research topics (spec #243, v1.8 10g.8). Gold spans tag the whole subject phrase as research_topic, entity names included
# ("lp(a) and heart disease"); the tagger often marks an entity inside it as a metric, leaves a function word untagged, or tags
# part of a word. The topic is the full span of whole words from the first to the last research_topic word, grown over entity
# words that touch it directly or across one coordinating word ("oxygen saturation dips and sleep apnea"), never past a clause end.
TOPIC_ENTITY = {'metric', 'record', 'profile_field'}
TOPIC_JOIN = {'and', 'or', '&', '+', 'vs', 'versus'}
NUMBERED_NAME = re.compile(r'\b(zone|omega|type|stage|phase|grade|class|glp|covid|il)[ -](\d{1,2}s?)\b(?![.,/%]\d)', re.I)   # names, not values
TOPIC_VALUE = re.compile(r'(?:\s*\b(?:of|at|around|about|near)\s+|\s*[=:]\s*)?(?<![\w-])[<>~]?\d[\d.,/%]*\s*'
                         r'(?:mg/dl|mmol/l|ng/ml|pg/ml|nmol/l|mg|g|kg|lbs?|%|bpm|ms|iu|units?)?(?![\w-])', re.I)
SELF_WORDS = re.compile(r"\b(?:me|my|mine|myself|i'm|i've|i'd)\b|(?<!type )(?<!phase )(?<!stage )(?<!class )(?<!grade )(?<!complex )\bi\b", re.I)
TOPIC_LEAD = {'the', 'a', 'an', 'any', 'some', 'and', 'or', 'of', 'for', 'on', 'in', 'with', 'like', 'as', 'to', 'vs', 'versus'}
TOPIC_TAIL = TOPIC_LEAD - {'a'}                    # a trailing letter is a name: "vitamin a", "urolithin a"
TOPIC_FUNCTION = TOPIC_EXCLUDED | TOPIC_LEAD | {'about', 'from', 'by', 'at'}
TOPIC_REFERENCE = {'this', 'that', 'these', 'those', 'it', 'its', 'their', 'them', 'they'}   # "this supplement": the subject is in history


def plain_month_range(text):
    if re.search(r'\b(?:compar\w*|chang\w*|difference|higher|lower|versus|vs)\b', text, re.I): return None
    month = '(?:' + '|'.join(sorted(_MONTHS, key=len, reverse=True)) + ')'
    endpoint = month + r'(?:\s+\d{1,4}(?:,?\s+\d{4})?)?'
    return re.search(r'\bbetween\s+' + endpoint + r'\s+and\s+' + endpoint + r'\b(?!\s+(?:\d|of\b|in\s+\d))', text, re.I)


def canonical_topic(t):
    """One string per subject: lowercase, single spaces, no edge punctuation or unmatched edge bracket, no leading article or
    determiner, no dangling connector ("The accuracy of X." and "accuracy of X" agree)."""
    t = ' '.join(t.lower().replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").split())
    while True:
        s = t.strip(' .,;:!?"\'<>=~/')            # incl. what a stripped value leaves ("HbA1c>6.5" -> "hba1c")
        if s.startswith('(') and s.count('(') > s.count(')'): s = s[1:]
        if s.endswith(')') and s.count(')') > s.count('('): s = s[:-1]
        w = s.split()
        if w and w[0] in TOPIC_LEAD: w = w[1:]
        if w and w[-1] in TOPIC_TAIL: w = w[:-1]
        if ' '.join(w) == t: return t
        t = ' '.join(w)


def research_topic(text, toks, lo, hi):
    """(topic, None) or (None, handoff code) for the research span of text[lo:hi]; toks = [(start, end, role)] of that part."""
    words = [(lo + m.start(), lo + m.end()) for m in re.finditer(r'\S+', text[lo:hi])]
    kind = []
    for a, z in words:
        roles = {r for ta, tz, r in toks if ta < z and tz > a}
        kind.append('topic' if 'research_topic' in roles else 'entity' if roles & TOPIC_ENTITY else None)
    found = [k for k, x in enumerate(kind) if x == 'topic']
    if not found:   # no word tagged as the topic, only entity names ("Papers about heat shock proteins and sauna bathing"): those names
        # are the subject, from the first to the last, within one clause; a negation or exclusion hands off (10g.32 item 8)
        found = [k for k, x in enumerate(kind) if x == 'entity']
        if not found or any(re.search(r'[.?!;:]$', text[words[k][0]:words[k][1]]) for k in range(found[0], found[-1])) or \
                re.search(r"\b(?:without|not|no|except|excluding|exclude|only|but)\b|n't\b", text[lo:hi], re.I): return None, 'unbound_research_topic'
    i, j = found[0], found[-1]
    word = lambda k: text[words[k][0]:words[k][1]] if 0 <= k < len(words) else ''
    entity = lambda k: 0 <= k < len(words) and kind[k] == 'entity'
    join = lambda k: word(k).lower() in TOPIC_JOIN
    closes = lambda k: bool(re.search(r'[.?!;:]$', word(k)))      # a clause ends after this word
    while True:
        if entity(i - 1) and not closes(i - 1): i -= 1
        elif join(i - 1) and entity(i - 2) and not closes(i - 2): i -= 2
        elif closes(j): break
        elif entity(j + 1): j += 1
        elif join(j + 1) and entity(j + 2): j += 2
        else: break
    topic = NUMBERED_NAME.sub(r'\1-\2', text[words[i][0]:words[j][1]])      # "zone 2", "omega 3", "type 2" are names
    if re.search(r'(?<![\w-])\d', topic): return None, 'unbound_research_topic'
    topic = TOPIC_VALUE.sub(' ', topic)                                     # privacy: values never leave ("an LDL of 160" -> "an LDL")
    topic = ' '.join(w for w in topic.split() if not re.match(r'[^A-Za-z0-9]*\d', w))   # nor any other number-led word ("2x", "(30)")
    if SELF_WORDS.search(re.sub(r'\b[A-Z]{2,}\b', ' ', topic)): return None, 'targeted_research_binding_unavailable'   # about the user (10g.8); "ME/CFS" is a name
    topic = canonical_topic(topic)
    words = topic.split()
    # Nothing public to search for, or a subject only named in earlier turns ("papers on that", "this supplement"): spec #243.
    if not words or words[0] in TOPIC_REFERENCE or not any(re.search('[a-z]', w) and w.strip("().,+&-'") not in TOPIC_FUNCTION for w in words):
        return None, 'unbound_research_topic'
    return topic, None


def above(x, thr):
    """Indices i with x[i] > thr, ascending: one comparison over the tensor, in its dtype, as x[i] > thr compares."""
    return [i for i, y in enumerate((x > thr).tolist()) if y]


def mention_runs(roles, offs, lo, hi, text=None):
    """Contiguous runs of mention-role tokens inside the current request (list splitting happens in Parser.split_run)."""
    runs, cur, last = [], [], None
    for j, ((a, z), r) in enumerate(zip(offs.tolist(), roles)):
        name = ROLES[r]
        if z > a and a >= lo and z <= hi and name in ('metric', 'record', 'profile_field'):
            if cur and name == last and j == cur[-1] + 1: cur.append(j)
            else:
                if cur: runs.append(cur)
                cur = [j]
            last = name
        elif cur and z > a:
            runs.append(cur); cur = []; last = None
    if cur: runs.append(cur)
    return runs


class KeywordAcute:
    def __init__(self):
        from query_selector import acute_or_crisis
        self.rule = acute_or_crisis
    def __call__(self, text): return self.rule(text)


def packed_hidden(enc, batches):
    """ModernBERT's forward (non-flash path) over several inputs in one pass, without padding: the tokens of all inputs go through
    the embeddings, norms, linear layers and MLPs together; attention runs per input with its own mask and positions, as in its own
    pass. Returns each input's last hidden state."""
    cut = [0]
    for b in batches: cut.append(cut[-1] + b['input_ids'].shape[1])
    masks = [enc._update_attention_mask(b['attention_mask'], output_attentions=False) for b in batches]
    attend = MODERNBERT_ATTENTION_FUNCTION[enc.config._attn_implementation]
    h = enc.embeddings(input_ids=torch.cat([b['input_ids'] for b in batches], dim=1))
    for layer in enc.layers:
        a = layer.attn; qkv = a.Wqkv(layer.attn_norm(h))
        h = h + a.out_drop(a.Wo(torch.cat([attend(a, qkv=qkv[:, x:y].view(1, -1, 3, a.num_heads, a.head_dim), attention_mask=m, sliding_window_mask=w,
                                                  position_ids=torch.arange(y - x).unsqueeze(0), local_attention=a.local_attention, bs=1, dim=a.all_head_size)[0]
                                           for x, y, (m, w) in zip(cut, cut[1:], masks)], dim=1)))
        h = h + layer.mlp(layer.mlp_norm(h))
    h = enc.final_norm(h)
    return [h[:, x:y].clone() for x, y in zip(cut, cut[1:])]


class Encoded:
    """A parser model whose encoder output is already computed (packed_hidden): JevParser's own forward does the rest."""
    def __init__(self, model, h): self._m, self._h = model, h
    def __getattr__(self, k): return getattr(self._m, k)
    def enc(self, input_ids, attention_mask): return SimpleNamespace(last_hidden_state=self._h)
    encode, forward = JevParser.encode, JevParser.forward


class InputTokenBudgetExceeded(ValueError):
    pass


class Parser:
    def __init__(self, path, vocab=DATA / 'vocabulary_enriched.json', acute=True, plan_threshold=None):
        self.tok = AutoTokenizer.from_pretrained(path); enc = AutoModel.from_pretrained(path)
        self.model = JevParser(enc, enc.config.hidden_size); load_parser_state(self.model, torch.load(path / 'heads.pt', map_location='cpu', weights_only=True)); self.model.eval()
        self.items = json.load(open(path / 'items.json'))
        with torch.inference_mode():
            self.I = torch.cat([self.model.item_embeddings(self.tok([t for _, _, t in self.items[i:i + 64]], return_tensors='pt', padding=True, truncation=True, max_length=96)) for i in range(0, len(self.items), 64)])
        V = json.load(open(vocab)); self.syn = []; self.syn_raw = []
        for kind, sec in (('metric', 'metrics'), ('record', 'records'), ('profile_field', 'profile_fields')):
            for key, e in (V.get(sec) or {}).items():
                for s in (e.get('synonyms') or []) + (e.get('abbreviations') or []):
                    s = s.lower().strip()
                    if len(s) >= 3: self.syn.append((re.compile(r'(?<![\w-])' + re.escape(s) + r'(?![\w-])'), kind, key)); self.syn_raw.append(s)
        self.syn_names = set(self.syn_raw)
        cpath = Path(path) / 'crisis.json'
        self.crisis_thr = json.load(open(cpath))['crisis_threshold'] if cpath.exists() else None
        # With a merged crisis head only the keyword rule remains outside the single pass.
        if self.crisis_thr is None: raise RuntimeError('learned parser requires the merged crisis head (crisis.json)')
        self.acute = KeywordAcute() if acute else None
        pos = {(k, key): i for i, (k, key, _) in enumerate(self.items)}; self.confuse = {}
        for key, e in (V.get('metrics') or {}).items():
            for c in e.get('confusable_with') or []:
                if ('metric', key) in pos and ('metric', c) in pos: self.confuse.setdefault(pos[('metric', key)], set()).add(pos[('metric', c)])
        for key in ('weight', 'height', 'bmi'):     # metric vs the stored profile value (catalog notes)
            for f in (key + '_kg', key + '_cm', key):
                if ('metric', key) in pos and ('profile_field', f) in pos:
                    self.confuse.setdefault(pos[('metric', key)], set()).add(pos[('profile_field', f)]); self.confuse.setdefault(pos[('profile_field', f)], set()).add(pos[('metric', key)])
        cal_path = Path(path) / 'calibration.json'
        cal = json.load(open(cal_path)) if cal_path.exists() else {}
        self.plan_threshold = plan_threshold if plan_threshold is not None else cal.get('plan_threshold', 0.5)
        self.conf_threshold = cal.get('conf_threshold', 0.0)      # joint confidence, set by conformal risk control (jevconformal.py)
        self._pack = self._packing_exact()

    def _batch(self, text):
        batch = self.tok(text, return_tensors='pt', truncation=False, return_offsets_mapping=True)
        if batch['input_ids'].shape[1] >= 190: raise InputTokenBudgetExceeded()
        return batch

    @torch.inference_mode()
    def _encode(self, texts):
        """Model outputs for several inputs from one packed encoder pass (packed_hidden)."""
        bs = [self._batch(t) for t in texts]
        for b in bs: b.pop('offset_mapping')
        return [Encoded(self.model, h).forward(b, self.I) for h, b in zip(packed_hidden(self.model.enc, bs), bs)]

    @torch.inference_mode()
    def _packing_exact(self):
        """Packing is used only where it changes nothing: on probes of several lengths (one past the local attention window) every
        output of the packed pass must equal the input's own pass bit for bit. A platform whose matrix products depend on the row
        count, or another encoder, keeps one pass per input."""
        enc = self.model.enc
        if getattr(enc.config, 'model_type', None) != 'modernbert' or enc.config._attn_implementation not in ('sdpa', 'eager') or enc.config.reference_compile: return False
        texts = [request_text('steps today', []), request_text('and my hrv?', ['how did I sleep last night']),
                 request_text(' '.join(['compare my resting heart rate with my HRV'] * 8), ['what was my deep sleep last week'] * 2)]
        own = []
        for t in texts:
            b = self._batch(t); b.pop('offset_mapping'); own.append(self.model(b, self.I))
        return all(torch.equal(v, got[k]) for n in (2, 3) for o, got in zip(own, self._encode(texts[:n])) for k, v in o.items())

    SPAN_MIN = 0.6      # cosine of a mention to its best item; calibrated on validation
    GROUP_THR = 0.0     # link-logit threshold for group words ("sleep", "lipids") and when no span is tagged

    LINK_THR = 0.5      # whole-input link logit threshold (validation-tuned)
    MARGIN = 0.0        # a mention expands to a group when several items score within this cosine margin of the best
    CATCH_THR = 99.0    # set-link logit for strong matches the tagger did not mark

    def split_run(self, run, offs):
        """A mention that spans a list separator ("HDL, LDL") is several mentions, unless the whole span is a catalog name
        ("cholesterol, HDL", "1,25-dihydroxy vitamin D", "B/P")."""
        text = self._text; whole = text[offs[run[0]][0]:offs[run[-1]][1]].strip().lower()
        if whole in self.syn_names: return [run]
        parts, cur = [], []
        for j in run:
            a, z = offs[j].tolist()
            if text[a:z].strip().lower() in SEPARATORS:
                if cur: parts.append(cur)
                cur = []
            else: cur.append(j)
        if cur: parts.append(cur)
        return parts or [run]

    @torch.inference_mode()
    def link(self, out, roles, offs, lo, hi, ok, inherit=None):
        h = out['hidden'][0]; t = F.normalize(self.model.tok_proj(h), dim=-1)
        allowed = torch.tensor([ok(k, key) for k, key, _ in self.items])
        setlink = lambda toks: ((t[toks] @ self.I.T).max(0).values * self.model.link_scale + self.model.link_bias).masked_fill(~allowed, -1e4)
        chosen = set(); self._runs = []; self._spans = []
        def within(items, toks):   # confusable pairs compete only inside one mention; items the user names separately both stay
            sc = {i: float(self.I[i] @ F.normalize(t[toks].mean(0), dim=-1)) for i in items}
            for i in sorted(items, key=lambda i: -sc[i]):
                if i in items:
                    for j in self.confuse.get(i, ()):
                        if j in items and sc[j] < sc[i] and not any({self.items[i][1], self.items[j][1]} <= b for b in BUNDLES): items.discard(j)
            return items
        for run in [p for run in mention_runs(roles, offs, lo, hi) for p in self.split_run(run, offs)]:
            e = F.normalize(t[run].mean(0), dim=-1); cos = (self.I @ e).masked_fill(~allowed, -1)
            best = float(cos.max()); g = setlink(run)
            if best >= self.SPAN_MIN:
                mine = {i for i, c in enumerate(cos.tolist()) if c >= best - self.MARGIN}
                group = set(above(g, self.LINK_THR))
                # A group word expands only to a complete spec bundle ("how did I sleep" -> core 5); a partial set keeps the
                # best item, so "how efficient is my sleep" stays sleep_efficiency (disagreement with the full view hands off).
                if any({self.items[i][1] for i in group} == b for b in BUNDLES): mine |= group
            else:
                mine = set(above(g, self.GROUP_THR))
            kept = within(mine, run); chosen |= kept
            self._runs.append((self._text[offs[run[0]][0]:offs[run[-1]][1]].lower(), {self.items[i][1] for i in kept}))
            self._spans.append((int(offs[run[0]][0]) - lo, int(offs[run[-1]][1]) - lo))
        cur = [j for j, (a, z) in enumerate(offs.tolist()) if z > a and a >= lo and z <= hi]
        if cur:   # strong matches the tagger missed (list items, short names)
            g = setlink(cur)
            for i in above(g, self.CATCH_THR):
                if not (self.confuse.get(i, set()) & chosen): chosen.add(i)
            if not chosen:
                chosen = set(above(g, self.GROUP_THR))
        self._inherit_read = not chosen                     # the inherit head decides only when the request itself links nothing
        if not chosen and inherit in ('subject', 'both'):   # follow-up: the subject comes from the previous request
            hist = [j for j, (a, z) in enumerate(offs.tolist()) if z > a and a > hi]
            if hist:
                g = setlink(hist); chosen = set(above(g, self.GROUP_THR))
        return chosen

    @staticmethod
    def _prior_window(p, ref):
        """(granularity, first day, last day) of a read window that a relative window shifts (10g.25): a current calendar period is the whole
        period; a between window is a day, a Mon-Sun week, a Sat-Sun weekend, a calendar month or year, or a range; a rolling window is
        ('rolling', N, unit). None: all history, or no window shape."""
        k = p.get('kind')
        if k == 'calendar':
            gran = p.get('period'); s = ref if gran == 'day' else ref - timedelta(days=ref.weekday()) if gran == 'week' else ref.replace(day=1) if gran == 'month' else ref.replace(month=1, day=1)
            e = s if gran == 'day' else s + timedelta(days=6) if gran == 'week' else s.replace(day=_cal.monthrange(s.year, s.month)[1]) if gran == 'month' else s.replace(month=12, day=31)
        elif k == 'between':
            s, e = date.fromisoformat(p['start_at'][:10]), date.fromisoformat(p['end_at'][:10])
            gran = ('day' if s == e else 'week' if (e - s).days == 6 and s.weekday() == 0 else
                    'month' if s.day == 1 and e.month == s.month and e.day == _cal.monthrange(e.year, e.month)[1] else
                    'year' if (s.month, s.day, e.month, e.day) == (1, 1, 12, 31) and s.year == e.year else
                    'weekend' if s.weekday() == 5 and (e - s).days == 1 else 'range')
        elif k == 'relative':
            n, u = int(p['amount']), p['unit'].rstrip('s'); gran = ('rolling', n, u)
            e = ref; s = ref - timedelta(days=n - 1) if u == 'day' else ref - timedelta(days=7 * n - 1) if u == 'week' else _months_back(ref, n) + timedelta(days=1)
        else: return None
        return gran, s, e

    @staticmethod
    def _shift_back(gran, s, e, cur):
        """The window (s, e) of granularity gran moved as the relative words in cur say (10g.25): "the previous N days" / "the N days before"
        by a rolling N-day window's length; "the previous/prior <unit>", "the <unit> before (that)" back one unit of the same granularity ("the
        one before" also a rolling window's length); year re-anchoring to the same window in that year. Returns (s, e), a handoff code (unit
        mismatch), or None (no relative words; a year-long window re-anchored: the words keep their standalone meaning)."""
        move = lambda d, years: d.replace(year=d.year + years, day=min(d.day, 28) if d.month == 2 else d.day)
        mu, mn, my = PREV_UNIT.search(cur), PREV_N.search(cur), YEAR_REANCHOR.search(cur)
        if mu and my and (mu.group(1) or mu.group(2) or '').lower() == 'year' and gran != 'year':
            mu = None       # "the year before?" after a window shorter than a year re-anchors it (10g.25), it is no unit mismatch
        if mn:
            n, u = int(mn.group(1) or mn.group(3)), (mn.group(2) or mn.group(4)).lower()
            if gran != ('rolling', n, u): return 'unresolved_inherited_period'
            length = (e - s).days + 1; s, e = s - timedelta(days=length), e - timedelta(days=length)
        elif mu:
            u = (mu.group(1) or mu.group(2) or '').lower().replace('night', 'day') or (gran if isinstance(gran, str) else '')
            if isinstance(gran, tuple):                            # rolling: "the one before" shifts by the window length
                if mu.group(0).lower().strip() != 'the one before': return 'unresolved_inherited_period'
                length = (e - s).days + 1; s, e = s - timedelta(days=length), e - timedelta(days=length)
            elif gran != u: return 'unresolved_inherited_period'
            elif u in ('day', 'week', 'weekend'): d = timedelta(days=1 if u == 'day' else 7); s, e = s - d, e - d
            elif u == 'month': s = (s.replace(day=1) - timedelta(days=1)).replace(day=1); e = s.replace(day=_cal.monthrange(s.year, s.month)[1])
            else: s, e = move(s, -1), move(e, -1)
        elif my:
            if gran == 'year' or (isinstance(gran, tuple) and (gran[2] == 'month' and gran[1] >= 12)): return None   # a year-long window: standalone
            years = (int(my.group(1)) - s.year) if my.group(1) else -1
            if years == 0: return 'unresolved_inherited_period'
            s, e = move(s, years), move(e, years)
        else: return None
        return s, e

    def relative_followup(self, req, cur, hist, ref, H):
        """Spec v1.8 10g.25: shift the prior read's window. "the previous/prior <unit>", "the <unit> before (that)", "the one before":
        back one unit when the unit matches the prior window's granularity (a rolling N-unit window shifts by its length); year
        re-anchoring moves a window shorter than a year to that year. A unit mismatch or a follow-up that changes something else
        hands off; no prior window -> None (standalone meaning)."""
        # The prior read must state its own window; a prior follow-up that inherited it (and maybe changed the metrics) is left to the model.
        if len(hist) > 1 and not self._has_window_words(hist[-1], ref): return None
        prev = self.select({**req, '_prev': True, 'state': {**req['state'], 'current_request': hist[-1], 'recent_user_requests': hist[:-1]}})
        if prev['status'] != 'planned' or any(q.get('kind') != 'health' for q in prev['queries']): return None
        periods = {json.dumps(q['period'], sort_keys=True) for q in prev['queries']}
        if len(periods) != 1: return H('unresolved_inherited_period')
        p = json.loads(periods.pop()); win = self._prior_window(p, ref)
        if win is None: return None if p.get('kind') in ('all_history', None) else H('unresolved_inherited_period')
        rest = cur
        for rx in (PREV_UNIT, PREV_N, YEAR_REANCHOR): rest = rx.sub(' ', rest)
        if re.search(r'[A-Za-z]{3,}', FOLLOW_FILLER.sub(' ', rest)): return H('unresolved_inherited_period')   # also changes something else
        moved = self._shift_back(*win, cur)
        if moved is None or isinstance(moved, str): return moved and H(moved)
        s, e = moved
        if e > ref: return H('future_period_unavailable')
        qs = [{**q, 'period': {'kind': 'between', 'start_at': s.isoformat(), 'end_at': e.isoformat()}} for q in prev['queries']]
        return {'status': 'planned', 'reason_codes': [], 'queries': qs, 'confidence': prev.get('confidence', 1.0),
                'decision_confidence': prev.get('decision_confidence', 1.0)}

    def units_ago_reads(self, metrics, cur, ref, src, op_basis, labs=False):
        """Spec v1.8 10g.24: "N units ago" is a point in time, anchor = ref - N units; always trend. Daily metrics (fresh_days <= 14)
        read the calendar period around the anchor (a day, its Mon-Sun week, its month; N years -> that month N years back); sparse
        metrics read (anchor - fresh_days) .. anchor. "since N units ago" = relative from the anchor (amended 2026-09-28: weeks 7N days,
        years 12N months; Vita computes the window). One read per window. labs: the lab reports current at the anchor, (anchor - 365,
        the longest lab freshness) .. anchor, latest ordering (10g.33 item 8)."""
        m = UNITS_AGO_ANY.search(cur); since, n, u = bool(m.group(1)), m.group(2).lower(), UNIT_ABBR.get(m.group(3).lower(), m.group(3).lower())
        n = int(n) if n.isdigit() else num_words(n) or 1
        lab = lambda p: [{'kind': 'health', 'metrics': [], 'records': ['labs'], 'operation': 'latest', 'period': p, 'source': None, 'profile_fields': []}] if labs else []
        if since:
            p = {'kind': 'relative', 'amount': n * {'week': 7, 'year': 12}.get(u, 1), 'unit': 'days' if u in ('day', 'week') else 'months'}   # labs: whole days (contract_period)
            return ([{'kind': 'health', 'metrics': list(metrics), 'records': [], 'operation': 'trend', 'source': src, 'profile_fields': [], 'date_basis': 'observed_at',
                      'period': p}] if metrics else []) + lab(dict(p))
        anchor = ref - timedelta(days=n) if u == 'day' else ref - timedelta(weeks=n) if u == 'week' else _months_back(ref, n * (12 if u == 'year' else 1))
        groups = {}
        for x in metrics:
            fd = (metric_definition(x) or {}).get('fresh_days', 365)
            if fd <= 14:
                if u == 'day': w = (anchor, anchor)
                elif u == 'week': a0 = anchor - timedelta(days=anchor.weekday()); w = (a0, a0 + timedelta(days=6))
                else: a0 = anchor.replace(day=1); w = (a0, a0.replace(day=_cal.monthrange(a0.year, a0.month)[1]))
            else: w = (anchor - timedelta(days=fd), anchor)
            groups.setdefault(w, []).append(x)
        qs = []
        for (a0, b0), ms in sorted(groups.items()):
            one_day = a0 == b0
            night = sorted(set(ms) & set(NIGHT_METRICS)) if one_day else []
            for part, basis in ((night, 'sleep_end_day'), (sorted(set(ms) - set(night)), 'observed_at')):
                if part: qs.append({'kind': 'health', 'metrics': part, 'records': [], 'operation': 'trend', 'source': src, 'profile_fields': [],
                                    'period': {'kind': 'between', 'start_at': a0.isoformat(), 'end_at': min(b0, ref).isoformat()}, 'date_basis': basis})
        return qs + lab({'kind': 'between', 'start_at': (anchor - timedelta(days=365)).isoformat(), 'end_at': anchor.isoformat()})

    @staticmethod
    def _calendar_wording(cur, time_spans):
        low = cur.lower()
        # Classify the calendar request, not discourse or pieces of a time
        # phrase left outside a partial neural tag (10g.19).
        low = re.sub(r'(^|[.!?;]\s*)next\s*,\s*', r'\1', low)
        low = re.sub(r'\b(?:past|previous|last|next|upcoming)\s+(?:(?:\d+|' + NUM_WORDS + r')\s+)?(?:days?|weeks?|weekends?|months?|years?)\b', ' ', low)
        for t in time_spans: low = low.replace(t.lower(), ' ')          # "past 30 days" is a window, not completion wording
        has = lambda ws: any(re.search(r'(?<![\w-])' + re.escape(w) + r'(?![\w-])', low) for w in ws)
        return has(CAL_BASIS['done']), has(CAL_BASIS['due'])

    def _calendar_read_text(self, cur, runs=()):
        """Completion words in an unrelated context sentence do not filter appointments."""
        reads = self._reading_sentences(cur, runs)
        # A sentence without a repeated record noun can still constrain those
        # records: "I already completed them", "scheduled ones please".
        return ' '.join(m.group().strip() for m in self.SENTENCE.finditer(cur)
                        if m.span() in reads or (re.search(r'\b(?:they|them|it)\b|\b(?:those|these|ones?|that|this)\b(?=\s*(?:[,.!?;]|$|(?:were|was|are|is|have|had|has|already|please)\b))', m.group(), re.I)
                                               and any(self._calendar_wording(m.group(), []))))

    def calendar_bases(self, cur, period, ref, time_spans):
        """Date bases for a calendar read that shares the plan's window (a mixed plan): done/due wording picks one basis,
        neutral wording reads both unless the window lies entirely in the future (spec v1.8 10g.19)."""
        done, due = self._calendar_wording(cur, time_spans)
        future = period.get('kind') == 'between' and date.fromisoformat(str(period['start_at'])[:10]) > ref
        bases = ['last_done_date'] if done and not due else ['next_due_date'] if due or future or period.get('kind') == 'all_history' else ['next_due_date', 'last_done_date']
        return [(b, dict(period)) for b in bases]

    def calendar_reads(self, kind, unit, offset, amount, time_text, cur, ref, time_spans):
        """Calendar-only request (spec v1.8 10g.19): done wording -> last_done_date, most recent past occurrence; due wording ->
        next_due_date, next occurrence; neutral -> both bases over a past/current window, due only for a future one; month names
        resolve within the current calendar year; current periods are whole periods. Returns [(basis, period)] or a handoff code."""
        done, due = self._calendar_wording(cur, time_spans)
        if done and due: return 'unresolved_calendar_basis'
        mode = 'past' if done else 'next' if due else 'year'
        if kind in ('none', 'all_history') and not time_text: return [('last_done_date' if done else 'next_due_date', {'kind': 'all_history'})]
        if kind == 'rolling':
            p = resolve_period(kind, amount, unit, offset, ref, time_text)
            if p is None: return 'unsupported_period'
            return [(b, p) for b in (['last_done_date'] if done else ['next_due_date'] if due else ['next_due_date', 'last_done_date'])]
        if kind in ('calendar', 'to_date') and unit in ('day', 'week', 'month', 'year') and not re.search(r'\b(?:next|upcoming|coming|tomorrow)\b|\d', time_text or ''):
            if offset == 'previous':
                p = resolve_period(kind, amount, unit, offset, ref); s, e = date.fromisoformat(p['start_at']), date.fromisoformat(p['end_at'])
            else:           # the current period: the whole period for due/neutral wording, to date for done wording
                s = ref if unit == 'day' else ref - timedelta(days=ref.weekday()) if unit == 'week' else ref.replace(day=1) if unit == 'month' else ref.replace(month=1, day=1)
                e = ref if unit == 'day' else s + timedelta(days=6) if unit == 'week' else s.replace(day=_cal.monthrange(s.year, s.month)[1]) if unit == 'month' else s.replace(month=12, day=31)
                if done: e = ref
        else:
            try: span = parse_time_span(time_text or cur, ref, mode)
            except AmbiguousDate: return 'ambiguous_numeric_date'
            except InvalidDate: return 'invalid_or_out_of_range_date'
            if span is None: return 'unsupported_period'
            s, e = span
            if kind == 'since': e = ref
            elif kind == 'day': e = s
        if s > e: return 'unsupported_period'
        if done and s > ref: return 'future_period_unavailable'
        bases = ['last_done_date'] if done else ['next_due_date'] if due or s > ref else ['next_due_date', 'last_done_date']
        return [(b, {'kind': 'between', 'start_at': s.isoformat(), 'end_at': e.isoformat()}) for b in bases]

    @classmethod
    def _has_window_words(cls, t, ref):
        return cls._has_date(t, ref) or bool(parse_rolling(t)) or bool(PREV_UNIT.search(t) or PREV_N.search(t) or YEAR_REANCHOR.search(t) or UNITS_AGO_ANY.search(t)) or \
            bool(re.search(r'\b(?:this|last|past|previous|current)\s+(?:week|month|year|weekend|night)\b|\b(?:yesterday|today|tonight|to date|so far)\b', t, re.I))

    @staticmethod
    def _has_date(t, ref):
        try: return bool(parse_time_span(t, ref))
        except (AmbiguousDate, InvalidDate): return True

    SENTENCE = re.compile(r'(?:[^.?!;\n]|[.?!;](?![.?!;]*(?:\s|$)))+[.?!;]*')     # a stop ends a sentence only before a space or the end ("Whoop 4.0")

    @staticmethod
    def _bind_event(read_text, back, cur, dated, times, ref):
        """The read text with its reference to a dated side event replaced by the event's dates, or None (spec v1.8 10g.28, clarified):
        "...What was my HRV that day?" -> "on 2026-09-14"; "...steps during the trip?" -> the trip's range; "since then" -> since the
        event; "the day after/before" -> the next/previous day. One dated sentence with one date only; any other reference -> None."""
        spans = [(a, z) for a, z in times if any(x <= a < y for x, y in dated)]
        if len(dated) != 1 or len(spans) != 1: return None
        try: sp = parse_time_span(cur[spans[0][0]:spans[0][1]], ref)
        except (AmbiguousDate, InvalidDate): return None
        if not sp or sp[1] > ref: return None
        (s, e), w = sp, re.sub(r'\s+', ' ', back.group(0).lower())
        one, day = s == e, timedelta(days=1)
        new = (f'on {s.isoformat()}' if one and re.fullmatch(r'(?:on )?that (?:day|date)', w) else
               f'from {s.isoformat()} to {e.isoformat()}' if not one and w.startswith('during ') else
               f'since {s.isoformat()}' if w == 'since then' else
               f'on {(s + day).isoformat()}' if one and w in ('the day after', 'the next day', 'the following day') and s + day <= ref else
               f'on {(s - day).isoformat()}' if one and w in ('the day before', 'the previous day') else None)
        return None if new is None else read_text[:back.start()] + new + read_text[back.end():]

    def _reading_sentences(self, t, runs=()):
        """Sentences of t (char spans) that carry the read: those holding a tagged mention (runs, t-relative), or, with no tagged
        mention (history turns), a catalog name. Context sentences around a read are ignored for planning (spec v1.8 10g.28);
        one sentence, or none that reads -> all of them."""
        sents = [(m.start(), m.end()) for m in self.SENTENCE.finditer(t)]
        if len(sents) < 2: return sents          # one sentence (or none): it carries the read, nothing to pick
        low = t.lower()
        reads = [(a, z) for a, z in sents if (any(x < z and y > a for x, y in runs) if runs else any(rx.search(low[a:z]) for rx, _, _ in self.syn))]
        return reads if reads and len(sents) > 1 else sents

    @staticmethod
    def _intake(t):
        """Food-intake wording (spec v1.8 10g.30; lists in food_intake.json)."""
        low = FOOD_EXEMPT.sub(' ', t.replace('’', "'"))
        return bool(FOOD_ALWAYS.search(low) or any(p.search(low) for p in FOOD_PATTERNS)
                    or (FOOD_CAL.search(low) and FOOD_WITH_CAL.search(low) and not FOOD_BURN.search(low)))

    def _since(self, t, ref, reading, runs=()):
        """'since <event>' in t (spec v1.8 10g.29, 10g.31; lists in since_anchors.json) -> (handoff code or None, anchor or None).
        Medical context in the event or in its sentence hands off, even with a date; the read's own names (tagged mentions runs, or
        with none every catalog name: "fasting insulin", "sleep apnea index") are not context. Otherwise only phrases in the reading
        sentences count. A dated phrase keeps the normal window rules; anchor ('data',) = the metric's first reading, ('device',
        source) = the start of that device's series; account start, a restart or any other event without a date cannot be resolved."""
        anchors, sents = set(), [(x.start(), x.end()) for x in self.SENTENCE.finditer(t)]
        context = t
        for a, z in runs or [x.span() for rx, _, _ in self.syn for x in rx.finditer(t.lower())]: context = context[:a] + ' ' * (z - a) + context[z:]
        for m in SINCE_PHRASE.finditer(t):
            s0, s1 = next(((a, z) for a, z in sents if a <= m.start() < z), (0, len(t)))
            ev = SINCE_END.sub('', m.group(1)).strip().lower().replace('’', "'")
            dated = bool(ev) and self._has_window_words(ev, ref)
            if SA_MEDICAL.search(ev) or SA_MEDICAL.search(context[s0:s1]): return ('unbound_request_constraint' if dated else 'unresolved_temporal_phrase'), None
            if dated or SA_BACKREF.match(ev) or not any(a <= m.start() < z for a, z in reading): continue
            if SA_ACCOUNT.search(ev) or SA_RESTART.search(ev) or SA_INTEGRATION.search(ev): return 'unresolved_temporal_phrase', None   # backfills predate a connection too
            devs = {v for al, v in SA_DEVICES.items() if re.search(r'(?<![\w-])' + re.escape(al) + r'(?![\w-])', ev)}
            if devs and (SA_REPLACE.search(ev) or SA_VERSION.search(ev)): return 'unresolved_temporal_phrase', None   # "my new Oura", "Whoop 4.0": same source before the event
            if len(devs) == 1 and SA_DEVICE_VERB.search(ev): anchors.add(('device', devs.pop()))
            elif any(p.match(ev) for p in SA_DATA): anchors.add(('data',))
            else: return 'unresolved_temporal_phrase', None
        if len(anchors) > 1: return 'unresolved_temporal_phrase', None
        return None, next(iter(anchors), None)

    @classmethod
    def _no_window(cls, cur, offs, roles, base, cur_end, time_spans, ref):
        """The request states no window: every tagged time span is vague recency or a point-in-time word, the date/period parsers
        find nothing, and no time word is left once the metric/record/source mentions ("daily max heart rate", "sleep start time")
        and those words ("vita sleep score right now") are blanked out."""
        vague = [v.lower() for rx in (VAGUE_RECENCY, POINT_NOW, MOST_RECENTLY) for v in rx.findall(cur)]
        if any(not any(t.lower() in v for v in vague) for t in time_spans) or cls._has_window_words(VAGUE_RECENCY.sub(' ', cur), ref): return False
        rest = list(cur)
        for (a, z), r in zip(offs.tolist(), roles):
            if z > a and a >= base and z <= cur_end and ROLES[r] in ('metric', 'record', 'profile_field', 'source'): rest[a - base:z - base] = ' ' * (z - a)
        return not TIME_WORD.search(POINT_NOW.sub(' ', VAGUE_RECENCY.sub(' ', ''.join(rest))))

    def latest_window_split(self, req, cur, offs, roles, base, cur_end, H, trailing_daily=False):
        """One stated window next to a clause that asks for the latest value ("my latest LDL and my HbA1c from last year",
        "What's my TSH, and my free T4 from last year?") is two reads (spec v1.8 10g.6). Split at the one strong boundary between
        the window side and the latest side; each half is decoded on its own and both must plan (all or nothing). None = no split."""
        spans = [(a - base, z - base, ROLES[r]) for (a, z), r in zip(offs.tolist(), roles) if z > a and a >= base and z <= cur_end and ROLES[r] != 'O']
        vague = [m.span() for rx in (VAGUE_RECENCY, STATE_DAYS) for m in rx.finditer(cur)]      # state words name no window (10g.13)
        tw = [(a, z) for a, z, r in spans if r == 'time' and not any(v0 <= a and z <= v1 for v0, v1 in vague)]; ms = [(a, z) for a, z, r in spans if r in ('metric', 'record', 'profile_field')]
        if not tw or len(ms) < 2: return None
        t0, t1 = min(a for a, _ in tw), max(z for _, z in tw)
        cands, unclear = [], False
        for b in CLAUSE_BOUNDARY.finditer(cur):
            left, right = cur[:b.start()], cur[b.end():]
            if not any(z <= b.start() for _, z in ms) or not any(a >= b.end() for a, _ in ms): continue      # mentions on both sides
            if t1 <= b.start(): win, other = left, right
            elif t0 >= b.end(): win, other = right, left
            else: continue                                                     # the window straddles the boundary
            if SERIES_CUE.search(other): continue                       # "What was my average X, Y since June" asks for a series
            if re.fullmatch(r'\s*[,;]\s*(?:and|plus)\s+', b.group(0), re.I) and not re.match(r'(?:my|the|your|what|how|show|give|pull|bring|tell|get|list|display|plot|graph|chart|see|can|could|would|i)\b', cur[b.end():], re.I) \
                    and not LATEST_CUE.search(other): continue          # "What were my A, B, and C today?": one list, one window (10g.6)
            cue = LATEST_CUE.search(other) or WHAT_IS.search(other) or re.search(r"\bhow(?:'s|\s+is|\s+are)\b", other, re.I) or STATE_DAYS.search(other)
            strong = bool(re.search(r'[,;.?!]|\balso\b|\bthen\b', b.group(0))) or bool(LATEST_CUE.search(other)) or \
                bool(re.match(r'\s*(?:what|how)\b', cur[b.end():], re.I) and WHAT_NOW.search(other))   # a new question beside "what's my X" (10g.6)
            if strong and cue: cands.append(b)
            elif win is left and re.match(r'\s*my\b', right, re.I): unclear = True   # "RHR this month plus my VO2 max": which window does VO2 max take?
        # A trailing item with no window of its own after the windowed clause is one more item of that request and takes its window (10g.34 item 1, 10g.35
        # item 4, 10g.36 item 7: "steps this week and my HRV" = one read of both, this week); only daily items: a sparse one meets the rider (10g.26).
        if not cands: return H('multiple_periods_require_clause_binding') if unclear and not trailing_daily else None   # writers read these both ways (spec 10g.6: unclear -> handoff)
        if len(cands) > 1: return H('multiple_periods_require_clause_binding')        # unclear attachment
        b = cands[0]; qs, conf, dec = [], 1.0, 1.0
        for part in (cur[:b.start()], cur[b.end():]):
            part = part.strip(' ,;.&?!')
            sub = {**req, '_clause': True, '_split': True, 'state': {**req['state'], 'current_request': part}}
            res = self.select(sub)
            if res['status'] != 'planned' or any(q.get('kind') != 'health' for q in res['queries']):
                return H('multiple_periods_require_clause_binding')             # all or nothing: never drop a clause
            qs += res['queries']; conf = min(conf, res.get('confidence', 1.0)); dec = min(dec, res.get('decision_confidence', 1.0))
        if len(qs) > 8 or sum(len(q['metrics']) for q in qs) > 16: return H('plan_breadth_exceeded')
        return {'status': 'planned', 'reason_codes': [], 'queries': qs, 'confidence': conf, 'decision_confidence': dec}

    def per_clause(self, req, cur, text, offs, roles, base, cur_end, H, fallback='multiple_periods_require_clause_binding'):
        if req.get('_clause') or req.get('_cmp'): return H(fallback)
        if CMP_SAME.search(cur): return H(fallback)     # "the same period last month" moves the window before it (10g.38 item c): window_compare's, not a follow-up's
        tagged = [(a, z, ROLES[r]) for (a, z), r in zip(offs.tolist(), roles) if z > a and a >= base and z <= cur_end]
        cuts, prev, start = [], None, base
        for a, z, role in tagged:                    # a clause ends where its window ends and the next mention begins
            # ... unless the window is worded before its own mention ("This month's weight and last month's body fat"): then its clause
            # starts with the window.
            if prev == 'time' and role not in ('time', 'source') and not re.match(r'\s*(?:from|using|via|on)\b', text[a:], re.I): cuts.append(start if re.match(r"['’]s\b", text[a:a + 3]) else a)
            if role == 'time' and prev != 'time': start = a
            prev = role
        cuts = [c for c in cuts if c > base]
        if not cuts:   # windows the tagger missed: split on a comparison connector (closed-class words, not domain vocabulary)
            cuts = [base + m.start() for m in re.finditer(r'\b(?:vs\.?|versus|compared (?:to|with))\s', cur, flags=re.I)]
            # ... but a side with no window of its own ("my HRV compared to last week") compares with an unstated reference: a handoff
            # (spec v1.8 10g.38 "Hand off", item 6); a latest word beside the comparison word states the latest value (items 7, f).
            first = text[base:cuts[0]] if cuts else ''
            if cuts and not (CMP_NOW.search(first) or LATEST_WORDS.search(first) or any(names_time(text[a:z]) for a, z, role in tagged if role == 'time' and z <= cuts[0])): return H(fallback)
        bounds = [base] + cuts + [cur_end]
        clauses = [(re.sub(r'\s+(?:and|plus|also|&)$', '', text[x:y].strip(' ,;.&?!'), flags=re.I), x, y) for x, y in zip(bounds, bounds[1:])]
        clauses = [(c, x, y) for c, x, y in clauses if re.search(r'[A-Za-z0-9]', c)]          # punctuation alone is not a clause
        clauses = [(re.sub(r'^(?:and|plus|also|then|&|vs\.?|versus|compared (?:to|with))\s+', '', c, flags=re.I), x, y) for c, x, y in clauses if c]
        if len(clauses) < 2: return H(fallback)
        qs, conf, dec, prev, last_ms, per = [], 1.0, 1.0, [], None, []
        for c, x, y in clauses:   # a clause with only a window ("... vs last week") is a follow-up of the clause before it
            sub = {**req, '_clause': True, 'state': {**req['state'], 'current_request': c, 'recent_user_requests': prev}}
            own = {role for a, z, role in tagged if x <= a and z <= y}
            if 'time' in own and own & {'metric', 'record', 'profile_field'}:
                sub['state'] = {**sub['state'], 'recent_user_requests': []}
            res = self.select(sub)
            # A clause with its own mention and its own window needs nothing from the clause before it (10g.6: one read per clause).
            # When it does not plan as that clause's follow-up (its items leak into the whole-input view: "LDL last year and ApoB
            # this year"), it is read on its own.
            own = {role if role != 'time' or names_time(text[a:z]) else 'O' for a, z, role in tagged if x <= a and z <= y}
            if res['status'] != 'planned' and prev and 'time' in own and own & {'metric', 'record', 'profile_field'}:
                res = self.select({**sub, 'state': {**sub['state'], 'recent_user_requests': []}})
            if res['status'] != 'planned' or any(q.get('kind') != 'health' for q in res['queries']):
                return H(fallback)                                    # all or nothing: never drop a clause
            ms = sorted({m for q in res['queries'] for m in q.get('metrics') or []})
            if qs and ms and ms == last_ms and len(ms) > 1: return H(fallback)   # "HRV and RHR this week and last week": unclear attachment
            last_ms = ms
            qs += res['queries']; conf = min(conf, res.get('confidence', 1.0)); dec = min(dec, res.get('decision_confidence', 1.0)); prev = [c]
            per.append(res['queries'])
        if len(qs) > 8 or sum(len(q['metrics']) for q in qs) > 16: return H('plan_breadth_exceeded')
        # Two clauses that read the same items over the same window asked for two windows: one of them was not read as stated (a window
        # the tagger missed takes the clause before it; 10g.38: identical windows hand off). One clause's own reads may share items and window
        # (a calendar clause reads its two date bases, 10g.19).
        sig = lambda q: json.dumps([sorted(q.get('metrics') or []), q.get('records'), q.get('period'), q.get('source'), q.get('date_basis')], sort_keys=True)
        sigs = [{sig(q) for q in c} for c in per]
        if any(sigs[i] & sigs[j] for i in range(len(sigs)) for j in range(i)): return H(fallback)
        return {'status': 'planned', 'reason_codes': [], 'queries': qs, 'confidence': conf, 'decision_confidence': dec}

    @staticmethod
    def period_span(p, ref):
        """The (first, last) day a decoded period reads, a current period to date; None for all history (a latest read)."""
        k = p.get('kind')
        if k == 'between': return date.fromisoformat(str(p['start_at'])[:10]), date.fromisoformat(str(p['end_at'])[:10])
        if k == 'calendar':
            per = p.get('period')
            if per not in ('day', 'week', 'month', 'year'): return None
            return (ref if per == 'day' else ref - timedelta(days=ref.weekday()) if per == 'week' else ref.replace(day=1) if per == 'month' else ref.replace(month=1, day=1)), ref
        if k == 'relative':
            n, u = int(p['amount']), p['unit'].rstrip('s')
            return (ref - timedelta(days=n - 1) if u == 'day' else ref - timedelta(days=7 * n - 1) if u == 'week' else _months_back(ref, n * (12 if u == 'year' else 1)) + timedelta(days=1)), ref
        return None

    @classmethod
    def _same_shift(cls, p, m, ref):
        """Spec v1.8 10g.38 item c: "the same period/days/week last <unit>" (CMP_SAME match m) = the window p moved back one unit, keeping its
        to-date end: this month (09-01..09-26) vs the same period last month = 08-01..08-26; "the same days last week" after Mon-Wed = the Mon-Wed
        before; "the same week last year" aligns ISO week and weekday (2026-09-21..26 -> 2025-09-22..27). None: no dated window to move."""
        span = cls.period_span(p, ref)
        if span is None: return None
        (s, e), word, by = span, m.group(1).lower(), (m.group(2) or m.group(3)).lower()
        if by == 'week': return s - timedelta(days=7), e - timedelta(days=7)
        if by == 'month': return _months_back(s, 1), _months_back(e, 1)
        if word == 'week':
            if s.weekday() != 0 or (e - s).days > 6: return None
            y, w, d = s.isocalendar()
            try: a = date.fromisocalendar(y - 1, w, d)
            except ValueError: return None                  # ISO week 53 in a year that has none
            return a, a + (e - s)
        back = lambda d: d.replace(year=d.year - 1, day=28) if (d.month, d.day) == (2, 29) else d.replace(year=d.year - 1)
        return back(s), back(e)

    def _cmp_windows(self, cur, offs, roles, base, cur_end, ref):
        """The windows of a comparison in cur, in order ([{'a', 'z', 'kind', 'm'}], cur offsets; kind abs|now|rel|same), and the mention spans
        (spec v1.8 10g.38). Tagged time runs as whole words, grown or completed by the latest, relative and "same period" windows the tagger
        misses; a run that lists several windows ("June, July and August", "this week vs last week") is split at its list words. A day range
        ("Monday to Wednesday", "Sept 3-9") stays one window (10g.36.6), and so does "from X to Y" unless both ends are periods (item 3)."""
        tok = [(a - base, z - base, ROLES[r]) for (a, z), r in zip(offs.tolist(), roles) if z > a and a >= base and z <= cur_end]
        runs = lambda names: [[a, z] for a, z in self._role_runs(tok, names)]
        ms = [tuple(x) for x in runs({'metric', 'record', 'profile_field'})]
        spans = []
        for a, z in runs({'time'}):
            while a > 0 and cur[a - 1].isalnum(): a -= 1
            while z < len(cur) and cur[z].isalnum(): z += 1
            left = re.search(r"\b(?:this|last|past|previous|current)\s+$", cur[:a], re.I) if re.match(r"(?:week|weekend|month|year|night)\b", cur[a:z], re.I) else None
            right = re.match(r"\s+(?:week|weekend|month|year|night)\b", cur[z:], re.I) if re.search(r"\b(?:this|last|past|previous|current)$", cur[a:z], re.I) else None
            spans.append([left.start() if left else a, z + right.end() if right else z])     # "last | week", "this | month": the tagger stopped inside the phrase
        for rx in (CMP_REL, CMP_SAME, CMP_NOW, CMP_STD):
            for m in rx.finditer(cur):
                if rx is CMP_NOW and re.fullmatch(r"(?:^|.*[.?!;]\s*)(?:(?:and|so|ok|okay|but)\s+)?", cur[:m.start()], re.I | re.S): continue   # a leading "now" is a discourse marker
                if rx is CMP_NOW and re.search(r"\b(?:up\s+to|up\s+until|until|till|til|to|through|thru|from|since|as\s+of|by)\s+$", cur[:m.start()], re.I): continue   # "up to now": a range end
                spans.append([m.start(), m.end()])
        spans.sort(); merged = []
        for a, z in spans:
            if merged and a < merged[-1][1]: merged[-1][1] = max(z, merged[-1][1])
            else: merged.append([a, z])
        # Windows count in the sentences of the read (with a mention) and in sentences that only continue it ("Also show the week before.");
        # a time phrase in any other sentence is context ("I started a new job two weeks ago.", "The kids are back at school now.": 10g.28).
        context = []
        for x in self.SENTENCE.finditer(cur):
            x0, x1 = x.span()
            if any(x0 <= a < x1 for a, _ in ms): continue
            rest = ''.join(' ' if any(a <= i < z for a, z in merged) else c for i, c in zip(range(x0, x1), cur[x0:x1]))
            if not CMP_JOIN.fullmatch(rest): context.append((x0, x1))
        ws = []
        for a, z in merged:
            if any(x <= a < y for x, y in context): continue
            for p, q in self._split_windows(cur, a, z, ref):
                m = CMP_EDGE.search(cur[p:q])        # list words the tagger took into the window ("last month up" of "last month up or down vs ...")
                while m and m.group(0):
                    p, q = (p + m.end(), q) if m.start() == 0 else (p, p + m.start()); m = CMP_EDGE.search(cur[p:q])
                t = cur[p:q]
                if not names_time(t) or FREQUENCY.fullmatch(t.strip()) or WHOLE_DAY.fullmatch(t.strip()) or VAGUE_RECENCY.fullmatch(t.strip()): continue
                pre = re.search(r"\b(?:in|on|for|during|over|at|from|since)\s+$", cur[:p], re.I)
                if pre and not re.match(r"(?:in|on|for|during|over|at|from|since|between)\b", t, re.I) and not CMP_NOW.fullmatch(t.strip()): p = pre.start()   # "in July", "for last week"
                bare = re.sub(r"^(?:in|on|for|during|over|at|from|since)\s+", '', cur[p:q].strip(), flags=re.I)
                kind = next((k for k, rx in (('now', CMP_NOW), ('same', CMP_SAME), ('rel', CMP_REL)) if rx.fullmatch(bare)), 'abs')
                ws.append({'a': p, 'z': q, 'kind': kind, 'm': CMP_SAME.fullmatch(bare) if kind == 'same' else None})
        return ws, ms

    @staticmethod
    def _role_runs(tok, names):
        """Contiguous runs of tokens whose role is in names: [(a, z)]."""
        out, last = [], None
        for j, (a, z, r) in enumerate(tok):
            if r in names:
                if last == j - 1: out[-1][1] = z
                else: out.append([a, z])
                last = j
        return [tuple(x) for x in out]

    @staticmethod
    def _period_end(t, cue):
        """An end of "from X to Y" that is a period (10g.38 item 3): "last week", "this month", "the week before"; with a comparison or change
        word in the request also a month name or a year ("Compare my calories from July to August"). Day and date ends make one span."""
        t = re.sub(r"^(?:from|in|during)\s+", '', t.strip(), flags=re.I)
        return bool(re.fullmatch(r"(?:(?:this|last|previous|the\s+previous|current|the\s+current)\s+(?:week|weekend|month|year)|(?:the\s+)?(?:week|weekend|month|year)\s+before(?:\s+that)?)", t, re.I)
                    or cue and re.fullmatch(r"(?:" + '|'.join(_MONTHS) + r")(?:\s+(?:19|20)\d{2})?|(?:19|20)\d{2}", t, re.I))

    def _split_windows(self, cur, a, z, ref):
        """One merged time span as its windows: split at list words ("June, July and August", "this week vs last week") when every part names
        a time; a day range stays whole; "from X to Y" splits only between two periods (_period_end)."""
        t = cur[a:z]
        try: day_range = parse_range(t, ref) is not None
        except (AmbiguousDate, InvalidDate): day_range = False
        try: n = len(_dates_in(t, ref))
        except (AmbiguousDate, InvalidDate): n = 0
        n += sum(len(rx.findall(t)) for rx in (CMP_REL, CMP_SAME, CMP_NOW)) + len(re.findall(r"\b(?:past|last|previous|prior|trailing)\s+(?:\d+|" + NUM_WORDS + r")\s+(?:day|week|month|year)s?\b", t, re.I))
        if day_range or n < 2 or re.match(r"\s*between\b", t, re.I): return [(a, z)]     # one date ("July 15, 2025"), one range, one window
        parts, x = [], 0
        for m in re.finditer(r"\s*(?:,\s*(?:and\s+|or\s+)?|;\s*|\s+(?:and\s+then|and|or|vs\.?|versus|then|&|\+|compared\s+(?:to|with)|against|relative\s+to|next\s+to)\s+)", t, re.I):
            parts.append((x, m.start())); x = m.end()
        parts.append((x, len(t)))
        out = []
        cue = bool(re.search(r"\b(?:compar\w*|chang\w*|improv\w*|difference|vs\.?|versus)\b", cur, re.I))
        for p, q in parts:
            r = re.search(r"\s+(?:to|through|thru|till|til|until)\s+", t[p:q], re.I)
            # "from July to August" itself supplies the comparison cue when
            # both ends are periods (10g.38.3). Day/date ends remain one span;
            # "through" without a comparison cue is still a continuous range.
            from_to = r and r.group().strip().lower() == 'to' and (re.match(r'\s*from\s+', t[p:q], re.I) or re.search(r'\bfrom\s*$', cur[:a + p], re.I))
            period_cue = cue or bool(from_to)
            if r and self._period_end(t[p:p + r.start()], period_cue) and self._period_end(t[p + r.end():q], period_cue): out += [(p, p + r.start()), (p + r.end(), q)]
            else: out.append((p, q))
        if len(out) > 1 and not all(names_time(t[p:q]) for p, q in out): return [(a, z)]
        return [(a + p, a + q) for p, q in out if t[p:q].strip()]

    def cmp_guard(self, text, ws=(), side=''):
        """A comparison that stays a handoff (spec v1.8 10g.38 "Hand off"; 10g.28 classes 5, 6, 9, 2, 3): a relationship or cause; a reference that
        is no stated window ("than usual", "vs my baseline", "compared to before"; "vs my average" with no window after it); another person or a
        norm; a normative judgement; medical context or an advice, judgement or explanation ask (the repo's 10g.28 lists: CMP_CARE and CMP_ASK in
        the comparison's sentences, text; CMP_CARE_SIDE and CMP_ASK_SIDE in the request's other sentences, side: "Compare my RHR this week and last
        week; my doctor asked me to", "My doctor asked me to keep an eye on it. Compare ..."). Returns the handoff code or None. ws: windows, as
        (start, end) in text."""
        if CMP_RELATION.search(text): return 'relationship_question'
        person = re.search(r"\b(?:wife|husband|partner|brother|sister|son|daughter|mom|mum|mother|dad|father|friend|boyfriend|girlfriend|colleague|coworker|kids?|child)(?:'s|s')?\b", text, re.I)
        if person: return 'other_person'
        if CMP_NORM.search(text) or CMP_JUDGE.search(text) or CMP_IMPLICIT.search(text): return 'unbound_request_constraint'
        # A style or language instruction and the verb that introduces it ("reply in a table", "answer in Romanian") keep a multi-read plan (10g.28
        # class 3 [refined 10g.38]): no instruction ask of the repo's lists ("reply", "respond").
        said, other = (CMP_STYLE.sub(' ', CMP_STYLE_ASK.sub(' ', CMP_PLAIN.sub(' ', CMP_SUPPLEMENT.sub(' ', canonicalize(t))))) if t.strip() else '' for t in (text, side))
        if CMP_CARE.search(said) or CMP_ASK.search(said) or CMP_CARE_SIDE.search(other) or CMP_ASK_SIDE.search(other): return 'unbound_request_constraint'
        for m in CMP_AVERAGE_REF.finditer(text):     # "steps today vs my average last week" names its window (item f); "vs my average" does not
            if not any(a >= m.end() and len(text[m.end():a].split()) <= 3 for a, _ in ws): return 'unbound_request_constraint'
        return None

    @staticmethod
    def _cmp_residue(t):
        """What is left of t, a clause beside a comparison's reads, once the words that may stand there are removed: discourse openers and
        pleasantries; a format or language instruction and a verb that only introduces it (the repo's style list, CMP_STYLE_ASK: "answer in
        Romanian"; a multi-read plan keeps it, 10g.28 class 3 [refined 10g.38]); the
        comparison scaffolding (CMP_FRAME, CMP_WHICH, CMP_REST: "how do they compare", "which week was higher", "any difference"), comparative words
        (CMP_MORE); a named device; read words and closed-class words (CMP_REST); time words (a window the tagger ended early: "the week before |
        last"). Any word left asks for something besides the reads."""
        t = CMP_WHICH.sub(' ', CMP_STYLE.sub(' ', CMP_STYLE_ASK.sub(' ', CMP_PLAIN.sub(' ', CMP_LEAD.sub(' ', canonicalize(t))))))
        for rx in CMP_FRAME: t = rx.sub(' ', t)
        t = CMP_REST.sub(' ', SRC_NAME.sub(' ', CMP_MORE.sub(' ', t)))
        return ' '.join(w for w in t.split() if not names_time(w))

    def _cmp_extra(self, cur, lo, hi, t0, ms):
        """A further ask beside a same-metric comparison (window_compare; spec v1.8 10g.38 "Hand off", item 6; 10g.28 classes 3, 7, 8, 9): a
        clause of the comparison's sentence outside its predicate - after it (from t0: "..., what happened?", "... and give me advice", "..., any
        idea why?") or before its list with no mention ("Quick question, ...") - that keeps a word once _cmp_residue has removed what may stand
        beside reads; and any other sentence that asks for something (no statement: CMP_STATEMENT, CMP_ADDRESS) and keeps such a word ("Any idea
        why?", "Give me advice."). The allowed words are listed, not the asks. A sentence that states something is plain context (10g.28);
        cmp_guard checks it for medical context and judgement."""
        left = lambda x, y: bool(re.search(r"[a-z]", self._cmp_residue(cur[x:y])))
        mentionless = lambda x, y: [c.span() for c in CMP_CLAUSE.finditer(cur, x, y) if not any(c.start() <= a < c.end() for a, _ in ms)]
        first = min([lo] + [a for a, _ in ms])
        sents = [x.span() for x in self.SENTENCE.finditer(cur)]
        s0 = next((x for x, y in sents if x <= first < y), 0)
        e0 = next((y for x, y in sents if x < t0 <= y), len(cur))       # the end of the sentence the comparison's predicate ends in
        if left(t0, e0) or any(left(x, y) for x, y in mentionless(s0, lo) + (mentionless(hi, t0) if t0 > hi else [])): return True
        for x, y in sents:
            if s0 <= x < e0 or (CMP_STATEMENT.match(cur[x:y]) and not CMP_ADDRESS.search(cur[x:y])): continue
            if left(x, y): return True
        return False

    def _name_blank(self, t):
        """t with its catalog names blanked: whole synonyms and abbreviations of the vocabulary (plurals too), item keys ("ferritin"), words with a digit,
        words in capitals ("HR"). What is left of a tagged mention names no item."""
        out, low, keys = list(t), t.lower(), {k for _, k, _ in self.items}
        spans = [m.span() for rx, _, _ in self.syn for m in rx.finditer(low)] + [m.span() for m in re.finditer(r"[\w-]+", t) if re.search(r"\d", m.group(0))
                 or (len(m.group(0)) > 1 and m.group(0).isupper()) or re.sub(r"(?:es|s)$", '', m.group(0).lower()) in self.syn_names or m.group(0).lower()[:-1] in self.syn_names   # plurals too
                 or m.group(0).lower() in keys]
        for a, z in spans: out[a:z] = ' ' * (z - a)
        return ''.join(out)

    def _direction(self, s, own, alone):
        """A direction or a change stated of the metric set in the sentence s (spec v1.8 10g.38 item 6: with one or no stated window a verdict hands
        off in v2), whatever its opening (VERDICT_CHANGE): "HRV up this week?", "HRV this week, up or down?", "Up or down: my HRV this week", "My
        HRV is up this week?", "My HRV went up this week, right?", "Tell me my HRV went up this week", "Did my HRV come back this week?", "Has my
        weight gotten back on track this month?". s is a sentence about the metric set (own: its mentions, as (start, end) in s) or with a pronoun
        subject ("Has it gone up?"); not a wh-question unless it offers the directions (VERDICT_ALT), and not a statement beside another sentence
        (plain context, 10g.28). The names in its mentions, device names and the clauses whose subject is a person are no part of it."""
        if not own and not VERDICT_PRONOUN.match(s): return False
        if VERDICT_WH.match(s) and not VERDICT_ALT.search(s): return False
        if not alone and CMP_STATEMENT.match(s) and not CMP_ADDRESS.search(s): return False
        blank = lambda m: ' ' * len(m.group(0))
        t = s
        for a, z in own: t = t[:a] + self._name_blank(t[a:z]) + t[z:]
        t = VERDICT_ADVERBIAL.sub(blank, SRC_NAME.sub(blank, t))
        for c in CMP_CLAUSE.finditer(t):
            if VERDICT_PERSON.match(c.group(0)) and not any(c.start() <= a < c.end() for a, _ in own): t = t[:c.start()] + blank(c) + t[c.end():]
        for m in VERDICT_CHANGE.finditer(t):
            if re.fullmatch(r"up|down", m.group(0), re.I) and VERDICT_PHRASAL.search(t[:m.start()]): continue
            if re.search(r"back$", m.group(0), re.I) and VERDICT_RECORD.search(s): continue      # a result that came back is a read
            return True
        return False

    def _verdict(self, cur, ms, times, cmp=False, direction=False):
        """A yes/no question about the metric set whose predicate is no plain read (spec v1.8 10g.38 item 6: a verdict with one or no stated window
        hands off in v2, the direct template cannot state it): "Did my HRV change this week?", "Are my steps down this week?", "Has my sleep score
        stabilized?", "Any change in my HRV this week?", "Has there been any change in my RHR since last week?", "How much has my weight changed
        since January?", "... tell me if my HRV recovered". Structural: a sentence that opens so about the metric set (VERDICT_OPEN or an embedded
        if/whether with the metric set as subject, never a pronoun; VERDICT_NP, a noun phrase of a change before it) and keeps a word once the
        names in its mentions, its times, device names and the words a plain read may add (PLAIN_READ) are removed: "Is my HRV data from last week
        available?", "Has my weight been logged this month?", "Any workouts this week?", "Any checkups scheduled for next week?", "How much do I
        weigh?" and "How much was my daily average heart rate last Tuesday?" are reads. With cmp (window_compare: a comparison of
        stated windows) the comparative words and the comparison scaffolding go too ("Is my HRV higher this week than last week?", item 6), so
        the complement of a yes/no comparison is checked before its clauses drop the auxiliary ("Is my HRV worryingly low this week vs last
        week?"). With direction (the plan path, not a comparison's clause) also a direction or change stated of the metric set whatever the
        sentence's opening (_direction). ms, times: (start, end) in cur."""
        sents = [s.span() for s in self.SENTENCE.finditer(cur)]
        alone = sum(bool(re.search(r"[a-z]", cur[x:y], re.I)) for x, y in sents) == 1
        for x, y in sents:
            own = [(a, z) for a, z in ms if x <= a < y]
            if direction and self._direction(cur[x:y], [(a - x, z - x) for a, z in own], alone): return True
            if not own: continue
            np_ = VERDICT_NP.match(cur[x:own[0][0]])
            starts = [x + np_.start('n')] if np_ else []          # a noun phrase of a change: the noun is the predicate
            for m in [VERDICT_OPEN.match(cur, x)] + list(VERDICT_IF.finditer(cur, x, y)):
                if not m: continue
                e = m.end(); nxt = next((a for a, _ in own if a >= e), None)
                # The subject is the metric set: "my"/"the" and a device name at most before the mention; not a pronoun ("did I", "do you", "How
                # much do I weigh?" with "I weigh" tagged as the mention) and not "there" (an existence question: VERDICT_NP).
                if nxt is None or re.match(r"\s*(?:i|i'm|you|u|we|they|he|she|it|there)\b", cur[e:], re.I) or \
                        re.search(r"[a-z]", re.sub(r"\b(?:my|the|our|your)\b", ' ', SRC_NAME.sub(' ', cur[e:nxt].lower()))): continue
                starts.append(e)
            for e in starts:
                t = ''.join(' ' if any(a <= i < z for a, z in times) else c for i, c in zip(range(e, y), cur[e:y]))
                for a, z in own:      # the names in a mention; the tagger may take the predicate into it ("steps | down", "HRV | recovered")
                    if z > e: t = t[:max(a, e) - e] + self._name_blank(t[max(a, e) - e:z - e]) + t[z - e:]
                rest = CMP_PLAIN.sub(' ', canonicalize(t))
                if cmp:
                    rest = CMP_WHICH.sub(' ', rest)
                    for rx in CMP_FRAME: rest = rx.sub(' ', rest)
                    rest = CMP_REST.sub(' ', CMP_MORE.sub(' ', rest))
                rest = PLAIN_READ.sub(' ', SRC_NAME.sub(' ', rest))
                if any(not names_time(w) for w in rest.split()): return True
        return False

    def window_compare(self, req, cur, offs, roles, base, cur_end, H):
        """Same metric set across stated windows (spec v1.8 10g.38): one read per window, whatever the phrasing (item 2) - a list ("HRV this week
        vs last week", "compare my steps this month with last month", "HRV in June, July and August", item g; fronted: "This week vs last week:
        HRV", item h), the metric restated ("RHR last week vs RHR the week before"), a second clause or sentence naming only a window ("Show my
        steps last week. Also show the week before.", "What was my HRV last week? What about the week before?", item 5), factual comparatives
        ("is my HRV higher this week than last week?", "which was higher, my RHR in June or July?", "did my RHR improve from last month to this
        month?", items 6, a, d) and a latest value beside a period ("What's my current RHR vs last month?", item 7). Each window's clause is the
        request with the list replaced by that window and the comparison scaffolding removed, decoded on its own; "the week before (that)"
        (item 1) and "the same period last month" (item c) move the window before it. All clauses must plan (10g.6, all or nothing), with one
        metric set and source; identical or overlapping windows hand off (a single day or latest value inside a period is part vs whole,
        item f). Returns a plan, a handoff, or None: no such comparison, the caller decodes the request as before."""
        if req.get('_clause') or req.get('_cmp'): return None
        ref = date.fromisoformat(req['state']['reference_date'])
        ws, ms = self._cmp_windows(cur, offs, roles, base, cur_end, ref)
        # A point-in-time word counts as a window only when a comparison or list word (CMP_LINK, mentions left out) joins it to a window beside it
        # (item 7). Otherwise the request compares no windows: "What was my most recent VO2 max in August?" and "Show my steps for this week now"
        # are one read (10g.15, 10g.32 item 9).
        blank = lambda x, y: ''.join(' ' if any(a <= i < z for a, z in ms) else cur[i] for i in range(x, y))
        ws = [w for i, w in enumerate(ws) if w['kind'] != 'now' or any(CMP_LINK.search(blank(v['z'], w['a']) if v['z'] <= w['a'] else blank(w['z'], v['a']))
                                                                      for v in ws[max(0, i - 1):i] + ws[i + 1:i + 2] if v['kind'] != 'now')]
        if len(ws) < 2 or not ms: return None
        att = None      # a point-in-time word right before its mention ("my current RHR") is that mention's latest read
        for w in ws:
            nxt = next(((a, z) for a, z in ms if a >= w['z']), None)
            if w['kind'] == 'now' and nxt and re.fullmatch(r"\s*", cur[w['z']:nxt[0]]):
                if att: return None
                att = (w, nxt)
        rest = [w for w in ws if not att or w is not att[0]]
        if not rest: return None
        lo, hi = rest[0]['a'], rest[-1]['z']
        inside, post_m = [m for m in ms if lo <= m[0] < hi], [m for m in ms if m[0] >= hi]
        gaps = [cur[x['z']:y['a']] for x, y in zip(rest, rest[1:])]
        cue = bool(re.search(r"\b(?:compar\w*|chang\w*|improv\w*|difference|vs\.?|versus)\b", cur, re.I))
        for x, y, g in zip(rest, rest[1:], gaps):           # "from X to Y" / "X - Y" between two windows: two reads only between periods (item 3)
            if re.search(r"\bcompar\w*\b", cur[:x['a']], re.I) and not re.match(r"from\b", cur[x['a']:x['z']], re.I) and re.fullmatch(r"\s*to\s+", g, re.I): continue   # "compare A to B"
            from_to = g.strip().lower() == 'to' and (re.match(r'\s*from\s+', cur[x['a']:x['z']], re.I) or re.search(r'\bfrom\s*$', cur[:x['a']], re.I))
            # Bare named months/years also describe continuous date ranges in
            # existing requests. Require a comparison cue to choose separate
            # periods; relative endpoints (last week to this week) are explicit.
            if from_to and not cue and not (self._period_end(cur[x['a']:x['z']], False) and self._period_end(cur[y['a']:y['z']], False)):
                return H('unbound_request_constraint')
            period_cue = cue or bool(from_to)
            if re.fullmatch(r"\s*(?:(?:to|through|thru|till|til|until|up\s+to|up\s+until)\s+(?:in\s+|the\s+)?|[-\u2013\u2014]\s*)", g, re.I) and not (
                    self._period_end(cur[x['a']:x['z']], period_cue) and self._period_end(cur[y['a']:y['z']], period_cue)): return None
        if re.search(r"\bbetween\s*$", cur[:lo], re.I) and not re.search(r"\bdifference\s+between\s*$", cur[:lo], re.I): return None   # "between X and Y": a range
        # The clause of each window, as (before, window, after): shared subject (all mentions outside the list, list words between windows) or
        # the mention restated in each clause ("RHR last week vs RHR the week before"); a window-only clause takes the clause before it.
        clauses = []
        if not inside:
            if not all(CMP_JOIN.fullmatch(g) for g in gaps): return None
            pre, post = cur[:lo], cur[hi:]
            if att:
                (nw, (ma, mz)) = att
                if mz > lo or any(mz <= a < lo for a, _ in ms if (a, _) != (ma, mz)) or not CMP_JOIN.fullmatch(cur[mz:lo]): return None
                clauses.append(('now', cur[:mz], '', post, None))
                pre = cur[:nw['a']] + cur[nw['z']:mz] + ' '
            if not re.search(r"[A-Za-z]", self._cmp_clean(pre, False)) and post_m:     # fronted windows: the window goes where the question ends
                body = self._cmp_clean(post, False).strip(' ,:;-'); end = re.search(r"[?.!]*\s*$", body)
                pre, post = body[:end.start()] + ' ', body[end.start():]
            lead = re.match(r"(?:in|on|for|during|over)\s+", cur[rest[0]['a']:rest[0]['z']], re.I)     # "in August vs September": the list shares its preposition
            for w in rest:
                wt = cur[w['a']:w['z']]
                clauses.append((w['kind'], pre, (lead.group(0) if lead and w['kind'] == 'abs' and not re.match(r"(?:in|on|for|during|over|at|from|since|the|this|last|past|previous|current)\b", wt, re.I) else '') + wt, post, w))
        else:
            if post_m or att or not any(m[1] <= lo for m in ms): return None
            post = cur[hi:]
            clauses.append((rest[0]['kind'], cur[:lo], cur[lo:rest[0]['z']], post, rest[0]))
            name = lambda a, z: re.sub(r"\W+", ' ', cur[a:z].lower()).strip()
            for w0, w, g in zip(rest, rest[1:], gaps):
                if any(w0['z'] <= a < w['a'] for a, _ in ms):
                    # A relative window moves the read before it (_cmp_reads); after another item ("steps last week and HRV the week before")
                    # the items are that clause's own: per-clause reads (10g.6, item i; the window relative to the clause before, 10g.25).
                    if w['kind'] in ('rel', 'same') and {name(a, z) for a, z in ms if w0['z'] <= a < w['a']} != {name(a, z) for a, z in ms if z <= lo}: return None
                    lead = re.match(r"(?:[\s,;:.?!&+/\u2013\u2014-]|\b(?:and|or|vs|versus|v|compared?|compares|to|with|against|relative|next|than|then|also|plus|as|well|opposed)\b)*", g, re.I)
                    clauses.append((w['kind'], g[lead.end():], cur[w['a']:w['z']], post, w))
                elif CMP_JOIN.fullmatch(g): clauses.append((w['kind'], clauses[-1][1], cur[w['a']:w['z']], post, w))
                else: return None
        # A comparison that stays a handoff, in the sentences of the comparison (those with a mention or one of its windows); the request's other
        # sentences are context (10g.28), checked for medical context and judgement only.
        sents = [(x.start(), x.end()) for x in self.SENTENCE.finditer(cur)]
        said = [(x, y) for x, y in sents if any(x <= a < y for a, _ in ms + [(w['a'], w['z']) for w in ws])]
        code = self.cmp_guard(' '.join(cur[x:y] for x, y in said), [(w['a'], w['z']) for w in ws] if said == [(0, len(cur))] else [],
                              side=' '.join(cur[x:y] for x, y in sents if (x, y) not in said))
        if code: return H(code)
        # A further ask beside the comparison is no part of its reads (10g.38 "Hand off", item 6; 10g.28 classes 3, 7, 8, 9): "Compare my weight this
        # month and last month, am I on track?", "HRV this week vs last week, what happened?", "Compare my steps this week and last week and give me
        # advice". The comparison's own predicate ends with its list, or, with its windows fronted ("For this week and last week, what was my
        # RHR?"), with its last mention. A yes/no comparison asks only a factual comparative (item 6): "Is my RHR this week vs last week a red flag?"
        # and "Is my HRV worryingly low this week vs last week?" hand off.
        pre_m = [m for m in ms if m[1] <= lo]
        t0 = hi if inside or pre_m else max(z for _, z in post_m)
        if self._cmp_extra(cur, lo, hi, t0, ms) or self._verdict(cur, ms, [(w['a'], w['z']) for w in ws], cmp=True): return H('unbound_request_constraint')
        if not inside and pre_m and post_m: return H('multiple_periods_require_clause_binding')   # "HRV this week vs last week and my RHR": unclear attachment (10g.6)
        # Decode each clause on its own; comparative words and a yes/no auxiliary ("is my HRV higher this week") are dropped, from every clause
        # alike, only when a clause does not plan with them.
        dec = lambda t: self.select({**req, '_cmp': True, 'state': {**req['state'], 'current_request': t, 'recent_user_requests': []}})
        for more in (False, True):
            got = self._cmp_reads(clauses, more, dec, ref)
            if not (isinstance(got, str) and more is False and any(CMP_MORE.search(b + ' ' + a) or CMP_YESNO.match(b) for _, b, _, a, _ in clauses)): break
        # A clause that cannot be read as one window of the same items, where the clauses name different items ("What's on my calendar this week
        # and what was my HRV last week?"), is left to per-clause reading (10g.6, item i): no same-metric comparison was failed.
        if isinstance(got, str): return None if inside and len({re.sub(r"\W+", ' ', cur[a:z].lower()).strip() for a, z in ms}) > 1 else H(got)
        reads, periods, conf, dconf = got
        items = lambda qs: json.dumps([sorted({m for q in qs for m in q.get('metrics') or []}), sorted({r for q in qs for r in q.get('records') or []}), sorted({q.get('source') or '' for q in qs})])
        if len({items(r) for r in reads}) != 1:
            return None if inside else H('multiple_periods_require_clause_binding')     # different items per window: per-clause reads (10g.6, item i)
        if not inside and len(ms) > 1 and all(re.fullmatch(r"(?:[\s,&]|\band\b)*", g, re.I) for g in gaps):
            return H('multiple_periods_require_clause_binding')      # "HRV and steps last week and today": unclear attachment (10g.6); "vs" compares
        for i in range(len(periods)):                        # identical or overlapping windows hand off; a day or the value now inside a period is part vs whole (item f)
            for j in range(i):
                a, b = periods[i], periods[j]
                if a is None and b is None: return H('unbound_request_constraint')
                if a is None or b is None: continue
                if a == b or (a[0] <= b[1] and b[0] <= a[1] and a[0] != a[1] and b[0] != b[1]): return H('unbound_request_constraint')
        qs = [q for r in reads for q in r]
        if len(qs) > 8: return H('plan_breadth_exceeded')
        return {'status': 'planned', 'reason_codes': [], 'queries': qs, 'confidence': conf, 'decision_confidence': dconf}

    def _cmp_reads(self, clauses, more, dec, ref):
        """The reads of each window's clause (window_compare): (reads per window, (first, last) day per window or None for the value now,
        confidence, decision confidence), or the handoff code of the first clause that cannot be read (all or nothing, 10g.6)."""
        reads, periods, conf, dconf = [], [], 1.0, 1.0
        for kind, before, wtext, after, w in clauses:
            if kind in ('rel', 'same'):
                # Shifting a prior read is valid only for the same clause with
                # a pure relative window. Copying it must not erase a newly
                # stated source, metric, filter or other constraint.
                if reads:
                    previous = clauses[len(reads) - 1]
                    scope = lambda b, a: re.sub(r'\W+', ' ', canonicalize(self._cmp_clean(b + ' ' + a, more))).strip()
                    if scope(before, after) != scope(previous[1], previous[3]):
                        return 'unbound_request_constraint'
                grammar = CMP_REL if kind == 'rel' else CMP_SAME
                window_words = re.sub(r'^(?:for|in|on|during|over)\s+', '', wtext.strip(' ,.;?!'), flags=re.I)
                window_words = re.sub(r'\s+(?:as\s+well|too)$', '', window_words, flags=re.I)
                if not grammar.fullmatch(window_words):
                    return 'unbound_request_constraint'
                if not reads or periods[-1] is None: return 'unresolved_inherited_period'     # nothing dated before it to move
                prior = reads[-1]; p0 = prior[0]['period']
                if kind == 'rel':
                    win = self._prior_window(p0, ref); moved = self._shift_back(*win, wtext) if win else None
                else: moved = self._same_shift(p0, w['m'], ref)
                if moved is None or isinstance(moved, str): return moved or 'unresolved_inherited_period'
                if moved[1] > ref: return 'future_period_unavailable'
                p = {'kind': 'between', 'start_at': moved[0].isoformat(), 'end_at': moved[1].isoformat()}
                reads.append([{**q, 'period': dict(p)} for q in prior]); periods.append(moved); continue
            res = dec(self._cmp_text(before, wtext, after, more))
            if res['status'] != 'planned' or not res['queries'] or any(q.get('kind') != 'health' for q in res['queries']):
                return res['reason_codes'][0] if res['status'] != 'planned' and res.get('reason_codes') else 'multiple_periods_require_clause_binding'   # all or nothing (10g.6)
            ps = {json.dumps(q['period'], sort_keys=True) for q in res['queries']}
            if len(ps) != 1: return 'multiple_periods_require_clause_binding'
            p = json.loads(ps.pop()); span = self.period_span(p, ref)
            ops = {q['operation'] for q in res['queries'] if q.get('metrics')}
            if kind == 'now' and (span is not None or ops != {'latest'}): return 'period_disagreement'    # the value now is one latest read (item 7)
            if kind != 'now':
                if span is None or ops - {'trend'}: return 'period_disagreement'     # a stated window reads its series (§3 item 9)
                # Verify a stated rolling duration independently of the child
                # prediction. Structurally valid reads of a different duration
                # must not satisfy this clause (e.g. 30 days for past 7 days).
                rolling = parse_rolling(wtext)
                if rolling:
                    expected = resolve_period('rolling', rolling[0], rolling[1], None, ref, wtext)
                    if expected is None or span != self.period_span(expected, ref):
                        return 'period_disagreement'
                stated = None
                if not re.search(r"\b(?:since|from|onwards?|till|til|until|to date|so far|ago|back|past|last\s+\d|last\s+(?:" + NUM_WORDS + r")\s)", wtext, re.I) and not rolling:
                    try: stated = parse_time_span(wtext, ref)
                    except (AmbiguousDate, InvalidDate): stated = None
                if stated:       # the clause read another window than it states: a series reads it to date; calendar records read whole
                    ok = {(stated[0], min(stated[1], ref))}     # periods by wording (10g.19: "my checkups this year" = 2026-01-01..12-31)
                    if any(q.get('records') for q in res['queries']):
                        for mode in ('year', 'next'):
                            try: ok.add(tuple(parse_time_span(wtext, ref, mode) or ()))
                            except (AmbiguousDate, InvalidDate): pass
                    if span not in ok: return 'period_disagreement'
            reads.append(res['queries']); periods.append(span); conf = min(conf, res.get('confidence', 1.0)); dconf = min(dconf, res.get('decision_confidence', 1.0))
        return reads, periods, conf, dconf

    def followup_compare(self, req, cur, hist, ref, H):
        """A follow-up comparing the prior read with the windows it names (spec v1.8 10g.38 item b): "What was my HRV in 2025?" -> "Compare that
        to 2024" / "how does that compare with the past 90 days?" / "vs last week?" = the prior window + the new one, metric and source inherited
        (10g.9; re-reading the prior window makes the answer self-contained); a follow-up naming two windows ("which was higher, June or July?")
        reads those two. Only a follow-up that says nothing but the comparison (FUP_CMP) and its windows; the guards and the identity and
        overlap rules of window_compare apply (a prior "this year" vs "the past 90 days" hands off). None: no such follow-up."""
        m = FUP_CMP.match(cur)
        rest = re.sub(r"(?:[.!?,]\s*|\s+)(?:thanks?|thank\s+you|thx|ty|cheers|please|pls)\b.*$", '', cur[m.end():], flags=re.I).strip(' ,.?!') if m else ''
        parts = [x.strip() for x in re.split(r"\s*(?:,|\bor\b|\band\b|\bvs\.?|\bversus\b)\s*", rest, flags=re.I) if x.strip()]
        if not parts or not all(time_only(t) for t in parts) or any(rx.search(cur.lower()) for rx, _, _ in self.syn): return None
        code = self.cmp_guard(cur)
        if code: return H(code)
        prior = self.select({**req, '_prev': True, 'state': {**req['state'], 'current_request': hist[-1], 'recent_user_requests': hist[:-1]}})
        if prior['status'] != 'planned' or any(q.get('kind') != 'health' for q in prior['queries']): return None
        ps = {json.dumps(q['period'], sort_keys=True) for q in prior['queries']}
        if len(ps) != 1: return None
        p0 = json.loads(ps.pop()); reads, periods = [], []
        for t in parts:
            bare = re.sub(r"^(?:in|on|for|during|over|at|since)\s+", '', t, flags=re.I)
            if CMP_REL.fullmatch(bare) or CMP_SAME.fullmatch(bare):
                base_p = reads[-1][0]['period'] if reads else p0
                win = self._prior_window(base_p, ref)
                moved = (self._shift_back(*win, bare) if win else None) if CMP_REL.fullmatch(bare) else self._same_shift(base_p, CMP_SAME.fullmatch(bare), ref)
                if moved is None or isinstance(moved, str): return H(moved or 'unresolved_inherited_period')
                if moved[1] > ref: return H('future_period_unavailable')
                qs = [{**q, 'period': {'kind': 'between', 'start_at': moved[0].isoformat(), 'end_at': moved[1].isoformat()}} for q in (reads[-1] if reads else prior['queries'])]
                reads.append(qs); periods.append(moved); continue
            if CMP_NOW.fullmatch(bare):
                qs = [{**q, 'operation': 'latest', 'period': {'kind': 'all_history'}} for q in prior['queries']]
                reads.append(qs); periods.append(None); continue
            res = self.select({**req, '_clause': True, 'state': {**req['state'], 'current_request': t, 'recent_user_requests': hist}})   # a window-change follow-up (10g.9)
            if res['status'] != 'planned' or any(q.get('kind') != 'health' for q in res['queries']):
                return H(res['reason_codes'][0] if res.get('reason_codes') else 'unresolved_inherited_period')
            sp = {json.dumps(q['period'], sort_keys=True) for q in res['queries']}
            if len(sp) != 1: return H('unresolved_inherited_period')
            reads.append(res['queries']); periods.append(self.period_span(json.loads(sp.pop()), ref))
        if len(parts) == 1: reads, periods = [prior['queries']] + reads, [self.period_span(p0, ref)] + periods
        items = lambda qs: json.dumps([sorted({m for q in qs for m in q.get('metrics') or []}), sorted({r for q in qs for r in q.get('records') or []}), sorted({q.get('source') or '' for q in qs})])
        if len({items(r) for r in reads}) != 1: return H('ambiguous_followup_subject')
        for i in range(len(periods)):
            for j in range(i):
                a, b = periods[i], periods[j]
                if a is None and b is None: return H('unbound_request_constraint')
                if a is None or b is None: continue
                if a == b or (a[0] <= b[1] and b[0] <= a[1] and a[0] != a[1] and b[0] != b[1]): return H('unbound_request_constraint')
        qs = [q for r in reads for q in r]
        if len(qs) > 8: return H('plan_breadth_exceeded')
        return {'status': 'planned', 'reason_codes': [], 'queries': qs, 'confidence': prior.get('confidence', 1.0), 'decision_confidence': prior.get('decision_confidence', 1.0)}

    @staticmethod
    def _cmp_clean(t, more):
        """t without the comparison scaffolding (CMP_FRAME; with more, also the comparative words and a yes/no auxiliary before them, CMP_MORE,
        CMP_YESNO), a style or language instruction (CMP_STYLE_CLAUSE: the multi-read plan keeps it, 10g.28 class 3 [refined 10g.38]) and
        pleasantries; "did I sleep more" reads how much I slept (CMP_SLEEP_MORE)."""
        t = CMP_STYLE_CLAUSE.sub(' ', CMP_SLEEP_MORE.sub('how much did I sleep', t))
        for rx in CMP_FRAME: t = rx.sub(' ', t)
        if more: t = CMP_YESNO.sub('', CMP_MORE.sub(' ', t))
        return re.sub(r"\b(?:please|pls|plz|thanks|thank\s+you|thx|cheers)\b", ' ', t, flags=re.I)

    @classmethod
    def _cmp_text(cls, before, wtext, after, more):
        """One window's clause: the cleaned words before the list, the window, the cleaned words after it."""
        t = cls._cmp_clean(before, more) + ' ' + wtext + ' ' + cls._cmp_clean(after, more)
        t = re.sub(r"\s+([,.;:?!])", r"\1", re.sub(r"\s+", ' ', t)).strip()
        t = re.sub(r"^[\s,.;:?!-]+", '', t); t = re.sub(r"([,;:])\s*([?.!]|$)", r"\2", t)
        return t

    def narrowing_text(self, cur, rs, times, runs):
        """A panel or area name with one of its members in one sentence of the read, from the text and the alias tables (10g.34 item 4,
        10g.36 item 5), whatever the tagger's spans are (r8 tags "specifically muscle mass" as one mention; a window may sit between the
        two). Returns (keys of the group to drop, handoff code). The member after a narrowing connective reads alone; after a list
        connective (or a new clause) the group stays as named; after any other joint, or with two members, or a non-member ("lipids:
        just glucose"), or a member before its group ("the LDL part of my lipid panel"), the scope is unsettled: hand off."""
        drop, names, merged = set(), [(rx, key) for rx, kind, key in self.syn if kind == 'metric' and key in PANELS], []
        for a, z in rs:
            if merged and re.search(r'(?:\.\.|\u2026)\s*$', cur[merged[-1][0]:merged[-1][1]]) and not cur[merged[-1][1]:a].strip(): merged[-1] = (merged[-1][0], z)
            else: merged.append((a, z))
        for s0, e0 in merged:
            s, low = cur[s0:e0], cur[s0:e0].lower()
            area = [m for m in re.finditer(r'(?<![\w-])sleep(?![\w-])', low) if any(x <= s0 + m.start() and s0 + m.end() <= y for x, y in runs) or re.search(r'(?:^\W*|\b(?:my|the|your|our)\s+)$', low[:m.start()])]     # the bare area word: tagged, or a noun
            if not area and not any(rx.search(low) for rx, _ in names): continue
            spans = {}
            for rx, kind, key in self.syn:
                for m in rx.finditer(low): spans.setdefault((m.start(), m.end()), set()).add(key)
            for term in APP_ANALYTES:
                for m in re.finditer(r'(?<![\w-])' + re.escape(term) + r'(?![\w-])', low): spans.setdefault((m.start(), m.end()), set()).add('app:' + term)
            kept = []
            for (a, z), ks in sorted(spans.items(), key=lambda x: (-(x[0][1] - x[0][0]), x[0][0])):     # the longest name wins
                if not any(a < z2 and z > a2 for a2, z2, _ in kept): kept.append((a, z, ks))
            for m in area:
                if not any(m.start() < z2 and m.end() > a2 for a2, z2, _ in kept): kept.append((m.start(), m.end(), {'sleep'}))
            kept.sort()
            tim = [(a - s0, z - s0) for a, z in times if s0 <= a < e0]
            gap = lambda x, y: ''.join(c for i, c in zip(range(x, y), s[x:y]) if not any(a <= i < z for a, z in tim))
            members = lambda key: set(AREAS['sleep']) if key == 'sleep' else set(PANELS[key])
            ids = lambda ks, mem: {k for k in ks if k in mem} | {i for k in ks if k.startswith('app:') for i in APP_ANALYTES[k[4:]]['ids'] if i in mem}
            for a, z, ks in kept:
                for key in [k for k in ks if k in PANELS or k == 'sleep']:
                    mem = members(key); grp = (set(AREAS['sleep']) | {'sleep'}) if key == 'sleep' else {key, *PANELS[key]}
                    nxt = next((k for k in kept if k[0] >= z), None); prv = next((k for k in reversed(kept) if k[1] <= a), None)
                    if nxt:
                        g = gap(z, nxt[0]); hit = ids(nxt[2], mem)
                        if NARROW_GAP.fullmatch(g):
                            chain = [nxt]; more = next((k for k in kept if k[0] >= nxt[1]), None)
                            while more and ids(more[2], mem) and gap(chain[-1][1], more[0]).strip() and CHAIN_GAP.fullmatch(gap(chain[-1][1], more[0])):
                                chain.append(more); more = next((k for k in kept if k[0] >= chain[-1][1]), None)
                            if not hit or len(chain) > 1: return set(), 'unresolved_health_scope'     # a non-member, or two members
                            drop |= grp - ids(nxt[2], mem)
                        elif hit and not (LIST_GAP.fullmatch(g) or (re.search(r'[A-Za-z]', g) and (CLAUSE_WORDS.search(g) or CLAUSE_BOUNDARY.search(g))) or SRC_BEFORE.search(low[:a]) or (not g.strip() and key == 'sleep')):
                            return set(), 'unresolved_health_scope'
                    if key != 'sleep' and prv and ids(prv[2], mem):
                        g = gap(prv[1], a)
                        if not (LIST_GAP.fullmatch(g) or (re.search(r'[A-Za-z]', g) and (CLAUSE_WORDS.search(g) or CLAUSE_BOUNDARY.search(g)))): return set(), 'unresolved_health_scope'
        return drop, None

    INTERNAL = ('_prev', '_clause', '_split', '_standalone', '_ctx', '_cmp')

    def select(self, req):
        try: return self._select_bounded(req)
        except InputTokenBudgetExceeded:
            return {'status': 'handoff', 'reason_codes': ['input_token_budget_exceeded'], 'queries': []}

    def _select_bounded(self, req):
        """The plan that leaves the decoder uses only period shapes the repo contract takes (contract_period). Re-entrant calls
        (the prior turn, clauses, the standalone re-read) keep the decoder's own shapes, which follow-up resolution reasons with
        (a rolling N weeks, the calendar year). A top-level call carries its own memo of model outputs by exact input text ('_memo'),
        passed on to its re-entrant calls with the request and gone when it returns: nothing is kept across requests."""
        # Count the complete inputs before any encoder call, including history that
        # normalization or topic reset would otherwise remove. Never infer on a prefix.
        if hasattr(self, 'tok') and 'current_request' in req['state']:
            st = req['state']; history = st.get('recent_user_requests') or []
            texts = [request_text(st['current_request'], history),
                     *[request_text(h, []) for h in history[-self.HISTORY_TURNS:]]]
            if any(len(self.tok(t, truncation=False)['input_ids']) >= 190 for t in texts):
                return {'status': 'handoff', 'reason_codes': ['input_token_budget_exceeded'], 'queries': []}
        if not any(req.get(k) for k in self.INTERNAL):
            req = {**req, '_memo': self._prefetch(req)}
            if self.history_crisis(req): return {'status': 'handoff', 'reason_codes': ['acute_or_crisis_requires_model'], 'queries': []}
        res = self._select(req)
        if res['status'] != 'planned' or any(req.get(k) for k in self.INTERNAL): return res
        ref = date.fromisoformat(req['state']['reference_date'])
        return {**res, 'queries': [{**q, 'period': contract_period(q['period'], ref, q.get('records') or ())} if q.get('kind') == 'health' else q for q in res['queries']]}

    def _prefetch(self, req):
        """A follow-up reads a second input besides its own: the prior turn (its window, operation or plan, re-read as a '_prev'
        request) or, when it is complete, itself without history (10g.26). Both go through the encoder in one packed pass
        (packed_hidden), which costs a fraction of a second pass. Returns the memo of those outputs."""
        if not getattr(self, '_pack', False) or not req['state'].get('recent_user_requests'): return {}   # no packing, or one input
        alone = self._history_alone(req)                  # each earlier turn on its own (history_crisis), in the same pass
        req, cur, hist = self._normalized(req)
        if not hist or (SAME_UNIT_IN.search(cur) and not YEAR_REANCHOR.search(cur)): texts = alone
        else:
            other = {**req['state'], 'recent_user_requests': []} if self_contained(cur) else {**req['state'], 'current_request': hist[-1], 'recent_user_requests': hist[:-1]}
            texts = list(dict.fromkeys([request_text(cur, hist), request_text(*self._normalized({**req, 'state': other})[1:]), *alone]))
        return dict(zip(texts, self._encode(texts)))

    HISTORY_TURNS = 4       # the contract's recent_user_requests (spec §2.4, 10f: the acute rule scans the current turn and the recent turns)

    def _history_alone(self, req):
        """The input text of each of the last HISTORY_TURNS earlier turns read as a request of its own (normalized as a current turn)."""
        return [request_text(*self._normalized({'state': {**req['state'], 'current_request': h, 'recent_user_requests': []}})[1:])
                for h in (req['state'].get('recent_user_requests') or [])[-self.HISTORY_TURNS:]]

    @torch.inference_mode()
    def history_crisis(self, req):
        """Crisis or medication content in an earlier turn hands off (spec §2.4, 10f). Every earlier turn is scored on its own, before a
        topic reset or the 3-turn input window drops it: in history the crisis head reads a turn as context (about 4e-5 on a crisis turn
        that scores 1.0 as the current one). Keyword rule OR the crisis head on the turn alone; uses the top-level memo (_prefetch)."""
        hist = (req['state'].get('recent_user_requests') or [])[-self.HISTORY_TURNS:]
        if not hist: return False
        if self.acute and any(self.acute(h) for h in hist): return True
        if self.crisis_thr is None: return False
        memo = req['_memo']
        for t in self._history_alone(req):
            if t not in memo:
                b = self._batch(t); b.pop('offset_mapping'); memo[t] = self.model(b, self.I)
            if float(memo[t]['crisis'][0].softmax(-1)[1]) >= self.crisis_thr: return True
        return False

    @staticmethod
    def _normalized(req):
        """(request, current turn, history) as the decoder reads them."""
        st = req['state']; cur = st['current_request']; hist = st.get('recent_user_requests') or []
        if any(calm(t) != t for t in [cur, *hist]):      # shouted turns are read in lowercase
            cur, hist = calm(cur), [calm(h) for h in hist]; req = {**req, 'state': {**st, 'current_request': cur, 'recent_user_requests': hist}}; st = req['state']
        if ADDRESS.search(cur) and ADDRESS.sub(' ', cur).strip(' ,.!'):     # "Hi Vita, show calendar for July" -> "show calendar for July"
            cur = ADDRESS.sub(' ', cur).strip(); req = {**req, 'state': {**st, 'current_request': cur}}; st = req['state']
        if any(plain(t) != t for t in [cur, *hist]):      # "Hey, respiratory rate this month thanks!" -> "respiratory rate this month"
            cur, hist = plain(cur), [plain(h) for h in hist]; req = {**req, 'state': {**st, 'current_request': cur, 'recent_user_requests': hist}}; st = req['state']
        if hist and TOPIC_RESET.search(cur):              # "Different topic — what's my ferritin?" does not inherit from earlier turns
            cur = TOPIC_RESET.sub(' ', cur).strip(' \u2014\u2013-:;,.') or cur        # the marker itself is not part of the request
            req = {**req, 'state': {**st, 'current_request': cur, 'recent_user_requests': []}}; st = req['state']; hist = []
        return req, cur, hist

    @torch.inference_mode()
    def _select(self, req):
        one_night = req.get('_one_night')          # decode a mixed_windows request as a plan (the handoff branch); never passed on
        if one_night: req = {k: v for k, v in req.items() if k != '_one_night'}
        req, cur, hist = self._normalized(req); st = req['state']
        ref = date.fromisoformat(st['reference_date'])
        H = lambda code: {'status': 'handoff', 'reason_codes': [code], 'queries': []}
        if hist and SAME_UNIT_IN.search(cur) and not YEAR_REANCHOR.search(cur): return H('unresolved_inherited_period')
        end = RANGE_END.match(cur) if hist and not any(req.get(k) for k in self.INTERNAL) else None
        if end and time_only(end.group('t')) and not any(rx.search(cur.lower()) for rx, _, _ in self.syn):
            # "to Wednesday" after "Show my steps from Monday" completes that request: re-read it, as the prior turn, with the new end (one span
            # 09-21..09-23 for day ends, 10g.38 item 3 / 10g.36.6), never the prior read + a new window (item b needs a comparison cue, FUP_CMP).
            done = hist[-1].rstrip(' ?.!') + ' ' + cur.strip()
            return self.select({**req, '_prev': True, 'state': {**st, 'current_request': done, 'recent_user_requests': hist[:-1]}})
        if hist and not any(req.get(k) for k in self.INTERNAL) and FUP_CMP.match(cur):     # the prior read vs the window a follow-up names (10g.38 item b)
            fc = self.followup_compare(req, cur, hist, ref, H)
            if fc is not None: return fc
        if hist and (PREV_UNIT.search(normalize_time_text(cur)) or PREV_N.search(normalize_time_text(cur)) or YEAR_REANCHOR.search(normalize_time_text(cur))):   # also on the re-read prior turn: a chain shifts from the most recent read window (10g.25)
            rel = self.relative_followup(req, normalize_time_text(cur), hist, ref, H)
            if rel is not None: return rel                        # None: no prior window -> the words keep their standalone meaning
        text = request_text(cur, hist); self._text = text
        b = self._batch(text); offs = b.pop('offset_mapping')[0]
        # BPE offsets include the leading space; start each token at its first character (same alignment as training).
        offs = torch.tensor([[a + len(text[a:z]) - len(text[a:z].lstrip()), z] for a, z in offs.tolist()])
        memo = req.get('_memo', {})           # this top-level request's model outputs by exact input text (select)
        if text not in memo: memo[text] = self.model(b, self.I)
        out = memo[text]
        # Crisis = keyword rule OR (merged crisis head, if trained, else the separate classifier). Never the model alone.
        crisis_head = self.crisis_thr is not None and float(out['crisis'][0].softmax(-1)[1]) >= self.crisis_thr
        if crisis_head or (self.acute and any(self.acute(t) for t in [cur, *hist])): return H('acute_or_crisis_requires_model')
        probs = out['status'][0].softmax(-1)
        status = STATUS[probs.argmax()]
        # Plan only when the model is confident; the threshold is calibrated on validation for a low false-plan rate.
        if status == 'plan' and float(probs[0]) < self.plan_threshold: return H('uncertain_model_decision')
        if req.get('_force_plan') and status == 'handoff': status = 'plan'   # checker training only: the plan a missed handoff would give
        if one_night and status == 'handoff': status = 'plan'
        if status == 'handoff' and HANDOFF[out['handoff'][0].argmax()] == 'mixed_windows' and plain_month_range(cur):
            one_night, status = True, 'plan'
        roles = out['roles'][0].argmax(-1).tolist(); base = len('request: '); cur_end = base + len(cur)
        if hist and not (req.get('_clause') or req.get('_prev') or req.get('_standalone')) and self_contained(cur) and mention_runs(roles, offs, base, cur_end):
            return self.select({**req, '_standalone': True, 'state': {**st, 'recent_user_requests': []}})   # 10g.26: a complete request stands alone
        if status == 'handoff':
            cls = HANDOFF[out['handoff'][0].argmax()] or 'handoff'
            res = H(MODEL_CODES['model_' + cls])
            # Older heads classify every comparison as unsupported. A closed,
            # explicit window comparison can instead compile into independently
            # decoded reads. window_compare checks the complete comparative,
            # all clauses, source equality, overlap and residual constraints;
            # parser agreement and the normal confidence/checker gates still apply.
            if cls == 'comparison' and not req.get('_clause'):
                comparison = self.window_compare(req, cur, offs, roles, base, cur_end, H)
                if comparison is not None:
                    return comparison
            # TEST ONLY (never set in production): read the model's "comparison" decision as "mixed_windows", to measure what this decoder
            # makes of same-metric window comparisons once retrained models say mixed_windows for them (spec v1.8 10g.38; jevtrain.py).
            route = 'mixed_windows' if cls == 'comparison' and TEST_COMPARISON_AS_MIXED else cls
            # Different windows per clause are planned clause by clause since spec v1.8 10g.6 (older labels said handoff).
            if route == 'mixed_windows' and not req.get('_clause'):     # comparisons stay handoffs ("did I sleep more than...")
                # A group, its window and its narrowed member ("sleep yesterday, namely total sleep", 10g.34 item 4) is one read: cutting it
                # into clauses at the window would read the group and the member as two reads.
                mr = [(int(offs[r[0]][0]) - base, int(offs[r[-1]][1]) - base) for r in mention_runs(roles, offs, base, cur_end)]
                tm = [(int(a) - base, int(z) - base) for (a, z), r in zip(offs.tolist(), roles) if z > a and a >= base and z <= cur_end and ROLES[r] == 'time']
                if self.narrowing_text(cur, self._reading_sentences(cur, mr), tm, mr)[0]:
                    one = self._select({**req, '_one_night': True})
                    if one['status'] == 'planned' and len({json.dumps(q['period'], sort_keys=True) for q in one['queries']}) == 1: return one
                split = self.latest_window_split(req, cur, offs, roles, base, cur_end, H) if not req.get('_split') else None
                # Its handoff is final too ("RHR this month plus my VO2 max": unclear attachment, 10g.6); per-clause decoding would
                # read the second clause as a follow-up of the first and give it the window.
                if split is not None: return split
                # The same metric set in several stated windows is one read per window (spec v1.8 10g.38); None: not such a comparison.
                wc = self.window_compare(req, cur, offs, roles, base, cur_end, H)
                if wc is not None: return wc
                res = self.per_clause(req, cur, text, offs, roles, base, cur_end, H, fallback='model_' + cls)
                if res['status'] == 'planned': return res
            if route == 'mixed_windows':
                # The trainer turns every plan whose reads differ in period kind into mixed_windows (jevtrain.py), and so the two reads
                # of one night (spec v1.8 10g.4: the night basis + a one-day observed_at window of the same day; "How did I sleep on
                # Tuesday?"). Decode such a request as a plan and keep it only when it is exactly that split.
                night = self._select({**req, '_one_night': True})
                if night['status'] == 'planned' and one_night_split(night['queries']): return night
            return res
        def span_runs(role, pos=False):   # contiguous tagged tokens are rebuilt from the original characters, not joined piecewise
            runs, last = [], None
            for j, ((a, z), r) in enumerate(zip(offs.tolist(), roles)):
                if z > a and ROLES[r] == role and a >= base and z <= cur_end:
                    if last == j - 1: runs[-1][1] = z
                    else: runs.append([a, z])
                    last = j
            if role == 'time':      # a time word split across tokens is read whole ("t|uesday last week": only "uesday..." tagged)
                whole = []
                for a, z in runs:
                    while a > base and text[a - 1].isalnum(): a -= 1
                    while z < cur_end and text[z].isalnum(): z += 1
                    # "<D> night" / "the night of D" started on D (spec v1.8 10g.4): the date parser needs the night word even when
                    # the tagger left it in the metric span ("How many hours I slept in the night | of Wednesday").
                    night = re.search(r'\bnight\s+$' if text[a:z].lower().startswith('of ') else r'\bnight\s+of\s+$', text[base:a], re.I)
                    if night: a = base + night.start()
                    night = re.match(r'\s+night\b', text[z:cur_end], re.I)
                    if night: z += night.end()
                    # "<weekday> last/this week" is one date (10g.3) even when the tagger stopped before the week words, as it does on a
                    # clipped or misspelt one ("Thursday | last wek."): the date parser needs them.
                    wd = WEEKDAY_END.search(text[a:z])
                    week = wd and re.match(r'\s+(?:week|wek|wk)\b' if wd.group(1) else r'\s+(?:last|this|lst)\s+(?:week|wek|wk)\b', text[z:cur_end], re.I)
                    if week: z += week.end()
                    if whole and a <= whole[-1][1]: whole[-1][1] = max(z, whole[-1][1])
                    else: whole.append([a, z])
                runs = whole
            return [(a - base, z - base) for a, z in runs] if pos else [text[a:z].strip() for a, z in runs]   # pos: offsets in cur
        span = lambda role: ' '.join(span_runs(role)).strip()
        if status == 'plan' and not hist and DEFINITION.search(cur) and not PERSONAL.search(cur) and not span('time') and not req.get('_clause'):
            return H('model_definition')                          # "What is ApoB?" asks what it is, not for the user's value
        if status == 'research':
            # Research about the user's own data ("papers relevant to my Lp(a) result") is interpretation, not a read (spec v1.8 10g.8).
            if re.search(r'\bmy\b', cur.lower()): return H('targeted_research_binding_unavailable')
            topic, code = research_topic(text, [(a, z, ROLES[r]) for (a, z), r in zip(offs.tolist(), roles) if z > a and a >= base and z <= cur_end], base, cur_end)
            if code: return H(code)
            if re.search(r"\b(?:not|no|without|except|excluding|exclude)\b", cur.lower()): return H('unbound_research_topic')   # dropping a negation inverts the topic ("people without diabetes"), as the repo's public_topic (10g.32 item 8)
            if not req.get('literature_available', True): return H('unbound_research_topic')
            return {'status': 'planned', 'reason_codes': [], 'queries': [{'kind': 'research', 'topic': topic}],
                    'confidence': float(probs[STATUS.index('research')]), 'decision_confidence': float(probs[STATUS.index('research')])}
        # Wording guards for reads. The tagged mentions mark the sentences that carry the read; context sentences are ignored (10g.28).
        runs = [(int(offs[r[0]][0]) - base, int(offs[r[-1]][1]) - base) for r in mention_runs(roles, offs, base, cur_end)]
        if any(EXTREME.search(cur[a:z]) for a, z in self._reading_sentences(cur, runs)): return H('model_extreme')   # an all-time extreme in the read
        said_sents, time_at = self._reading_sentences(cur, runs), span_runs('time', pos=True)
        code = temporal_guard(cur, said_sents, time_at, ref, prev_ok=bool(hist and PREV_N.search(cur)))
        if code: return H(code)
        # A dated event in a context sentence is context (spec v1.8 10g.28, clarified): "My half marathon was on the 14th. What's my resting
        # heart rate?" is not a read of the 14th. The read is decoded on its own sentences when it has its own window, or asks for a present
        # value and gets a latest read. A read referring back takes the event's date ("that day", "the day before", "since then": _bind_event)
        # or hands off when that is unclear ("then", "that day" after a month); one that may mean the event's date ("How did I sleep?") hands
        # off. A side sentence that is the read's window, a request or a correction (CTX_REQUEST) still binds, as before.
        if runs and not any(req.get(k) for k in self.INTERNAL):
            rs = self._reading_sentences(cur, runs); sents = [(x.start(), x.end()) for x in self.SENTENCE.finditer(cur)]; times = span_runs('time', pos=True)
            dated = [(x, y) for x, y in sents if (x, y) not in rs and any(x <= a < y for a, z in times)]
            event = lambda x, y: side_event(cur[x:y], [(a - x, z - x) for a, z in times if a < y and z > x])
            if len(sents) > 1 and len(rs) < len(sents) and dated and all(event(x, y) for x, y in dated):
                read_text = ' '.join(cur[x:y].strip() for x, y in rs)
                own = any(x <= a < y for a, z in times for x, y in rs)
                back = CTX_BACKREF.search(read_text)
                if back:        # the event's date binds the read (10g.28): the reference is replaced by the dates it names, then read as usual
                    bound = self._bind_event(read_text, back, cur, dated, times, ref)
                    if bound is None: return H('unresolved_temporal_phrase')
                    return self.select({**req, '_ctx': True, 'state': {**req['state'], 'current_request': bound}})
                if not (own or (PRESENT_READ.search(read_text) and not re.search(r"\bbeen\b", read_text, re.I))):
                    return H('unresolved_temporal_phrase')
                res = self.select({**req, '_ctx': True, 'state': {**req['state'], 'current_request': read_text}})
                if res['status'] == 'planned' and not own and any(q.get('operation') != 'latest' for q in res['queries']): return H('unresolved_temporal_phrase')
                return res
        intake_any, has_since = self._intake(cur), bool(SINCE_PHRASE.search(cur))
        reading = self._reading_sentences(cur, runs) if intake_any or has_since else []
        if intake_any and any(self._intake(cur[a:z]) for a, z in reading): return H('requested_metric_unavailable')   # 10g.30
        if hist and not runs and any(self._intake(h) and any(self._intake(h[a:z]) for a, z in self._reading_sentences(h)) for h in hist[-3:]):
            return H('requested_metric_unavailable')           # the subject comes from an intake question ("and today?")
        code, anchor = self._since(cur, ref, reading, runs) if has_since else (None, None)
        if code: return H(code)
        if hist and not anchor and not span('time') and not self._has_window_words(cur, ref):   # the window comes from an earlier turn
            for h in reversed(hist[-3:]):
                code, a = self._since(h, ref, self._reading_sentences(h)) if SINCE_PHRASE.search(h) else (None, None)
                if code: return H(code)
                if a and a[0] == 'device': return H('unresolved_inherited_period')   # a new metric does not keep the source (10g.9)
                if a or self._has_window_words(h, ref): break
        avail = set(req.get('available_metrics') or []); recs = set(req.get('available_record_types') or [])
        # Inventory ids and catalog ids name the same measurement (selector index inventory_to_catalog, e.g. hscrp -> hs_crp).
        resolve = {**{INV2CAT.get(m, m): m for m in avail}, **{m: m for m in avail}}
        panel_ok = lambda key: any(m in resolve or SPELLING.get(m, set()) & avail for m in PANELS[key])
        # A bare split id ("glucose") links when any of its ids is in the inventory, and reads those (spec v1.8 10g.27, 10g.1).
        ok = lambda k, key: (k == 'metric' and (key in resolve or (key in PANELS and panel_ok(key)) or bool(UNIT_SPLIT.get(key, set()) & avail))) or (k == 'record' and (key in recs or key in BUNDLE_DEFS)) or (k == 'profile_field' and 'profile' in recs)
        # Agreement check: plan only if two independent views of the model pick the same items
        # (whole-input late interaction vs. per-mention spans). Disagreement means "not sure" -> hand off.
        linked = set(above(out['link'][0], self.LINK_THR))
        full = {i for i, (k, key, _) in enumerate(self.items) if ok(k, key) and i in linked}
        chosen = self.link(out, roles, offs, base, cur_end, ok, INHERIT[int(out['inherit'][0].argmax())])
        profile_named = bool(re.search(r'\bprofile\b', cur.lower()))      # the user said "profile" (not a whole-input link guess)
        # ... in the clause of the weight/height ("profile weight", "weight in my profile"; not "my profile and my latest weight", 10g.2)
        profile_twin = any(re.search(r'\bprofile\b', c, re.I) and re.search(r'\b(?:weigh\w*|height|tall)\b', c, re.I) for c in CLAUSE_BOUNDARY.split(cur))
        pos = {(k, key): i for i, (k, key, _) in enumerate(self.items)}
        def prefer_metric(S):   # a metric and its profile twin (spec v1.8 10g.2): the metric, unless the user said "profile"
            keys = {self.items[i][1]: i for i in S}
            for m, f in TWINS.items():
                if not profile_twin and f in keys and m in resolve and ('metric', m) in pos: S = (S - {keys[f]}) | {pos[('metric', m)]}
                if profile_twin and m in keys and 'profile' in recs and ('profile_field', f) in pos: S = (S - {keys[m]}) | {pos[('profile_field', f)]}
            return S
        full, chosen = prefer_metric(full), prefer_metric(chosen)
        # BMI too: the catalog's bmi metric is not in the inventory ("BMI" -> profile bmi, there is no BMI metric; 10g.2)
        absent = {pos[('profile_field', f)] for m, f in [*TWINS.items(), ('bmi', 'bmi')] if m not in resolve and 'profile' in recs and ('metric', m) in pos
                  and ('profile_field', f) in pos and float(out['link'][0][pos[('metric', m)]]) > self.LINK_THR}
        full |= absent      # a twin metric the inventory lacks reads its profile field, as the span view does (10g.2 "otherwise the profile field")
        # A panel or area name narrowed to one member ("body composition - fat percentage", "lipids: just LDL", "sleep, specifically REM")
        # reads that member only; a member named as a separate item ("body composition and fat percentage"), or two members, does not
        # narrow (10g.34 item 4). Both views drop the rest of the group, and its links are no competitors (rest, below).
        narrow, R = set(), list(zip(self._runs, self._spans))
        for j in range(len(R) - 1):
            (t1, g1), (_, z1) = R[j]; (_, g2), (a2, _) = R[j + 1]
            grp = {x for k in g1 if k in PANELS for x in [k, *PANELS[k]]} | (set(AREAS['sleep']) if g1 <= set(AREAS['sleep']) and re.fullmatch(r'(?:my\s+)?sleep\W*', t1) else set())
            gap = ''.join(c for i, c in zip(range(z1, a2), cur[z1:a2]) if not any(a <= i < z for a, z in span_runs('time', pos=True)))   # "sleep last week, specifically ..."
            if g2 and g2 <= grp and NARROW_GAP.fullmatch(gap) and not any(g & grp for i, ((_, g), _) in enumerate(R) if i not in (j, j + 1)): narrow |= grp - g2
        tn, tcode = self.narrowing_text(cur, self._reading_sentences(cur, runs), span_runs('time', pos=True), runs)
        if tcode: return H(tcode)      # a group and its member joined in a way 10g.34 item 4 / 10g.36 item 5 do not settle
        narrow |= tn
        full, chosen = ({i for i in S if self.items[i][1] not in narrow} for S in (full, chosen))
        fa = FULL_AREA.search(cur)      # the whole read is the full sleep area: the wording names the reads, not its mention's links (10g.34 item 5)
        full_area = bool(fa) and all(fa.start() <= a and z <= fa.end() or g and g <= set(AREAS['sleep']) for (_, g), (a, z) in R)   # other mentions only of the area
        if full_area and narrow: return H('unresolved_health_scope')      # "full breakdown of my sleep, specifically REM": whole area and one member
        if full_area: full = chosen = set()
        # The two views are compared modulo known identities: ids of one analyte (spec v1.8 10g.1) name the same measurement, and a
        # profile field implies the profile record.
        def norm(S):
            keys = {(self.items[i][0], ANALYTE.get(resolve.get(self.items[i][1], self.items[i][1]), resolve.get(self.items[i][1], self.items[i][1]))
                     if self.items[i][0] == 'metric' else self.items[i][1]) for i in S}
            return keys - {('record', 'profile')}     # the profile record reads nothing without a field; a field implies it
        self._views = (norm(full), norm(chosen))                 # diagnostics only
        if norm(full) != norm(chosen) and not req.get('_force_plan'): return H('linking_disagreement')   # forced (checker data): span view only
        # A mention that is exactly a catalog-declared ambiguous name cannot pick one item.
        for run in [p for run in mention_runs(roles, offs, base, cur_end) for p in self.split_run(run, offs)]:
            term = text[offs[run[0]][0]:offs[run[-1]][1]].strip().lower()
            # Not ambiguous when the request names a longer catalog item containing the term ("RBC magnesium", "skin temperature").
            longer = any(term in n and n != term and re.search(r'(?<![\w-])' + re.escape(n) + r'(?![\w-])', cur.lower()) for n in self.syn_raw)
            typo = term not in AMBIGUOUS and term not in self.syn_raw and len(term) >= 5 and any(abs(len(a) - len(term)) <= 1 and _edits(term, a) <= 1 for a in AMBIGUOUS)
            if (term in AMBIGUOUS or typo) and not longer: return H('ambiguous_metric_alias')   # one typo away is still ambiguous ("magnesiun")
        app_terms = [t for t in APP_ANALYTES if re.search(r'(?<![\w-])' + re.escape(t) + r'(?![\w-])', cur.lower())]
        if not chosen and not app_terms and not full_area: return H('unresolved_metric_or_record')   # an app-known analyte binds below
        bundles = [self.items[i][1] for i in chosen if self.items[i][0] == 'record' and self.items[i][1] in BUNDLE_DEFS]
        if bundles:   # a bundle (health overview) is the whole plan: its defined reads intersected with the inventory
            if len(chosen) > 1: return H('plan_breadth_exceeded')
            if span('time') or self._has_window_words(cur, ref): return H('unbound_request_constraint')
            qs = [{'kind': 'health', 'metrics': sorted(set(r['metrics']) & avail), 'records': [], 'operation': r['operation'], 'period': dict(r['period']),
                   'source': None, 'profile_fields': [], 'date_basis': 'observed_at'} for r in BUNDLE_DEFS[bundles[0]]]
            qs = [q for q in qs if q['metrics']]
            return {'status': 'planned', 'reason_codes': [], 'queries': qs, 'confidence': float(probs[0]), 'decision_confidence': float(probs[0])} if qs else H('unresolved_metric_or_record')
        panel_keys = {self.items[i][1] for i in chosen if self.items[i][0] == 'metric' and self.items[i][1] in PANELS}
        if {'panel_cbc', 'panel_cbc_basic'} <= panel_keys:      # the basic count is inside the full one: "CBC without differential drops the differential" (10g.18)
            if not re.search(r'\b(?:without|w/o|no|minus)\s+(?:the\s+)?diff', cur, re.I): return H('unresolved_health_scope')
            panel_keys.discard('panel_cbc')
        metrics = {x for i in chosen if self.items[i][0] == 'metric' and self.items[i][1] not in PANELS
                   for x in ({resolve[self.items[i][1]]} if self.items[i][1] in resolve else UNIT_SPLIT[self.items[i][1]] & avail)}
        # A named group reads every present member (every inventory id of each analyte); bounded by its list, so not capped.
        panel_metrics = {x for k in panel_keys for m in PANELS[k] for x in ({resolve[m]} if m in resolve else set()) | (SPELLING.get(m, set()) & avail)}
        for m in list(metrics):          # every inventory id of the analyte, in one read (v1.8 10g.1)
            unit = UNIT_SPLIT.get(m, set()) if not re.search(r'%|percent', cur.lower()) else set()   # "neutrophil %" stays the % id
            metrics |= (SPELLING.get(m, set()) | unit) & avail
        # Analytes the app knows but the catalog does not ("free testosterone", "free T3"): bind to the ids in this request's inventory,
        # replacing the catalog look-alike linked for the mention; none present -> handoff (spec v1.8 10g.18/10g.23).
        for term in app_terms:
            e = APP_ANALYTES[term]; present = {x for x in e['ids'] if x in avail}
            if not present: return H('requested_metric_unavailable')
            drop = {k for t, ks in self._runs if term in t for k in ks} | set(e.get('instead_of') or [])
            metrics = (metrics - {resolve.get(k, k) for k in drop}) | present
        if full_area: metrics = set(AREAS['sleep']) & avail      # bounded by its own list (10g.34 item 5)
        if full_area and not metrics: return H('requested_metric_unavailable')
        # Breadth cap (spec §3 item 13, G10): at most 8 metrics outside a named panel/bundle; the ids of one analyte count once (10g.1).
        if not full_area and len({ANALYTE.get(m, m) for m in metrics}) > 8: return H('plan_breadth_exceeded')
        if panel_keys and not panel_metrics: return H('requested_metric_unavailable')
        metrics = sorted(metrics | panel_metrics)
        pm = CONTRIB.search(cur) if len(metrics) > 1 else None
        if pm:
            parents = {resolve.get(key, key) for rx, k, key in self.syn if k == 'metric' and rx.fullmatch(pm.group(1).strip().lower())}
            if parents & set(metrics) and set(metrics) - parents: metrics = sorted(set(metrics) - parents)
        if intake_any and set(metrics) & INTAKE_READS: return H('requested_metric_unavailable')   # intake wording + a calorie/glucose read
        bundle = full_area or set(metrics) == BUNDLES[0]      # the qualitative core 5 or the full sleep area: the wording sets window and operation
        records = sorted(self.items[i][1] for i in chosen if self.items[i][0] == 'record')
        fields = sorted(self.items[i][1] for i in chosen if self.items[i][0] == 'profile_field')
        # Next to other reads, a profile link with no field and no "profile" in the request reads nothing (norm() leaves it out of the
        # agreement check for that reason), so it is no read ("What distance did I cover in July?": "cover" linked the profile).
        if 'profile' in records and not fields and not profile_named and (metrics or len(records) > 1): records.remove('profile')
        cal_only = records == ['calendar'] and not metrics and not fields     # calendar records may read the future (spec v1.8 10g.19)
        kind = KIND[out['kind'][0].argmax()]; unit = UNIT[out['unit'][0].argmax()]; offset = OFFSET[out['offset'][0].argmax()]; amount = AMOUNT[out['amount'][0].argmax()]
        time_text = span('time') or ''
        # No stated window (spec v1.7 10f item 4): a standalone request that names no window reads the registry default window.
        # A "rolling"/"all_history" kind there only echoes the default that some training labels spelled out; its amount/unit
        # heads would invent a window the user did not state. same = the kind-head answers that give this plan's window.
        same = {kind}
        if not hist and self._no_window(cur, offs, roles, base, cur_end, span_runs('time'), ref):
            nw = {'none', 'rolling', 'all_history'}
            if span_runs('time') and all(any(v.start() <= a and z <= v.end() for rx in (VAGUE_RECENCY, MOST_RECENTLY) for v in rx.finditer(cur)) for a, z in span_runs('time', pos=True)):
                nw |= {'calendar', 'day', 'to_date'}       # "these days", "lately" or "most recently" tagged as a time span is not a calendar window (10g.32)
            if kind in nw: kind, same = 'none', nw
        if anchor: kind = 'all_history'; same = set(KIND)     # a start anchor with no date: all history (spec v1.8 10g.31)
        # A tagged window that only refers back ("on that same day", "over the same days", "for that week"; spec v1.8 10g.26 (c)) states
        # no window of its own: the read takes the window the prior turn read (10g.9: it inherits as a unit), whatever the kind head
        # guessed ("that week" after "last week" is last week); when that turn does not plan on its own, the date it states, as for an
        # untagged inherited window. A unit it names must fit that window; no window to take -> handoff.
        backref = None
        refs = [m for m in BACKREF_WINDOW.finditer(cur) if any(m.start() <= a and z <= m.end() for a, z in span_runs('time', pos=True))]
        if hist and time_text and not cal_only and all(any(m.start() <= a and z <= m.end() for m in refs) for a, z in span_runs('time', pos=True)):
            prev = self.select({**req, '_prev': True, 'state': {**req['state'], 'current_request': hist[-1], 'recent_user_requests': hist[:-1]}})
            periods = [q['period'] for q in prev['queries'] if q.get('kind') == 'health'] if prev['status'] == 'planned' else []
            if not periods:
                h = next((h for h in reversed(hist) if self._has_window_words(h, ref)), '')
                try: sp = parse_time_span(h, ref) if h and not parse_rolling(h) and not UNITS_AGO_ANY.search(h) else None
                except (AmbiguousDate, InvalidDate): sp = None
                periods = [{'kind': 'between', 'start_at': sp[0].isoformat(), 'end_at': sp[1].isoformat()}] if sp and sp[1] <= ref else []
            backref = periods[0] if periods and all(p == periods[0] for p in periods) else None
            if not backref or backref.get('kind') == 'all_history' or not all(window_fits(m.group(1), backref, ref) for m in refs):
                return H('unresolved_inherited_period')
            kind, same = 'between', set(KIND)
        # A calendar/to-date kind whose unit head names no unit ("so far today", "MTD", a window inherited by "calories also") is
        # read from the date syntax of the stated or inherited window below, as day/between windows are.
        no_unit = kind in ('calendar', 'to_date') and unit not in ('day', 'week', 'month', 'year') and not cal_only
        if (kind in ('day', 'between', 'since') or no_unit) and not time_text:     # inherited window: the most recent turn that states a date
            time_text = next((h for h in reversed(hist) if self._has_date(h, ref) or UNITS_AGO_ANY.search(h)), '')   # or "N units ago" (10g.24)
        ago_ok = bool(metrics or 'labs' in records) and not (set(records) - {'profile', 'labs'})   # reads units_ago_reads can build
        if hist and not time_text and ago_ok and not self_contained(cur):   # an elliptical follow-up keeps a prior "N units ago" point, whatever the kind head says (10g.26; 10g.33 items 3, 8)
            time_text = next((h for h in reversed(hist) if self._has_window_words(h, ref)), '')
            if not UNITS_AGO_ANY.search(time_text): time_text = ''
        # Qualitative sleep with no window (spec v1.8 10g.13, 10g.32): latest words -> last night; state words (now, currently, at the
        # moment, these days) -> the 30-day trend; otherwise the tense of the question picks the window, not the model's guess:
        # past ("how did I sleep", "how was my sleep") -> last night; present, present perfect or no verb -> the 30-day trend.
        vague = [m.span() for rx in (VAGUE_RECENCY, BUNDLE_STATE, BUNDLE_LATEST) for m in rx.finditer(cur)]
        if not hist and not anchor and bundle and not records and not fields \
                and not self._has_window_words(cur, ref) and all(any(v0 <= a and z <= v1 for v0, v1 in vague) for a, z in span_runs('time', pos=True)):
            said = SLEEP_PERFECT.sub(' been ', ' '.join(cur[a:z] for a, z in self._reading_sentences(cur, runs)).replace('’', "'"))
            if BUNDLE_LATEST.search(said): kind = 'night'
            elif SLEEP_PAST.search(said) and SLEEP_STATE.search(said): return H('period_disagreement')   # "how was my sleep and is it improving"; before state words (10g.32 item 2)
            elif BUNDLE_STATE.search(said): kind = 'none'
            else: kind = 'night' if SLEEP_PAST.search(said) else 'none'
            same = set(KIND)        # the wording sets the window, whatever the kind head says
        blank = TYPICAL_DAY.sub(lambda m: ' ' * len(m.group(0)), cur)      # typical-day wording is not a window; any other time word is
        typical = not hist and not anchor and blank != cur and self._no_window(blank, offs, roles, base, cur_end, [], ref) \
            and all(any(t.start() <= a and z <= t.end() for t in TYPICAL_DAY.finditer(cur)) for a, z in span_runs('time', pos=True))
        if typical: kind, same = 'none', set(KIND)     # "per day", "on a typical day", "usually": no window, whatever the kind head says (10g.33 item 4)
        # State words name no window ("How's my ApoB lately?", 10g.13); the state rule below sets the operation (10g.32 item 9). Time phrases
        # in a first-person side statement ("I'm trying to walk more in the evenings.") are context (10g.28); a window sentence counts.
        state, tspans = [m.span() for rx in (VAGUE_RECENCY, STATE_DAYS) for m in rx.finditer(cur)], span_runs('time', pos=True)
        side = lambda x, y: not any(x <= v0 < y for v0, _ in state) and side_event(cur[x:y], [(p - x, q - x) for p, q in tspans if x <= p < y])
        own_s = [(m.start(), m.end()) for m in self.SENTENCE.finditer(cur) if not side(m.start(), m.end())]
        own_text = ' '.join(cur[x:y] for x, y in own_s)
        if not hist and not bundle and state and not self._has_window_words(STATE_DAYS.sub(' ', VAGUE_RECENCY.sub(' ', own_text)), ref) \
                and all(any(v0 <= a and z <= v1 for v0, v1 in state) for a, z in tspans if any(x <= a < y for x, y in own_s)):
            kind, same = 'none', set(KIND)
        # A profile snapshot has no history: a window on a profile-only request cannot be read (spec v1.8 10g.20).
        if (fields or 'profile' in records) and not metrics and set(records) <= {'profile'} and kind not in ('none', 'all_history'): return H('profile_history_unavailable')
        # One plan carries one window: separate time phrases that resolve to different ranges need per-clause reads.
        since_at = [(m.start(), m.start() + 5 + len(SINCE_END.sub('', m.group(1)))) for m in SINCE_PHRASE.finditer(cur)] if kind == 'since' else []
        windows = {next((f'since{a}' for a, z in since_at if a <= x < z), re.sub(r'\W+', ' ', cur[x:y].lower()).strip())   # "since I switched to | Whoop | in June": one window (10g.32 item 3)
                   for x, y in span_runs('time', pos=True) if names_time(cur[x:y])} - {''}
        windows = {w for w in windows if not (FREQUENCY.fullmatch(w) or WHOLE_DAY.fullmatch(w))}      # "energy burned each day over the past 14 days": one window
        interval = plain_month_range(cur)
        if interval and any(a <= interval.start() and interval.end() <= z for a, z in said_sents) and all(interval.start() <= a and z <= interval.end() for a, z in span_runs('time', pos=True)):
            time_text = interval.group(0)
            kind, same, windows = 'between', set(KIND), {time_text}
        else:
            interval = None
        if windows and not interval and not req.get('_split'):   # the same metric set in several stated windows: one read per window (spec v1.8 10g.38)
            wc = self.window_compare(req, cur, offs, roles, base, cur_end, H)
            if wc is not None: return wc
        # A comparison with fewer than two stated windows hands off in v2 ("is my HRV higher than last month?", "has my RHR improved?", "HRV this week vs
        # my baseline", "my HRV compared to last week"): its reference is no stated window, and a single read would go to the direct template, which
        # cannot state a verdict (spec v1.8 10g.38 item 6 and "Hand off"; 10g.28 class 5). Two metrics compared in one window are one read (item e).
        # So does any yes/no question about the metric set whose predicate is no plain read ("Did my HRV change this week?", "Are my steps down
        # this week?", "Has my sleep score stabilized?", "Any change in my HRV this week?": _verdict).
        said_cmp = ' '.join(cur[a:z] for a, z in self._reading_sentences(cur, runs))
        if len(windows) <= 1 and (CMP_IMPLICIT.search(said_cmp) or CMP_VERDICT.search(said_cmp) or self._verdict(cur, runs, span_runs('time', pos=True), direction=not req.get('_cmp')) or (len(set(metrics)) + len(records) <= 1 and (
                CMP_AVERAGE_REF.search(said_cmp) or re.search(r"\b(?:vs\.?|versus|compared\s+(?:to|with)|relative\s+to|against)\s", said_cmp, re.I)))):
            return H('unbound_request_constraint')
        if len(windows) > 1:   # one read per clause (spec v1.8 10g.6): split after each window, all clauses must plan
            return self.per_clause(req, cur, text, offs, roles, base, cur_end, H)
        if len(windows) == 1 and not interval and not req.get('_split'):      # a latest clause next to a window clause: one read cannot carry both
            split = self.latest_window_split(req, cur, offs, roles, base, cur_end, H, trailing_daily=bool(metrics) and not records and not fields
                                             and all((metric_definition(m) or {}).get('fresh_days', 365) <= 14 for m in metrics))
            if split is not None: return split
        # "since <calendar period, day or date>" reads from the start of that period to now, whatever the kind head read ("since last
        # month" is no "last month"): a whole-day between (10g.35 item 3). "since N units ago" stays relative (10g.24).
        since_p = re.match(r'\s*since\s+(.+)$', time_text or '', re.I)
        if since_p and not UNITS_AGO_ANY.search(time_text):
            try: sp = parse_time_span(since_p.group(1), ref)
            except (AmbiguousDate, InvalidDate): sp = None
            if sp and sp[0] <= ref: kind, time_text, same = 'since', since_p.group(1), set(KIND)
        runs_t = [(a, z) for a, z in span_runs('time', pos=True) if names_time(cur[a:z])]
        if kind != 'since' and len(runs_t) == 1 and time_text == span('time'):
            (a, z), sents = runs_t[0], [(x.start(), x.end()) for x in self.SENTENCE.finditer(cur)]
            s0, e0 = next(((x, y) for x, y in sents if x <= a < y), (0, len(cur)))
            x = open_start(cur[s0:a], cur[a:z], cur[z:e0])
            if x is None and not hist and re.match(r"\W*going back to\b", cur[s0:z], re.I): return H('unresolved_temporal_phrase')   # "Going back to March, ...": since March or in March?
            if x:
                try: sp = parse_time_span(x, ref)
                except (AmbiguousDate, InvalidDate): sp = None
                if sp and sp[0] <= ref: kind, time_text, same = 'since', x, set(KIND)
        # Two views of the window must agree: the learned kind and the date syntax of the tagged span.
        try:
            stated = parse_time_span(span('time') or '', ref) if span('time') else None; stated_text = span('time') or ''
            if not stated and re.search(r'\b(?:tomorrow|next|upcoming|coming|ago|(?:before|preceding|prior\s+to)\s+(?:last|yesterday))\b|\d{1,2}/\d{1,2}', cur.lower()): stated, stated_text = parse_time_span(cur, ref), cur
        except AmbiguousDate: return H('ambiguous_numeric_date')
        except InvalidDate: return H('invalid_or_out_of_range_date')
        def is_range(t):     # the date syntax of t is one explicit range of days and is this span ("Sept 20 to 30", "Mon-Wed last week"; 10g.36 item 6)
            try: return stated[0] != stated[1] and parse_range(t, ref) == stated
            except (AmbiguousDate, InvalidDate): return False
        if stated is not None and not cal_only and (stated[0] > ref or (stated[1] > ref and is_range(stated_text))):
            return H('future_period_unavailable')     # observations cannot be in the future; nor can an explicit range of days reach it (10g.36 item 6)
        if NEXT_N.search(time_text or cur) and not cal_only: return H('future_period_unavailable')      # "over the next 7 days" is no past rolling window
        if LATER.search(cur) and not cal_only: return H('future_period_unavailable')      # "later this week", "the rest of the day"
        if re.search(r'\btonight\b', ' '.join([time_text] + [cur[a:z] for a, z in self._reading_sentences(cur, runs)]), re.I) and set(metrics) & NIGHT_READ:
            return H('future_period_unavailable')       # tonight's night ends tomorrow ("my steps tonight" stays today); not in a context sentence
        if re.search(r'\bq[1-4]\b|\bquarters?\b', cur.lower()): return H('unsupported_period')          # quarters are not in the period grammar
        if kind in ('none', 'all_history'):   # the model saw no window: the date syntax of the request must agree
            try: explicit = parse_time_span(re.sub(r'\b(?:right )?now\b', ' ', cur), ref)
            except AmbiguousDate: return H('ambiguous_numeric_date')
            except InvalidDate: return H('invalid_or_out_of_range_date')
            if explicit or TO_DATE_ABBR.search(cur): return H('period_disagreement')      # "hrv mtd" states a window the parser cannot read
            rs = self._reading_sentences(cur, runs)       # a rolling/calendar window the date parser does not read ("last week", "past month") disagrees too (§2.1); not one in a context sentence (10g.28)
            if self._has_window_words(re.sub(r'\bso far\b', ' ', ' '.join(cur[a:z] for a, z in span_runs('time', pos=True) if any(s <= a < e for s, e in rs))), ref): return H('period_disagreement')
        # one day or one explicit range of days ("Mon-Wed last week") is narrower than the "last week" that qualifies it (10g.3, 10g.36 item 6)
        fixed = stated is not None and not parse_rolling(time_text) and (stated[0] == stated[1] or is_range(stated_text) or
                not re.search(r'\b(?:this|last|previous|past|current|next)\s+(?:week|month|year|weekend)\b|\bto date\b|\bso far\b|\b[my]td\b', time_text.lower()))
        # One day fixed by the tagged span's date syntax: day, night and between read that day, and so do the relative kinds it replaces.
        if same == {kind} and stated is not None and stated[0] == stated[1] and span('time') and stated_text == span('time') and not cal_only:
            day = {'day', 'night', 'between'} | ({'rolling', 'calendar', 'to_date'} if fixed else set())
            if kind in day: same = day
        # An explicit range of days in the tagged span's date syntax ("Mon-Wed", "the 3rd to the 9th", "Sept 3-9") is the window: a day kind
        # reads it as between does (each end is that single day, 10g.3; an end before the start or after today stays a disagreement).
        if kind in ('day', 'between') and stated is not None and stated[0] < stated[1] <= ref and span('time') and stated_text == span('time') and not cal_only:
            try: explicit_range = parse_range(stated_text, ref) == stated
            except (AmbiguousDate, InvalidDate): explicit_range = False
            if explicit_range: kind, same = 'between', same | {'day', 'between'}
        if kind in ('rolling', 'calendar', 'to_date') and fixed:
            kind = 'day' if stated[0] == stated[1] else 'between'   # explicit date syntax ("2 days ago", "Sept 7 to 13") beats a relative guess
            time_text = stated_text
        if no_unit and kind in ('calendar', 'to_date'):
            # No unit from the head: a current week/month/year to date keeps its calendar shape (spec item 10: "this month" = MTD);
            # any other stated or inherited span is read as stated; no span -> unsupported_period, as before.
            try: span_ = parse_time_span(time_text, ref) if time_text else None
            except (AmbiguousDate, InvalidDate): span_ = None
            if span_:
                starts = {'week': ref - timedelta(days=ref.weekday()), 'month': ref.replace(day=1), 'year': ref.replace(month=1, day=1)}
                unit = next((u for u, s0 in starts.items() if span_ == (s0, ref) and s0 < ref), None)
                if unit: offset = 'current'
                else: kind = 'day' if span_[0] == span_[1] else 'between'
        # Two views of a calendar window: the offset head against the date syntax of the tagged span ("the current week" read as last week).
        if kind in ('calendar', 'to_date') and not no_unit and stated is not None and span('time') and stated_text == span('time') and (stated[1] == ref) != (offset != 'previous'):   # no offset = current, as resolve_period reads it
            return H('period_disagreement')
        if kind == 'night' and stated is not None:     # a stated date wins: "<D> night" / "the night of D" started on D (spec v1.8 10g.4)
            if stated[0] != stated[1]: return H('period_disagreement')
            kind = 'day'
        if kind == 'day' and stated is not None and stated[0] != stated[1]: return H('period_disagreement')
        # "N units ago" in this request or in the turn its window comes from (10g.24). In this request only as the read's own window:
        # inside a tagged window, or with none tagged in a sentence of the read (context is ignored, 10g.28: "I switched to green tea
        # about a month ago ... what was my HRV over the past 30 days?" reads the past 30 days).
        own = [m.group(0) for m in UNITS_AGO_ANY.finditer(cur) if (any(a < m.end() and z > m.start() for a, z in span_runs('time', pos=True))
               if span_runs('time') else any(s <= m.start() < e for s, e in self._reading_sentences(cur, runs)))]
        ago = own[0] if own else time_text if UNITS_AGO_ANY.search(time_text or '') else None
        if ago_out_of_range(' '.join(own) if own else ago): return H('unsupported_period')      # "HRV 99999999 days ago", "steps since 600 weeks ago" (10g.24)
        ago_read = bool(ago) and ago_ok     # "N units ago": units_ago_reads builds the reads (10g.24), a labs read too (10g.33 item 8)
        if cal_only:
            wording = self._calendar_read_text(cur, runs)
            if not mention_runs(roles, offs, base, cur_end) and hist:
                wording += ' ' + self._calendar_read_text(hist[-1])
            cal_reads = self.calendar_reads(kind, unit, offset, amount, time_text, wording, ref, span_runs('time'))
            if isinstance(cal_reads, str): return H(cal_reads)
            period, night = {'kind': 'all_history'}, False
        else:
            if kind == 'rolling' and not backref and not parse_rolling(time_text) and not UNITS_AGO_ANY.search(time_text or '') and SPOKEN_N.search(time_text or ''):
                n_said = next((int(m.group(1)) if m.group(1) else num_words(m.group(2)) for m in re.finditer(rf"\b(\d{{1,3}})\b|\b({NUM_WORDS})\b", time_text, re.I)), None)
                u_said = re.search(r'\b(day|week|month|year)s?\b', time_text, re.I)
                if n_said != amount or not u_said or u_said.group(1).lower() != unit:
                    return H('unsupported_period')      # a count the parser cannot read ("the last sixty days") must agree with the heads ("steps 7 days"); "since 2 weeks ago" is 10g.24's rule
            try: period = dict(backref) if backref else resolve_period(kind, amount, unit, offset, ref, time_text, parse_time_span)   # a back-reference: the prior read's window
            except AmbiguousDate: return H('ambiguous_numeric_date')        # a window inherited from an earlier turn ("Steps on 31.02." -> "and HRV?")
            except InvalidDate: return H('invalid_or_out_of_range_date')
            if period is None and ago_read: period = {'kind': 'all_history'}   # 10g.24: units_ago_reads builds the window below
            if period is None: return H('unsupported_period')
            night = period.pop('night', False)
        op = None       # set by a window or wording rule below, else the operation head decides (op_head)
        said_read = ' '.join(cur[a:z] for a, z in self._reading_sentences(cur, runs))   # the operation words of the read, not of context sentences (10g.28)
        state_said = ' '.join(m.group(0) for m in self.SENTENCE.finditer(cur) if STATE_DAYS.search(m.group(0)))   # "been" counts in the state word's own sentence
        if kind not in ('none', 'all_history'): op = 'trend'    # any stated window is a trend for metric reads (spec v1.7 §3 item 9)
        elif (LATEST_OP.search(said_read) or re.search(r'\blast\s+(?:[\w-]+\s+){0,6}(?:value|reading|measurement)\b', said_read, re.I)) and not bundle:
            op = 'latest'                                       # explicit latest words win (spec v1.8 10g.15); a leading "now" is a follow-up marker
        elif typical and metrics: op = 'trend'                  # typical-day wording: the default-window series carries the mean (10g.33 item 4)
        elif STATE_DAYS.search(said_read) and metrics and not records and not re.search(r'\bbeen\b', state_said, re.I) and not TREND_NOUNS.search(state_said) and not bundle \
                and len({(metric_definition(m) or {}).get('fresh_days', 365) >= 30 for m in metrics}) == 1:
            op = 'latest' if (metric_definition(metrics[0]) or {}).get('fresh_days', 365) >= 30 else 'trend'   # current state (10g.32)
        elif TREND_NOUNS.search(said_read) or bundle or CHANGE_CUE.search(said_read):
            op = 'trend'    # result nouns ask for a series (10g.15); the qualitative sleep bundle is never a single latest value (10g.13)
        elif hist and ADDITIVE.search(cur) and metrics and not req.get('_clause') and not self._has_window_words(cur, ref) and not TREND_NOUNS.search(said_read) and not re.search(r'\b(?:plot|graph|chart|visuali[sz]e)\b', said_read, re.I):
            prior = self.select({**req, '_prev': True, 'state': {**st, 'current_request': hist[-1], 'recent_user_requests': hist[:-1]}})
            ops = {q.get('operation') for q in prior.get('queries', []) if q.get('kind') == 'health'}
            if prior['status'] != 'planned' or len(ops) != 1: return H('unresolved_inherited_period')
            op = next(iter(ops))
        elif SHOW_VERBS.search(NARROW_SAY_RX.sub(' ', said_read) if narrow else said_read) and metrics: op = 'trend'   # "show / pull up / see my X" with no window: the series (spec §3 item 9); not "my CBC show for hemoglobin" (10g.36 item 5)
        elif (narrow or (NARROW_TEXT.match(said_read) and len(metrics) == 1)) and metrics: op = 'latest'   # one member picked from a panel or area ("Bdy comp: just muscle mass"), no window or trend word: item 9 (10g.34 item 4)
        elif not hist and metrics and any(k.startswith('panel_') for k in panel_keys) and set(re.findall(r"[a-z']+", ''.join(' ' if any(a <= i < z for a, z in runs) else c for i, c in enumerate(cur.lower())))) <= BARE_WORDS:
            op = 'latest'      # nothing but the panel name (and fillers): its latest read, whatever the operation head guessed (10g.18); series words were handled above
        elif hist and metrics and not req.get('_clause') and not SERIES_CUE.search(cur) and not self_contained(cur) and (
                ADDITIVE.search(cur) or FOLLOW_MARKER.match(FILLERS.sub('', cur.strip())) or INHERIT[int(out['inherit'][0].argmax())] in ('period', 'both')):
            # An elliptical follow-up that states no window or operation ("triglycerides too", "and my active calories?") keeps the
            # prior read's operation (spec v1.8 10f, 10g.9, 10g.26: window and operation inherit); the operation head only guesses it.
            # It must carry a 10g.26 marker or be read as inheriting by the model: "By the way, what's my X?" fails the predicate test
            # of self_contained but asks afresh.
            prev = self.select({**req, '_prev': True, 'state': {**req['state'], 'current_request': hist[-1], 'recent_user_requests': hist[:-1]}})
            ops = {q['operation'] for q in prev['queries'] if q.get('kind') == 'health' and q.get('metrics')} if prev['status'] == 'planned' else set()
            if len(ops) == 1: op = ops.pop()
        if anchor: op = 'trend'     # a start anchor asks for the series since then (10g.31)
        if absent & chosen and op == 'trend': return H('profile_history_unavailable')   # weight/height history or trend needs the metric (10g.2)
        if absent & chosen:     # ... and so does change or history wording on it ("How has my BMI changed over time?"); a name is not wording
            said = cur.lower()
            for a, z in runs + [x.span() for rx, _, _ in self.syn for x in rx.finditer(said)]: said = said[:a] + ' ' * (z - a) + said[z:]   # "family history"
            if PROFILE_SERIES.search(said): return H('profile_history_unavailable')
        # "so far" with no other window: to date in the metric's natural period (spec v1.8 10g.14), from registry freshness.
        if re.search(r'\bso far\b', cur.lower()) and {re.sub(r'\W+', ' ', t.lower()).strip() for t in span_runs('time')} <= {'so far', ''} and metrics:
            op = 'trend'
            flags = [(metric_definition(m) or {}).get('fresh_days', 365) <= 14 for m in metrics]
            if any(flags) and not all(flags): return H('multiple_periods_require_clause_binding')   # "steps and my weight so far": today for one, all history for the other (10g.14 is per metric): two reads, not one
            period = {'kind': 'calendar', 'period': 'day'} if all(flags) else {'kind': 'all_history'}
        op_head = op is None
        if op_head: op = OPERATION[out['operation'][0].argmax()]
        src = SOURCE[out['source'][0].argmax()]
        # Every named source must bind to the plan's one source (registry ids via source_aliases.json); two sources need separate
        # clauses (spec v1.8 10g.5, 10f 11); a name the aliases do not know cannot be bound.
        found = lambda t: {v for a, v in SOURCE_ALIASES.items() if re.search(r'(?<![\w-])' + re.escape(a) + r'(?![\w-])', t.lower())}
        # Only names in the sentences of the read count: a device named in a context sentence sets no source (10g.28, 10g.32).
        in_read = lambda a, rs=self._reading_sentences(cur, runs): any(s0 <= a < e0 for s0, e0 in rs)
        hits = [(m.start(), m.end(), v, a) for a, v in SOURCE_ALIASES.items() for m in re.finditer(r'(?<![\w-])' + re.escape(a) + r'(?![\w-])', cur.lower()) if in_read(m.start())]
        tagged = [(a, z) for a, z in span_runs('source', pos=True) if in_read(a)]     # a tagged span also names the aliases it overlaps ("ven" + "u")
        named = set().union(*[found(cur[a:z]) or {v for a0, z0, v, _ in hits if a0 < z and z0 > a} or {None} for a, z in tagged]) | {v for _, _, v, _ in hits}
        if any(in_read(m.start()) for m in NOT_INGESTED.finditer(cur)): named.add(None)     # a brand Vita does not ingest, even untagged ("my Strava distance")
        if any(SOURCE_EXCLUSION.search(cur[:a0]) for a0, _, _, _ in hits) or any(in_read(m.start()) for al in SOURCE_ALIASES for m in re.finditer(
                r'(?<![\w-])non-?\s*' + re.escape(al) + r'(?![\w-])|(?<![\w-])' + re.escape(al) + r'-(?:free|less)\b', cur.lower())):
            return H('exclusion_or_filter')     # an excluded source (SOURCE_EXCLUSION): leave-out, not a filter (10g.28 class 4)
        if len(named - {None}) > 1: return H('multiple_sources_require_separate_clauses')
        # The source head can miss a device name ("venu 3 resting hr", "iPhone distance"): the one named alias binds when the head says
        # none and the name is in the sentence of the read (not context: "Should I buy a Whoop? What's my HRV?"). An alias that is
        # also an everyday word ("quest", "polar") binds only where the tagger marks it as a source.
        if src is None and len(named) == 1 and None not in named and any(
                (al not in COMMON_ALIASES or any(a < z0 and z > a0 for a, z in tagged)) and any(s <= a0 < e for s, e in self._reading_sentences(cur, runs))
                for a0, z0, _, al in hits):
            src = next(iter(named))
        if named and (None in named or src not in named): return H('unbound_source')
        if anchor and anchor[0] == 'device' and (src != anchor[1] or set(records) - {'profile'} or
                                                 any(src not in ((metric_definition(m) or {}).get('valid_sources') or []) for m in metrics)):
            return H('unresolved_temporal_phrase')      # the device's own series cannot answer this read; its start date is unknown (10g.31)
        if src and not (named - {None}) and not (hist and not mention_runs(roles, offs, base, cur_end)):
            src = None      # 10g.9: a follow-up naming a new metric inherits window and operation, not the source; no unnamed sources
        if hist and not named and not mention_runs(roles, offs, base, cur_end):
            prior = self.select({**req, '_prev': True, 'state': {**st, 'current_request': hist[-1], 'recent_user_requests': hist[:-1]}})
            prior_qs = prior.get('queries', [])
            sources = {q.get('source') for q in prior_qs}
            if prior['status'] != 'planned' or len(sources) != 1: return H('unbound_source')
            src = next(iter(sources))
        if src and any(m not in CATALOG for m in metrics): return H('unbound_source')   # an id Vita's registry does not know ("hscrp", "lp_a") takes no source filter (10g.11, 10g.34 item 3)
        if src and any(k.startswith('panel_') for k in panel_keys): return H('unbound_source')   # a named panel takes no source (10g.11/10g.18); metric groups (workout_hr) use the members' sources
        if src and set(records) - {'profile'}: return H('unbound_source')    # record reads take no source filter (contract: query_plan.py)      # a group read takes no source until Vita validates sources on unregistered labs (10g.11/10g.18)
        # A named device that is not a valid source of every read metric cannot be planned (contract: query_plan.py; spec v1.7 10f 11).
        if src and any(src not in ((metric_definition(m) or {}).get('valid_sources') or []) for m in metrics): return H('unbound_source')
        # Joint confidence = the weakest decision the plan depends on (selective prediction).
        pmax = lambda n: float(out[n][0].softmax(-1).max())
        heads = ['operation', 'kind', 'source', 'inherit'] + (['unit', 'offset', 'amount'] if kind in ('rolling', 'calendar', 'to_date') else [])
        lp = torch.sigmoid(out['link'][0]); allowed = torch.tensor([ok(k, key) for k, key, _ in self.items])
        rest = lp.masked_fill(~allowed, 0); rest[list(chosen)] = 0
        rest[[i for i, (_, key, _) in enumerate(self.items) if full_area or key in narrow]] = 0   # accounted for by the wording (10g.34 items 4, 5)
        for i, (k, key, _) in enumerate(self.items):   # a bare split id outside the inventory reads only its present ids: once the plan
            if k == 'metric' and key in UNIT_SPLIT and key not in resolve and UNIT_SPLIT[key] & avail <= set(metrics): rest[i] = 0   # reads them it is no alternative
        link_in = float(lp[list(chosen)].min()) if chosen else 1.0      # an app-bound analyte has no catalog link
        link_conf = link_in * (1 - float(rest.max()))
        # A one-night decode rests on the model's mixed_windows decision (the trained target for those reads), not on a plan decision.
        p_status = float(probs[1]) * float(out['handoff'][0].softmax(-1)[HANDOFF.index('mixed_windows')]) if one_night else float(probs[0])
        conf = min([p_status, link_conf] + [pmax(n) for n in heads])
        if conf < self.conf_threshold: return {**H('low_joint_confidence'), 'confidence': conf}
        # Decision confidence (accept rule tier 2, exp/fixes/calibration/ACCEPT.md): the same parts, restricted to the decisions that
        # reach this plan, so never below conf. A head counts only where its answer is read: the operation when no window or wording
        # rule set it and a metric read carries it; the source when one is named in the read or a follow-up naming no item keeps it;
        # the inherit head with history or when the request itself links nothing; unit/offset/amount as the period grammar reads them
        # (a stated rolling number wins). The kind answers that give this window count together (same). A link competitor that the
        # agreement check maps onto the chosen items (the profile record, a chosen metric's profile twin, another id of a chosen
        # analyte) is no alternative.
        if ago_read: same = set(KIND)                                             # whatever the kind head said
        stated_n = kind == 'rolling' and parse_rolling(time_text)
        used = {'operation': op_head and bool(metrics) and not ago_read, 'source': bool(named) or bool(hist and not runs), 'inherit': bool(hist) or self._inherit_read,
                'unit': (kind in ('calendar', 'to_date') and not no_unit) or (kind == 'rolling' and not stated_n),     # no_unit: the date syntax set it
                'offset': kind in ('calendar', 'to_date') and not no_unit, 'amount': kind == 'rolling' and not stated_n}
        seen = norm(chosen)
        comp = next((float(v) for v, i in zip(*rest.sort(descending=True)) if not norm(prefer_metric({int(i)})) <= seen), 0.0)
        decision = min([p_status, link_in * (1 - comp), float(out['kind'][0].softmax(-1)[[i for i, k in enumerate(KIND) if k in same]].sum())]
                       + [pmax(n) for n in heads if n != 'kind' and used[n]])
        # Today has one shape: a calendar day (a calendar kind with unit day, the "so far" rule) is the one-day window on the
        # reference date that a day kind reads for "today" (spec 10f: "my X today" = a bounded one-day window; 10g.4: "today" = the
        # night ending on the reference date), so the two readings of one request give one plan.
        if period == {'kind': 'calendar', 'period': 'day'}: period = {'kind': 'between', 'start_at': ref.isoformat(), 'end_at': ref.isoformat()}
        # "last night" keeps the night basis only for night-capable metrics; others read the reference day (spec v1.7 10f item 5).
        # One night or one named day: NIGHT_METRICS read on the sleep_end_day basis, every other metric on a one-day observed_at
        # window of the same day (spec v1.8 10g.4); longer windows stay a single observed_at read.
        one_day = night or (period.get('kind') == 'between' and period.get('start_at') == period.get('end_at'))
        night_part = sorted(set(metrics) & set(NIGHT_METRICS)) if one_day else []
        # No stated window on a trend: the default is explicit (v1.7 10f item 4), from Vita's registry freshness:
        # fresh_days <= 14 (daily/nightly wearables) -> 30 days; sparse (labs, body composition) -> all history.
        classes = None
        daily_m = {m for m in metrics if (metric_definition(m) or {}).get('fresh_days', 365) <= 14}
        mixed = bool(daily_m) and len(daily_m) < len(metrics)
        # Daily and sparse metrics with no stated window: each takes its own default, so two reads (10f item 4; 10g.6). With state words ("these
        # days", "lately") each takes its own state reading (10g.32 item 9): the daily one a 30-day trend, the sparse one its latest value.
        state_mixed = mixed and not records and not bundle and bool(STATE_DAYS.search(said_read)) and not re.search(r'\bbeen\b', state_said, re.I) \
            and not TREND_NOUNS.search(state_said) and not LATEST_OP.search(said_read)
        if kind == 'none' and period == {'kind': 'all_history'} and metrics and (op == 'trend' or state_mixed):   # not a "so far" day (10g.14)
            if state_mixed: classes = [(sorted(daily_m), 'trend', {'kind': 'relative', 'amount': 30, 'unit': 'days'}), (sorted(set(metrics) - daily_m), 'latest', dict(period))]
            elif mixed: classes = [(sorted(daily_m), 'trend', {'kind': 'relative', 'amount': 30, 'unit': 'days'}), (sorted(set(metrics) - daily_m), 'trend', dict(period))]
            elif daily_m: period = {'kind': 'relative', 'amount': 30, 'unit': 'days'}
        qs = []
        if fields: records = sorted(set(records) | {'profile'})
        if metrics:
            rest = sorted(set(metrics) - set(night_part))
            # A current period reads in its canonical calendar shape (10g.35 item 1): one day on the reference date ("today", a daily "so
            # far") is the calendar day, "this week"/"this month" (as worded: "since Monday" stays between) the calendar week/month; a night
            # read (the 10g.4 split), a "since" window and past periods keep whole-day between.
            wtext = ' '.join([time_text or ''] + [cur[a:z] for a, z in self._reading_sentences(cur, runs)])
            now = 'day' if period == {'kind': 'between', 'start_at': ref.isoformat(), 'end_at': ref.isoformat()} else \
                  next((u for u, rx in (('week', WEEK_NOW), ('month', MONTH_NOW)) if rx.search(wtext)), None)
            start = {'day': ref, 'week': ref - timedelta(days=ref.weekday()), 'month': ref.replace(day=1)}.get(now)
            shaped = {'kind': 'calendar', 'period': now} if now and not night_part and kind != 'since' and period in (
                {'kind': 'calendar', 'period': now}, {'kind': 'between', 'start_at': start.isoformat(), 'end_at': ref.isoformat()}) else period
            for ms, basis, o, p in ([(ms, 'observed_at', o, p) for ms, o, p in classes] if classes else
                                    [(ms, basis, op, dict(shaped if basis == 'observed_at' else period)) for ms, basis in ((night_part, 'sleep_end_day'), (rest, 'observed_at'))]):
                if ms: qs.append({'kind': 'health', 'metrics': ms, 'records': [], 'operation': o, 'period': p, 'source': src,
                                  'profile_fields': [], 'date_basis': basis})
        for r in records:
            if r == 'calendar':     # one read per date basis (spec v1.8 10g.19)
                for b, p in (cal_reads if cal_only else self.calendar_bases(self._calendar_read_text(cur, runs), period, ref, span_runs('time'))):
                    qs.append({'kind': 'health', 'metrics': [], 'records': ['calendar'], 'operation': 'latest', 'period': p, 'source': None,
                               'profile_fields': [], 'date_basis': b})
                continue
            qs.append({'kind': 'health', 'metrics': [], 'records': [r], 'operation': 'latest',
                       'period': {'kind': 'all_history'} if r == 'profile' else period, 'source': src if r != 'profile' else None,
                       'profile_fields': (fields or (['all'] if (records == ['profile'] and not metrics) or profile_named else [])) if r == 'profile' else []})   # "show my profile" (10g.32), "my profile and my steps" (10g.20)
        if hist and (not span('time') or backref) and kind not in ('none', 'all_history') and not req.get('_clause') and not ago_read:   # the window was inherited (an inherited "N units ago" is a point anchor, re-applied below: 10g.33 item 3)
            def span_days(p):
                bounds = self.period_span(p, ref)
                return (bounds[1] - bounds[0]).days + 1 if bounds else 10**9
            short = [x for x in metrics if (metric_definition(x) or {}).get('fresh_days', 365) >= 30 and span_days(period) < (metric_definition(x) or {}).get('fresh_days', 365)]
            if short or 'labs' in records: return H('unresolved_inherited_period')
        if ago_read:     # "HRV two weeks ago" (spec v1.8 10g.24)
            qs = self.units_ago_reads(metrics, ago, ref, src, None, 'labs' in records) + [q for q in qs if q.get('records') and q['records'] != ['labs']]
        # A window-change follow-up that resolves to the window the previous turn already read asked for a different one
        # ("the previous week?" after "last week"): unclear which, so hand off.
        if hist and not req.get('_prev') and not req.get('_clause') and span('time') and not mention_runs(roles, offs, base, cur_end):
            prev = self.select({**req, '_prev': True, 'state': {**req['state'], 'current_request': hist[-1], 'recent_user_requests': hist[:-1]}})
            key = lambda r: json.dumps(sorted(json.dumps(q, sort_keys=True) for q in r['queries']))
            if prev['status'] == 'planned' and key(prev) == key({'queries': qs}): return H('multiple_periods_require_clause_binding')
            # It re-reads the prior read's items and nothing else (spec §3 item 11, 10g.9): a subject that differs was guessed.
            items = lambda r: {x for q in r['queries'] for x in (q.get('metrics') or []) + (q.get('records') or [])}
            if prev['status'] == 'planned' and items(prev) != items({'queries': qs}): return H('ambiguous_followup_subject')
            if prev['status'] == 'planned' and len(items(prev)) > 1 and re.search(r'\b(?:that one|this one)\b', cur, re.I): return H('ambiguous_followup_subject')
        return {'status': 'planned', 'reason_codes': [], 'queries': qs, 'confidence': conf, 'decision_confidence': decision}
