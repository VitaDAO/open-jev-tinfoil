"""Strict supplementary grading shared by local and sealed evaluations.

Topic expectations come from gold annotations or the *entire* explicit research
subject in the request, never from model output. Unsupported gold shapes remain
ungraded rather than being reported correct.
"""
import json
import re


def topic_key(text):
    text = text.casefold().replace('–', '-').replace('—', '-')
    text = re.sub(r'\b(zone|omega|type|stage|phase|grade|class|glp|covid|il)[ -]+(\d+)', r'\1\2', text)
    text = re.sub(r'^\s*(?:the|an?)\s+', '', text)
    text = re.sub(r'^effects? of\s+', '', text)
    text = re.sub(r'\blp\s*\(?a\)?\b', 'lpa', text)
    # Punctuation does not change biomedical names: Lp(a), NAD+, HRV.
    text = re.sub(r'[()]', ' ', text)
    return re.findall(r"[a-z0-9]+(?:\+[a-z0-9]*)?", text)


def expected_topic(case):
    expected = case['expected']
    if expected.get('topic'):
        return expected['topic']
    if expected.get('topics') and len(expected['topics']) == 1:
        return expected['topics'][0]
    text = case['current_request'].strip()
    # Research cue and subject delimiter must occur in that order. Capture the
    # complete remaining subject, including negation, populations and numbers.
    match = re.search(r'\b(?:research|studies|study|papers?|literature|science|evidence|trials?|meta-analys[ei]s|published(?: data)?)\b.*?\b(?:about|on|for|regarding|comparing|linking|that)\s+(.+)', text, re.I)
    if not match:
        match = re.match(r'^\s*(?:papers?|research|studies|literature)\s*:\s*(.+)', text, re.I)
    if not match:
        return None
    subject = match[1].strip(' .?!')
    previous = None
    while subject != previous:
        previous = subject
        subject = re.sub(r'[.,?!;]?\s+(?:please|pls|plz|ty|ta|cheers|much appreciated|appreciate it|thanks(?: a lot| in advance| very much)?|thank you(?: very much)?)$', '', subject, flags=re.I).rstrip(' .?!')
    return subject or None


def grade(case, result, health_match):
    expected = case['expected']; status = expected['status']
    if result['status'] != 'planned':
        return 'missed' if status in ('plan', 'research') else 'ok_handoff'
    if status == 'handoff':
        return 'FALSE_ACCEPT'
    queries = result.get('queries') or []
    research = [q for q in queries if q.get('kind') == 'research']
    if status == 'research':
        topic = expected_topic(case)
        if topic is None:
            return 'UNSCORED_RESEARCH'
        return 'ok_research' if len(queries) == len(research) == 1 and topic_key(topic) == topic_key(research[0].get('topic', '')) else 'FALSE_ACCEPT'
    if status == 'either' and not expected.get('reads'):
        return 'UNSCORED_PLAN'
    if len(queries) != len(expected.get('reads') or []):
        return 'FALSE_ACCEPT'
    if research or not queries:
        return 'FALSE_ACCEPT'
    signatures = [json.dumps(q, sort_keys=True) for q in queries]
    if len(signatures) != len(set(signatures)):
        return 'FALSE_ACCEPT'
    # A distinct query must satisfy each read, including explicitly annotated
    # metadata. Per-pair matching preserves calendar reads with distinct bases.
    reads = expected.get('reads', [])
    def same(read, query):
        return health_match([read], [query]) and all(query.get(k) == read[k] for k in ('limit', 'date_basis') if k in read)
    def assign(index, used):
        return index == len(reads) or any(j not in used and same(reads[index], query) and assign(index + 1, used | {j}) for j, query in enumerate(queries))
    return 'ok_plan' if assign(0, frozenset()) else 'FALSE_ACCEPT'
