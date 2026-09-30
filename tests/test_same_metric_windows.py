"""Same metric, two windows (spec v1.8 10g.38) in the repo guard layer. A comparison of the same metric across explicitly stated,
non-overlapping windows is one read per window, whatever the phrasing: a side sentence that names only a window, or compares the read
with another window, continues the read (item 5); a factual comparative with two windows reaches the parser (items 6, a); a style or
format instruction keeps a multi-read plan and hands off a single read (10g.28 class 3 [refined 10g.38]). Relationships, implicit
references, verdicts with one window or an open one, other people and normative judgement still hand off, and acute content first."""
import pytest

import query_selector
from query_plan import QueryRequest
from query_selector import unexplained_sentence
from test_p0_rule_fixes import selector  # noqa: F401  (fixture)

METRICS = ['heart_rate_variability', 'resting_heart_rate', 'steps', 'sleep_score', 'ldl', 'weight', 'calories_active', 'vo2_max']


def request(text, history=()):
    return QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': list(history), 'reference_date': '2026-09-26', 'time_zone': 'Europe/Bucharest'},
        'reference_time': '2026-09-25T22:41:32Z', 'available_metrics': METRICS,
        'available_record_types': ['profile', 'labs', 'workouts', 'calendar'], 'available_sources': ['oura', 'whoop'], 'literature_available': True})


def read(start, end):
    return {'kind': 'health', 'metrics': ['heart_rate_variability'], 'records': [], 'operation': 'trend',
            'period': {'kind': 'between', 'start_at': start, 'end_at': end}, 'source': None, 'date_basis': 'observed_at'}


ONE = [read('2026-09-14', '2026-09-20')]
TWO = [read('2026-09-14', '2026-09-20'), read('2026-09-07', '2026-09-13')]


class StubParser:
    def __init__(self, queries): self.queries, self.calls = queries, 0
    def select(self, req):
        self.calls += 1
        return {'status': 'planned', 'queries': self.queries}


def outcome(monkeypatch, selector, text, queries=TWO, history=()):
    """(status, reason, parser calls) on the learned path with a parser that returns `queries`."""
    stub = StubParser(queries)
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: stub)
    result = selector.select_query(request(text, history))
    return result['status'], (result['reason_codes'] or [None])[0], stub.calls


@pytest.mark.parametrize('text', [
    # 10g.38 item 5: a second sentence naming only a window is a clause-bound continuation, not another question (class 8).
    'Show my steps last week. Also show the week before.',
    'What was my HRV last week? What about the week before?',
    'What was my HRV last week? And the week before?',
    'Show my steps last week. What about the week before?',
    'Show my steps this week. Then the week before that.',
    'Show my resting heart rate this month. Also show the same period last month.',        # 10g.38 c
    'What was my HRV on Tuesday? And the day before?',
    'Show my steps last month. And 2 months before?',
    'Show my HRV last week. And the one before that?',
    'Show my steps yesterday. And the day before?',
    'What was my HRV last month? What about the month before?',
    'Show my HRV yesterday. And the day before yesterday?',
    'Show my steps last week. And the week before last?',
    'Show my HRV last week. Hey, and the week before?',                                     # an interjection is not content
    'Show my steps this week. Also last week.', 'Steps yesterday. Then today.',            # already plain window clauses
])
def test_a_side_sentence_naming_only_a_window_continues_the_read(monkeypatch, selector, text):
    assert not unexplained_sentence(request(text)), text
    assert outcome(monkeypatch, selector, text) == ('planned', None, 1), text


def test_a_follow_up_naming_only_a_window_after_an_interjection_continues_the_read():
    assert not unexplained_sentence(request('yo, the week before? plz', ['What was my HRV last week?']))


@pytest.mark.parametrize('text', [
    # A relative window is a window for 10g.38, not a date that makes its sentence a window clause: another question next to it is
    # still another question (10g.28 class 8, 10g.6 all or nothing), whatever the parser returns.
    'Show my HRV today. What did I eat the day before?',
    'Show my steps today. Where was I the day before?',
    'Show my HRV today. What was my mood the day before?',
    'Show my HRV this week. Did I drink much the week before?',
    'Show my steps last week. What was I doing the week before?',
    'Show my HRV yesterday. How much coffee did I have the day before?',
    'Show my sleep score last night. What time did I go to bed the night before?',
    'Show my resting heart rate this week. What was my calorie intake the same week?',
    'Show my steps last week. Who did I train with the week before?',
    'Show my HRV this month. What did my doctor say the month before?',
    'Show my HRV last week. Can you remind me what I ate the week before?',
    'Show my steps today. What will the weather be the day after?',
    'Show my HRV today. Was I stressed the day before?',
    'Show my HRV this week. I felt awful the week before.',
    'Show my steps this week. I had a fever the week before.',
    'Show my HRV today. My wife was sick the day before.',
    'Show my HRV this week. Should I rest the same week next month?',
    'Show my resting heart rate last week. Is it normal to feel tired the week before?',
    'Show my steps last week. Was I travelling the week before?',
    'Show my HRV last month. Was I on holiday the month before?',
    'Show my HRV last week. What happened the week before?',
    'Show my steps today. Did I walk the dog the day before?',
    'What was my resting heart rate last week? What was I up to the week before?',
    'Show my steps this week. How many hours did I work the week before?',
    'Show my HRV last week. Did I get enough sunlight the week before?',
    'Show my HRV today. Did I go out the night before?',
    'Show my HRV today. Did I take my supplements the day before?',
    'Show my sleep score last night. What did I watch the night before?',
    'Show my HRV today. What did I do the previous day?',
    'Show my steps yesterday. Where did I go the day before?',
    'Show my steps last week. How much did I spend the week before?',
    'Show my HRV today. What was I thinking the day before?',
    'Show my weight this month. What did I weigh myself on the month before?',
    'Show my HRV this week. Did I meditate the week before?',
    'Show my steps today. Which route did I take the day before?',
    'Show my HRV this week. Did I fly anywhere the week before?',
    'Show my sleep score last week. Did I use my new pillow the week before?',
    # A window anchored on an event with no date is unresolved (10g.29 analogue), and "the same week" alone is the read's own window.
    'Show my steps last week. What about the week before my birthday?',
    'Show my HRV last week. And the week before my marathon?',
    'What was my HRV yesterday? And the day before my flight?',
    'Show my sleep score last week. And the night before my exam?',
    'Show my resting heart rate last month. Also show the month before my wedding.',
    'Show my steps this week. And the week before I moved?',
    'Show my steps last week. What about the week before my vacation?',
    'What was my resting heart rate on Tuesday? And the day before my presentation?',
    'Show my steps last week. Also 2 weeks before my operation.',
    'Show my steps last week. And the week before my holiday?',
    'Show my HRV last week. What about the week before my holiday?',
    'Show my sleep score last week. What about the week before my deadline?',
    'What was my HRV yesterday? And the day before my run?',
    'Show my steps this month. And the month before my move?',
    'Show my resting heart rate this week. And the same week as my last race?',
    'Show my HRV last week. And the week before my new job started?',
    'Show my steps last week. And the week before I got my Oura?',
    'Show my HRV last week. What about the week before I quit coffee?',
    'Show my HRV last week. And the week before my trip to Rome?',
    'Show my HRV last week. What about the same week?',
    'Show my steps this month. How does that compare with the same period?',
])
def test_a_relative_window_next_to_another_question_or_an_event_hands_off(monkeypatch, selector, text):
    assert unexplained_sentence(request(text)), text
    for queries in (ONE, TWO):
        assert outcome(monkeypatch, selector, text, queries) == ('handoff', 'unbound_request_constraint', 0), text


READS = ['What was my HRV last week?', 'Show my resting heart rate this month.', "What's my latest LDL?"]


@pytest.mark.parametrize('text', [
    # 10g.38 items 2, 6, 7, a, f, h: a comparison of the read across two stated windows, in its own sentence, before or after the read.
    *(t for side in ['How does that compare with last year?'] for r in READS for t in (r + ' ' + side, side + ' ' + r)),
    'What was my HRV last week? How does that compare with the week before?',
    'Show my resting heart rate this month. How does that compare with last month?',
    'Show my resting heart rate this month. Compare it with last month.',
    'Show my resting heart rate this month. Compared to last month?',
    'Show my resting heart rate this month. Versus last month?',
    'Show my resting heart rate this month. vs last month please.',
    'What was my HRV last week? Was it higher than the week before?',
    'What was my HRV last week? Was that better than the week before?',                      # a: better/worse between two windows
    'Show my steps this week. How does it stack up against last week?',
    "What's my current resting heart rate? How does it compare with last month?",            # 7: current vs a period
    "What's my resting heart rate now? How does that compare with last month?",
    'What was my HRV in June and July? Which was higher?',                                    # 6: the read states both windows
    'Show my HRV this week. Did it improve from last week?',                                  # 6: a change against a stated window
    'Show my HRV this week. Has it gone up compared to last week?',
    'Show my resting heart rate last week. Was it lower the week before?',
    'Show my HRV last week. Is it better now?',                                               # 7, f: a current value vs the read's window
    'Show my steps yesterday and today. Put them side by side.',                              # 4: the layout of a multi-read plan
])
def test_a_side_sentence_comparing_the_read_across_two_windows_continues_it(monkeypatch, selector, text):
    assert not unexplained_sentence(request(text)), text
    assert outcome(monkeypatch, selector, text) == ('planned', None, 1), text


@pytest.mark.parametrize('text', [
    # 10g.38 item 6: a verdict with one explicit window (the read states none), or an open one ("since"), is not a comparison of two windows.
    "What's my HRV? Is it higher than last week?",
    "What's my HRV? How does that compare with last week?",
    'Show my HRV this week. Has it gone up since last week?',
    # A change with no explicit window after it ("has it gone up?", "did it improve?") has an implicit reference, even next to a read of two
    # windows (10g.38 hand-off list, item 6).
    'Show my HRV this week and last week. Did it improve?',
    # A change over a window of its own is a verdict over that window, not a comparison of two (10g.38 item 6).
    'Show my HRV this week. Has it improved this month?',
    'Show my HRV this week. Has it gone up over the past month?',
    'Show my HRV last week. Has it gone up now?',
    'Show my resting heart rate last week. Did it improve the week before?',
    'Show my resting heart rate this month. Did it drop last month?',
    'Show my HRV since June. How does that compare with last week?',
    'Show my HRV last week. Is it higher than last week?',                                    # the same window twice
    # Implicit references (class 5), relationships, other people and norms (classes 5, 6), judgement (class 9), filters (class 4).
    *(t for side in ['How does that compare with my usual?', 'How does that compare with before?'] for r in READS for t in (r + ' ' + side, side + ' ' + r)),
    'Show my HRV last week. Is that higher than normal?',
    'Show my HRV last week. Compare it to my baseline last month.',
    'Show my HRV last week. Was it higher than usual the week before?',
    'Show my HRV last week. Did my training affect it the week before?',
    'Show my HRV last week. Compare with my brother last week.',
    'Show my HRV last week. vs people my age last month.',
    'Show my HRV last week. And my wife\'s the week before?',
    'Show my HRV last week. Is that good compared to the week before?',
    'Show my HRV last week. Why was it lower than the week before?',
    'Show my HRV this week and last week. Is it good?',
    'Show my HRV this week and last week. Is it on the same track as my step count?',
    'Show my HRV last week. Excluding the week before.',
    'Show my HRV last week. Also what\'s the weather the week before?',
])
def test_a_verdict_implicit_reference_or_protected_class_in_a_side_sentence_hands_off(monkeypatch, selector, text):
    assert unexplained_sentence(request(text)), text
    assert outcome(monkeypatch, selector, text) == ('handoff', 'unbound_request_constraint', 0), text


@pytest.mark.parametrize('text', [
    # 10g.38 items 3, 6, a, d: factual comparatives and changes between two windows the read states.
    'has my sleep score got better this month than last month?',
    'has my HRV got higher this week compared to last week?',
    'has my sleep gotten worse this week vs last week?',
    'tell me if my HRV dropped from last week to this week',
    'tell me whether my resting heart rate changed from June to July',
    'I wonder if my HRV was higher this week than last week',
    'curious whether my steps were higher in June or July',
    # already reaching the parser, kept that way
    'is my HRV higher this week than last week?', 'which was higher, my HRV in June or July?', 'did my resting heart rate improve from last month to this month?',
    'was my resting heart rate better last week or this week?', 'is my sleep score worse this month than last month?',
    'how does my HRV this week compare to last week?', 'steps last week vs the week before', "What's the difference between my steps this week and last week?",
    'has my HRV got worse last week than the week before?',                                   # a relative window with a resolved one
])
def test_a_factual_comparative_between_two_windows_reaches_the_parser(monkeypatch, selector, text):
    assert not unexplained_sentence(request(text)), text
    assert outcome(monkeypatch, selector, text) == ('planned', None, 1), text


@pytest.mark.parametrize('text', [
    # One window, an open one or an event: a verdict (10g.38 item 6); normative wording (class 9) and relationships (class 5) always.
    'has my sleep score got better this month?',
    'has my HRV got better since last week?',
    'tell me if my HRV dropped since I started running',
    'has my sleep gotten worse this week vs last week since I started drinking coffee?',
    'has my sleep score got worse this month than before my trip last month?',
    'has my HRV got healthier this month than last month?',
    'I wonder if my HRV was higher this week than last week because of my new job',
    'tell me whether my resting heart rate changed from June to July due to the heat',
    'does my HRV this week vs last week correlate with my steps?',
    "What's my weight and how closely it tracks my step count.",
    'has my HRV got worse last week than the week before my birthday?',
    # A clause after the windows (a judgement, a reason, a purpose) is content only the model may interpret (classes 5, 9).
    'I wonder if my HRV this week vs last week is normal',
    "I'm curious whether my RHR got higher this week than last week because I'm overtraining",
    'Tell me whether my HRV improved from last month to this month so I know if the new diet is working',
    'I wonder if my steps this week vs last week reflect the new job',
    'Curious if my RHR was lower this month than last month from all the running',
])
def test_a_change_verdict_without_two_windows_or_with_a_relationship_hands_off(text):
    assert unexplained_sentence(request(text)), text


@pytest.mark.parametrize('text', [
    'Show my steps this week and last week in a table.',
    'Put my steps this week and last week in a table.',
    'My HRV this week vs last week as a chart please.',
    'Show my HRV this week and last week. Put it in a table.',
    'Show my HRV this week and last week. Keep it short.',
    'Show my resting heart rate this month and last month in German please.',
    "What's my current resting heart rate vs last month, as bullet points?",
    'Show my steps last week and the week before in a table.',
    'Show my HRV last week. And the week before, in a table?',
])
def test_a_style_instruction_keeps_a_multi_read_plan_and_hands_off_a_single_read(monkeypatch, selector, text):
    # [refined 10g.38]: a multi-read plan composes, so the instruction keeps the plan; a single read goes to the direct template, which cannot honour it.
    assert outcome(monkeypatch, selector, text, TWO) == ('planned', None, 1), text
    assert outcome(monkeypatch, selector, text, ONE) == ('handoff', 'text_transformation_requires_native_context', 1), text


@pytest.mark.parametrize('text', [
    # One window stated: the plan cannot be multi-read over windows, so the handoff comes before any parser, as before.
    'Show my steps this week in a table.', 'Show my steps this week. Put it in a table.', "What's my HRV this week? Keep it short.",
    "What's my latest weight right now, as a table?",                                          # "latest" and "right now" are one current value
    # Unit systems (10g.36.3) and creative rewrites or text transformations hand off in every plan.
    'Show my weight this month and last month, imperial only.',
    'Compare my steps this week with last week and write me a haiku about autumn.',
    'Translate my HRV this week and last week into Romanian.',
])
def test_a_style_instruction_hands_off_before_the_parser_when_the_plan_cannot_compose(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text, TWO) == ('handoff', 'text_transformation_requires_native_context', 0), text


@pytest.mark.parametrize('text', [
    # Two windows in the text but one read in the plan (a window in a context sentence, a format reference): the single read hands off.
    'Show my HRV last week. Put it in a table. I trained hard this week.',
    'Put my HRV last week in a table, same format as the week before.',
])
def test_a_style_instruction_with_a_single_read_hands_off_after_the_parser(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text, ONE) == ('handoff', 'text_transformation_requires_native_context', 1), text


def test_a_style_request_that_hands_off_for_another_reason_keeps_that_reason(monkeypatch, selector):
    # The style code is the reason only when style is (a single read, or fewer than two windows); any other handoff keeps its own code.
    text = "Show my HRV this week and last week in a table. Is it normal?"
    assert outcome(monkeypatch, selector, text, TWO) == ('handoff', 'unbound_request_constraint', 0)
    text = 'Show my HRV this month vs the last 30 days in a table.'
    overlapping = [read('2026-09-01', '2026-09-26'), read('2026-08-28', '2026-09-26')]
    assert outcome(monkeypatch, selector, text, overlapping) == ('handoff', 'unbound_request_constraint', 1)
    assert outcome(monkeypatch, selector, text, ONE) == ('handoff', 'text_transformation_requires_native_context', 1)


@pytest.mark.parametrize('text', [
    'I have chest pain. Compare my HRV this week and last week.',
    'Compare my HRV this week and last week. I want to kill myself.',
    'What was my HRV last week? How does that compare with the week before? I think I am dying.',
    "My HRV this week vs last week in a table. I can't breathe properly.",
    'Show my steps last week. Also show the week before. I took too many of my sleeping pills.',
    'I ran out of my blood pressure meds. Show my resting heart rate this week vs last week.',
    'Show my resting heart rate this month. vs last month? My chest feels crushing.',
])
def test_acute_content_next_to_a_two_window_comparison_hands_off_first(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text, TWO) == ('handoff', 'acute_or_crisis_requires_model', 0), text


def reads(metric, *periods, operation='trend'):
    return [{'kind': 'health', 'metrics': [metric], 'records': [], 'operation': operation, 'period': period, 'source': None,
             'date_basis': 'observed_at'} for period in periods]


MONTH, WEEK = {'kind': 'calendar', 'period': 'month'}, {'kind': 'calendar', 'period': 'week'}
DAYS = lambda n: {'kind': 'relative', 'amount': n, 'unit': 'days'}
DATES = lambda start, end: {'kind': 'between', 'start_at': start, 'end_at': end}


@pytest.mark.parametrize('text,queries', [
    # 10g.38: overlapping windows ("VO2 max since July 15 vs last 3 weeks", follow-up f "this month vs past 90 days") and identical ones
    # ("past 14 days vs past 2 weeks") hand off. The guard cannot resolve windows, so the plan decides: two trend reads of a shared metric
    # whose windows overlap, neither one day or the latest value.
    ('Show my steps this month. Versus the last 30 days?', reads('steps', MONTH, DAYS(30))),
    ('Show my HRV for the last 30 days. Versus this month?', reads('heart_rate_variability', DAYS(30), MONTH)),
    ('Show my sleep score this month. Versus the last 30 days?', reads('sleep_score', MONTH, DAYS(30))),
    ('Show my resting heart rate this month. Versus the last 30 days?', reads('resting_heart_rate', MONTH, DAYS(30))),
    ('Show my weight this month. Versus the last 30 days?', reads('weight', MONTH, DAYS(30))),
    ('Show my resting heart rate this month. Against the past 30 days.', reads('resting_heart_rate', MONTH, DAYS(30))),
    ('My resting heart rate this year vs last month?', reads('resting_heart_rate', DATES('2026-01-01', '2026-09-26'), DATES('2026-08-01', '2026-08-31'))),
    ('HRV this month vs last week', reads('heart_rate_variability', MONTH, DATES('2026-09-14', '2026-09-20'))),
    ('Show my HRV this month. How does that compare with last week?', reads('heart_rate_variability', MONTH, DATES('2026-09-14', '2026-09-20'))),
    ('Show my HRV this year. How does that compare with the past 90 days?', reads('heart_rate_variability', DATES('2026-01-01', '2026-09-26'), DAYS(90))),
    ('Show my steps this week. Versus the last 7 days?', reads('steps', WEEK, DAYS(7))),
    ('VO2 max since July 15 vs last 3 weeks', reads('vo2_max', DATES('2026-07-15', '2026-09-26'), DAYS(21))),
    ('Show my steps for the past 14 days. Versus the past 2 weeks?', reads('steps', DAYS(14), DAYS(14))),     # identical
    ('Steps past 14 days vs past 2 weeks', reads('steps', DAYS(14), DAYS(14))),
    ('HRV this week vs last week vs the last 30 days', reads('heart_rate_variability', WEEK, DATES('2026-09-14', '2026-09-20'), DAYS(30))),
])
def test_overlapping_or_identical_windows_hand_off_on_the_plan(monkeypatch, selector, text, queries):
    assert outcome(monkeypatch, selector, text, queries) == ('handoff', 'unbound_request_constraint', 1), text


@pytest.mark.parametrize('text,queries', [
    # Windows that do not overlap, part vs whole against one day or the latest value (follow-up f), a shifted period (c), one read per
    # window for three windows (g), different sources or different metrics.
    ('Show my steps this month. Versus last month?', reads('steps', MONTH, DATES('2026-08-01', '2026-08-31'))),
    ('Show my HRV for the last 30 days. And the 30 days before?', reads('heart_rate_variability', DAYS(30), DATES('2026-07-29', '2026-08-27'))),
    ('Show my resting heart rate this month. Also show the same period last month.', reads('resting_heart_rate', MONTH, DATES('2026-08-01', '2026-08-26'))),
    ('Show my steps today. How does that compare with this week?', reads('steps', DATES('2026-09-26', '2026-09-26'), WEEK)),
    ('Show my steps yesterday. Versus the last 30 days?', reads('steps', DATES('2026-09-25', '2026-09-25'), DAYS(30))),
    ("What's my current resting heart rate? How does it compare with the last 30 days?",
     reads('resting_heart_rate', {'kind': 'all_history'}, operation='latest') + reads('resting_heart_rate', DAYS(30))),
    ('HRV this week vs last week vs the week before', reads('heart_rate_variability', WEEK, DATES('2026-09-14', '2026-09-20'), DATES('2026-09-07', '2026-09-13'))),
    ('Compare my Oura HRV with my Whoop HRV this month.', [{**reads('heart_rate_variability', MONTH)[0], 'source': 'oura'},
                                                          {**reads('heart_rate_variability', MONTH)[0], 'source': 'whoop'}]),
    ('Show my steps this month and my HRV for the last 30 days.', reads('steps', MONTH) + reads('heart_rate_variability', DAYS(30))),
])
def test_windows_that_do_not_overlap_or_a_day_against_its_period_keep_the_plan(monkeypatch, selector, text, queries):
    assert outcome(monkeypatch, selector, text, queries) == ('planned', None, 1), text


def test_overlapping_windows_hand_off_on_the_rule_path_too(monkeypatch):
    from query_plan import HealthRead
    from query_selector import QuerySelector
    selector = QuerySelector.__new__(QuerySelector)
    monkeypatch.delenv('OPEN_JEV_PARSER', raising=False)
    monkeypatch.setattr(selector, 'health', lambda text, req, diagnostics: [HealthRead(metrics=['steps'], period=DAYS(7 if 'week' in text else 30))])
    result = selector.select_query(request('Show my steps last week; show my steps last month'))
    assert (result['status'], result['reason_codes']) == ('handoff', ['unbound_request_constraint'])


@pytest.mark.parametrize('text', [
    # A side sentence that continues the read across windows asks for one read per window (10g.38 rule, items 2, 5, 7): a single read has
    # dropped one, whatever the parser decided.
    'Show my steps last week. Also show the week before.',
    'What was my HRV last week? What about the week before?',
    'What was my HRV on Tuesday? And the day before?',
    'Show my steps this month. Versus the last 30 days?',
    'Show my resting heart rate this month. Against the past 30 days.',
    'What was my HRV last week? How does that compare with the week before?',
    'Show my resting heart rate this month. Versus last month?',
    "What's my current resting heart rate? How does it compare with last month?",
    'Show my HRV this week. Did it improve from last week?',
    'Show my HRV last week. Is it better now?',
])
def test_a_single_read_for_a_side_sentence_continuing_the_read_across_windows_hands_off(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text, ONE) == ('handoff', 'unbound_request_constraint', 1), text


@pytest.mark.parametrize('text,history', [
    # A follow-up that names only a window is the read's own window (10g.25): one read. So is a plain window clause for a read without one.
    ('Hey. and the month before? Cheers.', ['HRV last month']),
    ('yo, the week before? plz', ['What was my HRV last week?']),
    ('Show my HRV. Last week please.', []),
    ('Show my steps. For this week.', []),
])
def test_a_window_that_is_the_reads_own_keeps_a_single_read(monkeypatch, selector, text, history):
    assert outcome(monkeypatch, selector, text, ONE, history) == ('planned', None, 1), text


@pytest.mark.parametrize('text', [
    # A factual comparative between two windows plans only with nothing but its wording outside its names and windows (10g.38 items 6, a):
    # an implicit reference, a norm, a manner, another person or a group, before or after the windows, hands off (10g.38 hand-off list,
    # classes 5, 6, 9), whatever the parser returns.
    'Has my HRV got worse than usual this week and last week?',
    'I wonder if my HRV dropped below my baseline this week and last week',
    "Curious if my HRV got lower than what's normal for my age between last week and this week",
    "Has my HRV got worse than Anna's from last week to this week?",
    "Wondering if my HRV has gotten lower than my ex's between last week and this week",
    'I wonder if my HRV dropped more than it did for other runners from last week to this week',
    'Has my weight got higher than my ideal weight between last month and this month?',
    'I wonder if my resting heart rate changed worryingly from last week to this week',
    'Has my resting heart rate got higher than my target from last month to this month?',
    'I wonder if my steps dropped below my goal between last week and this week',
    'Has my HRV got lower than expected this week and last week?',
    'Tell me whether my HRV got lower than it should be from last week to this week',
    'Curious if my resting heart rate got higher than the normal range between last month and this month',
    "Has my sleep score got worse than what's typical for me this week and last week?",
    'I wonder if my steps dropped below my previous best from last month to this month',
    'Wondering if my resting heart rate changed abnormally between last week and this week',
    'Has my HRV got worse than the average this week and last week?',
    'I wonder if my weight got higher than most people from last month to this month',
    "Has my HRV got lower than Tom's between last week and this week?",
    'Tell me if my sleep score got worse than the recommended level from last week to this week',
    'I wonder if my HRV dropped from last week to this week compared to my usual',
])
def test_an_implicit_reference_norm_or_other_person_in_a_two_window_comparative_hands_off(monkeypatch, selector, text):
    assert unexplained_sentence(request(text)), text
    for queries in (ONE, TWO):
        assert outcome(monkeypatch, selector, text, queries) == ('handoff', 'unbound_request_constraint', 0), text


@pytest.mark.parametrize('text', [
    # The wording of a factual comparative in the frames the reference-less forms use (10g.38 items 6, a, d).
    'I want to know if my HRV dropped from last week to this week',
    "I'd like to know whether my HRV improved from June to July",
    'keen to know if my HRV dropped from last week to this week',
    'can you tell me if my sleep score got worse this week than last week?',
    'has my HRV gone up or down from last month to this month?',
])
def test_the_frame_of_a_factual_comparative_reaches_the_parser(monkeypatch, selector, text):
    assert not unexplained_sentence(request(text)), text
    assert outcome(monkeypatch, selector, text) == ('planned', None, 1), text


# Verifier final round.
@pytest.mark.parametrize('text', [
    # A change is a comparison of two windows only when an explicit window follows it: a manner, judgement or implicit-reference word in
    # the change ("really", "finally", "as expected"), or no window after it, leaves a verdict (10g.38 hand-off list, item 6).
    'Show my HRV this week and last week. Did it really improve?',
    'Show my HRV this week. Did it really improve from last week?',
    'Show my weight this month and last month. Has it finally gone down?',
    'Show my weight this month. Has it finally gone down compared to last month?',
    'Show my steps last week and this week. Did it improve as expected?',
    'Show my steps this week. Did it improve as expected compared to last week?',
    'Show my HRV this week. Has it actually gone up compared to last week?',
    'Show my HRV this week. Did it slightly improve from last week?',
    'Show my HRV this week and last week. Has it gone up?',
    'Show my HRV this week and last week. Is it up?',
    'Show my HRV this week and last week. How much did it change?',
])
def test_a_change_with_a_free_word_or_no_window_after_it_hands_off(monkeypatch, selector, text):
    assert unexplained_sentence(request(text)), text
    for queries in (ONE, TWO):
        assert outcome(monkeypatch, selector, text, queries) == ('handoff', 'unbound_request_constraint', 0), text


@pytest.mark.parametrize('text', [
    # A factual comparative is between readings of a metric the sentence names: one made only of frame words is a self-state or symptom
    # comparison with nothing to read, and stays class 5 (or 2).
    'Am I more tired this week than last week?',
    'Do I feel worse this week vs last week?',
    'Have I got worse this week than last week?',
    'Have I gotten better this week compared to last week?',
    'Am I getting worse this week vs last week?',
    'I wonder if I got better from last week to this week',
    'Have my readings got worse this week than last week?',
    'Have my levels got better this week than last week?',
])
def test_a_comparative_without_a_metric_hands_off(monkeypatch, selector, text):
    assert unexplained_sentence(request(text)), text
    for queries in (ONE, TWO):
        assert outcome(monkeypatch, selector, text, queries) == ('handoff', 'unbound_request_constraint', 0), text


@pytest.mark.parametrize('text,history', [
    # A correction replaces the read's window: one read of the new window, never an added second window.
    ('Show my steps last week. No, the week before.', ()),
    ('Show my steps last week. Sorry, I mean the week before.', ()),
    ('Show my steps last week. Sorry, the week before.', ()),
    ('Show my steps last week. Switch to the month before.', ()),
    ('Show my steps last week. Rather the week before, please.', ()),
    ('Show my steps last week. Actually, the week before.', ()),
    ('Show my steps last week. The week before instead.', ()),
    ('Show my steps last week. No, last month.', ()),
    ('Show my steps this week instead of last week.', ()),
    ('No, the week before.', ('Show my steps last week',)),
    ('Sorry, I mean last month.', ('Show my steps last week',)),
])
def test_a_correction_never_adds_a_second_window(monkeypatch, selector, text, history):
    assert outcome(monkeypatch, selector, text, TWO, history) == ('handoff', 'unbound_request_constraint', 1), text
    assert all(query_selector.continuation(s, r, request(text, history)) != 'windows' for s, r in query_selector.sentence_roles(request(text, history)))


@pytest.mark.parametrize('text', [
    # Look-alikes of a correction keep a two-window plan.
    'No need for a chart. Compare my steps this week with last week.',
    'Compare my steps this week with last week, no rush.',
    'Actually, compare my steps this week with last week.',
    "I've been taking the stairs instead of the lift. Compare my steps this week with last week.",
])
def test_a_correction_look_alike_keeps_the_two_window_plan(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text, TWO) == ('planned', None, 1), text


@pytest.mark.parametrize('text', [
    # Class 7: a write joined to the read by "and" is a request, with or without a style word.
    'Show my HRV this week in a table next to last week and email it to me.',
    'Show my HRV this week next to last week and email it to me.',
    'Show my steps this week and last week in a table and send it to my coach.',
    'Compare my steps this week with last week and export them as a PDF.',
    'Show my HRV this week next to last week and save it to my files.',
    'Show my steps this week and last week in a table and delete last week afterwards.',
])
def test_a_write_joined_by_and_hands_off(monkeypatch, selector, text):
    assert unexplained_sentence(request(text)), text
    for queries in (ONE, TWO):
        assert outcome(monkeypatch, selector, text, queries) == ('handoff', 'unbound_request_constraint', 0), text


@pytest.mark.parametrize('text', ['Show my pre and post workout heart rate last week.', 'Show my steps last week and the week before and add them up.'])
def test_a_joined_verb_without_an_object_or_a_sum_is_not_an_action(text):
    assert not unexplained_sentence(request(text)), text
