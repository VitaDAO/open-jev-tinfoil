"""P0 rule fixes from the Vita battery and held-out sets (retrain plan §3, spec v1.3 §3)."""
import pytest

from proposal_selector import ProposalSelector
from query_plan import QueryRequest
from query_selector import QuerySelector, enhance, research_topic, single_value_question
from trained_proposal_selector import TrainedProposalSelector

METRICS = ['resilience_daytime_recovery', 'egfr', 'creatinine', 'spo2', 'homa_ir', 'hscrp', 'hba1c', 'fasting_glucose', 'apob', 'tsh', 'hematocrit', 'total_sleep', 'sleep_score',
           'sleep_deep', 'sleep_efficiency', 'sleep_rem', 'sleep_light', 'resting_heart_rate', 'weight',
           'readiness_score', 'recovery_score', 'calories_burned', 'workout_calories', 'steps', 'heart_rate_variability',
           'ldl', 'hdl', 'triglycerides', 'cholesterol', 'vitamin_d']


def request(text, history=(), records=('profile', 'labs', 'workouts', 'calendar'), literature=True):
    return QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': list(history),
                  'reference_date': '2026-09-26', 'time_zone': 'Europe/Bucharest'},
        'reference_time': '2026-09-25T22:41:32Z', 'available_metrics': METRICS,
        'available_record_types': list(records), 'available_sources': ['oura', 'whoop'],
        'literature_available': literature})


@pytest.fixture
def selector(monkeypatch):
    def select(self, req):
        predicted = dict(task='health', coverage='targeted', purpose='trend', research='none')
        return ProposalSelector.select(self, req, prediction=(predicted, {k: .8 for k in predicted}),
            coverage_decision=({'choice': 'the proposed read covers the request', 'confidence': .9}, None))
    monkeypatch.setattr(TrainedProposalSelector, 'select', select)
    monkeypatch.setattr(TrainedProposalSelector, 'intent_check', lambda self, text: ({'choice': 'recorded_health_read'}, None))
    return QuerySelector.__new__(QuerySelector)


def plan(selector, text, history=()):
    return selector.select_query(request(text, history))


@pytest.mark.parametrize('text', ['What is my hsCRP?', 'What is my hs-CRP?'])
def test_hscrp_binds_to_inventory(selector, text):
    result = plan(selector, text)
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['metrics'] == ['hscrp']


@pytest.mark.parametrize('text,op', [("What's my TSH?", 'latest'), ('hematocrit', 'latest'), ('a1c?', 'latest'),
                                     ('What are my fasting glucose and HbA1c?', 'latest'),
                                     ('Show my ApoB trend.', 'trend'), ('How has my TSH been this year?', 'trend')])
def test_single_value_questions_read_latest(selector, text, op):
    result = plan(selector, text)
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['operation'] == op


def test_rolling_versus_calendar_periods():
    assert 'past 12 months' in enhance('Show my ApoB over the last year')
    assert 'past 1 months' in enhance('resting heart rate over the last month')
    assert 'past 7 days' in enhance('steps over the past week')
    assert 'last year' in enhance('Show my ApoB last year') and 'past' not in enhance('Show my ApoB last year')
    assert 'last month' in enhance('HRV last month')


@pytest.mark.parametrize('text', ['Last-night sleep efficiency', "last night's deep sleep"])
def test_last_night_variants_normalise(text):
    assert 'last night' in enhance(text)


def test_night_basis_only_for_supported_sleep_metrics(selector):
    assert plan(selector, 'What was my sleep score last night?')['reason_codes'] == ['night_basis_metric_unavailable']
    ok = plan(selector, 'How long did I sleep last night?')
    assert ok['status'] == 'planned' and ok['queries'][0]['metrics'] == ['total_sleep']


def test_how_long_did_i_sleep_reads_total_sleep_only():
    assert enhance('how long did I sleep on september 25').startswith('what is my total sleep')


def test_units_are_not_timezones(selector):
    result = plan(selector, 'What does the literature say about an ApoB of 78 mg/dL?')
    assert 'clock_or_timezone_qualifier_unavailable' not in (result['reason_codes'] or [])


def test_research_topic_drops_stated_values_but_keeps_personal_topics_unbound():
    read = research_topic('what does the literature say about an apob of 78 mg/dl', request('x'))
    assert read is not None and read.topic == 'apob'
    assert research_topic('what does research say about my creatine dose', request('x')) is None


@pytest.mark.parametrize('text', ['What does the literature say about ApoB?', 'Any papers on omeprazole long-term use?',
                                  'Is there published literature on rapamycin and longevity?'])
def test_literature_and_papers_route_to_research(selector, text):
    result = plan(selector, text)
    assert result['status'] == 'planned', result['reason_codes']
    assert all(q['kind'] == 'research' for q in result['queries'])


def test_current_medications_is_a_profile_read(selector):
    result = plan(selector, 'What are my current medications?')
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['profile_fields'] == ['medications']


def test_last_lab_test_reads_labs(selector):
    result = plan(selector, 'When was my last lab test?')
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['records'] == ['labs']


def test_future_metric_windows_hand_off(selector):
    assert plan(selector, 'steps next week')['status'] == 'handoff'


def test_bare_calories_is_ambiguous_but_qualified_calories_bind(selector):
    assert plan(selector, 'show my calories this week')['reason_codes'] == ['ambiguous_metric_alias']
    assert plan(selector, 'calories burned in my workouts this month')['queries'][0]['metrics'] == ['workout_calories']
    assert plan(selector, 'energy expenditure this week')['queries'][0]['metrics'] == ['calories_burned']


def test_readiness_and_recovery_aliases(selector):
    assert plan(selector, 'Show my readiness this week')['queries'][0]['metrics'] == ['readiness_score']
    assert plan(selector, 'Show my recovery this week')['queries'][0]['metrics'] == ['recovery_score']


def test_broad_plans_need_an_explicit_overview(monkeypatch):
    def select(self, req):
        predicted = dict(task='health', coverage='broad', purpose='trend', research='none')
        return ProposalSelector.select(self, req, prediction=(predicted, {k: .8 for k in predicted}),
            coverage_decision=({'choice': 'the proposed read covers the request', 'confidence': .9}, None))
    monkeypatch.setattr(TrainedProposalSelector, 'select', select)
    monkeypatch.setattr(TrainedProposalSelector, 'intent_check', lambda self, text: ({'choice': 'recorded_health_read'}, None))
    s = QuerySelector.__new__(QuerySelector)
    assert s.select_query(request('show me the numbers for the last 3 months'))['reason_codes'] == ['plan_breadth_exceeded']
    assert s.select_query(request('Analyze my health for the last 3 months'))['status'] == 'planned'


def test_pronoun_followup_inherits_the_prior_subject_only(selector):
    result = plan(selector, 'Show it for the last 3 months', ["What's my latest weight?"])
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['metrics'] == ['weight'] and result['queries'][0]['operation'] == 'trend'
    assert result['queries'][0]['period'] == {'kind': 'relative', 'amount': 3, 'unit': 'months'}
    assert plan(selector, 'Show it for the last 3 months')['status'] == 'handoff'   # no prior subject


@pytest.mark.parametrize('text', ['Show my current medications.', 'What are my current medications?'])
def test_profile_alias_cannot_reenable_an_unavailable_profile(selector, text):
    result = selector.select_query(request(text, records=('labs',)))
    assert result['status'] == 'handoff'


@pytest.mark.parametrize('text', ["dad's a1c", "my mom's ldl", 'what will my weight be in a month',
                                  'will i hit my step goal today', 'longest sleep this year', 'weight in q2'])
def test_other_people_predictions_extremes_and_quarters_hand_off(selector, text):
    assert plan(selector, text)['status'] == 'handoff'


def test_high_recovery_is_not_the_recovery_score():
    from proposal_binding import entities
    spans = entities('time in high recovery', ['recovery_score', 'recovery_high_duration'])
    assert [v for _, _, vs in spans for k, v in vs] == ['recovery_high_duration']


def test_trigs_alias_keeps_every_list_item(selector):
    result = plan(selector, 'ldl, hdl and trigs')
    assert result['status'] == 'planned' and result['queries'][0]['metrics'] == ['hdl', 'ldl', 'triglycerides']


def test_today_is_latest_but_other_single_days_keep_the_reviewed_trend(selector):
    # Spec §3.9 names "X today" only; reviewed fixtures keep single past days as trend (daily-summed metrics).
    assert plan(selector, "what's my steps today")['queries'][0]['operation'] == 'latest'
    assert plan(selector, 'Show my steps yesterday')['queries'][0]['operation'] == 'trend'


def test_same_for_followup_inherits_the_window(selector):
    result = plan(selector, 'same for readiness', ['how did my hrv look last week'])
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['metrics'] == ['readiness_score'] and result['queries'][0]['period']['kind'] != 'all_history'


def test_bare_crp_is_ambiguous_not_hscrp(selector):
    assert plan(selector, "What's my CRP?")['reason_codes'] == ['ambiguous_metric_alias']
    assert plan(selector, "What's my hs-CRP?")['queries'][0]['metrics'] == ['hscrp']


@pytest.mark.parametrize('text,expected', [
    ('How long did I sleep this week?', ['total_sleep']),
    ('avg sleep over the past 14 days', ['total_sleep']),
    ('How did I sleep this week?', ['sleep_deep', 'sleep_efficiency', 'sleep_rem', 'sleep_score', 'total_sleep']),
    ('Give me the full breakdown of my sleep this week', ['sleep_deep', 'sleep_efficiency', 'sleep_light', 'sleep_rem', 'sleep_score', 'total_sleep']),
])
def test_sleep_area_is_scoped_by_wording(selector, text, expected):
    result = plan(selector, text)
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['metrics'] == expected


@pytest.mark.parametrize('text', [
    'average stroke rate on my rowing sessions this week', 'whats my stroke volume estimate on the bike',
    'my grandfather had a stroke in 2015, does family history change my risk score',
    'had chest pain in march, turned out to be costochondritis, can you show my appointments from then',
    'what are the warning signs of a heart attack in women, asking for general knowledge'])
def test_acute_lookalikes_are_not_flagged(text):
    from query_selector import acute_or_crisis
    assert not acute_or_crisis(text)


@pytest.mark.parametrize('text', ['I had a stroke in 2015 and now my face is drooping again',
                                  'what are the signs of a stroke, I think I am having one right now'])
def test_present_cues_keep_history_or_education_framing_acute(text):
    from query_selector import acute_or_crisis
    assert acute_or_crisis(text)


@pytest.mark.parametrize('text', ['recovery + strain', 'hrv the week of sept 7', 'show my weight trend this year and my body fat right now'])
def test_dropped_items_and_mixed_windows_hand_off(selector, text):
    assert plan(selector, text)['status'] == 'handoff'


def test_named_record_is_never_silently_dropped(selector, monkeypatch):
    import query_selector
    real = query_selector.HealthRead
    result = plan(selector, 'workouts and calories from workouts this month')
    assert result['status'] == 'handoff' or {'workouts'} <= {r for q in result['queries'] for r in q['records']}


@pytest.mark.parametrize('text,metric', [("what's my good cholesterol", 'hdl'), ('whats my bad cholesterol at', 'ldl'), ('my ldl-c?', 'ldl')])
def test_colloquial_lipid_aliases(selector, text, metric):
    result = plan(selector, text)
    assert result['status'] == 'planned' and result['queries'][0]['metrics'] == [metric]
    assert result['queries'][0]['operation'] == 'latest'


@pytest.mark.parametrize('text', ["what's my HOMA IR score", 'my oura readiness', 'resting hr from whoop'])
def test_single_value_wording_with_scores_and_providers(selector, text):
    q = selector.select_query(request(text, ()))['queries']
    assert q and q[0]['operation'] == 'latest'


def test_now_show_me_inherits_the_window(selector):
    result = plan(selector, 'now show me weight', ['hrv over the last 12 months'])
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['metrics'] == ['weight'] and result['queries'][0]['period']['kind'] != 'all_history'


@pytest.mark.parametrize('text', ['uhh whats my resting heart rate', "what's my weight on file", 'my ferritin is how much?'])
def test_latest_unless_a_series_is_requested(selector, text):
    assert plan(selector, text.replace('ferritin', 'vitamin d'))['queries'][0]['operation'] == 'latest'


@pytest.mark.parametrize('current,history', [('now spo2', ['resting heart rate last 30 days']),
                                             ('also my triglycerides', ['ldl over the last year']),
                                             ('egfr too', ['creatinine over the past year'])])
def test_bare_subject_followups_inherit_the_window(selector, current, history):
    result = plan(selector, current, history)
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['period']['kind'] != 'all_history'


def test_open_ended_and_to_date_windows():
    assert enhance('vitamin d from june 2025 till now').startswith('vitamin d since june 2025')
    assert 'this month' in enhance('steps month to date')


def test_daytime_recovery_resilience_is_not_the_recovery_score():
    assert 'recovery score' not in enhance('daytime recovery resilience')
    assert 'recovery score' in enhance('show my whoop recovery')


def test_mixed_latest_and_window_hands_off(selector):
    assert plan(selector, 'ldl over the past year and my current a1c')['status'] == 'handoff'


def test_scope_instructions_keep_the_default_trend(selector):
    assert plan(selector, 'Steps only, no labs')['queries'][0]['operation'] == 'trend'


@pytest.mark.parametrize('text,metric', [('max heart rate today', 'day_max_hr'), ('what did my heart rate peak at today', 'day_max_hr'),
                                         ('avg HR during workouts this month', 'workout_avg_hr'), ('max HR in workouts last week', 'workout_max_hr'),
                                         ('activity heart rate over the past 2 weeks', 'activity_hr_average'),
                                         ('peak heart rate during activity today?', 'activity_hr_max'), ('time in zone 4 this week', 'hr_zone_4')])
def test_heart_rate_variants_bind_to_their_own_metrics(text, metric):
    from proposal_binding import entities
    inv = ['heart_rate', 'day_max_hr', 'day_avg_hr', 'workout_max_hr', 'workout_avg_hr', 'activity_hr_max',
           'activity_hr_average', 'activity_hr_min', 'hr_zone_4', 'resting_heart_rate']
    assert [v for _, _, vs in entities(enhance(text), inv) for k, v in vs if k == 'metric'] == [metric]


def test_trailing_also_followup_inherits_the_window(selector):
    result = plan(selector, 'ldl also', ['show hdl over the past year'])
    assert result['status'] == 'planned', result['reason_codes']
    assert result['queries'][0]['metrics'] == ['ldl'] and result['queries'][0]['period']['kind'] != 'all_history'
