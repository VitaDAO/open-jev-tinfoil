"""Missed, stopped or changed medication always hands off, even beside a plain read (spec v1.8 10g.36.4, §2.4 class 1, v14-0857).
Supplements and vitamins are not medication (plain context); a supplement a clinician prescribed is medical context (class 2)."""
import pytest

import query_selector
from query_plan import QueryRequest
from query_selector import acute_or_crisis, unexplained_sentence
from test_p0_rule_fixes import selector  # noqa: F401  (fixture)


def request(text, history=()):
    return QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': list(history), 'reference_date': '2026-09-26', 'time_zone': 'Europe/Bucharest'},
        'reference_time': '2026-09-25T22:41:32Z', 'available_metrics': ['resting_heart_rate', 'heart_rate_variability', 'blood_pressure',
        'systolic_blood_pressure', 'vitamin_d', 'ldl', 'fasting_insulin', 'steps', 'total_sleep', 'apob'],
        'available_record_types': ['profile', 'labs', 'workouts', 'calendar'], 'available_sources': ['oura', 'whoop'], 'literature_available': True})


class StubParser:
    def __init__(self): self.calls = 0
    def select(self, req):
        self.calls += 1
        return {'status': 'planned', 'queries': [{'kind': 'health', 'metrics': ['apob'], 'records': [], 'operation': 'latest',
                                                  'period': {'kind': 'all_history'}, 'source': None, 'date_basis': 'observed_at'}]}


CHANGES = [
    "I ran out of my blood pressure meds last week.",                            # v14-0857
    'I ran out of my BP meds', 'skipped my meds yesterday', 'I forgot my dose this morning', 'I doubled my dose last night', "I've been off my statin for a month",
    'went without my insulin for two days', 'I stopped taking my blood thinners', 'I missed my thyroid pills all week', 'forgot to take my metformin',
    'I quit my antidepressants cold turkey', 'I halved my dose of levothyroxine', 'cut my pills in half to make them last', "I'm tapering off my beta blocker",
    'been weaning myself off my sertraline', 'ran out of refills on my inhaler', "I'm out of insulin", 'my prescription ran out on Monday',
    'my refill is overdue', 'my dose was doubled by mistake', 'my medication got changed last week', 'took an extra dose of my blood pressure tablets',
    'I took double my usual pills', 'took two of my tablets instead of one', "haven't taken my heart medication since Sunday", "didn't take my meds today",
    'not taking my cholesterol pills anymore', 'no longer on my blood pressure medication', 'came off the pill in June', 'I switched my medication to a lower dose',
    'increased my dose of ozempic', 'lowered my insulin dose', 'my doctor told me to reduce my dose and I did', 'I decreased my prednisone', 'I paused my statins for a week',
    "can't afford my meds this month", "couldn't get my inhaler refilled", 'I dropped my levothyroxine for a while', 'been skipping my asthma inhaler',
    'ditched the blood pressure pills', 'my tablets ran out so I went without', 'I lapsed on my prescription', 'I am missing doses of my seizure medication',
    'Forgot my tablet again',
]


@pytest.mark.parametrize('text', CHANGES)
def test_a_medication_change_is_acute_content(text):
    assert query_selector.medication_change(text.lower()) and acute_or_crisis(text), text


@pytest.mark.parametrize('side', CHANGES)
def test_a_medication_change_next_to_a_plain_read_hands_off_before_any_parser(monkeypatch, selector, side):
    stub = StubParser()
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: stub)
    for text in (side + " What's my resting heart rate?", "What's my resting heart rate? " + side, "Show my HRV last week, " + side):
        result = selector.select_query(request(text))
        assert result['status'] == 'handoff' and result['reason_codes'] == ['acute_or_crisis_requires_model'] and stub.calls == 0, text


def test_v14_0857_hands_off(selector):
    result = selector.select_query(request("I ran out of my blood pressure meds last week. What's my resting heart rate?"))
    assert result['status'] == 'handoff' and result['reason_codes'] == ['acute_or_crisis_requires_model']


@pytest.mark.parametrize('text', [
    # Supplements, vitamins and devices are not medication.
    'I took all my supplements this morning. Vitamin D level?', 'I ran out of my vitamin D pills', 'forgot my magnesium tablets', 'I stopped taking creatine',
    'skipped my fish oil this week', 'I missed my multivitamin', 'ran out of protein powder and electrolytes', 'I stopped my zinc and my vitamin C tablets',
    'I switched to a new tablet for reading my data', 'I dropped my Samsung tablet', 'forgot my iPad tablet at the gym', 'I dropped my phone',
    # Plain reads and context that only look like a change.
    "What's my fasting insulin?", 'skip the trend and show my fasting insulin', 'Show my insulin level, skip the chart', 'list my meds', 'what meds am I on',
    'Which medications are listed in my profile?', 'What does the literature say about statin-associated muscle symptoms?', 'Summarize the literature on metformin for healthy aging.',
    'I double checked my HRV setup. What was my HRV last night?', 'I changed my watch strap. Steps today?', 'I switched phones and lost a day of steps',
    'I lowered the volume on my alarm', 'I stopped by the gym on the way home', 'I missed my train but still hit 10k steps', 'I forgot my charger at the hotel',
    'I cut the grass for an hour this morning', 'Skipped breakfast and lunch today, so mostly coffee', 'I quit my job last month. Steps this week?',
])
def test_supplements_devices_and_look_alikes_are_not_a_medication_change(text):
    assert not query_selector.medication_change(text.lower()), text
    assert not acute_or_crisis(text), text


@pytest.mark.parametrize('text', [
    # A supplement a clinician prescribed or directed is class 2.
    "My doctor put me on 50,000 IU vitamin D. What's my vitamin D level?", 'I was prescribed vitamin D. What is my vitamin D level?',
    'My GP had me start magnesium last month. What was my HRV last week?', 'The vitamin D my cardiologist recommended is working, I think. What was my vitamin D level?',
    'My dietitian prescribed fish oil for my triglycerides. What is my LDL?', "I'm on prescription vitamin D. What's my vitamin D level?",
    "Vitamin D as directed by my doctor. What's my latest vitamin D?", 'My nurse practitioner told me to take iron. What is my HRV this week?',
])
def test_a_prescribed_or_clinician_directed_supplement_is_medical_context(text):
    assert unexplained_sentence(request(text)), text


@pytest.mark.parametrize('text', ['I took all my supplements this morning. Vitamin D level?', "I've been taking creatine and fish oil. What's my HRV last week?"])
def test_plain_supplement_statements_stay_as_before(text):
    # The 'care' list (supplements, vitamins) already hands a supplement statement in a side sentence off; the medication rule does not add to it.
    assert not query_selector.medication_change(text.lower()) and not acute_or_crisis(text)
