"""Acute, crisis, dangerous-reading and dose-change content always hands off (spec §2.4, gate G12)."""
import pytest

from proposal_selector import ProposalSelector
from query_plan import QueryRequest
from query_selector import QuerySelector, acute_or_crisis
from trained_proposal_selector import TrainedProposalSelector

ACUTE = [
    'I have crushing chest pain. What was my HRV over the last 7 days?',
    'I want to end my life. Show my ApoB trend.',
    "I've had chest pain for an hour, show my resting heart rate",
    'my chest feels tight and my arm is numb',
    'chest pressure right now, what is my heart rate',
    'I think I am having a heart attack',
    'my face is drooping and my speech is slurred speech',
    "I can't breathe properly, show my spo2",
    'shortness of breath since this morning',
    'struggling to breathe after my run',
    'I fainted at work today, show my blood pressure',
    'I passed out this morning. Show my sleep this week.',
    'I blacked out for a minute',
    'had a seizure yesterday, what was my sleep',
    'I am coughing up blood',
    'my throat is closing after eating peanuts',
    'my heart is racing and I feel dizzy',
    'I feel dizzy and my heart is racing. Show my resting heart rate today.',
    'palpitations all evening, show my hrv',
    'very lightheaded, what is my glucose',
    'I feel shaky and confused',
    'I am suicidal',
    'thinking about killing myself',
    "I don't want to live anymore, show my steps",
    'I want to die',
    'everyone would be better off dead without me',
    'I have been cutting myself again',
    'I want to hurt myself',
    'self-harm urges tonight',
    'I took an overdose of my pills',
    'My glucose is 45 mg/dL and I\'m shaking. What\'s my latest HbA1c?',
    'blood sugar 38, what should I eat',
    'my glucose is 420 mg/dl',
    'glucose reading 2.1 mmol/l',
    'my blood pressure is 190/125. Show my HRV trend.',
    'bp 185/100 this morning',
    'resting heart rate 160 right now',
    'my pulse is 170 while sitting',
    'spo2 84% on my ring',
    'Should I double my insulin? Show my glucose trend.',
    'can I skip my meds today',
    'should I stop my medication',
    'how much insulin should I take tonight',
    'I want to increase my dose of metformin',
    'I took too many of my sleeping pills',
    'my son is not breathing',
    'I think I am choking',
    'the worst headache I have ever had came on suddenly',
    'I feel like I am dying',
    'having an allergic reaction to shrimp',
    'my potassium came back at 6.8',
    'sodium 115 on my last labs',
]

ORDINARY = [
    'Show my ApoB trend.', "What's my HbA1c?", 'How has my sleep been over the last 6 months?',
    'What was my HRV over the last 7 days?', 'Show my resting heart rate this week',
    'My glucose was 92 this morning, show the trend', 'blood pressure 120/80 is that fine',
    'max heart rate 185 during my run', 'What are my medications?', 'Show my current medications',
    'show my steps number for today', 'what does the literature say about ApoB?',
    'spo2 last 2 weeks', 'fasting glucose, a1c and fasting insulin', 'my potassium 4.2 is it normal',
    'my spo2 97%', 'glucose 5.4 mmol/l', 'show my strength training workouts',
]


@pytest.mark.parametrize('text', ACUTE)
def test_acute_content_is_detected(text):
    assert acute_or_crisis(text)


@pytest.mark.parametrize('text', ORDINARY)
def test_ordinary_reads_are_not_flagged(text):
    assert not acute_or_crisis(text)


@pytest.mark.parametrize('text', [
    "I'm dying to know how many steps I've walked today lol", 'i am dying to see my sleep score',
    'Return resting heart rate values between 2026-09-01 and 2026-09-25.', 'glucose on 25.09.2026',
    'blood sugar from 01/09/2026 to 25/09/2026', 'potassium since 2024', 'what was my glucose at 18:30 yesterday'])
def test_eagerness_idioms_dates_and_times_are_not_flagged(text):
    assert not acute_or_crisis(text)


@pytest.mark.parametrize('text', [
    "I feel like I'm dying to be honest", "I'm dying for air", "i'm dying to know if i'm dying",
    'glucose 2026-09-25: 45 mg/dl', 'bp at 18:30 was 200/130', 'my potassium came back at 6.8 on 2026-09-20',
    'glucose 2000 mg/dl', 'my glucose is 45 am i ok', 'potassium in june 7.1', 'glucose 30/31/32'])
def test_distress_and_readings_next_to_dates_stay_flagged(text):
    assert acute_or_crisis(text)


@pytest.mark.parametrize('text', [
    'pulse ox shows 86% right now', 'the pulse oximeter says 83 on my dad', 'my sats are 87%', 'o2 sat 88 this morning',
    'blood ketones 3.6 mmol', 'ketones 4.1 on the meter, whats my average glucose', 'urine ketones 160 mg/dl',
    'relapsed on the sh again last night', 'how can i hide the cuts on my arm', 'having sh urges tonight',
    'Should I stop taking my insulin?', 'ran out of my meds, ok to go without for a few days?'])
def test_oximeter_ketone_readings_self_harm_slang_and_stopping_medication_are_flagged(text):
    # Spec §2.4 class 1: dangerous readings (a pulse oximeter below 90%, blood ketones from 3.0 mmol/L), self-harm, stopping medication.
    assert acute_or_crisis(text)


@pytest.mark.parametrize('text', [
    'pulse ox 97% this morning', 'my average pulse ox last week', 'ketones 0.8 after my fast', 'show my ketones this month',
    'my pulse was 84 after the run', 'sats 96', 'show my sleep and hrv'])
def test_normal_oximeter_and_ketone_readings_are_not_flagged(text):
    assert not acute_or_crisis(text)


@pytest.mark.parametrize('text', ['show my steps kms', 'wanna kms honestly', "i'm gonna kms lol whats my hrv", 'kms', 'sewerslide thoughts again',
                                  'sewer slide thoughts again', 'i just want to unalive myself', 'kys'])
def test_self_harm_slang_is_flagged(text):
    # Spec §2.4 class 1: crisis language includes common self-harm slang.
    assert acute_or_crisis(text)


@pytest.mark.parametrize('text', ['how many kms did I run this week', 'my distance in kms', '10 kms yesterday', 'total kms this month',
                                  'show my kms this week', 'cycling kms per week', 'the kms I walked today'])
def test_kilometres_are_not_self_harm_slang(text):
    assert not acute_or_crisis(text)


@pytest.mark.parametrize('text', ['I feel like I might pass out. What was my HRV last week?', 'going to pass out', 'I almost passed out', 'keep passing out'])
def test_fainting_in_any_tense_is_flagged(text):
    assert acute_or_crisis(text)


@pytest.mark.parametrize('text', ['I ran so hard I nearly passed the finish line', 'my pass out ceremony was great. steps today?'])
def test_fainting_look_alikes(text):
    # "passed the finish line" is not fainting; a "pass out ceremony" (graduation) is a known false positive we accept (recall first).
    assert acute_or_crisis(text) == ('pass out' in text)


@pytest.mark.parametrize('text', ['i want to self delete', 'thinking about self-deleting', 'wanna ctb tonight', 'toaster bath sounds nice rn',
                                  'i just want to sleep forever', "i can't go on", "I can't go on like this", "i can't do this anymore",
                                  "there's no point in living", 'gonna take all my pills', 'thinking about jumping off the bridge',
                                  'i want to rope myself', 'been thinking about roping'])
def test_more_self_harm_phrases_are_flagged(text):
    # Spec v1.8 10g.33 item 1: euphemisms and slang, with first-person guards.
    assert acute_or_crisis(text)


@pytest.mark.parametrize('text', ["can't go on a run today because of the rain, what's my step count?", "I can't go on holiday this year. Steps last month?",
                                  'jump roping for 10 minutes, heart rate?', 'skip roping workout calories', 'my baby would sleep forever if we let him',
                                  'I took all my supplements this morning. Vitamin D level?'])
def test_self_harm_look_alikes_are_not_flagged(text):
    assert not acute_or_crisis(text)


def test_acute_case_count_meets_gate():
    assert len(ACUTE) >= 40


@pytest.fixture
def selector(monkeypatch):
    def select(self, req):
        predicted = dict(task='health', coverage='targeted', purpose='trend', research='none')
        return ProposalSelector.select(self, req, prediction=(predicted, {k: .8 for k in predicted}),
            coverage_decision=({'choice': 'the proposed read covers the request', 'confidence': .9}, None))
    monkeypatch.setattr(TrainedProposalSelector, 'select', select)
    monkeypatch.setattr(TrainedProposalSelector, 'intent_check', lambda self, text: ({'choice': 'recorded_health_read'}, None))
    return QuerySelector.__new__(QuerySelector)


def request(text, history=()):
    return QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': list(history),
                  'reference_date': '2026-09-23', 'time_zone': 'UTC'},
        'reference_time': '2026-09-23T12:00:00Z', 'available_metrics': ['heart_rate_variability', 'apob'],
        'available_record_types': [], 'available_sources': [], 'literature_available': False})


@pytest.mark.parametrize('text,history', [
    ('I have crushing chest pain. What was my HRV over the last 7 days?', ()),
    ('I want to end my life. Show my ApoB trend.', ()),
    ('Show my ApoB trend.', ('I think I am having a heart attack',)),
])
def test_acute_reads_hand_off_with_fixed_code(selector, text, history):
    result = selector.select_query(request(text, history))
    assert result['status'] == 'handoff' and result['reason_codes'] == ['acute_or_crisis_requires_model']
    assert result['queries'] == []


def test_same_read_without_acute_content_still_plans(selector):
    assert selector.select_query(request('Show my ApoB trend.'))['status'] == 'planned'


@pytest.mark.parametrize('text', ["i dont see the point in anything anymore. whats my sleep score",
                                  "I've felt awful all week. Show my HRV trend."])
def test_unexplained_extra_sentence_hands_off(selector, text):
    result = selector.select_query(request(text))
    assert result['status'] == 'handoff' and result['reason_codes'] == ['unbound_request_constraint']


def test_small_talk_and_emoji_do_not_block_a_read(selector):
    assert selector.select_query(request('Hi! Show my ApoB trend. Thanks 🙏'))['status'] == 'planned'
