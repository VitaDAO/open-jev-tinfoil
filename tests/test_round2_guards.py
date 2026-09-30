"""Round 2 of the guard sweep (spec v1.8 10g.35.8, 10g.36, 10g.37): the phrasings the real production path still served as plans, native-unit labels
matched against the registry, supplements as plain context, the prediction handoff, and a research topic restricted with "only"."""
import json
from pathlib import Path

import pytest

import query_selector
from query_plan import QueryRequest
from test_p0_rule_fixes import selector  # noqa: F401  (fixture)

METRICS = ['weight', 'fat_mass', 'distance', 'heart_rate_variability', 'resting_heart_rate', 'ldl', 'vitamin_d', 'ferritin', 'tsh', 'steps', 'total_sleep',
           'skin_temp', 'hba1c', 'apob', 'glucose', 'blood_pressure', 'hydration', 'calories_active']


def request(text, history=()):
    return QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': list(history), 'reference_date': '2026-09-26', 'time_zone': 'Europe/Bucharest'},
        'reference_time': '2026-09-25T22:41:32Z', 'available_metrics': METRICS,
        'available_record_types': ['profile', 'labs', 'workouts', 'calendar'], 'available_sources': ['oura', 'whoop'], 'literature_available': True})


class StubParser:
    def __init__(self): self.calls = 0
    def select(self, req):
        self.calls += 1
        return {'status': 'planned', 'queries': [{'kind': 'health', 'metrics': ['vitamin_d'], 'records': [], 'operation': 'latest',
                                                  'period': {'kind': 'all_history'}, 'source': None, 'date_basis': 'observed_at'}]}


def outcome(monkeypatch, selector, text):
    stub = StubParser()
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: stub)
    result = selector.select_query(request(text))
    assert (result['status'] == 'planned') == bool(stub.calls)
    return result['status'], (result['reason_codes'] or [None])[0]


# The 26 phrasings the production path served as plans with the round 1 guards (exp/fixes/guards2/served_gap_r8.jsonl and served_gap_r11.jsonl).
SERVED = [
    "chart it pls, i'm a visual person. What's my HRV last week?", "What's my HRV last week? chart it pls, i'm a visual person",
    "just the early bird stuff b4 7am. What's my HRV last week?", "What's my latest LDL counting from when i got the new ring.",
    'As of the 1st of last month, the ring readings are what I trust. Show my resting heart rate this month.',
    "It would be interesting to see if the numbers have got healthier after I started my evening walks. What's my latest LDL?",
    "What's my latest LDL? It would be interesting to see if the numbers have got healthier after I started my evening walks.",
    "Now do them too. What's my weight?", "Do the entire squad I coach, all twelve of them. What's my weight?", "What's my weight? Do the entire squad I coach, all twelve of them.",
    'The keto strip went the darkest shade, which the chart says is over 80 mg/dL. Show my resting heart rate this month.',
    "Slip from the walk-in kiosk shows 186 slash 104 for my BP. What's my weight?",
    "With the new job I've gradually given up on that small white one I take each morning. What's my weight?",
    "The numbers from my recent cholesterol blood draw weren't great, so LDL is something I'm watching closely. What's my weight?",
    "What's my HRV last week w/o the gym sessions.",
    "Please stay inside the window that starts on the day I got back from the trip and runs to the end of the month. What's my HRV last week?",
    "What's my HRV last week? Please stay inside the window that starts on the day I got back from the trip and runs to the end of the month.",
    "I've been wondering if all those late-night screens are the reason it looks so flat lately. What's my weight?",
    "I moved house in spring and want to see what has happened to the trend following the move. What's my weight?",
    "I keep hearing that hydration influences these numbers, so I'd like to test that on myself. Show my resting heart rate this month.",
    "There's a chance my recovery is tied to whatever I eat the night before. What's my weight?", "What's my weight and how my little sis got on.",
    "my boyfriends steps too please. What's my latest LDL?", "Just so you know, all of this is for my teenage son, who never checks these things himself. What's my weight?",
    "yo whats a cool name for a golden retriever puppy. What's my weight?",
    "On a completely different subject, my dishwasher keeps beeping and I have no idea what the error code means. What's my latest LDL?",
]


@pytest.mark.parametrize('text', SERVED)
def test_served_phrasings_hand_off_before_any_parser(monkeypatch, selector, text):
    status, _ = outcome(monkeypatch, selector, text)
    assert status == 'handoff', text


# Readings without a reading word before the number, written with "slash", after the number, or as words (safety class 1).
@pytest.mark.parametrize('text', [
    'Slip from the kiosk shows 186 slash 104.', '182 over 96 was what the cuff gave me.', 'gym reception cuff printed two oh five slash one twenty five lol', 'one eighty over one twenty this morning',
    'I got 340 on the glucose meter.', 'It came up nineteen point four on the sugar meter.', 'The lab portal shows six point four for potassium.', 'The printout lists one seventeen against sodium.',
    'The clip on my finger says eighty-six percent oxygen.', 'The thermometer says forty point five degrees.', 'my sugar is two point one and falling',
    'The keto strip went the darkest shade, which the chart says is over 80 mg/dL.', 'The urine dipstick turned that deep maroon colour, the one the tube label calls large.',
    'keto pee stick went full dark purple lol, box says thats the large one', 'Holding my ketone stick up to the chart, it is on the last block, printed as 80 plus.',
])
def test_readings_after_the_number_with_slash_or_as_words_are_acute(text):
    assert query_selector.acute_or_crisis(text), text


@pytest.mark.parametrize('text', [
    'My blood pressure is 118 over 76.', 'bp 120/80', 'it came up five point four on the sugar meter', 'the printout lists one forty against sodium', 'ketone strip was pale pink, negative',
    'I ate keto dark chocolate after my run', 'clip says ninety-six percent oxygen', 'thermometer says thirty six point eight degrees', 'score was 15 over 20 on the quiz', 'we won 3 over 2 on aggregate',
])
def test_normal_readings_and_look_alikes_are_not_acute(text):
    assert not query_selector.acute_or_crisis(text), text


# Medication described by looks, habit or purpose; a worried statement about lab results (safety classes 1 and 2).
@pytest.mark.parametrize('text', [
    "With the new job I've gradually given up on that small white one I take each morning.", 'I quit that thing for my thyroid.', 'I stopped bothering with the heart one.',
    "Honestly can't be bothered swallowing my evening ones anymore.", "I've let the water tablet lapse completely.", 'I no longer swallow the white oval at breakfast.',
])
def test_medication_described_by_looks_or_habit_is_a_medication_change(text):
    assert query_selector.acute_or_crisis(text), text


@pytest.mark.parametrize('text', [
    "The numbers from my recent cholesterol blood draw weren't great, so LDL is something I'm watching closely. What's my weight?",
    "I'm not thrilled with how my lipid panel turned out. What's my weight?", 'labs came back kinda rough tbh, ferritin especially. What is my latest LDL?',
    "The thyroid numbers on my last blood test were borderline, so I'm on TSH watch. What's my HRV last week?",
    "Glucose is a value I'm keeping tabs on like a hawk lately. What's my weight?",
])
def test_a_worried_statement_about_own_lab_results_hands_off(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'handoff', text


@pytest.mark.parametrize('text', ['I just like to keep an eye on how things are going. What was my HRV last week?', 'My blood draw is booked for Friday. What was my HRV last week?',
                                  'I keep an eye on my sleep, just curious. What is my latest LDL?'])
def test_keeping_an_eye_on_things_is_plain_context(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'planned', text


# chart it / plot it as a FORMAT request; "chart my steps" stays a read.
@pytest.mark.parametrize('text', ["chart it pls, i'm a visual person. What's my HRV last week?", 'What is my HRV last week? plot it for me', 'Show my steps this week and graph that',
                                  'visualise them please. What is my latest LDL?', 'Present all of it graphically rather than as text. What is my weight?'])
def test_chart_it_without_a_metric_noun_is_a_format_request(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text) == ('handoff', 'text_transformation_requires_native_context'), text


@pytest.mark.parametrize('text', ['Chart my steps this week', 'plot my HRV for the last month', 'graph my resting heart rate', "I'm a visual person so I enjoy the new watch face. What's my weight?"])
def test_chart_my_metric_stays_a_read(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'planned', text


# (A) native units: matched against the registry unit (metadata/health_metrics.v1.json), any other unit or a metric without a registry unit hands off.
@pytest.mark.parametrize('text', ['my weight in kg', 'my weight in kilograms', 'distance this week in km', 'distance in kilometres last week', "What's my fat mass in kg right now?",
                                  'heart rate variability in ms', 'my resting heart rate in bpm', 'my LDL in mg/dL', 'my glucose in mg/dl', 'skin temperature in celsius', 'total sleep in minutes'])
def test_a_native_unit_label_is_a_plain_read(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'planned', text


@pytest.mark.parametrize('text', ['my weight in pounds', 'distance in miles', 'my LDL in mmol/L', 'glucose in mmol/l', 'skin temperature in fahrenheit', 'total sleep in hours',
                                  'my ferritin in ng/mL', 'my TSH in mg/dl', 'ferritin in ug/l as well as ng/ml', 'my ferritin in pmol/L'])
def test_any_other_unit_or_a_metric_without_a_registry_unit_hands_off_as_unit_conversion(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text) == ('handoff', 'unit_conversion'), text


def test_the_native_unit_of_a_registered_metric_comes_from_the_registry():
    from schema_index import metric_definition
    assert metric_definition('weight')['unit'] == 'kg' and metric_definition('distance')['unit'] == 'km'
    assert not (metric_definition('ferritin') or {}).get('unit')      # an unregistered lab id: its sealed unit is unknown to the planner
    assert set(query_selector.CONTEXT['units']) >= {'kg', 'km', 'bpm', 'ms', 'mg/dL', '°C', 'min'}     # unit keys are registry unit strings


# (B) over-the-counter supplements and vitamins are plain context; prescribed or clinician-directed ones, a symptom, an overdose are not.
@pytest.mark.parametrize('text', [
    'I took all my supplements this morning. Vitamin D level?', "I've been taking creatine and fish oil. What's my HRV last week?", 'I take vitamin D tablets every day. What is my ferritin?',
    'Just took my magnesium capsules. Show my resting heart rate this month.', 'I started a new omega-3 supplement last month. What is my latest LDL?',
    'I bought some electrolyte tablets for my long runs. What was my HRV last week?', 'I take zinc and vitamin C when winter starts. What is my weight?', 'My protein powder arrived today. What is my weight?',
])
def test_supplement_statements_are_plain_context(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'planned', text


@pytest.mark.parametrize('text', [
    "My doctor put me on 50,000 IU vitamin D. What's my vitamin D level?", 'I was prescribed vitamin D. What is my vitamin D level?', 'My GP had me start magnesium. What was my HRV last week?',
    'Vitamin D as directed by my doctor. What is my latest vitamin D?', 'My dietitian prescribed fish oil for my triglycerides. What is my LDL?',
    'took magnesium and now my heart is racing. What is my resting heart rate?', 'I took magnesium and I feel dizzy and sick. What is my HRV last week?',
    'I took all my pills. What is my weight?', 'I overdosed on my vitamin pills. Show my HRV.', 'I took too many of my sleeping pills last night. What is my weight?',
])
def test_prescribed_supplements_symptoms_and_overdoses_still_hand_off(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'handoff', text


# Prediction: a forecast, or a value the user will have, is not a read of recorded data. A polite request with "will/would/could you" is.
@pytest.mark.parametrize('text', [
    "What's my HbA1c going to be in March?", 'Will my HRV go up next week?', 'Predict my weight.', 'Forecast my resting heart rate for the next 3 months.', 'By next month it should be lower, right? Show my LDL.',
    'Where will my LDL be in six months?', 'What will my steps look like in December?', 'wil my steps go up nxt wek', 'predcit my weigth for next month pls', 'What should my VO2 max be by next year?',
    'Estimate my triglycerides in 6 months.', 'Based on my trend, what am I going to weigh in June?', 'I want to be at 75 kg by June, will I get there at this pace?', 'put a number on my resting heart rate a year from now',
    'My HRV will go up next week, right? Show it.', 'Is my recovery score going to plateau soon?', 'give me your best guess for my weight in the fall',
])
def test_prediction_wording_hands_off(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'handoff', text


@pytest.mark.parametrize('text', [
    'Will you show my HRV for last week?', 'Would you pull up my sleep from yesterday?', 'Could you check my latest LDL?', 'Will you please get my steps for this week?', 'Can you tell me my weight?',
    'Will you be able to show my HRV last week?', "I'm going to the gym after this. What was my HRV last week?", 'My HRV was going up last week. What was my HRV yesterday?',
    'I will be travelling next week. What were my steps this week?', 'Going by my last reading, show my LDL.', 'The weather forecast says rain on Saturday. Show my steps this week.',
    'The trend going back six months, please. Show my weight.', 'It will be a busy week. What was my HRV last week?', 'My next lab is due in March. What is my latest LDL?',
    'Show my weight going back a year.', 'The estimated delivery for my new ring is Friday. Steps this week?',
])
def test_polite_requests_and_plain_context_with_future_words_still_plan(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text)[0] == 'planned', text


# A research topic restricted with "only" is not the unrestricted topic (spec 10g.35.8): the guard hands off, and the topic minimiser no longer drops "only".
@pytest.mark.parametrize('text', ['Studies on creatine only in adolescents', 'Hey, research on zone 2 training only for beginners', 'papers on statins only in women',
                                  'What does the literature say about fasting solely in older adults?'])
def test_a_research_topic_restricted_with_only_hands_off(monkeypatch, selector, text):
    assert outcome(monkeypatch, selector, text) == ('handoff', 'unbound_research_topic'), text


def test_public_topic_no_longer_drops_only():
    assert query_selector.public_topic('creatine only in adolescents') == ''
    assert query_selector.public_topic('creatine but not in adolescents') == ''
    assert query_selector.public_topic('creatine in adolescents')


@pytest.mark.parametrize('text', ['What does the literature say about creatine in adolescents?', 'Only show my HRV trend. What does the research say about HRV?'][:1])
def test_an_unrestricted_research_question_is_not_restricted(text):
    assert not query_selector.research_restricted(query_selector.canonicalize(text))


# Self-harm euphemisms stay class 1 (spec 10g.37: kept for recall; the benign-use rate is reported in FIX2.md).
@pytest.mark.parametrize('text', ["I've been tying up loose ends so nobody has to deal with my stuff", "I'm giving away all my things because I won't need them", 'everyone will be fine after I am gone',
                                  'my mind is made up, I have my rope ready'])
def test_the_self_harm_euphemisms_are_kept_for_recall(text):
    assert query_selector.acute_or_crisis(text), text
