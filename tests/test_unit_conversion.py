"""A value asked for in another unit hands off with its own published code (spec v1.8 10g.36.3, v14-0292). A metric's own unit used as a
label is a plain read, and style, format and language instructions keep the text-transformation code."""
import pytest

import query_selector
from query_plan import QueryRequest
from test_p0_rule_fixes import selector  # noqa: F401  (fixture)

METRICS = ['weight', 'fat_mass', 'distance', 'glucose', 'skin_temp', 'total_sleep', 'heart_rate_variability', 'steps', 'hba1c', 'body_fat',
           'calories_burned', 'ldl', 'apob', 'sleep_efficiency', 'height', 'vitamin_d']


def request(text):
    return QueryRequest.model_validate({'schema_version': 'vita-selector/v2',
        'state': {'current_request': text, 'recent_user_requests': [], 'reference_date': '2026-09-26', 'time_zone': 'Europe/Bucharest'},
        'reference_time': '2026-09-25T22:41:32Z', 'available_metrics': METRICS,
        'available_record_types': ['profile', 'labs', 'workouts', 'calendar'], 'available_sources': ['oura', 'whoop'], 'literature_available': True})


class StubParser:
    def __init__(self): self.calls = 0
    def select(self, req):
        self.calls += 1
        return {'status': 'planned', 'queries': [{'kind': 'health', 'metrics': ['apob'], 'records': [], 'operation': 'latest',
                                                  'period': {'kind': 'all_history'}, 'source': None, 'date_basis': 'observed_at'}]}


CONVERSIONS = [
    'Give me my weight in pounds.',                                             # v14-0292
    "What's my weight in lbs?", 'my weight in stone please', 'Show my fat mass in pounds.',
    'How much do I weigh in ounces?', 'my height in feet and inches', 'height in cm',
    'distance this week in miles', 'How many miles did I walk yesterday?', 'my steps in miles for last week', 'how far did I go, in kilometres',
    'distance in metres last week', 'what was my distance in feet',
    'skin temperature in Fahrenheit', 'my skin temp in degrees F', 'skin temp in kelvin', 'my temperature in °F',
    'Convert my glucose to mmol/L.', 'my glucose in mmol/l', 'glucose in mmol per liter', 'LDL in mmol/L', 'my ApoB in nmol/l', 'convert it to mg/dL',
    'Can you convert that to metric?', 'Show my weight in imperial units', 'give me everything in metric', 'use imperial units for my weight',
    'sleep in hours', 'my total sleep in seconds', 'HRV in seconds', 'calories in kilojoules', 'calories burned in kJ', 'my vitamin D in nmol/L',
    'weight in pounds last month', 'What is my weight, in pounds, right now?', 'my weight instead of kg in pounds', 'weight, lbs instead of kg',
    'Show my weight as pounds', 'unit conversion please: weight to lb', 'a conversion of my weight to pounds',
]


@pytest.mark.parametrize('text', CONVERSIONS)
def test_unit_conversion_requests_hand_off(text):
    assert query_selector.unit_conversion(request(text)), text


@pytest.mark.parametrize('text', ['Give me my weight in pounds.', 'Convert my glucose to mmol/L.', 'distance this week in miles',
                                  'Show my weight in stone. Thanks!', 'Hi! Could you show my weight in lbs?'])
def test_the_learned_path_never_plans_a_conversion(monkeypatch, selector, text):
    stub = StubParser()
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: stub)
    result = selector.select_query(request(text))
    assert result['status'] == 'handoff' and result['reason_codes'] == ['unit_conversion'] and stub.calls == 0


def test_the_rule_path_hands_off_a_conversion_too(selector):
    result = selector.select_query(request('Give me my weight in pounds.'))
    assert result['status'] == 'handoff' and result['reason_codes'] == ['unit_conversion']


@pytest.mark.parametrize('text', [
    'my weight in kg', "What's my fat mass in kg right now?", 'Show my weight in kilograms.', 'distance this week in km', 'how many km did I walk this week',
    'my glucose in mg/dL', 'skin temperature in celsius', 'my total sleep in minutes', 'HRV in ms', 'sleep efficiency in percent', 'my HbA1c as a percentage',
    'calories burned in kcal', 'my height in metres', 'body fat in %', 'my LDL in mg/dl', 'my ApoB in mg/dL',
    # Not asks for another unit: a value stated as context, a measure of time, a quantity, a profile read of the unit system.
    "I ran 12 kms yesterday morning. What's my HRV today?", 'I lost 5 pounds since spring. What was my weight last month?',
    'How many minutes was I in HR zone 3 yesterday?', 'Hey, could you pull my time in HR zone 5 during the past 5 days?',
    'How many hours did I sleep last night?', 'how many minutes did I sleep on Tuesday', 'Which unit system is set in my profile?',
    'What unit system is on my profile?', 'Metric or imperial: what is my profile using?', 'my HbA1c in the second quarter',
    'What was my HRV in a minute of rest?', 'Show my HRV in the first hours after waking',
    'how many steps did I take in the last hour', 'What was my resting heart rate in the last hours of the night',
    'my lipid results from the meter',
])
def test_a_metrics_own_unit_or_plain_wording_is_not_a_conversion(text):
    assert not query_selector.unit_conversion(request(text)), text


@pytest.mark.parametrize('text', ["What's my HRV this week? Answer in Romanian.", 'Show my steps this week and put it in a table.',
                                  'Keep it short. What was my resting heart rate yesterday?'])
def test_style_instructions_keep_the_text_transformation_code(selector, text):
    result = selector.select_query(request(text))
    assert result['status'] == 'handoff' and result['reason_codes'] == ['text_transformation_requires_native_context']
