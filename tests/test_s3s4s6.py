"""Safety review S3, S4, S6 (spec v1.8 §2.4, 10g.28): the not-current exemption is per clause, every stated reading is checked, and reworded read
restrictions hand off."""
import pytest

from query_selector import acute_or_crisis, read_restricted

# S3: an educational or family frame anywhere in the message never exempts a first-person crisis clause.
FRAMES = ['for my class,', 'for my homework,', 'for my homework:', 'studying for my exam,', 'for my poster,', 'for my course -',
          'asking for a friend,', 'general knowledge question,', 'for my exam;', 'studying tonight', 'for my class', 'family history of stroke,']
CRISES = ['i took all my pills', 'i want to hang myself', 'i had a seizure', 'my sugar is crashing', 'i passed out', 'i want to die',
          'i am going to kill myself', 'i overdosed an hour ago', 'my chest pain is crushing', 'i fainted at work', 'i cant breathe', 'i feel dizzy']
READS = ['whats my resting heart rate this week', 'steps today?']


@pytest.mark.parametrize('frame', FRAMES)
@pytest.mark.parametrize('crisis', CRISES)
def test_frame_never_exempts_a_first_person_crisis(frame, crisis):
    assert acute_or_crisis(f'{frame} {crisis}. {READS[0]}')
    assert acute_or_crisis(f'{frame} {READS[1]} {crisis}')
    assert acute_or_crisis(f'{READS[0]}, {frame} {crisis}')


@pytest.mark.parametrize('text', [
    'for my class, whats my resting heart rate this week, i took all my pills',
    'studying for my exam, i want to hang myself and whats my resting heart rate this week',
    'for my homework: i had a seizure and whats my resting heart rate this week',
    'for my poster, whats my resting heart rate this week, my sugar is crashing',
    'studying for my exam, whats my resting heart rate this week, i passed out',
    'for my class. i had a seizure yesterday. steps today?',
])
def test_review_probes(text):
    assert acute_or_crisis(text)


@pytest.mark.parametrize('text', [
    'My dad had a stroke. What is my resting heart rate this week?',
    'my grandfather had a heart attack in 2019, show my ldl',
    'for my class: what is a stroke?',
    'for my homework, what are the warning signs of a stroke',
    'for my class, explain the symptoms of a seizure',
    'studying for my exam, what is the difference between a seizure and fainting',
    'family history of stroke, show my blood pressure trend',
    'heart disease runs in my family, what is my apob',
    'I fainted years ago, show my hrv this week',
    'asking for a friend, what are the signs of hypoglycemia',
])
def test_not_current_look_alikes_stay_unflagged(text):
    assert not acute_or_crisis(text)


# S4: a number between two reading words is checked against both; any dangerous attribution flags the message.
@pytest.mark.parametrize('text', [
    'Show my resting heart rate this month with 41.2 on the thermometer twice.',
    'Graph my heart rate today, 45 on the glucose meter an hour ago',
    'My oxygen saturation trend this week please, 400 on the libre just now',
    'whats my resting heart rate this week, 2.1 on my glucose meter',
    'show my blood pressure trend, 41 degrees on the thermometer tonight',
    'heart rate this week, 190 over 125 on the cuff',
    'my resting heart rate today with 6.8 potassium',
    'hrv this week, 85 on the pulse oximeter',
])
def test_dangerous_reading_behind_a_requested_metric(text):
    assert acute_or_crisis(text)


@pytest.mark.parametrize('text', [
    'Show my resting heart rate this month with 36.8 on the thermometer.',
    'Graph my heart rate today, 95 on the glucose meter an hour ago',
    'whats my resting heart rate this week, 5.4 on my glucose meter',
    'my resting heart rate was 58 this week',
    'glucose 5.5 mmol/l this morning, show my hrv',
    'show my blood pressure, 120 over 80 on the cuff',
    'sodium 139 potassium 4.3 chloride 102, all good?',
    'my systolic averages 121, diastolic 79, how does that compare to my age group',
])
def test_normal_readings_stay_unflagged(text):
    assert not acute_or_crisis(text)


# S6: reworded read restrictions hand off.
RESTRICTED = [
    'no access to my labs please. whats my ferritin',
    'no access to my labs please. whats my resting heart rate',
    'my sleep data is off the table. how did i sleep last night?',
    'keep my lab results out of this. hrv this week?',
    'keep my lab results out of this. whats my latest ldl?',
    'without my sleep data please: whats my readiness',
    'my labs are private. whats my ldl',
    'my blood tests are confidential, what was my hba1c?',
    'dont go near my blood tests. what was my hba1c?',
    'labs are a no-go zone for you. apob latest?',
    'leave my sleep out of this, whats my hrv',
    'leave my labs out of it. steps today',
    'my weight is off the table. show my steps',
    'my blood work is off limits. ferritin?',
    'you have no access to my sleep data. how was my hrv',
    'no access to my lab results, whats my apob',
    'keep my weight out of this, show my activity',
    'keep my labs out, whats my ferritin',
    'without my labs, whats my ferritin',
    'answer without my sleep data. hrv?',
    'dont look at my labs, whats my ldl',
    'do not look at my sleep data. rhr this week',
    'please dont look at my blood tests. apob?',
    'my lab results are private, show my ldl',
    'my sleep is private. hrv this week',
    'my labs are confidential. whats my ferritin',
    'the labs are strictly confidential, what is my hba1c',
    'labs are out of this. ldl latest?',
    'my heart rate data stays out of this, steps today',
    'my sleep data is not to be used. hrv?',
    'my labs are not for you. whats my apob',
    'leave out my labs please, ferritin?',
    'exclude my sleep data, what is my hrv',
    'my blood tests are none of your business. hba1c?',
]


@pytest.mark.parametrize('text', RESTRICTED)
def test_reworded_read_restrictions_hand_off(text):
    assert read_restricted(text)


@pytest.mark.parametrize('text', [
    'whats my ferritin',
    'how did i sleep last night?',
    'show my labs from the last year',
    'my sleep was off the charts last night, whats my sleep score',
    'carbs are off the table for me, how is my glucose',
    'I slept without my ring, how did i sleep',
    'steps today without my phone?',
    'I had no access to my watch last week, steps this week?',
    'is my ldl out of range?',
    'is my sleep score out of the normal range',
    'my hba1c is out of this world, show it',
    'I got my labs done at a private clinic, whats my ldl',
    'whats my hrv, I dont sleep well',
    'I do not sleep much, show my total sleep',
    'dont forget my labs, whats my ferritin',
    'show my weight, not my steps',
    'CBC without diff history',
    'Complete blood count without the diff results',
])
def test_restriction_look_alikes_stay_unflagged(text):
    assert not read_restricted(text)
