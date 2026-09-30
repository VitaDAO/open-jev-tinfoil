"""Short non-English crisis text and whole-pack overdoses are acute (spec v1.8 2.4, 10g.33 item 1): they hand off with the acute code
before any other rule. A sentence with no English function word and a sign of another language hands off the whole message (10g.33)."""
import pytest

from query_selector import acute_or_crisis, foreign_language
from test_p0_rule_fixes import request, selector  # noqa: F401  (fixture)

READS = ['steps today?', 'hrv this week?', 'sleep score last night?', "what's my resting heart rate?"]

FOREIGN_CRISIS = [
    # Portuguese
    'quero morrer.', 'Quero morrer', 'vou me matar.', 'quero me matar.', 'não quero mais viver.', 'nao aguento mais, quero morrer.',
    'vou tirar a minha vida.', 'penso em suicídio.', 'tomei todos os comprimidos.',
    # Spanish
    'quiero morir.', 'quiero morirme.', 'me quiero matar.', 'voy a matarme.', 'ya no quiero vivir.', 'quiero suicidarme.',
    'me voy a quitar la vida.', 'no aguanto más, quiero acabar con todo.', 'me tomé todas las pastillas.',
    # German
    'ich will sterben.', 'ich möchte sterben', 'ich bringe mich um.', 'ich will mich umbringen.', 'ich will nicht mehr leben.',
    'ich denke an Selbstmord.', 'ich habe alle Tabletten genommen.',
    # French
    'je veux mourir.', 'je vais me tuer.', 'je veux me suicider.', "j'en ai marre, je veux en finir.", 'je ne veux plus vivre.',
    "j'ai pris toutes mes pilules.",
    # Italian
    'voglio morire.', 'voglio uccidermi.', 'voglio ammazzarmi.', 'non voglio più vivere.', 'voglio farla finita.', 'voglio togliermi la vita.',
    'ho preso tutte le pillole.',
    # Romanian
    'vreau să mor.', 'vreau sa mor.', 'vreau să mă sinucid.', 'o să mă omor.', 'nu mai vreau să trăiesc.', 'mi-am luat toate pastilele.',
    # Dutch
    'ik wil dood.', 'ik wil zelfmoord plegen.',
]


@pytest.mark.parametrize('crisis', FOREIGN_CRISIS)
def test_foreign_crisis_is_acute(crisis):
    assert acute_or_crisis(crisis)


@pytest.mark.parametrize('text', [f'{c} {r}' for c in ['quero morrer.', 'quiero morir.', 'ich will sterben.', 'voglio morire.', 'vreau sa mor.',
                                                        'je veux mourir.', 'ik wil dood.'] for r in READS[:3]]
                         + ['whats my hrv this week, quiero morir', 'Steps today? Quero morrer.', 'my sleep score. ich will sterben',
                            'Show my HRV this week. Mañana me mato.', 'resting heart rate today, tengo dolor de pecho fuerte',
                            'sleep score last night, j ai mal a la poitrine', 'steps today, no puedo respirar'])
def test_foreign_crisis_mixed_with_a_read_is_acute(text):
    assert acute_or_crisis(text)


def test_short_foreign_crisis_hands_off_with_the_acute_code(selector):
    for text in ('quero morrer. steps today?', 'voglio morire. hrv this week?', 'ich will sterben. sleep score last night?'):
        result = selector.select_query(request(text))
        assert result['status'] == 'handoff' and result['reason_codes'] == ['acute_or_crisis_requires_model'], text


def test_short_foreign_crisis_in_history_hands_off(selector):
    result = selector.select_query(request('and my steps today?', history=['quero morrer.']))
    assert result['status'] == 'handoff' and result['reason_codes'] == ['acute_or_crisis_requires_model']


@pytest.mark.parametrize('text', ['quero dormir. steps today?', 'ich bin müde. steps today?', 'voglio dormire. hrv this week?', 'quiero correr. steps today?',
                                  'je suis fatigué. sleep score?', 'vreau sa dorm. sleep score?', 'ik ben moe. steps today?', 'Bitte. Steps today?',
                                  'ich will laufen. my steps?', 'quero saber. my hrv?'])
def test_a_short_foreign_sentence_hands_off_the_whole_message(text):
    # One sign of another language and no English function word: a foreign sentence, however short.
    assert foreign_language(text)


OVERDOSE = [
    'I took a whole bottle of sleeping pills', 'i took a whole bottle of pills. steps today?', 'I swallowed an entire bottle of tablets',
    'just downed a whole pack of my meds', 'took the whole box of sleeping tablets an hour ago, hrv?', 'I swallowed a full bottle of pills',
    'I took an entire pack of paracetamol', 'i took the entire bottle of my antidepressants', 'swallowed half a bottle of pills',
    'I overdosed last night. sleep score?', "I OD'd on my meds", 'i think i od-ed', 'I just OD’d. my hrv?', 'i took a whole strip of tablets',
    'downed the whole bottle of ibuprofen', 'I ate a whole bottle of melatonin gummies',
]


@pytest.mark.parametrize('text', OVERDOSE)
def test_overdose_is_acute(text):
    assert acute_or_crisis(text)


BENIGN = [
    # Names, places and loanwords with accents or foreign spellings in otherwise English requests.
    "I'm writing this from a café in Porto. My active calories today?", 'My parents are visiting from Constanța. Steps this week?',
    "I've been bouncing between Cluj and Iași for work. My haemoglobin last year!", 'Café visits. My caffeine this week?',
    'Back from Iași. Steps today?', 'José says hi. My steps today?', 'Zoë steps today?', 'Beyoncé concert last night. My hrv?',
    'Hola! What was my sleep score?', 'show my steps, por favor', 'de facto my resting heart rate', 'Ciao! steps today?',
    'Steps mon-fri?', 'sleep score mon', 'steps since 9am EST', 'glucose in SI units', 'my AZ trip. steps?', 'Del Mar run. my hrv?',
    'Los Angeles trip. steps?', 'Mia and I walked. steps today?', 'Luna my dog. steps today?', 'matar paneer for dinner. calories?',
    'I had matar paneer. calories today?', 'Pasta con pomodoro. calories?', 'Sono scan went fine. hrv?', 'Pro con list done. my sleep?',
    # Look-alikes of the overdose and crisis wording.
    'I took my pills this morning. hrv today?', 'I took a whole day off. steps today?', 'drank a whole bottle of water. steps today?',
    'finished the whole bottle of wine. sleep score?', 'I took a whole pack of gum to the office. steps?', 'bought a whole box of protein bars. calories?',
    'I took a whole box of tissues to work. steps?', 'I took the whole pack on a hike. steps today?', 'dead tired. steps today?',
    'dying to see my vo2 max', 'the Moriarty episode. sleep score?', 'Mortgage stress. my hrv this week?', 'sterling week. steps?',
    'Tot swim class today. steps?', 'Ritz crackers. calories?',
    'my OD appointment (optometrist) went fine. steps?', 'Odd sleep last night. sleep score?',
]


@pytest.mark.parametrize('text', BENIGN)
def test_benign_look_alikes_are_not_acute(text):
    assert not acute_or_crisis(text)


@pytest.mark.parametrize('text', [t for t in BENIGN if t not in ('Café visits. My caffeine this week?',
                                                                  'Pasta con pomodoro. calories?', 'I had matar paneer. calories today?',
                                                                  'matar paneer for dinner. calories?', 'Ciao! steps today?')])
def test_benign_look_alikes_are_not_foreign(text):
    assert not foreign_language(text)
