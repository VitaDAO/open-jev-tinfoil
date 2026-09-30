"""A sentence outside English hands off the whole message (spec v1.8 10g.33): the evaluated contract is English."""
import pytest

from query_selector import foreign_language
from test_p0_rule_fixes import request, selector  # noqa: F401  (fixture)


@pytest.mark.parametrize('text', ["vreau sa vad somnul. what's my HRV?", 'Nu mai pot. Show my sleep', 'cati pasi am facut azi',
                                  'Wie viele Schritte habe ich heute gemacht?', "combien de pas j'ai fait aujourd'hui",
                                  'quanti passi ho fatto oggi', 'cuantos pasos di hoy', 'arată-mi somnul', 'Qual é a minha frequência cardíaca?',
                                  'Cât am dormit azi-noapte?', 'Привет, сколько я спал?', '我昨晚睡了多久', 'Hur mycket sov jag i natt?', 'Πόσα βήματα έκανα σήμερα;'])
def test_non_english_sentences_are_detected(text):
    assert foreign_language(text)


@pytest.mark.parametrize('text', ["What's my HRV this week?", 'my ferritin lately', 'Hola! What was my sleep score?', 'show my steps, por favor',
                                  'I did a 10k with my mates. Resting heart rate this week?', 'de facto my resting heart rate', 'PE class tired me out. My steps today?',
                                  "I'm writing this from a café in Porto. My active calories today?", 'My parents are visiting from Constanța. Steps this week?',
                                  'my γ-GT over time', "I've been bouncing between Cluj and Iași for work. My haemoglobin last year!", 'my homocysteine in µmol/L'])
def test_english_requests_are_not_flagged(text):
    assert not foreign_language(text)


def test_a_foreign_sentence_hands_off_the_whole_message(selector):
    result = selector.select_query(request("vreau sa vad pasii. What's my resting heart rate?"))
    assert result['status'] == 'handoff' and result['reason_codes'] == ['language_outside_evaluated_contract']
