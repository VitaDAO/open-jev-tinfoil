"""An earlier turn that states something about the user's own medication hands off the follow-up (S1; spec §2.4 class 1, 10g.28,
10g.36.4). In the current turn the same sentence already hands off; in history only the models read it, and the crisis head does
not score history. Questions about medications, other people's medication and supplements stay as before."""
import pytest

import query_selector
from test_medication_change import StubParser, request
from test_p0_rule_fixes import selector  # noqa: F401  (fixture)

STATED = ['I gave my prescriptions a holiday this week.', "haven't touched my blood pressure pills since monday.",
          'binned my metformin last week, sick of it.', 'I ditched my antidepressants cold turkey on Friday.',
          "I've been on a break from my statins. What is my LDL?", 'my insulin is sitting in the fridge unopened this week']
NOT_STATED = ['what meds am I on', "what's my ldl?", 'my mom takes statins for her cholesterol.', 'I take vitamin D pills',
              'my doctor wants me to watch my LDL.', 'that was a killer workout this morning. how many steps did I take?',
              'my legs are dead after the run', 'I finished a murder mystery novel last night.', "I'm dying to know how I slept"]


@pytest.mark.parametrize('text', STATED)
def test_own_medication_statement_in_history_is_flagged(text):
    assert query_selector.history_medication(text), text


@pytest.mark.parametrize('text', NOT_STATED)
def test_questions_other_people_and_supplements_are_not(text):
    assert not query_selector.history_medication(text), text


@pytest.mark.parametrize('position', [1, 2, 4])
@pytest.mark.parametrize('text', STATED[:3])
def test_a_follow_up_after_a_medication_statement_hands_off_before_any_parser(monkeypatch, selector, text, position):
    stub = StubParser()
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: stub)
    history = [text] + ['steps today', 'how did I sleep last night', 'HRV this week'][:position - 1]    # text sits at turn -position
    for current in ('and my HRV?', "Different topic - what's my resting heart rate this month?"):
        result = selector.select_query(request(current, history))
        assert result['status'] == 'handoff' and result['reason_codes'] == ['unbound_request_constraint'], (current, history, result)
    assert stub.calls == 0


@pytest.mark.parametrize('text', NOT_STATED[:3])
def test_a_follow_up_after_a_medication_question_still_reaches_the_parser(monkeypatch, selector, text):
    stub = StubParser()
    monkeypatch.setenv('OPEN_JEV_PARSER', 'learned')
    monkeypatch.setattr(query_selector, 'learned_parser', lambda: stub)
    result = selector.select_query(request('and my ApoB?', [text]))
    assert stub.calls == 1 and result['status'] == 'planned', result
