from scripts.grade_learned import grade, topic_key


def test_only_nonsemantic_topic_formatting_is_normalized():
    assert topic_key('The accuracy of Lp(a) testing') == topic_key('accuracy of lp a testing')
    assert topic_key('vitamin A and sleep') != topic_key('vitamin and sleep')


def test_research_subject_cannot_lose_negation_population_or_number():
    for subject in ('5-HTP and sleep', 'sleep in adults over 65', 'people without diabetes'):
        case = {'current_request': 'Find papers on ' + subject, 'expected': {'status': 'research'}}
        result = {'status': 'planned', 'queries': [{'kind': 'research', 'topic': subject}]}
        assert grade(case, result, None) == 'ok_research'
        result['queries'][0]['topic'] = 'sleep'
        assert grade(case, result, None) == 'FALSE_ACCEPT'


def test_unknown_research_gold_is_unscored_not_correct():
    assert grade({'current_request': 'Tell me more', 'expected': {'status': 'research'}},
                 {'status': 'planned', 'queries': [{'kind': 'research', 'topic': 'sleep'}]}, None) == 'UNSCORED_RESEARCH'


def test_explicit_record_limit_and_date_basis_are_graded():
    case = {'expected': {'status': 'plan', 'reads': [{'records': ['labs'], 'limit': 5, 'date_basis': 'exam_date'}]}}
    result = {'status': 'planned', 'queries': [{'kind': 'health', 'records': ['labs'], 'limit': 200, 'date_basis': 'exam_date'}]}
    assert grade(case, result, lambda a, b: True) == 'FALSE_ACCEPT'


def test_extra_calendar_basis_is_not_collapsed_away():
    case = {'expected': {'status': 'plan', 'reads': [{'records': ['calendar'], 'date_basis': 'next_due_date'}]}}
    result = {'status': 'planned', 'queries': [
        {'kind': 'health', 'records': ['calendar'], 'date_basis': 'next_due_date'},
        {'kind': 'health', 'records': ['calendar'], 'date_basis': 'last_done_date'}]}
    assert grade(case, result, lambda a, b: True) == 'FALSE_ACCEPT'


def test_two_calendar_bases_use_distinct_queries():
    case = {'expected': {'status': 'plan', 'reads': [
        {'records': ['calendar'], 'date_basis': 'next_due_date'},
        {'records': ['calendar'], 'date_basis': 'last_done_date'}]}}
    result = {'status': 'planned', 'queries': [
        {'kind': 'health', 'records': ['calendar'], 'date_basis': 'last_done_date'},
        {'kind': 'health', 'records': ['calendar'], 'date_basis': 'next_due_date'}]}
    assert grade(case, result, lambda a, b: True) == 'ok_plan'
    result['queries'][1]['date_basis'] = 'last_done_date'
    assert grade(case, result, lambda a, b: True) == 'FALSE_ACCEPT'


def test_politeness_is_not_part_of_the_research_subject():
    for suffix in ('Thank you very much.', 'Thanks in advance!', 'Much appreciated.', 'Ta.', 'Thanks a lot.'):
        case={'current_request':'What is the evidence for NAD+ supplements? '+suffix, 'expected':{'status':'research'}}
        result={'status':'planned','queries':[{'kind':'research','topic':'NAD+ supplements'}]}
        assert grade(case,result,None)=='ok_research'


def test_second_sentence_constraint_is_not_discarded():
    case={'current_request':'Find research on sleep. Only studies in adults over 65.', 'expected':{'status':'research'}}
    result={'status':'planned','queries':[{'kind':'research','topic':'sleep'}]}
    assert grade(case,result,None)=='FALSE_ACCEPT'


def test_parenthesized_names_and_multiple_politeness_clauses():
    assert topic_key('lipoprotein(a) and heart disease') == topic_key('lipoprotein a and heart disease')
    case={'current_request':'research on caffeine timing and sleep quality pls. Cheers.', 'expected':{'status':'research'}}
    result={'status':'planned','queries':[{'kind':'research','topic':'caffeine timing and sleep quality'}]}
    assert grade(case,result,None)=='ok_research'
