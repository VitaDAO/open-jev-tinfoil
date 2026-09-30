from datetime import date
import pytest
from learned_parser.decode import Parser

@pytest.mark.parametrize('text,spans', [
    ('Next, show my calendar last week.', ['last week']),
    ('Show appointments over the past 3 months', ['3 months']),
    ('Show appointments in the previous month', ['month']),
])
def test_discourse_and_partial_time_tags_do_not_choose_calendar_basis(text, spans):
    assert Parser._calendar_wording(text, spans)==(False,False)

@pytest.mark.parametrize('text,expected', [
    ('Show past appointments', (True,False)),
    ('Show completed checkups', (True,False)),
    ('Show my next appointment', (False,True)),
    ('Show scheduled appointments', (False,True)),
])
def test_actual_calendar_qualifiers_survive(text,expected):
    assert Parser._calendar_wording(text,[])==expected


def test_completion_in_plain_context_is_not_calendar_wording():
    parser=Parser.__new__(Parser)
    text='I completed a long project. Show appointments in August.'
    start=text.index('appointments')
    assert parser._calendar_read_text(text,[(start,start+len('appointments'))])=='Show appointments in August.'

@pytest.mark.parametrize('text', ['Show steps from June to July', 'HRV from 2024 to 2025'])
def test_from_named_period_to_named_period_is_two_windows(text):
    parser=Parser.__new__(Parser)
    start=text.index('from')
    assert len(parser._split_windows(text,start,len(text),date(2026,9,26)))==2

@pytest.mark.parametrize('text', ['Show steps from Monday to Wednesday', 'Show steps between June and July', 'Show steps from June through July'])
def test_explicit_ranges_remain_one_window(text):
    parser=Parser.__new__(Parser)
    start=text.index('from') if 'from' in text else text.index('between')
    assert len(parser._split_windows(text,start,len(text),date(2026,9,26)))==1

@pytest.mark.parametrize('model', ['r8a','r8b'])
def test_actual_neural_calendar_context_and_month_comparison(model):
    from pathlib import Path
    from scripts.learned_replay import load
    parser,cases=load(Path(__file__).parent/'fixtures/learned_replay'/('jevparse_'+model))
    texts={'Next, show my calendar last week.','I completed a long project. Show appointments in August.','Show my steps from June to July'}
    outcomes={req['state']['current_request']:parser.select(req) for req,_ in cases if req['state']['current_request'] in texts}
    assert set(outcomes)==texts
    for text,result in outcomes.items():
        if text.startswith('Show my steps'):
            assert result['status']=='handoff',result
            assert result['reason_codes']==['unbound_request_constraint']
            continue
        assert result['status']=='planned',result
        assert len(result['queries'])==2
        assert {q['date_basis'] for q in result['queries']}=={'next_due_date','last_done_date'}


def test_anaphoric_completion_stays_with_calendar_request():
    parser=Parser.__new__(Parser)
    text='Show my appointments in November. I already completed them.'
    start=text.index('appointments')
    wording=parser._calendar_read_text(text,[(start,start+len('appointments'))])
    assert 'completed them' in wording
    assert parser.calendar_reads('between',None,None,0,'November',wording,date(2026,9,26),['November'])==[
        ('last_done_date',{'kind':'between','start_at':'2025-11-01','end_at':'2025-11-30'})]


def test_demonstrative_on_an_unrelated_noun_is_not_a_calendar_backreference():
    parser=Parser.__new__(Parser)
    text='I completed this project. Show appointments in August.'
    start=text.index('appointments')
    assert parser._calendar_read_text(text,[(start,start+len('appointments'))])=='Show appointments in August.'


def test_dated_month_range_without_comparison_cue_does_not_guess():
    parser=Parser.__new__(Parser)
    text='Show my triglycerides from January to March 2026'
    a=text.index('from'); z=text.index(' to '); b=z+4
    parser._cmp_windows=lambda *args: ([{'a':a,'z':z,'kind':'abs'},{'a':b,'z':len(text),'kind':'abs'}],[(8,21)])
    req={'state':{'reference_date':'2026-09-26'}}
    result=parser.window_compare(req,text,None,None,0,len(text),lambda reason: {'status':'handoff','reason':reason})
    assert result=={'status':'handoff','reason':'unbound_request_constraint'}
