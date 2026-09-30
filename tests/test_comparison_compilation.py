"""Comparison composition must preserve explicit windows and reject dropped constraints."""
from datetime import date
from pathlib import Path

import pytest

from learned_parser.decode import Parser
from scripts.learned_replay import load


@pytest.mark.parametrize('period', [
    {'kind':'relative','unit':'days','amount':30},
    {'kind':'relative','unit':'months','amount':7},
    {'kind':'all_history'},
])
def test_rolling_comparison_rejects_a_child_that_changed_the_requested_window(period):
    parser=Parser.__new__(Parser)
    result={'status':'planned','queries':[{'kind':'health','metrics':['steps'],'records':[],
        'operation':'trend','period':period,'source':None,'date_basis':'observed_at'}], 'confidence':.99}
    actual=parser._cmp_reads([('abs','Show my steps ','past 7 days','',None)],False,lambda text:result,date(2026,9,26))
    assert actual=='period_disagreement'


def test_rolling_comparison_retains_the_exact_requested_window():
    parser=Parser.__new__(Parser)
    period={'kind':'relative','unit':'days','amount':7}
    result={'status':'planned','queries':[{'kind':'health','metrics':['steps'],'records':[],
        'operation':'trend','period':period,'source':None,'date_basis':'observed_at'}], 'confidence':.99}
    actual=parser._cmp_reads([('abs','Show my steps ','past 7 days','',None)],False,lambda text:result,date(2026,9,26))
    assert not isinstance(actual,str)
    assert actual[0][0][0]['period']==period


@pytest.mark.parametrize('model',['r8a','r8b'])
def test_real_comparison_heads_compile_only_complete_unambiguous_reads(model):
    parser,cases=load(Path(__file__).parent/'fixtures/learned_replay'/('jevparse_'+model))
    texts={'Compare my steps this week with last week.':('steps',{'kind':'calendar','period':'week'}, {'kind':'between','start_at':'2026-09-14','end_at':'2026-09-20'}),
           'Compare my HRV this month with last month.':('heart_rate_variability',{'kind':'calendar','period':'month'},{'kind':'between','start_at':'2026-08-01','end_at':'2026-08-31'}),
           'Compare my steps in June and July.':('steps',{'kind':'between','start_at':'2026-06-01','end_at':'2026-06-30'},{'kind':'between','start_at':'2026-07-01','end_at':'2026-07-31'})}
    seen=set()
    for req,_ in cases:
        text=req['state']['current_request']
        if text not in texts:continue
        seen.add(text);metric,a,b=texts[text];r=parser.select(req)
        assert r['status']=='planned'
        assert [q['metrics'] for q in r['queries']]==[[metric],[metric]]
        assert [q['period'] for q in r['queries']]==[a,b]
        assert all(q['source'] is None and q['operation']=='trend' for q in r['queries'])
    assert seen==set(texts)


@pytest.mark.parametrize('model',['r8a','r8b'])
def test_real_comparison_heads_reject_constraint_mutations_and_overlap(model):
    parser,cases=load(Path(__file__).parent/'fixtures/learned_replay'/('jevparse_'+model))
    required={'Compare my HRV this week with last week. Only readings above 50.',
              'Compare my HRV this week with last week. Is that healthy?',
              "Compare my HRV this week with last week. Also show my wife's readings.",
              'Compare my HRV this week with last week. And exclude Garmin.',
              'Compare my HRV this week with last week. I have crushing chest pain.',
              'Compare my HRV this month with the past 90 days.',
              'Compare my steps past 14 days with past 2 weeks.'}
    seen=set()
    for req,_ in cases:
        if req['state']['current_request'] in required:
            seen.add(req['state']['current_request']);r=parser.select(req)
            assert r['status']=='handoff' and not r['queries'],r
    assert seen==required


@pytest.mark.parametrize('before,after', [
    ('my Oura steps ', ''),
    ('my Garmin steps above 100 ', ''),
    ('my Garmin steps ', ' excluding weekends'),
])
def test_relative_comparison_never_copies_past_a_new_clause_constraint(before,after):
    parser=Parser.__new__(Parser)
    first={'status':'planned','queries':[{'kind':'health','metrics':['steps'],'records':[],
        'operation':'trend','period':{'kind':'calendar','period':'week'},'source':'garmin',
        'date_basis':'observed_at'}],'confidence':.99}
    clauses=[('abs','Compare my Garmin steps ','this week','',None),
             ('rel',before,'the week before',after,None)]
    assert parser._cmp_reads(clauses,False,lambda text:first,date(2026,9,26))=='unbound_request_constraint'


@pytest.mark.parametrize('window', ['the week before', 'for the week before', 'for the week before as well'])
def test_relative_comparison_shifts_only_an_unchanged_clause(window):
    parser=Parser.__new__(Parser)
    first={'status':'planned','queries':[{'kind':'health','metrics':['steps'],'records':[],
        'operation':'trend','period':{'kind':'calendar','period':'week'},'source':'garmin',
        'date_basis':'observed_at'}],'confidence':.99}
    clauses=[('abs','Compare my Garmin steps ','this week','',None),
             ('rel','my Garmin steps ',window,'',None)]
    got=parser._cmp_reads(clauses,False,lambda text:first,date(2026,9,26))
    assert not isinstance(got,str)
    assert got[0][1][0]['source']=='garmin'
    assert got[0][1][0]['period']=={'kind':'between','start_at':'2026-09-14','end_at':'2026-09-20'}


def test_relative_window_text_cannot_hide_a_filter():
    parser=Parser.__new__(Parser)
    first={'status':'planned','queries':[{'kind':'health','metrics':['steps'],'records':[],
        'operation':'trend','period':{'kind':'calendar','period':'week'},'source':None,
        'date_basis':'observed_at'}],'confidence':.99}
    clauses=[('abs','Show my steps ','this week','',None),
             ('rel','Show my steps ','the week before above 100','',None)]
    assert parser._cmp_reads(clauses,False,lambda text:first,date(2026,9,26))=='unbound_request_constraint'


def test_served_decoder_change_changes_the_selector_pin(monkeypatch,tmp_path):
    import query_selector as Q
    import trained_proposal_selector
    monkeypatch.setattr(Q,'ROOT',tmp_path)
    monkeypatch.setattr(trained_proposal_selector,'identity',lambda:'base')
    for name in ['query_plan.py','query_selector.py','query_execution.py','metadata/reason-codes.v1.json',
                 'metadata/context-classes.v1.json','learned_parser/parse.py','learned_parser/decode.py',
                 'learned_parser/verify.py','metadata/learned/panels.json']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    monkeypatch.setenv('OPEN_JEV_PARSER','learned');original=Q.identity()
    (tmp_path/'learned_parser/decode.py').write_text('changed decoder')
    assert Q.identity()!=original
    original=Q.identity();(tmp_path/'metadata/learned/panels.json').write_text('changed panel definition')
    assert Q.identity()!=original
    monkeypatch.delenv('OPEN_JEV_PARSER');rules=Q.identity()
    (tmp_path/'learned_parser/decode.py').write_text('another decoder')
    assert Q.identity()==rules
