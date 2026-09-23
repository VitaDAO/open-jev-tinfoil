from datetime import UTC, datetime
import importlib.util
from pathlib import Path

import pytest
from learned_selector import LearnedSelector, resolve_request
from selector import SelectorRequest
from contracts.vita_read_contract import read_arguments


def request(text, history=()):
    return SelectorRequest.model_validate({'schema_version':'vita-selector/v1',
        'available_metrics':['steps','apob','respiratory_rate','total_sleep','sleep_efficiency'],
        'literature_available':True,'state':{'current_request':text,
        'recent_user_requests':list(history),'reference_date':'2026-09-23','time_zone':'UTC'}})


def run(text, history=(), **overrides):
    # Isolate argument/guard behavior from changing model accuracy.
    selector=LearnedSelector.__new__(LearnedSelector);selector.torch=None
    predicted=dict(task='health',coverage='targeted',purpose='unsupported',research='none',
                   period_kind='unstated',context='inherit',calendar_basis='current_plans',
                   profile='no',workouts='no',labs='no',calendar='no')
    predicted.update(overrides)
    margins={k:.8 if k in ('task','coverage','research') else .001 for k in predicted}
    req=request(text,history)
    result=selector.select(req,prediction=(predicted,margins))
    plan=read_arguments(result,req.available_metrics,literature_available=True,
        record_types=('profile','workouts','labs','calendar'),now=datetime(2026,9,23,12,tzinfo=UTC))
    return result,plan


def test_exact_arguments_do_not_depend_on_irrelevant_low_margin_heads():
    result,plan=run('Please show my latest respiratory rate yesterday, please.')
    assert result['status']=='selected'
    assert plan['health_reads'][0]['purpose']=='latest'
    assert plan['health_reads'][0]['concepts']==['respiratory_rate']
    assert plan['health_reads'][0]['range']=={'kind':'between','start_at':'2026-09-22','end_at':'2026-09-22'}


def test_period_correction_preserves_latest_and_replaces_all_dates():
    _,plan=run('Actually, make that last month',['Show my latest ApoB in 2025'])
    read=plan['health_reads'][0]
    assert read['purpose']=='latest' and read['concepts']==['apob']
    assert read['range']=={'kind':'between','start_at':'2026-08-01','end_at':'2026-08-31'}


@pytest.mark.parametrize('previous',[
    'Show my ApoB from Oura this year', 'Show my last 5 ApoB readings',
    'Show my steps excluding weekends', 'Show my latest 5 lab reports',
])
def test_followups_do_not_drop_prior_qualifiers(previous):
    result,plan=run('What about last month?',[previous])
    assert result['status']=='unsupported' and plan is None


def test_new_subject_does_not_inherit_prior_period_or_source():
    resolved=resolve_request(request('Show my respiratory rate',['Show my ApoB from Oura last month']))
    assert resolved.state.recent_user_requests==[]
    _,plan=run(resolved.state.current_request)
    assert plan['health_reads'][0]['range']=={'kind':'all_history'}
    assert plan['health_reads'][0]['concepts']==['respiratory_rate']


@pytest.mark.parametrize('text',[
    'Show my latest 5 lab reports', 'Show my 3 most recent reports',
    'Show my latest ApoB readings 3', 'Show my latest five workouts',
    'Show my steps on weekends last month', 'Show my steps in the mornings this week',
    "Please analyze my partner's health", 'Analyze my health excluding labs',
    'Show my steps and delete my history', 'Show my steps in pounds',
    'Analyze my health and sleep', 'Show my steps on 2026-04-31',
    'Show my latest steps trend', 'Show my health plans last month',
])
def test_unrepresented_constraints_cannot_become_executable(text):
    result,plan=run(text,coverage='broad')
    assert result['status']=='unsupported' and plan is None


def test_records_bind_without_model_record_flags_and_calendar_basis_is_preserved():
    _,plan=run('Show my screening events completed yesterday')
    assert plan['health_reads'][0]['record_types']==['calendar']
    assert plan['health_reads'][0]['calendar_date_basis']=='last_done_date'


def test_descriptive_summary_does_not_add_research_from_wrong_model_head():
    result,plan=run('Show me a snapshot of my health past 14 days',coverage='broad',research='broad_overview')
    assert result['status']=='selected' and 'literature_reads' not in plan


def test_exact_grader_rejects_wrong_query_and_separates_errors():
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_selector_exact.py'
    spec=importlib.util.spec_from_file_location('exact_grader',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,plan=run('Show my steps yesterday')
    gold={'acceptable_plans':[plan],'handoff_ok':False}
    assert module.grade({'status':'selected'},plan,gold)=='correct_plan'
    _,wrong=run('Show my ApoB last month')
    assert module.grade({'status':'selected'},wrong,gold)=='wrong_plan'
    assert module.grade({'status':'selected'},None,gold)=='invalid_response'
    assert module.grade({'status':'unsupported'},None,{'acceptable_plans':[],'handoff_ok':True},'KeyError')=='harness_error'


def test_model_failure_is_not_disguised_as_a_safe_handoff():
    class BrokenModel:
        def tok(self,*args,**kwargs):
            raise ValueError('synthetic model failure')
    selector=LearnedSelector.__new__(LearnedSelector)
    selector.torch=None;selector.model=BrokenModel()
    with pytest.raises(ValueError,match='synthetic model failure'):
        selector.select(request('Show my steps'))
