from datetime import UTC, datetime

import pytest
from selector import SelectorRequest
from proposal_binding import canonicalize, temporal, resolve_context
from proposal_selector import ProposalSelector
from contracts.vita_read_contract import read_arguments


def request(text, history=(), metrics=None):
    return SelectorRequest.model_validate({'schema_version':'vita-selector/v1',
        'available_metrics':metrics or ['steps','apob','respiratory_rate','total_sleep','sleep_efficiency'],
        'literature_available':True, 'state':{'current_request':text,
        'recent_user_requests':list(history),'reference_date':'2026-09-23','time_zone':'UTC'}})


def run(text, history=(), semantic_error=None, **overrides):
    selector=ProposalSelector.__new__(ProposalSelector)
    predicted=dict(task='health',coverage='targeted',purpose='trend',research='none')
    predicted.update(overrides)
    result=selector.select(request(text,history), prediction=(predicted,{k:.8 for k in predicted}),
        coverage_decision=({'choice':'the proposed read covers the request','confidence':.9},semantic_error))
    plan=read_arguments(result,request(text).available_metrics,literature_available=True,
        record_types=('profile','workouts','labs','calendar'), now=datetime(2026,9,23,12,tzinfo=UTC))
    return result,plan


def test_prose_is_not_a_pre_model_vocabulary_gate():
    _,a=run('Show my steps last month')
    result,b=run('Last month, could you pull up the steps I managed to rack up, thanks?')
    assert result['status']=='selected'
    assert a==b


def test_model_judges_semantics_before_bad_date_is_rejected():
    seen=[]
    class Model:
        def tok(self,*args,**kwargs):return {'input_ids':[1,2]}
    selector=ProposalSelector.__new__(ProposalSelector);selector.model=Model()
    def predict(text):
        seen.append(text)
        return dict(task='health',coverage='targeted',purpose='trend',research='none'),{'task':.8}
    selector.predict=predict
    result=selector.select(request('Show steps on February 30, 2026'))
    assert len(seen)==1 and result['status']=='unsupported'
    assert 'invalid_or_out_of_range_date' in result['reason_codes']


@pytest.mark.parametrize('suffix',[
    'from Oura', 'on weekends', 'excluding runs', 'and email it to my doctor',
    'in pounds', 'below 20', 'from my phone',
])
def test_constraints_cannot_disappear_when_date_moves(suffix):
    for text in [f'Show steps last month {suffix}', f'Last month show steps {suffix}']:
        result,plan=run(text)
        assert result['status']=='unsupported' and plan is None


def test_unrecognized_constraint_requires_semantic_coverage():
    result,plan=run('Show steps near my workplace',semantic_error='semantic_coverage_unconfirmed')
    assert result['status']=='unsupported' and plan is None


def test_followup_carries_source_filter_and_replaces_period():
    result,plan=run('Now do last month',['Show steps from Oura in 2025'])
    assert result['status']=='unsupported' and plan is None
    assert 'source_filter' in result['reason_codes']
    assert result['diagnostics']['proposal']['period']['calendar_unit']=='month'


def test_new_subject_drops_old_source_and_date():
    result,plan=run('Show respiratory rate',['Show steps from Oura in 2025'])
    assert result['status']=='selected'
    assert plan['health_reads'][0]['concepts']==['respiratory_rate']
    assert plan['health_reads'][0]['range']=={'kind':'all_history'}


@pytest.mark.parametrize('text,start,end',[
    ('Steps from June 3 through June 17','2026-06-03','2026-06-17'),
    ('Steps between December 20, 2025 and January 5','2025-12-20','2026-01-05'),
    ('Steps from March 2025 through May 2025','2025-03-01','2025-05-31'),
    ('Steps for 2026-05','2026-05-01','2026-05-31'),
])
def test_complete_date_ranges(text,start,end):
    result,plan=run(text)
    assert result['status']=='selected',result['reason_codes']
    assert plan['health_reads'][0]['range']=={'kind':'between','start_at':start,'end_at':end}


@pytest.mark.parametrize('text',[
    'Steps from March 10 to March 3','Steps before June','Steps for June 31',
    'Steps for 2026-13','Steps in 2025 and sleep in 2026','Steps since June 1',
    'Steps for the past 200 days',
])
def test_invalid_or_unrepresentable_dates_handoff(text):
    result,plan=run(text)
    assert result['status']=='unsupported' and plan is None


def test_model_errors_propagate_not_disguised_as_handoff():
    class Broken:
        def tok(self,*args,**kwargs):raise RuntimeError('model failed')
    selector=ProposalSelector.__new__(ProposalSelector);selector.model=Broken()
    with pytest.raises(RuntimeError,match='model failed'):
        selector.select(request('Show steps'))


def test_native_output_validation():
    class Model:
        def tok(self,*args,**kwargs):return {'input_ids':[1,2]}
        def decide(self,*args):return [{'choice':'the proposed read covers the request','confidence':float('nan')}]
    selector=ProposalSelector.__new__(ProposalSelector);selector.model=Model()
    result,_=run('Show steps')
    from proposal_binding import ReadProposal
    _,error=selector.coverage_check('Show steps',ReadProposal.model_validate(result['diagnostics']['proposal']))
    assert error=='invalid_semantic_decision'


def test_index_binds_catalog_aliases_to_exact_inventory_ids():
    from proposal_binding import entities
    pairs = [(k,v) for _,_,values in entities('Show LDL cholesterol and SpO2', ['ldl_cholesterol','oxygen_saturation']) for k,v in values]
    assert pairs == [('metric','ldl_cholesterol'),('metric','oxygen_saturation')]
    pairs = [(k,v) for _,_,values in entities('Show sleep and resting heart rate', ['total_sleep','sleep_efficiency']) for k,v in values]
    assert ('unavailable','resting_heart_rate') in pairs


def test_absent_record_category_does_not_become_a_read():
    selector=ProposalSelector.__new__(ProposalSelector)
    req=request('Show my lab reports').model_copy(update={'available_record_types':[]})
    result=selector.select(req,prediction=({'task':'health','coverage':'targeted','purpose':'trend','research':'none'},{'task':.8}),
        coverage_decision=({},None))
    assert result['status']=='unsupported'
    assert 'requested_record_category_unavailable' in result['reason_codes']
    assert result['answers']['labs']=={'noul':0.0}


def test_followup_subject_swap_preserves_time_and_unsupported_source():
    _,plan=run('What about my steps then?',['Show my sleep past 3 weeks'])
    assert plan['health_reads'][0]['range']=={'kind':'relative','unit':'days','amount':21}
    result,plan=run('What about my steps then?',['Show my sleep from Oura past 3 weeks'])
    assert result['status']=='unsupported' and plan is None


def test_schema_metadata_is_not_permission_or_data_presence():
    result,plan=run('Show sleep and resting heart rate past 2 weeks')
    assert result['status']=='unsupported' and plan is None
    assert 'requested_metric_unavailable' in result['reason_codes']


def test_dictionary_separates_backend_capabilities_from_adapter():
    from schema_index import INDEX
    assert INDEX['adapter_capabilities']['source_filter'] is False
    assert INDEX['backend_capabilities_observed']['sources']=='authorized source filters supported'
    assert INDEX['records']['labs']['adapter_detail_limit']==3


def test_trained_head_identity_is_pinned_before_use(monkeypatch):
    import trained_proposal_selector as trained
    monkeypatch.setattr(trained.ProposalSelector,'__init__',lambda self, model:None)
    monkeypatch.setattr(trained,'INTENT_SHA256','0'*64)
    with pytest.raises(ValueError,match='Unapproved read-intent adapter'):
        trained.TrainedProposalSelector(None)


def test_complete_grammar_binding_does_not_need_a_learned_veto():
    selector=ProposalSelector.__new__(ProposalSelector)
    def unexpected(*args):raise AssertionError('Already completely bound')
    selector.coverage_check=unexpected
    result=selector.select(request('Show my steps yesterday'),prediction=(
        {'task':'health','coverage':'targeted','purpose':'trend','research':'none'},{'task':.8}))
    assert result['status']=='selected'
    assert result['diagnostics']['semantic_coverage']=={'method':'complete_grammar_binding'}


def test_profile_snapshot_cannot_claim_historical_fields():
    result,plan=run('Show my profile in 2024')
    assert result['status']=='unsupported' and plan is None
