import asyncio
import copy
import httpx
import pytest
from examples.vita_orchestrator import prepare_model_turn,ORCHESTRATOR_INSTRUCTIONS
from query_plan import QueryPlan,HealthRead,ResearchRead,request_identity
from routing import MODEL_REVISION
from test_query_plan import request,metric_payload

def candidate(req,queries=None):
 return QueryPlan(status='planned',queries=queries or [HealthRead(metrics=['steps'],period={'kind':'all_history'})],
  time_zone=req.state.time_zone,selector_sha256='1'*64,adapter_sha256='2'*64,
  model_revision=MODEL_REVISION,request_sha256=request_identity(req))

def prepare(req,p,payload=None,error=None):
 calls=[]
 class Client:
  def select(self,request):
   if error:raise error
   return p
 async def authorized(kind,operation):calls.append(operation);return copy.deepcopy(payload)
 result=asyncio.run(prepare_model_turn(Client(),req,authorized))
 return result,calls

def test_complete_acquisition_preserves_original_task_and_model_still_must_answer():
 req=request();result,calls=prepare(req,candidate(req),metric_payload())
 assert len(calls)==1 and result['user_message']==req.state.current_request
 assert result['selector_evidence']['all_requested_delivered'] is True
 assert result['additional_system_instructions']==ORCHESTRATOR_INSTRUCTIONS
 assert 'never an answer' in ORCHESTRATOR_INSTRUCTIONS
 assert 'diagnostics' not in result['selector_evidence']

@pytest.mark.parametrize('mode',['handoff','invalid','network','attestation'])
def test_failed_selection_preserves_question_for_existing_model_and_does_no_io(mode):
 req=request();p=candidate(req);error=None
 if mode=='handoff':p=p.model_copy(update={'status':'handoff','queries':[],'reason_codes':['unsupported']})
 if mode=='invalid':p=p.model_copy(update={'request_sha256':'3'*64})
 if mode=='network':error=httpx.ReadTimeout('synthetic-private-exception')
 if mode=='attestation':error=RuntimeError('synthetic-private-exception')
 result,calls=prepare(req,p,error=error)
 assert calls==[] and result['user_message']==req.state.current_request
 assert not result['selector_evidence']['all_requested_delivered']
 assert 'synthetic-private-exception' not in str(result)

@pytest.mark.parametrize('research',[False,True])
def test_missing_health_or_unreviewed_research_is_not_a_completed_answer(research):
 req=request()
 p=candidate(req,[ResearchRead(targets=['sleep'])]) if research else candidate(req)
 payload={'status':'ok','sources':[]} if research else {'status':'unavailable'}
 result,calls=prepare(req,p,payload)
 assert len(calls)==1 and not result['selector_evidence']['all_requested_delivered']
 assert result['selector_evidence']['delivery_status']==('needs_evidence_review' if research else 'incomplete')
