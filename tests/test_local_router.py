import httpx
import pytest
from examples.local_vita_router import MODEL_REVISION, QUESTIONS, route


def test_only_record_access_is_replaced():
    requests = []
    answers = [{'choice':'explain a concept'}, {'noul':0.9}, {'score':0.0}]
    def respond(request):
        import json
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200,json={'model_revision':MODEL_REVISION,'answers':answers if len(requests)==1 else [{'noul':.1}]})
    with httpx.Client(base_url='http://local',transport=httpx.MockTransport(respond)) as client:
        result = route(client, 'What is REM sleep?')
    assert requests == [{'state':'What is REM sleep?','questions':list(QUESTIONS)}, {'state':'What is REM sleep?','questions':[QUESTIONS[1]]}]
    assert result['action'] == 'explain a concept'
    assert result['record_access_probability'] == .1
    assert result['answers'] == [answers[0], {'noul':.1}, answers[2]]


@pytest.mark.parametrize('status,body', [(503,{}),(200,{'model_revision':'wrong','answers':[{}]})])
def test_errors_do_not_return_partial_decisions(status,body):
    with httpx.Client(base_url='http://local',transport=httpx.MockTransport(lambda request:httpx.Response(status,json=body))) as client:
        with pytest.raises((httpx.HTTPStatusError,RuntimeError)):
            route(client,'synthetic input')
