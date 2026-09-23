"""Synthetic real-model contract and rejection tests against local/verified proxy."""
import json
import os
import statistics
import time
import urllib.error
import urllib.request

base = os.environ.get('BASE_URL', 'http://127.0.0.1:18080')
headers = {'Authorization': 'Bearer ' + os.environ['OPEN_JEV_API_KEY'], 'Content-Type': 'application/json'}
body = {'state': 'I was charged twice for the same order. I want my money back.', 'questions': [
    {'type': 'choice', 'instructions': 'Which team should handle this?', 'options': ['billing', 'technical support', 'sales']},
    {'type': 'score', 'instructions': 'How positive is the sentiment?', 'options': ['negative', 'neutral', 'positive']},
    {'type': 'noul', 'instructions': 'The customer is asking for a refund.'}]}

def post(payload, auth=True):
    request = urllib.request.Request(base + '/decide', data=json.dumps(payload).encode(), headers=headers if auth else {'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)

assert post(body, False)[0] == 401
assert post({'state':'test','questions':[]})[0] == 422
assert post({**body, 'state': 'word ' * 300})[0] == 422
assert post({**body, 'questions': [body['questions'][0]] * 32})[0] == 422
elapsed, inference = [], []
for _ in range(10):
    start = time.perf_counter()
    status, result = post(body)
    assert status == 200, (status, result)
    elapsed.append((time.perf_counter() - start) * 1000)
    inference.append(result['inference_ms'])
    answers = result['answers']
    assert len(answers) == 3
    for answer in answers[:2]:
        assert abs(sum(answer['probabilities'].values()) - 1) < 1e-5
        assert all(0 <= p <= 1 for p in answer['probabilities'].values())
    assert answers[0]['choice'] in body['questions'][0]['options']
    assert 0 <= answers[1]['score'] <= 2
    assert 0 <= answers[2]['noul'] <= 1
print(json.dumps({'requests':10,'p50_ms':round(statistics.median(elapsed),2), 'max_ms':round(max(elapsed),2), 'inference_p50_ms':round(statistics.median(inference),2), 'last_response':result, 'negative_checks':'unauthenticated, empty questions, state overflow, total context overflow passed'}, indent=2))
