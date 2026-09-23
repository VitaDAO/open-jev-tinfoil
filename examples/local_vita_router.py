"""Experimental local router: isolate the record-access question from routing.

Uses the existing loopback server and its service key. Outputs are proposals,
never permission grants or authorization to read records/change settings.
"""
from pathlib import Path
import time
import httpx

QUESTIONS = (
    {'type':'choice', 'instructions':'Classify the requested action.',
     'options':['read existing data','explain a concept','change settings']},
    {'type':'noul', 'instructions':'Does the user ask to access their existing personal records?'},
    {'type':'score', 'instructions':'How urgent is this request?',
     'options':['not urgent','moderately urgent','very urgent']},
)
MODEL_REVISION = '19bf9a64815add579fbf6c907bef584d9277a8e4'


def route(client: httpx.Client, state: str) -> dict:
    """Client must target the authenticated local server; no partial success."""
    started = time.perf_counter()
    def decide(questions):
        response = client.post('/decide', json={'state': state, 'questions': questions})
        response.raise_for_status()
        data = response.json()
        if data.get('model_revision') != MODEL_REVISION or len(data.get('answers', [])) != len(questions):
            raise RuntimeError('Unexpected model response')
        return data['answers']
    answers = decide(list(QUESTIONS))
    answers[1] = decide([QUESTIONS[1]])[0]
    return {'action': answers[0]['choice'], 'record_access_probability': answers[1]['noul'],
            'urgency_score': answers[2]['score'], 'answers': answers,
            'elapsed_ms': round((time.perf_counter()-started)*1000, 2),
            'mode': 'isolated_record_check', 'model_revision': MODEL_REVISION}


def local_client() -> httpx.Client:
    key = (Path.home()/'.config/open-jev-tinfoil/api.key').read_text().strip()
    return httpx.Client(base_url='http://127.0.0.1:18080', timeout=15,
                        headers={'Authorization': 'Bearer '+key})


if __name__ == '__main__':
    import json
    with local_client() as client:
        print(json.dumps(route(client, 'Explain what REM sleep means in general.'), indent=2))
