"""Verify attestation before synthetic inference, and prove a wrong pin fails."""
import json
import os
import statistics
import time
from pathlib import Path
from tinfoil import SecureClient

host = 'open-jev.vitality-now.containers.tinfoil.dev'
repo = 'VitaDAO/open-jev-tinfoil'
key = os.environ['OPEN_JEV_API_KEY']
body = {'state': 'I was charged twice and want my money back.', 'questions': [
    {'type': 'choice', 'instructions': 'Which team should handle this?', 'options': ['billing', 'technical support', 'sales']},
    {'type': 'score', 'instructions': 'How positive is the sentiment?', 'options': ['negative', 'neutral', 'positive']},
    {'type': 'noul', 'instructions': 'The customer wants a refund.'}]}
results = {}
for mode in ['tls', 'ehbp']:
    verifier = SecureClient(enclave=host, repo=repo, transport=mode)
    started = time.perf_counter()
    with verifier.make_secure_http_client() as client:
        verification_ms = (time.perf_counter() - started) * 1000
        health = client.get(f'https://{host}/health', timeout=30)
        health.raise_for_status()
        assert client.post(f'https://{host}/decide', json=body, timeout=30).status_code == 401
        headers = {'Authorization': 'Bearer ' + key}
        for invalid in [{'state':'test','questions':[]}, {**body,'state':'word '*300}, {**body,'questions':[{'type':'noul','instructions':'word '*30}]*32}]:
            assert client.post(f'https://{host}/decide', json=invalid, headers=headers, timeout=30).status_code == 422
        durations = []
        inferences = []
        for _ in range(10):
            start = time.perf_counter()
            response = client.post(f'https://{host}/decide', json=body, headers=headers, timeout=30)
            response.raise_for_status()
            durations.append((time.perf_counter()-start)*1000)
            data = response.json()
            assert data['model_revision'] == '19bf9a64815add579fbf6c907bef584d9277a8e4'
            assert len(data['answers']) == 3
            for answer in data['answers'][:2]:
                assert abs(sum(answer['probabilities'].values())-1) < 1e-5
                assert all(0 <= p <= 1 for p in answer['probabilities'].values())
            assert 0 <= data['answers'][1]['score'] <= 2
            assert 0 <= data['answers'][2]['noul'] <= 1
            inferences.append(data['inference_ms'])
        boundary = client.post(f'https://{host}/decide', json={'state':'word '*256,'questions':[{'type':'noul','instructions':'word '*240}]},headers=headers,timeout=30)
        boundary.raise_for_status()
        results[mode] = {'health':health.json(),'verification_ms':round(verification_ms,2),'requests':10,'p50_ms':round(statistics.median(durations),2),'max_ms':round(max(durations),2),'inference_p50_ms':round(statistics.median(inferences),2),'boundary_inference_ms':boundary.json()['inference_ms'],'last_response':data,'verification':verifier.get_verification_document().to_dict()}
# No application key or body is supplied to this deliberately wrong workload pin.
wrong = SecureClient(enclave=host, measurement={'snp_measurement':'0'*96}, transport='tls')
try:
    with wrong.make_secure_http_client():
        pass
except Exception as exc:
    results['wrong_measurement_rejected'] = type(exc).__name__
else:
    raise AssertionError('Wrong measurement accepted')
Path('evidence/live-verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a not in ['last_response','verification']} if isinstance(v,dict) else v for k,v in results.items()},indent=2))
