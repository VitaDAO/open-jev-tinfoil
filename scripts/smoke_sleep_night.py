"""Synthetic sleep contract regression against an explicit loopback serving image."""
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from scripts.evaluate_query_plan import request
from query_plan import QueryPlan, compile_batch
from query_selector import identity, INTENT_SHA256
from routing import MODEL_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    url = urlparse(args.base_url)
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port or url.path not in ('', '/'):
        raise ValueError('Synthetic smoke requires explicit loopback URL')
    rows = []
    cases = [
        ('Show my deep sleep last night', ['sleep_deep']),
        ('Show my total sleep and deep sleep last night', ['sleep_deep', 'total_sleep']),
        ('Show my total sleep last night', ['total_sleep']),
        ('How much deep sleep did I get last night?', None),
        ('Show my steps last night', None),
    ]
    with httpx.Client(timeout=30) as client:
        for prompt, expected in cases:
            req = request(prompt)
            req.available_metrics = ['total_sleep', 'sleep_efficiency', 'sleep_deep', 'steps']
            response = client.post(args.base_url.rstrip('/') + '/v1/select',
                json=req.model_dump(mode='json'),
                headers={'Authorization': 'Bearer ' + os.environ['OPEN_JEV_API_KEY']})
            response.raise_for_status()
            plan = QueryPlan.model_validate(response.json())
            assert (plan.selector_sha256, plan.adapter_sha256, plan.model_revision) == (
                identity(), INTENT_SHA256, MODEL_REVISION)
            if expected is None:
                assert plan.status == 'handoff' and not plan.queries
                if prompt.startswith('How much'):
                    assert plan.reason_codes == ['learned_read_intent_unconfirmed']
            else:
                assert plan.status == 'planned' and len(plan.queries) == 1
                query = plan.queries[0]
                assert sorted(query.metrics) == expected
                assert query.date_basis == 'sleep_end_day'
                assert query.period == {'kind': 'between', 'start_at': '2026-09-23', 'end_at': '2026-09-23'}
                assert len(compile_batch(plan, req)['health_reads']) == 1
            rows.append({'request': req.model_dump(mode='json'), 'plan': plan.model_dump(mode='json')})
        # Unavailable deep sleep cannot be silently replaced by total sleep.
        req = request('Show my deep sleep last night')
        req.available_metrics = ['total_sleep']
        response = client.post(args.base_url.rstrip('/') + '/v1/select',
            json=req.model_dump(mode='json'),
            headers={'Authorization': 'Bearer ' + os.environ['OPEN_JEV_API_KEY']})
        response.raise_for_status()
        plan = QueryPlan.model_validate(response.json())
        assert plan.status == 'handoff' and not plan.queries
        rows.append({'request': req.model_dump(mode='json'), 'plan': plan.model_dump(mode='json')})
    args.output.write_text(json.dumps({'selector_sha256': identity(), 'cases': rows}, indent=2) + '\n')
    print(json.dumps({'passed': len(rows), 'scope': 'synthetic selection only; no sleep records executed'}))


if __name__ == '__main__':
    main()
