"""Public full-catalog synthetic acceptance against an explicit loopback image."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from query_plan import QueryRequest, QueryPlan, compile_batch
from query_selector import identity, INTENT_SHA256
from routing import MODEL_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    url = urlparse(args.base_url)
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port or url.path not in ('', '/'):
        raise ValueError('Synthetic smoke requires explicit loopback URL')
    base = json.loads((ROOT / 'evidence/full-catalog/request.json').read_text())
    rows = []
    with httpx.Client(timeout=30) as client:
        for prompt in ('Summarize my health trends this week',
                       'Summarize my health trends this month', 'Analyze me.'):
            request = QueryRequest.model_validate({**base, 'state': {**base['state'], 'current_request': prompt}})
            started = time.perf_counter()
            response = client.post(args.base_url.rstrip('/') + '/v1/select',
                json=request.model_dump(mode='json'),
                headers={'Authorization': 'Bearer ' + os.environ['OPEN_JEV_API_KEY']})
            response.raise_for_status()
            plan = QueryPlan.model_validate(response.json())
            assert plan.status == 'planned', plan.reason_codes
            assert (plan.selector_sha256, plan.adapter_sha256, plan.model_revision) == (
                identity(), INTENT_SHA256, MODEL_REVISION)
            metrics = [m for q in plan.queries if q.kind == 'health' for m in q.metrics]
            assert sorted(metrics) == sorted(request.available_metrics)
            batch = compile_batch(plan, request)
            count = len(batch['required_operation_ids'])
            assert count == len(plan.queries) <= 8
            rows.append({'prompt': prompt, 'metric_count': len(metrics), 'operation_count': count,
                         'elapsed_ms': (time.perf_counter() - started) * 1000,
                         'plan': plan.model_dump(mode='json')})
    args.output.write_text(json.dumps({'selector_sha256': identity(), 'cases': rows}, indent=2) + '\n')
    print(json.dumps({'passed': len(rows), 'operation_counts': [r['operation_count'] for r in rows]}))


if __name__ == '__main__':
    main()
