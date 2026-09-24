"""Public schema index plus per-request inventory, never user record contents."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX = json.loads((ROOT / 'metadata/selector-index.v1.json').read_text())
CATALOG_BYTES = (ROOT / 'metadata/health_metrics.v1.json').read_bytes()
if hashlib.sha256(CATALOG_BYTES).hexdigest() != INDEX['catalog_sha256']:
    raise RuntimeError('Schema index/catalog mismatch')
CATALOG = json.loads(CATALOG_BYTES)['metrics']


def metric_definition(inventory_id):
    canonical = INDEX['inventory_to_catalog'].get(inventory_id, inventory_id)
    return CATALOG.get(canonical)


def metric_labels(available):
    """Bind catalog aliases to the caller's exact IDs, preserving unavailable IDs."""
    labels = {}
    for metric in available:
        definition = metric_definition(metric) or {}
        for label in [metric, metric.replace('_', ' '), definition.get('display_name', ''), *definition.get('aliases', [])]:
            if label:
                labels.setdefault(label.lower().replace('_', ' '), set()).add(metric)
    return {label: sorted(values) for label, values in labels.items()}


def describe_targets(metrics, records):
    descriptions = []
    for metric in metrics:
        definition = metric_definition(metric)
        descriptions.append(definition['display_name'] if definition else metric.replace('_', ' '))
    descriptions.extend(INDEX['records'][record]['description'] for record in records)
    return '; '.join(descriptions)
