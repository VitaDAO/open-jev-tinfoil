"""Opt-in Open-JEV acquisition inside Vita's native manager loop.

This module runs in the Vita backend, with its existing jsonschema dependency.
It replaces only a representable first planning round, never source execution,
the capability broker, the answering provider or terminal answer validation.
All native instructions, admitted inventory, tools and history stay intact.
"""
import asyncio
import hashlib
import json
import re
import time
from copy import deepcopy
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx

from query_plan import QueryRequest, QueryPlan, HealthRead, compile_batch, MAX_METRICS_PER_READ


class _Ineligible(ValueError):
    """Only locally generated, fixed eligibility codes may enter receipts."""


def _prior_user_requests(messages):
    """Read native admitted recall, never arbitrary tool text or assistant prose."""
    prior, seen = [], set()
    index = 0
    while index < len(messages):
        message = messages[index]
        content = message.get('content', [])
        recall = [b for b in content if b.get('type') == 'tool-call'
                  and b.get('name') == 'vita_prior_user_context']
        if recall:
            block = recall[0]
            call_id = block.get('id')
            source = message.get('source', {})
            if (len(content) != 1 or message.get('role') != 'assistant'
                    or source.get('kind') != 'model' or source.get('provider') != 'vita-admitted'
                    or not isinstance(call_id, str)
                    or not re.fullmatch(r'ctx_prior_user_[0-9a-f]{24}', call_id)
                    or call_id in seen or block.get('arguments') != '{}'
                    or index + 1 >= len(messages)):
                raise _Ineligible('selector_history_untrusted')
            result = messages[index + 1]
            outputs = result.get('content', [])
            if (result.get('role') != 'user'
                    or result.get('source') != {'kind':'tool', 'callId':call_id}
                    or len(outputs) != 1 or outputs[0].get('type') != 'tool-result'
                    or outputs[0].get('toolCallId') != call_id):
                raise _Ineligible('selector_history_untrusted')
            values = outputs[0].get('content', [])
            try:
                if len(values) != 1 or values[0].get('type') != 'text':
                    raise ValueError()
                payload = json.loads(values[0]['text'])
                if (not isinstance(payload, dict) or set(payload) != {'schema','message'}
                        or payload['schema'] != 'vita-prior-user-context/v1'):
                    raise ValueError()
                text = payload['message']
                if isinstance(text, list):
                    if not text or any(not isinstance(b, dict) or b.get('type') != 'input_text'
                                       or not isinstance(b.get('text'), str) for b in text):
                        raise ValueError()
                    text = '\n'.join(b['text'] for b in text)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                raise _Ineligible('selector_history_untrusted') from None
            prior.append(text.strip())
            seen.add(call_id)
            index += 2
            continue
        if (message.get('role') == 'user' and len(content) == 1
                and content[0].get('type') == 'text'):
            prior.append(content[0]['text'].strip())
        index += 1
    return prior


def native_batch(plan, request, tool_schema):
    """Validate against THIS turn's model-facing contract before proposing IO.

    A schema-present metric is an allowed query, not proof of a populated value.
    The native scheduler independently authorizes each read and registers evidence.
    """
    from jsonschema import Draft202012Validator, ValidationError, SchemaError
    from vita_agent.kernel.health_range_input import normalize_health_ranges
    from vita_agent.health.query_tool import _build_range
    from vita_agent.health.metric_registry import canonical_authorized_metric_ids

    plan = QueryPlan.model_validate(plan)
    batch = compile_batch(plan, request)
    if batch is None:
        raise ValueError('selector_handoff')
    # These standalone projections depend on a post-read consumer. Do not
    # substitute a raw native read for that consumer or relabel research basis.
    if any(not isinstance(q, HealthRead) or q.date_basis == 'sleep_end_day' for q in plan.queries):
        raise ValueError('native_projection_requires_planner')
    batch = {key: value for key, value in batch.items() if value}
    # Resolve aliases with the live Vita registry, rather than assuming that
    # the standalone catalog's spellings match this runtime's authority enum.
    chunks = [q.metrics[i:i + MAX_METRICS_PER_READ] for q in plan.queries
              for i in range(0, max(1, len(q.metrics)), MAX_METRICS_PER_READ)]
    for op, metrics in zip(batch['health_reads'], chunks, strict=True):
        op['concepts'] = list(canonical_authorized_metric_ids(metrics))
    # Bind relative ranges once, with Vita's own arithmetic and the trusted
    # selector turn clock. This prevents midnight/elapsed-time scope drift.
    for op in batch['health_reads']:
        if op['range']['kind'] not in ('relative', 'calendar'):
            continue
        normalized = normalize_health_ranges({'health_reads': [op]})['health_reads'][0]
        window = _build_range(range_mode=normalized['range_mode'],
            start_at=None, end_at=None, history_days=normalized.get('history_days'),
            history_months=normalized.get('history_months'), now=request.reference_time,
            time_zone=request.state.time_zone)
        start, end = window.start_at, window.end_at
        if op['record_types'] == ['calendar']:
            # Native relative/calendar record reads cover whole local days.
            # Date-only bounds preserve that expansion after freezing the clock.
            zone = ZoneInfo(request.state.time_zone)
            start, end = start.astimezone(zone).date(), end.astimezone(zone).date()
        op['range'] = {'kind': 'between', 'start_at': start.isoformat(),
                       'end_at': end.isoformat()}
    try:
        Draft202012Validator.check_schema(tool_schema)
        Draft202012Validator(tool_schema).validate(batch)
    except (ValidationError, SchemaError):
        raise ValueError('native_contract_mismatch') from None
    return batch


class NativeSelectorProvider:
    """One instance per admitted turn, wrapping the existing VitaProvider.

    `request` must come from the trusted admitted clock/inventory, never the
    browser's claimed grants or a global list of every possible metric. The
    live native tool schema is the final proposal contract, not this inventory.
    `authorize` is manager.check_disclosure. Its failures propagate.

    On unsupported questions or selector failure, forward the EXACT original
    envelope, including app tools, coverage, grant states and conversations.
    Only the first ordinary user-input round is eligible; never compaction,
    continuation, an attachment turn, a forced tool choice, or a resumed read.
    """
    def __init__(self, client, request, provider, *, authorize, timeout_seconds=1.0):
        self.client = client
        try:
            self.request = QueryRequest.model_validate(deepcopy(request))
        except (ValueError, TypeError):
            # Long or otherwise unrepresentable user requests still reach Vita.
            self.request = None
        self.provider = provider
        self.authorize = authorize
        if not 0 < timeout_seconds <= 5:
            raise ValueError('selector_timeout_out_of_bounds')
        self.timeout_seconds = timeout_seconds
        self.attempted = False
        # Safe counters/codes only. No prompts, identifiers, evidence or errors.
        self.receipt = {'status': 'not_attempted', 'selector_ms': 0.0,
                        'native_schema_sha256': None, 'provider_calls': 0, 'reason_code': None}

    def _schema(self, envelope):
        if self.request is None:
            raise _Ineligible('selector_request_unrepresentable')
        request = envelope['request']
        if envelope['controls'].get('toolChoice', 'auto') not in ('auto', 'required'):
            raise _Ineligible('native_tool_choice')
        messages = request['messages']
        if not messages or messages[-1]['role'] != 'user':
            raise _Ineligible('not_current_user_round')
        content = messages[-1]['content']
        if content != [{'type': 'text', 'text': self.request.state.current_request}]:
            raise _Ineligible('original_request_mismatch')
        prior = _prior_user_requests(messages[:-1])
        recent = self.request.state.recent_user_requests
        if recent and prior[-len(recent):] != recent:
            raise _Ineligible('selector_history_mismatch')
        tools = [tool for tool in request.get('tools', []) if tool['name'] == 'acquire_sources']
        if len(tools) != 1:
            raise _Ineligible('native_reader_unavailable')
        schema = tools[0]['parameters']
        self.receipt['native_schema_sha256'] = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return schema

    async def __call__(self, envelope):
        await self.authorize()
        batch = None
        if not self.attempted:
            self.attempted = True
            started = time.monotonic()
            stage = 'native_envelope_invalid'
            try:
                schema = self._schema(envelope)
                stage = 'selector_failed'
                plan = await asyncio.wait_for(
                    asyncio.to_thread(self.client.select, self.request), self.timeout_seconds)
                stage = 'native_projection_rejected'
                batch = native_batch(plan, self.request, schema)
            except _Ineligible as exc:
                self.receipt.update(status='native_planner', reason_code=exc.args[0])
            except TimeoutError:
                self.receipt.update(status='native_planner', reason_code='selector_timeout')
            except (httpx.HTTPError, RuntimeError, ValueError, KeyError, TypeError):
                # Never leak provider exceptions or interpret failure as no data.
                self.receipt.update(status='native_planner', reason_code=stage)
            except ImportError:
                self.receipt.update(status='native_dependency_unavailable', reason_code='native_dependency_unavailable')
            else:
                self.receipt['status'] = 'native_acquisition_proposed'
            finally:
                self.receipt['selector_ms'] = round((time.monotonic() - started) * 1000, 3)
        # Re-check after selection; a revoked lease must not produce a proposal.
        await self.authorize()
        if batch is not None:
            block = {'type': 'tool-call', 'id': 'jev-' + uuid4().hex,
                     'name': 'acquire_sources', 'arguments': json.dumps(batch)}
            yield {'type': 'block-start', 'index': 0, 'blockType': 'tool-call'}
            yield {'type': 'tool-call-delta', 'index': 0, 'id': block['id'],
                   'name': block['name'], 'argumentsDelta': block['arguments']}
            yield {'type': 'block-end', 'index': 0, 'block': block}
            yield {'type': 'finish', 'reason': {'kind': 'tool-calls'}}
            return
        self.receipt['provider_calls'] += 1
        stream = self.provider(envelope)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()
