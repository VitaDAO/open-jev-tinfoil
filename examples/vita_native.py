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


def _current_question_index(messages):
    # SessionInputPolicy preserves admitted provenance when projecting these
    # synthetic pairs. A name in arbitrary model/user text is not sufficient.
    users = [i for i, m in enumerate(messages) if m.get('role') == 'user'
             and m.get('content') and m['content'][0].get('type') == 'text']
    if not users:
        raise _Ineligible('not_current_user_round')
    index = users[-1]
    if (any(m.get('source', {}).get('form') == 'attachment' for m in messages)
            or any(b.get('name') == 'vita_attachment_slice'
           for m in messages for b in m.get('content', []))):
        raise _Ineligible('attachments_present')
    suffix = messages[index + 1:]
    allowed = {'vita_skc': ('skc', 'vita-skc/v1'),
               'vita_profile_planning': ('profile_planning', 'vita-profile-planning/v1')}
    if len(suffix) % 2 or len(suffix) > 4:
        raise _Ineligible('not_current_user_round')
    seen = set()
    for call, result in zip(suffix[::2], suffix[1::2]):
        blocks, outputs = call.get('content', []), result.get('content', [])
        if len(blocks) != 1 or len(outputs) != 1:
            raise _Ineligible('untrusted_bootstrap')
        block, output = blocks[0], outputs[0]
        name, call_id = block.get('name'), block.get('id')
        if name not in allowed or name in seen:
            raise _Ineligible('not_current_user_round')
        kind, schema = allowed[name]
        source = call.get('source', {})
        if (call.get('role') != 'assistant' or block.get('type') != 'tool-call'
                or source.get('kind') != 'model' or source.get('provider') != 'vita-admitted'
                or not isinstance(call_id, str)
                or not re.fullmatch('ctx_' + kind + '_[0-9a-f]{28}', call_id)
                or block.get('arguments') != '{}'
                or result.get('role') != 'user'
                or result.get('source') != {'kind': 'tool', 'callId': call_id}
                or output.get('type') != 'tool-result' or output.get('toolCallId') != call_id):
            raise _Ineligible('untrusted_bootstrap')
        content = output.get('content', [])
        if len(content) != 1 or content[0].get('type') != 'text':
            raise _Ineligible('untrusted_bootstrap')
        try:
            payload = json.loads(content[0]['text'])
        except (ValueError, TypeError, KeyError):
            raise _Ineligible('untrusted_bootstrap') from None
        if not isinstance(payload, dict) or payload.get('schema') != schema:
            raise _Ineligible('untrusted_bootstrap')
        seen.add(name)
    return index


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
        index = _current_question_index(messages)
        content = messages[index]['content']
        if content != [{'type': 'text', 'text': self.request.state.current_request}]:
            raise _Ineligible('original_request_mismatch')
        prior = [m['content'][0]['text'] for m in messages[:index]
                 if m['role'] == 'user' and len(m['content']) == 1
                 and m['content'][0].get('type') == 'text']
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
