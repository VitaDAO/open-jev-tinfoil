# Structured proposals, schema index and request inventory

Local synthetic experiment, 2026-09-24. This candidate remains disabled by
default and is not approved for deployment or Vita integration.

## What changed

The experimental `/v1/select` now builds a typed proposal before producing the
existing `vita-selector/v1` answer map. Every request within the 256-token state
limit reaches the frozen Open-JEV encoder. There is no word whitelist ahead of
inference. Explicit entities, operations, dates and unsupported constraints are
bound separately. Simple requests that match the complete existing grammar can
be validated deterministically; other phrasing passes a learned read-intent gate
and a native Open-JEV proposal-coverage question. Neither decision is a permission
check or a calibrated guarantee of correctness.

The schema index combines the existing 120-metric catalog with record category
aliases, fields, date meanings and adapter limits. For example, uploaded lab
report metadata is separate from numeric ApoB measurements; lab dates are exam
dates, workout dates are session start times, and calendar due/completion dates
are separate fields. The index distinguishes capabilities of the locally
inspected backend from those expressible by this legacy adapter. The backend's
source filters, profile projections and paging do not become adapter capabilities
merely by listing them.

`available_metrics` remains the per-request measurement inventory. The optional
`available_record_types` adds the corresponding record inventory. Supply it
explicitly; its default of all four record types preserves old clients and is
not evidence of permission or data presence. A request for an absent category is
handed off. A broad request with a partial record inventory is also handed off,
because the old consumer's broad flag could otherwise re-enable missing records.
The caller still owns authorization and actual data access. Inventory contains
identifiers only, never record values or decryption keys.

Catalog aliases map to the caller's exact metric IDs (including legacy LDL and
oxygen identifiers). A recognized but unavailable metric cannot disappear from
a mixed request. Named/ISO date ranges, dates anywhere in a sentence, spelled
rolling durations and unambiguous period or subject corrections are supported.
Inherited provider filters survive follow-up resolution. Unsupported source,
count, unit, exclusion, calculation and write constraints remain handoffs.

A new 1,024-weight read-intent head uses the frozen encoder. Its 454 synthetic
training requests and 24 separate calibration requests are committed here.
Regularization and threshold are selected using calibration families only:
11/12 positive calibration requests accepted, 0/12 negative requests accepted.
These are fitted calibration results, not independent validation. Original
encoder/native-head weights and the original selector adapter are unchanged.
The additional intent adapter is hash-pinned at
`aa67ad90ebfe34be36d9c4799f665ccb7df71d2bd50e4dde184c2e19ba97539c`.

## Evaluation and provenance

The prior 59 and 160 are now explicitly development/regression sets. The fresh
48 were authored and frozen before implementation at SHA-256
`af5f63d217d95519ec33e5daf8fcf9dcb98e21130bf79ae6824a29fd9051b4e1`.
They contain 24 required plans and 24 required handoffs. Their labels use manually
chosen canonical equivalents rendered through the existing extracted consumer.
The implementing assistant authored this set, so it is not independent or
human-adjudicated. No exact fresh request text occurs in the training set.
Do not confuse that with absence of semantic/template overlap.

Exact compiled-plan equality is the metric. The oracle and candidate compiler
share the extracted legacy consumer; this does not establish database execution,
record completeness or answer quality. Existing fixed record limits and profile
field subsets remain limitations. The fresh baseline uses the previous bounded
selector after an equivalent encoder-method extraction. The final evaluator
hashes sources before inference and refuses a run if sources change during it.

The category index is a source-code snapshot, not a live database scan. Field
names were read from local `vita-agent-deepseek-v32` source at checkout HEAD
`8ffcdde225e7e0296038078c1679a999699cb9fb`. `query_contracts.py` and
`query_tool.py` had pre-existing local modifications; the index records their
actual file hashes plus `full_session_sources.py`, rather than claiming the
checkout commit alone identifies them. No files in that repository were changed.
Backend capabilities were not verified in a deployed environment.

## Measured results

| Exact plan or correct handoff | Previous bounded adapter | Revised adapter |
| --- | ---: | ---: |
| Original 59 regressions | 58/59 | 58/59 |
| Broader 160 regressions | 90/160 | 124/160 |
| Fresh 48, first evaluation | 29/48 | 43/48 |

No wrong executable plans, harness errors or invalid responses were observed in
these final runs. Required-read coverage is more informative than total accuracy:
25/26 to 25/26 on the original set, **9/79 to 43/79** on the broader set, and
**5/24 to 19/24** on the fresh set. This is improved coverage, not replacement
acceptance. Sources were frozen for the final runs; no code, thresholds, weights
or expected labels changed after seeing the fresh results.

The original remaining miss is m10, "What is the most recent ApoB value I have?",
rejected by the learned intent head. The old miss, "Analize me", is now accepted.
The five fresh misses are:

- "the newest one" is conservatively treated as an unbound quantity;
- a harmless "from the" left around a duration is treated as an unresolved source;
- "Bring up my complete profile, please" fails the learned intent gate;
- "Tell me which health plans are due next week" fails that gate;
- "Throughout last week, what were my steps like?" also fails that gate.

Among 36 broader missed reads, ten fail the learned intent gate, six lack entity
binding, six need combined personal-data/research binding, and the rest concern
follow-ups, dates, source wording, conflicting operators or confidence. The
schema index improves entity/capability binding; it does not solve these remaining
language and legacy-contract limitations.

## Reproduction

```sh
PYTHONPATH=vendor .venv/bin/python -m pytest -q tests
PYTHONPATH=vendor .venv/bin/python scripts/train_read_intent.py
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v2/regression-59.json --output /tmp/proposal-59.json
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v2/additional-160.json --output /tmp/proposal-160.json
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v3/fresh-48.json --output /tmp/proposal-fresh.json
```

`--model ridge` selects the previous bounded adapter; `--model proposal` selects
the diagnostic native-intent variant. The default `trained` matches the new
experimental endpoint. Retraining is a local development action; output hash
changes require review and a new pin, never a silent runtime update.

## API

Authenticated `POST /v1/select`, with `ENABLE_EXPERIMENTAL_SELECTOR=1` only for
local evaluation. It is disabled by default. Example synthetic body:

```json
{
  "schema_version": "vita-selector/v1",
  "available_metrics": ["steps", "respiratory_rate"],
  "available_record_types": ["workouts"],
  "literature_available": false,
  "state": {
    "current_request": "Show my breathing rate yesterday",
    "recent_user_requests": [],
    "reference_date": "2026-09-23",
    "time_zone": "UTC"
  }
}
```

Responses retain `selected`/`unsupported` and the legacy answer map, with proposal
and gate diagnostics added. `unsupported` always returns inert answers. The
client now expects `open_jev_structured_proposal_experimental` and still requires
explicit reviewed release, selector and adapter pins. No approved new release
pin exists; the selector identity covers binding code, catalog/index and the
new intent adapter. This candidate does not change the live enclave release.

## Verification and timing

- 168 unit/contract tests pass, including unavailable inventory, alias identity,
  date boundaries, inherited constraints, invalid model output and inert handoff.
- Real CPU Engine over loopback HTTP: auth 401, exact latest/date plan, semantic
  paraphrase plan, inherited-provider handoff, unavailable-category handoff and
  the updated client response checks pass. Default-disabled 503 was checked
  through ASGI. See `http-smoke.json` and `scripts/smoke_proposal.py`.
- Warm local HTTP, 10 requests per path: complete-grammar median **101.66 ms**,
  p95 **104.20 ms**; semantic proposal median **341.81 ms**, p95 **344.81 ms**.
  These small samples exclude cold startup, WAN and enclave overhead. The
  semantic path currently needs three encoder passes; it is slower than the
  earlier narrow classifier. Do not use the mixed-suite median as read latency.
- Fresh accepted reads, 19 cases: median **358.70 ms**, p95 **379.27 ms**.
- The test server was stopped after verification. No Linux image build, database
  execution, external inference, enclave deployment or new attestation was run.
- Existing tokenizer-regex and Starlette/httpx warnings persist; no tokenizer,
  dependency or pretrained model changes were introduced.

Quality remains insufficient to replace Vita's current selector. The next
contract work is to carry explicit source, projection, count and clause bindings
to the backend capabilities already identified in the index. It must be assessed
with a versioned consumer and new exact-plan expectations, not by changing these
old labels to count unsupported requests as successes.
