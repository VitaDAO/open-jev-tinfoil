# Experimental Vita selector and retrieval-plan adapter

**Not approved to replace Venice. Not deployed.** `/v1/select` is disabled by default and requires `ENABLE_EXPERIMENTAL_SELECTOR=1` for synthetic experiments. The existing hosted v0.2.1 enclave has failed application health. No Vita code, settings, secrets, or deployment was changed by this work.

## Model and measured result

One frozen Open-JEV DeBERTa encoder pass feeds 11 ridge heads (task, coverage, purpose, research, period kind, calendar basis, context inheritance, and four record types). The encoder is shared with `/route` and `/decide`, including the inference lock. No per-metric model calls. Dynamic identifiers are resolved using exact names/aliases; uncertain or unrepresentable requests hand off, rather than grant permissions. Date validation/arithmetic is deterministic. Scores/margins are not calibrated probabilities.

- Source model: `19bf9a64815add579fbf6c907bef584d9277a8e4`, FP16 storage / FP32 compute.
- Learned adapter: `52e885aefec1230c71b9f68f65cca7c94b67341cce962e03674430ef828ba69c`.
- Selector implementation identity at evaluation: `94e82c8fc795100c29e08bcb21b88166a000e126ee02f945386635e63bb7dd72` (learned source + date/contract helper source + adapter pin).
- Training: 449 synthetic development examples. Per-head class balancing and ridge selected by analytic leave-one-out over development data; template siblings make this optimistic. No external teacher or personal health data.
- Frozen new synthetic evaluation: 59 cases, 45 correct dispositions, 43 fallbacks, 15 of 27 expected selectable queries accepted. Two incorrect accepts: latest respiratory rate interpreted with the wrong read purpose; last-five weight readings accepted without a representable N limit. **These block integration.** We did not retrain on these evaluation failures.
- The 59 cases are assistant-authored, not independently human-labelled. Earlier 36 cases are development evidence for the learned model. Their first grammar-only report remains historical baseline evidence.
- Full local HTTP selector + pure plan, 120 canonical metrics, 101 broad requests: process setup 3.32s; first 125.7ms; warm n=100 median107.9ms / p95119.1ms. Target <1s warm is met locally. These exclude WAN, attestation, database reads and Fable, and are not hosted CPU results. Cold process may use warm OS caches.
- Venice's reported 2.2–4.4s full-selector range came from the integration task, not a contemporaneous matched experiment here. Local speed does not compensate for lower coverage/correctness.

`model-heldout-results.json` contains category outcomes and every resulting plan from the legacy compatibility consumer. `model-heldout.json` was frozen before evaluation (SHA256 `576e34e4cd0b5c78ced289a25d2fb1db67be22eade9a21d5a30409977eca830d`).

## Wire contract and verified client

`POST /v1/select`, bearer authentication, JSON:

```json
{
  "schema_version": "vita-selector/v1",
  "available_metrics": ["total_sleep", "steps", "ldl", "weight"],
  "literature_available": true,
  "state": {
    "current_request": "Analyze me",
    "recent_user_requests": [],
    "reference_date": "2026-09-23",
    "time_zone": "Europe/Bucharest"
  }
}
```

Bounds:64KiB body,512 unique metric identifiers,1200 characters per request, at most4 previous user requests,256 tokens for the combined model state. Exceeding a bound returns422/413, never truncates. Schema rejects health records/extra fields. Current request and admitted recent user requests are intent; no retrieved text or tool output field is accepted. API auth, advisory output and transport attestation do not establish permission to read or write.

Response includes `status: selected|unsupported`, reason, the complete `answers` map consumed by Vita `read_arguments`, raw decision margins, model/adapter/selector identities, and `advisory:true`. `unsupported` uses task=other with no executable plan. The caller retains the original request and hands it to the existing capable model/tool path **once**, with no hidden retry cascade or window expansion. The endpoint being unapproved returns503 before inference.

`examples/selector_client.py` verifies Tinfoil TLS attestation and caller-supplied reviewed release/selector/adapter pins before reading/sending the API key. No approved release digest exists for this candidate. It validates answer keys and enums, model identity, adapter identity, implementation and advisory state. `from_venice_request()` explicitly maps the current request_body wire shape. `client.plan()` adds local deterministic planning under the supplied trusted clock, authorized record types and budgets. It does not run a database or authorize access.

`POST /v1/select-baseline` is an optional authenticated grammar baseline. Its outputs are rejected by the learned client's implementation/adapter checks. Its millisecond timing must not be described as Open-JEV model speed.

## Database plan and evidence coverage

`plan_adapter.build_plan(selection, available_metrics, now=..., time_zone=..., record_types=..., literature_available=..., operation_budget=8, summary_token_budget=16000)` is pure. Output schema `vita-read-plan/v1` contains:

- `batch`: existing Vita health/literature operations, accepted after `normalize_health_ranges()` by the actual `SourceBatchRequest`.
- `coverage.required`: operation, domain, concepts/records, purpose, exact range, scope origin, required summary and token allowance. Initially selected, never marked delivered.
- `coverage.deferred`: every inventory concept and record deferred by overview/operation budget or unavailable authorization.
- Catalog version/hash, trusted local reference date, advisory flag and recovery boundary.

The full canonical120-metric catalog is snapshotted under `metadata/`, with an explicit exhaustive domain policy. Canonical aliases map to current catalog IDs (e.g. ldl_cholesterol -> ldl). Unknown catalog IDs fail explicitly; the policy must be extended with a future catalog, rather than guessing a database column. The catalog itself does not supply clinical domains/priorities: the experimental policy overlay is visible and needs product/clinical review.

Undated broad overview: six domain reads, up to8 lab metrics and8 body metrics as **latest individual observations/all history**, and up to4 metrics each for sleep/activity/recovery/vitals as30-day trends. This is32 measurements rather than an exhaustive120-metric scan. Profile and literature consume the other two slots. Remaining88 metrics plus workouts/lab-report/calendar record reads are explicitly deferred (91 entries). Latest all-history may still incur source scan work; no database latency claim is made. Narrow reads preserve all selected concepts or return budget-unsupported. Explicit windows override overview defaults; no automatic three-month sleep expansion or older-body fallback for an explicit window. Period means are never used as latest raw observations. This policy is not permission or a clinical prioritization oracle.

`audit_delivery()` inspects the **final model-visible** operation payloads, not the retained full result. It checks query scope, purpose, per-concept presence, raw-vs-grouped latest semantics, units/source, stale last-known dates, scan completeness, paging, missing admission, errors and complete-empty intervals. Retained-but-unshown concepts remain not_delivered. Source rows remain separate. It does not deduplicate observations, join datasets or perform source arithmetic.

`project_required_summaries()` is a proposed Vita-side admission function, **not installed in Vita**. It uses the existing `summary_evidence()` output, preserves values/units/timestamps/source scope/provenance, drops unnecessary aggregate statistics from latest-observation presentation, and budgets a round-robin first projection across operations. It returns the exact candidate visible bundle plus audit. It refuses insufficient budgets and never counts omitted entries as complete. Records require the existing authorized field projection. This must be integrated with source retention/evidence binding in Vita before it can affect answers.

## Actual contract/projection verification

`verify_vita_plan_contract.py` imports actual read-only modules from `vita-agent-deepseek-v32` (path asserted, source SHA256s recorded), validates the plan, calls the current result pager and final serializer, and applies the proposed compact projection. Synthetic fixtures contain120 metrics, two source rows per selected metric, sparse old body/latest readings, explicit calendar windows, partial pages, wrong grouped-latest data, errors, complete empty results and canonical aliases.

Current paged projection gets evidence from all six domains but misses some selected metrics. The proposed projection delivers all32 selected concepts (64 synthetic source-series) plus profile within the16k total budget, including a2k literature reservation. `all_requested_delivered` remainsfalse because91 items are explicitly deferred. See current measured `vita-plan-projection.json` for exact token count and source hashes.

This verifies **typed planning and actual model-summary projection**, not execution against a database/provider. Duplicate-upload/session dedup, delayed-import timestamp filtering, authorization enforcement, source aggregates, actual row counts, real scans, and end-to-end Fable behavior remain unverified here; use existing Vita resolver/session tests and integration acceptance. Do not report zero follow-up model calls from this fixture.

## Remaining blockers and next work

1. Learned intent/temporal/record coverage and two false accepts fail the replacement gate. More representative, independently labelled supported-query training/evaluation is needed; preserve this frozen set as evidence if it becomes development data.
2. Current selector does not represent source filters, N-limits, custom aggregation, comparisons/correlations, arbitrary research, recall or writes. It must hand those to existing supported tools, not remove them. See `query-coverage.md`.
3. Vita-side summary admission integration, resolver/database oracle fixtures and real acceptance remain with the integration task. The standalone repo cannot make retained data visible merely by selecting IDs.
4. Hosted v0.2.1 is failed. A read-only-root/tmpfs reproduction is prepared in CI; no unverified runtime diagnosis, new enclave release, pin weakening or deployment is claimed.
5. No approved attested selector release exists. Default-disabled endpoint, draft status and exact pin requirements remain until quality and hosted checks pass.
