# Single Vita query client and structured plan

Local synthetic candidate, 2026-09-24. One client (`VitaClient` in
`examples/vita_client.py`), one selector endpoint (`POST /v1/select`), one closed
plan (`vita-query-plan/v2`). This branch is not deployed or installed in Vita.
The selector remains disabled unless `ENABLE_EXPERIMENTAL_SELECTOR=1` is set.
No reviewed selector enclave release is embedded in the client.

## What changed

The old answer map could not preserve a full profile, field selection, source,
record limit, distinct windows or a specific research target. The public client
and endpoint now use typed health/research clauses directly. The old selector
client and `/v1/select-baseline` route are removed. Historical parsers, adapters
and regression artifacts remain offline references, not parallel client paths.
The same `VitaClient` also exposes the existing generic `/decide` and coarse
`/route` capabilities; they are separate tasks, not fallback selector APIs.

The frozen Open-JEV encoder is unchanged. A third 1,024-weight read-intent head
was trained on 843 synthetic requests and calibrated on 36 wording-family
examples. It binds read intent to deterministic schema, date, inventory and
constraint checks, plus the existing native proposal-coverage decision. Profile
and finite research-slot requests can bind directly without encoder inference.
This is not unrestricted natural-language SQL or a medical-answer model.

Supported clauses include full 23-field profiles with explicit missing fields,
selected profile fields, latest N record details (1–200), metric provider filters,
distinct per-clause windows, sleep metric exclusions, context corrections and
finite public research targets/interventions. Catalog aliases compile to Vita's
canonical IDs (for example `ldl_cholesterol` → `ldl`). Unknown or unsupported
clauses make the entire plan an inert handoff. Nothing is partially executed from
a rejected plan. All requests remain advisory and subject to Vita's own authority.

## API and one client

The request schema is `vita-selector/v2`, served only at `/v1/select`.
Old `vita-selector/v1` request bodies receive 422. The retired baseline route
returns 404 after authentication. Disabled selector requests receive 503.

Run from this repository in Vita's trusted backend, with the pinned client
requirements installed. The placeholders below must come from a reviewed,
attested release; never copy pins from an untrusted server response.

```python
from datetime import datetime, UTC
from query_plan import QueryRequest
from query_execution import execute_plan
from examples.vita_client import VitaClient

now = datetime.now(UTC)  # trusted turn clock; reuse through execution
request = QueryRequest(
    schema_version="vita-selector/v2",
    state={
        "current_request": "Show my latest 5 lab reports in 2025",
        "recent_user_requests": [],
        "reference_date": now.date().isoformat(),
        "time_zone": "UTC",
    },
    reference_time=now,
    available_metrics=[],
    available_record_types=["labs"],  # actual authorized inventory
    available_sources=[],
    literature_available=False,
)
client = VitaClient(
    release_digest=REVIEWED_RELEASE_DIGEST,
    selector_sha256=REVIEWED_SELECTOR_DIGEST,
    adapter_sha256=REVIEWED_INTENT_ADAPTER_DIGEST,
)
try:
    plan = client.select(request)
finally:
    client.close()

# Caller supplies this callback through Vita's own capability broker and source
# executor. It must authorize EVERY operation and record continuation, retain
# the trusted turn clock/fence, and return the unabridged source result.
delivery = await execute_plan(plan, request, run_authorized_vita_operation)
```

The client verifies Tinfoil attestation and the approved release before reading
`OPEN_JEV_API_KEY`. There is no plaintext fallback. It validates the closed plan,
model/selector/adapter identities, exact request hash, timezone, caller inventory
and eight-operation budget before returning it. The example's callback is the
integration boundary; it is not supplied with database credentials by this repo.
Vita integration has not been installed or browser-tested here.

`compile_batch()` is an acquisition plan only. Use `execute_plan()` when consuming
it, and honor `status` / `all_requested_delivered`:

- `handoff`: unsupported request; zero operations.
- `complete`: every requested health projection passed its completion checks.
- `incomplete`: failed, denied, capped, missing or unproven health evidence.
- `needs_evidence_review`: research was acquired but still needs Vita's existing
  evidence/answer validation. Successful retrieval is not proof of an answer.

The consumer uses the existing authorized Vita query executor; it has no database
connection, credentials or SQL. Record continuations preserve the same query,
source fence and limit, with an eight-page cap per read. Missing profile values
remain explicitly missing. It never widens empty date windows. Latest metric
facts are checked against raw source/time/value series and projected separately
from history and whole-window statistics.

For “last night,” a plan requests sleep episodes ending on the reference local
date. The current backend lacks an episode-end index, so acquisition is raw
all-history followed by explicit episode-timing projection. Complete coverage
and aligned, aware start/end metadata are required. Missing timing, a same-day
possible nap or multiple candidate episodes remains incomplete. Recorded-at is
never treated as proof of episode time. After-midnight starts and absent metadata
can therefore need the existing model/clarification path; no universal sleep
semantics or performance guarantee is claimed.

## Final verified results

| Check | Result | What was checked |
| --- | ---: | --- |
| Original 59 prompts, v2 contract | 59/59 | 31 exact plans, 28 correct handoffs |
| Broader 160 prompts, v2 contract | 160/160 | 81 exact plans, 79 correct handoffs |
| Previously missed 19 (subset of 160) | 19/19 | Full query semantics, not only accepted status |
| V5 development set | 48/48 | 24 plans, 24 handoffs |
| New compositions | 24/24 | 12 plans, 12 handoffs; first run also 24/24 |
| Unit/contract suite | 258 passed | Includes existing offline regressions and new client/consumer/orchestrator checks |
| Future constraint boundaries | 40/40 | Unsupported clocks, sources, filters, writes and research constraints hand off |
| Profile review cases | 16/16 | Explicit projections and recorded-read intent |
| Actual Vita executor, synthetic source rows | 16/16 | Profiles, 405-record pagination, dates, sources, latest facts, workouts, sleep and negative coverage |
| Actual Vita batch schema | 167 planned cases | Includes overlapping development suites |
| Real local HTTP through the single client | 24/24 | Complete query equality, identity, authentication and retired/invalid API rejection |

There are zero wrong plans, missed required plans or compiler failures in those
final language suites. The 59/160 oracles were explicitly migrated for the new
contract, with corrections detailed below; this is not a claim that the legacy
wire format stayed byte-identical or that every natural-language request works.

Warm loopback HTTP, four CPU threads on this M5 Max, 10 repetitions per path:

| Path | Median | p95 (nearest rank) |
| --- | ---: | ---: |
| Simple metric read | 136.1 ms | 194.0 ms |
| Semantic latest-value wording | 328.8 ms | 415.5 ms |
| Two independently scoped clauses | 269.9 ms | 321.6 ms |
| Explicit profile fields/full profile (no encoder pass) | 2.1 ms | 2.5 ms |

A paired alternating benchmark toggling request-local feature reuse only measured
semantic requests at 418.0 → 305.9 ms (27% less time, three → two encoder passes),
and repeated semantic clauses at 819.5 → 444.9 ms (46% less time, six → three
passes), with identical plans. Single-pass and direct-profile paths were roughly
unchanged. No cross-request input cache is used.

These timings include HTTP and client validation, not attestation, WAN, database
reads, research or final answer generation. They are not an enclave RAM/performance
benchmark. Raw reports: `regression-59.json`, `additional-160.json`,
`evaluation-48.json`, `remaining-19.json`, `holdout-evaluation-24.json`,
`actual-executor.json`, `http-smoke.json`; identities are recorded in `manifest.json`.

Selector SHA-256: `f9cecdb0fdf09167da7bb62c5ef8ce6891ac4480a16a9b8820c108d30365dd46`.
Intent adapter SHA-256: `31111ec06e545f0e68f7a2c7ad62cd5c1c6d554402e881a3ef0fbe85cf2d24f0`.

## Orchestrator and canonical battery coverage

`examples/vita_orchestrator.py` exposes `prepare_model_turn(client, request,
run_authorized_operation)` using the same `VitaClient`. It returns a fixed
`additional_system_instructions` block, the unchanged `user_message`, and
`selector_evidence` as structured tool data. Add the fixed block to the existing
orchestrator policy; retain its permissions, consent and evidence checks. Keep
source data out of the system instructions and send the result only through
the existing approved model path. This helper sends nothing to DeepSeek itself.

Invalid/unavailable selection and handoffs preserve the question for normal
model planning. Source failures become incomplete delivery with sanitized
reasons, preserving earlier successful operations. Cancellation propagates.
Even complete acquisition still requires the model to perform interpretation,
arithmetic, comparison and research appraisal when requested. The helper adds
no model round and does not authorize any data access.

The exact 20 canonical user prompts were run locally against the final selector
with a synthetic complete catalog of 120 metrics and the previous four user
requests. **7/20 produce plans; 13/20 hand off.** This is coverage, not answer
accuracy or a passing 20-question browser run.

| Prompt IDs | Directly planned acquisition |
| --- | --- |
| 5, 6 | Six-month sleep and seven-day HRV histories |
| 9, 10 | ApoB trend; all three requested fasting-glucose/HbA1c/hsCRP targets |
| 11, 12, 13 | Allergies/medications, goals, chronic conditions |

Counts, provider comparisons, episode alignment, medication/personalized
literature, prioritization and broad full-inventory analysis need normal model
planning. The last-lab-test date phrasing also currently hands off. These are
not silently simplified into a smaller completed task. The initial battery
probe exposed a missing hsCRP alias; that omission is fixed and unknown
metric-list items now reject the entire plan.

The helper contract is tested locally, but it is not installed in Vita and no
DeepSeek response was evaluated in this patch. A system prompt and selector
tests cannot establish that all 20 answers are correct. See [the review
assessment](REVIEW.md) for reproduced findings, repairs and review limitations.

## Evaluation provenance

All cases are synthetic and authored by the implementing assistant. These are
engineering regression results, not an independent clinical or general-language
accuracy study. The frozen encoder has not been fine-tuned and no external
inference service or private data was used.

The canonical Vita 20-prompt selector probe is a separate diagnostic, using the
existing prompt-only artifact and a synthetic 120-metric catalog inventory. It
does not read patient answers or source records. Its report stores prompt IDs,
source hash and plans, not the original prompt text. It is not a 20-answer score,
and it does not replace the unchanged sequential browser acceptance run.

- `frozen-48.json` was frozen before v5 implementation. `expanded-48.json` fills
  explicit schema defaults and supplies the complete-query oracle. Expansion was
  done after initial implementation but before the first run; it is not independent.
- Its first run had 22 correct plans, 24 correct handoffs, one miss and one wrong
  plan/compile failure. `iteration-1-*` preserves that run. Repairs made those
  cases development regressions; no claim of untouched holdout performance.
- `remaining-19-oracle.json` assigns full clause semantics to the previously
  missed requests. The first diagnostic outputs had already been inspected, so
  these are reviewed development cases, not blind labels. Counting only a
  `planned` status was replaced with exact complete-query comparison.
- `regression-59-v2.json` and `additional-160-v2.json` port the old prompts to the
  single richer contract. Profile projection is now `all`, unbounded record
  requests no longer silently mean only 3/8 records, explicit calendar dates are
  retained, and research uses enum slots. Newly supported source/field/N/window
  requests have explicit positive oracles. Old v1-v4 fixtures/results are unchanged.
- The first 59-port run reported eight list-order mismatches and one obsolete
  negative oracle for “Sleep only, no labs.” Those artifacts and the initial
  oracle are retained. Record order was standardized; that one oracle was
  corrected to the requested sleep-only plan, with no lab reads.
- `holdout-24.json` was authored before evaluating those new phrases. Its first
  run passed 24/24 and is preserved separately. The phrases were not added to
  training. They were selected by the same implementing author.
- The first broader v2 run scored 158/160: an overview tail was split as another
  read, and a calendar year-to-date range incorrectly retained an instant.
  `iteration-3-additional-160.json` preserves it. The repair keeps unbound tails
  under full intent/coverage validation and uses whole-day record boundaries.
  That exposed one porting-oracle error: bA-059 had timestamp bounds on its
  date-only lab/calendar clauses. The initial oracle and the 159/160 intermediate
  run are preserved; those two end bounds were corrected to the same local date.
  The metric clauses still end at the exact trusted reference instant.

## Reproduce

```sh
PYTHONPATH=vendor .venv/bin/python -m pytest -q
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_query_plan.py
PYTHONPATH=vendor .venv/bin/python scripts/smoke_proposal.py
PYTHONDONTWRITEBYTECODE=1 \
  VITA_SOURCE_ROOT=/path/to/verified/vita-candidate \
  /path/to/verified/vita-candidate/vita/py/.venv/bin/python \
  scripts/verify_query_execution.py
```

The HTTP smoke starts/stops only its own loopback service with a random synthetic
key, checks the running selector identity, and routes the sole client through an
explicit test transport. It does not perform or simulate successful attestation.
`--base-url http://127.0.0.1:PORT --output PATH` exercises an existing synthetic
service and is the container workflow's current selector smoke. Default output
is always in `selector-v5`; old evidence is not overwritten.

The executor integration asserts imports from the chosen actual Vita checkout,
uses that checkout's real closed batch contracts, materialization, resolver,
query executor, record pager and date logic with synthetic source rows, and records
source hashes. No live DB, provider, encrypted production records, research
provider, final-answer renderer or browser journey is exercised.

## Limits and release state

A correct plan does not establish evidence availability, clinical correctness,
end-to-end answer quality or latency. The executor explicitly reports incomplete
coverage instead of claiming all data was delivered. Requests needing unsupported
aggregation, metric latest-N, upload-time ordering, clinical filters, writes,
unknown providers/populations/research targets or too many operations hand off.
Large full-inventory overviews can exceed the eight-operation budget and hand
off; this version does not silently omit metrics to fit it.

Source code, selector identity, adapter digest, container image, signed release and
running enclave are separate identities. Local tests do not produce a deployment
or attestation approval. The draft branch still excludes the separate main-branch
startup fix and is not a release candidate. Linux image/8 GB runtime checks and
private end-to-end Vita acceptance are not established by these local results.
Recovery is to retain the disabled selector or revert this checkpoint. No live
Tinfoil, Vita, keys, grants or attestation pins were changed.
