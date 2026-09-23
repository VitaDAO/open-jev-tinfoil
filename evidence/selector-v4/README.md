# Selector v4: read intent and argument binding

Local synthetic experiment, 2026-09-24. The experimental endpoint is disabled by
default. No Vita files, live enclave, attestation pins or secrets were changed.

## Behavior

The same frozen Open-JEV encoder now uses a second version of the small read-intent
adapter. Training adds general personal-read questions, multi-target reads,
calendar wording and matched non-read/filter/action examples. Schema synonyms
are normalized for that intent decision; the full request still reaches the
separate native proposal-coverage question. The older coarse task classifier
cannot veto an explicitly bound target before these checks run. A disagreement
with the coarse classifier also prevents a grammar parse from bypassing them.

Argument binding now distinguishes date prepositions from source names, singular
latest-value references from counts, negated trend operators from positive ones,
and explicit research opt-outs from research requests. Period corrections retain
inherited filters. An explicit follow-up can replace latest with trend, or narrow
a previously named metric group to one unambiguous member. Colliding catalog
aliases hand off rather than silently choosing an inventory ID. Explicit inventory
IDs and intentional domain expansion remain distinct.

The legacy profile response projects only nine fields. An explicit full/whole/
complete-profile request now hands off with `legacy_profile_projection_incomplete`.
This corrects a limitation of the old evaluation oracle; its expected labels have
not been rewritten to increase the score. Ordinary profile-summary reads retain
the existing projection.

## Training and evaluation provenance

The 59, 160 and previous 48 cases are development/regression sets. Their observed
misses informed this work. The new 40-case set was authored and frozen before
implementation at SHA-256
`be1153ffaa4f6f689f83914b3d1bc7e87befedb700621bd458c6564288c3a419`.
It has 20 intended reads and 20 intended handoffs. As with the earlier set, the
implementing assistant authored it; it is not independent or human-adjudicated.
Its canonical-plan oracle shares the extracted legacy consumer, including two
overpermissive complete-profile labels. Raw labels and raw scores are preserved.

The first evaluation of those 40 found a real incorrect plan: "May I view..."
was interpreted as the month May. It scored 36/40, including one wrong plan;
the v3 baseline also had that wrong plan. That output is preserved in
`frozen-first-evaluation.json`. The modal/month ambiguity was then repaired;
consequently the final 40-case score is a regression score, not an untouched
holdout. Before that repair, an additional 24-case temporal set was frozen at
SHA-256 `0124a850cc3310a08c0cd645ca9a881ea782d6a2e1ffebb288cea06d9b0a8ebc`.
It includes modal questions, actual May dates, both in one sentence, calendar
follow-ups and unsupported constraints. These author-created compositions test
the repair but do not constitute independent broad generalization evidence.

The adapter was trained on 681 synthetic requests and calibrated on 36 separate
wording-family requests. The first attempt used the old largest-regularizer
tiebreaker; it regressed to 57/59. That attempt and its 127/160 result are preserved
under `iteration-1/`. Before opening the new 40-case results, selection was changed
to minimize calibration squared error among candidates with equal accepted-positive
counts. This development choice was informed by the regression failure. Final
alpha is 0.001 and threshold is 0.0. Fitted calibration: 18/18 positive accepts,
0/18 negative accepts. Those fitted results are not independent validation.

Final intent adapter SHA-256:
`45946119ca98492c1d781a76eb3f153bdba0fe89cdeb2409fb9615707a93f757`.
Original Open-JEV weights and the original multi-head selector adapter are unchanged.
No external inference or paid training was used.

## Final measured results

| Suite | v3 | v4 | Correct required reads in v4 |
| --- | ---: | ---: | ---: |
| Original regressions | 58/59 | **59/59** | 26/26 |
| Broader regressions | 124/160 | **141/160** | 60/79 |
| Previous 48, now regressions | 43/48 | **47/48** | 23/24 |
| New 40, after repairing its discovered date bug | 29/40 | **37/40** | 17/20 |
| New 24 temporal compositions, first evaluation | Not run | **24/24** | 12/12 |

No wrong plans, harness errors or invalid responses occurred in these final runs.
The earlier incorrect May-filter plan is retained in the first-evaluation evidence.
All required handoffs remained handoffs. The remaining miss in the 48-case set,
and two of the three in the 40-case set, are the intentional complete-profile
rejections described above. The other 40-case miss is "The newest oxygen saturation
one, can you pull it up?", rejected by the intent head.

- **186 unit/contract tests pass.** Real-model evaluation is separate from these
  mostly binding/contract tests.
- **14/14 synthetic record-ID execution cases pass.**
- Real loopback API/client smoke passes: authentication, latest ApoB regression,
  exact date/metric plans, semantic paraphrase, inherited source constraint,
  absent category and client identity validation. Default-disabled 503 passes
  through ASGI.
- Warm local HTTP, 10 requests per path: grammar median **107.37 ms**, p95
  **108.47 ms**; semantic median **350.93 ms**, p95 **357.11 ms**.
- Broader accepted-plan timing (59 warm accepted cases): median **397.19 ms**,
  p95 **575.00 ms**. Rejection timings are reported separately in raw JSON.
- Measured encoder calls per selected request: **one** for a complete grammar
  certificate, **three** for the semantic path. Reusing identical request features
  could reduce this, but no latency optimization was mixed into this repair.

Final source hashes, every grade/summary, frozen fixture hashes, adapter/dataset
hashes and HTTP selector identity were checked. Neither new fixture has an exact
request-text match in training/calibration; semantic-family overlap remains.
Existing tokenizer-regex and Starlette-deprecation warnings persisted. No Linux
image build, real Vita DB execution, deployed attestation, integration or release
was performed. The draft branch is also not a release candidate: fetched main
contains a separate enclave-startup fix not included in this experiment branch.

## Verification boundaries

Exact-plan evaluation hashes operative sources and weights before/after inference.
It reports accepted-plan latency separately from handoff latency and measures
backbone forward-pass counts using a hook. Local CPU results exclude WAN, enclave
startup, database reads and Vita answer generation.

`execution.json` supplements plan comparison with independently specified record
IDs from a small synthetic dataset. Its deliberately bounded fixture executor
supports explicit calendar date windows and latest-per-metric selection. It checks
UTC/Bucharest midnight boundaries, leap days, two metrics, empty/sparse windows,
new-subject resets, inherited source/count constraints and unavailable inventory.
It never widens an empty window. This is **not execution against the actual Vita
vault/database**, and it does not verify report paging, production tie-breaking,
source filtering, aggregates, broad/profile completeness or final-answer quality.

The API smoke uses the real local CPU Engine over loopback HTTP and the shipped
client's response validation. Attestation is intentionally not exercised by that
local transport. An approved release pin is still required for private use.

## Remaining limits and direction

The dictionary/index is useful for entity identity and capability boundaries; it
cannot supply missing intent or make the legacy answer map express more operations.
Remaining misses include indirect/elliptical read wording, spelling and scope
ambiguity, combined personal-data/research requests, explicit exclusions, and
complete profile requests. The old "last night" label maps to latest over all
history; accepting that would silently lose the requested window.

The next contract needs separately validated clauses with explicit fields, source,
date basis, window, ordering and count. Research clauses need to preserve the
actual general question, rather than replace it with one of three broader canned
questions. Report selection needs verified ordering/ties and paging. These are
consumer/execution contracts, not just extra aliases or a lower model threshold.
The existing `query_ir.py` is still an experimental schema/compiler, not an active
decoder or a deployed replacement. Keep unsupported requests on Vita's existing
path until these contracts and execution checks are implemented.

## Reproduction

```sh
PYTHONPATH=vendor .venv/bin/python -m pytest -q
PYTHONPATH=vendor .venv/bin/python scripts/train_read_intent.py --version 2
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v2/regression-59.json --output /tmp/v4-59.json
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v2/additional-160.json --output /tmp/v4-160.json
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v3/fresh-48.json --output /tmp/v4-48.json
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v4/frozen-40.json --output /tmp/v4-40.json
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v4/temporal-24.json --output /tmp/v4-24.json
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_execution.py
```

Training is a development action; any changed adapter hash requires deliberate
review and repinning. No runtime training or download is enabled. Recovery is to
leave the endpoint disabled or revert the standalone checkpoint; the deployed
service is unaffected.
