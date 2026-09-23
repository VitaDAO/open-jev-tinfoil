# Adapter repair and exact-plan evaluation

2026-09-24. Local synthetic experiments only. **Not a replacement-ready selector;
do not deploy or integrate.** The experimental endpoint remains disabled by
default. No Vita source, deployed service, credentials or model weights changed.

## What changed

The existing frozen Open-JEV encoder plus ridge adapter now resolves period-only
follow-ups before classification and validates the entire resolved request.
Source/count/qualifier text from the previous request survives that resolution.
A new subject drops unrelated history. Courtesy wording is normalized, metrics
and records bind to the supplied inventory, and complete dates plus explicit
latest/trend cues are handled by code. Unbound words/numerals cannot silently
disappear from an accepted plan. Only model heads that own a remaining semantic
decision gate acceptance; low confidence in unused record/date heads does not
veto a resolved argument. Model failures propagate, distinct from intentional
handoff. Historical `conditional` head-mask ablations require their old source
revision and are not a selectable policy of this repaired candidate.

`scripts/evaluate_selector_exact.py` compares complete compiled plans, reports
wrong plans separately from missed plans/handoffs, and distinguishes exceptions
and selected-but-uncompilable responses. The old `evaluate_selector.py` now uses
the exact oracle for the hash-pinned 59-case fixture. Other legacy fixture
formats explicitly retain their weaker label scoring and must not be used as
exact-plan evidence.

This remains a hybrid of explicit parsing and the existing learned adapter. It
is not a fine-tuned native `/decide` head or a new Laya comparison.

## Before and after

Both columns use the same frozen corrected contract-plan expectations. The
before source is commit `8c6b5615c2e4729557ed09a4ab7c0440b32ef913`. Baseline 59
predictions are regraded from the preserved selector-v1 output. Baseline 160
predictions were newly run and saved before changing the adapter. Their scores
were not read until the repaired candidate had also been evaluated.

| Measure | Before: 59 | Repaired: 59 | Before: 160 | Repaired: 160 |
| --- | ---: | ---: | ---: | ---: |
| Correct plan or correct handoff | 47 | **58** | 89 | **90** |
| Correct executable plans | 15 | 25 | 11 | 9 |
| Wrong executable plans | 1 | **0** | 4 | **0** |
| Correct handoffs | 32 | 33 | 78 | 81 |
| Missed required plans | 11 | 1 | 67 | 70 |
| Harness errors / invalid responses | 0 | 0 | 0 | 0 |

The 59 oracle requires 26 plans. It corrects the unavailable-weight case to
handoff and permits either an exact latest-three lab plan or handoff for m37.
The baseline's 15 correct plans include that optional m37 plan; it serves 14 of
the 26 required-plan cases. The repaired candidate serves 25/26 and hands off
m37. Its only missed required case is m05, `Analize me` (unbound wording).

The additional set requires 79 plans and 81 handoffs. Correct read coverage fell
from **11/79 to 9/79**, even as wrong accepts fell from 4 to 0. Among the repaired
70 misses, 47 fail the bounded-wording check, 21 fail date parsing, and 2 fail
follow-up parsing. The 58/59 result therefore does not generalize. This candidate
is a conservative repair diagnostic, not sufficient acceptance for replacing
the active Vita path. No tuning was performed using the additional scores.

## Evaluation provenance and limits

`fixture-provenance.json` hashes the supplied wf1 gold and the two independently
authored/relabelled 80-case sets. Author and reviewer labels compile to identical
plans for all 160 cases, so none were excluded from the consensus metric.
Some examples and policies were discussed in the supplied report before this
run; these are additional prospective cases, not claimed completely blind or
human-adjudicated. The original 59 have already guided development.

Gold plans come from the extracted legacy Vita consumer, which is also the
candidate compiler. Equality proves compatibility with those expected plans;
it does not independently prove correct database results, permissions, all
requested records reaching the answer, clinical correctness or end-to-end
quality. For example, both oracle and consumer cap lab-report details at three.
Record completeness needs independent synthetic DB fixtures and paging checks.

The supplied deliberately-wrong selector previously scored 59/59; after the
scorer repair it scores **33/59**, with 26 wrong executable plans. Its research
case still has a genuinely matching plan, hence 33 rather than 32 correct.

Raw predictions, plans, exceptions and timings are preserved in
`baseline-*.json` and `repaired-*.json`. Each repaired report pins the fixture and
all operative selector/grader/consumer source files by SHA-256. Those source
hashes were checked against the final candidate. Weights remain the existing
adapter `52e885aefec1230c71b9f68f65cca7c94b67341cce962e03674430ef828ba69c`.

## Verification and timing

- `PYTHONPATH=vendor .venv/bin/python -m pytest -q tests`: **135 passed**.
- Real CPU `Engine` exercised through the ASGI API: unauthenticated 401,
  authenticated correct selection, inherited source-filter handoff, and disabled
  selector 503 all passed. See `http-smoke.json`; this is not network/enclave
  transport verification.
- `git diff --check` passed. The container copy list includes the exact grader
  used by its mounted tests; no new image build or enclave release was run.
- Local four-thread CPU model-bearing requests: regression warm n=24 median
  **105.99 ms**, p95 **116.72 ms**; additional warm n=9 median **111.19 ms**,
  p95 **119.87 ms**. Samples are small. Most additional requests hand off before
  inference, so their ~0.18 ms whole-suite median is not model inference speed.
- Existing tokenizer-regex and Starlette/httpx deprecation warnings occurred.
  Runtime/tokenizer settings were identical for baseline and candidate; no
  dependency or tokenizer changes were mixed into this comparison.

Reproduce the candidate results:

```sh
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v2/regression-59.json --output /tmp/selector-regression.json
PYTHONPATH=vendor .venv/bin/python scripts/evaluate_selector_exact.py --cases evidence/selector-v2/additional-160.json --output /tmp/selector-additional.json
```

## Next engineering decision

The immediate remaining bottleneck is the bounded language/date binding path,
before inference, rather than merely a confidence threshold. Broader typed
argument extraction, explicit handling of unsupported constraints, and proper
semantic training need a separately frozen evaluation. Repeatedly adding words
to fit these 160 would turn them into another development suite. Preserve the
current results, retain the active Vita selector, and verify real execution
before any future acceptance or attested release.
