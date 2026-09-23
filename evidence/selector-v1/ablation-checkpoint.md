# Development ablations after the original evaluation

The original59-case report at5f1fa06 is preserved. Alex requested further work after inspecting its failures; that set is now diagnostic development evidence for these changes, not an independent final validation set.

## 1. Conditional head relevance

Same449-example learned weights, same .10 margin threshold, same cached model predictions:

| Decision policy | Correct dispositions | Accepted expected-select cases | Incorrect accepts | Fallbacks |
|---|---:|---:|---:|---:|
| All11heads |45/59|15/27|2|43|
| Conditional dependency mask |44/59|15/27|3|42|

Only m51 changed: a non-English request became incorrectly accepted. **Reject conditional mask; default remains all_heads.** This was a hypothesis, not a proven optimization. The implementation now returns bounded reason_codes, required/rejected heads and predicted decision diagnostics. Thresholds are not calibrated probabilities. The model is still not integration-ready.

The baseline's exact accepted disposition accuracy is14/16, not45/59 database answer accuracy. At43/59 fallback, a rough sequential latency estimate is selector latency +0.729*fallback-selector latency, excluding DB/Fable. With observed Linux selector727ms and the integration task's Venice2.2–4.4s range, this gives about2.33–3.93s. This is an assumption-based estimate, not a matched end-to-end measurement, and ignores the unacceptable false accepts.

## 2. Typed query representation

`query_ir.py` defines a bounded8-clause closed IR, including operation, metrics/domains/exclusions, sources, exact temporal selection, aggregate/grouping intent, sort/N intent, record/profile fields, calendar timestamp basis, context reference, literature, memory, conversation and action/no-read handoff.

`compile_ir()` emits only existing tool fields. Provider filters and exclusions are valid operations, not intrinsically unsafe keywords. Separate periods/providers can form parallel reads. LatestN, upload-date filtering and unimplemented sums/medians are explicitly rejected rather than downgraded. Source authorization remains in Vita; enabled capability and available inventory inputs are not new permissions. New explicit metric IDs in a trusted available inventory can pass through without invented domain metadata; domain expansion requires metadata.

This is schema/compiler coverage, **not evidence that the current learned model emits this IR**. Date span extraction, richer binding and an IR decoder still need work. The existing experimental endpoint remains legacy-compatible and disabled by default. Independent fresh acceptance has not been rerun after these changes.
