# CPU release acceptance

This is the release candidate for the single `/v1/select` API and
`vita-query-plan/v2` contract. It proposes bounded acquisition plans; Vita's
native manager, capability broker, evidence registry and answering model remain
responsible for execution and answers. A handoff preserves the original request.

## Model decision

Tev1-4B at its original BF16 language-weight precision was tested without integer
quantization on this Mac. Three adapter iterations produced 339/366, 356/406 and
398/406 exact-plan-or-handoff matches. The last still proposed seven incorrect
plans and missed one supported plan. Its 64/64 result on the earlier, smaller
finite-option test did not predict full-adapter correctness. The saved v3 report
contains every remaining failure and exact model/runtime hashes. BF16 is not FP32. A separate CPU probe (four cases, two option orders each)
passed 8/8 but took 12.98 s median / 17.98 s p95 and reached 8.53 GiB sampled
RSS. This is Mac CPU feasibility evidence, not a Tinfoil AMD benchmark; it
misses the one-second optional-selector budget.

The existing OpenJEV encoder and learned Vita adapters passed all 366 previous
fixture entries. Their first evaluation on a separately authored 40-case
challenge passed 38/40, with two safe handoffs. General date-year and discourse
parsing fixes now pass 406/406: 193 exact plans and 213 correct handoffs. The 19-case
suite overlaps the 160-case suite, leaving 387 unique request/history pairs.
A further independently authored 24-case challenge initially passed 22/24,
with two safe handoffs and no wrong plans. Those expose provider-adjective
binding ("Garmin steps") and a historical-year follow-up. Preserve this first
attempt; the repaired 24/24 result is regression evidence. The HTTP release
gate now covers all 430 entries (411 unique request/history pairs).
No existing oracle was relabeled. The challenge is now regression evidence,
not unseen validation. Full Vita answers require separate browser acceptance.

Mac model jobs overlapped for part of this comparison; the local report's timing
is not an isolated speed comparison or a Tinfoil latency guarantee. Exact Linux
HTTP and live enclave results are the release gates.

## Changes and checks

- Shared compiler splits health concepts at Vita's six-concept schema boundary
  and counts those real operations against the eight-operation budget.
- Native adapter proposes only a first eligible `acquire_sources` call. The
  original provider handles later rounds, unrepresentable requests and failures.
  Trusted inventory, turn clock/timezone, permission checks and current tool
  schema remain authoritative. Default selector wait is one second.
- Calendar relative windows freeze to local date-only bounds, preserving whole
  days, current-day future events and DST semantics. Metric windows retain instants.
- The Python client checks the approved release digest before its initial TLS
  transport and before any SDK transport rebuild/retry. The actual pinned SDK's
  same-release and different-release rotation paths are tested.
- 316 standalone tests pass; native compatibility is checked separately against
  the actual v32 source. Its quiet 19/19 run passed without changing the 20-second
  deadline. An earlier native run timed out; its cause was not established and
  the first-attempt evidence remains in the local experiment directory.

## Release and recovery

The existing dedicated service is 4 CPU / 8 GB with no GPU, outbound network, debug,
SSH or automatic updates. Keep the existing `OPEN_JEV_API_KEY` secret binding.
No real personal or health records are used in these fixtures.

Before release, build and test the exact Linux image, run all fixture entries
through its HTTP API, pin its immutable digest, publish measured configuration,
and verify live attestation and authenticated positive/negative calls. Record
source commit, image digest, configuration commit/tag, measurement and API pins
separately. This document alone does not claim those steps have completed.

Rollback requires both the previous v0.2.2 enclave tag and GitHub's latest-release
pointer to match that release: the SDK verifies against the latest published
release. Switching only the container tag is insufficient. Its approved digest
is `b0185c159a2ec83c771695a32984414a9a85d7dbe91332e58b70ed00f5579946`.
Remove the optional native wrapper to restore Vita's original planning loop.
