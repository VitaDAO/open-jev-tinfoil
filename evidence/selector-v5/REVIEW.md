# Robustness and latency review — 2026-09-24

Scope: the owned single-client v5 patch over
`29a6e33e793a946505c198dd3358ac5169810712` in `codex/verified-route-v021`.
Seven pre-existing experimental files were excluded and left untouched. Two
`codex review --uncommitted` runs inspected isolated snapshots of the owned patch.
`review-source-manifest.json` binds the second snapshot's source files. These
were reviews of local changes, not the deployed enclave or Vita runtime.

## Findings verified and fixed

| Finding | Reproduction | Repair |
| --- | --- | --- |
| Full-profile wording overrides a narrow projection | “Show only my allergies from my full profile” requested all fields | Explicit restrictions keep the named fields |
| Clinical permission mistaken for a recorded-data read | “Can I take medications?” produced a medications read | Closed read constructions distinguish recorded lists from treatment permission |
| Later page conceals missing source coverage | A continuation with a top-level gap and a complete nested page claimed completion | Every continuation rechecks gaps, denials and scan continuation |
| Additive profile request silently narrowed | “Show my profile and medications” requested medications alone | Additive profile scope remains full; conflicting restrictions hand off |
| Operational exceptions abort model preparation | A second callback raised a timeout after a successful first read | Sanitized incomplete status preserves prior results; cancellation still propagates |

Both model-review runs produced actionable findings; all five above were accepted
after inspecting/reproducing the code paths. No finding was dismissed as merely
intentional. The second review independently passed 77 focused tests; its wider
run lacked PyTorch. The local repository's correctly configured Python 3.12
environment supplies the full test and model results in the manifest.

The last two repairs followed the second review: seven newly added regressions
failed before repair and passed afterward; the cancellation control passed both
times. A small qualifier grammar adjustment also followed the reviewed snapshot.
These final edits received targeted tests and manual inspection, not a third
model-review stamp. No claim of exhaustive correctness is made.

## Additional probes

Forty unsupported-constraint requests were frozen before execution. First run:
38/40, with two unsafe whole-day plans that discarded explicit clock/timezone
qualifiers. The decoder now rejects unrepresented clocks and timezone overrides.
The unchanged boundary set passes 40/40. First results remain in
`future-boundaries-first.json`.

The exact canonical Vita prompt-only battery was probed separately with synthetic
full-catalog availability, preserving the last four user requests. The initial
result was 4 plans / 16 handoffs, including one omitted hsCRP target. The repair
binds its common alias and rejects unbound metric-list items in any position.
Direct “What are my …” profile-field questions now bind the requested snapshot.
Final selector coverage is recorded in `vita-battery20-selector.json`; a handoff
is not a successfully answered battery question. The initial result remains in
`vita-battery20-first.json`.

Sixteen profile scope/intent regression cases pass. Their first run missed one
read because the modal word “May” was parsed as a month. Date parsing now operates
after the recognized read prefix. Tightening list coverage initially caused
conservative misses on supported no-research/no-labs qualifiers; preserved
`review-list-*` results record those misses and their repairs.

## Performance change

Identical frozen encoder inputs are reused only inside one selector call. The
request-local context is isolated across concurrent calls and cleared on both
success and failure. There is no cross-request text/feature cache. The model,
weights, precision, intent check, proposal-coverage check and validation remain.

`feature-reuse-benchmark.json` compares alternating reused/uncached runs with the
same model and verifies identical plans. `http-smoke.json` measures the final
single client through actual loopback HTTP. These timings exclude attestation,
WAN, database, research and answer generation. The unchanged backbone sometimes
still requires two or more passes; faster lookup does not establish faster or
better end-to-end medical answers.

## Orchestrator boundary

`examples/vita_orchestrator.py` prepares original user text plus typed evidence
for the existing answering model and supplies a fixed instruction block to add
to its existing policy. Handoff/invalid selection invokes no source operation;
incomplete evidence and unreviewed research remain explicit. The helper introduces
no model call, credentials, provider selection or authority grant.

Contract tests verify this preparation, not DeepSeek compliance. The helper is
not wired into the Vita app in this patch. A full canonical browser battery with
real authorized execution, answer-quality review and latency remains required
before claiming integrated acceptance. Enclave release, Linux/8 GB checks and
attestation of this candidate are also not established here.
