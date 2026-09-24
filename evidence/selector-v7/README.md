# Intent and binding repair — local candidate

This candidate is not deployed. It retains the existing acquisition-only contract:
Open-JEV proposes personal-record/literature reads, native Vita code validates and
executes them, and Fable interprets and answers. Handoffs are safe continuations,
not avoided model rounds or evidence of better end-to-end answer quality.

## Diagnosis

- Grammar coverage previously bypassed a negative learned intent decision for a
  general definition such as “What is ApoB?” A grammar match now retains the
  learned personal-read check.
- The v4 head adds generic explanation negatives to the synthetic development
  corpus. It is a conservative supported-read gate, not a general reasoning model.
- The v4-only candidate missed a previously supported report dates/provider
  request (bB-033). The executor already supports lab metadata; the learned gate
  was the failing layer. A complete, explicit personal metadata grammar now binds
  the catalog's lab report aliases to one existing labs read. It consumes every
  clause and does not erase filters, dates, counts, actions, or other people.
- Zero record counts, duplicate reads, standing no-read restrictions, and the
  stale HRV catalog mapping are code-level defects, repaired independently of
  model confidence.

## Training provenance

`v4-training-protocol.json` contains 915 synthetic training rows and 60 calibration
rows. Its SHA256 matches the pinned v4 artifact's dataset SHA256. Calibration is
used to select regularization and threshold; it is not held-out accuracy proof.
The fixed encoder is Open-JEV revision 19bf9a64815add579fbf6c907bef584d9277a8e4,
FP16 checkpoint storage loaded for FP32 CPU computation. Model weights are unchanged.

Reproduce with `scripts/reproduce_read_intent_v4.py --model-dir <pinned-model-fp16>
--output-dir <empty-directory>`. The script uses only the frozen protocol, not
acceptance packets. The existing tokenizer regex warning is unchanged.

The semantic-only head, global metadata refit (v5), and anchored update (v6) were
rejected offline; none is installed in the candidate. The global refit lost at
least five ordinary reads in its first 40 regression cases and was stopped early.
The anchored update failed its zero-false-accept calibration condition. They have
no complete passing release score.

## Scope of evidence

The 22- and 73-case packets were independently authored by the Vita task but are
now disclosed regression cases. The 36-case family expansion is author-created
development evidence. The historical 430-case oracle has one documented strict
mismatch, bB-075: its expected handoff differs from a valid Oura sleep-efficiency
seven-day read. Preserve the raw strict grade rather than silently relabeling it.

No production data, live retrieval, enclave attestation, final-answer quality, or
end-to-end latency is established by these local synthetic selector tests. Existing
native authorization, inventory checks, the one-second deadline, and deployed
service/client remain unchanged.

## Fresh acceptance failures and bounded repair

An independently frozen 42-case packet initially found three wrong acquisitions:
translation of a quoted read, a dropped zero workout count, and an ignored
standing request to keep records closed. Three metadata variants also missed
reads. These are real failures; the original 9 correct plans / 27 handoffs /
3 wrong / 3 missed result is retained. Schema validation accepted all three wrong
plans, demonstrating why structural validation is not semantic correctness.

The repair hands text-transformation tasks to native handling before embedded
commands bind; rejects unbound record quantities after recognized count/date
binding; recognizes closed/off-limits standing restrictions; adds the shared
blood-reports catalog alias; and completely parses lists of metadata fields.
After repair, the same (now disclosed) packet is 12 correct plans / 30 handoffs /
0 wrong / 0 missed. This is regression repair, not a new unseen test result.

Latest local checks: 388 unit tests pass, one skipped; 35 native adapter tests
pass; 73-case and 22-case suites remain clean; 430-case corpus is 429 exact
matches plus the documented bB-075 oracle mismatch. See local-verification.json
for source identity, counts and raw artifact hashes. No new wrong acquisitions
were observed in these packets; this is not a universal guarantee.

Vita independently replayed the repaired 42 cases from a frozen snapshot: all
outputs matched, all 12 planned outputs passed native projection, and its audit
confirmed all 430 rows and the retained oracle mismatch. Source review found no
new blocker in this bounded patch. This does not establish browser acceptance or
a deployment identity.
