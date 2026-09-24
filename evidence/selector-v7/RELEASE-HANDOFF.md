# Release handoff — preparation only

Candidate branch: codex/intent-certificate-repair. The immutable source commit is
its PR head; the packet is committed with the tested code. Base: ef853f80d0ddeaead5e6f1f875c2e137e6a65bad.

Selector SHA256: 6beaf54f171b47a6f068c785e63f85a2e2aac64e63a73e0b4eb36777a55b5cc5
Intent adapter SHA256: 79057d0e2673aa2813a8ea64193d74182a33392b1221c7c06d4afc7afe18dc36
Model revision: 19bf9a64815add579fbf6c907bef584d9277a8e4
Contract: POST /v1/select, vita-selector/v2 -> vita-query-plan/v2.

This packet does not authorize merging, building/publishing a release image,
selecting an enclave release, or changing Vita's endpoint/client/deadline.
No candidate image digest, attestation, release tag or live timing exists yet.
The earlier HTTP smoke belongs to an older source identity and is not acceptance
for this candidate. Do not point Vita at this candidate until the gates below pass.

## Accepted local evidence

See local-verification.json, README.md and fresh-repaired-independent-review.json.
388 unit tests pass, one skipped; 35 native adapter tests pass. Model regression
packets: 42/42, 73/73, 22/22; historical430 has429 exact matches plus the retained
reviewed bB-075 oracle mismatch. Independent Vita replay matches all42 outputs;
all12 planned outputs pass native projection. All are now disclosed regression
checks, not universal accuracy or browser acceptance.

## Release gates and handoff fields

1. Review/accept the immutable repair PR. PR8 separately changes broad-research
   native projection; it is not included in this branch. Reconcile source and
   native client identities before testing mixed research in Vita.
2. Before a main merge, obtain the relevant principal's exact revision decision.
   Main push triggers the existing image build/test/publish workflow. Do not
   equate that image publication with a selected enclave deployment.
3. Record the accepted source commit and resulting environment-neutral image
   digest. Run container startup, authenticated /v1/select and negative auth
   checks on that exact image under4CPU/8GiB and read-only/tmpfs constraints.
4. Recheck the actual target configuration repository/tag sequence and current
   rollback selection. Pin the tested digest, then record configuration commit,
   candidate tag, release/attestation digest, target enclave/domain, rollback tag,
   named operator/action, verification and recovery scope before lifecycle action.
5. After an authorized lifecycle action, verify SDK attestation and /health
   against the expected selector/model/adapter identities. Never disable
   verification to obtain a successful request. Retain closed egress and current
   secret binding; no credentials belong in the packet.
6. Measure synthetic cold and warm requests using the same inputs and client:
   profile read, simple metric read, lab metadata read, and a descriptive weekly
   summary. Record attestation/connection/inference/total timing separately;
   run sequentially without competing benchmarks. Preserve the1second Vita
   deadline unless separately changed by the principal.
7. Supply Vita: source commit, image digest, configuration commit/tag, attestation
   identity, verified host, selector/adapter/model hashes, exact native client
   commit, API request/response evidence, cold/warm timing artifact, and rollback
   packet. Then run the four live candidate journeys and compare acquisition
   scope, avoided model rounds, answer quality and total time.

## Failure and recovery

On a failed gate, do not advance the endpoint/client pin or claim release ready.
An already selected candidate may be reverted only through the recorded rollback
scope. Metadata pages are capped at200 and must retain completeness information;
exam_date must not become an asserted issuance/upload date. Ambiguous or
unrepresentable requests stay with native Fable handling. Existing privacy and
authorization checks remain authoritative.
