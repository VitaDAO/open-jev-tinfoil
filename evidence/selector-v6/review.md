# Focused release review

The configured Codex review of the initial native-wrapper changes identified a
calendar date-boundary issue. The wrapper now preserves date-only bounds for
calendar records and exact instants for metrics. Six regressions cover the real
Vita executor, current-day expansion and DST. The quiet current-v32 suite passes
19/19 with unchanged deadlines. An unrelated finding concerned a pre-existing
untracked segmented-selector experiment, which is excluded from this release.

A separate reviewer inspected the new release-pinned client against the actual
tinfoil0.14 SDK. Initial synchronous transport creation, automatic synchronous
TLS re-verification/retry and the analogous asynchronous rebuild all invoke the
public verify override before constructing a replacement transport. No
implementation defect was found in the used sync path. The reviewer found that
CI would skip actual-SDK tests without tinfoil installed; the unit-test step now
explicitly installs tinfoil0.14. Same-approved-release rotation succeeds and a
changed release cannot receive a retried private request in the synthetic tests.
A real SDK smoke against the existing enclave also rejected a wrong release pin
and verified approved public health.

Operational finding: the SDK verifies GitHub's latest published release. The
recovery procedure therefore restores that release pointer together with the
previous enclave tag; a container-only rollback would not restore verification.

The semantic fixtures are finite and partly reused. Correct handoffs are not
answered questions, and neither schemas nor attestation prove answer quality.
The Vita task owns live browser acceptance using the actual answering provider.

A fresh reviewer challenged provider-adjective parsing with negated sources
and differently scoped second subjects. The binder now rejects negated provider
filters and ambiguous multi-subject adjective scope before removing the provider
name. All 15 focused source tests pass, including the reviewer's failures;
separately stated source/date clauses retain their independent queries.
