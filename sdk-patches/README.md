# Opt-in public trust cache

This client-only change requires a reviewed `tinfoil==0.14.0+vita1` wheel. It does
not change the model, enclave image, deployment or installed shared SDK.
The source wheel is PyPI `tinfoil==0.14.0`:

- upstream wheel SHA256: `f04e1c0ed98e22619c03e6fc8b9a2cab60a4e9ed2001ffec58ab5cd848d77642`
- patched wheel SHA256: `8fdd63b508956e46a3b46cf869fcf5124b5730e0e31e5a4847109ca4e527e259`
- cache dependency contract: `sigstore==4.5.0`, `tuf==6.0.0`

Reproduce with Python 3.12 (the builder itself uses only the standard library):

```sh
python -m pip download --no-deps --only-binary=:all: tinfoil==0.14.0 -d /tmp/sdk-input
python scripts/build_trust_sdk.py /tmp/sdk-input/tinfoil-0.14.0-py3-none-any.whl /tmp/tinfoil-0.14.0+vita1-py3-none-any.whl
```

The builder checks the entire upstream wheel and both changed source hashes,
preserves other package code/licenses, updates version and RECORD metadata, and
writes deterministic ZIP entries. The supplied unified patch is the review diff;
the builder is the reproducible source of the wheel. Two builds matched exactly.
Do not replace a shared installation in place. Vita must pin the artifact/hash
consistently in its own lockfile and image packaging after independent review.

## API and process ownership

```python
# Parent: public metadata only. No VitaClient or private request enters this step.
from examples.trust_cache import PublicTrustCache
cache = PublicTrustCache()
cache.refresh()                         # Online authenticated TUF update.
snapshot = cache.snapshot()             # Pass this immutable value to a child.

# Child: construct fresh crypto objects and a fresh attested TLS client.
client = VitaClient(release_digest=approved_release,
                    selector_sha256=approved_selector,
                    adapter_sha256=approved_adapter,
                    trust_snapshot=snapshot)
```

`SecureClient(..., sigstore_verifier_factory=...)` is the sole SDK extension.
It invokes a factory at each code-verification step, including re-verification
following key rotation. Tinfoil's existing signature, certificate identity,
transparency, exact release/tag, measurement and enclave/TLS binding checks stay
in place. The factory creates a new child-local Sigstore verifier from public
root bytes; it does not return a verified enclave or skip verification.
Without an injected factory the SDK retains its original behavior.

Snapshots expire at the earlier of five minutes from refresh start or the earliest
expiry among the authenticated TUF metadata loaded for the trust root (including
delegations). Age is checked with both wall and monotonic clocks, and the root
bytes are hash-checked. A failed refresh does not replace the previous snapshot
or extend its validity. Before expiry an existing valid snapshot can still be
used. Missing/expired/tampered snapshots fail closed: the caller uses Vita's
normal planner. Do not silently initialize an uncached client after cache failure.

Refresh runs in a short-lived helper process with a sanitized environment: no
API keys or unrelated credentials are forwarded. The helper performs online TUF
verification and returns only public root bytes, their hash and fixed expiry
metadata over a private pipe. It is bounded at 30 seconds; a timeout/failure
leaves the previous snapshot and its original expiry unchanged. This keeps
network-library native threads and cryptographic initialization out of the parent.
A macOS check confirmed one native parent thread before and after refresh, and a
subsequent real fork emitted no warning. No helper process persists after refresh.

The host should call `refresh()` when `refresh_due` is true (normally after four
minutes, sooner near signed expiry), outside the request's critical path. The
cache starts no background thread, scheduler or endpoint. Refresh is serialized
in the owning parent. A child must receive a snapshot, not call the inherited
cache object. Do not share a client, verifier, TLS pool or lock between processes.
If refreshing in a background thread, retain the host's established safe child
creation strategy; this helper does not make forking a multithreaded host safe.

The snapshot is trusted in-memory configuration created by this module, not an
untrusted serialized cache format. There is no disk snapshot loader or API to
accept arbitrary caller-supplied roots. Host IPC must preserve that boundary.
Expiry rejects new select/route/decide operations and re-verification; close in
the owning process remains allowed for cleanup. Obtain a current snapshot and
construct a new client for later turns after expiry. Initialization and selection
must remain under Vita's combined two-second waiting budget; synchronous-worker
cancellation/cleanup remains the host's responsibility.

## Verification and measured limits

Tests cover five-minute and earlier signed expiry, refresh failure without stale
extension, clock rollback/forward changes, modified roots, exact release/repo/tag
and signature rejection, SDK key rotation, process guards and cleanup.
The test fixtures are public Sigstore material from the v0.3.2 release, never
application credentials, user records or a production trust-cache input.

A four-sample alternating live comparison used separate child processes and the
same public ApoB request. Baseline client setup: 2879/2065 ms. Cached setup:
1503/1435 ms. Cached setup plus fresh TLS selection: 2113/2036 ms. Attestation,
release identity and typed results matched. VitaClient/QueryRequest imports were excluded; lazy SDK imports inside the
constructor were included. That first experiment had already loaded Sigstore
modules in the parent, and disk caches were retained. This small measurement shows an improvement, not a sub-two-second
end-to-end guarantee. The helper still performs fresh enclave verification and
release lookup; it does not cache either away.

The final helper-process cold-path retest deliberately leaves SDK modules out of
parent memory. Baseline setup was 6938/4697 ms; cached setup 4276/5045 ms, with
fresh TLS selection adding 678/643 ms in the cached arms. These totals include
lazy SDK imports in the constructor and network variability. Neither run proves
reliable selection in the two-second caller budget. Do not compare absolute
latency between the different parent-preload conditions. Both evidence files are
preserved under `evidence/client-trust-cache/`.

## Async server lifecycle

`await cache.async_refresh()` runs the same isolated helper and uses exactly the
same snapshot decoder and validation as `refresh()`. Use it in an async host's
lifespan before admitting requests and for proactive refresh when `refresh_due`
becomes true. Helper startup plus communication has a 30-second limit. Timeout,
nonzero exit, bad output and cancellation leave the prior snapshot and expiry
unchanged. Cancellation during launch is shielded until the process handle is
obtained; cleanup kills/drains/reaps the helper even on repeated cancellation.

The same parent-owned gate serializes sync/async refresh. Async waiters yield to
the event loop; a concurrent sync refresh raises `trust_refresh_in_progress`
instead of blocking the event loop. Cancelling a waiting async caller does not
cancel the in-progress refresh. No process-wide loop or signal policy is changed.

Supported async subprocess implementations are uvloop and the stdlib nonthreaded
Pidfd/Safe/Fast child watchers. A stdlib ThreadedChildWatcher is refused before
spawn, because it creates a parent wait thread. Missing/unsupported cache remains
unavailable and the host should select its normal planner path. Configure the
host's known-safe process policy explicitly; do not silently switch it here.

A real uvloop public-refresh test completed in891ms while the event loop advanced
82times. Parent native threads stayed1 before/after; a subsequent actual fork
reported no warning and its child validated the snapshot. Focused tests also
cover timeout reaping, launch-window cancellation, repeated cancellation, failed
refresh, and sync/async serialization. Evidence is `async-lifecycle.json`.
