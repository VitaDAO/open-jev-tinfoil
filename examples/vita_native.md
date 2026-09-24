# Native Vita integration

Use `NativeSelectorProvider` inside the existing admitted Vita manager turn. It
proposes a representable first acquisition and leaves the same manager,
answering provider, capability broker, scheduler, evidence registry and terminal
validation in control of the rest of the turn.

```python
from examples.vita_native import NativeSelectorProvider

# Existing Vita-owned objects. The request uses the admitted turn clock,
# authorized inventory and history, never a global list of possible metrics.
model = NativeSelectorProvider(
    client, request, native_provider, authorize=manager.check_disclosure,
)
result = await manager.run(original_question, model=model, history=history,
                           conversation_id=conversation_id)
```

`client` is the single attested `VitaClient`, with reviewed release, selector
and adapter pins from the reviewed release acceptance packet. Initialize this
shared client before admitted turns; verification may require network round trips.
The loopback client in the synthetic runner is test-only.

The model-facing `acquire_sources` schema for the **current turn** supplies the
available native operations and exact parameter constraints. Native Vita's
existing instructions carry its capability and grant/coverage descriptions.
These tools, instructions and history are forwarded unchanged to that provider on a
handoff, selector error, unrepresentable request or later reasoning round. Do
not append a duplicate global capability list or interpret a schema field as a
populated user value. Unknown, partial, empty and withheld remain distinct.
Native digit-leading metric slugs are preserved in the admitted inventory;
identifier syntax never grants permission to read an unadmitted metric.

`receipt.reason_code` is a fixed, non-sensitive diagnostic: eligibility failures
such as `selector_request_unrepresentable`, `native_tool_choice`, or
`original_request_mismatch`; `selector_timeout`; `selector_failed`; or
`native_projection_rejected`. Provider exception text is never recorded.


Follow-up history can be plain prior user text or Vita's admitted
`vita_prior_user_context` recall pairs. The adapter validates the native source,
synthetic call ID, empty arguments, matching result and exact recall schema,
then compares the resulting questions in order with the selector request.
Assistant recall and ordinary source outputs never become user history.
Malformed or mismatched recall falls back with `selector_history_untrusted`
or `selector_history_mismatch`, preserving the original envelope. No recalled
question is promoted into a new instruction or used to expand inventory.

A selector proposal is checked against that live schema, resolves metric names
using the current Vita registry, and binds relative dates using Vita's date
arithmetic and the trusted request clock. Calendar-record periods retain whole
local days, including the rest of the current day; metric periods retain exact
instants. Its `acquire_sources` call then enters
the normal manager review/tool/scheduler path. It does not call a database or a
raw executor callback. The broker still checks each read and native source IDs
remain available to final-answer validation. No `all_requested_delivered` flag
is substituted for answer correctness.

Unsupported counts, comparisons, clinical interpretation, arbitrary research,
app actions and contextual questions use the original native planner. The
standalone sleep-end projection and source-derived research still require that
planner too. A single explicit public research request can go directly to
`literature_reads` when the entire question fits the closed grammar and its
targets, diet/exercise interventions and improve/lower goal exactly match the
selector plan. This preserves the original study wording (including randomized
trials) and uses `explicit_subjects_in_current_user_message` with no source IDs.
Personal values, dates, population filters, unresolved references, mixed reads
and other unmatched qualifiers retain the complete planner handoff. Do not relabel a research subject or omit a clause to manufacture a
fast-path success. Failure of the optional selector is not failure of Vita.

The wrapper is one instance per turn. It checks disclosure before and after
selection; authority failures and cancellation propagate. Selector waiting is
bounded at one second by default. A timed-out transport thread may finish its
network request in the background but cannot dispatch a source operation.
Share the same `VitaClient` across turns: it admits only one outstanding selector
request and rejects later calls immediately while that worker is still running,
so those turns use the native planner. Admission is released only when the worker
exits, including on errors. HTTP's 15-second phase inactivity timeouts and the
SDK's separate attestation timeouts are not an overall wall-clock deadline;
the one-second wrapper deadline does not cancel synchronous network I/O.

## Verified scope

Current native-wrapper synthetic target:
`vita-agent-deepseek-v32@8ffcdde225e7e0296038078c1679a999699cb9fb`, including
its existing local changes. The logged September 24 rerun recorded source
hashes and passed all 19 original wrapper tests with the existing 20-second native
deadline. The diagnostic follow-up also exercises the real session-policy
bootstrap prefix and native manager acquisition. The active answering provider is Fable medium; its existing
MockTransport stream/settings test also passed. These tests use synthetic
readers and stub providers, not live model inference.

- Native manager tests exercise actual brokered HRV acquisition, source identity
  and accepted final rendering, plus exact fallback-context preservation, long
  questions, contract mismatches, clock binding and revoked authority. Calendar
  regressions cover relative, current-month and current-day periods across
  non-UTC date rollover and DST, using the actual native executor and validator.
- `scripts/verify_vita_native_battery.py --dry-run` checks the 20 canonical
  question tool schemas and 16 independent synthetic source oracles. This does
  not invoke an answering model and does not grade 20 final answers. The runner
  and its earlier dry-run evidence remain pinned to the separate
  `vita-agent-deepseek@e98298039153a24f9a1649e28a9fa93341ca0592` sibling; that
  evidence is not canonical-20 acceptance for v32.
- `--live` requires explicit local selector/config arguments and limits the
  complete run to 20 questions, four rounds each, 2,048 output tokens per call,
  80 provider calls and 3 MB total serialized provider inputs. Results preserve
  terminal acceptance separately from acquisition and independent correctness.

The first v32 wrapper attempt had one native deadline failure; its quiet
logged rerun passed without increasing the deadline. The first attempt lacked
stage timings, so its cause remains unproven. Neither result establishes live
answer quality or end-to-end inference latency.

The v32 mounted runtime has a separate optional `backbone.local_jev` first-round
hook. When adopting this wrapper, replace that hook instead of layering two
selectors. Its current activation was not inspected during synthetic testing.

Check [issue #1](https://github.com/VitaDAO/open-jev-tinfoil/issues/1) and the
release's acceptance assets for exact live endpoint, source/image/config identities,
reviewed pins, measured latency and application acceptance state. A healthy enclave
does not establish completed browser acceptance. Remove the opt-in wrapper to
return to the existing provider.

## Fork-per-request hosts

Construct `VitaClient` only inside the admitted request child, after the fork.
Its TLS pool, SDK transport locks and selection lock belong to that process.
`select`, `route`, `decide` and `close` reject use from another PID before touching
those resources. Do not reset or close an inherited client in the child; create
a fresh one. Keep parent startup free of initialized Open-JEV transports.

For a two-second application budget, place both construction/attestation and
selection inside the same timed operation. The wrapper's existing timeout covers
`client.select` only: initialization performed before it is outside that budget.
A lazy child-owned adapter can initialize within `select`. A timed-out thread is
not cancelled; do not treat `asyncio.wait_for` as a hard network/worker deadline.
If strict worker termination is required, isolate the operation in a disposable
process created by the admitted child and terminate/reap it on expiry. Decrypted
requests and results must not travel through the opaque-capsule parent.

The client reads `OPEN_JEV_API_KEY` from the child environment after attestation
verification. It does not implement a secret-file loader or reuse a developer
key path. Use the enclave's configured secret injection. Release and selector
pins remain mandatory. Cold attestation costs are additional to warm selection
latency and must be measured in the actual process architecture.

An optional five-minute public trust snapshot can reduce repeated Sigstore trust
root refreshes without inheriting a connection. See
[`sdk-patches/README.md`](../sdk-patches/README.md) for the exact opt-in wheel,
parent/child API, expiry policy, measured limits and packaging requirements.
