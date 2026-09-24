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
and adapter pins. There is no approved release pin for this experimental
integration yet. The loopback client in the synthetic runner is test-only.

The model-facing `acquire_sources` schema for the **current turn** supplies the
available native operations and exact parameter constraints. Native Vita's
existing instructions carry its capability and grant/coverage descriptions.
These tools, instructions and history are forwarded unchanged to that provider on a
handoff, selector error, unrepresentable request or later reasoning round. Do
not append a duplicate global capability list or interpret a schema field as a
populated user value. Unknown, partial, empty and withheld remain distinct.

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
standalone sleep-end projection and research provenance currently require that
planner too. Do not relabel a research subject or omit a clause to manufacture a
fast-path success. Failure of the optional selector is not failure of Vita.

The wrapper is one instance per turn. It checks disclosure before and after
selection; authority failures and cancellation propagate. Selector waiting is
bounded at one second by default. A timed-out transport thread may finish its
network request in the background but cannot dispatch a source operation. The
existing client bounds that transport independently.

## Verified scope

Current native-wrapper synthetic target:
`vita-agent-deepseek-v32@8ffcdde225e7e0296038078c1679a999699cb9fb`, including
its existing local changes. The logged September 24 rerun recorded source
hashes and passed all 19 wrapper tests with the existing 20-second native
deadline. The active answering provider is Fable medium; its existing
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

The native wrapper has not been enabled in the running browser app or deployed
to Tinfoil. Complete canonical browser acceptance and an attested release are
still required. Remove the opt-in wrapper to return to the existing provider.
