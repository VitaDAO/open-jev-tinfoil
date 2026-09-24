# Optional client-only broad research projection

The native adapter can now propose a mixed health/research batch for a complete current-message broad analysis request recognized by the existing closed grammar (for example, `Analyze me`). The request must have no explicit period, the plan must contain health reads, and its single research operation must retain the exact canonical sleep/physical-activity/cardiometabolic-health topics, diet/exercise interventions and improve goal. This is a projection compatibility change, not a model accuracy improvement.

Vita already permits `general_overview_research` with empty `basis_source_ids` alongside health reads. Those topics are general suggestions, not findings or priorities inferred from records. No topic, source ID, health range or operation is invented by this adapter. The existing request/inventory binding and actual turn schema still validate the complete batch; native authorization and execution remain Vita's responsibility.

Explicit topic or numerical research, derived/contextual followups, extra clauses, unmatched qualifiers, date-qualified analyses and sleep-end-day reads continue to hand off the whole plan. Other broad paraphrases may still hand off. Forced tool choices, attachments, resumed reads and subsequent rounds retain existing eligibility behavior. No model loop, source IO, external study search or timeout change is added.

## Verification

- Actual Vita v32 native suite: 64 passed, including genuine mixed-capability manager schema, normalized native contract, unchanged health output, complete operation IDs, whole-envelope fallback, capability/metric negatives and round eligibility. The adapter matches only the unqualified broad-analysis grammar production, avoiding the general parser's dangling-preposition stripping.
- Standalone suite: 353 passed, one native-environment-dependent module skipped.
- Offline comparison of the saved 430-case v0.3.1 synthetic report using actual Vita health-plus-literature schemas: compatible projections 175 -> 176; projection handoffs 31 -> 30; selector handoffs remain 224. All 175 previously accepted batches are byte-equivalent after canonical JSON serialization. Only `m02`, `Please analyze me.`, changes. No network/source IO.
- The original report SHA256 is `fc45d4df6eff17945d96eafa12a7d2cd093701c1afc545e964910f1036231f5a`. Original strict/reviewed selector scores are unchanged; this is not a new inference run or browser acceptance.

## Activation and recovery

No dependency, serving source, image, model, selector, intent-adapter or attestation pin changes. Compatible with deployed v0.3.1: release digest `8a8d94c40328cc8ee9e2ccc186ba9afc4799f9e22e67c5c1523fce4cb880863e`, selector `d38dfecc0d701a3cd9c3ed74d7a6ef5256ace53927ee158e1340434952005b9f`, intent adapter `31111ec06e545f0e68f7a2c7ad62cd5c1c6d554402e881a3ef0fbe85cf2d24f0`, model `19bf9a64815add579fbf6c907bef584d9277a8e4`.

The Vita owner must independently verify and opt into the new frozen client. This change does not deploy or activate anything. Roll back the client checkout to `3396d7c49d243b9f446cdfa0f1f672e1487946a8` or disable the optional selector wrapper. Keep the one-second budget and native fallback. This patch cannot repair selector network timeouts or promise perfect final answers.
