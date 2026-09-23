# Learned local intent adapter — 2026-09-23

The requested 24-case set now scores **24/24 routing and 24/24 record-access decisions**. Those cases were explicitly promoted to training/development after the user asked to fix them; this is training fit, not unseen accuracy.

## Model change

`examples/local_learned_router.py` uses Open Jev's fixed DeBERTa backbone to encode the state together with the original routing question/options. It mean-pools only the state tokens, L2-normalizes the 1024-dimensional representation, and applies a learned 1024x5 linear ridge head: three routing scores and two record-access scores. The backbone and tokenizer are unchanged. The 5,120 weights are stored in `adapters/vita-intent-v1.json`, with model revision and training-input hashes. The upstream generic decision head is replaced for these two tasks only. This is a narrow local classifier, not an improvement to every generic Open Jev question.

Training: 58 unique previously seen synthetic examples (original10, first validation24, previous confirmation24). Regularization selected among six values by analytic leave-one-out accuracy across both tasks; ties prefer stronger regularization. Lambda0.01 gives113/116 leave-one-out decisions correct and116/116 full-fit training decisions. Leave-one-out here is model selection, not an independent performance estimate. No fresh-case labels selected features, regularization, or weights. No input-string rules, test-case lookups, provider calls, GPU training, temperature adjustment, or lowered evaluation threshold were used.

Before tuning, a new 30-case fixture was written and its SHA256 recorded by the prompt-search script. That same hash is present in the training report and was reverified afterward. First, ten routing and ten yes/no prompt variants were evaluated on the target24; single prompts and tested ensembles did not reach24/24. The learned adapter was then fitted and evaluated once on fresh30. Its saved float32 weights were reloaded and predictions checked against the original fit; results are identical. Reverification is not another independent sample.

## Results

| Set / approach | Routing | Record access |
|---|---:|---:|
| Target24, previous hybrid | 22/24 | 21/24 |
| Target24, learned adapter (now training data) | 24/24 | 24/24 |
| Fresh30, original joint model | 24/30 | 25/30 |
| Fresh30, previous two-call hybrid | 24/30 | 24/30 |
| Fresh30, learned adapter | 30/30 | 29/30 |

Fresh30 local median latency: learned adapter100.05ms, original121.58ms, previous hybrid217.94ms. The adapter runs in-process and supplies two decisions, whereas HTTP baselines include urgency as a third task; these are observed path timings, not an equal-work model speed benchmark. Model initialization is excluded. Target24 adapter median92.92ms. No hosted timing or resource-usage claim follows.

Remaining fresh error: “Explain what an activity target is, without looking at my own target.” Routing correctly says explanation, but the separate record-access head says yes. This was recorded, not used to retune. Treat conflicts as requiring further handling in a future integration; the current classifier grants no access and executes no actions.

All examples and labels were authored by the implementing assistant. The suite is English-only, balanced, single-intent, and closely related to the training domain. Record-access labels are correlated with routing in these examples. It does not cover mixed requests, adversarial text, long histories, actual users, or clinical interpretation. A larger independently labeled set is required before production use. The reported scores are uncalibrated ridge outputs, deliberately not named probabilities.

## Use, verification, and scope

From this checkout with local model weights available:

```python
from examples.local_learned_router import LocalLearnedRouter
router = LocalLearnedRouter()
result = router.route("What is my currently selected language? Please leave it unchanged.")
```

Use `.venv/bin/python examples/local_learned_router.py` for a synthetic demo. `experiments/train_adapter.py` reproduces training and writes a separate candidate artifact; `experiments/verify_adapter.py` checks the shipped adapter. Training saves a candidate and does not automatically replace the shipped weights.

20 tests pass. Real-model verification covers all54 target/fresh cases, saved-artifact prediction parity, empty/overlong rejection, and zero retained collator cache entries. Identity, shape and finite-weight validation are tested. Upstream tokenizer emits a generic Mistral-regex warning even for this pinned DeBERTa snapshot; tokenizer behavior was not changed during comparison.

The current `/decide` server, generic upstream weights, Tinfoil release and Vita production are unchanged. New local router is optional, in-process and advisory. It does not expose an HTTP endpoint, supply urgency, or claim attestation for the modified computation. Any later enclave integration must bundle/pin the adapter and publish a newly measured release. Recovery is to use the existing generic server or prior local client.
