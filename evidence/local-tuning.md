# Local Open Jev tuning — 2026-09-23

Implemented an experimental local Vita router in `examples/local_vita_router.py`. It keeps the original joint call for routing and urgency, then replaces only the record-access answer with a second call containing that question alone. No model weights, temperature, threshold, server, attested image, or deployment changed. It requires the existing authenticated loopback server on port 18080. Run `.venv/bin/python examples/local_vita_router.py` for a synthetic example, or import `local_client` and `route`.

## Experiment and selection

The original ten cases are development data. Five candidates were evaluated: original questions, a declarative record-access question, longer descriptive routing options, isolated original questions, and isolated descriptive questions. The first 24-case fixture was written before development calls, and its SHA256 saved in the development result. Isolating all questions won development (18/20 correct decisions versus 15/20 baseline), but first validation showed a routing regression: baseline choice21/24 + record17/24 versus isolated choice20/24 + record19/24. Longer descriptions were worse on development.

After examining that first validation, the hybrid was chosen. That fixture is therefore validation data, not an untouched confirmation set for the final hybrid. A separate 24-case confirmation fixture was written before running the final baseline/hybrid comparison. The arms alternate order for each case; no retries, cloud calls, or training were used. Exact fixtures, scripts and raw outputs are committed alongside this report.

| Confirmation result (24 new cases) | Baseline | Hybrid |
|---|---:|---:|
| Routing correct | 22/24 | 22/24 |
| Record-access correct at p >= 0.5 | 17/24 | 21/24 |
| Median end-to-end local time | 116.925 ms | 211.29 ms |
| Model calls per input | 1 | 2 |

Five record-access errors improved and one regressed. “What does a hydration goal mean in general?” improved from p(access)=0.627 to0.211. “Change my hydration goal to two liters” regressed from0.485 to0.556. Several decisions remain close to0.5. Routing and urgency outputs were exactly equal between arms on every confirmation case. Urgency has no gold numeric labels and was not evaluated for correctness.

This is an assistant-authored, small synthetic English intent suite, not independent or blinded judging, clinical validation, broad model improvement, or a significance claim. Fixtures use a narrow three-intent taxonomy and define viewing current account settings as accessing existing personal information; this is not a final Vita product-policy decision. Contradictory, mixed-intent, adversarial and multilingual inputs require separate evaluation. Correctness gains may not generalize, and hosted latency will include network overhead for two sequential requests. TypeSafe was not rerun on this new fixture.

The model jointly attends to state and all question text, so separate calls remove an avenue for cross-question interference. This mechanism is consistent with the observed behavior; it does not guarantee stronger predictions in other tasks. The upstream model card explicitly identifies unfamiliar instructions and domains as limitations: https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large .

## Validation and recovery

12 tests pass, including that only the record answer is replaced, identical state is supplied to both calls, and wrong revision/HTTP errors fail without a partial result. A live synthetic call through the new client passed. No secret or personal record is included in artifacts. This client remains optional and advisory; it does not grant permissions or execute changes. Recovery is to stop using this example and continue the unchanged one-call API. Substantial routing improvements likely require representative labeled Vita data and measured domain fine-tuning, rather than more verbose instructions.
