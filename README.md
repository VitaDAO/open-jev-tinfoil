# Open Jev on Tinfoil

Standalone authenticated CPU serving for `com-kotobalabs/open-jev-deberta-v3-large`.
Source model revision: `19bf9a64815add579fbf6c907bef584d9277a8e4`.
Independent reproduction of the Jev interface; not TypeSafe's proprietary Jev.

## Contract

`POST /decide` requires `Authorization: Bearer <OPEN_JEV_API_KEY>` and JSON:

```json
{"state":"I was charged twice and want a refund.","questions":[{"type":"choice","instructions":"Which team should handle this?","options":["billing","technical support","sales"]},{"type":"noul","instructions":"The customer wants a refund."}]}
```

Returns `answers`, `model_revision`, and `inference_ms`. Choice and score include
probability distributions; score is a zero-based expected level, and noul is p(yes).
`POST /route` requires the same bearer key and accepts `{"state":"Show my saved step goal."}`.
It returns `action`, `record_access`, uncalibrated `decision_scores`, `elapsed_ms`,
`model_revision`, `adapter_sha256`, and `advisory: true`. It does not produce an
urgency score or authorize any operation. It shares the encoder and inference
lock with `/decide`, so concurrent work is rejected with429 rather than queued.
The adapter is digest-checked before startup; changed weights require a new pin
and release. `GET /health` is public and reports readiness and adapter identity
after model load and warm-up.

Limits: 64 KiB body, 32 questions, 256 state tokens, 512 total tokens including
questions/options. Oversized states are rejected rather than silently truncated.
Choice accepts 2–255 distinct options; score accepts 2–10 ordered distinct levels.
Question/option text must be nonempty. The total token limit takes precedence over
nominal question/option counts. Errors: 401 unauthorized, 413 oversized body,
422 invalid input/token overflow, 429 busy (retry later). One inference at a time.

## Privacy and attestation

- Code, vendored loader and exact model snapshot are bundled in a digest-pinned image.
- Hash-locked Python dependencies and pinned Python base image.
- Model file SHA-256 manifest is included at `/opt/model/manifest.json`.
- Tinfoil release workflow measures configuration and signs/publishes attestation.
- Use Tinfoil SecureClient with repository `VitaDAO/open-jev-tinfoil` to verify the
  workload and establish attestation-pinned TLS before sending any private input.
  Ordinary curl/requests does not provide this verification.
- No runtime downloads, outbound service network, request access logs, telemetry,
  persistence, or cross-request text cache. Upstream caching is disabled by wrapper.
- Dedicated service key; no shared Vita credentials. Keep the key in the Vita
  agent enclave, never in browser JavaScript. Authentication does not replace
  consent or capability checks in Vita.
- `examples/vita_client.py` additionally pins the approved v0.1.0 release digest,
  and refuses a different release before reading/sending the application key.
- No debug SSH, automatic updates, GPU or production Vita integration.

Attestation establishes which code is running, not the correctness of decisions.
The model was evaluated on banking/sentiment/BoolQ; evaluate Vita cases separately.
No claim of memory zeroization is made; request objects exist transiently in RAM.

## Build and run

GitHub Actions builds linux/amd64, runs API tests and real-model smoke tests under
4 CPU / 8 GB limits, then pushes the tested image to GHCR. Pin that digest in
`tinfoil-config.yml`, publish a Tinfoil release, and explicitly create the enclave.
Release publication alone does not deploy it.

Local development (Python 3.12):

```sh
uv venv --python 3.12
uv pip install -r requirements.lock pytest==9.1.1 httpx==0.28.1
MODEL_DIR=model .venv/bin/python scripts/download_model.py
# Supply OPEN_JEV_API_KEY through a secret file/environment; never commit it.
PYTHONPATH=vendor MODEL_DIR=model HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/uvicorn server:create_app --factory --host 127.0.0.1 --port 18080 --no-access-log
```

Recovery: stop only the new `open-jev` container. Existing Vita services are independent.
See issue #1 and `evidence/` for deployment state, immutable identities and verification.

## Attribution

Loader files under `vendor/typed_decisions` are unmodified from the pinned model
snapshot, with the upstream Apache-2.0 license and notice. Base model
`microsoft/deberta-v3-large` is MIT. Dataset provenance and model limitations:
https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large

## Experimental local Vita routing

`examples/local_vita_router.py` preserves joint routing/urgency and separately
rechecks record access. On 24 fresh synthetic confirmation cases, record-access
correctness improved from 17/24 to 21/24, routing stayed 22/24, and local median
latency increased from 117 to 211 ms. This optional client does not change weights,
server behavior or the attested release. See `evidence/local-tuning.md` for the
selection process, regressions and limits; outputs are advisory, never permission grants.

### Learned local adapter

The newer `examples/local_learned_router.py` fits a small task-specific head over
the frozen Open Jev encoder. It reaches 24/24 on the requested set after using
that set for training. On 30 freshly frozen synthetic examples it scores30/30
routing and29/30 record access, versus24/30 and25/30 for the original model.
Local median for these two decisions is100ms. It does not produce urgency;
its scores are not calibrated probabilities. This optional in-process path does
not change `/decide` or the attested release. See `evidence/local-v3.md` for
training provenance, remaining error, reproducibility and evaluation limits.

The shared router implementation is now `routing.py`; the local example imports
that same implementation. `scripts/smoke_route.py` verifies all54 recorded
predictions against HTTP. `examples/vita_client.py` exposes `.route(state)` and
checks the attested release, model revision, adapter digest and response types.
`scripts/verify_route_live.py` runs the synthetic route suite through that verified
client. See issue1 for the actual deployed release/readiness; code presence is
not evidence of deployment.
