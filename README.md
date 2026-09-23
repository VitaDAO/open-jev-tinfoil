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
`GET /health` is public and reports readiness after model load and warm-up.

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
