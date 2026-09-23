# Open Jev / TypeSafe Jev comparison — 2026-09-23

30 successful requests per arm: 10 frozen synthetic Vita-style states, each with the same three questions (choice, noul, score), repeated three times. Alternating arm order, persistent HTTP connections, no retries. TypeSafe version pinned and returned as `jev-1.13.0`. The first client request is separate; neither is a measured model cold start. No real personal records were used.

| Arm | First request | Warm median (n=29) | Warm p95 (nearest rank) | Routing / 10 | Record-access yes/no / 10 |
|---|---:|---:|---:|---:|---:|
| Open Jev, local macOS CPU, 4 threads | 122.45 ms | 121.40 ms | 141.78 ms | 9 | 6 |
| TypeSafe Jev API, WAN included | 385.51 ms | 312.07 ms | 397.87 ms | 10 | 10 |
| Open Jev, Tinfoil enclave | unavailable: HTTP 503 | unavailable | unavailable | not run | not run |

These latency rows are not an equal-infrastructure speed ranking. Local requests omit WAN/enclave overhead. The separate Linux 4-CPU / 8-GB container smoke measured 561.7 ms median on a different three-question support packet (10 requests). Hosted Open Jev end-to-end latency remains unmeasured. A 503 response is not inference latency.

Correctness uses the 10 unique cases' first responses and labels frozen before calls; noul threshold is 0.5. Repetitions do not provide 30 independent accuracy cases. This assistant-authored smoke set is not held-out, blinded, clinically validated, or sufficient to establish production reliability. Confidence numbers are not compared because definitions/calibration differ. Score outputs are preserved in JSON but have no pre-registered numeric gold labels.

Open Jev misrouted “Show my notification settings, but do not change them” to changing settings. It also classified generic REM/HRV explanation requests as requests to access personal records. The model must not grant access or execute changes; Vita's deterministic consent/capability checks remain necessary.

| Case | Open choice | TypeSafe choice | Open p(record access) | TypeSafe p(record access) |
|---|---|---|---:|---:|
| sleep-read | read existing data | read existing data | 0.748 | 0.950 |
| sleep-explain | explain a concept | explain a concept | 0.614 | 0.010 |
| notifications | change settings | change settings | 0.549 | 0.020 |
| steps-read | read existing data | read existing data | 0.759 | 0.950 |
| hrv-explain | explain a concept | explain a concept | 0.630 | 0.010 |
| units | change settings | change settings | 0.349 | 0.020 |
| negated-read | explain a concept | explain a concept | 0.038 | 0.060 |
| read-only | change settings | read existing data | 0.333 | 0.790 |
| urgent-setting | change settings | change settings | 0.335 | 0.030 |
| comparison | read existing data | read existing data | 0.541 | 0.930 |

TypeSafe reported 11,073 input tokens across 30 calls. At the published $0.042/million input tokens and free output tokens, the estimated usage cost is $0.000465; this is an estimate, not a billing receipt. Sources: [API](https://docs.typesafe.ai/api), [models/pricing](https://docs.typesafe.ai/models).

## Deployment and attestation

Repository: https://github.com/VitaDAO/open-jev-tinfoil
Release: https://github.com/VitaDAO/open-jev-tinfoil/releases/tag/v0.1.0

The new 4-CPU, 8192-MB, zero-GPU enclave was provisioned with debug and automatic updates disabled. CPU attestation and exact signed release verification pass. Both initial boot and one bounded stop/start recovery remained pending at the Docker image-pull phase; no model process or successful live inference has been observed. The image is public and its manifest is anonymously accessible. Current evidence does not establish an out-of-memory problem. App health, live rejection/concurrency tests, EHBP inference, and production Vita integration remain incomplete. Existing Vita services were not changed.

Recovery: stop only the new `open-jev` container. No automatic recovery or shared registry changes are authorized by this report. See issue #1 for the current checkpoint and exact identities.
