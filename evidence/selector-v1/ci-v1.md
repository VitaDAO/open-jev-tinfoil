# Linux runtime checkpoint at 5f1fa06

[Build35915590517](https://github.com/VitaDAO/open-jev-tinfoil/actions/runs/35915590517) passed at source5f1fa06a693840251135cba7ef3e7cc09c5d45b4.

- Tested image sha256:a74e05b4b3525156fc80fcd0bf402b6908c9bef1f7b2363018cd0cd19f2953c9; not tagged as a Tinfoil release or deployed.
- 75 tests passed,1 skipped (external synthetic integration fixture absent). Generic and54 routing smoke checks passed.
- Full learned selector HTTP on Linux4CPU/8GB: warm n100 median727.05ms/p95755.95ms; no WAN/attestation/database/plan execution. This is model inference, distinct from grammar baseline4.11ms.
- Memory:2.043GiB reported during idle; memory.peak2,216,681,472bytes. Cgroup accounting may exclude build-charged image file pages; not live enclave RAM evidence.
- New image works with read-only root, dropped capabilities, no-new-privileges and64MiB scoped /tmp tmpfs.
- **The exact old v0.2.1 image successfully imported torch without a writable /tmp.** Exit0. Therefore simple torch import did not reproduce the suspected startup failure. Full Engine initialization under exact read-only restrictions is the next diagnostic; do not present tmpfs as a proven fix yet.
- Hosted24h metrics returned zero totals/utilizations, so they provide no usable memory/OOM evidence for the failed boot.

CI passing proves runtime/contract smoke, not language quality acceptance. The two model false accepts remain blockers; the experimental selector stays disabled by default.
