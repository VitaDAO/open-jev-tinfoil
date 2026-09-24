# Compact routing release v0.2.1

- Runtime/image source:2a96084e2caa9215aaa3958444c6ad8caf9426a2.
- Tested image:ghcr.io/vitadao/open-jev-tinfoil@sha256:40e09453a5a74f7039ef59c65b2b1b790573a894c4be0c4480e400fde7212356.
- Build35905764231:32 tests and all54 route predictions pass under4CPU/8GB, plus generic smoke. Route median648–649ms on the Linux runner.
- Configuration/tag commit:f29d437cfaccca1d2afe5a7f0be5b8f42fb44119, v0.2.1; PR #3 merged with equal source/target trees.
- Release/measurement workflows35906306674 and35906336709 succeeded.
- Signed deployment digest:ad1f173859cb04203b367800cb4ae5b031e0ec1e606336b3e4c9fdcfca9c380f.
- Adapter SHA256:35fbdc0351280e1720dd61776a4f6492d270f1bf689858162bd6deacfcfc12a6, unchanged.
- Model source revision:19bf9a64815add579fbf6c907bef584d9277a8e4; FP16-stored derived backbone, FP32 computation. Source/derived file hashes in image `/opt/model/manifest.json`.
- Registry storage estimate:1,162,743,011 compressed +2,108,667,392 uncompressed tar =3,271,410,403 bytes, below4GiB. This is not live RAM consumption.
- Target:open-jev /6cc904e0-f08a-4162-b478-57321a8f42f1, open-jev.vitality-now.containers.tinfoil.dev,4CPUs/8192MB/0GPUs, debug and automatic updates disabled.
- Only secret binding:OPEN_JEV_API_KEY. No migrations, other services, Vita app edits or real health data used.
- Recovery:stop only this standalone container. Prior v0.2.0/v0.1.0 configurations never reached working inference and are not known-working service rollbacks.

Manifest SHA256 and exact decoded configuration verified. Live verification is recorded separately after startup; tag publication alone does not prove readiness.

Current verified status: pull/start completed, then container failed health (`unhealthy`). No hosted inference has succeeded. Read-only temporary-directory mismatch is being reproduced; live attestation alone does not establish app readiness.
