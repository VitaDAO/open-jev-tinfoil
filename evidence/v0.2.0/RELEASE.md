# Routing release v0.2.0

- Accepted source:PR #2, merged as f3f7bedb96053b22f7087ba9ea67434e0512bbe4; merge tree equals reviewed branch tree4155adb749c299104887e51559fd67e81355584d.
- Tested image source:1ba182b72c8391c17a57b54541a5b082978c617c.
- Image:ghcr.io/vitadao/open-jev-tinfoil@sha256:ba7da2df50a4242721a4f5b4ca64ce1605384c887e08bb27ad55e10097f883d6.
- Build/test:Actions35903692961 passed Linux4CPU/8GB,29 unit tests, generic smoke, all54 route predictions. Route medians642–647ms on that runner. Local current suite31 tests; HTTP route medians97–99ms. Measurements are machine-specific.
- Configuration/tag commit:f3f7bedb96053b22f7087ba9ea67434e0512bbe4 / v0.2.0.
- Release workflow35904196657 and measure/publish35904218107 succeeded.
- Signed deployment digest:9868aca7564ebc9a5cd678d4f9cae9a621cde3d99b5aa44f68616037fd1cd1f9.
- Adapter SHA256:35fbdc0351280e1720dd61776a4f6492d270f1bf689858162bd6deacfcfc12a6.
- Model revision:19bf9a64815add579fbf6c907bef584d9277a8e4.
- Target:container6cc904e0-f08a-4162-b478-57321a8f42f1, open-jev.vitality-now.containers.tinfoil.dev, control.inf6.tinfoil.sh,4CPUs/8192MB/0GPUs; confidential mode on; debug and automatic updates off.
- Only secret binding:OPEN_JEV_API_KEY; no secret value recorded. No migrations, no Vita app deployment.
- Recovery:stop this standalone container or return to v0.1.0. The previous tag never reached healthy inference, so it is a configuration rollback, not a known-working service.

The manifest SHA256 matches tinfoil.hash and its decoded config is byte-identical to tinfoil-config.yml. The new image manifest is publicly pullable without credentials. Publishing attestation is distinct from live readiness. Live evidence is tracked in issue #1 and subsequent verification files.
