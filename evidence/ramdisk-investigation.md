# Image-pull storage investigation

Pinned Tinfoil CVM0.14.7 configures Docker data-root at `/mnt/ramdisk/private/docker` and containerd under the same RAM disk, with Docker's containerd snapshotter enabled. Its sizing code assigns a fixed4GiB RAM disk when detected RAM is below32GiB; both8GB and16GB configurations therefore use this cap.

Primary sources at exact release:
- https://github.com/tinfoilsh/cvmimage/blob/v0.14.7/image/rootfs/etc/docker/daemon.json
- https://github.com/tinfoilsh/cvmimage/blob/v0.14.7/image/rootfs/etc/containerd/config.toml
- https://github.com/tinfoilsh/cvmimage/blob/v0.14.7/tinfoil/internal/pid1/runtime/setup.go#L184

Measured v0.2.0 image footprint from registry layer lengths and gzip uncompressed-size trailers:
- compressed1,892,800,473 bytes;
- uncompressed tar2,975,918,080 bytes;
- combined4,868,718,553 bytes, versus4,294,967,296-byte RAM-disk cap.

Tar byte counts include headers/padding; this is an image-storage estimate, not observed live filesystem utilization. Containerd retains compressed blobs and unpacked snapshots. The difference is large enough to make capacity the leading explanation for repeated pull stalls; unavailable runtime logs do not directly prove ENOSPC. Attestation, configuration, network, certificates and registry setup complete; application start is never reached.

Remediation:FP16 storage for backbone, unchanged FP32 runtime and unchanged adapter. The derived manifest records original backbone hash, derived per-file hashes, source revision and storage/compute dtype. All54 local route labels remain identical; generic smoke maximum probability change0.000550747. This is a lossy storage change and does not establish parity for every possible prompt. The smaller image must pass Linux tests, fit the cap, receive a new measured release and pass live readiness before the fix is accepted.

The first FP16 packing build35905247364 failed the non-root real-model startup test: safe_open could not open the packed backbone (reported FileNotFoundError). The packing writer created mode0600. Setting derived public model weights to0644 fixed access; corrected build35905764231 passes the non-root real-model suite. A unit assertion now checks mode0644, and the CI readiness loop detects an exited container immediately instead of waiting the full timeout.
