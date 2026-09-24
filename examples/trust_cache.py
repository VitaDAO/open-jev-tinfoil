"""Public-only five-minute TUF snapshots; never cache clients, tokens or requests.

Parent calls refresh before admitting children and again before refresh_due.
Children receive snapshot() and construct their own verifier and TLS transport.
"""
from dataclasses import dataclass
import hashlib
import asyncio
import base64
import json
import subprocess
import sys
import os
from pathlib import Path
from threading import Lock
import time

MAX_AGE_SECONDS = 300
REFRESH_AFTER_SECONDS = 240
WORKER_TIMEOUT_SECONDS = 30


def _validated_root():
    from importlib.metadata import version
    if version('sigstore') != '4.5.0' or version('tuf') != '6.0.0':
        raise RuntimeError('trust_cache_dependency_mismatch')
    from sigstore._internal.tuf import TrustUpdater, DEFAULT_TUF_URL
    updater = TrustUpdater(DEFAULT_TUF_URL)  # Online TUF signature/rollback/expiry checks.
    path = updater.get_trusted_root_path()
    target = updater._updater.get_targetinfo('trusted_root.json')
    if target is None:
        raise RuntimeError('trust_root_unavailable')
    raw = Path(path).read_bytes()
    target.verify_length_and_hashes(raw)
    # Read authenticated metadata in memory, not independently reread disk files.
    expires = min(metadata.expires.timestamp()
                  for metadata in updater._updater._trusted_set)
    return raw, expires


@dataclass(frozen=True)
class TrustSnapshot:
    root_bytes: bytes
    sha256: str
    issued_at: float
    issued_monotonic: float
    expires_at: float

    def assert_valid(self):
        now, elapsed = time.time(), time.monotonic() - self.issued_monotonic
        if (not 0 <= elapsed < MAX_AGE_SECONDS or now < self.issued_at
                or now >= self.expires_at
                or self.expires_at > self.issued_at + MAX_AGE_SECONDS):
            raise RuntimeError('trust_snapshot_expired')
        if hashlib.sha256(self.root_bytes).hexdigest() != self.sha256:
            raise RuntimeError('trust_snapshot_integrity')

    def make_verifier(self):
        self.assert_valid()
        from sigstore.models import TrustedRoot, trustroot_v1
        from sigstore.verify import Verifier
        # New process-local crypto objects from immutable, TUF-authenticated bytes.
        return Verifier(trusted_root=TrustedRoot(trustroot_v1.TrustedRoot.from_json(self.root_bytes)))


def _refresh_snapshot(issued, monotonic):
    raw, signed_expiry = _validated_root()
    candidate = TrustSnapshot(raw, hashlib.sha256(raw).hexdigest(), issued,
                              monotonic, min(issued + MAX_AGE_SECONDS, signed_expiry))
    candidate.assert_valid()
    candidate.make_verifier()
    return candidate


def _worker_spec(issued, monotonic):
    env = {key:value for key,value in os.environ.items() if key in (
        'PATH','HOME','TMPDIR','SYSTEMROOT','XDG_CACHE_HOME','XDG_DATA_HOME',
        'SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE','PYTHONPATH')}
    return ([sys.executable, str(Path(__file__).resolve()),
             '--refresh-worker', str(issued), str(monotonic)], env)


def _decode_snapshot(stdout, issued, monotonic):
    data = json.loads(stdout)
    raw = base64.b64decode(data.pop('root_base64'), validate=True)
    candidate = TrustSnapshot(root_bytes=raw, **data)
    if candidate.issued_at != issued or candidate.issued_monotonic != monotonic:
        raise RuntimeError('trust_snapshot_clock_mismatch')
    candidate.assert_valid()
    return candidate


def _fetch_snapshot(issued, monotonic):
    args, env = _worker_spec(issued, monotonic)
    response = subprocess.run(args, env=env, capture_output=True, text=True,
                              timeout=WORKER_TIMEOUT_SECONDS, check=True)
    return _decode_snapshot(response.stdout, issued, monotonic)


def _check_async_watcher():
    # asyncio's default macOS child watcher starts a thread per helper. Never
    # silently change the host's process-wide event loop or signal policy.
    uvloop = sys.modules.get('uvloop')
    if uvloop is not None and isinstance(asyncio.get_running_loop(), uvloop.Loop):
        return  # uvloop owns its subprocess transport; no stdlib child watcher.
    watcher = asyncio.get_child_watcher()
    allowed = tuple(getattr(asyncio, name) for name in
                    ('PidfdChildWatcher', 'SafeChildWatcher', 'FastChildWatcher')
                    if hasattr(asyncio, name))
    if not isinstance(watcher, allowed):
        raise RuntimeError('trust_refresh_requires_nonthreaded_child_watcher')


async def _cleanup_worker(launch):
    process = await launch
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.communicate()  # Drain pipes and reap even after cancellation.


async def _finish_cleanup(task):
    # Repeated cancellation must not abandon a spawned helper.
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


async def _async_fetch_snapshot(issued, monotonic):
    _check_async_watcher()
    args, env = _worker_spec(issued, monotonic)
    launch = asyncio.create_task(asyncio.create_subprocess_exec(
        *args, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE))
    try:
        async with asyncio.timeout(WORKER_TIMEOUT_SECONDS):
            process = await asyncio.shield(launch)
            stdout, _ = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError('trust_refresh_worker_failed')
        return _decode_snapshot(stdout, issued, monotonic)
    except BaseException:
        # Shield launch too: cancellation can arrive after OS spawn but before
        # create_subprocess_exec returns the process handle.
        cleanup = asyncio.create_task(_cleanup_worker(launch))
        await _finish_cleanup(cleanup)
        raise


class PublicTrustCache:
    def __init__(self):
        self._owner_pid = os.getpid()
        self._refresh_lock = Lock()
        self._snapshot = None

    def _check_owner(self):
        if os.getpid() != self._owner_pid:
            raise RuntimeError('trust_cache_parent_only')

    def refresh(self):
        self._check_owner()  # Before touching an inherited lock.
        if not self._refresh_lock.acquire(blocking=False):
            raise RuntimeError('trust_refresh_in_progress')
        try:
            issued, monotonic = time.time(), time.monotonic()
            candidate = _fetch_snapshot(issued, monotonic)
            candidate.assert_valid()
            self._snapshot = candidate  # Publish atomically only after successful verification.
            return candidate
        finally:
            self._refresh_lock.release()

    async def async_refresh(self):
        self._check_owner()
        # Never wait on a threading lock in the host event loop. The same gate
        # serializes sync and async refreshes; cancellation while waiting is inert.
        while not self._refresh_lock.acquire(blocking=False):
            await asyncio.sleep(.01)
        try:
            issued, monotonic = time.time(), time.monotonic()
            candidate = await _async_fetch_snapshot(issued, monotonic)
            candidate.assert_valid()
            self._snapshot = candidate
            return candidate
        finally:
            self._refresh_lock.release()

    def snapshot(self):
        self._check_owner()
        value = self._snapshot
        if value is None:
            raise RuntimeError('trust_cache_not_ready')
        value.assert_valid()
        return value

    @property
    def refresh_due(self):
        self._check_owner()
        try:
            value = self.snapshot()
        except RuntimeError:
            return True
        return (time.monotonic() - value.issued_monotonic >= REFRESH_AFTER_SECONDS
                or value.expires_at - time.time() <= MAX_AGE_SECONDS - REFRESH_AFTER_SECONDS)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--refresh-worker':
        raise SystemExit('Expected public refresh worker invocation')
    value = _refresh_snapshot(float(sys.argv[2]), float(sys.argv[3]))
    print(json.dumps({'root_base64':base64.b64encode(value.root_bytes).decode(),
        'sha256':value.sha256, 'issued_at':value.issued_at,
        'issued_monotonic':value.issued_monotonic, 'expires_at':value.expires_at}))
