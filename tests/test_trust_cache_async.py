import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import pytest
from examples import trust_cache as tc

uvloop=pytest.importorskip('uvloop')


def run(coro):
    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        return runner.run(coro)


def fake_worker(monkeypatch, *, delay=0, pid_file=None, exit_code=0):
    original=tc._worker_spec
    def spec(issued, monotonic):
        _,env=original(issued,monotonic)
        code=f'''
import os,sys,time,json,base64,hashlib
path={str(pid_file)!r}
if path!='None':open(path,'w').write(str(os.getpid()))
time.sleep({delay})
raw=b'public-test-root'
print(json.dumps({{'root_base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),
'issued_at':float(sys.argv[1]),'issued_monotonic':float(sys.argv[2]),'expires_at':float(sys.argv[1])+300}}))
sys.exit({exit_code})
'''
        return [sys.executable,'-c',code,str(issued),str(monotonic)],env
    monkeypatch.setattr(tc,'_worker_spec',spec)


def test_uvloop_refresh_keeps_event_loop_responsive_and_fork_safe(monkeypatch):
    fake_worker(monkeypatch,delay=.1)
    async def scenario():
        cache=tc.PublicTrustCache();ticks=[]
        task=asyncio.create_task(cache.async_refresh())
        while not task.done():
            ticks.append(1);await asyncio.sleep(.01)
        value=await task
        assert cache.snapshot() is value and len(ticks)>=3
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pid=os.fork()
            if pid==0:
                try:value.assert_valid();os._exit(0)
                except BaseException:os._exit(1)
            assert os.waitpid(pid,0)[1]==0
        assert not [w for w in caught if issubclass(w.category,DeprecationWarning)]
    run(scenario())


def test_async_cancel_kills_reaps_and_preserves_previous_expiry(monkeypatch,tmp_path):
    pid_file=tmp_path/'pid';fake_worker(monkeypatch,delay=30,pid_file=pid_file)
    async def scenario():
        cache=tc.PublicTrustCache()
        old=tc.TrustSnapshot(b'old',hashlib.sha256(b'old').hexdigest(),time.time(),time.monotonic(),time.time()+90)
        cache._snapshot=old
        task=asyncio.create_task(cache.async_refresh())
        async with asyncio.timeout(3):
            while not pid_file.exists():await asyncio.sleep(.01)
        pid=int(pid_file.read_text());task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        with pytest.raises(ChildProcessError):os.waitpid(pid,os.WNOHANG)
        assert cache.snapshot() is old
        assert cache._refresh_lock.acquire(blocking=False)
        cache._refresh_lock.release()
    run(scenario())


def test_async_failure_and_decode_failure_do_not_publish(monkeypatch):
    fake_worker(monkeypatch,exit_code=2)
    async def scenario():
        cache=tc.PublicTrustCache()
        with pytest.raises(RuntimeError,match='worker_failed'):await cache.async_refresh()
        with pytest.raises(RuntimeError,match='not_ready'):cache.snapshot()
        fake_worker(monkeypatch)
        monkeypatch.setattr(tc,'_decode_snapshot',lambda *a: (_ for _ in ()).throw(ValueError('invalid snapshot')))
        with pytest.raises(ValueError):await cache.async_refresh()
        assert cache._snapshot is None
    run(scenario())


def test_sync_async_are_serialized_without_blocking_loop(monkeypatch):
    fake_worker(monkeypatch,delay=.1)
    async def scenario():
        cache=tc.PublicTrustCache();first=asyncio.create_task(cache.async_refresh())
        await asyncio.sleep(.01)
        with pytest.raises(RuntimeError,match='in_progress'):cache.refresh()
        second=asyncio.create_task(cache.async_refresh());await asyncio.sleep(.01)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):await second
        assert not first.done()
        value=await first;assert cache.snapshot() is value
        assert await cache.async_refresh() is not value
    run(scenario())


def test_threaded_watcher_refused_before_spawn(monkeypatch):
    async def scenario():
        watcher=asyncio.ThreadedChildWatcher()
        monkeypatch.setattr(asyncio,'get_child_watcher',lambda:watcher)
        with pytest.raises(RuntimeError,match='nonthreaded'):
            await tc.PublicTrustCache().async_refresh()
    asyncio.run(scenario())


def test_async_timeout_kills_and_reaps(monkeypatch,tmp_path):
    pid_file=tmp_path/'pid';fake_worker(monkeypatch,delay=30,pid_file=pid_file)
    monkeypatch.setattr(tc,'WORKER_TIMEOUT_SECONDS',.2)
    async def scenario():
        cache=tc.PublicTrustCache()
        with pytest.raises(TimeoutError):await cache.async_refresh()
        pid=int(pid_file.read_text())
        with pytest.raises(ChildProcessError):os.waitpid(pid,os.WNOHANG)
        assert cache._snapshot is None
    run(scenario())


def test_cancellation_during_launch_and_repeated_cancel_reaps(monkeypatch,tmp_path):
    pid_file=tmp_path/'pid';fake_worker(monkeypatch,delay=30,pid_file=pid_file)
    original=asyncio.create_subprocess_exec
    async def scenario():
        created=asyncio.Event();release=asyncio.Event();processes=[]
        async def launch(*args,**kwargs):
            process=await original(*args,**kwargs)
            processes.append(process);created.set()
            await release.wait()
            return process
        monkeypatch.setattr(asyncio,'create_subprocess_exec',launch)
        cache=tc.PublicTrustCache();task=asyncio.create_task(cache.async_refresh())
        await created.wait();task.cancel();await asyncio.sleep(.01)
        task.cancel();await asyncio.sleep(.01);release.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert processes[0].returncode is not None
        with pytest.raises(ChildProcessError):os.waitpid(processes[0].pid,os.WNOHANG)
        assert cache._snapshot is None
    run(scenario())
