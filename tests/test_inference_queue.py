"""Bounded queue behavior without loading model weights."""
import asyncio
import threading
import pytest
from fastapi import HTTPException
from server import InferenceQueue

class Connection:
    def __init__(self):
        self.closed = asyncio.Event()
    async def receive(self):
        await self.closed.wait()
        return {'type': 'http.disconnect'}

async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(.001)

def test_sixteen_waiters_fifo_and_seventeenth_rejected():
    async def check():
        queue = InferenceQueue()
        queue.wait_seconds = 2
        started, release = threading.Event(), threading.Event()
        calls = []
        def infer(n):
            calls.append(n)
            if n == 0:
                started.set()
                assert release.wait(3)
            return n
        active = asyncio.create_task(queue.run(Connection(), infer, 0))
        await until(started.is_set)
        tasks = []
        try:
            for i in range(1, 17):
                tasks.append(asyncio.create_task(queue.run(Connection(), infer, i)))
                await until(lambda: queue.waiting == i)
            with pytest.raises(HTTPException) as exc:
                await queue.run(Connection(), infer, 17)
            assert exc.value.status_code == 429
            assert exc.value.headers == {'Retry-After': '1'}
            assert calls == [0]
        finally:
            release.set()
        assert await asyncio.gather(active, *tasks) == list(range(17))
        assert calls == list(range(17))
        assert queue.waiting == 0 and not queue.lock.locked()
    asyncio.run(check())

@pytest.mark.parametrize('removal', ['timeout', 'disconnect', 'cancel'])
def test_removed_waiter_never_runs_and_frees_capacity(removal):
    async def check():
        queue = InferenceQueue()
        queue.wait_seconds = .03 if removal == 'timeout' else 2
        await queue.lock.acquire()
        calls = []
        connection = Connection()
        task = asyncio.create_task(queue.run(connection, calls.append, 'removed'))
        await until(lambda: queue.waiting == 1)
        if removal == 'disconnect':
            connection.closed.set()
        elif removal == 'cancel':
            task.cancel()
        if removal == 'cancel':
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(HTTPException) as exc:
                await task
            assert exc.value.status_code == (429 if removal == 'timeout' else 499)
        assert queue.waiting == 0 and calls == []
        queue.lock.release()
        await queue.run(Connection(), calls.append, 'next')
        assert calls == ['next'] and not queue.lock.locked()
    asyncio.run(check())

def test_cancelled_active_request_does_not_overlap_inference():
    async def check():
        queue = InferenceQueue()
        queue.wait_seconds = 2
        started, release = threading.Event(), threading.Event()
        calls = []
        def infer(n):
            calls.append(n)
            if n == 0:
                started.set()
                assert release.wait(3)
            return n
        first = asyncio.create_task(queue.run(Connection(), infer, 0))
        await until(started.is_set)
        first.cancel()
        second = asyncio.create_task(queue.run(Connection(), infer, 1))
        await until(lambda: queue.waiting == 1)
        try:
            assert queue.lock.locked() and calls == [0]
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == 1 and calls == [0, 1]
        assert not queue.lock.locked()
    asyncio.run(check())

def test_inference_error_releases_slot():
    async def check():
        queue = InferenceQueue()
        def fail(_):
            raise ValueError('synthetic')
        with pytest.raises(ValueError):
            await queue.run(Connection(), fail, None)
        assert await queue.run(Connection(), lambda _: 'recovered', None) == 'recovered'
    asyncio.run(check())

@pytest.mark.parametrize('removal', ['disconnect', 'cancel'])
def test_removal_racing_with_slot_release_does_not_leak_lock(removal):
    async def check():
        queue = InferenceQueue()
        await queue.lock.acquire()
        calls = []
        connection = Connection()
        task = asyncio.create_task(queue.run(connection, calls.append, 'removed'))
        await until(lambda: queue.waiting == 1)
        queue.lock.release()
        if removal == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            connection.closed.set()
            with pytest.raises(HTTPException) as exc:
                await task
            assert exc.value.status_code == 499
        assert calls == [] and queue.waiting == 0 and not queue.lock.locked()
        await queue.run(Connection(), calls.append, 'next')
        assert calls == ['next']
    asyncio.run(check())
