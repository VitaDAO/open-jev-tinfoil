"""Authenticated, bounded CPU inference. No request logging or text retention."""
import asyncio
import hashlib
import hmac
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from starlette.concurrency import run_in_threadpool
from routing import LocalLearnedRouter, ADAPTER_SHA256
from query_plan import QueryRequest, QueryPlan
from query_selector import (QuerySelector, identity as selector_identity, INTENT_SHA256, REASON_CODES, REASON_CODES_BYTES,
                            learned_parser_enabled, learned_parser, learned_parser_identity)

MODEL_REVISION = '19bf9a64815add579fbf6c907bef584d9277a8e4'
Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=4096)]

class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['choice', 'score', 'noul']
    instructions: Text
    options: list[Text] | None = Field(default=None, max_length=255)

    @model_validator(mode='after')
    def check_options(self):
        if self.type == 'noul':
            if self.options is not None:
                raise ValueError('noul has fixed no/yes options; omit options')
        elif self.options is None or not 2 <= len(self.options) <= (10 if self.type == 'score' else 255):
            raise ValueError('invalid option count')
        if self.options and len(set(self.options)) != len(self.options):
            raise ValueError('duplicate options')
        return self

class RouteRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    state: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=16384)]

class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    state: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16384)]
    questions: list[Question] = Field(min_length=1, max_length=32)

class RequestGuard:
    """Authenticate before reading bodies, and cap even chunked request bodies."""
    def __init__(self, app, token):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        async def reject(status, detail):
            await JSONResponse({'detail': detail}, status_code=status)(scope, receive, send)
        if scope['path'] != '/health':
            headers = dict(scope['headers'])
            expected = ('Bearer ' + self.token).encode()
            if not hmac.compare_digest(headers.get(b'authorization', b''), expected):
                return await reject(401, 'Unauthorized')
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > 65536:
                return await reject(413, 'Request body exceeds 64 KiB')
            if not message.get('more_body', False):
                break
        delivered = False
        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()
        await self.app(scope, bounded_receive, send)

class InferenceQueue:
    """One running inference, at most 16 FIFO waiters, 500 ms queue deadline."""
    capacity = 16
    wait_seconds = 0.5

    def __init__(self):
        self.lock = asyncio.Lock()
        self.waiting = 0

    @staticmethod
    def busy():
        return HTTPException(429, 'Inference busy; retry later', headers={'Retry-After': '1'})

    async def run(self, request, function, payload):
        acquired = False
        try:
            if not self.lock.locked() and not self.waiting:
                await self.lock.acquire()
                acquired = True
            else:
                if self.waiting >= self.capacity:
                    raise self.busy()
                self.waiting += 1
                acquire = asyncio.create_task(self.lock.acquire())
                async def disconnected():
                    while True:
                        if (await request.receive())['type'] == 'http.disconnect':
                            return
                disconnect = asyncio.create_task(disconnected())
                try:
                    done, _ = await asyncio.wait(
                        (acquire, disconnect), timeout=self.wait_seconds,
                        return_when=asyncio.FIRST_COMPLETED)
                    if disconnect in done:
                        # Client is gone; remove its waiter without running inference.
                        raise HTTPException(499, 'Client disconnected')
                    if acquire not in done:
                        raise self.busy()
                    acquire.result()
                finally:
                    for task in (acquire, disconnect):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(acquire, disconnect, return_exceptions=True)
                    acquired = (not acquire.cancelled() and acquire.exception() is None
                                and acquire.result())
                    self.waiting -= 1

            # Cancelling a request cannot stop a Python inference thread. Keep the
            # slot until that thread actually finishes, preventing overlapping work.
            worker = asyncio.create_task(run_in_threadpool(function, payload))
            cancelled = False
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    break
            if cancelled:
                # Retrieve any exception before propagating request cancellation.
                if not worker.cancelled():
                    worker.exception()
                raise asyncio.CancelledError
            return worker.result()
        finally:
            if acquired:
                self.lock.release()

class Engine:
    def __init__(self):
        import torch
        from typed_decisions.open_jev import OpenJev
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
        torch.set_num_interop_threads(1)
        self.model = OpenJev.from_pretrained(os.environ.get('MODEL_DIR', '/opt/model'), device='cpu')
        if next(self.model.model.backbone.parameters()).dtype != torch.float32:
            raise RuntimeError('Backbone must compute in FP32')
        with open(os.path.join(os.environ.get('MODEL_DIR', '/opt/model'), 'manifest.json')) as manifest:
            self.weight_storage_dtype = json.load(manifest).get('weight_storage_dtype', 'float32')
        # Upstream Collator caches input strings indefinitely. Use no cache at all.
        self.model.collator._ids = lambda text: self.model.tok(text, add_special_tokens=False)['input_ids']
        self.model.collator._cache.clear()
        self.router = LocalLearnedRouter(model=self.model)
        self.selector = QuerySelector(self.model)
        if learned_parser_enabled():   # load and warm the learned parser before serving (OPEN_JEV_PARSER=learned)
            learned_parser()
            self.learned_parser_sha256 = learned_parser_identity()
        self.router.route('This is a test.')
        self.decide(DecisionRequest(state='This is a test.', questions=[Question(type='noul', instructions='This is a test.')]))

    def select(self, request):
        return self.selector.select_query(request)

    def route(self, request):
        return {**self.router.route(request.state), 'weight_storage_dtype': self.weight_storage_dtype,
                'backbone_compute_dtype': 'float32'}

    def decide(self, request):
        state_tokens = len(self.model.tok(request.state, add_special_tokens=False)['input_ids'])
        if state_tokens > self.model.collator.max_state:
            raise ValueError('State exceeds 256 tokens; shorten it explicitly')
        questions = [q.model_dump(exclude_none=True) for q in request.questions]
        started = time.perf_counter()
        answers = self.model.decide(request.state, questions)
        return {'answers': answers, 'model_revision': MODEL_REVISION,
                'inference_ms': round((time.perf_counter() - started) * 1000, 2),
                'weight_storage_dtype': self.weight_storage_dtype, 'backbone_compute_dtype': 'float32'}

def create_app(engine_factory=Engine, token=None, selector_enabled=None):
    selector_enabled = (os.environ.get("ENABLE_EXPERIMENTAL_SELECTOR") == "1"
                        if selector_enabled is None else selector_enabled)
    token = token if token is not None else os.environ.get('OPEN_JEV_API_KEY', '')
    if len(token) < 32:
        raise RuntimeError('OPEN_JEV_API_KEY must contain at least 32 characters')
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine_factory()
        yield
        app.state.engine = None
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestGuard, token=token)
    queue = InferenceQueue()
    app.state.inference_queue = queue
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Pydantic's default error includes raw user inputs; do not echo them.
        return JSONResponse({'detail': 'Invalid request schema'}, status_code=422)
    @app.get('/health')
    async def health():
        return {'status': 'ready', 'model_revision': MODEL_REVISION, 'device': 'cpu',
                'inference_queue_capacity': queue.capacity, 'inference_queue_wait_ms': int(queue.wait_seconds * 1000),
                'max_state_tokens': 256, 'max_sequence_tokens': 512, 'adapter_sha256': ADAPTER_SHA256,
                'selector_enabled': selector_enabled, 'selector_schema': 'vita-selector/v2',
                'selector_sha256': selector_identity(), 'selector_adapter_sha256': INTENT_SHA256,
                'reason_codes': sorted(REASON_CODES), 'reason_codes_sha256': hashlib.sha256(REASON_CODES_BYTES).hexdigest(),
                'selector_parser': 'learned' if learned_parser_enabled() else 'rules',
                'learned_parser_sha256': getattr(app.state.engine, 'learned_parser_sha256', None)}

    @app.post('/v1/select', response_model=QueryPlan)
    async def select_reads(request: QueryRequest, http_request: Request):
        if not selector_enabled:
            raise HTTPException(503, "Experimental selector has not passed acceptance")
        try:
            return await queue.run(http_request, app.state.engine.select, request)
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(422, 'Input exceeds model token limits') from None
        except Exception:
            raise HTTPException(500, 'Inference failed') from None

    @app.post('/decide')
    async def decide(request: DecisionRequest, http_request: Request):
        try:
            return await queue.run(http_request, app.state.engine.decide, request)
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(422, 'Input exceeds model token limits') from None
        except Exception:
            raise HTTPException(500, 'Inference failed') from None
    @app.post('/route')
    async def route(request: RouteRequest, http_request: Request):
        try:
            return await queue.run(http_request, app.state.engine.route, request)
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(422, 'Input exceeds model token limits') from None
        except Exception:
            raise HTTPException(500, 'Inference failed') from None
    return app
