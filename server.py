"""Authenticated, bounded CPU inference. No request logging or text retention."""
import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from starlette.concurrency import run_in_threadpool

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

class Engine:
    def __init__(self):
        import torch
        from typed_decisions.open_jev import OpenJev
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
        torch.set_num_interop_threads(1)
        self.model = OpenJev.from_pretrained(os.environ.get('MODEL_DIR', '/opt/model'), device='cpu')
        # Upstream Collator caches input strings indefinitely. Use no cache at all.
        self.model.collator._ids = lambda text: self.model.tok(text, add_special_tokens=False)['input_ids']
        self.model.collator._cache.clear()
        self.decide(DecisionRequest(state='This is a test.', questions=[Question(type='noul', instructions='This is a test.')]))

    def decide(self, request):
        state_tokens = len(self.model.tok(request.state, add_special_tokens=False)['input_ids'])
        if state_tokens > self.model.collator.max_state:
            raise ValueError('State exceeds 256 tokens; shorten it explicitly')
        questions = [q.model_dump(exclude_none=True) for q in request.questions]
        started = time.perf_counter()
        answers = self.model.decide(request.state, questions)
        return {'answers': answers, 'model_revision': MODEL_REVISION,
                'inference_ms': round((time.perf_counter() - started) * 1000, 2)}

def create_app(engine_factory=Engine, token=None):
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
    lock = asyncio.Lock()
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Pydantic's default error includes raw user inputs; do not echo them.
        return JSONResponse({'detail': 'Invalid request schema'}, status_code=422)
    @app.get('/health')
    async def health():
        return {'status': 'ready', 'model_revision': MODEL_REVISION, 'device': 'cpu',
                'max_state_tokens': 256, 'max_sequence_tokens': 512}
    @app.post('/decide')
    async def decide(request: DecisionRequest):
        if lock.locked():
            raise HTTPException(429, 'Inference busy; retry later', headers={'Retry-After': '1'})
        async with lock:
            try:
                return await run_in_threadpool(app.state.engine.decide, request)
            except ValueError:
                raise HTTPException(422, 'Input exceeds model token limits') from None
            except Exception:
                raise HTTPException(500, 'Inference failed') from None
    return app
