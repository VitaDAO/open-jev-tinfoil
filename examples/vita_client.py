"""The single Vita client. Attestation precedes credentials or private requests.

Caller supplies the reviewed release digest. Selector pins are also required for
/v1/select. No approved selector release is embedded or learned from the server.
Run in Vita's trusted backend; execution still uses its own capability broker.
"""
import os
import re

from routing import MODEL_REVISION, ADAPTER_SHA256
from query_plan import QueryRequest, QueryPlan, compile_batch

HOST = 'open-jev.vitality-now.containers.tinfoil.dev'
REPO = 'VitaDAO/open-jev-tinfoil'


class VitaClient:
    def __init__(self, *, release_digest, selector_sha256=None, adapter_sha256=None):
        for value in (release_digest,):
            if not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{64}',value):
                raise ValueError('Reviewed release SHA256 pin required')
        if (selector_sha256 is None)!=(adapter_sha256 is None):
            raise ValueError('Both selector pins are required')
        for value in (selector_sha256,adapter_sha256):
            if value is not None and (not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{64}',value)):
                raise ValueError('Reviewed selector SHA256 pins required')
        from tinfoil import SecureClient
        class ReleasePinnedClient(SecureClient):
            def verify(inner):
                ground_truth=super().verify()
                document=inner.get_verification_document()
                # The SDK re-verifies after TLS key rotation. Check the release
                # before it builds a replacement transport or retries a request,
                # not only when this application client is constructed.
                if document is None or document.security_verified is not True or document.release_digest!=release_digest:
                    raise RuntimeError('Unapproved enclave release')
                return ground_truth
        self.verifier=ReleasePinnedClient(enclave=HOST,repo=REPO,transport='tls')
        self.http=self.verifier.make_secure_http_client()
        try:
            document=self.verifier.get_verification_document()
            if document is None or document.security_verified is not True or document.release_digest!=release_digest:
                raise RuntimeError('Unapproved enclave release')
            self.token=os.environ['OPEN_JEV_API_KEY']
        except Exception:
            self.http.close()
            raise
        self.selector_sha256=selector_sha256
        self.adapter_sha256=adapter_sha256

    def select(self, request):
        if not self.selector_sha256 or not self.adapter_sha256:
            raise ValueError('Reviewed selector pins required for /v1/select')
        request=QueryRequest.model_validate(request)
        response=self.http.post(f'https://{HOST}/v1/select',
            headers={'Authorization':f'Bearer {self.token}'},
            json=request.model_dump(mode='json'),timeout=15)
        response.raise_for_status()
        raw=response.json()
        if not isinstance(raw,dict) or raw.get('schema_version')!='vita-query-plan/v2' or raw.get('advisory') is not True:
            raise RuntimeError('Unexpected selector wire contract')
        result=QueryPlan.model_validate(raw)
        if (result.selector_sha256!=self.selector_sha256 or result.adapter_sha256!=self.adapter_sha256
                or result.model_revision!=MODEL_REVISION):
            raise RuntimeError('Unexpected selector identity')
        # Validate request identity, inventory, every clause and operation budget.
        # The compiled batch is acquisition only, not a completed answer.
        compile_batch(result,request)
        return result

    def decide(self, state, questions):
        response = self.http.post(
            f'https://{HOST}/decide',
            headers={'Authorization': f'Bearer {self.token}'},
            json={'state': state, 'questions': questions},
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
        if result.get('model_revision') != MODEL_REVISION:
            raise RuntimeError('Unexpected model revision')
        return result

    def route(self, state):
        response = self.http.post(
            f'https://{HOST}/route',
            headers={'Authorization': f'Bearer {self.token}'},
            json={'state': state}, timeout=15,
        )
        response.raise_for_status()
        result = response.json()
        if (result.get('model_revision') != MODEL_REVISION
                or result.get('adapter_sha256') != ADAPTER_SHA256
                or result.get('advisory') is not True
                or result.get('action') not in ['read existing data', 'explain a concept', 'change settings']
                or type(result.get('record_access')) is not bool):
            raise RuntimeError('Unexpected routing response')
        return result

    def close(self):
        self.http.close()
