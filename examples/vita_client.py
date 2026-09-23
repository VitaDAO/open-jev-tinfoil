"""Run within Vita's trusted backend/enclave, not the user's browser.

pip install -r examples/client-requirements.txt
OPEN_JEV_API_KEY must be supplied by the caller's secret store.
No plaintext fallback: verification/network errors propagate to the caller.
"""
import os

HOST = 'open-jev.vitality-now.containers.tinfoil.dev'
REPO = 'VitaDAO/open-jev-tinfoil'
RELEASE_DIGEST = 'ad1f173859cb04203b367800cb4ae5b031e0ec1e606336b3e4c9fdcfca9c380f'
ADAPTER_SHA256 = '35fbdc0351280e1720dd61776a4f6492d270f1bf689858162bd6deacfcfc12a6'
MODEL_REVISION = '19bf9a64815add579fbf6c907bef584d9277a8e4'

class VitaDecisionClient:
    def __init__(self):
        from tinfoil import SecureClient
        self.verifier = SecureClient(enclave=HOST, repo=REPO, transport='tls')
        # Attestation verifies code against signed repo release, and binds TLS.
        self.http = self.verifier.make_secure_http_client()
        document = self.verifier.get_verification_document()
        if document is None or not document.security_verified or document.release_digest != RELEASE_DIGEST:
            self.http.close()
            raise RuntimeError('Enclave does not match the approved v0.2.1 release')
        self.token = os.environ['OPEN_JEV_API_KEY']

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

if __name__ == '__main__':
    client = VitaDecisionClient()
    try:
        print(client.decide('Please show my sleep trends.', [
            {'type': 'choice', 'instructions': 'Which topic is requested?',
             'options': ['sleep', 'nutrition', 'exercise', 'other']}
        ]))
    finally:
        client.close()
