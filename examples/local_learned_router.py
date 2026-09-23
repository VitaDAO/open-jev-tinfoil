"""Experimental in-process router with a learned head over frozen Open Jev.

No outbound calls or request cache. Scores are uncalibrated ridge outputs, not
probabilities. Decisions are advisory and never grant permissions.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = '19bf9a64815add579fbf6c907bef584d9277a8e4'
LABELS = ['read existing data', 'explain a concept', 'change settings']


def validate_artifact(data):
    if data.get('format_version') != 1 or data.get('model_revision') != MODEL_REVISION or data.get('labels') != LABELS:
        raise ValueError('Unsupported adapter identity')
    import math
    weights = data.get('weights', [])
    if len(weights) != 1024 or any(len(row) != 5 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row) for row in weights):
        raise ValueError('Invalid adapter weights')


class LocalLearnedRouter:
    def __init__(self, model_dir=None, adapter_path=None):
        import torch
        sys.path.insert(0, str(ROOT/'vendor'))
        from typed_decisions.open_jev import OpenJev
        from typed_decisions.schema import Question
        data = json.loads(Path(adapter_path or ROOT/'adapters/vita-intent-v1.json').read_text())
        validate_artifact(data)
        torch.set_num_threads(4)
        self.torch = torch
        self.model = OpenJev.from_pretrained(str(model_dir or ROOT/'model'), device='cpu')
        self.model.collator._ids = lambda text: self.model.tok(text, add_special_tokens=False)['input_ids']
        self.model.collator._cache.clear()
        self.question = Question('intent', 'choice', data['question'], LABELS, 0)
        self.weights = torch.tensor(data['weights'], dtype=torch.float64)

    def route(self, state):
        if not isinstance(state, str) or not state.strip() or len(state.encode()) > 65536:
            raise ValueError('State must be nonempty text of at most 64 KiB')
        started = time.perf_counter()
        torch = self.torch
        with torch.inference_mode():
            tokens = self.model.tok(state, add_special_tokens=False)['input_ids']
            if not 1 <= len(tokens) <= 256:
                raise ValueError('State must contain 1 to 256 tokens')
            batch = self.model.collator([(state, [self.question])], self.model.device)
            hidden = self.model.model.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
            pooled = hidden[0, 2:2+len(tokens)].mean(0).double()
            scores = torch.nn.functional.normalize(pooled, dim=0) @ self.weights
        return {'action': LABELS[int(scores[:3].argmax())],
                'record_access': bool(scores[3:].argmax()),
                'decision_scores': {'action': dict(zip(LABELS, scores[:3].tolist())),
                                    'record_access': dict(zip(['no', 'yes'], scores[3:].tolist()))},
                'elapsed_ms': round((time.perf_counter()-started)*1000, 2),
                'mode': 'frozen_encoder_learned_adapter', 'model_revision': MODEL_REVISION}


if __name__ == '__main__':
    router = LocalLearnedRouter()
    print(json.dumps(router.route('What is my currently selected language? Please leave it unchanged.'), indent=2))
