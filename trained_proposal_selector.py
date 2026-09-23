"""Vita read-intent head over frozen Open-JEV features; local experiment only."""
import hashlib
import json
import math
import re
from pathlib import Path

from proposal_selector import ProposalSelector, identity as proposal_identity
from routing import MODEL_REVISION
from schema_index import INDEX

ROOT = Path(__file__).resolve().parent
INTENT_SHA256 = '45946119ca98492c1d781a76eb3f153bdba0fe89cdeb2409fb9615707a93f757'


def identity():
    return hashlib.sha256(proposal_identity().encode() + Path(__file__).read_bytes() + INTENT_SHA256.encode()).hexdigest()


class TrainedProposalSelector(ProposalSelector):
    def __init__(self, model):
        super().__init__(model)
        raw = (ROOT / 'adapters/vita-read-intent-v2.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != INTENT_SHA256:
            raise ValueError('Unapproved read-intent adapter')
        data = json.loads(raw)
        if (data['format_version'] != 1 or data['model_revision'] != MODEL_REVISION
                or data['question'] != self.question.instructions):
            raise ValueError('Read-intent encoder mismatch')
        self.intent_weights = self.torch.tensor(data['weights'], dtype=self.torch.float64)
        self.intent_threshold = data['threshold']
        if (self.intent_weights.shape != (1024,) or not self.torch.isfinite(self.intent_weights).all()
                or not math.isfinite(self.intent_threshold)):
            raise ValueError('Invalid read-intent adapter')

    def intent_check(self, current):
        # Align schema synonyms with training feature names. The full original
        # request still reaches the separate proposal-coverage decision.
        aliases=INDEX['metric_aliases']
        pattern=r'(?<!\w)(?:'+'|'.join(re.escape(x) for x in sorted(aliases,key=len,reverse=True))+r')(?!\w)'
        normalized=re.sub(pattern,lambda m:aliases[m[0]].replace('_',' '),current)
        score = float(self.encode(normalized) @ self.intent_weights)
        accepted = score >= self.intent_threshold
        return {'choice':'recorded_health_read' if accepted else 'handoff',
                'score':score, 'threshold':self.intent_threshold,
                'adapter_sha256':INTENT_SHA256}, None if accepted else 'learned_read_intent_unconfirmed'

    def select(self, request, **kwargs):
        result = super().select(request, **kwargs)
        result['selector_sha256'] = identity()
        result['intent_adapter_sha256'] = INTENT_SHA256
        return result
