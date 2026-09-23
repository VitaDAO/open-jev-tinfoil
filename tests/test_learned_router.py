import copy
import json
from pathlib import Path
import pytest
from examples.local_learned_router import LocalLearnedRouter, validate_artifact

ARTIFACT = json.loads((Path(__file__).resolve().parents[1]/'adapters/vita-intent-v1.json').read_text())


def test_saved_adapter_has_supported_identity_and_finite_weights():
    validate_artifact(ARTIFACT)


@pytest.mark.parametrize('mutation', ['revision','shape','nonfinite'])
def test_invalid_adapter_is_rejected(mutation):
    data = copy.deepcopy(ARTIFACT)
    if mutation == 'revision': data['model_revision'] = 'wrong'
    elif mutation == 'shape': data['weights'].pop()
    else: data['weights'][0][0] = float('nan')
    with pytest.raises(ValueError): validate_artifact(data)


@pytest.mark.parametrize('state', ['', '   ', None, 'x'*65537])
def test_invalid_state_is_rejected_before_inference(state):
    router = LocalLearnedRouter.__new__(LocalLearnedRouter)
    with pytest.raises(ValueError): router.route(state)
