"""A head tagged with the training device must restore on a CPU-only server."""
import json
from types import SimpleNamespace
import torch
from learned_parser import verify

def test_mps_tagged_checker_head_restores_identical_cpu_weights(tmp_path, monkeypatch):
    state = torch.nn.Linear(2, 1).state_dict()
    with monkeypatch.context() as saving:
        registry = torch.serialization._package_registry
        saving.setattr(torch.serialization, '_package_registry',
            [(0, lambda storage: 'mps', lambda storage, location: None), *registry])
        torch.save(state, tmp_path/'verifier_head.pt')
    (tmp_path/'verifier.json').write_text(json.dumps({'text_version': 2}))
    encoder = torch.nn.Identity()
    encoder.config = SimpleNamespace(hidden_size=2)
    monkeypatch.setattr(verify.AutoTokenizer, 'from_pretrained', lambda path: object())
    monkeypatch.setattr(verify.AutoModel, 'from_pretrained', lambda path: encoder)
    checker = verify.Verifier(tmp_path)
    for name, tensor in checker.m.head.state_dict().items():
        assert tensor.device.type == 'cpu'
        assert torch.equal(tensor, state[name])
