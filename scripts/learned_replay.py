"""Record neural outputs, replay the real decoder/linker without encoder weights.

Only use trusted local recordings. The archive contains Python regex objects.
No sealed datasets are accepted by the recorder.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from learned_parser.decode import Parser


def key(batch):
    return hashlib.sha256(batch['input_ids'].numpy().tobytes()).hexdigest()


class RecordedModel:
    def __init__(self, data):
        self.frames = data['frames']
        self.tok_proj = data['tok_proj']
        self.link_scale = data['link_scale']
        self.link_bias = data['link_bias']

    def __call__(self, batch, items):
        # Clone because downstream link processing must not mutate a recording.
        return {n: t.clone() for n, t in self.frames[key(batch)].items()}


def record(parser, requests, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    parser._pack = False  # record individual passes; packing has its own parity gate
    parser.tok.save_pretrained(directory / 'tokenizer')
    state = {k: copy.deepcopy(v) for k, v in vars(parser).items()
             if k not in ('tok', 'model', 'I', 'acute') and not k.startswith('_')}
    data = {'state': state, 'I': parser.I.clone(), 'frames': {},
            'tok_proj': copy.deepcopy(parser.model.tok_proj),
            'link_scale': parser.model.link_scale.detach().clone(),
            'link_bias': parser.model.link_bias.detach().clone()}

    def hook(model, args, output):
        data['frames'][key(args[0])] = {n: t.detach().clone() for n, t in output.items()}

    handle = parser.model.register_forward_hook(hook)
    results = []
    try:
        with torch.inference_mode():
            for request in requests:
                results.append(parser.select(request))
    finally:
        handle.remove()
    torch.save(data, directory / 'neural.pt')
    (directory / 'cases.json').write_text(json.dumps(list(zip(requests, results)), indent=1))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [directory / 'neural.pt', directory / 'cases.json']}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return results


def load(directory):
    directory = Path(directory)
    for name, digest in json.loads((directory / 'manifest.json').read_text()).items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Replay recording checksum mismatch')
    data = torch.load(directory / 'neural.pt', map_location='cpu', weights_only=False)
    parser = Parser.__new__(Parser)
    parser.__dict__.update(data['state'])
    parser.I = data['I']
    parser.model = RecordedModel(data)
    parser.tok = AutoTokenizer.from_pretrained(directory / 'tokenizer', local_files_only=True)
    from learned_parser.decode import KeywordAcute
    parser.acute = KeywordAcute()
    parser._pack = False
    return parser, json.loads((directory / 'cases.json').read_text())


def verify(directory):
    parser, cases = load(directory)
    with torch.inference_mode():
        for request, expected in cases:
            actual = parser.select(request)
            assert actual == expected, (request['state']['current_request'], expected, actual)
    return len(cases)
