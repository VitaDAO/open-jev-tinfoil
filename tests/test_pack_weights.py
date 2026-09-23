import hashlib,json
import torch
from safetensors.torch import save_file,load_file
from scripts.pack_weights import pack


def test_pack_records_source_and_derived_hashes_without_changing_head(tmp_path):
    source=tmp_path/'model.safetensors';head=tmp_path/'head.safetensors'
    save_file({'weight':torch.tensor([.123456,1.0],dtype=torch.float32)},source,metadata={'format':'pt'})
    save_file({'weight':torch.ones(1)},head)
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest();head_bytes=head.read_bytes()
    (tmp_path/'manifest.json').write_text(json.dumps({'revision':'synthetic'}))
    pack(tmp_path)
    m=json.loads((tmp_path/'manifest.json').read_text())
    assert m['source_backbone_sha256']==source_sha
    assert m['files']['model.safetensors']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert m['weight_storage_dtype']=='float16' and m['compute_dtype']=='float32'
    assert load_file(source)['weight'].dtype==torch.float16
    assert head.read_bytes()==head_bytes
