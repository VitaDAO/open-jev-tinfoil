"""Derive FP16-stored backbone from pinned FP32 source; compute remains FP32.
Keep the small generic head unchanged. Record source and derived file hashes.
"""
import hashlib,json,sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def pack(root):
    path=root/'model.safetensors'
    source=digest(path)
    with safe_open(path,framework='pt') as f:metadata=f.metadata()
    weights=load_file(path)
    if not any(t.dtype==torch.float32 for t in weights.values()):
        raise ValueError('Expected original FP32 weights')
    packed={k:t.to(torch.float16) if t.dtype==torch.float32 else t for k,t in weights.items()}
    temporary=root/'model.fp16.safetensors'
    save_file(packed,temporary,metadata=metadata)
    temporary.chmod(0o644)  # The serving container runs as non-root UID10001.
    temporary.replace(path)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['source_backbone_sha256']=source
    manifest['weight_storage_dtype']='float16'
    manifest['compute_dtype']='float32'
    manifest['files']={p.name:digest(p) for p in sorted(root.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':pack(Path(sys.argv[1]))
