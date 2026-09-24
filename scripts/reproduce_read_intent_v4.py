"""Reproduce the frozen v4 intent head from its synthetic development protocol."""
import os,sys,json,hashlib,time
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
args=p.parse_args();OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
if any((OUT/n).exists() for n in ('vita-read-intent-v4.json','v4-calibration.json')):
    raise ValueError('Output already exists')
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from proposal_binding import canonicalize
from routing import MODEL_REVISION
raw=(ROOT/'evidence/selector-v7/v4-training-protocol.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='3931dff809a8d2d4aff6254c2fb606b32bdcfc5e59acc2e3d6992c587af2d541'
protocol=json.loads(raw);train=protocol['train'];cal=protocol['calibration']
(OUT/'v4-training-protocol.json').write_bytes(raw)
import torch
from typed_decisions.open_jev import OpenJev
from learned_selector import LearnedSelector
torch.set_num_threads(4);model=OpenJev.from_pretrained(str(args.model_dir),device='cpu');model.collator._ids=lambda t:model.tok(t,add_special_tokens=False)['input_ids'];model.collator._cache.clear();encoder=LearnedSelector(model)
cache={}
def encode(text):
 text=canonicalize(text)
 if text not in cache:cache[text]=encoder.encode(text)
 return cache[text]
xs=[]
for i,r in enumerate(train):
 xs.append(encode(r['request']))
 if (i+1)%100==0:print('encoded',i+1,flush=True)
x=torch.stack(xs);y=torch.tensor([2*r['label']-1 for r in train],dtype=torch.float64)
cx=torch.stack([encode(r['request']) for r in cal]);cy=torch.tensor([r['label'] for r in cal]);gram=x@x.T
candidates=[]
for alpha in protocol['alphas']:
 w=x.T@torch.linalg.solve(gram+alpha*torch.eye(len(x),dtype=x.dtype),y);scores=cx@w
 threshold=max(0.,float(scores[cy==0].max())+.05)
 tp=int(((scores>=threshold)&(cy==1)).sum());fp=int(((scores>=threshold)&(cy==0)).sum());loss=float(((scores-(cy.double()*2-1))**2).mean())
 candidates.append((tp,-loss,-alpha,w,threshold,scores,fp))
best=max(candidates,key=lambda c:c[:3]);tp,negative_loss,negative_alpha,w,threshold,scores,fp=best
artifact={'format_version':1,'model_revision':MODEL_REVISION,'question':encoder.question.instructions,'dataset_sha256':hashlib.sha256((OUT/'v4-training-protocol.json').read_bytes()).hexdigest(),'alpha':-negative_alpha,'threshold':threshold,'weights':w.tolist()}
(OUT/'vita-read-intent-v4.json').write_text(json.dumps(artifact,separators=(',',':'))+'\n')
report={'training_cases':len(train),'calibration_cases':len(cal),'true_positive':tp,'false_positive':fp,'threshold':threshold,'alpha':-negative_alpha,'search':[{'alpha':-c[2],'tp':c[0],'fp':c[6],'threshold':c[4],'mse':-c[1]} for c in candidates],'rows':[{**r,'score':float(s),'accepted':bool(s>=threshold)} for r,s in zip(cal,scores)]}
(OUT/'v4-calibration.json').write_text(json.dumps(report,indent=2)+'\n');torch.save(cache,OUT/'training-features.pt');print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
