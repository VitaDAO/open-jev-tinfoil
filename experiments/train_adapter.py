"""Fit a small ridge intent adapter on frozen DeBERTa state embeddings.
Training uses previous synthetic development/validation cases only. Fresh-v3 is
confirmation-only and is never consulted for feature, regularization or fit.
"""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'vendor'))
from typed_decisions.open_jev import OpenJev
from typed_decisions.schema import Question

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
model=OpenJev.from_pretrained(str(root/'model'),device='cpu')
model.collator._ids=lambda text:model.tok(text,add_special_tokens=False)['input_ids']
model.collator._cache.clear()
train=json.loads((root/'evidence/typesafe-comparison.json').read_text())['cases']+json.loads((root/'experiments/heldout.json').read_text())+json.loads((root/'experiments/confirmation.json').read_text())
fresh=json.loads((root/'experiments/fresh-v3.json').read_text())
assert hashlib.sha256((root/'experiments/fresh-v3.json').read_bytes()).hexdigest() == json.loads((root/'evidence/local-v3-search.json').read_text())['fresh_sha256']
labels=['read existing data','explain a concept','change settings']
q=Question('intent','choice','Classify the requested action.',labels,0)

@torch.inference_mode()
def embed(state):
 ids=model.tok(state,add_special_tokens=False)['input_ids']
 if len(ids)>256:raise ValueError('State exceeds 256 tokens')
 b=model.collator([(state,[q])],model.device)
 h=model.model.backbone(input_ids=b['input_ids'],attention_mask=b['attention_mask']).last_hidden_state
 return h[0,2:2+len(ids)].mean(0).double()
start=time.perf_counter()
x=torch.stack([embed(c[1]) for c in train]);print('Embedded training',len(train),flush=True)
x=torch.nn.functional.normalize(x,dim=1)
y=torch.tensor([[float(c[2]==l) for l in labels]+[float(not c[3]),float(c[3])] for c in train],dtype=torch.float64)
# Kernel ridge closed form; analytic leave-one-out chooses regularization without fresh labels.
k=x@x.T;eye=torch.eye(len(train),dtype=torch.float64)
trials=[]
for lam in [.0001,.001,.01,.1,1.,10.]:
 inv=torch.linalg.inv(k+lam*eye);alpha=inv@y
 loo=y-alpha/inv.diag()[:,None]
 good=int((loo[:,:3].argmax(1)==y[:,:3].argmax(1)).sum()+(loo[:,3:].argmax(1)==y[:,3:].argmax(1)).sum())
 trials.append((good,lam))
best=max(trials)[1]
w=x.T@torch.linalg.solve(k+best*eye,y)
print('LOO',trials,'selected',best,flush=True)
report={'training_cases':len(train),'method':'frozen backbone mean state embedding, L2 normalized; linear ridge outputs; lambda selected by analytic leave-one-out total correct; ties prefer larger lambda. No calibrated probabilities.','regularization':best,'loo':trials,'fresh_sha256':hashlib.sha256((root/'experiments/fresh-v3.json').read_bytes()).hexdigest(),'results':{}}
for split,cases in [('train',train),('fresh',fresh)]:
 rows=[]
 for i,c in enumerate(cases):
  t=time.perf_counter();v=x[i] if split=='train' else torch.nn.functional.normalize(embed(c[1]),dim=0)
  s=v@w;pred=labels[int(s[:3].argmax())];read=bool(s[3:].argmax())
  rows.append({'case':c[0],'action':pred,'record_access':read,'choice_correct':pred==c[2],'record_correct':read==c[3],'scores':s.tolist(),'ms':(time.perf_counter()-t)*1000})
 report['results'][split]={'rows':rows,'choice_correct':sum(r['choice_correct'] for r in rows),'record_correct':sum(r['record_correct'] for r in rows)}
 print(split,report['results'][split]['choice_correct'],report['results'][split]['record_correct'],len(cases),flush=True)
(root/'evidence/local-v3-adapter.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'experiments/intent-adapter.retrained.json').write_text(json.dumps({'model_revision':'19bf9a64815add579fbf6c907bef584d9277a8e4','labels':labels,'regularization':best,'question':q.instructions,'weights':w.float().tolist(),'format_version':1,'training_cases':len(train),'selection':'Regularization selected by analytic leave-one-out on previous synthetic cases; fresh-v3 excluded','training_source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'evidence/typesafe-comparison.json',root/'experiments/heldout.json',root/'experiments/confirmation.json']}},indent=2)+'\n')
