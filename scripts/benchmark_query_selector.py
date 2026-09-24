"""Paired synthetic CPU benchmark of request-local feature reuse only."""
import hashlib,json,statistics,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
from query_selector import QuerySelector,identity
from scripts.evaluate_query_plan import request

def main():
 import torch
 from typed_decisions.open_jev import OpenJev
 torch.set_num_threads(4)
 model=OpenJev.from_pretrained(str(ROOT/'model-fp16'),device='cpu')
 model.collator._ids=lambda text:model.tok(text,add_special_tokens=False)['input_ids'];model.collator._cache.clear()
 class Uncached(QuerySelector):
  def encode(self,current):return super(QuerySelector,self).encode(current)
 selectors={'reused':QuerySelector(model),'uncached':Uncached(model)}
 paths={'simple':'Show my steps yesterday','semantic':'What is the most recent ApoB value I have?',
        'two_clauses':'Show my ApoB in 2023; show my steps in 2024','profile':'Show my complete profile',
        'repeated_semantic':'What is the most recent ApoB value I have?; what is the most recent ApoB value I have?'}
 rows=[];calls=[]
 hook=model.model.backbone.register_forward_pre_hook(lambda *_:calls.append(1))
 start=identity()
 try:
  for name,text in paths.items():
   req=request(text);expected=None;measured={key:[] for key in selectors};passes={key:[] for key in selectors}
   for selector in selectors.values():selector.select_query(req)
   for i in range(12):
    for key in (list(selectors) if i%2==0 else list(reversed(selectors))):
     calls.clear();began=time.perf_counter();result=selectors[key].select_query(req)
     measured[key].append((time.perf_counter()-began)*1000);passes[key].append(len(calls))
     comparable={k:v for k,v in result.items() if k!='diagnostics'}
     if expected is None:expected=comparable
     assert result['status']=='planned' and comparable==expected,(name,key)
   rows.append({'path':name,'samples_per_arm':12,'arms':{key:{'p50_ms':statistics.median(values),'p95_ms':sorted(values)[-1],
     'backbone_passes':sorted(set(passes[key]))} for key,values in measured.items()},'identical_plans':True})
 finally:hook.remove()
 assert start==identity()
 report={'selector_sha256':start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'scope':'Paired alternating local CPU 4-thread inference; identical model and checks, reuse toggled only. No HTTP, database or enclave.',
   'rows':rows}
 (ROOT/'evidence/selector-v5/feature-reuse-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
