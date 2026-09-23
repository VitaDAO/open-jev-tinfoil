"""Local selector-only probe of canonical prompts; never loads patient answers.

Supply the canonical prompt-only JSON separately. Output contains IDs, plans and
source hashes, not the input prompt text. This is not a browser or answer test.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
from query_selector import QuerySelector,identity
from query_plan import QueryRequest,compile_batch
from schema_index import CATALOG,INDEX

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prompts',type=Path,required=True)
 parser.add_argument('--output',type=Path,default=ROOT/'evidence/selector-v5/vita-battery20-selector.json')
 args=parser.parse_args();raw=args.prompts.read_bytes();prompts=json.loads(raw)['prompts']
 assert len(prompts)==20
 import torch
 from typed_decisions.open_jev import OpenJev
 torch.set_num_threads(4);model=OpenJev.from_pretrained(str(ROOT/'model-fp16'),device='cpu')
 model.collator._ids=lambda t:model.tok(t,add_special_tokens=False)['input_ids'];model.collator._cache.clear()
 selector=QuerySelector(model);start=identity();history=[];rows=[]
 aliases={v:k for k,v in INDEX['inventory_to_catalog'].items()}
 inventory=sorted(aliases.get(m,m) for m in CATALOG)
 sources=sorted({s for m in CATALOG.values() for s in m['valid_sources']})
 for case in prompts:
  req=QueryRequest.model_validate({'schema_version':'vita-selector/v2','available_metrics':inventory,
   'available_record_types':['profile','labs','workouts','calendar'],'available_sources':sources,
   'literature_available':True,'reference_time':'2026-09-24T12:00:00Z',
   'state':{'current_request':case['prompt'],'recent_user_requests':history[-4:],'reference_date':'2026-09-24','time_zone':'UTC'}})
  began=time.perf_counter();result=selector.select_query(req);batch=compile_batch(result,req)
  rows.append({'id':case['id'],'status':result['status'],'queries':result['queries'],'reasons':result['reason_codes'],
   'operation_count':len(batch['required_operation_ids']) if batch else 0,'elapsed_ms':(time.perf_counter()-began)*1000})
  history.append(case['prompt'])
 assert start==identity()
 report={'scope':'Exact canonical prompt sequence with last four user prompts only; synthetic full catalog inventory, no user records, no assistant answers, no browser, DeepSeek or final answer evaluation.',
  'selector_sha256':start,'prompt_source_sha256':hashlib.sha256(raw).hexdigest(),
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'metric_inventory_count':len(inventory),
  'planned':sum(r['status']=='planned' for r in rows),'handoff':sum(r['status']=='handoff' for r in rows),'total':20,'rows':rows}
 args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
