"""Verify serialized adapter against recorded fit/confirmation predictions."""
import json,statistics,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from examples.local_learned_router import LocalLearnedRouter
start=time.perf_counter();router=LocalLearnedRouter();load_ms=(time.perf_counter()-start)*1000
recorded=json.loads((root/'evidence/local-v3-adapter.json').read_text())
report={'load_ms':load_ms,'results':{}}
for name,path,split in [('target24','confirmation.json','train'),('fresh30','fresh-v3.json','fresh')]:
 reference={r['case']:r for r in recorded['results'][split]['rows']}
 cases=json.loads((root/'experiments'/path).read_text());rows=[]
 for case,state,choice,read in cases:
  result=router.route(state)
  assert (result['action'],result['record_access']) == (reference[case]['action'],reference[case]['record_access'])
  rows.append({'case':case,'choice_correct':result['action']==choice,'record_correct':result['record_access']==read,'result':result})
 report['results'][name]={'rows':rows,'summary':{'cases':len(rows),'choice_correct':sum(r['choice_correct'] for r in rows),'record_correct':sum(r['record_correct'] for r in rows),'median_ms':statistics.median(r['result']['elapsed_ms'] for r in rows),'max_ms':max(r['result']['elapsed_ms'] for r in rows)}}
assert len(router.model.collator._cache)==0
for state in ['', 'word '*300]:
 try:router.route(state)
 except ValueError:pass
 else:raise AssertionError('Invalid input accepted')
report['negative_checks']='empty and overlong rejected; request cache stays empty; serialized predictions match'
(root/'evidence/local-v3-verified.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v['summary'] for k,v in report['results'].items()},indent=2))
