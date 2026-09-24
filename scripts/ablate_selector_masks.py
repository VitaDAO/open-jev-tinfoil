"""Diagnostic ablation on the former held-out59; no longer independent validation."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor'),str(ROOT/'scripts')]
from typed_decisions.open_jev import OpenJev
from learned_selector import LearnedSelector
from selector import SelectorRequest
from evaluate_selector import evaluate

if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    model=OpenJev.from_pretrained(str(ROOT/'model-fp16'),device='cpu')
    model.collator._ids=lambda text:model.tok(text,add_special_tokens=False)['input_ids'];model.collator._cache.clear()
    selector=LearnedSelector(model)
    os.environ['SELECTOR_CASES']=str(ROOT/'evidence/selector-v1/model-heldout.json')
    os.environ['EVALUATION_LABEL']='Diagnostic development reuse after baseline failures; NOT fresh held-out validation. Same learned weights and .10 threshold.'
    cached={};responses={}
    def baseline(body):
        key=json.dumps(body,sort_keys=True);r=selector.select(SelectorRequest.model_validate(body))
        cached[key]=(r['diagnostics']['predicted_decisions'],r['decision_margins']);responses[key]=r
        return r
    all_heads=evaluate(baseline)
    def conditional(body):
        return selector.select(SelectorRequest.model_validate(body),confidence_policy='conditional',
                               prediction=cached[json.dumps(body,sort_keys=True)])
    conditional_report=evaluate(conditional)
    changes=[]
    for before,after in zip(all_heads['results'],conditional_report['results']):
        if before['status']!=after['status'] or before['correct']!=after['correct']:
            changes.append({'id':before['id'],'before_status':before['status'],'before_correct':before['correct'],
                            'after_status':after['status'],'after_correct':after['correct']})
    print(json.dumps({'all_heads':all_heads,'conditional':conditional_report,'changed':changes,
        'decision':'reject if incorrect_selected increases; these scores are dispositions, not full database plan correctness',
        'diagnostics':list(responses.values())},indent=2))
