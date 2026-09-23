"""Train a small frozen-encoder read-intent head, never on evaluation fixtures.

Synthetic template families are development data. Calibration uses separate
wording families, not random rows of the same template. Not clinical validation.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from learned_selector import LearnedSelector
from proposal_binding import canonicalize
from routing import MODEL_REVISION


def dataset():
    targets = ['ApoB', 'LDL cholesterol', 'oxygen saturation', 'respiratory rate',
               'sleep efficiency', 'steps', 'total sleep', 'custom metric',
               'lab reports', 'workouts', 'profile', 'health plans']
    positive = ['Show my {x}', 'What is my latest {x}?', 'How has my {x} evolved?',
                'Pull up my {x} for last month', 'Review my {x} from 2025',
                'Could you display my {x} during February?', 'I want to see my {x}',
                'What {x} have I got on file?', 'Let me inspect my {x} history',
                'Can you retrieve my {x} from the past 14 days?', 'My {x} yesterday please',
                'Tell me about my recorded {x}', 'Show {x} between June 3 and June 17',
                'What was my {x} like in 2024?', 'Graph my {x} over time',
                'Summarize my {x} this year']
    negative = ['What is {x}?', 'Explain what {x} means', 'Why does {x} matter?',
                'Is {x} a medical term?', 'How does {x} work in general?',
                'Does exercise cause changes in my {x}?', 'Does my {x} affect my sleep?',
                'Is my {x} higher than last year?', 'Compare my {x} with last month',
                'Show {x} only while I was sick', 'Show {x} near my office',
                'Find {x} restricted to manual entries', 'Show the earliest {x}',
                'Send my {x} to someone else', 'Change my {x}', 'What did you say about my {x} before?',
                'Was ist {x}?', 'Zeig mir meine {x}', 'Montre mes {x}',
                'Tell me if my {x} is normal', 'How many {x} measurements do I have?']
    train = [{'request':template.format(x=x), 'label':label}
             for label, templates in [(1,positive),(0,negative)] for template in templates for x in targets]
    train += [{'request':s,'label':1} for s in ['Analyze me','Analize me','Analyse my health',
        'Assess my overall health', 'Give me a health summary', 'How is my health doing?',
        'I want a descriptive health overview','Evaluate my health and suggest improvements',
        'Give me a comprehensive health analysis','Show my complete health profile']]
    calibration = []
    for label, templates in [(1,['Please fetch my {x} data for me', 'What do my stored {x} measurements show?']),
                             (0,['Teach me about the meaning of {x}', 'Is there a connection between my {x} and my mood?'])]:
        for template in templates:
            for x in targets[::2]:calibration.append({'request':template.format(x=x),'label':label})
    return train, calibration


def main():
    import torch
    from typed_decisions.open_jev import OpenJev
    torch.set_num_threads(4)
    model = OpenJev.from_pretrained(str(ROOT / 'model-fp16'), device='cpu')
    model.collator._ids=lambda t:model.tok(t,add_special_tokens=False)['input_ids']
    model.collator._cache.clear()
    encoder = LearnedSelector(model)
    train, calibration = dataset()
    x = torch.stack([encoder.encode(canonicalize(r['request'])) for r in train])
    y = torch.tensor([2*r['label']-1 for r in train], dtype=torch.float64)
    cx = torch.stack([encoder.encode(canonicalize(r['request'])) for r in calibration])
    cy = torch.tensor([r['label'] for r in calibration])
    # Select regularization and acceptance threshold only on the separate
    # calibration families. The evaluation fixtures are never loaded here.
    candidates=[]
    gram=x@x.T
    for alpha in (.001,.003,.01,.03,.1):
        weights=x.T@torch.linalg.solve(gram+alpha*torch.eye(len(x),dtype=x.dtype),y)
        scores=cx@weights
        threshold=max(0.,float(scores[cy==0].max())+.05)
        candidates.append((int(((scores>=threshold)&(cy==1)).sum()),alpha,threshold,weights,scores))
    _,alpha,threshold,weights,scores=max(candidates,key=lambda item:(item[0],item[1]))
    report={'training_cases':len(train),'calibration_cases':len(calibration),'threshold':threshold,
        'alpha':alpha,'calibration_search':[{'accepted_correct':n,'alpha':a,'threshold':t} for n,a,t,_,_ in candidates],
        'calibration_accepted_correct':int(((scores>=threshold)&(cy==1)).sum()),
        'calibration_accepted_wrong':int(((scores>=threshold)&(cy==0)).sum()),
        'calibration_rows':[{**r,'score':float(s),'accepted':bool(s>=threshold)} for r,s in zip(calibration,scores)],
        'note':'Threshold fitted on these calibration families; zero wrong accepts here is not an independent guarantee.'}
    dataset_bytes=(json.dumps({'train':train,'calibration':calibration},indent=2)+'\n').encode()
    (ROOT/'evidence/selector-v3/read-intent-training.json').write_bytes(dataset_bytes)
    data={'format_version':1,'model_revision':MODEL_REVISION,'question':encoder.question.instructions,
        'dataset_sha256':hashlib.sha256(dataset_bytes).hexdigest(), 'alpha':alpha,
        'threshold':threshold,'weights':weights.tolist()}
    path=ROOT/'adapters/vita-read-intent-v1.json';path.write_text(json.dumps(data,separators=(',',':'))+'\n')
    report['adapter_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'evidence/selector-v3/read-intent-calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='calibration_rows'},indent=2))


if __name__=='__main__':main()
