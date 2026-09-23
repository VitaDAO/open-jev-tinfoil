"""Train a small frozen-encoder read-intent head, never on evaluation fixtures.

Synthetic template families are development data. Calibration uses separate
wording families, not random rows of the same template. Not clinical validation.
"""
import hashlib
import json
import sys
import argparse
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


def dataset_v2():
    train, _ = dataset()
    targets = ['ApoB', 'LDL cholesterol', 'oxygen saturation', 'respiratory rate',
               'sleep efficiency', 'steps', 'total sleep', 'lab reports', 'workouts']
    positive = ['Which {x} readings have been collected for me?',
        'Fetch the {x} stored in my account', 'For yesterday, what did my {x} look like?',
        'Retrieve the newest {x} value that I have', 'Can I inspect the {x} I have logged?',
        'What is the most recent {x} result available in my records?',
        'Give me a view of the recorded {x}, please', 'Bring up my {x} measurements',
        'Throughout the previous year, how were my {x} measurements doing?',
        'Let me look through my {x} records', 'My {x} from the last nine days please']
    negative = ['Are the {x} readings I have healthy?', 'What could my {x} indicate?',
        'Fetch my {x} restricted to the hospital source', 'Bring up my {x} when I was unwell',
        'Explain the newest {x} value I have', 'What should I do about my {x}?',
        'Give me a view of {x} only from my watch', 'Retrieve my {x} excluding outliers',
        'How was my {x} related to my diet?', 'What is the difference between {x} and my last result?']
    train += [{'request':template.format(x=x), 'label':label}
        for label,templates in [(1,positive),(0,negative)] for template in templates for x in targets]
    for a,b in [('ApoB','LDL cholesterol'),('steps','workouts'),('total sleep','oxygen saturation')]:
        for template in ['Bring up my {a} and {b} for 2024', 'What have my {a} and {b} looked like this year?',
                         'Show the recorded {a} together with my {b}']:
            train.append({'request':template.format(a=a,b=b),'label':1})
    for noun in ['health appointments','screening events','health plans']:
        for template in ['Do I have any {x} scheduled for next week?', 'Which {x} are due tomorrow?',
                         'Bring up the {x} that I saved', 'What {x} were completed yesterday?',
                         'Are there any {x} coming up next year?']:
            train.append({'request':template.format(x=noun),'label':1})
        for template in ['Create {x} for tomorrow', 'Cancel my {x}', 'How many {x} are due tomorrow?']:
            train.append({'request':template.format(x=noun),'label':0})
    train += [{'request':s,'label':1} for s in ['Bring up my whole health profile',
        'Open my stored profile', 'I would like to look at my health profile',
        'Show a descriptive summary of my health without advice', 'Summarize all of my health data']]
    # New calibration wording families, fixed before either evaluation is read.
    calibration=[]
    for label,templates in [(1,['Could you fetch the personal {x} history kept for me?',
                                'Can we go through my stored {x} results?']),
                            (0,['Do my stored {x} results suggest a disease?',
                                'Retrieve personal {x} only on days when I exercised'])]:
        for template in templates:
            for x in targets:calibration.append({'request':template.format(x=x),'label':label})
    return train, calibration


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version', choices=(1,2), type=int, default=1)
    args=parser.parse_args()
    output=ROOT / ('evidence/selector-v3' if args.version==1 else 'evidence/selector-v4')
    import torch
    from typed_decisions.open_jev import OpenJev
    torch.set_num_threads(4)
    model = OpenJev.from_pretrained(str(ROOT / 'model-fp16'), device='cpu')
    model.collator._ids=lambda t:model.tok(t,add_special_tokens=False)['input_ids']
    model.collator._cache.clear()
    encoder = LearnedSelector(model)
    train, calibration = (dataset if args.version==1 else dataset_v2)()
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
    # Equal classification counts do not justify the largest regularizer: that
    # can shift both classes toward a high acceptance threshold. Break ties by
    # calibration squared error against the training target, never eval scores.
    loss=lambda item:float(((item[4]-(cy.double()*2-1))**2).mean())
    chosen=max(candidates,key=lambda item:(item[0],-loss(item),-item[1])) if args.version==2 else max(candidates,key=lambda item:(item[0],item[1]))
    _,alpha,threshold,weights,scores=chosen
    report={'training_cases':len(train),'calibration_cases':len(calibration),'threshold':threshold,
        'alpha':alpha,'calibration_search':[{'accepted_correct':item[0],'alpha':item[1],
            'threshold':item[2],'squared_error':loss(item)} for item in candidates],
        'calibration_accepted_correct':int(((scores>=threshold)&(cy==1)).sum()),
        'calibration_accepted_wrong':int(((scores>=threshold)&(cy==0)).sum()),
        'calibration_rows':[{**r,'score':float(s),'accepted':bool(s>=threshold)} for r,s in zip(calibration,scores)],
        'note':'Threshold fitted on these calibration families; zero wrong accepts here is not an independent guarantee.'}
    dataset_bytes=(json.dumps({'train':train,'calibration':calibration},indent=2)+'\n').encode()
    (output/'read-intent-training.json').write_bytes(dataset_bytes)
    data={'format_version':1,'model_revision':MODEL_REVISION,'question':encoder.question.instructions,
        'dataset_sha256':hashlib.sha256(dataset_bytes).hexdigest(), 'alpha':alpha,
        'threshold':threshold,'weights':weights.tolist()}
    path=ROOT/f'adapters/vita-read-intent-v{args.version}.json';path.write_text(json.dumps(data,separators=(',',':'))+'\n')
    report['adapter_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'read-intent-calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='calibration_rows'},indent=2))


if __name__=='__main__':main()
