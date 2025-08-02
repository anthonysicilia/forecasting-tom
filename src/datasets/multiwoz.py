import json
import hashlib
import pandas as pd

if __name__ == '__main__':

    with open('raw/MWOZ.txt', 'r') as text:
        lines = text.readlines()

    data = []
    inputs = ''
    for line in lines:

        if 'OVERALL' in line:
            scores = line.split('\t')[-1].split(',')
            target = sum(map(float, scores)) / len(scores)
            data.append({
                'input' : inputs,
                'gt': target,
                'context': 'a human user is interacting with an AI task assistant',
                'question': 'is the user that they (the user) are more satisfied than would occur by chance',
                'instance_id': hashlib.md5(inputs.encode('utf-8')).hexdigest()
            })
            inputs = ''
        else:
            if not line.strip():
                continue
            inputs += '\n' + ': '.join(line.split('\t')[0:2])
    
    mags = [x['gt'] for x in data]
    for x in data:
        x['p'] = sum([x['gt'] > m for m in mags]) / len(mags)
    
    with open('data/multiwoz.jsonl', 'w') as out:
        for x in data:
            out.write(json.dumps(x) + '\n')