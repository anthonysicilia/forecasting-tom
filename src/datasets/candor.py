import hashlib
import json
import os
import pandas as pd
import numpy as np
import sys

if __name__ == '__main__':

    second_order = False
    if len(sys.argv) > 1: second_order = True
    data = []

    for convo in os.listdir('raw/no_media'):

        if convo == '.DS_Store':
            continue
        transcript = pd.read_csv(f'raw/no_media/{convo}/transcription/transcript_audiophile.csv')
        speakers = transcript['speaker'].unique()
        norm = [s for s in list(sorted(speakers))]
        utts = []
        for s, u in zip(transcript['speaker'], transcript['utterance']):
            utts.append(f'Speaker {norm.index(s)}: {u}')
        inputs = '\n'.join(utts)
        survey = pd.read_csv(f'raw/no_media/{convo}/survey.csv')
        A = speakers[0]
        B = speakers[1]

        fo_gt = survey[survey['user_id'] == A]['i_like_you'].unique()[0]
        so_gt = survey[survey['user_id'] == B]['you_like_me'].unique()[0]

        demographics = {
            f'Speaker {norm.index(s)}' : {
                k : survey[survey['user_id'] == s][k].unique()[0]
                for k in ['age', 'sex', 'race', 'edu', 'politics']

            }
            for s in speakers
        }

        data.append({
            'input' : inputs,
            'gt': so_gt if second_order else fo_gt,
            'context': 'two strangers are talking through a video communication platform',
            'question': 'is Speaker 1 that Speaker 0 likes them (Speaker 1) more than would occur by chance' \
                if second_order else 'is Speaker 0 that they (Speaker 0) like Speaker 1 more than would occur by chance',
            'demographics': demographics,
            'instance_id': hashlib.md5(inputs.encode('utf-8')).hexdigest()
        })
    
    mags = [x['gt'] for x in data]
    for x in data:
        x['p'] = sum([x['gt'] > m for m in mags]) / len(mags)
    
    file = 'data/candor-so.jsonl' if second_order else 'data/candor-fo.jsonl'
    with open(file, 'w') as out:
        for x in data:
            out.write(json.dumps(x) + '\n')



