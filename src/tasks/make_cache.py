import json
import os

from src.tasks.utils import get_features

if __name__ == '__main__':

    for file in os.listdir('outputs'):

        if '.jsonl' not in file: continue

        with open(f'outputs/{file}', 'r') as outputs:
            outputs = [json.loads(line) for line in outputs]

        _ = get_features(outputs, cache=file)
        
