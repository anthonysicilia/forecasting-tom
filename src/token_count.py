import transformers
import json

if __name__ == '__main__':

    tkn = transformers.AutoTokenizer.from_pretrained('gpt2', padding_side='left')

    files = [
        'data/candor-fo.jsonl',
        'data/casino.jsonl',
        'data/multiwoz.jsonl'
    ]

    for file in files:
        with open(file, 'r') as data:
            data = [json.loads(line) for line in data]
        lens = [len(tkn(x['input'])['input_ids']) for x in data]
        print(file, sum(lens) / len(lens), min(lens), max(lens))