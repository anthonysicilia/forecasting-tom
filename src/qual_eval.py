import json
import argparse
import matplotlib.pyplot as plt


from src.tasks.utils import get_basics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--bag', type=int, default=0)
    parser.add_argument('--tuq', type=int, default=0) # 2TUQ == EUQ with slight changes
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_samp', type=int, default=10)
    args = parser.parse_args()

    import matplotlib
    matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1, 3, figsize=(8,3), sharex=True, sharey=True)

    model = 'gpt-4o-2024-05-13'
    fo_file = f'outputs/{model}-candor-fo-d=1.jsonl'
    so_file = f'outputs/{model}-candor-so-d=1.jsonl'

    with open(fo_file, 'r') as outputs:
        fo_outputs = [json.loads(line) for line in outputs]
    with open(so_file, 'r') as outputs:
        so_outputs = [json.loads(line) for line in outputs]

    fo_probas = get_basics(fo_outputs, args)['p_hat']
    so_probas = get_basics(so_outputs, args)['p_hat']
    diffs = [b-a for a,b in zip(fo_probas, so_probas)]

    ax.flat[0].hist(diffs, bins=5, color='r')
    ax.flat[0].set_xlabel('GPT 4o')

    model = 'open-mixtral-8x22b'
    fo_file = f'outputs/{model}-candor-fo-d=1.jsonl'
    so_file = f'outputs/{model}-candor-so-d=1.jsonl'

    with open(fo_file, 'r') as outputs:
        fo_outputs = [json.loads(line) for line in outputs]
    with open(so_file, 'r') as outputs:
        so_outputs = [json.loads(line) for line in outputs]

    fo_probas = get_basics(fo_outputs, args)['p_hat']
    so_probas = get_basics(so_outputs, args)['p_hat']
    diffs = [b-a for a,b in zip(fo_probas, so_probas)]

    ax.flat[1].hist(diffs, bins=5, color='b')
    ax.flat[1].set_xlabel('Mixtral 8x22B')


    fo_probas = get_basics(fo_outputs, args)['p_true']
    so_probas = get_basics(so_outputs, args)['p_true']
    diffs = [b-a for a,b in zip(fo_probas, so_probas)]

    ax.flat[2].hist(diffs, bins=5, color='g')
    ax.flat[2].set_xlabel('Ground Truth')

    fig.suptitle('Predicted and True Difference in Perspective (CANDOR)')

    plt.tight_layout()
    plt.show()

