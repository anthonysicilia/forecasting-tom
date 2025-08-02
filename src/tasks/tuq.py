import argparse
import json
import numpy as np
import random
import pandas as pd
import os

from scipy.special import logit, expit

from scipy.stats import spearmanr, pearsonr

from src.tasks.utils import get_basics, get_features
from src.tasks.utils import explained_variance, parse_file_name, linear_regression, sgd_regression, fit_predict
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# standard platt scaling
class DoNothing:

    def fit(self, yhat, y, clipval=1000):

        self.predict = lambda x: x

        return self

# standard platt scaling
class PlattScale:

    def fit(self, yhat, y, clipval=1000):

        yhat = logit(yhat).reshape(-1, 1)
        y = logit(y).reshape(-1, 1)
        yhat = np.clip(yhat, -clipval, clipval) # expit 1000 is already 1.0
        y = np.clip(y, -clipval, clipval) # expit 1000 is already 1.0
        clf = LinearRegression().fit(yhat, y)
        self.predict = lambda x: expit(clf.predict(np.clip(logit(x), -clipval, clipval)))

        return self

def make_eu_problem(file, args):
        
        assert '-so-' in file, '2TUQ only implemented for Second-Order Candor'

        with open(f'outputs/{file}', 'r') as outputs:
            outputs = [json.loads(line) for line in outputs]
        
        basics = get_basics(outputs, args)
        p_so_hat = basics['p_hat']
        p_so_true = basics['p_true']

        fo_file = file.replace('-so-', '-fo-')
        with open(f'outputs/{fo_file}', 'r') as fo_outputs:
            fo_outputs = [json.loads(line) for line in fo_outputs]
        fo_basics = get_basics(fo_outputs, args)
        p_fo_hat = fo_basics['p_hat']
        p_fo_true = fo_basics['p_true']

        eu_hat = (p_so_hat - p_fo_hat)
        eu_hat = np.stack((p_so_hat, p_fo_hat), -1)
        if args.linear_eu:
            eu = (p_fo_true - p_so_true)
        else:
            eu = (p_fo_true - p_so_true) ** 2

        return eu_hat, eu

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('files', type=str, nargs='+',
        help='Model outputs files to evaluate. First arg is taken to be the directory for the files while args following are a list of keywords to exclude.')
    parser.add_argument('--bag', type=int, default=0,
        help='Whether bootstrap aggregating (self-consistency) should be used.')
    parser.add_argument('--human', type=int, default=0,
        help='Whether human results should be shown for CANDOR')
    parser.add_argument('--eu', type=int, default=0,
        help='Triggers 2TUQ evaluation (1TUQ otherise).')
    parser.add_argument('--linear_eu', type=int, default=0,
        help='For 2TUQ switches between FUnQ (linear) and its sqaure, which is traditional epistemic uncertainty.')
    parser.add_argument('--n_train', type=int, default=100,
        help='Number of examples to use for Random forest or regression.')
    parser.add_argument('--n_samp', type=int, default=10,
        help='Number of model samples to use from the outputs files.')
    args = parser.parse_args()

    results = []

    if args.files[0].endswith("/"):
        # first arg is a directory so just use that, all others used as exclusion cases
        exclude = args.files[1:] if len(args.files) > 1 else []
        args.files = [file for file in os.listdir(args.files[0]) if '.jsonl' in file]
        for e in exclude:
            args.files = [file for file in args.files if e not in file]
    
    emb_seen = set()

    for file in args.files:

        file = file.split('outputs/')[-1]
        mod, dat = parse_file_name(file)

        try:
            with open(f'outputs/{file}', 'r') as outputs:
                outputs = [json.loads(line) for line in outputs]
        except FileNotFoundError:
            print('Skipping', file)
            continue
        
        basics = get_basics(outputs, args)
        p_true = basics['p_true']
        p_hat = basics['p_hat']

        # check for not too many nans
        # print(file, np.mean(np.isnan(p_hat)))
        # continue

        if args.human and not args.eu:
            assert '-fo-' in file, 'human baseline only supported with First-Order Candor'
            so_file = file.replace('-fo-', '-so-')
            with open(f'outputs/{so_file}', 'r') as so_outputs:
                so_outputs = [json.loads(line) for line in so_outputs]
            so_basics = get_basics(so_outputs, args)
            p_human = so_basics['p_true']
        else:
            p_human = None

        if args.eu:
            # replaces these with predicted eu and actual eu
            # p_hat_fo, p_true_fo = make_eu_problem(file, args)   
            fo_file = file.replace('-so-', '-fo-')
            with open(f'outputs/{fo_file}', 'r') as fo_outputs:
                fo_outputs = [json.loads(line) for line in fo_outputs]
            fo_basics = get_basics(fo_outputs, args)
            p_fo_hat = fo_basics['p_hat']
            p_fo_true = fo_basics['p_true']
            eu_true = (p_true - p_fo_true) if args.linear_eu else (p_true - p_fo_true) ** 2

        for seed in [0, 1, 11, 111, 42]:

            random.seed(seed)

            train = set(random.sample([i for i in range(len(p_true))], args.n_train))
            train = np.array([i in train for i in range(len(p_true))])

            train_large = set(random.sample([i for i in range(len(p_true))], 8 * args.n_train))
            train_large = np.array([i in train_large for i in range(len(p_true))])
                
            feats = [('df', p_hat)]
            if p_human is not None and not args.eu:
                feats += [('h', p_human)]

            for name, feat in feats:

                if args.eu:
                    p_hat = fit_predict(file, p_hat, p_true, train)
                    p_fo_hat = fit_predict(file, p_fo_hat, p_fo_true, train)
                    linear_regression(results, file, name, (p_hat - p_fo_hat), eu_true, train, model=DoNothing)
                    linear_regression(results, file, name + '-c', np.stack((p_hat, p_fo_hat), -1), eu_true, train)
                else:
                    linear_regression(results, file, name + '-lr', feat, p_true, train)
                    linear_regression(results, file, name + '-ps', feat, p_true, train, model=PlattScale)
                    linear_regression(results, file, name, feat, p_true, train, model=DoNothing)
            
            if args.eu:
                linear_regression(results, file, 'df-xl', (p_hat - p_fo_hat), eu_true, train_large, model=DoNothing)
            else:
                linear_regression(results, file, name + '-lr-xl', p_hat, p_true, train_large)
 
            # P(IK) -> P(TK)
            if f'{dat}-{seed}' not in emb_seen:
                emb_seen.add(f'{dat}-{seed}')
                emb = get_features(outputs, cache=file)
                if args.eu:
                    emb_fo = get_features(outputs, cache=file.replace('-so-', '-fo-'))
                    p_hat = fit_predict(file, emb, p_true, train, model=RandomForestRegressor, max_depth=5)
                    p_fo_hat = fit_predict(file, emb_fo, p_fo_true, train, model=RandomForestRegressor, max_depth=5)
                    linear_regression(results, file, 'pik', (p_hat - p_fo_hat), eu_true, train, model=DoNothing)
                    emb = np.concatenate((emb, emb_fo), -1)
                    linear_regression(results, file, 'pik-c', emb, p_true, train_large, model=RandomForestRegressor, max_depth=5)
                else:
                    sgd_regression(results, file, 'pik', emb, p_true, train)
                    sgd_regression(results, file, 'pik-xl', emb, p_true, train_large)
                    # linear_regression(results, file, 'pik-nn', emb, p_true, train_large, model=MLPRegressor)
                    # linear_regression(results, file, 'pik-rf', emb, p_true, train_large, model=RandomForestRegressor, max_depth=5)

    df = pd.DataFrame(results)
    df['ones'] = 1
    df['s'] = np.sqrt(df['v'])
    # NOTE: example metrics below for correlation, mae
    print(df.groupby(['name'])[['r', 'rh', 'mae', 'bias', 'v']].agg(['mean']).round(3))
    print(df.groupby(['name'])[['r', 'rh', 'mae', 'bias', 'v']].agg(['std']).round(3))
    # import matplotlib.pyplot as plt
    # plt.hist(df['bias']); plt.show()
    print(df.groupby(['name'])['r2'].agg(['mean', 'min', 'max']).round(3))
    print(df.groupby(['model','name'])['r2'].agg(['mean', 'min', 'max']).round(3))
    print(df.groupby(['model','name'])['mae'].agg(['mean', 'min', 'max']).round(3))
    print(df.groupby(['data','name'])['r2'].agg(['mean', 'min', 'max']).round(3))
    print(df[df['name']=='df-lr'].groupby(['model','data'])['bias'].agg(['mean']).round(3))
    print(df[df['name']=='pik-xl'].groupby(['data'])['r2'].agg(['mean']).round(3))
