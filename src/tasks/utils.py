import json
import numpy as np
import re
import os

from tqdm import tqdm
from sklearn.linear_model import LinearRegression, SGDRegressor
from scipy.stats import spearmanr, pearsonr

TOKENS = json.load(open('tokens.json', 'r'))
TKN = TOKENS['HF']
OAIK = TOKENS['OAI']
TGIK = TOKENS['TG']

def get_probas(arr):

    for text in arr:

        r = re.search(r'certainty\s*=\s*(\d+)', text.lower())
        try:
            p = int(r.group().split('=')[-1]) / 10
            if p > 1:
                # manually checked, almost all cases show model returns %
                p = p / 10
            if 0 <= p and p <= 1:
                yield p 
            # some wierd edge cases where model says 810% etc.
            # let these pass through
        except:
            # print(text)
            pass

def _get_features(text):
        
        if text is None:
            return np.empty(768).fill(np.nan)

        import requests

        url = "https://api.together.xyz/v1/embeddings"

        payload = {
            "model": "togethercomputer/m2-bert-80M-8k-retrieval",
            "input": text
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {TGIK}"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return np.array(json.loads(response.text)['data'][0]['embedding'])
        except:
            return np.empty(768).fill(np.nan)

def _get_prompt(o):
    try:
        return o['completion'][0].split(o['response'][0])[0]
    except:
        return None

def get_features(outputs, cache):

    try:
        return np.load(f'features/{cache}.npz')['arr_0']
    except:
        # added since embeddings don't actually change
        _, data = parse_file_name(cache)
        demo = '-d=1' if '-d=1' in cache else '-d=0'
        existing = [file for file in sorted(os.listdir('features')) if data in file and demo in file]
        if existing:
            e = existing[0].split('.npz')[0]
            # check
            # print('Sub embeddings:', e, cache)
            return get_features(outputs, e)
        np.savez(f'features/{cache}.npz', np.array([_get_features(_get_prompt(o)) for o in tqdm(outputs)]))
        return get_features(outputs, cache)

def get_basics(outputs, args):

    probas = [list(get_probas(o['response']))[:args.n_samp] for o in outputs]

    p_true = np.array([o['p'] for o in outputs])
    gt = np.array([o['gt'] for o in outputs])
    p_rand = np.array([plist[0] if plist else np.nan for plist in probas])
    p_bag = np.array([np.mean(plist) if plist else np.nan for plist in probas])
    p_hat = p_bag if args.bag else p_rand

    return {'probas' : probas, 'p_true' : p_true, 'p_hat' : p_hat, 'gt' : gt}

def explained_variance(model, x, y, train, test, **kwargs):
    if len(x.shape) <= 1:
        x = x.reshape(-1, 1)
    yhat = model(**kwargs).fit(x[train], y[train]).predict(x[test])
    yhat = np.clip(yhat, 0, 1)
    mse = np.mean((y[test] - yhat) ** 2)
    ref = np.mean(y[train])
    ref_mse = np.mean((y[test] - ref) ** 2)
    return 1 - mse / ref_mse, (yhat - y[test]).mean(), np.abs(y[test] - yhat).mean(), np.std(yhat)

def parse_file_name(file):
    data = ['candor', 'casino', 'multiwoz']
    for d in data:
        if d in file:
            return file.split(f'-{d}')[0], d 

def linear_regression(results, file, name, x, y, train, model=LinearRegression, **kwargs):
    is_nan_x = np.isnan(x)
    if len(is_nan_x.shape) > 1: is_nan_x = is_nan_x.sum(-1)
    nas = ~np.logical_or(is_nan_x, np.isnan(y))
    test = (~train) & nas
    # if nas.mean() < 0.8 or test.sum() < 100:
    if nas.mean() < 0.78 or test.sum() < 100: # gemma no temp was only slightly worse
        print('Skipping, not enough data:', name, file, nas.mean(), test.sum())
        return
    tr = train & nas
    if len(x.shape) == 1:
        rho = spearmanr(x[test], y[test]).statistic
        r = pearsonr(x[test], y[test]).statistic
    else:
        rho = r = np.nan
    mo, dat = parse_file_name(file)
    try:
        r2, bias, mae, v = explained_variance(model, x, y, tr, test, **kwargs)
        results.append({
            'name' : name, 
            'model' : mo, 
            'data' : dat, 
            'r' : r, 'rh' : rho, 'r2' : r2,
            'bias' : bias, 'mae' : mae, 'v' : v})
    except ValueError as e:
        # from scipy.special import logit
        # print([xi for xi in x[test] if np.isnan(logit(xi))]); exit()
        print('Skipping, Value Error:', file, model)

def sgd_regression(results, file, name, x, y, train):
    is_nan_x = np.isnan(x)
    if len(is_nan_x.shape) > 1: is_nan_x = is_nan_x.sum(-1)
    nas = ~np.logical_or(is_nan_x, np.isnan(y))
    test = (~train) & nas
    if nas.mean() < 0.8 or test.sum() < 100: 
        print('Skipping, not enough data:', file)
        return
    tr = train & nas
    rho = r = np.nan
    model, data = parse_file_name(file)
    try:
        r2, bias, mae, v = explained_variance(SGDRegressor, x, y, tr, test)
        results.append({
            'name' : name, 
            'model' : model, 
            'data' : data, 
            'r' : r, 'rh' : rho, 'r2' : r2,
            'bias' : bias, 'mae' : mae, 'v' : v})
    except ValueError:
        print('Skipping, Value Error:', file)

# in progress... maybe drop embedding
def fit_predict(file, x, y, train, model=LinearRegression, **kwargs):
    is_nan_x = np.isnan(x)
    if len(is_nan_x.shape) > 1: is_nan_x = is_nan_x.sum(-1)
    nas = ~np.logical_or(is_nan_x, np.isnan(y))
    test = (~train) & nas
    tr = train & nas
    if nas.mean() < 0.8 or test.sum() < 100: 
        print('Skipping fit_predict, not enough data:', file)
        return
    x_tr = x[tr].reshape(-1, 1) if len(x.shape) == 1 else x[tr]
    x_full = x[nas].reshape(-1, 1) if len(x.shape) == 1 else x[nas]
    model = model(**kwargs).fit(x_tr, y[tr])
    yhat = np.ones_like(y) * np.nan
    yhat[nas] = model.predict(x_full)
    return yhat
