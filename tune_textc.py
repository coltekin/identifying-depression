#!/usr/bin/env python3

import sys, os, os.path
import json
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import numpy as np
import itertools
import random
from hashlib import md5
import logging
from logging import debug, info, error, basicConfig
basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def read_logs(filename):
    with open(filename, 'r') as fp:
        for line in fp:
            if (len(line) > 1 and line[0] != '#'):
                log_data = json.loads(line)
                tuned_params = log_data['params']
                model_params = log_data['model_params']

                # hack for supporting older log format without sd
                try:
                    _ = log_data['scores'][0][0]
                except:
                    log_data['scores'] = zip(log_data['scores'], [0, 0, 0, 0])
                score_names = ('acc', 'acc_sd', 'prec', 'prec_sd', 'rec', 'rec_sd',
                               'f1', 'f1_sd')
                scores = [x for t in log_data['scores'] for x in t]
                scores = dict(zip(score_names, scores))
                yield tuned_params, scores, model_params


def add_args(ap):
    ap.add_argument('params',
            help=('Model parameters\n'
                  'A string that can be interpreted by python eval() '
                  'as a sequence whose members are triples '
                  'where, the first element is the name of the option '
                  'the second element is the type  '
                  'and the last argument is another sequence, '
                  'either a tuple or a variable-length sequence '
                  'depending on the type. '
                  '"type" is one of "real", "int" or "categorical". '
                  'for real and "int" the third argument is a tuple '
                  'indicating the range, and for "categorical" '
                  'the last argument is a list '
                  'Example: (("C", "real", (0.5, 1.5)), '
                  '("lowercase", "cat", ("word", "char", "both")))'))
    ap.add_argument('--search-method', '-s', choices=('grid', 'random'),
            default='grid', help="Method used for hyper parmeter search")
    ap.add_argument('--optimize', choices=('F1-macro', 'F1-micro', 'F1-binary'),
            default='F1-macro', help="The score to maximize.")
    ap.add_argument('--real-step', type=float, default=None,
            help=("Increment step size for real-valued parameters "
                 "during grid search"))
    ap.add_argument('--round-digits', type=int, default=None,
            help=("Number of digits after decimal point to round"
                  " for real-valued parameters after sampling"
                 " (for random search)"))
    ap.add_argument('--max-iter', '-m', type=int, default=-1,
            help=('Maximum number of hyperparameter combinations '
                  'to compare. Default (-1) means until '
                  'the search space is exhausted'))
    ap.add_argument('--k-folds', '-k', type=int, default=None,
            help=('Use k-fold cross validation. '
                  'Ignored if -t is given.'))
    ap.add_argument('--test-ratio', '-r', type=float, default=0.2,
            help=('Ratio of held-out data. '
                  'Ignored if --test option is given'))
    ap.add_argument('--n-splits', '-n', type=int, default=1,
            help=('Number of splits. Ignored if -t or -k is given'))
    ap.add_argument('--save', '-S', metavar='LOGFILE', 
            help=('Save intermediate parameter values and scores '
                  'to given log file. '
                  'Use - for standard output'))
    ap.add_argument('--resume-from', '-R', metavar='LOGFILE',
            help=('Resume tuning, skipping the parameters that '
                  'are logged in the log file.'))

def cmdline(args, prog=None):
    import argparse
    ap = argparse.ArgumentParser()
    if prog: ap.prog = prog
    else: ap.prog = 'tune'
    add_args(ap)
    ap.add_argument('train', help="Training data")
    ap.add_argument('--test', '-t', default=None, help="Test set")
    opt = ap.parse_args(args)
    return opt, ap

def random_iter(params, max_iter=1000, real_round=None):
    seen = set()
    rejected = 0
    while True:
        opt = []
        for param, type_, seq in params:
            if 'integer'.startswith(type_):
                opt.append((param, random.choice(range(seq[0], seq[1]+1))))
            elif 'real'.startswith(type_):
                rand_val = random.uniform(seq[0], seq[1])
                if real_round:
                    round(rand_val, real_round)
                opt.append((param, rand_val))
            elif 'categorical'.startswith(type_):
                opt.append((param, random.choice(seq)))
        opt_hash = md5(str(opt).encode()).digest()
        if opt_hash not in seen:
            seen.add(opt_hash)
            rejected = 0
            yield dict(opt)
        else:
            rejected += 1
            if rejected == max_iter:
                info("More than {} iterations with already drawn parameters. "
                      "The search space is probably exhausted".format(max_iter))
                return

def grid_iter(params, real_step=None):
    optlist = []
    for param, type_, seq in params:
        if 'integer'.startswith(type_):
            optlist.append(range(seq[0], seq[1]+1))
        elif 'real'.startswith(type_):
            if real_step is None:
                # TODO: 5 steps is arbitrary
                real_step = (seq[1] - seq[0]) / 5
            optlist.append(np.arange(seq[0], seq[1]+real_step, real_step))
        elif 'categorical'.startswith(type_):
            optlist.append(seq)
    for opt in itertools.product(*optlist):
        yield dict(zip((p[0] for p in params), opt))

def tune(model, params, train, test=None,
        method='grid',
        real_step=None, round_digits=None,
        max_iter=-1, optimize='F1-macro',
        k=None, split_r=0.2, n_splits=1,
        save=sys.stdout,
        skip_params=None):
    """ Fit and evaluate a model repeatedly,
     the best one based on params.

     Args:
        model:  A class that is initialized with the given params,
                and supports fit() and predict() methods
        train:  a tuple (features, labels) in a form to be consumed by
                the model, if training set is not given, it is used
                both for training and testing
        test:   same as 'train'
        params: a dict-like object whose keys are the option names
                 and values are tuples of (type, range) or (type,
                 list). For types is one of 'real', 'int' or 'categorical'.
                 For 'real' and 'int' a range,for 'categorical' a list
                 of values is required.
        method: 'grid', 'random' or a iterable that returns values
                from the option range.
        real_step: Step size for incrementing the real values during
                grid search.
        round_digits: If specified, the sampled real numbers during
                random search are rounded to have round_digts precision.
        max_iter: terminate after trying max_iter fit/predict/eval cycles
        k:      Use k-fold CV.
        split_r: ratio of the held-out set, ignored if test set or
                 argument `k' is given
        n_splits: number of splits, sklearn StratifiedShuffleSplit is
                used for n-splits
        save:   file-like object save the results after each iteration
        skip_params: A sequence of dictionaries containing individual
                parameter values to skip. Useful for resuming a search. 
    """

    if method == 'grid':
        param_iter = grid_iter(params, real_step=real_step)
    elif method == 'random':
        param_iter = random_iter(params, real_round=round_digits)
    else:
        param_iter = method(params)

    if test[0] is not None:
        tune_str = "test data".format()
    else:
        if k:
            splits = StratifiedKFold(n_splits=k, shuffle=True)
            tune_str = "{}-fold CV".format(k)
        else:
            splits = StratifiedShuffleSplit(
                                n_splits=n_splits, test_size=split_r)
            tune_str = "{} splits of {} ratio".format(n_splits, split_r)
        feat, lab = np.array(train[0]), np.array(train[1])
        splits = list(splits.split(feat, lab))


    def fit_eval(m, trnX, trnY, tstX, tstY, average='micro', **param):
        m.set_params(**param)
        m.fit(trnX, trnY, tstX, tstY)
        pred = m.predict(tstX)
        prec, rec, f1, _ = prfs(tstY, pred, average=average)
        acc = accuracy_score(tstY, pred)
        return (acc, prec, rec, f1)

    def param_to_str(p_dict):
        p_list = sorted(tuple(p_dict.items()))
        p_fmt = "{}={} " * len(p_list)
        return p_fmt.format(*[x for t in p_list for x in t])

    if skip_params:
        skip_params = set(
                [md5(param_to_str(p).encode()).digest()\
                        for p in skip_params])
    else:
        skip_params = set()

    best_sc = (0, 0, 0, 0)
    best_param = None
    m = model()
    for param in param_iter:
        scores = []
        p_str = param_to_str(param)
        p_hash = md5(p_str.encode()).digest()
        if p_hash in skip_params:
            info('Skipping: {}'.format(p_str))
            continue
        info('Tuning with {}: {}'.format(tune_str, p_str))
        if test[0] is not None:
            scores.append(
                fit_eval(m, train[0], train[1], test[0], test[1],
                    average=optimize.split('-')[-1], **param))
        else:
            for ti, vi in splits:
                scores.append(fit_eval(m, feat[ti], lab[ti], feat[vi], lab[vi],
                                **param))
        scores = np.array(scores)
        sc_mean = scores.mean(axis=0)
        sc_sd = scores.std(axis=0)
        if sc_mean[2] > best_sc[2]:
            best_sc = sc_mean
            best_param = param
        if save:
            json.dump({'params': param,
                       'model_params': m.get_params(),
                       'scores': tuple(zip(sc_mean.tolist(), sc_sd.tolist()))},
                    save, ensure_ascii=False)
            print('', file=save, flush=True)
        max_iter -= 1
        if max_iter == 0:
            break
        stop_file = '.stop-tune' + str(os.getpid())
        if os.path.isfile(stop_file):
            os.remove(stop_file)
            break
    return best_param, best_sc

if __name__ == '__main__':
    class Dummy:
        def __init__(self, **kwargs):
            print('init:', kwargs)
        def get_params(self):
            return {'a': 1, 'b': 2}
        def fit(self, X, y, valX=None, valY=None):
            pass
        def re_fit(self, X, y, valX=None, valY=None, **kwargs):
            pass
        def predict(self, X):
            return np.ones(len(X))

    tune(Dummy,
         (('a', 'int', (0,3)), ('b', 'cat', [10, 11, 12])),
         ((0,1,2,3,4,5,6,7,8,9),(1,1,1,1,1,2,2,2,2,2)),
         method='random',
         max_iter=10000
    )
