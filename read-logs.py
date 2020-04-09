#!/usr/bin/env python3
""" Read/process the logs produced during tuning trac models.
    Default prints out the best score, -t argument produces a table. 
"""

import json
import sys
import argparse
from collections import OrderedDict # used as an `ordered' set
from tune_textc import read_logs

ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('logfile', help='Log file.')
ap.add_argument('--all-params', '-a', action='store_true',
        help='Output all model parameters, including defaults.')
grp = ap.add_mutually_exclusive_group()
grp.add_argument('--tsv', '-t', action='store_true',
        help='Output a tab-seperated file table.')
grp.add_argument('--top-n', '-T', default=20, type=int,
        help='Only output records with TOP-N F1-scores.')
opt = ap.parse_args()


if opt.tsv:
    for i, (tuned_p, scores, model_p) in enumerate(read_logs(opt.logfile)):
        if i == 0:
            col_names = sorted(tuned_p.keys()) + sorted(scores.keys())
            if opt.all_params:
                col_names += sorted(scores.keys())
            fmt = "{}" + "\t{}" * (len(col_names) - 1)
            print(fmt.format(*list(col_names)))

        # out = {**tuned_p, **scores} #python 3.5+ only
        out = tuned_p
        out.update(scores)
        if opt.all_params: out.update(model_p)
        print(fmt.format(*[out[x] for x in col_names]))
else:
    scores = []
    params = []
    best_params = []
    best_sc = {'f1': 0.0}
    n = 0
    for i, (tuned_p, sc, _) in enumerate(read_logs(opt.logfile)):
        scores.append(sc)
        params.append(tuned_p)
        if sc['f1'] > best_sc['f1']:
            best_sc = sc
            best_params = tuned_p
        n += 1

    fmt = 4 * "{:0.2f}±{:0.2f} "
    print('Based on {} entries.'.format(n))
    print('Best score (p r f a):',
        fmt.format(100*best_sc['prec'], 100*best_sc['prec_sd'],
                   100*best_sc['rec'], 100*best_sc['rec_sd'],
                   100*best_sc['f1'], 100*best_sc['f1_sd'],
                   100*best_sc['acc'], 100*best_sc['acc_sd']))
    print('Top {}:'.format(opt.top_n))
    sc_param = zip(scores,params)
    for sc, param  in \
      sorted(sc_param, key=lambda x: x[0]['f1'], reverse=True)[:opt.top_n]:
        print("{:0.2f}±{:0.2f} ".format(
            100*sc['f1'], 100*sc['f1_sd']), end="")
        print_end = ","
        for i, k in enumerate(sorted(param)):
            v = param[k]
            if i == (len(param) - 1):
                print_end = ""
            if isinstance(v, float):
                print('{}={:0.4f}'.format(k,v), end=print_end)
            else:
                print('{}={}'.format(k,v), end=print_end)
        print()
