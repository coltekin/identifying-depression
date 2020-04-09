#!/usr/bin/env python3
"""Train/tune 'bag-of-n-grams' (BoNG) models.
   These are essentially wrappers areound sklearn linear classifiers.
"""

import sys, re
import numpy as np
import logging
from logging import debug, info, warning, basicConfig
basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

from sklearn.svm import LinearSVC
import tune_textc


class BongVectorizer:
    W_TOK = re.compile("\w+|[^ \t\n\r\f\v\w]+").findall

    PARAMS = {'min_df': 1,
              'c_ngmin': 1, 'c_ngmax': 1,
              'w_ngmin': 1, 'w_ngmax': 1,
              'lowercase': None,
              'tokenizer': W_TOK,
              'vectorizer': 'tfidf',
              'b': 0.75,
              'k1': 2.0}

    def __init__(self, **kwargs):
        self.cache_file = kwargs.get('cache_file', None)
        self.v = None
        self._docid = None
        for k, v in self.PARAMS.items(): setattr(self, k, v)
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.PARAMS:
                warning("Ignoring unknown parameter {}.".format(k))
            else:
                setattr(self, k, v)

    def get_params(self, **kwargs):
        return {k: getattr(self, k) for k in self.PARAMS}

    def _init_vectorizer(self):
        if self.vectorizer not in {'tfidf', 'bm25'}:
            info("Unknown/unimplemented vectorizer {}"
                 ", falling back to TF-IDF".format(self.vectorizer))
            self.vectorizer = 'tfidf'
        if self.vectorizer == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.v = TfidfVectorizer(sublinear_tf=True,
                                     min_df=self.min_df,
                                     analyzer=self.doc_analyzer)
        elif self.vectorizer == 'bm25':
            from bm25 import BM25Vectorizer
            self.v = BM25Vectorizer(sublinear_tf=True,
                                     min_df=self.min_df,
                                     analyzer=self.doc_analyzer,
                                     b = self.b, k1 = self.k1)

    def fit(self, docs):
        self._init_vectorizer()
        self.v.fit(docs)

    def fit_transform(self, docs):
        self._init_vectorizer()
        return self.v.fit_transform(docs)

    def transform(self, docs):
        return self.v.transform(docs)

    def get_ngrams(self, s, ngmin, ngmax, separator="",
                   bos="<", eos=">", suffix="", flatten=True):
        """ For the given sequence s. Return all ngrams in range ngmin-ngmax.
            spearator is useful for readability
            bos/eos symbols are added to indicate beginning and end of seqence
            suffix is an arbitrary string useful for distinguishing
                    in case differernt types of ngrams
            if flatten is false, a list of lists is returned where the
                    first element contains the ngrams of size ngmin
                    and the last contains the ngrams of size ngmax
        """

        # return a single dummy feature if there are no applicable ngrams
        # probably resulting in a mojority-class classifier
        if ngmax == 0 or (ngmax - ngmin < 0) :
            return ['__dummy__']

        ngrams = [[] for x in range(1, ngmax + 1)]
        s = [bos] + s + [eos]
        for i, ch in enumerate(s):
            for ngsize in range(ngmin, ngmax + 1):
                if (i + ngsize) <= len(s):
                    ngrams[ngsize - 1].append(
                            separator.join(s[i:i+ngsize]) + suffix)
        if flatten:
            ngrams = [ng for nglist in ngrams for ng in nglist]
        return ngrams

    def doc_analyzer(self, doc):
        """ Convert document to word/char ngrams with optional
            case normaliztion.
        """

        if self.lowercase is None or self.lowercase == 'none':
            lowercase = set()
        elif self.lowercase in {'both', 'all'}:
            lowercase = {'char', 'word'}
        else: lowercase = {self.lowercase}

        # character n-grams
        if 'char' in lowercase:
            docfeat = self.get_ngrams(list(doc.lower()),
                    self.c_ngmin, self.c_ngmax)
        else:
            docfeat = self.get_ngrams(list(doc),
                    self.c_ngmin, self.c_ngmax)
        # word n-grams
        if 'word' in lowercase:
            docfeat.extend(self.get_ngrams(self.tokenizer(doc.lower()),
                                    self.w_ngmin, self.w_ngmax,
                                    suffix="⅏", separator=" "))
        else:
            docfeat.extend(self.get_ngrams(self.tokenizer(doc),
                                    self.w_ngmin, self.w_ngmax,
                                    suffix="⅏", separator=" "))
        return docfeat


class Bong:
    _params = {'pp': {'min_df', 'c_ngmin', 'c_ngmax',
                      'w_ngmin', 'w_ngmax', 'lowercase', 'vectorizer', 'b', 'k1'},
              'all': {'classifier', 'random_state', 'n_jobs', 'multi_class'},
              'svm': {'dual', 'C', 'class_weight'},
              'rf': {'n_estimators'},
              'lr': {'C', 'class_weight'}
            }
    def __init__(self, **kwargs):
        # default options
        self.classifier='svm'
        self.multi_class = 'ovr'
        self.n_jobs = -1
        self.C=1
        self.dual = True
        self.class_weight = 'balanced'
        self.n_estimators = 50
        self.random_state=None
        self.min_df=1
        self.c_ngmin, self.c_ngmax = 1, 5
        self.w_ngmin, self.w_ngmax = 1, 3
        self.lowercase='word'
        self.vectorizer = 'tfidf'
        self.b = 0.75
        self.k1 = 2.0
        self.v = None
        self.model = None
        #
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def get_param_names(cls, type_=None):
        if type_ and type_.startswith('prep'):
            type_ = 'pp'

        if type_:
            return cls._params[type_]
        else:
            return  set([k for dd in cls._params\
                       for k in cls._params[dd]])

    def get_params(self, param_names=None, type_=None):
        if type_ and type_.startswith('prep'):
            type_ = 'pp'
        elif type_ and type_ == 'model':
            type_ = self.classifier

        if param_names and type_:
                return {k: getattr(self, k) for k in param_names\
                            if k in _params[type_]}
        elif param_names:
            return {k: getattr(self, k) for k in param_names}
        else:
            return  {k: getattr(self, k) for dd in self._params\
                                         for k in self._params[dd]}


    def fit(self, docs, labels, val_docs=None, val_labels=None):
        info("Converting documents to BoNG vectors")
        self.v = BongVectorizer(
                min_df=self.min_df,
                c_ngmin = self.c_ngmin,
                c_ngmax = self.c_ngmax,
                w_ngmin = self.w_ngmin,
                w_ngmax = self.w_ngmax,
                lowercase = self.lowercase,
                vectorizer = self.vectorizer,
                b = self.b,
                k1 = self.k1)
        docs = self.v.fit_transform(docs)
        info("Number of features: {}".format(
                            len(self.v.v.vocabulary_)))

        if self.classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression
        elif self.classifier == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier
        else:
            from sklearn.svm import LinearSVC
            clf = LinearSVC

        classifier_opt = {k: getattr(self,k) \
                for k in self._params[self.classifier]}
        self.model = clf(**classifier_opt)

        if self.multi_class:
            if self.multi_class == 'ovo':
                from sklearn.multiclass import OneVsOneClassifier
                self.model = OneVsOneClassifier(self.model, n_jobs=self.n_jobs)
            elif self.multi_class == 'ovr':
                from sklearn.multiclass import OneVsRestClassifier
                self.model = OneVsRestClassifier(self.model, n_jobs=self.n_jobs)

        info("Fitting the model")
        self.model.fit(docs, np.array(labels))

    def predict(self, docs):
        assert self.v and self.model, "Predicting on untrained model."
        v = self.v.transform(docs)
        return self.model.predict(v)

def _predict(train, test, opt):

    def str_to_param(pstr):
        try:
            params = eval(pstr)
        except:
            def to_val(s):
                try: return(eval(s))
                except: return s
            params = [x.split('=') for x in pstr.split(',')]
            params = {x: to_val(y) for x, y in params}
        return params

    params = {}
    if opt.param_file:
        params = str_to_param(open(opt.param_file, 'r').read().strip())

    params.update(str_to_param(opt.params))

    m = Bong(**params)
    info('Training')
    m.fit(train.docs, train.labels)

    info('Testing')
    pred = m.predict(test.docs)

    return pred


def score(train, test, opt):
    from sklearn.metrics import precision_recall_fscore_support as prfs
    pred = _predict(train, test, opt)
    gold = test.labels

    prec, rec, f1, _ = prfs(gold, pred, average='macro')
 
    print("Precision {}, Recall: {}, F-score: {}".format(
        prec, rec, f1))

def tune(data, devdata, opt):
    import tune_textc
    try:
        params = eval(opt.params)
    except:
        params = '(' + opt.params + ')'

    m = Bong

    skip = None
    if opt.resume_from:
        skip = [x for (x, _, _) in tune_textc.read_logs(opt.resume_from)]

    savefp = None
    if opt.save:
        if opt.save == '-':
            savefp = sys.stdout
        else:
            if skip and opt.resume_from == opt.save:
                savefp = open(opt.save, 'a')
            else:
                savefp = open(opt.save, 'w')

    devdocs, devlabels = None, None
    if devdata:
        devdocs = devdata.docs
        devlabels = trndata.num_labels(devdata.labels)

    best_param, best_sc = tune_textc.tune(m, params, 
                  (trndata.docs, trndata.num_labels()),
                  test=(devdocs, devlabels),
                  method=opt.search_method,
                  real_step = opt.real_step,
                  round_digits = opt.round_digits,
                  max_iter=opt.max_iter,
                  k=opt.k_folds,
                  split_r=opt.test_ratio,
                  n_splits=opt.n_splits,
                  save=savefp,
                  optimize=opt.optimize,
                  skip_params=skip)
    print('best params:', best_param)
    print('best score:', best_sc)

    if savefp and savefp != sys.stdout:
        savefp.close()

if __name__ == '__main__':
    from cmdline import cmdline

    trndata, devdata, opt = cmdline()

    info('Classes: {}'.format(trndata.label_set))
    func = globals()[opt.command]
    func(trndata, devdata, opt)
