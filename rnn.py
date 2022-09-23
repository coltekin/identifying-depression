#!/usr/bin/env python3
"""Train/tune RNN model for document classification.
"""
import numpy as np
import logging
from logging import debug, info, basicConfig
basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import sys, re
from collections import Counter

from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score

from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import SpatialDropout1D
from keras.layers.merge import concatenate
from keras.models import Model

import tune_textc

class VectRNN:
    tokenizer = re.compile(r"\w+|[^ \t\n\r\f\v\w]+").findall
    PAD = 0
    OOV = 1
    START = 2
    END = 3

    def __init__(self, 
                 min_df=None,
                 c_cutoff=0, w_cutoff=0,
                 c_maxlen=None, w_maxlen=None,
                 lowercase=None):
        self.min_df = min_df
        self.c_cutoff = c_cutoff
        self.w_cutoff = w_cutoff
        self.c_maxlen = c_maxlen
        self.w_maxlen = w_maxlen
        self.lowercase = lowercase

        self.c_vocab = dict()
        self.w_vocab = dict()

    def fit(self, docs):
        max_c = 0
        max_w = 0
        c_count = Counter()
        w_count = Counter()
        for doc in docs:
            if self.c_maxlen != 0:
                c_count.update(doc)
                if self.c_maxlen is None and max_c < len(doc):
                    max_c = len(doc)
            if self.w_maxlen != 0:
                doc_t = self.tokenizer(doc)
                w_count.update(doc_t)
                if self.w_maxlen is None and max_w < len(doc_t):
                    max_w = len(doc_t)
        if self.c_maxlen is None: self.c_maxlen = max_c + 2
        if self.w_maxlen is None: self.w_maxlen = max_w + 2
        for i, (k, v) in enumerate(c_count.most_common()):
            if self.c_cutoff and v < self.c_cutoff:
                break
            self.c_vocab[k] = i + 4
        for i, (k, v) in enumerate(w_count.most_common()):
            if self.w_cutoff and v < self.w_cutoff:
                break
            self.w_vocab[k] = i + 4
#        print(self.w_vocab, self.c_vocab)

    def transform(self, docs):
        char_seq = []
        word_seq = []
        for doc in docs:
            if self.c_maxlen:
                char_seq.append(self.to_numseq(list(doc),
                            self.c_vocab, self.c_maxlen))
            if self.w_maxlen:
                word_seq.append(self.to_numseq(self.tokenizer(doc),
                            self.w_vocab, self.w_maxlen))
        return np.array(char_seq), np.array(word_seq)

    def fit_transform(self, docs):
        self.fit(docs)
        d = self.transform(docs)
        return d

    def to_numseq(self, doc, vocab, maxlen):
        """ Transform a single document to a sequence of numeric values 
        """
        x = [self.START]
        for token in doc:
            x.append(vocab.get(token, self.OOV))
            if len(x) > (maxlen - 2):
                break
        x.append(self.END)
        if len(x) < maxlen: 
            x += [self.PAD] * (maxlen - len(x))
        return x

class TextRNN:
    _params = {'pp': {'min_df', 'lowercase', 
                      'c_cutoff', 'w_cutoff',
                      'c_maxlen', 'w_maxlen'},
               'all': {'random_state', 'batch_size', 'epoch',
                       'optimizer', 'loss', 'best_epoch', 'class_weight'},
               'model': {'c_embdim', 'c_embdrop', 'c_featdim', 'c_featdrop',
                         'w_embdim', 'w_embdrop', 'w_featdim', 'w_featdrop',
                         'rnn'}
            }

    def __init__(self, **kwargs):
        # default options
        self.min_df = 1
        self.lowercase = 'word'
        self.c_cutoff,  self.w_cutoff = 0, 0
        self.c_maxlen,  self.w_maxlen = None, None
        self.c_embdim,  self.c_embdrop =   64, 0.2
        self.c_featdim, self.c_featdrop = 64, 0.2
        self.w_embdim,  self.w_embdrop =   64, 0.2
        self.w_featdim, self.w_featdrop = 64, 0.2
        self.rnn = 'GRU'
        self.epoch = 30
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.batch_size = 128
        self.random_state = None
        self.best_epoch = None
        self.class_weight = 'balanced'
        #
        self._preproc_params = None
        self.set_params(**kwargs)

    # TODO: duplication - can be inherited
    def set_params(self, **kwargs):
        pp_opt = dict()
        for k, v in kwargs.items():
            setattr(self, k, v)
            if k in self._params['pp']:
                pp_opt[k] = v
        if pp_opt != self._preproc_params:
            # If preprocessing parameters did not change, do not reset
            # vectorizer. This helps by not re-training vectorizer
            # during tuning
            self.vectorizer = None 
        self.model = None 

    # TODO: duplication - can be inherited
    @classmethod
    def get_param_names(cls, type_=None):
        if type_ and type_.startswith('prep'):
            type_ = 'pp'

        if type_:
            return cls._params[type_]
        else:
            return  set([k for dd in cls._params\
                       for k in cls._params[dd]])

    # TODO: duplication - can be inherited
    def get_params(self, param_names=None, type_=None):
        if type_ and type_.startswith('prep'):
            type_ = 'pp'

        if param_names and type_:
                return {k: getattr(self, k) for k in param_names\
                            if k in _params[type_]}
        elif param_names:
            return {k: getattr(self, k) for k in param_names}
        else:
            return  {k: getattr(self, k) for dd in self._params\
                                         for k in self._params[dd]}

    def _build_model(self, n_labels):

        if self.rnn.lower() == 'gru':
            from keras.layers import GRU
            rnn = GRU
        elif self.rnn.lower() == 'lstm':
            from keras.layers import LSTM
            rnn = LSTM
        else:
            from keras.layers import SimpleRNN
            rnn = SimpleRNN

        c_inp = Input(shape=(self.c_maxlen,), name='char_input')
        w_inp = Input(shape=(self.w_maxlen,), name='word_input')

        c_emb = Embedding(len(self.vectorizer.c_vocab) + 4,
                          self.c_embdim,
                          mask_zero=True,
                          name='char_embedding')(c_inp)
        c_emb = SpatialDropout1D(self.c_embdrop)(c_emb)
        w_emb = Embedding(len(self.vectorizer.w_vocab) + 4,
                          self.w_embdim,
                          mask_zero=True,
                          name='word_embedding')(w_inp)
        w_emb = SpatialDropout1D(self.w_embdrop)(w_emb)

        c_fw = rnn(self.c_featdim,
                   dropout=self.c_featdrop, name='char_fw_rnn')(c_emb)
        c_bw = rnn(self.c_featdim, dropout=self.c_featdrop,
                    go_backwards=True, name='char_bw_rnn')(c_emb)
        c_feat = concatenate([c_fw, c_bw])

        w_fw = rnn(self.w_featdim,
                dropout=self.w_featdrop,
                name='word_fw_rnn')(w_emb)
        w_bw = rnn(self.w_featdim, dropout=self.w_featdrop,
                go_backwards=True, name='word_bw_rnn')(w_emb)
        w_feat = concatenate([w_fw, w_bw])

        h = concatenate([c_feat, w_feat])

        clf = Dense(n_labels, activation='softmax', name='out')(h)

        self.model = Model(inputs=[c_inp, w_inp], outputs=[clf])
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def re_fit(self, docs, labels, val_docs=None, val_labels=None, **kwargs):
        # Not implemented yet.
        self.set_params(**kwargs)
        self.fit(docs, labels, val_docs=val_docs, val_labels=val_labels)

    def fit(self, docs, labels, val_docs=None, val_labels=None):
        info("Converting documents to numeric sequences")
        if self.vectorizer is None:
            self.vectorizer = VectRNN(min_df=self.min_df,
                                c_cutoff = self.c_cutoff,
                                w_cutoff = self.w_cutoff,
                                c_maxlen = self.c_maxlen,
                                w_maxlen = self.w_maxlen,
                                lowercase = self.lowercase)
            c_trn, w_trn = self.vectorizer.fit_transform(docs)
            info("Number of features: c:{}, w:{}".format(
                len(self.vectorizer.c_vocab),
                len(self.vectorizer.w_vocab)))
        else:
            c_trn, w_trn = self.vectorizer.transform(docs)

        lab_trn = to_categorical(np.array(labels))


        self._build_model(len(np.unique(labels)))

        debug("{}".format(Counter(labels)))
        if self.class_weight == 'balanced':
            from sklearn.utils import class_weight
            weights = class_weight.compute_class_weight('balanced',
                    classes=np.unique(labels), y=labels)
#            c_count = Counter(labels)
#            keys = c_count.keys()
#            weights = np.log(np.array([c_count[k] for k in keys]).reshape(1,-1))
#            weights = weights.sum() / weights 
            c_weight  = {k: v for k,v in zip(np.unique(labels), weights)}
        else:
            c_weight = None

        debug("{}".format(c_weight))

        info("Fitting the model")

        # This is custom, not Keras' EarlyStopping
        early_stop = EarlyStop(patience=7)

        if val_docs is not None and val_labels is not None:
            c_val, w_val = self.vectorizer.transform(val_docs)
            lab_val = to_categorical(np.array(val_labels))
            self.model.fit(x={'char_input': c_trn, 'word_input': w_trn},
              y=lab_trn,
              batch_size=self.batch_size,
              class_weight=c_weight,
              verbose=0,
              epochs=self.epoch,
              validation_data=({'char_input': c_val, 'word_input': w_val},
                                lab_val),
              callbacks=[early_stop])
            self.best_epoch = early_stop.best_epoch + 1
        else:
            self.model.fit(x={'char_input': c_trn, 'word_input': w_trn},
              y=lab_trn,
              batch_size=self.batch_size,
              class_weight=c_weight,
              verbose=0,
              epochs=self.epoch)

    def predict_prob(self, docs):
        assert self.vectorizer and self.model, "Predicting on untrained model."
        c, w = self.vectorizer.transform(docs)
        return self.model.predict({'char_input': c, 'word_input': w})

    def predict(self, docs):
        return np.argmax(self.predict_prob(docs), axis=1)

class EarlyStop(Callback):
    """Stop training when F1 score is not improving for 'patience'
    epochs. Most of it is borrowed from Keras 'EarlyStopping'
    callback.
    """
    def __init__(self, patience=0, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self._reset

    def _reset(self):
        self.best = 0
        self.best_epoch = self.params['epochs']
        self.wait = 0
        self.best_weights = None

    def on_train_begin(self, logs={}):
        self._reset()

    def on_epoch_end(self, epoch, logs={}):
        predict = np.argmax(
                self.model.predict(self.validation_data[:2],
                                   batch_size = self.params['batch_size']),
                axis=1)
        targ = np.argmax(self.validation_data[2], axis=1)
        prec, rec, f1, _ = prfs(targ, predict, average='macro')
        if self.verbose:
            info("epoch: {} p: {:0.4f} r: {:0.4f} f:{:0.4f}".format(
                epoch, prec, rec, f1))
        if f1 > self.best:
            self.best = f1
            self.best_epoch = epoch
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
                info("Stopping at epoch {}, best: e {}, f: {}.".format(
                    epoch, self.best_epoch, self.best))
            else:
                self.wait += 1
        if epoch == (self.params['epochs'] - 1):
            info("Stopping at final epoch {}, best: e {}, f: {}.".format(
                    epoch, self.best_epoch, self.best))
            if self.best < f1:
                self.model.set_weights(self.best_weights)

    def __str__(self):
        return 'prfa: {} {} {} {}'.format(
                self.val_precision, self.val_recall, self.val_f1,
                self.val_accuracy)


def tune(data, devdata, opt):
    import tune_textc
    try:
        params = eval(opt.params)
    except:
        params = '(' + opt.params + ')'

    m = TextRNN

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
                  skip_params=skip)
    print('best params:', best_param)
    print('best score:', best_sc)

    if savefp and savefp != sys.stdout:
        savefp.close()


if __name__ == '__main__':
    from cmdline import cmdline

    trndata, devdata, opt = cmdline()

    func = globals()[opt.command]
    func(trndata, devdata, opt)
