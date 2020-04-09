#!/usr/bin/env python3

import logging
from collections import Counter, OrderedDict
from logging import debug, info, error, basicConfig
basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


import sys, re
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gzip


class TextcCSV:
    """Data structure/reader for simple CSV files 
    Data should be a real CSV file with delimiter comma,
    with a header proper quoting (with double quotes) and escaping.
    The labels should have a column header 'label', and the text part
    should have a column header 'text'.
    The other fields, if exists ignored.
    """
    def __init__(self,
            path=None,
            label_filter=None,
            negative_class=None):
        self.docs = []
        self.labels = []
        self.label_set = OrderedDict()
        if path:
            self.load(path, label_filter=label_filter,
                    negative_class=negative_class)

    def _filter(self, label, f):
        if f is None: return label
        for inp, out in f:
            if inp == label:
                if not out: return None
                return out
        return label

    def load(self, path, negative_class=None, label_filter=None):
        if path.endswith(".gz"):
            fp = gzip.open(path, 'rt')
        else:
            fp = open(path, 'rt')
        csv_r = csv.DictReader(fp)
        for row in csv_r:
            label = row['label']
            if label_filter:
                label = self._filter(label, label_filter)
                if label is None:
                    continue
            self.labels.append(label)
            self.docs.append(row['text'])
        if (not self.label_set) and negative_class:
            self.label_set[negative_class] = 0
        lset = Counter(self.labels)
        self.label_set.update(lset.most_common())
        fp.close()

    def num_labels(self, labels=None):
        """Return numeric labels instead of string-valued readable ones.
        """
        if labels is None: labels=self.labels
        label_list = list(self.label_set.keys())
        return [label_list.index(lab) for lab in labels]

    def str_labels(self, seq):
        """Return string-valued labels for a given numeric label
        """
        labels = self.label_set.keys()
        return [labels[i] for i in seq]

def stats(data, histogram=True):
    tokenizer = re.compile(r"\w+|[^ \t\n\r\f\v\w]+").findall

    print('Label distribution:')
    for k,v in Counter(data.labels).most_common():
        print('{}: {:>6d} ({:2.2f}%)'.format(k, v, 100*v/len(data.labels)))
    print("Total: {:>6d}".format(len(data.labels)))
    print()

    dlen_c = np.array([len(x) for x in data.docs])
    print("Document length in characters (mean/sd/min/max): "
          "{:0.2f}/{:0.2f}/{}/{}".format(
                dlen_c.mean(), dlen_c.std(), dlen_c.min(), dlen_c.max()))
    dlen_w = np.array([len(tokenizer(x)) for x in data.docs])
    print("Document length in words (mean/sd/min/max): "
          "{:0.2f}/{:0.2f}/{}/{}".format(
                dlen_w.mean(), dlen_w.std(), dlen_w.min(), dlen_w.max()))
    _, plts = plt.subplots(nrows=1,ncols=2)
    plts[0].hist(dlen_c, bins=100)
    plts[0].set_title("Char")
    plts[1].hist(dlen_w, bins=100)
    plts[1].set_title("Word")
    plt.show()

if __name__ == "__main__":

    input_file = "data/blogs.csv"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    d = TextcCSV(input_file, label_filter=[('breastc', '')])
    
    stats(d)
