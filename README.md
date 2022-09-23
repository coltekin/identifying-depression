## Identifying depression on Reddit: the effect of training data

This repository contains code and data used in paper

Inna Pirina and Çağrı Çöltekin (2018) [Identifying Depression on
Reddit: The Effect of Training Data](https://www.aclweb.org/anthology/W18-5903/).
In: Proceedings of the 2018 EMNLP Workshop SMM4H: The 3rd Social Media Mining
for Health Applications Workshop & Shared Task, pages 9–12

The results presented in the paper uses the bag-of-n-grams SVM
classifiers implemented in `bong.py`. The following demonstrates a
typical use case. Brief descriptions of other options are provided
with the `-h` command line option.

The first step is tuning the hyperparameters: 

```bash
./bong.py -i data/train.csv # traning data \
    -l ds: -l ff: -l do: -l ds: -l bc: -l nd: # we remove these labels from the training data \
    --negative-class=ndf # explicity state the negative class \
    tune # this is the command \
    -S dsf-ndf.log  # save results to this log file \
    -s random -k 5 # random search with 5-fold CV \
    -m 1000 # try 1000 hyperparameter settings
    '(("w_ngmax", "int", (1, 2)), ("c_ngmax", "int", (2, 4)), ("C", "real", (0.1, 2.0)), ("lowercase", "cat", ("word", "char", "both")))'

```
Yes, the cmdline interface, particularly filtering bit is somewhat convoluted.
If you want clener command line you can also split the traning data
into separate files with binary class labels.

The rest of the tunable hyperparameters can be found in the `__init__`
method of class `Bong` in `bong.py`. For more information on other
options for tuning see `./bong.py tune -h`
The above will crunch numbers for a while and write the hyperparmeter
settings and evaluation metrics in file `dsf-ndf.log`.

To get the best hyperparameters and scores (example is based on a
short run):

```bash
./read-logs.py dsf-ndf.log
Based on 34 entries.
Best score (p r f a): 94.50±1.08 94.50±1.08 94.50±1.08 94.50±1.08
Top 20:
94.50±1.08 C=0.6762,c_ngmax=3,lowercase=char,w_ngmax=1
94.38±0.79 C=1.6789,c_ngmax=2,lowercase=both,w_ngmax=1
94.38±1.12 C=0.5059,c_ngmax=3,lowercase=both,w_ngmax=1
94.25±1.21 C=1.4373,c_ngmax=3,lowercase=both,w_ngmax=1
94.25±1.21 C=1.8086,c_ngmax=3,lowercase=both,w_ngmax=1
...
```

Now we can retrain the model with the best parameters,
and test it on the test data.

```bash
./bong.py -i data/train.csv -t data/test.csv \
    -l ds: -l ff: -l do: -l ds: -l bc: -l nd: \
    -l dsf:do -l ndf:nd  # these two are new, maps trainig file labels to test file labels \
    score # now we want the score \
    C=0.6762,c_ngmax=3,lowercase=char,w_ngmax=1
2022-09-23 22:05:49,866 Classes: OrderedDict([('do', 400), ('nd', 400)])
2022-09-23 22:05:49,866 Training
2022-09-23 22:05:49,867 Converting documents to BoNG vectors
2022-09-23 22:05:51,773 Number of features: 27081
2022-09-23 22:05:51,775 Fitting the model
2022-09-23 22:05:52,936 Testing
Precision 0.6377374671898959, Recall: 0.6325000000000001, F-score:
0.6289729238574195

```
`predict` instead of prints out the labels instead.

The dataset(s) are included in `data` directory. The class labels match the
class labels used in the paper.
