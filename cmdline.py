import tune_textc

def cmdline():
    import argparse
    from textc_csv import TextcCSV

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', help="Path to the training data")
    ap.add_argument('--test', '-t', help="Path to the testing data")
    ap.add_argument('--param-file', '-p', 
            help="Path to file with parameters.")
    ap.add_argument('--output', '-o', default='-',
                        help="Output file. `-' means stdout.")
    ap.add_argument('--replace-label', '-l', action='append',
                    help=("pairs of 'inp:out'. The 'inp' labels are"
                          "replaced with 'out' labels. This is useful"
                          "if one needs to collapse classes"))
    ap.add_argument('--negative-class', 
                    help=('Mark the given class as the negative class.'
                          'Useful for binary classification'))
    ap.add_argument('--filter-test', action='store_true')

    subp = ap.add_subparsers(help="Command")

    tunep = subp.add_parser('tune')
    tunep.set_defaults(command='tune')
    tune_textc.add_args(tunep)

    predp = subp.add_parser('predict')
    predp.add_argument('params',
            help=('Model parameters\n'
                  'A string that can be interpreted by python eval() '
                  'as a sequence whose members are pairs '
                  'of, parameter=value'))
    predp.set_defaults(command='predict')

    testp = subp.add_parser('score')
    testp.add_argument('params',
            help=('Model parameters\n'
                  'A string that can be interpreted by python eval() '
                  'as a sequence whose members are pairs '
                  'of, parameter=value'))
    testp.set_defaults(command='score')


    opt = ap.parse_args()

    label_replace = None
    if opt.replace_label:
        label_replace = []
        for s  in opt.replace_label:
            label_replace.append(s.split(':'))
    trndata = TextcCSV(opt.input, label_filter=label_replace,
            negative_class=opt.negative_class)
    devdata = None
    if opt.test:
        if opt.filter_test:
            devdata =  TextcCSV(opt.test, label_filter=label_replace,
                            negative_class=opt.negative_class)
        else:
            devdata =  TextcCSV(opt.test, negative_class=opt.negative_class)

    return trndata, devdata, opt
