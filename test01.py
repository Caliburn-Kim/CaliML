from __future__ import print_function

import threading
import timeit

import joblib
import numpy as np
from sklearn.metrics import f1_score

from cklib import ckstd, threshold
from cklib.ckstd import fprint
from cklib.cktime import date

if __name__ == "__main__":
    seed = 22
    ts = timeit.default_timer()
    print('Load dataframe')
    step = joblib.load('./bin/step.dat')
    df = joblib.load('./bin/dataframe.dat')
    ppreds_train = df['ppred']
    pprobs_train = df['pprob']
    spreds_train = df['spred']
    sprobs_train = df['sprob']
    train_flows = df['flow']
    label_encoder = df['le']
    y_true = df['y_true']

    for s in range(len(step)):
        step[s] = step[s][step[s] >= 0.5]

    new_step = []
    for s in step:
        new_step.append(list(reversed(s.tolist())))
    te = timeit.default_timer()
    print('---> Done ({:.6f} sec)'.format(te - ts))

    lthclass = threshold.LThreshold(h_threshold = 1.0, random_state = seed, verbose = True)

    lthclass.initalizing(
        ppreds = ppreds_train,
        pprobs = pprobs_train,
        spreds = spreds_train,
        sprobs = sprobs_train,
        flows = train_flows,
        delta = new_step,
        classes = label_encoder.transform(label_encoder.classes_),
        y_true = label_encoder.transform(y_true)
    )

    lthclass.approximate()
    lthclass.gradient(times = 1, limit = -1)
    l_threshold = lthclass.getThreshold()

    print(l_threshold)
