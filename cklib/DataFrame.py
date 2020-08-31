'''
    Library: cklib.ckdata
    
    For management dataset
'''

# All imports
from __future__ import print_function

import itertools
import numpy as np
import timeit
import gc
import pandas as pd
import multiprocessing
import os, sys

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

import cklib.ckconst as ckc
from cklib import cksess
from cklib.ckstd import fprint
from cklib.cktime import date
from cklib import ckstd

class Flow_Dataset:
    def __init__(self, split = 0.7, logging = True, random_state = None):
        self.dataset = None
        self.session = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_session = None
        self.test_session = None
        
        self.flows = None
        self.train_flows = None
        self.test_flows = None
        
        self.split_ratio = split
        self.train_size = None
        self.seed = random_state
        self.skip_datas = []
        
        use_cores = multiprocessing.cpu_count() // 3 * 2
        
        self.pclf = RandomForestClassifier(n_jobs = use_cores, random_state = random_state)
        self.sclf = RandomForestClassifier(n_jobs = use_cores, random_state = random_state)
        self.le = LabelEncoder()
        self.pscaler = MinMaxScaler()
        self.sscaler = MinMaxScaler()
        
        self.spreds_train = None
        self.spreds_test = None
        self.sprobs_train = None
        self.sprobs_train_all = None
        self.sprobs_test = None
        self.sprobs_test_all = None
        
        self.ppreds_train = None
        self.ppreds_test = None
        self.pprobs_train = None
        self.pprobs_train_all = None
        self.pprobs_test = None
        self.pprobs_test_all = None

        self.pkt_train_ptime_mean = None
        self.pkt_test_ptime_mean = None
        
        if logging:
            self.log = open('/tf/md0/thkim/log/' + date() + '.log', 'a')
        else:
            self.log = None
        
    def read_csv(self, path, encoding = ckc.ISCX_DATASET_ENCODING):
        fprint(self.log, 'Reading dataset: {}'.format(path))
        ts = timeit.default_timer()
        self.dataset = pd.read_csv(filepath_or_buffer = path, encoding = encoding).values
        
        fprint(self.log, 'Skip data: {}'.format(self.skip_datas))
        for word in self.skip_datas:
            self.dataset = self.dataset[self.dataset[:, -1] != word]
            
        self.flows = cksess.get_flows(dataset = self.dataset)
        self.train_size = int(len(self.flows) * self.split_ratio)
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Shuffling dataset by flows')
        ts = timeit.default_timer()
        self.dataset, _ = cksess.shuffle_flow(dataset = self.dataset, flows = self.flows, random_state = self.seed)
        self.flows = cksess.get_flows(dataset = self.dataset)
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Creating training & test dataset')
        ts = timeit.default_timer()
        self.session = self.dataset[[flow[-1] for flow in self.flows]]
        self.train_session = self.session[:self.train_size]
        self.test_session = self.session[self.train_size:]
        self.train_dataset = self.dataset[cksess.flatten(self.flows[:self.train_size])]
        self.test_dataset = self.dataset[cksess.flatten(self.flows[self.train_size:])]
        self.train_flows = cksess.get_flows(dataset = self.train_dataset)
        self.test_flows = cksess.get_flows(dataset = self.test_dataset)
        gc.collect()
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        return '<Function: read & shuffling csv>'
        
    def skip_data(self, *skip):
        for word in skip:
            self.skip_datas.append(word)
        return '<Function: skip data>'
    
    def getLabelEncoder(self):
        return self.le
    
    def getClassifier(self):
        return self.pclf, self.sclf
    
    def modelling(self):
        fprint(self.log, 'Training label encoder and scaler')
        ts = timeit.default_timer()
        self.le.fit(self.session[:, -1])
        self.pscaler.fit(self.train_dataset[:, 2:-1])
        self.sscaler.fit(self.train_session[:, 1:-1])
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Training random forest model')
        ts = timeit.default_timer()
        self.pclf.fit(
            X = self.pscaler.transform(self.train_dataset[:, 2:-1]),
            y = self.le.transform(self.train_dataset[:, -1])
        )
        
        self.sclf.fit(
            X = self.sscaler.transform(self.train_session[:, 1:-1]),
            y = self.le.transform(self.train_session[:, -1])
        )
        
        gc.collect()
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)'.format(te - ts))
        
        return '<Function: modelling>'
    
    def predict(self):
        pred_ts = timeit.default_timer()
        
        fprint(self.log, 'Predict session training dataset')
        ts = timeit.default_timer()
        self.spreds_train = self.sclf.predict(self.sscaler.transform(self.train_session[:, 1:-1]))
        te = timeit.default_timer()
        fprint(self.log, 'Session training dataset predict time: {} seconds'.format(te - ts))

        fprint(self.log, 'Predict session test dataset')
        ts = timeit.default_timer()
        self.spreds_test = self.sclf.predict(self.sscaler.transform(self.train_session[:, 1:-1]))
        te = timeit.default_timer()
        fprint(self.log, 'Session test dataset predict time: {} seconds'.format(te - ts))
        self.sprobs_train_all = self.sclf.predict_proba(self.sscaler.transform(self.train_session[:, 1:-1]))
        self.sprobs_train = np.max(self.sprobs_train_all, axis = 1)
        self.sprobs_test_all = self.sclf.predict_proba(self.sscaler.transform(self.test_session[:, 1:-1]))
        self.sprobs_test = np.max(self.sprobs_test_all, axis = 1)

        fprint(self.log, 'Predict packet training dataset')
        ts = timeit.default_timer()
        self.ppreds_train = self.pclf.predict(self.pscaler.transform(self.train_dataset[:, 2:-1]))
        te = timeit.default_timer()
        self.pkt_train_ptime_mean = (te - ts) / len(self.ppreds_train)
        self.ppreds_train = [self.ppreds_train[flow] for flow in self.train_flows]
        self.pprobs_train_all = self.pclf.predict_proba(self.pscaler.transform(self.train_dataset[:, 2:-1]))
        self.pprobs_train = np.max(self.pprobs_train_all, axis = 1)
        self.pprobs_train_all = [self.pprobs_train_all[flow] for flow in self.train_flows]
        self.pprobs_train = [self.pprobs_train[flow] for flow in self.train_flows]
        fprint(self.log, 'Packet training dataset predict time: {} seconds'.format(te - ts))

        fprint(self.log, 'Predict packet test dataset')
        ts = timeit.default_timer()
        self.ppreds_test = self.pclf.predict(self.pscaler.transform(self.test_dataset[:, 2:-1]))
        te = timeit.default_timer()
        packet_test_pred_time = te - ts
        self.pkt_test_ptime_mean = packet_test_pred_time / len(self.ppreds_test)
        self.ppreds_test = [self.ppreds_test[flow] for flow in self.test_flows]
        self.pprobs_test_all = self.pclf.predict_proba(self.pscaler.transform(self.test_dataset[:, 2:-1]))
        self.pprobs_test = np.max(self.pprobs_test_all, axis = 1)
        self.pprobs_test_all = [self.pprobs_test_all[flow] for flow in self.test_flows]
        self.pprobs_test = [self.pprobs_test[flow] for flow in self.test_flows]
        fprint(self.log, 'Packet test dataset predict time: {} seconds'.format(te - ts))

        pred_te = timeit.default_timer()
        fprint(self.log, 'Processing of predict part is finished ({} seconds)'.format(pred_te - pred_ts))
        
        return '<Function: predict>'
    
    def getTrainPredict(self):
        return self.ppreds_train, self.spreds_train, self.pprobs_train, self.sprobs_train
    
    def getTrainProbability(self):
        return self.pprobs_train_all
    
    def getTestPredict(self):
        return self.ppreds_test, self.pprobs_train
    
    def getTestProbability(self):
        return self.pprobs_test_all
    
    def getTrainFlow(self):
        return self.train_flows
    
    def getTrainLabel(self):
        return self.train_session[:, -1]


class Session_Dataset:
    def __init__(self, split = 0.7, logging = True, random_state = None):
        self.train_dataset = None
        self.test_dataset = None

        self.split_ratio = split
        self.train_size = None
        self.seed = random_state
        self.skip_datas = []
        
        np.random.seed(random_state)
        use_cores = multiprocessing.cpu_count() // 3 * 2
        self.sclf = RandomForestClassifier(n_jobs = use_cores, random_state = random_state)
        self.le = LabelEncoder()
        self.scaler = MinMaxScaler()
        
        self.spreds_train = None
        self.spreds_test = None

        if logging:
            self.log = open('./log/' + date() + '.log', 'a')
        else:
            self.log = None

    def skip_data(self, *skip):
        for word in skip:
            self.skip_datas.append(word)
        return '<Function: skip data>'

    def read_csv(self, path, encoding = ckc.ISCX_DATASET_ENCODING):
        fprint(self.log, 'Reading dataset: {}'.format(path))
        ts = timeit.default_timer()
        dataset = pd.read_csv(filepath_or_buffer = path, encoding = encoding).values
        
        fprint(self.log, 'Skip data: {}'.format(self.skip_datas))
        for word in self.skip_datas:
            dataset = dataset[dataset[:, -1] != word]
            
        flows = cksess.get_flows(dataset = dataset)
        dataset = dataset[[flow[-1] for flow in flows]]
        self.train_size = int(len(flows) * self.split_ratio)
        flows = None
        del flows
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Shuffling dataset by flows')
        ts = timeit.default_timer()
        np.random.shuffle(dataset)
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Creating training & test dataset')
        ts = timeit.default_timer()
        self.train_dataset = dataset[:self.train_size]
        self.test_dataset = dataset[self.train_size:]
        gc.collect()
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        return '<Function: read & shuffling csv>'

    def modelling(self):
        fprint(self.log, 'Training label encoder and scaler')
        ts = timeit.default_timer()
        self.le.fit(self.train_dataset[:, -1])
        self.le.fit(self.test_dataset[:, -1])
        self.scaler.fit(self.train_dataset[:, 1:-1])
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Training random forest model')
        ts = timeit.default_timer()

        self.sclf.fit(
            X = self.scaler.transform(self.train_dataset[:, 1:-1]),
            y = self.le.transform(self.train_dataset[:, -1])
        )
        
        gc.collect()
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)'.format(te - ts))
        
        return '<Function: modelling>'

    def predict(self):
        pred_ts = timeit.default_timer()
        
        fprint(self.log, 'Predict session training dataset')
        ts = timeit.default_timer()
        self.spreds_train = self.sclf.predict(self.scaler.transform(self.train_dataset[:, 1:-1]))
        te = timeit.default_timer()
        fprint(self.log, 'Session training dataset predict time: {} seconds'.format(te - ts))

        fprint(self.log, 'Predict session test dataset')
        ts = timeit.default_timer()
        self.spreds_test = self.sclf.predict(self.scaler.transform(self.train_dataset[:, 1:-1]))
        te = timeit.default_timer()
        fprint(self.log, 'Session test dataset predict time: {} seconds'.format(te - ts))

    def getTrainPredict(self):
        return self.spreds_train

    def getTestPredict(self):
        return self.spreds_test

    def getLabelEncoder(self):
        return self.le

    def getTrainLabel(self):
        return self.train_dataset[:, -1]

    def getTestLabel(self):
        return self.test_dataset[:, -1]

    def getClassifier(self):
        return self.sclf
