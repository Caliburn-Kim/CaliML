'''
    Library: cklib.ckmachine
    
    For supporting machine learning methods
'''

# All imports
from __future__ import print_function

import itertools
import numpy as np
import timeit
import gc
import pandas as pd
import os, sys
import threading

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import cklib.ckconst as ckc
from cklib.ckstd import fprint
from cklib.cktime import time_stamp
from cklib import ckstd

import matplotlib.pyplot as plt

class HThreshold:
    def __init__(self, random_state = None, verbose = True):
        self.ppreds = None
        self.pprobs = None
        self.spreds = None
        self.flows = None
        self.classes = None
        self.y_true = None
        self.h_threshold = None
        self.init_th = None
        self.initial_threshold = None
        self.step = None
        self.isInit = False
        self.verbose = verbose
        np.random.seed(random_state)
        
        if verbose:
            self.log = open('/tf/md0/thkim/log/' + time_stamp() + '.log', 'a')
        else:
            self.log = None
            
    def initalizing(self, ppreds, pprobs, spreds, flows, classes, y_true, initial_threshold = 0.9):
        self.ppreds = ppreds
        self.pprobs = pprobs
        self.spreds = spreds
        self.flows = flows
        self.classes = classes
        self.y_true = y_true
        self.initial_threshold = initial_threshold
        self.h_threshold = [round((np.random.rand() % 0.05) + self.initial_threshold, 3) for c in self.classes]
        self.h_threshold = [initial_threshold for c in self.classes]
        self.step = int(100 - (self.initial_threshold * 100) + 1)
        self.init_th = self.h_threshold.copy()
        self.isInit = True
        fprint(self.log, 'Initializing compilte')
        return '<Initializing Threshold class>'
    
    def momentom(self, prev_delta, diff, step):
        return round(0.9 * step + (10 * diff + 0.001) * (step / prev_delta), 3)
    
    def approximate(self):
        assert self.isInit, 'Class Threshold is not initialized'
        
        fprint(self.log, 'Processing finding approximate threshold')
        
        percentage = 0
        timer_deviding = self.step * len(self.classes) / 100
        
        ts = timeit.default_timer()
        
        for ci, cs in enumerate(self.classes):
            f1_scores = []
            for h_step in range(self.step):
                classified = []
                for flow_idx, fpreds, fprobs in zip(range(len(self.flows)), self.ppreds, self.pprobs):
                    found = False

                    for pkt_idx, pred, prob in zip(range(len(fpreds)), fpreds, fprobs):
                        if (self.h_threshold[pred] <= prob):
                            classified.append(pred)
                            found = True
                            break

                    if not found:
                        classified.append(self.spreds[flow_idx])

                f1_scores.append(f1_score(y_true = self.y_true, y_pred = classified, labels = self.classes, average = 'macro'))

                self.h_threshold[ci] += 0.01
                self.h_threshold[ci] = round(self.h_threshold[ci], 2)
                percentage += 1
                te = timeit.default_timer()
                if self.verbose:
                    print('Processing {:.3f}% ({:.4f} seconds)'.format(percentage / timer_deviding, te - ts), end = '\r')

            self.h_threshold[ci] = round(self.init_th[ci] + (np.argmax(f1_scores) / 100), 2)
            fprint(self.log, 'Now threshold: {}'.format(self.h_threshold))
            
        self.h_threshold = [{hth > self.initial_threshold:round(hth - 0.001, 3)}.get(True, hth) for hth in self.h_threshold]
        fprint(self.log, 'Found approximate threshold: {}'.format(self.h_threshold))
        return '<Appoximate function>'
    
    def gradient(self, delta_step = 0.01, times = 1, limit = 10):
        assert self.isInit != None, 'Class Threshold is not initialized'
        
        search_count = 0
        solstice = False
        delta = delta_step
        prev_base = []
        fprint(self.log, 'Start threshold: {}'.format(self.h_threshold))
        fprint(self.log, 'Print threshold every {} times'.format(times))
        
        ts = timeit.default_timer()
        prev_base.append(self.h_threshold.copy())
        while (not solstice):
            search_count += 1
            f1_scores = []
            '''
                Base point F1-score
            '''
            classified = []
            for flow_idx, fpreds, fprobs in zip(range(len(self.flows)), self.ppreds, self.pprobs):
                found = False

                for pkt_idx, pred, prob in zip(range(len(fpreds)), fpreds, fprobs):
                    if (self.h_threshold[pred] <= prob):
                        classified.append(pred)
                        found = True
                        break

                if not found:
                    classified.append(self.spreds[flow_idx])

            classified = np.array(classified)
            base_f1 = f1_score(y_true = self.y_true, y_pred = classified, labels = self.classes, average = 'macro')

            '''
                Surronding points F1-score
            '''
            
            for ci, cs in enumerate(self.classes):
                for i in [-delta, delta]:
                    self.h_threshold[ci] += i
                    self.h_threshold[ci] = round(self.h_threshold[ci], 3)

                    if self.h_threshold[ci] > 1:
                        self.h_threshold[ci] -= i
                        self.h_threshold[ci] = round(self.h_threshold[ci], 3)
                        f1_scores.append([ci, i, 0.])
                        continue
                    if self.h_threshold[ci] < 0.5:
                        self.h_threshold[ci] -= i
                        self.h_threshold[ci] = round(self.h_threshold[ci], 3)
                        f1_scores.append([ci, i, 0.])
                        continue

                    classified = []
                    for flow_idx, fpreds, fprobs in zip(range(len(self.flows)), self.ppreds, self.pprobs):
                        found = False

                        for pkt_idx, pred, prob in zip(range(len(fpreds)), fpreds, fprobs):
                            if (self.h_threshold[pred] <= prob):
                                classified.append(pred)
                                found = True
                                break

                        if not found:
                            classified.append(self.spreds[flow_idx])

                    classified = np.array(classified)
                    f1_scores.append([ci, i, f1_score(y_true = self.y_true, y_pred = classified, labels = self.classes, average = 'macro')])

                    self.h_threshold[ci] -= i
                    self.h_threshold[ci] = round(self.h_threshold[ci], 3)

            chg_th_idx = np.argmax(f1_scores, axis = 0)[-1]
            max_f1 = f1_scores[chg_th_idx]
            te = timeit.default_timer()
            
            if max_f1[2] < base_f1:
                solstice = True
                fprint(self.log, 'Total process count: {} ({:.4f} seconds)'.format(search_count, te - ts))
            else:
                diff = max_f1[2] - base_f1
#                 delta = self.momentom(prev_delta = delta, diff = diff, step = max_f1[1])
                prev_th = self.h_threshold[max_f1[0]]
                self.h_threshold[max_f1[0]] += max_f1[1]
                
                if (self.h_threshold[max_f1[0]] > 1) & (prev_th != 0.999):
                    self.h_threshold[max_f1[0]] = 0.999
                elif ((self.h_threshold[max_f1[0]] > 1) & (prev_th == 0.999)) or ((self.h_threshold[max_f1[0]] > 1) & (prev_th == 1.0)):
                    self.h_threshold[max_f1[0]] = 1.
                if self.h_threshold[max_f1[0]] < 0.5:
                    self.h_threshold[max_f1[0]] = 0.5
                    
                self.h_threshold[max_f1[0]] = round(self.h_threshold[max_f1[0]], 3)
                
                if self.h_threshold in prev_base:
                    solstice = True
                    fprint(self.log, 'Threshold is looping --> process will be terminated')
                    fprint(self.log, 'Total process count: {} ({:.4f} seconds)'.format(search_count, te - ts))
#                     fprint(self.log, 'Now: {}'.format(self.h_threshold))
#                     fprint(self.log, 'Prv: {}'.format(prev_base))
                else:
                    prev_base.append(self.h_threshold.copy())
                
                if self.verbose:
                    print('[Base: {}] [Max: {}] [difference: {}] [delta: {}]'.format(base_f1, max_f1[2], diff, max_f1[1]))

            if search_count % times == 0:
                fprint(self.log, '{} --> {} ({:.4f} seconds)'.format(search_count, self.h_threshold, te - ts))

            if search_count > limit:
                fprint(self.log, 'Process count is over than {} --> Stop process ({:.4f} seconds)'.format(limit, te - ts))
                break

        print('')
        if solstice:
            fprint(self.log, 'Found threshold: {}'.format(self.h_threshold))
        else:
            fprint(self.log, 'Process stoped threshold: {}'.format(self.h_threshold))
        
        return '<Gradient function>'
    
    def getThreshold(self):
        return self.h_threshold

    
class LThreshold:
    def __init__(self, h_threshold = 1.0, random_state = None, verbose = True):
        self.ppreds = None
        self.pprobs = None
        self.spreds = None
        self.sprobs = None
        self.flows = None
        self.classes = None
        self.y_true = None
        self.h_threshold = h_threshold
        self.l_threshold = None
        self.init_th = None
        self.initial_threshold = None
        self.step = None
        self.isInit = False
#         self.delta = None
        self.verbose = verbose
        np.random.seed(random_state)
        
        if verbose:
            self.log = open('/tf/md0/thkim/log/' + time_stamp() + '.log', 'a')
        else:
            self.log = None
            
    def initalizing(self, ppreds, pprobs, spreds, sprobs, flows, classes, y_true, initial_threshold = 0.5):
        self.ppreds = ppreds
        self.pprobs = pprobs
        self.spreds = spreds
        self.sprobs = sprobs
        self.flows = flows
        self.classes = classes
#         self.delta = delta
        self.y_true = y_true
        self.initial_threshold = initial_threshold
#         self.l_threshold = [round((np.random.rand() % 0.05) + self.initial_threshold, 3) for c in self.classes]
        self.l_threshold = [self.initial_threshold for c in self.classes]
        self.step = self.getStep()
        self.init_th = self.l_threshold.copy()
        self.isInit = True
        fprint(self.log, 'Initializing compilte')
        return '<Initializing Threshold class>'
    
    def getStep(self):
        base = int(self.h_threshold * 10) * 10
        step = int(base - (self.initial_threshold * 100) + 1)
        return step
    
    def approximate(self):
        assert self.isInit, 'Class Threshold is not initialized'
        
        fprint(self.log, 'Processing finding approximate threshold')
        
        percentage = 0
        timer_deviding = self.step * len(self.classes) / 100
        
        ts = timeit.default_timer()
        
        for ci, cs in enumerate(self.classes):
            f1_scores = []
            for h_step in range(self.step):
                classified = []
                for flow_idx, fpreds, fprobs in zip(range(len(self.flows)), self.ppreds, self.pprobs):
                    found = False

                    for pkt_idx, pred, prob in zip(range(len(fpreds)), fpreds, fprobs):
                        if (self.h_threshold <= prob):
                            classified.append(pred)
                            found = True
                            break
                            
                    if not found:
                        if (self.l_threshold[self.spreds[flow_idx]] <= self.sprobs[flow_idx]):
                            classified.append(self.spreds[flow_idx])
                        else:
                            max_prob_idx = np.argmax(fprobs)
                            classified.append(fpreds[max_prob_idx])

                f1_scores.append(f1_score(y_true = self.y_true, y_pred = classified, labels = self.classes, average = 'macro'))

                self.l_threshold[ci] += 0.01
                self.l_threshold[ci] = round(self.l_threshold[ci], 2)
                percentage += 1
                te = timeit.default_timer()
                if self.verbose:
                    print('Processing {:.3f}% ({:.4f} seconds)'.format(percentage / timer_deviding, te - ts), end = '\r')

            self.l_threshold[ci] = round(self.init_th[ci] + (np.argmax(f1_scores) / 100), 2)
            fprint(self.log, 'Max F1: {} --> Now threshold: [{}]{}'.format(np.argmax(f1_scores), self.h_threshold, self.l_threshold))
            
#         self.l_threshold = [{th > self.initial_threshold:round(th - 0.01, 2)}.get(True, th) for th in self.l_threshold]
        fprint(self.log, 'Found approximate threshold: {}'.format(self.l_threshold))
        return '<Appoximate function>'
    
    def gradient(self, delta_step = 0.01, times = 1, limit = 10):
        assert self.isInit != None, 'Class Threshold is not initialized'
        
        search_count = 0
        path_count = 0
        solstice = False
        delta = delta_step
        prev_base = []
        fprint(self.log, 'Start threshold: [{}]{}'.format(self.h_threshold, self.l_threshold))
        fprint(self.log, 'Print threshold every {} times'.format(times))
        
        ts = timeit.default_timer()
        prev_base.append(self.l_threshold.copy())
        while (not solstice):
            search_count += 1
            f1_scores = []
            '''
                Base point F1-score
            '''
            classified = []
            for flow_idx, fpreds, fprobs in zip(range(len(self.flows)), self.ppreds, self.pprobs):
                found = False

                for pkt_idx, pred, prob in zip(range(len(fpreds)), fpreds, fprobs):
                    if (self.h_threshold <= prob):
                        classified.append(pred)
                        found = True
                        break

                if not found:
                    if (self.l_threshold[self.spreds[flow_idx]] <= self.sprobs[flow_idx]):
                        classified.append(self.spreds[flow_idx])
                    else:
                        max_prob_idx = np.argmax(fprobs)
                        classified.append(fpreds[max_prob_idx])

            classified = np.array(classified)
            base_f1 = f1_score(y_true = self.y_true, y_pred = classified, labels = self.classes, average = 'macro')

            '''
                Surronding points F1-score
            '''
            
            for ci, cs in enumerate(self.classes):
                for i in [-delta, delta]:
                    self.l_threshold[ci] += i
                    self.l_threshold[ci] = round(self.l_threshold[ci], 3)

                    if self.l_threshold[ci] > 1:
                        self.l_threshold[ci] -= i
                        self.l_threshold[ci] = round(self.l_threshold[ci], 3)
                        f1_scores.append([ci, i, 0.])
                        continue
                    if self.l_threshold[ci] < 0.5:
                        self.l_threshold[ci] -= i
                        self.l_threshold[ci] = round(self.l_threshold[ci], 3)
                        f1_scores.append([ci, i, 0.])
                        continue
                        
                    if self.l_threshold in prev_base:
                        self.l_threshold[ci] -= i
                        self.l_threshold[ci] = round(self.l_threshold[ci], 3)
                        f1_scores.append([ci, i, 0.])
                        continue

                    classified = []
                    for flow_idx, fpreds, fprobs in zip(range(len(self.flows)), self.ppreds, self.pprobs):
                        found = False

                        for pkt_idx, pred, prob in zip(range(len(fpreds)), fpreds, fprobs):
                            if (self.h_threshold <= prob):
                                classified.append(pred)
                                found = True
                                break

                        if not found:
                            if (self.l_threshold[self.spreds[flow_idx]] <= self.sprobs[flow_idx]):
                                classified.append(self.spreds[flow_idx])
                            else:
                                max_prob_idx = np.argmax(fprobs)
                                classified.append(fpreds[max_prob_idx])

                    classified = np.array(classified)
                    f1_scores.append([ci, i, f1_score(y_true = self.y_true, y_pred = classified, labels = self.classes, average = 'macro')])

                    self.l_threshold[ci] -= i
                    self.l_threshold[ci] = round(self.l_threshold[ci], 3)

            chg_th_idx = np.argmax(f1_scores, axis = 0)[-1]
            max_f1 = f1_scores[chg_th_idx]
            te = timeit.default_timer()
            
            if max_f1[2] < base_f1:
                solstice = True
                fprint(self.log, 'Total process count: {} ({:.4f} seconds)'.format(search_count, te - ts))
            else:
                diff = max_f1[2] - base_f1
                
                prev_th = self.l_threshold[max_f1[0]]
                self.l_threshold[max_f1[0]] += max_f1[1]
        
                if self.l_threshold[max_f1[0]] > 1.:
                    self.l_threshold[max_f1[0]] = 1.
                    
                if self.l_threshold[max_f1[0]] < 0.5:
                    self.l_threshold[max_f1[0]] = 0.5
                    
                self.l_threshold[max_f1[0]] = round(self.l_threshold[max_f1[0]], 3)
                
                prev_base.append(self.l_threshold.copy())
                
                if self.verbose:
                    print('[Base: {}] [Max: {}] [difference: {}] [delta: {}]'.format(base_f1, max_f1[2], diff, max_f1[1]))

            if search_count % times == 0:
                fprint(self.log, '{} --> {} ({:.4f} seconds)'.format(search_count, self.l_threshold, te - ts))

            if search_count > limit:
                fprint(self.log, 'Process count is over than {} --> Stop process ({:.4f} seconds)'.format(limit, te - ts))
                break

        print('')
        if solstice:
            fprint(self.log, 'Found threshold: [{}]{}'.format(self.h_threshold, self.l_threshold))
        else:
            fprint(self.log, 'Process stoped threshold: [{}]{}'.format(self.h_threshold, self.l_threshold))
        
        return '<Gradient function>'
    
    def getThreshold(self):
        return self.l_threshold
    
