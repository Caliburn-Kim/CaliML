from __future__ import print_function

import joblib
from cklib import ckstd
from cklib.ckstd import fprint
from cklib import ckmachine
import timeit
import numpy as np
from sklearn.metrics import f1_score
from cklib.cktime import date

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
        self.step = None
        self.isInit = False
        self.delta = None
        self.verbose = verbose
        np.random.seed(random_state)
        
        if verbose:
            self.log = open('./log/' + date() + '.log', 'a')
        else:
            self.log = None
            
    def initalizing(self, ppreds, pprobs, spreds, sprobs, flows, classes, y_true, delta):
        self.ppreds = ppreds
        self.pprobs = pprobs
        self.spreds = spreds
        self.sprobs = sprobs
        self.flows = flows
        self.classes = classes
        self.delta = delta
        self.y_true = y_true
        self.l_threshold = [delta[c][0] for c in range(len(self.classes))]
        self.step = self.getStep()
        self.init_th = self.l_threshold.copy()
        self.isInit = True
        fprint(self.log, 'Initializing compilte')
        return '<Initializing Threshold class>'
    
    def getStep(self):
        return int((self.h_threshold - 0.5) * 10) * 10 + 1
    
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

                self.l_threshold[ci] -= 0.01
                self.l_threshold[ci] = round(self.l_threshold[ci], 2)
                percentage += 1
                te = timeit.default_timer()
                if self.verbose:
                    print('Processing {:.3f}% ({:.4f} seconds)'.format(percentage / timer_deviding, te - ts), end = '\r')

            self.l_threshold[ci] = round(self.init_th[ci] - (np.argmax(f1_scores) / 100), 2)
            fprint(self.log, 'Max F1: {} --> Now threshold: [{}]{}'.format(np.argmax(f1_scores), self.h_threshold, self.l_threshold))
            
#         self.l_threshold = [{th > self.initial_threshold:round(th - 0.01, 2)}.get(True, th) for th in self.l_threshold]
        fprint(self.log, 'Found approximate threshold: {}'.format(self.l_threshold))
        return '<Appoximate function>'
    
    def gradient(self, times = 1, limit = -1):
        assert self.isInit, 'Class Threshold is not initialized'
        
        start_point = []
        last_idx = None
        for c in range(len(self.classes)):
            for i, k in enumerate(self.delta[c]):
                if k >= self.l_threshold[c]:
                    last_idx = i
            start_point.append(last_idx)

        search_count = 0
        path_count = 0
        solstice = False
        prev_base = []
        fprint(self.log, 'Start threshold: [{}]{}'.format(self.h_threshold, self.l_threshold))
        if limit < 0:
            fprint(self.log, 'Find infinity')
        else:
            fprint(self.log, 'Find limit: {}'.format(limit))
        fprint(self.log, 'Print threshold every {} times'.format(times))
        d_position = [start_point[i] for i in range(len(self.classes))]
        max_point = [len(self.delta[c]) - 1 for c in range(len(self.classes))]
        
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

            for class_idx, c in enumerate(self.classes):
                if (d_position[class_idx] + 1  > max_point[class_idx]):
                    continue
                self.l_threshold[class_idx] = self.delta[class_idx][d_position[class_idx] + 1]
                self.l_threshold[class_idx] = round(self.l_threshold[class_idx], 3)

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
                f1_scores.append([class_idx, f1_score(y_true = self.y_true, y_pred = classified, labels = self.classes, average = 'macro' )])

                self.l_threshold[class_idx] = self.delta[class_idx][d_position[class_idx]]
                self.l_threshold[class_idx] = round(self.l_threshold[class_idx], 3)
            
            tmp_f1 = np.max(f1_scores, axis = 0)[-1]
            f1_scores = np.array(f1_scores, dtype = np.object)
            tmp_f1 = f1_scores[f1_scores[:, 1] == tmp_f1]
            max_f1 = np.squeeze(tmp_f1[np.random.choice(np.arange(len(tmp_f1)), 1)])
            te = timeit.default_timer()

            if max_f1[1] < base_f1:
                solstice = True
                prev_base = []
                prev_base.append(self.l_threshold.copy())
                fprint(self.log, 'Total process count: {} ({:.4f} seconds)'.format(search_count, te - ts))
            else:
                diff = (max_f1[1] - base_f1)
                d_position[max_f1[0]] += 1
                self.l_threshold[class_idx] = self.delta[max_f1[0]][d_position[max_f1[0]]]

                if max_f1[1] > base_f1:
                    prev_base = []
                    
                prev_base.append(self.l_threshold.copy())

                if self.verbose:
                    print(
                        '[{:3d}][Base: {:.6f}] [Max: {:.6f}] [diff: {:.6f}] [class: {:2d}] [delta: {:.3f}] ({:.4f} sec)'.format(
                            search_count,
                            base_f1,
                            max_f1[1],
                            diff,
                            max_f1[0],
                            self.delta[max_f1[0]][d_position[max_f1[0]]],
                            te - ts
                        ), end = '\r'
                    )

            if (search_count > limit) & (limit > -1):
                fprint(self.log, 'Process count is over than {} --> Stop process ({:.4f} seconds)'.format(limit, te - ts))
                break

        print('')
        if solstice:
            print('[{:3d}][Base: {:.6f}] [Max: {:.6f}] [diff: {:.6f}] [class: {:2d}] [delta: {:.3f}] ({:.4f} sec)'.format(search_count, base_f1, max_f1[1], diff, max_f1[0], self.delta[max_f1[0]][d_position[max_f1[0]]], te - ts))
            fprint(self.log, 'Found threshold: [{}]{}'.format(self.h_threshold, self.l_threshold))
        else:
            last_th = self.l_threshold.copy()
            self.l_threshold = prev_base[0]
            fprint(self.log, 'Last threhsold will be set the same f1-score threhsold')
            fprint(self.log, '{} ---> {}'.format(last_th, self.l_threshold))
        
        return '<Gradient function>'
    
    def getThreshold(self):
        return self.l_threshold
