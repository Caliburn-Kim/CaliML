from __future__ import print_function

import cklib.ckconst as ckc
from cklib.ckstd import fprint
from cklib.cktime import time_stamp
from cklib import ckstd

import matplotlib.pyplot as plt

class BidataPlot(threading.Thread):
    def __init__(self, ppreds, pprobs, spreds, sprobs, flows, classes, y_true, lock, class_id = 0, th_range = 10):
        threading.Thread.__init__(self)
        self.ppreds = ppreds
        self.pprobs = pprobs
        self.spreds = spreds
        self.sprobs = sprobs
        self.flows = flows
        self.classes = classes
        self.y_true = y_true
        self.f1s = []
        self.h_threshold = 1.0
        self.find_idx = None
        self.th_range = th_range
        self.lock = lock
        self.cid = class_id
        
        
    def setXY(self, x, y):
        self.x = x
        self.y = y
        
    def run(self):
        ts = timeit.default_timer()
        print('[{}] Preprocessing'.format(self.cid))
        l_thresholds = [[[round(i / self.th_range, 3), round(j / self.th_range, 3)] for j in range(self.th_range + 1)] for i in range(self.th_range + 1)]
        x = self.x
        y = self.y
        
        self.find_idx = np.squeeze(np.hstack([np.where(self.y_true == x), np.where(self.y_true == y)]))
        np.sort(self.find_idx)
#         find_idx = self.y_true.tolist().index(x) + self.y_true.tolist().index(y)
#         find_idx.sort()
        self.ppreds = np.array(self.ppreds)[self.find_idx].tolist()
        self.pprobs = np.array(self.pprobs)[self.find_idx].tolist()
        self.spreds = np.array(self.spreds)[self.find_idx].tolist()
        self.sprobs = np.array(self.sprobs)[self.find_idx].tolist()
        self.flows = np.array(self.flows)[self.find_idx].tolist()
        c = 0
        new_flows = []
        for flow in self.flows:
            tmp = []
            for f in flow:
                tmp.append(c)
                c += 1
            new_flows.append(tmp)
        self.flows = new_flows
        self.y_true = self.y_true[self.find_idx]
        
        percentage = 0
        tot_len = len(l_thresholds) ** 2
        
        te = timeit.default_timer()
        print('[{}]---> Done ({:.4f} seconds)'.format(self.cid, te - ts))
        
        print('[{}]Create f1-scores'.format(self.cid))
        ts = timeit.default_timer()
        for l_threshold in l_thresholds:
            f1_scores = []
            for lth in l_threshold:
                classified = []
                for flow_idx, fpreds, fprobs in zip(range(len(self.flows)), self.ppreds, self.pprobs):
                    found = False

                    for pkt_idx, pred, prob in zip(range(len(fpreds)), fpreds, fprobs):
                        if (self.h_threshold <= prob):
                            if (pred != x) & (pred != y):
                                classified.append(y + 1)
                            else:
                                classified.append(pred)
                            found = True
                            break

                    if not found:
                        if (self.spreds[flow_idx] != x) & (self.spreds[flow_idx] != y):
                            classified.append(y + 1)
                        else:
                            if (lth[self.spreds[flow_idx] - x] <= self.sprobs[flow_idx]):
                                classified.append(self.spreds[flow_idx])
                            else:
                                max_prob_idx = np.argmax(fprobs)
                                classified.append(fpreds[max_prob_idx])

                classified = np.array(classified)
                f1_scores.append(f1_score(y_true = self.y_true, y_pred = classified, average = 'macro'))
                te = timeit.default_timer()
                if percentage % 7 == 0:
                    self.lock.acquire()
                    print('Loading:[{}][{:3.4f}% ({:5.6f} sec)]'.format(self.cid, percentage / tot_len * 100, te - ts), end = '\r')
                    self.lock.release()
                percentage += 1
            
            self.f1s.append(f1_scores)
        self.lock.acquire()
        print('[{}]Loading: {:3.4f}% ({:5.6f} sec)'.format(self.cid, percentage / tot_len * 100, te - ts))
        print('[{}]---> Done ({:.4f} seconds)'.format(self.cid, te - ts))
        self.lock.release()
        return '<Function: processing>'
        
    def plotting(self):
        x = [round(i / self.th_range, 3) for i in range(self.th_range + 1)]
        y = [round(i / self.th_range, 3) for i in range(self.th_range + 1)]

        fig = plt.figure()
        cs = plt.contourf(x, y, self.f1s)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(cs)
        plt.show()
        return '<Function: plotting>'