from __future__ import print_function

from cklib.cktime import time_stamp

import math
import itertools
import numpy as np
from IPython.display import display_html
from pyarrow import csv
import imp
import cklib.cktime as cktime
import threading
import timeit
import time as timelib
import numpy as np

def fprint(fobject = None, message = '', end = '\n'):
    print('{} {}'.format(time_stamp(), message), end = end)
    
    if fobject is not None:
        fobject.write('{} {}'.format(time_stamp(), message) + end)
        fobject.flush()
    return

def floor(number, digit):
    return math.floor(number * (10 ** digit)) / float(10 ** digit)

def restart():
    return display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)

def read_csv(path):
    return csv.read_csv(path).to_pandas()

def flatten(__list__):
    return list(itertools.chain.from_iterable(__list__))

def print_cfmx(cfmx, length = 10):
    for i in range(len(cfmx)):
        for j in range(len(cfmx[0])):
            print('{value:{width}}'.format(value = cfmx[i][j], width = length), end = '')
        print('')
    print('\nRow: Predict, Col: Real')
    return

def reload(module_name):
    return imp.reload(module_name)

def time(sec):
    min = sec // 60
    hour = min // 60
    sec = sec % 60
    if hour == 0:
        if min == 0:
            return '{} seconds'.format(sec)
        else:
            return '{} minutes {} seconds'.format(min, sec)
    else:
        return '{} hours {} minutes {} seconds'.format(hour, min, sec)
    
def log_format(*args):
    strname = '[' + cktime.date() + ']'
    for arg in args[:-1]:
        strname = strname + '[' + arg + ']'
    strname = strname + args[-1] + '.log'
    return strname

def out_format(*args):
    strname = ''
    for arg in args[:-1]:
        strname = strname + '[' + arg + ']'
    strname = strname + args[-1] + '.out'
    return strname

class Timer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name='Timer Thread')
        self.stopper = False
        
    def run(self):
        count = 0
        ts = timeit.default_timer()
        te = timeit.default_timer()
        while (not self.stopper):
            timelib.sleep((np.random.rand() % 0.3) + 0.7)
            te = timeit.default_timer()
            print('Loading: {:.4f} seconds'.format(te - ts), end = '\r')
        print('Loading: {:.4f} seconds'.format(te - ts))
        
    def stop(self):
        self.stopper = True
        print('Kill timer')