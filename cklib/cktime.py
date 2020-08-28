import datetime

def time_stamp():
    return '[{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}]'.format((datetime.datetime.now() + datetime.timedelta(hours = 9)).year, (datetime.datetime.now() + datetime.timedelta(hours = 9)).month, (datetime.datetime.now() + datetime.timedelta(hours = 9)).day, (datetime.datetime.now() + datetime.timedelta(hours = 9)).hour, (datetime.datetime.now() + datetime.timedelta(hours = 9)).minute, (datetime.datetime.now() + datetime.timedelta(hours = 9)).second)

def date():
    return '{}-{:02d}-{:02d}'.format((datetime.datetime.now() + datetime.timedelta(hours = 9)).year, (datetime.datetime.now() + datetime.timedelta(hours = 9)).month, (datetime.datetime.now() + datetime.timedelta(hours = 9)).day)