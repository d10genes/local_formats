import time
import numpy.random as nr


class Timer(object):

    def __init__(self, start=True):
        self.end_time = None
        if start:
            self.start()
        else:
            self.st_time = None

    def start(self):
        self.st_time = time.perf_counter()

    def end(self):
        self.end_time = time.perf_counter()
        return self.time

    @property
    def time(self):
        return self.end_time - self.st_time


get_dtypes = lambda df, tp='category': df.columns[df.dtypes == tp]


def mod_cols(df, f=None, cols=None):
    df = df.copy()
    if cols is None:
        cols = get_dtypes(df, object)

    for c in cols:
        df[c] = f(df[c].copy())
    return df


def add_nulls(s, size=100):
    rand_ixs = nr.choice(s.index, size=size, replace=False)
    s.loc[rand_ixs] = None
    return s

