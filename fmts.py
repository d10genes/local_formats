
# coding: utf-8

# In[ ]:

# import datetime as dt
import itertools as it
import os
import time

import numpy.random as nr
from pandas import pandas as pd, DataFrame
# from pandas.compat import lmap
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')
pd.options.display.notebook_repr_html = False
pd.options.display.width = 120
(";")


# # Load and set up df
# 
# tps = DataFrame({'Nu': dforig.apply(pd.Series.nunique), 'Tp': dforig.dtypes})

# ### String columns

# In[ ]:

# http://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
with open('data/census_col_name_desc_mod.txt', 'r') as f:
    csd = f.read().splitlines()


dforig = pd.read_csv('data/census-income.data', header=None).iloc[:, :-2]
dforig.columns = [l.split()[-1].capitalize() for l in it.takewhile(bool, csd)]


# In[ ]:

# Get a few of each dtype
# ctypes = [c for dtp, gdf in dforig.dtypes.reset_index(drop=0).groupby(0) for c in gdf['index'][:4]]
get_dtypes = lambda df, tp='category': df.columns[df.dtypes == tp]
ctypes = 'Hhdfmx Pemntvty Grinst Areorgn Marsupwt Divval Caploss'.split()
dfs = dforig[ctypes].copy()


# ### Nulls in String cols

# In[ ]:

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


nr.seed(0)
dfsnulls = mod_cols(dfs, f=add_nulls)


# ### Convert string cols to categorical

# In[ ]:

tocat = lambda x: x.astype('category')

dfc = mod_cols(dfs, f=tocat)
dfcnulls = mod_cols(dfsnulls, f=tocat)


# ### Convert string to categorical

# In[ ]:

from collections import OrderedDict


# In[ ]:

def summarize_types(df):
    return DataFrame(OrderedDict([('Dtypes', df.dtypes), ('Nulls', df.isnull().sum())]))


# In[ ]:

summarize_types(dfs)


# In[ ]:

DataFrame({'S': summarize_types(dfs)})


# In[ ]:

summarize_types(dfsnulls)


# In[ ]:

ix = pd.MultiIndex.from_product([
    ['Str', 'Str_null', 'Cat', 'Cat_null'],
    ['Dtypes', 'Nulls']])

d = DataFrame(pd.concat(map(
    summarize_types,
    [dfs, dfsnulls, dfc, dfcnulls, ]
    ), axis=1))
d.columns = ix
d


# ## Test formats

# In[ ]:

import fastparquet
import feather


# In[ ]:

# import myutils as mu
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


# In[ ]:

dfs[:2]


# In[ ]:

DataFrame.to_csv(index=None)


# In[ ]:

pd.read_csv('test/t.csv')[:2]


# In[ ]:

from functools import partial as part


# In[ ]:

def bench(fn, df, writer, reader, desc=''):
    twrite = Timer(start=1)
    writer(df, fn)
    twrite.end()
    
    tread = Timer(start=1)
    rdf = reader(fn)
    tread.end()
    assert df.shape == rdf.shape, '{} != {}'.format(df.shape, rdf.shape)
    assert (df.dtypes == rdf.dtypes).all(), '{}\n\n != \n\n{}'.format(df.dtypes, rdf.dtypes)
    
    return twrite.time, tread.time, os.path.getsize(fn) / 10**6


# In[ ]:

def pq_writer(**kw):
    def write_pq(df, fn):
        fastparquet.write(fn, df, **kw)
    return write_pq

def pq_reader(categories=None):
    def read_pq(fn):
        f = fastparquet.ParquetFile(fn)
        return f.to_pandas(categories=categories)
    return read_pq


# In[ ]:

def run_tests(df):
    res = [
        bench('test/t.csv', df, part(DataFrame.to_csv, index=None), pd.read_csv) + ('Csv',),
        bench('test/t.fth', df, feather.write_dataframe, feather.read_dataframe) + ('Feather',),
        bench('test/t_snap.parq', df, pq_writer(compression='SNAPPY'), pq_reader()) + ('Pq-Snappy',),
        bench('test/t_unc.parq', df, pq_writer(compression='UNCOMPRESSED'), pq_reader()) + ('Pq-Uncompressed',),
        bench('test/t_gzip.parq', df, pq_writer(compression='GZIP'), pq_reader()) + ('Pq-Gzip',),
    ]
    return res


# In[ ]:

res_ = run_tests(dfs)
resnull_ = run_tests(dfsnulls)


# In[ ]:

todf = lambda x: DataFrame(x, columns=['Write_time', 'Read_time', 'Mb', 'Fmt'])

res = todf(res_)
resnull = todf(resnull_)


# ## Plot

# In[ ]:

def label(df, x, y, txt, ax):
    for i, point in df.iterrows():
        ax.text(point[x], point[y], str(point[txt]))

def plot_scatter(df, x=None, y=None, lab=None, ax=None):
    df.plot.scatter(x=x, y=y, ax=ax)
    label(df, x, y, lab, ax or plt.gca())


# In[ ]:

def plot_times_size(res):
    _, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    plot_scatter(res, x='Write_time', y='Read_time', lab='Fmt', ax=ax1)
    plot_scatter(res, x='Read_time', y='Mb', lab='Fmt', ax=ax2)


# In[ ]:

plot_times_size(res)


# In[ ]:

plot_times_size(resnull)


# In[ ]:




# In[ ]:

pq_writer(compression='SNAPPY')(dfs, '/tmp/x.parq')


# In[ ]:

f = fastparquet.ParquetFile('/tmp/x.parq')


# In[ ]:

f.to_pandas(categories=categories)


# In[ ]:

f


# In[ ]:




# In[ ]:

t = Timer(start=1)


# In[ ]:

t.end()


# In[ ]:

get_ipython().magic('mkdir test')


# In[ ]:

dforig[:2]

