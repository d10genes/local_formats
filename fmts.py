
# coding: utf-8

# In[ ]:

# import datetime as dt
from importlib import reload
import utils; reload(utils); from utils import Timer
import itertools as it
import os

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
dfb = mod_cols(dfs, f=lambda x: x.str.encode('utf-8'))
dfbnulls = mod_cols(dfb, f=add_nulls)


# In[ ]:

low_card = dfs[['Hhdfmx']]


# In[ ]:

f = lambda x: x
f


# In[ ]:

def gen_diff_types(df):
    "df with str dtype"
    dfsnulls = mod_cols(df, f=add_nulls)
    dfb = mod_cols(df, f=lambda x: x.str.encode('utf-8'))
    dfbnulls = mod_cols(dfb, f=add_nulls)
    all_dfs = lambda: None
    all_dfs.__dict__.update(dict(
        dfs=df, 
        dfsnulls=dfsnulls,
        dfb=dfb,
        dfbnulls=dfbnulls,
    ))
    return all_dfs
    
gen_diff_types()


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


ix = pd.MultiIndex.from_product([
    ['Str', 'Str_null',
     'Cat', 'Cat_null',
     'Bytes', 'Bytes_null'],
    ['Dtypes', 'Nulls']])

d = DataFrame(pd.concat(map(
    summarize_types,
    [dfs, dfsnulls,
     dfc, dfcnulls,
     dfb, dfbnulls,
    ]
    ), axis=1))
d.columns = ix
d


# ## Bcolz

# import bcolz
#     # !rm -r 'data/df.bcolz'
# %time ct = bcolz.ctable.fromdataframe(dfs[:10000000], rootdir='data/df.bcolz', mode='w')

# param_perf = {}

# cparams_ = {'clevel': 5, 'shuffle': True, 'cname': 'lz4'}
# cparams_.update({
#     'clevel': 1,
#     # 'cname': 'snappy',
#     'shuffle': 0,
# })
# cparams = bcolz.cparams(**cparams_)
# 
# !rm -r 'data/df.bcolz'
# %time ct = bcolz.carray(dfs.Hhdfmx[:2000], rootdir='data/df.bcolz', cparams=cparams)
# param_perf[frozenset(cparams_.items())] = ct.nbytes / ct.cbytes
# ct
# - ================================================================
# def merge(d1, d2):
#     d = d1.copy()
#     d.update(d2)
#     return d
# 
# DataFrame([merge(dict(fs), {'Ratio': rat}) for fs, rat in param_perf.items()])

# ## Test formats

# In[ ]:

import fastparquet
import feather

from functools import partial as part


# In[ ]:

def max_len(s, nulls=True):
    if nulls:
        s = s[s == s]
    return s.map(len).max()

def pq_writer(get_lens=False, **kw):
    def write_pq(df, fn):
        if get_lens:
            lns = {c: max_len(df[c], nulls=True) for c in get_dtypes(df, object)}
        else:
            lns = None
        fastparquet.write(fn, df, fixed_text=lns, **kw)
    return write_pq

def pq_reader(categories=None):
    def read_pq(fn):
        f = fastparquet.ParquetFile(fn)
        return f.to_pandas(categories=categories)
    return read_pq


# ### Proto
# 
# fn = '/tmp/t.parq'
# fastparquet.write(fn, d, object_encoding='infer', fixed_text={'A': 2})
# 
# pf = fastparquet.ParquetFile(fn)
# 
# pf.to_pandas()

# In[ ]:

def bench(fn, df, writer, reader, desc=''):
    twrite = Timer(start=1)
    writer(df, fn)
    twrite.end()
    
    tread = Timer(start=1)
    rdf = reader(fn)
    tread.end()
    global _df, _rdf
    _df, _rdf = df, rdf
    assert df.shape == rdf.shape, '{} != {}'.format(df.shape, rdf.shape)
    assert (df.dtypes == rdf.dtypes).all(), '{}\n\n != \n\n{}'.format(df.dtypes, rdf.dtypes)
    
    return twrite.time, tread.time, os.path.getsize(fn) / 10**6

def try_bench(*a, **kw):
    known_errors = [
        'with dtype bytes',
        "'NoneType' object has no attribute 'encode'",
        "'bytes' object has no attribute 'encode'",
    ]
    try:
        return bench(*a, **kw)
    except Exception as e:
        print(e)
        print(a[0])
        if any(s in str(e) for s in known_errors):
            na = float('nan')
            return na, na, na
        else:
            raise(e)


# In[ ]:

has_nulls=False,
              object_encoding='utf8',
              fixed_text={'store_and_fwd_flag': 1})


# In[ ]:

obj_col


# In[ ]:

v


# In[ ]:

(dfsnulls[obj_col] == dfsnulls[obj_col])


# In[ ]:

def get_obj_type(df, as_str=True):
    [obj_col, *_] = get_dtypes(df, object)
    s = df[obj_col]
    nonull_val = s[s == s].values[0]
    if as_str:
        return enc_dct[type(nonull_val)]
    return type(nonull_val)


# In[ ]:




# In[ ]:

enc_dct[get_obj_type(dfs)]


# In[ ]:

get_obj_type(dfb)


# In[ ]:

for v in dfsnulls[obj_col]:
    if v != v:
        print('!')
        break


# In[ ]:

def stack_results(res):
    return pd.concat([
        (df.assign(Enc=type)
           .assign(Null=null))
        for df, type, null in res
    ], ignore_index=True)


def run_tests(df, asdf=True, cats=None):
    obj_tp = get_obj_type(df) if cats is None else 'infer'
        
    pqr = pq_reader(categories=cats)
    csv_dtype = (None if cats is None else
        dict(zip(cats, it.repeat('category')))
    )
    res = [
        try_bench('test/t.csv', df, part(DataFrame.to_csv, index=None),
              part(pd.read_csv, dtype=csv_dtype)) + ('Csv',),
        try_bench('test/t.fth', df, feather.write_dataframe, feather.read_dataframe) + ('Feather',),
        try_bench('test/t_snap.parq', df, pq_writer(compression='SNAPPY'), pqr) + ('Pq-Snappy',),
        try_bench('test/t_snap_utf8.parq', df, pq_writer(compression='SNAPPY', object_encoding=obj_tp), pqr) + ('Pq-Snappy-enc',),
        try_bench('test/t_snap_f.parq', df,
                  pq_writer(get_lens=True, compression='SNAPPY'),
                  pqr) + ('Pq-Snappy-ft',),
        try_bench('test/t_unc.parq', df, pq_writer(compression='UNCOMPRESSED'), pqr) + ('Pq-Uncompressed',),
        try_bench('test/t_gzip.parq', df, pq_writer(compression='GZIP'), pqr) + ('Pq-Gzip',),
    ]
    if asdf:
        return todf(res)
    else:
        return res

todf = lambda x: DataFrame(x, columns=['Write_time', 'Read_time', 'Mb', 'Fmt'])


# In[ ]:

res


# In[ ]:

cats = get_dtypes(dfc).tolist()
assert cats


# In[ ]:




# In[ ]:

dfs[:2
   ]


# In[ ]:

res = run_tests(dfs, asdf=1, cats=None)
resnull = run_tests(dfsnulls, asdf=1, cats=None)


# In[ ]:

resc = run_tests(dfc, asdf=1, cats=cats)
rescnull = run_tests(dfcnulls, asdf=1, cats=cats)


# In[ ]:

resb = run_tests(dfb, asdf=1, cats=None)
resbnull = run_tests(dfbnulls, asdf=1, cats=None)


# In[ ]:

allres = stack_results([
    (res, 'Str', 'False'),
    (resnull, 'Str', 'True'),
    (resc, 'Cat', 'False'),
    (rescnull, 'Cat', 'True'),
    (resb, 'Byte', 'False'),
    (resbnull, 'Byte', 'True'),
])


# ## Plot

# In[ ]:

def label_df(df, x, y, txt, ax):
    for i, point in df.iterrows():
        ax.text(point[x], point[y], str(point[txt]))

def plot_scatter(df, x=None, y=None, lab=None, ax=None, size=None):
    df.plot.scatter(x=x, y=y, ax=ax, s=size)
    label_df(df, x, y, lab, ax or plt.gca())


# In[ ]:

def scatter_df(df, x=None, y=None, s=None):
    p = plt.scatter(df[x], df[y], s=df[s] * 2)
    plt.xlabel(x)
    plt.ylabel(y)


# In[ ]:

def plot_times_size(res):
    _, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    plot_scatter(res, x='Read_time', y='Write_time', lab='Fmt', ax=ax1, size='Mb')
    plot_scatter(res, x='Read_time', y='Mb', lab='Fmt', ax=ax2)


# In[ ]:




# In[ ]:

plot_times_size(res)


# In[ ]:

plot_times_size(resnull)


# In[ ]:

plot_times_size(res)


# In[ ]:

NUDGE = .035

def s2df(ss):
    return DataFrame(OrderedDict([(s.name, s) for s in ss]))

def label(x, y, txt):
    x, y, txt = df = s2df([x, y, txt])
    ax = plt.gca()
    for i, row in df.iterrows():
        ax.text(row[x] + NUDGE, row[y] + NUDGE, str(row[txt]))
        
def scatter(x=None, y=None, s=None):
    p = plt.scatter(x, y, s=s * 2)
    plt.xlabel(x)
    plt.ylabel(y)
    # plt.legend()
    
def plot_scatter2(x, y, size, lab=None, ax=None, color=None):
    scatter(x=x, y=y, s=size)
    # scatter_df(res, x=x, y=y, size=s)
    label(x, y, lab)


# In[ ]:

def outlier_val(s, nsd=2.5):
    s = s.dropna()
    m = s.mean()
    sd = s.std()
    return m + nsd * sd


# In[ ]:

def trim_outliers(df, cs=[], nsd=2.5):
    for c in cs:
        s = df[c]
        v = outlier_val(s, nsd=nsd)
        df = df[s <= v]
    return df


# In[ ]:

s = allres.Write_time.dropna()


# In[ ]:

import numpy as np


# In[ ]:




# In[ ]:




# In[ ]:

np.percentile(s, 90)


# In[ ]:

allres.ix[np.setdiff1d(allres.index, _allres.index)].sort_values(['Fmt'], ascending=True)


# In[ ]:

# _allres = trim_outliers(allres, cs=['Write_time', 'Read_time'])
_allres = allres.dropna(axis=0)
g = sns.FacetGrid(_allres, row='Enc', col='Null', aspect=1.2, size=4)
# g.map(plot_scatter2, 'Read_time', 'Write_time', 'Mb', 'Fmt')
g.map(plot_scatter2, 'Write_time', 'Read_time', 'Mb', 'Fmt')


# In[ ]:

g = sns.FacetGrid(allres.assign(One=10), row='Enc', col='Null', aspect=1.2, size=4)
# g.map(plot_scatter2, 'Read_time', 'Write_time', 'Mb', 'Fmt')
g.map(plot_scatter2, 'Mb', 'Read_time', 'One', 'Fmt')


# In[ ]:




# In[ ]:

allres[:2]


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

