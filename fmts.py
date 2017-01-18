
# coding: utf-8

# In[ ]:

# import datetime as dt
from importlib import reload
import utils; reload(utils); from utils import Timer, get_dtypes, mod_cols, add_nulls
import itertools as it
import os

import uuid
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


df = pd.read_csv('data/census-income.data', header=None).iloc[:, [1]].rename(columns={1: 'Low_card'}) #[['Hhdfmx']]
# dforig.columns = [l.split()[-1].capitalize() for l in it.takewhile(bool, csd)]


# In[ ]:

# Get a few of each dtype
# ctypes = [c for dtp, gdf in dforig.dtypes.reset_index(drop=0).groupby(0) for c in gdf['index'][:4]]
df = df[['Low_card']].assign(Unq=[uuid.uuid4().hex for _ in range(len(df))])


# In[ ]:

df.apply(lambda x: x.nunique())


# ### Nulls in String cols
# 
# ### Convert string cols to categorical

# In[ ]:

tocat = lambda x: x.astype('category')

# dfc = mod_cols(dfs, f=tocat)
# dfcnulls = mod_cols(dfsnulls, f=tocat)


# 
# dfsnulls = mod_cols(dfs, f=add_nulls)
# dfb = mod_cols(dfs, f=lambda x: x.str.encode('utf-8'))
# dfbnulls = mod_cols(dfb, f=add_nulls)

# In[ ]:

def gen_diff_types(df):
    "df with str dtype"
    all_dfs = lambda: None
    dct = all_dfs.__dict__
    dct.update(dict(
        dfs=df, 
        dfsnulls=mod_cols(df, f=add_nulls),
        dfb=mod_cols(df, f=lambda x: x.str.encode('utf-8')),
        dfc=mod_cols(df, f=tocat),
    ))
    dct.update(dict(
        dfbnulls=mod_cols(all_dfs.dfb, f=add_nulls),
        dfcnulls=mod_cols(all_dfs.dfsnulls, f=tocat),
    ))
    return all_dfs
    
nr.seed(0)
dflo = gen_diff_types(df[['Low_card']])
dfunq = gen_diff_types(df[['Unq']])


# ### Convert string to categorical

# In[ ]:

from collections import OrderedDict

def summarize_types(df):
    return DataFrame(OrderedDict([('Dtypes', df.dtypes), ('Nulls', df.isnull().sum())]))


ix = pd.MultiIndex.from_product([
    ['Str', 'Str_null',
     'Cat', 'Cat_null',
     'Bytes', 'Bytes_null'],
    ['Dtypes', 'Nulls']])

d = DataFrame(pd.concat(map(
    summarize_types,
    [dflo.dfs, dflo.dfsnulls,
     dflo.dfc, dflo.dfcnulls,
     dflo.dfb, dflo.dfbnulls,
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

def get_obj_type(df, as_str=True):
    """Get type of first non-null element in first column
    with object dtype. With `as_str` convert to arg for
    `fastparquet.write`'s `object_encoding` param.
    """
    [obj_col, *_] = get_dtypes(df, object)
    s = df[obj_col]
    nonull_val = s[s == s].values[0]
    if as_str:
        return enc_dct[type(nonull_val)]
    return type(nonull_val)

enc_dct = {str: 'utf8', bytes: 'bytes'}


# In[ ]:

cols = ['Low_card', 'Unq']


# In[ ]:

def stack_results(res):
    return pd.concat([
        (df.assign(Enc=type)
           .assign(Null=null))
        for df, type, null in res
    ], ignore_index=True)


def run_writers(df, asdf=True, cats=None):
    obj_tp = get_obj_type(df) if cats is None else 'infer'
        
    pqr = pq_reader(categories=cats)
    csv_dtype = (None if cats is None else
        dict(zip(cats, it.repeat('category')))
    )
    res = [
        try_bench('test/t.csv', df,
                  DataFrame.to_csv,
#                   part(DataFrame.to_csv, index=None),
              part(pd.read_csv, dtype=csv_dtype, index_col=0)) + ('Csv',),
        try_bench('test/t.fth', df, feather.write_dataframe, feather.read_dataframe) + ('Feather',),
        try_bench('test/t_snap.parq', df, pq_writer(compression='SNAPPY'), pqr) + ('Pq-Snappy',),
        try_bench('test/t_snap_utf8.parq', df, pq_writer(compression='SNAPPY', object_encoding=obj_tp), pqr) + ('Pq-Snappy-enc',),
        try_bench('test/t_snap_f.parq', df,
                  pq_writer(get_lens=True, compression='SNAPPY'),
                  pqr) + ('Pq-Snappy-ft',),
        try_bench('test/t_unc.parq', df, pq_writer(compression='UNCOMPRESSED'), pqr) + ('Pq-Uncompressed',),
#         try_bench('test/t_gzip.parq', df, pq_writer(compression='GZIP'), pqr) + ('Pq-Gzip',),
    ]
    if asdf:
        return todf(res)
    else:
        return res

todf = lambda x: DataFrame(x, columns=['Write_time', 'Read_time', 'Mb', 'Fmt'])


# In[ ]:

def run_dfs(dfholder):
    d = dfholder
    global res, resnull, resc, rescnull, resb, resbnull
    res = run_writers(d.dfs, asdf=1, cats=None)
    print('rnull!')
    resnull = run_writers(d.dfsnulls, asdf=1, cats=None)

    resc = run_writers(d.dfc, asdf=1, cats=cols)
    rescnull = run_writers(d.dfcnulls, asdf=1, cats=cols)

    resb = run_writers(d.dfb, asdf=1, cats=None)
    resbnull = run_writers(d.dfbnulls, asdf=1, cats=None)
    
#     allres = stack_results([
#         (res, 'Str', 'False'),
#         (resnull, 'Str', 'True'),
#         (resc, 'Cat', 'False'),
#         (rescnull, 'Cat', 'True'),
#         (resb, 'Byte', 'False'),
#         (resbnull, 'Byte', 'True'),
#     ])
    allres = stack_results([
        (res, 'Str', False),
        (resnull, 'Str', True),
        (resc, 'Cat', False),
        (rescnull, 'Cat', True),
        (resb, 'Byte', False),
        (resbnull, 'Byte', True),
    ])
    return allres


# In[ ]:

reslo = run_dfs(dflo)


# In[ ]:

resunq = run_dfs(dfunq)


# res = run_tests(dfs, asdf=1, cats=None)
# resnull = run_tests(dfsnulls, asdf=1, cats=None)
# 
# resc = run_tests(dfc, asdf=1, cats=cats)
# rescnull = run_tests(dfcnulls, asdf=1, cats=cats)
# 
# resb = run_tests(dfb, asdf=1, cats=None)
# resbnull = run_tests(dfbnulls, asdf=1, cats=None)
# 
# allres = stack_results([
#     (res, 'Str', 'False'),
#     (resnull, 'Str', 'True'),
#     (resc, 'Cat', 'False'),
#     (rescnull, 'Cat', 'True'),
#     (resb, 'Byte', 'False'),
#     (resbnull, 'Byte', 'True'),
# ])

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

plot_times_size(res)


# In[ ]:

plot_times_size(resnull)


# In[ ]:

plot_times_size(res)


# In[ ]:

from functools import partial
# import numpy as np
NUDGE = .005

def s2df(ss):
    return DataFrame(OrderedDict([(s.name, s) for s in ss]))

def label(x, y, txt):
    x, y, txt = df = s2df([x, y, txt])
    ax = plt.gca()
    for i, row in df.iterrows():
        ax.text(row[x] + NUDGE, row[y] + NUDGE, str(row[txt]))
        
def scatter(x=None, y=None, s=None, sz_fact=30):
    p = plt.scatter(x, y, s=s * sz_fact, alpha=.25)
    plt.xlabel(x)
    plt.ylabel(y)
    # plt.legend()
    
def plot_scatter2(x, y, size, lab=None, ax=None, sz_fact=30, color=None):
#     print('size={}\t lab={}\t ax={}\t sz_fact={}\t color={}\t'.format(size, lab, ax, sz_fact, color))
    scatter(x=x, y=y, s=size, sz_fact=sz_fact)
    # scatter_df(res, x=x, y=y, size=s)
    label(x, y, lab)

def outlier_val(s, nsd=2.5):
    s = s.dropna()
    m = s.mean()
    sd = s.std()
    return m + nsd * sd

def trim_outliers(df, cs=[], nsd=2.5):
    for c in cs:
        s = df[c]
        v = outlier_val(s, nsd=nsd)
        df = df[s <= v]
    return df

def part(f, *a, **kw):
    wrapper = partial(f, *a, **kw)
    wrapper.__module__ = '__main__'
    return wrapper


# In[ ]:

allres.ix[np.setdiff1d(allres.index, _allres.index)].sort_values(['Fmt'], ascending=True)


# In[ ]:

comp_rat.plot.scatter(x='Read_time', y='Compression_ratio')


# In[ ]:

_allres = reslo.dropna(axis=0)
g = sns.FacetGrid(_allres, row='Enc', col='Null', aspect=1.2, size=4)
g.map(plot_scatter2, 'Write_time', 'Read_time', 'Mb', 'Fmt')


# In[ ]:

reslo.Null == 'True'


# In[ ]:

reslo.query('Null').sort_values(['Enc', 'Read_time'], ascending=True)


# In[ ]:




# In[ ]:

fastparquet.write()


# In[ ]:

_allres = resunq.dropna(axis=0)
g = sns.FacetGrid(_allres, row='Enc', col='Null', aspect=1.2, size=4)
g.map(part(plot_scatter2, sz_fact=10), 'Write_time', 'Read_time', 'Mb', 'Fmt')


# - CSV suffers dramatically in speed on categorical encodings
# - Low cardinality strings
#     - Everything besides CSV has similar speed characteristics for categoricals
# - Unique strings
#     
# - Feather is really hard to beat in speed, except for unique byte strings with nulls (and [crashes for bytes with no nulls in the column](https://github.com/wesm/feather/issues/283))
# 
# 
# My personal takeaways:
# - For columns with lots of redundant strings, convert to categorical if possible and use fastparquet with Snappy
# - Otherwise, try to use bytes instead of strings and use `fastparquet.write`'s `fixed_text` argument

# In[ ]:

comp_rat = reslo.assign(Compression_ratio=lambda x: x.Mb.max() / x.Mb).query('Fmt != "Pq-Gzip"')
sns.stripplot(x="Fmt", y="Compression_ratio", data=comp_rat, jitter=True)
plt.xticks(rotation=75);


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

