
# coding: utf-8

# After years of CSV file format domination in the python world, I'm finally seeing more compelling options for local column-oriented data storage. Some give advantages in read speed, file size or both. 
# 
# While I've had good results in these formats for floats and ints, a bottleneck I still have is storing columns with strings. The file format landscape is improving fast, so I won't be surprised if the speed results of the benchmarks here are obsolete before they're published, but this can at least serve as a performance snapshot.
# 
# [Feather](https://github.com/wesm/feather) is a storage format that aims for speed and interoperability between languages based on the Apache Arrow spec. It shares a creator [Wes McKinney](https://github.com/wesm) with Pandas, and integrates nicely into the python data ecosystem.
# 
# [Fastparquet](https://github.com/dask/fastparquet) is a recent fork of python's implementation of the parquet file format, with an unsurprising focus on speed. Both parquet and [Bcolz](https://github.com/Blosc/bcolz) offer compression as a feature, potentially speeding up IO and shrinking the required size on disk.
# 
# 
# ## String data
# A cursory look at these formats shows an obvious performance increase for numerical data, but I quickly found that performance is much more hit-or-miss for data involving strings. My hope here was to get an empirical look at my options when dealing with string data.
# 
# A difficulty with using strings in Python is that pandas seems to treat them as arbitrary python objects, eliminating any performance gains you can get from numerical datatypes using numpy underneath. A couple of options that you have are to convert the series to a categorical data type, or convert all of the string elements to bytes.
# 
# In this post I compare the performance of these different file formats on string data encoded as strings, categoricals, and bytes. While I sometimes treat categorical encoding as a silver bullet for performance, since it is represented internally as integers, the benefits only outweigh the overhead when there are just a few unique strings in a column that are repeated many times. To compare the performance of the different formats in both scenarios, I test them on a series with mostly unique strings, and another with just a few repeated strings. 
# 
# The benchmarks below compare pandas' CSV, feather and fastparquet. I couldn't find any setting where Bcolz had reasonable speed on the string data, so threw that out. While the ultimate file size is important, it is secondary to speed for me, so I also ignored compression options like gzip and brotli.
# 
# I'm only using series with about 200k rows because I don't have all day to wait on the benchmarks to finish.

# In[ ]:

# import datetime as dt
import imports; imports.reload(imports); from imports import *
import utils; reload(utils); from utils import (
    Timer, get_dtypes, mod_cols, add_nulls, get_obj_type,
    plot_scatter, part, getsize
)
get_ipython().magic('matplotlib inline')
# ;;


# # Load and set up df

# ### String columns
# Here's the base dataframe with a column for unique values (`Unq`) and one for repetitive values (`Low_card`).

# In[ ]:

nr.seed(0)
N = 200000
chars = np.array(list(string.ascii_lowercase) + lmap(str, range(10)))
nchars = len(chars)
rix = nr.randint(0, high=nchars, size=(N, 32))
rstrs = [''.join(chars[ix]) for ix in rix]

cols = ['Unq', 'Low_card']
df = DataFrame(dict(Unq=rstrs, Low_card=nr.choice(rstrs[:100], size=N)))[cols]


# In[ ]:

df.apply(lambda x: x.nunique())


# In[ ]:

# from castra import Castra
# c = Castra(path='/tmp/test/df.castra', template=df)
# c.extend(df)

# c


# !rm -rf /tmp/test/df.castra

# def castro_writer():
#     def write_castro(df, fn):
#         c = Castra(path=fn, template=df)
#         c.extend(df)
#     return write_castro
# 
# 
# def castro_reader(categories=None):
#     def read_castro(fn):
#         c = Castra(fn)
#         return c[:]
#     return read_castro
# 
# writer = castro_writer()
# reader = castro_reader()
# 
# writer(df, )
# 
# c = Castra('/tmp/test/df.castra')

# ## Bcolz

# In[ ]:

import bcolz
# from numba import njit

# from pandas.util.testing import assert_frame_equal


# fn = '/tmp/test/df.bc'
# 
# !rm -r $fn
# %time ct = write_bcolz(df, fn=fn, asdf=True)
# 
# !rm -r $fn
# %time ctl = write_bcolz(df, fn=fn, asdf=False, convert_series=tolist)
# 
# !rm -r $fn
# %time ctl = write_bcolz(df, fn=fn, asdf=False, convert_series=to_unicode)

# %time bu = bcolz.carray(a)
# %time bu2 = bcolz.carray(df['Unq'].tolist())
# %time b2 = bcolz.ctable.fromdataframe(df[['Unq']])

# In[ ]:

tolist = lambda x: x.tolist()
to_unicode = lambda x: x.values.astype('U')

def write_bcolz(df, fn=None, asdf=True, convert_series=None):
    if asdf:
        ct = bcolz.ctable.fromdataframe(df, rootdir=fn)
    else:
        transform = convert_series or z.identity
        cs = [transform(col) if (col.dtype == 'O') else col for _, col in df.iteritems()]
        ct = bcolz.ctable(columns=cs, names=list(df), rootdir=fn)
    return ct


def read_bcolz(fn):
    ct = bcolz.open(fn, mode='r')
    return DataFrame(ct[:])

mk_bcolz_writer = lambda **kw: part(write_bcolz, **kw)
# mk_bcolz_reader = lambda **_: read_bcolz
# read_bcolz(fn)[:2]


# In[ ]:




# ### Bytes | categorical | strings & Nulls | Nonulls
# This function returns a container with combinations of the different string data encodings, and some with nulls randomly inserted, since this can have a big effect on parquet's performance.

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
    
tocat = lambda x: x.astype('category')
nr.seed(0)
dflo = gen_diff_types(df[['Low_card']])
dfunq = gen_diff_types(df[['Unq']])


# Here's a description:

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
    [dflo.dfs, dflo.dfsnulls,
     dflo.dfc, dflo.dfcnulls,
     dflo.dfb, dflo.dfbnulls,
    ]
    ), axis=1))
d.columns = ix
d


# ## Parquet writer/reader
# Here are some helper functions to read and write with fastparquet using different options

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


# ### Benchmarkers
# And here are some functions to automate the benchmarking, iterating through different inputs and read/write options. I wrapped them all in a sloppy try/except block in a fit of undisciplined laziness.

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
    
    return twrite.time, tread.time, getsize(fn) / 10**6

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

def stack_results(res):
    return pd.concat([
        (df.assign(Enc=type)
           .assign(Null=null))
        for df, type, null in res
    ], ignore_index=True)


def run_writers(df, asdf=True, cats=None, dirname='/tmp/test'):
    obj_tp = get_obj_type(df) if cats is None else 'infer'
        
    pqr = pq_reader(categories=cats)
    csv_dtype = (None if cats is None else
        dict(zip(cats, it.repeat('category')))
    )
    
    os.rmdir(dirname)
    os.mkdir(dirname)
    dir = lambda x: path.join(dirname, x)
    
    csv_reader = partial(pd.read_csv, dtype=csv_dtype, index_col=0))
    res = [
        try_bench(dir('t.csv'), df, DataFrame.to_csv,
              csv_reader + ('Csv',),
        try_bench(dir('t.fth'), df, feather.write_dataframe, feather.read_dataframe) + ('Feather',),
        try_bench(dir('t_snap.parq'), df, pq_writer(compression='SNAPPY'), pqr) + ('Pq-Snappy',),
        try_bench(dir('t_snap_utf8.parq'), df, pq_writer(compression='SNAPPY', object_encoding=obj_tp), pqr
                 ) + ('Pq-Snappy-enc',),
        try_bench(dir('t_snap_f.parq'), df,
                  pq_writer(get_lens=True, compression='SNAPPY'),
                  pqr) + ('Pq-Snappy-ft',),
        try_bench(dir('t_unc.parq'), df, pq_writer(compression='UNCOMPRESSED'), pqr
                 ) + ('Pq-Uncompressed',),
        
        # try_bench('/tmp/test/t_gzip.parq', df, pq_writer(compression='GZIP'), pqr
        # ) + ('Pq-Gzip',),  # <= slow writes
    ]
    if asdf:
        return todf(res)
    else:
        return res

todf = lambda x: DataFrame(x, columns=['Write_time', 'Read_time', 'Mb', 'Fmt'])


# In[ ]:




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
    
    allres = stack_results([
        (res, 'Str', False),
        (resnull, 'Str', True),
        (resc, 'Cat', False),
        (rescnull, 'Cat', True),
        (resb, 'Byte', False),
        (resbnull, 'Byte', True),
    ])
    return allres


# ## Actually run benchmarks

# In[ ]:




# In[ ]:

get_ipython().system('rm -rf /tmp/test/')
get_ipython().magic('mkdir /tmp/test/')


# In[ ]:

reslo = run_dfs(dflo)


# In[ ]:

resunq = run_dfs(dfunq)


# ## Plots

# In[ ]:




# In[ ]:

allres.ix[np.setdiff1d(allres.index, _allres.index)].sort_values(['Fmt'], ascending=True)


# In[ ]:

comp_rat.plot.scatter(x='Read_time', y='Compression_ratio')


# In[ ]:

_allres = reslo.dropna(axis=0)
g = sns.FacetGrid(_allres, row='Enc', col='Null', aspect=1.2, size=4)
g.map(plot_scatter, 'Write_time', 'Read_time', 'Mb', 'Fmt')


# In[ ]:

reslo.query('Null').sort_values(['Enc', 'Read_time'], ascending=True)


# In[ ]:

_allres = resunq.dropna(axis=0)
g = sns.FacetGrid(_allres, row='Enc', col='Null', aspect=1.2, size=4)
g.map(part(plot_scatter, sz_fact=10), 'Write_time', 'Read_time', 'Mb', 'Fmt')


# - CSV suffers dramatically in speed on categorical encodings
# - Low cardinality strings
#     - Everything besides CSV has similar speed characteristics for categoricals
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

