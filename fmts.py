
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
    plot_scatter, part, getsize, INFEASIBLE, nan,
    check_args
)
get_ipython().magic('matplotlib inline')
# ;;


# In[ ]:

def null_type(s):
    n = (~(s == s)).sum() > 0
    t1 = s.dtype
    if t1 == object:
        tp = type(s.iloc[0])
        return n, tp
    return n, t1


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


# ## Bcolz

# In[ ]:

import bcolz
# from numba import njit

# from pandas.util.testing import assert_frame_equal


# %time bu = bcolz.carray(a)
# %time bu2 = bcolz.carray(df['Unq'].tolist())
# %time b2 = bcolz.ctable.fromdataframe(df[['Unq']])

# ### Bytes | categorical | strings & Nulls | Nonulls
# This function returns a container with combinations of the different string data encodings, and some with nulls randomly inserted, since this can have a big effect on parquet's performance.

# def gen_diff_types(df):
#     "df with str dtype"
#     all_dfs = lambda: None
#     dct = all_dfs.__dict__
#     dct.update(dict(
#         dfs=df, 
#         dfsnulls=(df, f=add_nulls),
#         dfb=mod_cols(df, f=lambda x: x.str.encode('utf-8')),
#         dfc=mod_cols(df, f=tocat),
#     ))
#     dct.update(dict(
#         dfbnulls=mod_cols(all_dfs.dfb, f=add_nulls),
#         dfcnulls=mod_cols(all_dfs.dfsnulls, f=tocat),
#     ))
#     return all_dfs
#     
# nr.seed(0)
# dflo = gen_diff_types(df[['Low_card']])
# dfunq = gen_diff_types(df[['Unq']])

# In[ ]:

def new_cols(df):
    c = df.columns[0]
    s = df[c]
    df = df.assign(
        # Str=s, 
        Str_nls=lambda x: add_nulls(x[c]),
        Bytes=lambda x: x[c].str.encode('utf-8'),
        Cat=lambda x: tocat(x[c])
    ).assign(
        Bytes_nls=lambda x: add_nulls(x['Bytes']),
        Cat_nls=lambda x: add_nulls(x['Cat']),
    )
    return df.rename(columns={c: 'Str'})

tocat = lambda x: x.astype('category')


# In[ ]:

d_txf = new_cols(df[['Unq']])


# In[ ]:

DataFrame({'Nulls': d_txf.isnull().sum(), 'Dtypes': d_txf.dtypes})


# (d_txf.Cat_nls == d_txf.Cat_nls).mean()

# Here's a description:

# def summarize_types(df):
#     return DataFrame(OrderedDict([('Dtypes', df.dtypes), ('Nulls', df.isnull().sum())]))
# 
# 
# ix = pd.MultiIndex.from_product([
#     ['Str', 'Str_null',
#      'Cat', 'Cat_null',
#      'Bytes', 'Bytes_null'],
#     ['Dtypes', 'Nulls']])
# 
# d = DataFrame(pd.concat(map(
#     summarize_types,
#     [dflo.dfs, dflo.dfsnulls,
#      dflo.dfc, dflo.dfcnulls,
#      dflo.dfb, dflo.dfbnulls,
#     ]
#     ), axis=1))
# d.columns = ix
# d

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
    res = writer(df, fn)
    if res is INFEASIBLE:
        # print('INFEASIBLE')
        return nan, nan, nan
    twrite.end()
    # print('Written with', writer)
    
    tread = Timer(start=1)
    rdf = reader(fn)
    tread.end()

    assert df.shape == rdf.shape, '{} != {}'.format(df.shape, rdf.shape)
    # assert (df.dtypes == rdf.dtypes).all(), '{}\n\n != \n\n{}'.format(df.dtypes, rdf.dtypes)
    
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

# import datetime as dt
import imports; imports.reload(imports); from imports import *
import utils; reload(utils); from utils import (
    Timer, get_dtypes, mod_cols, add_nulls, get_obj_type,
    plot_scatter, part, getsize
)
get_ipython().magic('matplotlib inline')
# ;;


# In[ ]:

import enum

class BEnc(enum.Enum):
    "Bcolz encoding"
    df = 1
    list = 3
    utf8 = 2
    
class StrEnc(enum.Enum):
    "Series encoding for string col"
    str = 2
    cat = 1
    byte = 3


# In[ ]:

@check_args
def feasible_bcolz(method: BEnc=None, str_enc: StrEnc=None,
                   null: bool=None, **_):
    "Some bcolz settings are unbearably slow."   
    if method == BEnc.df:
        return (str_enc, null) == (StrEnc.str, False)
    elif method == BEnc.list:
        return (str_enc in {StrEnc.str, StrEnc.byte}) and (not null)
    elif method == BEnc.utf8:
        return str_enc != StrEnc.cat
    raise TypeError("Shouldn't reach here")
    
    
feasible_bcolz(method=BEnc.df, str_enc=StrEnc.str, null=False)


# combos = [
#     (benc, strenc, null)
#     for benc in BEnc.__members__.values()
#     for strenc in StrEnc.__members__.values()
#     for null in [True, False]
# 
# ]
# 
# bads = slow.sort_values(['Fmt', 'Enc', 'Null',], ascending=True).copy()
# bads.Fmt = bads.Fmt.map({'Bcolz-df': BEnc.df, 'Bcolz-lst': BEnc.list, 'Bcolz-uni': BEnc.utf8})
# bads.Enc = bads.Enc.map({'Str': StrEnc.str, 'Cat': StrEnc.cat, 'Byte': StrEnc.byte,})
# 
# bad_combos = list(bads['Fmt Enc Null'.split()].itertuples(index=False, name=None))
# 
# for combo in combos:
#     method, str_enc, null = combo
#     inbad = combo in bad_combos
#     feas = feasible_bcolz(method=method, str_enc=str_enc, null=null)
#     assert inbad != feas
# #     print(inbad, feas)
# #     if inbad == feas:
# #         print(combo)
#     

# In[ ]:

def todf(x):
    d = DataFrame(x).T
    d.columns = ['Write_time', 'Read_time', 'Mb']
    d.index.name = 'Fmt'
    return d.reset_index(drop=0)

def stack_results(res):
    return pd.concat([
        (df.assign(Enc=type)
           .assign(Null=null))
        for df, type, null in res
    ], ignore_index=True)


# In[ ]:

tolist = lambda x: x.tolist()
to_unicode = lambda x: x.values.astype('U')

    
@check_args
def write_bcolz(df, fn=None, method: BEnc=None, str_enc: StrEnc=None,
                   null: bool=None):
    if not feasible_bcolz(method=method, str_enc=str_enc, null=null):
        return INFEASIBLE
    if method == BEnc.df:
        ct = bcolz.ctable.fromdataframe(df, rootdir=fn)
    else:
        converter = {BEnc.list: tolist , BEnc.utf8: to_unicode}[method]
        cs = [converter(col) if (col.dtype == 'O')
              else col for _, col in df.iteritems()]
        ct = bcolz.ctable(columns=cs, names=list(df), rootdir=fn)
    return ct

def read_bcolz(fn):
    ct = bcolz.open(fn, mode='r')
    return DataFrame(ct[:])


mk_bcolz_writer = lambda **kw: part(write_bcolz, **kw)
# mk_bcolz_reader = lambda **_: read_bcolz
# read_bcolz(fn)[:2]


# In[ ]:

d_txf[:2]


# In[ ]:

import sys

def fprint(*a, **k):
    print(*a, **k)
    sys.stdout.flush()


# In[ ]:

@check_args
def run_writers(df, asdf=True, cats=None, dirname='/tmp/test',
                str_enc: StrEnc=None, null: bool=False):
    """For given df (should be single column), run series of
    reads/writes
    """
    null_, _ = null_type(df.iloc[:, 0])
    assert null_ == null
    obj_tp = get_obj_type(df) if cats is None else 'infer'
        
    pqr = pq_reader(categories=cats)
    csv_dtype = (None if cats is None else
        dict(zip(cats, it.repeat('category')))
    )
    
    if os.path.exists(dirname):
        print('Deleting', dirname, '... ', end='')
        shutil.rmtree(dirname)
        fprint('Done')
        
    os.mkdir(dirname)
    dir = lambda x: os.path.join(dirname, x)
    
#     method=BEnc.df, null=null, str_enc=str_enc
    
    csv_reader = partial(pd.read_csv, dtype=csv_dtype, index_col=0)
    pq_write_enc = pq_writer(compression='SNAPPY', object_encoding=obj_tp)
    pq_write_len = pq_writer(get_lens=True, compression='SNAPPY')
    
    bc_mkr_mkr = part(mk_bcolz_writer, null=null, str_enc=str_enc)
    blosc_df_wrt = bc_mkr_mkr(method=BEnc.df)
    blosc_uni_wrt = bc_mkr_mkr(method=BEnc.utf8)
    blosc_lst_wrt = bc_mkr_mkr(method=BEnc.list)
#     blosc_df_wrt = mk_bcolz_writer(method=BEnc.df, null=null, str_enc=str_enc)
#     blosc_uni_wrt = mk_bcolz_writer(method=BEnc.utf8, null=null, str_enc=str_enc)
#     blosc_lst_wrt = mk_bcolz_writer(method=BEnc.list, null=null, str_enc=str_enc)
    res = {
        'Csv': try_bench(dir('t.csv'), df, DataFrame.to_csv, csv_reader),
        'Bcolz-df': try_bench(dir('t_df.blsc'), df, blosc_df_wrt, read_bcolz),
        'Bcolz-uni': try_bench(dir('t_uni.blsc'), df, blosc_uni_wrt, read_bcolz),
        'Bcolz-lst': try_bench(dir('t_lst.blsc'), df, blosc_lst_wrt, read_bcolz),
        'Feather': try_bench(dir('t.fth'), df, feather.write_dataframe, feather.read_dataframe),
        'Pq-Snappy': try_bench(dir('t_snap.parq'), df, pq_writer(compression='SNAPPY'), pqr),
        'Pq-Snappy-enc': try_bench(dir('t_snap_utf8.parq'), df, pq_write_enc, pqr),
        'Pq-Snappy-ft': try_bench(dir('t_snap_f.parq'), df, pq_write_len, pqr),
        'Pq-Uncompressed': try_bench(dir('t_unc.parq'), df, pq_writer(compression='UNCOMPRESSED'), pqr),
    }
    # try_bench('/tmp/test/t_gzip.parq', df, pq_writer(compression='GZIP'), pqr) + ('Pq-Gzip',),  # <= slow writes
    if asdf:
        return todf(res)
    else:
        return res


# In[ ]:

d_txf[:3]


# In[ ]:

def run_dfs(base_df):
    """For base df with single col, generate new columns
    based on original with different ways of encoding
    the string type. Run the full battery of
    read/write benchmarks on each of these new columns.
    """
    d = new_cols(base_df)
    #global res, resnull, resc, rescnull, resb, resbnull
    res = run_writers(d[['Str']], str_enc=StrEnc.str)
    resnull = run_writers(d[['Str_nls']], str_enc=StrEnc.str, null=True)

    resc = run_writers(d[['Cat']], cats=cols, str_enc=StrEnc.cat)
    rescnull = run_writers(d[['Cat_nls']], cats=cols, str_enc=StrEnc.cat, null=True)

    resb = run_writers(d[['Bytes']], str_enc=StrEnc.byte)
    resbnull = run_writers(d[['Bytes_nls']], str_enc=StrEnc.byte, null=True)
    
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

r = run_dfs(df[['Low_card']][:2000])


# def run_dfs(dfholder):
#     d = dfholder
#     global res, resnull, resc, rescnull, resb, resbnull
#     res = run_writers(d.dfs, asdf=1, cats=None)
#     print('rnull!')
#     resnull = run_writers(d.dfsnulls, asdf=1, cats=None)
# 
#     resc = run_writers(d.dfc, asdf=1, cats=cols)
#     rescnull = run_writers(d.dfcnulls, asdf=1, cats=cols)
# 
#     resb = run_writers(d.dfb, asdf=1, cats=None)
#     resbnull = run_writers(d.dfbnulls, asdf=1, cats=None)
#     
#     allres = stack_results([
#         (res, 'Str', False),
#         (resnull, 'Str', True),
#         (resc, 'Cat', False),
#         (rescnull, 'Cat', True),
#         (resb, 'Byte', False),
#         (resbnull, 'Byte', True),
#     ])
#     return allres

# ## Actually run benchmarks

# reslo_old = reslo

# In[ ]:

get_ipython().magic("time reslo = run_dfs(df[['Low_card']])")


# In[ ]:

mu.ping()


# In[ ]:

get_ipython().magic("time resunq = run_dfs(df[['Unq']])")


# In[ ]:

mu.ping()


# In[ ]:

get_ipython().system('rm -rf /tmp/test/')
get_ipython().magic('mkdir /tmp/test/')


# reslo = run_dfs(dflo)
# 
# resunq = run_dfs(dfunq)

# ## Plots

# In[ ]:

allres.ix[np.setdiff1d(allres.index, _allres.index)].sort_values(['Fmt'], ascending=True)


# In[ ]:

comp_rat.plot.scatter(x='Read_time', y='Compression_ratio')


# In[ ]:

slow = reslo.sort_values('Read_time', ascending=0)[:11].reset_index(drop=1)


# In[ ]:

Bcolz-lst   78.783165  194.710648  15.797601   Cat   True
Bcolz-lst   47.681886    6.519006  15.800197   Cat  False
Bcolz-lst   56.181895   58.864246  15.796801   Str   True
Bcolz-lst   79.398800   15.782561  15.796803  Byte   True


# In[ ]:

slow.sort_values('Fmt', ascending=True)


# In[ ]:

_allres = reslo.dropna(axis=0)
g = sns.FacetGrid(_allres, row='Enc', col='Null', aspect=1.2, size=4)
g.map(plot_scatter, 'Write_time', 'Read_time', 'Mb', 'Fmt')


# In[ ]:

_allres = resunq.dropna(axis=0)
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

